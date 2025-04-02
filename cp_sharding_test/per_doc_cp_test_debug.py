#!/usr/bin/env python
import sys
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# add project root to path if needed.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from per_document_cp_sharding.per_document_cp_sharding import AttnFuncWithAllGatherPerDocSharding
try:
    from flash_attn.flash_attn_interface import _flash_attn_forward as _flash_attn_fwd
except ImportError:
    _flash_attn_fwd = None

def generate_global_tokens():
    """
    Generate a global token sequence for a context window of 24 tokens,
    consisting of 2 documents:
      - Document 1: tokens 0-6 are regular tokens; token 7 is <eos> (-1)
      - Document 2: tokens 8-22 are regular tokens; token 23 is <eos> (-1)
    Returns a tensor of shape [B, 24] with B = 1.
    """
    global_tokens = []
    for i in range(7):
        global_tokens.append(i)
    global_tokens.append(-1)  # <eos> for doc1
    for i in range(8, 23):
        global_tokens.append(i)
    global_tokens.append(-1)  # <eos> for doc2
    assert len(global_tokens) == 24, f"Expected 24 tokens, got {len(global_tokens)}"
    return torch.tensor([global_tokens], dtype=torch.int64)

def embed_tokens(tokens):
    """
    Embeds tokens into vectors of size 8.
    For demonstration, each non-<eos> token is mapped to a vector of size 8 with all entries equal
    to the token id (as float), and <eos> tokens (represented as -1) remain as -1.
    The output shape is [B, T, 1, 8] and is converted to bfloat16.
    """
    B, T = tokens.shape
    emb = tokens.float().unsqueeze(-1).unsqueeze(-1).expand(B, T, 1, 8)
    return emb.to(torch.bfloat16)

def global_to_local(global_tokens, global_doc_lens, rank, cp_size):
    """
    Maps a global token tensor (shape [B, 24]) to local tokens for a given CP rank.
    For cp_size=2 (num_chunks=4), each document is split into 4 chunks.
    For each document:
      - For rank 0: local tokens = concatenation of chunk0 and chunk3.
      - For rank 1: local tokens = concatenation of chunk1 and chunk2.
    Returns a tensor of shape [B, T_local].
    """
    B, T = global_tokens.shape
    num_chunks = 2 * cp_size  # 4
    local_tokens_list = []
    for b in range(B):
        tokens = global_tokens[b]
        local_doc_list = []
        start = 0
        for L in global_doc_lens[b]:
            doc = tokens[start:start+L]
            base = L // num_chunks
            rem = L % num_chunks
            # For simplicity assume rem==0.
            chunks = [doc[j*base:(j+1)*base] for j in range(num_chunks)]
            if rank == 0:
                local_doc = torch.cat([chunks[0], chunks[3]], dim=0)
            else:
                local_doc = torch.cat([chunks[1], chunks[2]], dim=0)
            local_doc_list.append(local_doc)
            start += L
        local_tokens_list.append(torch.cat(local_doc_list, dim=0).unsqueeze(0))
    return torch.cat(local_tokens_list, dim=0)

def compute_local_doc_lens(global_doc_lens, cp_size, rank):
    """
    Given global doc lengths (e.g. [8,16]), compute local doc lengths for a given CP rank.
    For cp_size=2, each document's local length = global_length // 2.
    """
    return [L // 2 for L in global_doc_lens]

def map_local_to_global(doc_lens, gathered_outputs):
    """
    Reassemble the global output from gathered local outputs.
    
    Args:
      doc_lens: a list of global document lengths [L0, L1, ..., L_{D-1}] for B=1.
                Each Ld is assumed divisible by (2*cp_size).  
      gathered_outputs: a list of length cp_size, where
          gathered_outputs[r] is rank r's local output tensor of shape [1, local_sum, nHeads, headDim]
          with local_sum = sum_d (Ld // cp_size).
    
    For each document d, let each worker’s local output have two shards:
         front = first half (length = Ld//(2*cp_size))
         back  = second half (length = Ld//(2*cp_size))
    The global output for document d is assembled by concatenating:
         [F[0], F[1], …, F[R-1], B[R-1], …, B[0]]
    for each document, and then concatenating the documents in order.
    
    This implementation assumes B = 1.
    """
    cp_size = len(gathered_outputs)
    # For each document, local doc length = Ld_local = Ld // cp_size.
    # And each shard length = Ld_local // 2 = Ld // (2*cp_size).
    shard_len_list = [L // (2 * cp_size) for L in doc_lens]
    # Compute boundaries for each document in the local outputs.
    boundaries = [0]
    for L in doc_lens:
        boundaries.append(boundaries[-1] + (L // cp_size))
    reassembled_docs = []
    for d in range(len(doc_lens)):
        front_shard_len = shard_len_list[d]
        back_shard_len = shard_len_list[d]
        # For each CP worker r, extract its portion for document d.
        doc_shards = []
        for r in range(cp_size):
            local_out = gathered_outputs[r][0]  # shape [local_sum, nHeads, headDim]
            # The portion for document d is from boundaries[d] to boundaries[d+1].
            doc_portion = local_out[boundaries[d]:boundaries[d+1], :, :]
            # Split into front and back halves.
            front = doc_portion[:front_shard_len, :, :]
            back  = doc_portion[front_shard_len:, :, :]
            doc_shards.append((front, back))
        # Assemble global document d: first shards in order, then back shards in reverse.
        front_parts = [doc_shards[r][0] for r in range(cp_size)]
        back_parts  = [doc_shards[r][1] for r in reversed(range(cp_size))]
        doc_global = torch.cat(front_parts + back_parts, dim=0)
        reassembled_docs.append(doc_global)
    global_out = torch.cat(reassembled_docs, dim=0).unsqueeze(0)
    return global_out

def run(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    
    # Initialize NCCL process group.
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    cp_group = dist.new_group(ranks=list(range(world_size)))
    
    global_doc_lens = [8, 16]
    global_tokens = generate_global_tokens()  # shape [1, 24]
    global_tokens = global_tokens.cuda()
    print(f"[Rank {rank}] Global tokens:\n{global_tokens}")
    
    # Compute local tokens.
    local_tokens = global_to_local(global_tokens, [global_doc_lens], rank, world_size)
    print(f"[Rank {rank}] Local tokens (after sharding, shape {local_tokens.shape}):\n{local_tokens}")
    
    # Compute local document lengths.
    local_doc_lens = compute_local_doc_lens(global_doc_lens, world_size, rank)
    print(f"[Rank {rank}] Local document lengths: {local_doc_lens}")
    
    # Embed local tokens.
    q = embed_tokens(local_tokens)  # shape [B, T_local, 1, 8]
    k = q.clone()
    v = q.clone()
    print(f"[Rank {rank}] Local token embeddings (squeezed to [B, T_local, 8]):\n{q.squeeze(2)}")
    
    # Call the per-document forward pass (local mode).
    local_doc_lens_nested = [local_doc_lens]  # e.g. [[4,8]]
    local_out = AttnFuncWithAllGatherPerDocSharding.apply(
        True,         # is_training
        q,            # [B, T_local, 1, 8]
        k,            # k tensor
        v,            # v tensor
        local_doc_lens_nested,     # local document lengths
        0.0,          # dropout_p
        None,         # softmax_scale
        "bshd",       # qkv_format placeholder
        "causal",     # attn_mask_type
        "no_bias",    # attn_bias_type
        None,         # attn_bias
        False,        # deterministic
        False,        # use_fused_attention (using varlen kernel)
        (0, 0),       # window_size tuple
        cp_group,
        torch.cuda.current_stream()
    )
    print(f"[Rank {rank}] Local output from per-document forward pass (shape: {local_out.shape}):\n{local_out}")
    
    # Gather outputs from all CP workers.
    gathered = [torch.empty_like(local_out) for _ in range(world_size)]
    dist.all_gather(gathered, local_out)
    if rank == 0:
        global_out = map_local_to_global(global_doc_lens, gathered)
        print(f"[Rank {rank}] Global output (reassembled, shape {global_out.shape}):\n{global_out.squeeze(2)}")
        
    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 2
    mp.spawn(run, args=(world_size,), nprocs=world_size)
    
    # --- Extra Section: Compare standard flash attention forward outside distributed setting ---
    # This call is made in a non-distributed context.
    global_tokens = generate_global_tokens().cuda()  # shape [1, 24]
    global_q = embed_tokens(global_tokens)  # keep shape [1, 24, 1, 8]
    global_k = global_q.clone()
    global_v = global_q.clone()
    global_out_std = _flash_attn_fwd(
        q=global_q,
        k=global_k,
        v=global_v,
        dropout_p=0.0,
        softmax_scale=global_q.shape[-1] ** -0.5,
        causal=True,
        window_size=(0, 0),
        alibi_slopes=None,
        return_softmax=False
    )
    print(f"Standard flash attention forward (global) output (shape {global_out_std[0].shape}):\n{global_out_std[0]}")



