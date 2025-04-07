#!/usr/bin/env python
import sys
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# add project root to path 
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from transformer_engine.pytorch.attention import AttnFuncWithCPAndKVAllGather
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
    Each token is mapped to a vector of size 8 with all entries equal to the token id (as float).
    The output shape is [B, T, 1, 8] and is converted to bfloat16.
    """
    B, T = tokens.shape
    emb = tokens.float().unsqueeze(-1).unsqueeze(-1).expand(B, T, 1, 8)
    return emb.to(torch.bfloat16)

def map_local_to_global_seq_custom(doc_lens, gathered_outputs):
    """
    Reassemble the global output from gathered local outputs for per-sequence CP using symmetric reordering.
    
    Args:
      doc_lens: a list of global document lengths [L0, L1, ..., L_{D-1}] for B=1.
                Each L is assumed divisible by (2 * cp_size), so that each CP worker’s local portion has
                length L_local = L // cp_size.
      gathered_outputs: a list of length cp_size, where
          gathered_outputs[r] is the local output tensor from CP worker r with shape [T_local, nHeads, headDim],
          where T_local = sum_d (L_d // cp_size).
    
    For each document d, each worker’s local slice (for doc d) is of length L_local = L_d // cp_size.
    Each such slice is split evenly into two halves:
         front: the first half (length = L_d // (2*cp_size))
         back:  the second half (length = L_d // (2*cp_size))
    The desired global order for document d is:
         [front of worker 0, front of worker 1, …, front of worker (R-1),
          back of worker (R-1), …, back of worker 0]
    The function returns a tensor of shape [1, sum_d L_d, nHeads, headDim] with tokens in the correct global order.
    
    Note: This implementation assumes B = 1.
    """
    cp_size = len(gathered_outputs)
    # For each document, each worker’s local portion length = L_local = L // cp_size.
    # And each shard length = L_local // 2 = L // (2*cp_size).
    shard_len_list = [L // (2 * cp_size) for L in doc_lens]
    # Compute boundaries for each document in the local outputs.
    boundaries = [0]
    for L in doc_lens:
        boundaries.append(boundaries[-1] + (L // cp_size))
    reassembled_docs = []
    for d in range(len(doc_lens)):
        front_shard_len = shard_len_list[d]
        # For each CP worker r, use the entire gathered output (shape [T_local, nHeads, headDim])
        doc_shards = []
        for r in range(cp_size):
            local_out = gathered_outputs[r]  # shape [T_local, nHeads, headDim]
            doc_portion = local_out[boundaries[d]:boundaries[d+1], :, :]
            front = doc_portion[:front_shard_len, :, :]
            back  = doc_portion[front_shard_len:, :, :]
            doc_shards.append((front, back))
        front_parts = [doc_shards[r][0] for r in range(cp_size)]
        back_parts  = [doc_shards[r][1] for r in reversed(range(cp_size))]
        doc_global = torch.cat(front_parts + back_parts, dim=0)
        reassembled_docs.append(doc_global)
    global_out = torch.cat(reassembled_docs, dim=0).unsqueeze(0)
    return global_out

def global_to_local_seq_embeddings(
    global_embeddings: torch.Tensor,
    global_doc_lens: list[list[int]],
    rank: int,
    cp_size: int
) -> torch.Tensor:
    """
    Shard a globally-embedded tensor for a per-sequence CP split,
    analogous to global_to_local_seq_custom but operating on embeddings directly.
    
    For each document length L, assume L is divisible by 2*cp_size.
    Split into 2*cp_size chunks. The local shard for rank r is:
        [chunk[r], chunk[2*cp_size - 1 - r]]
    Concatenate shards across documents in order.

    Args:
      global_embeddings: shape [B, T, ..., ...], e.g. [B, T, nHeads, headDim].
      global_doc_lens: a list of lists of doc lengths
      rank: current CP rank
      cp_size: context-parallel group size

    Returns:
      A tensor [B, T_local, ..., ...], where T_local = sum of (L // cp_size) for each doc.
    """
    B = global_embeddings.shape[0]
    num_chunks = 2 * cp_size
    local_shards_per_sample = []

    for b in range(B):
        sample_embed = global_embeddings[b]  # shape [T, ..., ...]
        doc_lens_for_b = global_doc_lens[b]
        local_doc_shards = []
        seq_start = 0

        for L in doc_lens_for_b:
            chunk_size = L // num_chunks
            # Slice out the doc from [seq_start : seq_start+L]
            doc_embed = sample_embed[seq_start : seq_start + L]
            seq_start += L

            # Split into num_chunks contiguous chunks
            chunks = [doc_embed[i*chunk_size : (i+1)*chunk_size] for i in range(num_chunks)]
            # For rank r => chunk[r] + chunk[num_chunks - 1 - r]
            local_doc = torch.cat([chunks[rank], chunks[num_chunks - 1 - rank]], dim=0)
            local_doc_shards.append(local_doc)

        local_sample_shard = torch.cat(local_doc_shards, dim=0).unsqueeze(0)
        local_shards_per_sample.append(local_sample_shard)

    local_embeddings = torch.cat(local_shards_per_sample, dim=0)
    return local_embeddings

def run(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)


    global_doc_lens = [8, 16] 
    global_tokens = generate_global_tokens()  # shape [1, 24]
    global_tokens = global_tokens.cuda()
    global_embeddings = embed_tokens(global_tokens)  # shape [1, 24, 1, 8]
    print(f"[Rank {rank}] Global embeddings (shape {global_embeddings.shape}):\n{global_embeddings}", flush=True)
    
    # Initialize the NCCL distributed process group.
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    cp_group = dist.new_group(ranks=list(range(world_size)))
    
    # Compute local tokens for per-sequence CP using the custom symmetric mapping.
    local_embeddings = global_to_local_seq_embeddings(global_embeddings, [global_doc_lens], rank, world_size)
    print(f"[Rank {rank}] Per-sequence local tokens (shape {local_embeddings.shape}):\n{local_embeddings}", flush=True)
    
    # Embed local tokens.
    q = local_embeddings  # shape [B, T_local, 1, 8]
    k = q.clone()
    v = q.clone()
    print(f"[Rank {rank}] Local token embeddings (squeezed to [B, T_local, 8]):\n{q.squeeze(2)}", flush=True)

    T_local = q.shape[1]
    cp_size = world_size

    # max_seqlen_q = longest local sequence length for q
    # max_seqlen_kv = longest local sequence length for k and v
    max_seqlen_q = T_local * (cp_size) 
    max_seqlen_kv = T_local * (cp_size)

    # Build cu_seqlens_q
    cu_seqlens_q = torch.tensor([0, max_seqlen_q], device=q.device, dtype=torch.int32)
    cu_seqlens_q_padded = None  # if not used
    
    # Create cumulative sequence lengths for local tokens.
    B, T_local, _, _ = q.shape
    cu_seqlens = torch.arange(0, (B+1)*T_local, step=T_local, dtype=torch.int32, device=q.device)
    
    # Call the per-sequence forward pass.
    out = AttnFuncWithCPAndKVAllGather.apply(
        True,                   # is_training
        q,                      # q: [B, T_local, 1, 8]
        k,                          # k tensor
        v,                      # v tensor
        cu_seqlens_q,           # cu_seqlens_q
        max_seqlen_q,           # max_seqlen_q
        max_seqlen_kv,          # max_seqlen_kv
        cu_seqlens_q_padded,    # cu_seqlens_q_padded (None for now)
        0.0,                    # dropout_p
        None,                   # softmax_scale
        "bshd",                 # qkv_format placeholder
        "causal",               # attn_mask_type
        "no_bias",              # attn_bias_type
        None,                   # attn_bias
        False,                  # deterministic
        False,                  # use_fused_attention
        (0, 0),                 # window_size tuple
        cp_group,
        torch.cuda.current_stream()
    )
    print(f"[Rank {rank}] Local output from per-sequence forward pass (shape: {out.shape}):\n{out}", flush=True)
    
    # Gather outputs from all CP workers.
    gathered = [torch.empty_like(out) for _ in range(world_size)]
    dist.all_gather(gathered, out)
    if rank == 0:
        global_out = map_local_to_global_seq_custom(global_doc_lens, gathered)
        print(f"[Rank {rank}] Global output (reassembled, shape {global_out.shape}):\n{global_out.squeeze(2)}", flush=True)
    
    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 2
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)
    
    # --- Extra Section: Compare standard flash attention forward outside distributed setting ---
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
    print(f"Standard flash attention forward (global) output (shape {global_out_std[0].shape}):\n{global_out_std[0].squeeze(2)}", flush=True)
