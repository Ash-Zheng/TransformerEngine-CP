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
      gathered_outputs: a list of length R (the cp world size), where 
          gathered_outputs[r] is rank r's local output tensor of shape 
          [1, local_sum, nHeads, headDim], with 
          local_sum = sum_d (Ld // cp_size).
    
    For each document d, the local token count per worker is:
         local_doc_len = Ld // cp_size.
    Each worker’s local output for document d is assumed to be the concatenation of two shards,
    each of length shard_len = Ld // (2*cp_size).
    
    The desired global token order for document d is:
         F[0], F[1], …, F[R-1], B[R-1], B[R-2], …, B[0],
    where for each worker r:
         F[r] = first half of worker r’s local output for doc d,
         B[r] = second half of worker r’s local output for doc d.
    
    The function processes each document in order and concatenates the results,
    returning a tensor of shape [1, sum(Ld), nHeads, headDim] that has the tokens in the correct global order.
    
    Note: This implementation assumes B = 1.
    """
    cp_size = len(gathered_outputs)  # Number of CP workers (R)
    
    # Compute cumulative boundaries for each document in each worker's local output.
    # Each document d has local length = Ld_local = Ld // cp_size.
    boundaries = [0]
    for L in doc_lens:
        boundaries.append(boundaries[-1] + (L // (2 * cp_size)) * 2)  
        # Because each doc is split into 2 shards per worker, so total tokens per doc per worker is Ld//cp_size.
    
    # Now, for each document, we reassemble from each worker.
    reassembled_docs = []
    # For each document d, local tokens per worker = Ld_local = Ld // cp_size.
    # And each shard length = shard_len = Ld // (2 * cp_size).
    for d in range(len(doc_lens)):
        shard_len = doc_lens[d] // (2 * cp_size)
        # For each CP worker r, extract its portion for document d from gathered_outputs[r].
        # The local output for document d in worker r is at positions:
        #   start = boundaries[d], end = boundaries[d+1] (same for all workers)
        doc_shards = []  # List of tensors from each worker, but each worker provides two shards.
        for r in range(cp_size):
            # gathered_outputs[r] is shape [1, total_local, nHeads, headDim]
            local_doc = gathered_outputs[r][0, boundaries[d]:boundaries[d+1], :, :]  # shape [Ld_local, nHeads, headDim]
            # Split into front and back halves.
            front = local_doc[:shard_len, :, :]
            back  = local_doc[shard_len:, :, :]
            doc_shards.append((front, back))
        # Now reassemble document d in the following order:
        #   first, all front shards in order r = 0...R-1,
        #   then all back shards in reverse order r = R-1...0.
        front_parts = [doc_shards[r][0] for r in range(cp_size)]
        back_parts  = [doc_shards[r][1] for r in reversed(range(cp_size))]
        doc_global = torch.cat(front_parts + back_parts, dim=0)  # shape [Ld_local * cp_size = Ld, nHeads, headDim]
        reassembled_docs.append(doc_global)
    # Finally, concatenate all document outputs.
    global_out = torch.cat(reassembled_docs, dim=0).unsqueeze(0)  # shape [1, sum(Ld), nHeads, headDim]
    return global_out


def run(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    
    # Initialize NCCL distributed process group.
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
    
    # Call the per-document forward pass (always in local mode).
    local_doc_lens_nested = [local_doc_lens]  # e.g. [[4,8]]
    local_out = AttnFuncWithAllGatherPerDocSharding.apply(
        True,         # is_training
        q,            # [B, T_local, 1, 8] (local tokens)
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
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)
