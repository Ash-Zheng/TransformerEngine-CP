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

def global_to_local_doc_embeddings(global_embeddings: torch.Tensor,
                               global_doc_lens: list[list[int]],
                               rank: int,
                               cp_size: int
                              ) -> torch.Tensor:
    """
    Shard a globally-embedded tensor by documents, returning the local shard
    for a particular CP rank.

    Args:
      global_embeddings: shape [B, T, ...], e.g. [B, T, nHeads, headDim].
         This is the entire global embedding for all documents concatenated.
      global_doc_lens: a list of lists, where global_doc_lens[b] is
         the per-document lengths for sample b. For B=1 and doc_lens=[8,16],
         we have [[8,16]].
      rank: The current CP rank (0 to cp_size-1).
      cp_size: The context-parallel group size.

    Returns:
      A tensor of shape [B, T_local, ...], where T_local = sum of
        (doc_length // cp_size) for each doc in each batch. If cp_size=2,
        each doc is split into 4 chunks (2×cp_size=4). The logic for rank=0
        picks chunk0 + chunk3, rank=1 picks chunk1 + chunk2, etc.
    """
    # For cp_size=2 => 4 total chunks per doc => chunk0..chunk3.
    num_chunks = 2 * cp_size

    B = global_embeddings.shape[0]
    # We'll build up [B, T_local, ...] in a list-of-tensors manner.
    local_shards_per_sample = []

    # We'll track the "sequence" dimension as dim=1 for shape [B, T, ...].
    for b in range(B):
        sample_embed = global_embeddings[b]  # shape [T, ...] for sample b
        doc_lens_for_b = global_doc_lens[b]

        local_doc_shards = []
        seq_start = 0
        for L in doc_lens_for_b:
            # Split doc of length L into num_chunks pieces => chunk_size = L // num_chunks
            # (assuming L is divisible by num_chunks).
            chunk_size = L // num_chunks
            # Collect the sub-embeddings for each chunk
            chunks = []
            for c in range(num_chunks):
                cstart = seq_start + c*chunk_size
                cend   = seq_start + (c+1)*chunk_size
                chunks.append(sample_embed[cstart : cend])

            # For each doc, rank=0 => chunk0 + chunk3, rank=1 => chunk1 + chunk2,
            # or in general rank=r => chunk[r] + chunk[num_chunks-r-1].
            # This matches your earlier doc-chunk logic.
            local_front_chunk = chunks[rank]
            local_back_chunk  = chunks[num_chunks - 1 - rank]
            # Concatenate those pieces for this doc
            local_doc = torch.cat([local_front_chunk, local_back_chunk], dim=0)
            local_doc_shards.append(local_doc)

            seq_start += L  # move past this doc

        # Combine doc shards for sample b => shape [T_local_b, ...]
        local_sample_shard = torch.cat(local_doc_shards, dim=0)
        # Store it as [1, T_local_b, ...]
        local_shards_per_sample.append(local_sample_shard.unsqueeze(0))

    # Finally cat all samples => shape [B, T_local, ...]
    local_embeddings = torch.cat(local_shards_per_sample, dim=0)
    return local_embeddings


def compute_local_doc_lens(global_doc_lens, cp_size):
    """
    Given global doc lengths, compute local doc lengths for a given CP rank.
    For cp_size=2, each document's local length = global_length // 2.
    """
    # For each document length, divide by cp_size.
    return [doc_len // cp_size for doc_len in global_doc_lens[0]]

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
         back:  second half (length = Ld//(2*cp_size))
    The global output for document d is assembled by concatenating:
         [F[0], F[1], …, F[R-1], B[R-1], …, B[0]]
    for each document, and then concatenating the documents in order.
    
    This implementation assumes B = 1.
    """
    cp_size = len(gathered_outputs)
    shard_len_list = [L // (2 * cp_size) for L in doc_lens]
    boundaries = [0]
    for L in doc_lens:
        boundaries.append(boundaries[-1] + (L // cp_size))
    reassembled_docs = []
    for d in range(len(doc_lens)):
        front_shard_len = shard_len_list[d]
        back_shard_len = shard_len_list[d]
        doc_shards = []
        for r in range(cp_size):
            local_out = gathered_outputs[r][0]
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

def run(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)

    global_doc_lens = [[8, 16]]
    global_tokens = generate_global_tokens()  # shape [1, 24]
    global_tokens = global_tokens.cuda()
    global_embeddings = embed_tokens(global_tokens)  # shape [1, 24, 1, 8]
    print(f"[Rank {rank}] Global embeddings (shape {global_embeddings.shape}):\n{global_embeddings}")
    
    # Initialize NCCL distributed process group.
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    cp_group = dist.new_group(ranks=list(range(world_size)))
    
    
    # Compute local embeddings for per-document CP.
    local_embeddings = global_to_local_doc_embeddings(global_embeddings, global_doc_lens, rank, world_size)
    
    # Compute local document lengths.
    local_doc_lens = compute_local_doc_lens(global_doc_lens, world_size)
    print(f"[Rank {rank}] Local document lengths: {local_doc_lens}")
    
    # Embed local tokens.
    # NOTE: We'll set requires_grad_ so we can do backward.
    q = local_embeddings.clone().requires_grad_()  # shape [B, T_local, 1, 8]
    k = q.clone().requires_grad_()
    v = q.clone().requires_grad_()
    print(f"[Rank {rank}] Local token embeddings (shape {q.shape}):\n{q.squeeze(2)}")
    
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
    torch.cuda.synchronize(device=rank)

    # For demonstration: backprop a simple scalar loss from local_out.
    # We'll just sum up local_out and call backward, then print gradients.
    local_loss = local_out.float().sum()
    local_loss.backward()

    print(f"[Rank {rank}] Grad for local Q: {q.grad}")
    print(f"[Rank {rank}] Grad for local K: {k.grad}")
    print(f"[Rank {rank}] Grad for local V: {v.grad}")

    # Gather outputs from all CP workers.
    gathered = [torch.empty_like(local_out) for _ in range(world_size)]
    dist.all_gather(gathered, local_out)
    if rank == 0:
        global_out = map_local_to_global(global_doc_lens[0], gathered)
        print(f"[Rank {rank}] Global output (reassembled, shape {global_out.shape}):\n{global_out.squeeze(2)}")
        
    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 2
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)
    
    # --- Extra Section: Compare standard flash attention forward outside distributed setting ---
    global_tokens = generate_global_tokens().cuda()  # shape [1, 24]
    global_q = embed_tokens(global_tokens).clone().requires_grad_()  # keep shape [1, 24, 1, 8]
    global_k = global_q.clone().requires_grad_()
    global_v = global_q.clone().requires_grad_()
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

    # Similarly, let's do a trivial backward pass on the standard output, printing gradients:
    std_loss = global_out_std[0].float().sum()
    std_loss.backward()

    print("Standard flash attention Q grad:\n", global_q.grad)
    print("Standard flash attention K grad:\n", global_k.grad)
    print("Standard flash attention V grad:\n", global_v.grad)
