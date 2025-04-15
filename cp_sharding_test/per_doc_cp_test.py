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

################################################################################
# 1) Random Input Generation
################################################################################

def generate_random_embeddings_with_eos(
    doc_lens: list[int],
    B: int,
    n_heads: int,
    head_dim: int,
    device: torch.device
) -> torch.Tensor:
    """
    Create random bf16 embeddings of shape [B, T, n_heads, head_dim],
    with T = sum(doc_lens). For each document, the *last token* is set to -1
    across all heads/dim (indicating <eos>).

    Example:
      doc_lens = [8,16], so T=24. Then index 7 is eos for doc1,
      and index 23 is eos for doc2 (assuming B=1).
    """
    T = sum(doc_lens)
    global_embeddings = torch.randn(
        (B, T, n_heads, head_dim),
        dtype=torch.bfloat16,
        device=device
    )
    seq_start = 0
    for length in doc_lens:
        eos_idx = seq_start + length - 1
        # Mark the entire vector at [eos_idx, :] as -1
        # shape is [n_heads, head_dim].
        global_embeddings[0, eos_idx, :, :] = -1.0
        seq_start += length
    return global_embeddings

################################################################################
# 2) Per-doc CP Sharding Helpers
################################################################################

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
      global_doc_lens: a list of lists, where global_doc_lens[b] is
         the per-document lengths for sample b. For B=1 and doc_lens=[8,16],
         we have [[8,16]].
      rank: The current CP rank (0 to cp_size-1).
      cp_size: The context-parallel group size.

    Returns:
      A tensor of shape [B, T_local, ...].
      E.g. if cp_size=2, each doc is split into 4 chunks, with rank=0 => chunk0+chunk3,
      rank=1 => chunk1+chunk2, etc.
    """
    num_chunks = 2 * cp_size
    B = global_embeddings.shape[0]
    local_shards_per_sample = []

    for b in range(B):
        sample_embed = global_embeddings[b]  # shape [T, ...]
        doc_lens_for_b = global_doc_lens[b]

        local_doc_shards = []
        seq_start = 0
        for L in doc_lens_for_b:
            chunk_size = L // num_chunks
            chunks = []
            for c in range(num_chunks):
                cstart = seq_start + c * chunk_size
                cend   = seq_start + (c + 1) * chunk_size
                chunks.append(sample_embed[cstart:cend])
            local_front_chunk = chunks[rank]
            local_back_chunk  = chunks[num_chunks - 1 - rank]
            local_doc = torch.cat([local_front_chunk, local_back_chunk], dim=0)
            local_doc_shards.append(local_doc)
            seq_start += L

        local_sample_shard = torch.cat(local_doc_shards, dim=0).unsqueeze(0)
        local_shards_per_sample.append(local_sample_shard)

    local_embeddings = torch.cat(local_shards_per_sample, dim=0)
    return local_embeddings

def compute_local_doc_lens(global_doc_lens, cp_size):
    """
    For each doc length L, local length = L // cp_size.
    """
    # For B=1 scenario, global_doc_lens = [[8,16]], so global_doc_lens[0] = [8,16].
    return [L // cp_size for L in global_doc_lens[0]]

def map_local_to_global(doc_lens, gathered_outputs):
    """
    Reassemble the global output from gathered local outputs.

    Args:
      doc_lens: a list of global document lengths, e.g. [8,16].
      gathered_outputs: list of length cp_size. gathered_outputs[r] has shape
                       [1, local_sum, nHeads, headDim].
    Returns:
      A shape [1, sum(doc_lens), nHeads, headDim].
    """
    cp_size = len(gathered_outputs)
    shard_len_list = [L // (2 * cp_size) for L in doc_lens]
    boundaries = [0]
    for L in doc_lens:
        boundaries.append(boundaries[-1] + (L // cp_size))
    reassembled_docs = []
    for d in range(len(doc_lens)):
        front_shard_len = shard_len_list[d]
        doc_shards = []
        for r in range(cp_size):
            local_out = gathered_outputs[r][0]  # shape: [local_sum, nHeads, headDim]
            doc_portion = local_out[boundaries[d] : boundaries[d+1], :, :]
            front = doc_portion[:front_shard_len, :, :]
            back  = doc_portion[front_shard_len:, :, :]
            doc_shards.append((front, back))
        front_parts = [doc_shards[r][0] for r in range(cp_size)]
        back_parts  = [doc_shards[r][1] for r in reversed(range(cp_size))]
        doc_global = torch.cat(front_parts + back_parts, dim=0)
        reassembled_docs.append(doc_global)
    global_out = torch.cat(reassembled_docs, dim=0).unsqueeze(0)
    return global_out

################################################################################
# 3) Main Distributed Run
################################################################################

def run(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)

    # Initialize NCCL distributed process group.
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    cp_group = dist.new_group(ranks=list(range(world_size)))

    # Suppose we have doc_lens = [8,16] => total T=24 tokens
    doc_lens = [4]
    B = 1
    n_heads = 1  
    head_dim = 8
    device = torch.device(rank)

    # 1) Generate random global embeddings [1, 24, 1, 8] in bf16
    #    Mark each doc's last token with all -1
    global_embeddings = generate_random_embeddings_with_eos(
        doc_lens, B, n_heads, head_dim, device
    )
    # Show shape on each rank (optional)
    print(f"[Rank {rank}] Global embeddings shape: {global_embeddings.shape}", flush=True)

    # 2) Shard for local doc CP
    local_embeddings = global_to_local_doc_embeddings(global_embeddings, [doc_lens], rank, world_size)

    # 3) Prepare Q, K, V
    q = local_embeddings
    k = q.clone()
    v = q.clone()

    # 4) Local doc_lens (each doc is halved if cp_size=2)
    local_doc_lens = compute_local_doc_lens([doc_lens], world_size)

    # 5) Run the per-document forward pass
    local_doc_lens_nested = [local_doc_lens] 
    local_out = AttnFuncWithAllGatherPerDocSharding.apply(
        True,               # is_training
        q,                  # [B, T_local, nHeads, headDim]
        k,
        v,
        local_doc_lens_nested,
        0.0,                # dropout_p
        None,               # softmax_scale
        "bshd",             # qkv_format
        "causal",           # attn_mask_type
        "no_bias",          # attn_bias_type
        None,               # attn_bias
        False,              # deterministic
        False,              # use_fused_attention
        (0, 0),             # window_size
        cp_group,
        torch.cuda.current_stream()
    )

    # 6) All-gather local outputs to rank 0
    gathered = [torch.empty_like(local_out) for _ in range(world_size)]
    dist.all_gather(gathered, local_out)

    # 7) Reassemble the global output on rank 0
    if rank == 0:
        global_out_doc = map_local_to_global(doc_lens, gathered)
        print(f"[Rank 0] Global output (doc-CP) shape: {global_out_doc.shape}", flush=True)

    dist.destroy_process_group()

    #####################################################################
    # 8) Compare with standard flash attention
    #####################################################################
    if rank == 0 and _flash_attn_fwd is not None:
        # We'll reuse the same global_embeddings as Q, K, V
        global_q_std = global_embeddings.clone()
        global_k_std = global_embeddings.clone()
        global_v_std = global_embeddings.clone()

        global_out_std = _flash_attn_fwd(
            q=global_q_std,
            k=global_k_std,
            v=global_v_std,
            dropout_p=0.0,
            softmax_scale=global_q_std.shape[-1] ** -0.5,  # or None
            causal=True,
            window_size=(0, 0),
            alibi_slopes=None,
            return_softmax=False
        )
        global_out_std_tensor = global_out_std[0]  # shape [B, T, nHeads, headDim]
        print(f"[Rank 0] Standard flash attention output shape: {global_out_std_tensor.shape}")

        # 9) Print difference
        if global_out_doc.shape == global_out_std_tensor.shape:
            max_abs_diff = (global_out_doc.float() - global_out_std_tensor.float()).abs().max()
            print(f"[Rank 0] Max abs diff (doc-CP vs standard): {max_abs_diff.item()}", flush=True)
        else:
            print("[Rank 0] Mismatch in shape for doc-CP vs standard flash attention.", flush=True)

    elif rank == 0:
        print("[Rank 0] flash_attn not installed, skipping standard attention comparison.")

if __name__ == "__main__":
    world_size = 2
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)
