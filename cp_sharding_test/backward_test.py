#!/usr/bin/env python
import sys
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# Add project root to path 
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Per-document forward pass.
from per_document_cp_sharding.per_document_cp_sharding import AttnFuncWithAllGatherPerDocSharding
# Per-sequence forward pass.
from transformer_engine.pytorch.attention import AttnFuncWithCPAndKVAllGather

try:
    from flash_attn.flash_attn_interface import _flash_attn_forward as _flash_attn_fwd
except ImportError:
    _flash_attn_fwd = None

###########################################################################
# 1) Sharding Functions
###########################################################################

def global_to_local_doc_embeddings(global_embeddings: torch.Tensor,
                                   global_doc_lens: list[list[int]],
                                   rank: int,
                                   cp_size: int
                                  ) -> torch.Tensor:
    """
    Shard a globally-embedded tensor by documents, returning the local shard
    for a particular CP rank.

    Args:
      global_embeddings: shape [B, T, nHeads, headDim].
      global_doc_lens: a list of lists, where global_doc_lens[b] is
         the per-document lengths for sample b.
      rank: The current CP rank (0 to cp_size-1).
      cp_size: The context-parallel group size.

    Returns:
      A tensor of shape [B, T_local, nHeads, headDim].
    """
    num_chunks = 2 * cp_size
    B = global_embeddings.shape[0]

    local_shards_per_sample = []

    for b in range(B):
        sample_embed = global_embeddings[b]  # shape [T, nHeads, headDim]
        doc_lens_for_b = global_doc_lens[b]

        local_doc_shards = []
        seq_start = 0
        for L in doc_lens_for_b:
            chunk_size = L // num_chunks
            chunks = []
            for c in range(num_chunks):
                cstart = seq_start + c * chunk_size
                cend = seq_start + (c + 1) * chunk_size
                chunks.append(sample_embed[cstart : cend])

            local_front_chunk = chunks[rank]
            local_back_chunk = chunks[num_chunks - 1 - rank]
            local_doc = torch.cat([local_front_chunk, local_back_chunk], dim=0)
            local_doc_shards.append(local_doc)

            seq_start += L

        local_sample_shard = torch.cat(local_doc_shards, dim=0).unsqueeze(0)
        local_shards_per_sample.append(local_sample_shard)

    local_embeddings = torch.cat(local_shards_per_sample, dim=0)
    return local_embeddings

def global_to_local_seq_embeddings(
    global_embeddings: torch.Tensor,
    global_doc_lens: list[list[int]],
    rank: int,
    cp_size: int
) -> torch.Tensor:
    """
    Shard a globally-embedded tensor for a per-sequence CP split.
    
    For each document length L, assume L is divisible by 2*cp_size.
    Split into 2*cp_size chunks. The local shard for rank r is:
        [chunk[r], chunk[2*cp_size - 1 - r]]
    Concatenate shards across documents in order.

    Args:
      global_embeddings: shape [B, T, nHeads, headDim].
      global_doc_lens: a list of lists of doc lengths
      rank: current CP rank.
      cp_size: context-parallel group size.

    Returns:
      [B, T_local, nHeads, headDim].
    """
    B = global_embeddings.shape[0]
    num_chunks = 2 * cp_size
    local_shards_per_sample = []

    for b in range(B):
        sample_embed = global_embeddings[b]  # shape [T, nHeads, headDim]
        doc_lens_for_b = global_doc_lens[b]
        local_doc_shards = []
        seq_start = 0

        for L in doc_lens_for_b:
            chunk_size = L // num_chunks
            doc_embed = sample_embed[seq_start : seq_start + L]
            seq_start += L

            chunks = [doc_embed[i * chunk_size : (i+1)*chunk_size]
                      for i in range(num_chunks)]
            local_doc = torch.cat([chunks[rank], chunks[num_chunks - 1 - rank]], dim=0)
            local_doc_shards.append(local_doc)

        local_sample_shard = torch.cat(local_doc_shards, dim=0).unsqueeze(0)
        local_shards_per_sample.append(local_sample_shard)

    local_embeddings = torch.cat(local_shards_per_sample, dim=0)
    return local_embeddings

###########################################################################
# 2) Reassembly Helpers (unchanged from original)
###########################################################################

def compute_local_doc_lens(global_doc_lens, cp_size, rank):
    """
    For per-document CP, each document's local length = global_length // cp_size.
    """
    return [L // cp_size for L in global_doc_lens]

def map_local_to_global(doc_lens, gathered_outputs):
    """
    Reassemble the global output from gathered local outputs for per-document CP.
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

def map_local_to_global_seq_custom(doc_lens, gathered_outputs):
    """
    Reassemble the global output from gathered local outputs for per-sequence CP using symmetric reordering.
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
            local_out = gathered_outputs[r]  # shape: [T_local, nHeads, headDim]
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

###########################################################################
# 3) Global Embeddings Generation
###########################################################################

def generate_global_embeddings(
    doc_lens: list[int],
    B: int,
    n_heads: int,
    head_dim: int,
    device: torch.device
) -> torch.Tensor:
    """
    Create random bf16 embeddings [B, T, n_heads, head_dim],
    where T = sum(doc_lens). Mark each document's *last token* as -1.0
    across all heads/dim (indicating <eos>).

    doc_lens: list of doc lengths
    B: batch size (usually 1 here).
    n_heads: number of heads.
    head_dim: dimension per head.
    device: GPU device.

    Returns:
      global_embeddings: [B, T, n_heads, head_dim], dtype=bfloat16
    """
    T = sum(doc_lens)
    global_embeddings = torch.randn(
        (B, T, n_heads, head_dim),
        dtype=torch.bfloat16,
        device=device
    )
    # Mark <eos> for each doc's last token
    seq_start = 0
    for length in doc_lens:
        eos_idx = seq_start + length - 1
        global_embeddings[0, eos_idx, :, :] = -1.0
        seq_start += length

    return global_embeddings

###########################################################################
# 4) Main Distributed Run
###########################################################################

def run(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    
    # Initialize NCCL process group.
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    cp_group = dist.new_group(ranks=list(range(world_size)))
    
    # We'll define 29 documents: one long doc of 100k tokens, plus 28 short docs of 1k each.
    # So total length T = 128,000
    # doc_lens_doc = [100000] + [1000] * 28  # 100k + 28k => 128k
    # # For the per-sequence approach, we treat all 128k tokens as a single doc
    # doc_lens_seq = [128000]
    doc_lens_doc = [16]
    doc_lens_seq = [16]

    B = 1
    n_heads = 1
    head_dim = 8
    device = torch.device(rank)

    if rank == 0:
        # Create the global embeddings on rank 0 only:
        global_embeddings = generate_global_embeddings(
            doc_lens_doc, B, n_heads, head_dim, device
        )
    else:
        # Allocate an empty tensor of the correct size on other ranks
        T = sum(doc_lens_doc)
        global_embeddings = torch.empty((B, T, n_heads, head_dim),
                                        dtype=torch.bfloat16,
                                        device=device)

    # Now broadcast from rank 0 to all ranks:
    dist.broadcast(global_embeddings, src=0)

    ########################################################################
    # Per-Document CP
    ########################################################################
    # Shard
    local_doc_embeddings = global_to_local_doc_embeddings(
        global_embeddings,
        [doc_lens_doc],  # shape: [B, #docs]
        rank,
        world_size
    )
    # We pass the same local embeddings as Q, K, V
    q_doc = local_doc_embeddings.clone().detach().requires_grad_()
    k_doc = local_doc_embeddings.clone().detach().requires_grad_()
    v_doc = local_doc_embeddings.clone().detach().requires_grad_()

    # Compute local doc_lens
    local_doc_lens = compute_local_doc_lens(doc_lens_doc, world_size, rank)
    local_doc_lens_nested = [local_doc_lens]

    out_doc = AttnFuncWithAllGatherPerDocSharding.apply(
        True,
        q_doc,
        k_doc,
        v_doc,
        local_doc_lens_nested,
        0.0,
        global_embeddings.shape[-1] ** -0.5,
        "bshd",
        "causal",
        "no_bias",
        None,
        False,
        False,
        (0, 0),
        cp_group,
        torch.cuda.current_stream()
    )

    # Backward
    grad_out_doc = torch.randn_like(out_doc, device=device)
    out_doc.backward(grad_out_doc)
    dq_doc_local = q_doc.grad
    dk_doc_local = k_doc.grad
    dv_doc_local = v_doc.grad

    # Gather local grads
    gathered_dq_doc = [torch.empty_like(dq_doc_local) for _ in range(world_size)]
    gathered_dk_doc = [torch.empty_like(dk_doc_local) for _ in range(world_size)]
    gathered_dv_doc = [torch.empty_like(dv_doc_local) for _ in range(world_size)]
    dist.all_gather(gathered_dq_doc, dq_doc_local, group=cp_group)
    dist.all_gather(gathered_dk_doc, dk_doc_local, group=cp_group)
    dist.all_gather(gathered_dv_doc, dv_doc_local, group=cp_group)

    ########################################################################
    # Per-Sequence CP
    ########################################################################
    local_seq_embeddings = global_to_local_seq_embeddings(
        global_embeddings,
        [doc_lens_seq],
        rank,
        world_size
    )
    q_seq = local_seq_embeddings.clone().detach().requires_grad_()
    k_seq = local_seq_embeddings.clone().detach().requires_grad_()
    v_seq = local_seq_embeddings.clone().detach().requires_grad_()

    T_local_seq = q_seq.shape[1]
    # maximum local doc length
    max_seqlen_q = T_local_seq * world_size
    max_seqlen_kv = max_seqlen_q

    cu_seqlens_q = torch.tensor([0, max_seqlen_q], device=device, dtype=torch.int32)
    cu_seqlens_q_padded = None

    out_seq = AttnFuncWithCPAndKVAllGather.apply(
        True,
        q_seq,
        k_seq,
        v_seq,
        cu_seqlens_q,
        max_seqlen_q,
        max_seqlen_kv,
        cu_seqlens_q_padded,
        0.0,
        global_embeddings.shape[-1] ** -0.5,
        "bshd",
        "causal",
        "no_bias",
        None,
        False,
        False,
        (0, 0),
        cp_group,
        torch.cuda.current_stream()
    )
    # Backward
    grad_out_seq = torch.randn_like(out_seq, device=device)
    out_seq.backward(grad_out_seq)
    dq_seq_local = q_seq.grad
    dk_seq_local = k_seq.grad
    dv_seq_local = v_seq.grad

    # Gather local grads
    gathered_dq_seq = [torch.empty_like(dq_seq_local) for _ in range(world_size)]
    gathered_dk_seq = [torch.empty_like(dk_seq_local) for _ in range(world_size)]
    gathered_dv_seq = [torch.empty_like(dv_seq_local) for _ in range(world_size)]
    dist.all_gather(gathered_dq_seq, dq_seq_local, group=cp_group)
    dist.all_gather(gathered_dk_seq, dk_seq_local, group=cp_group)
    dist.all_gather(gathered_dv_seq, dv_seq_local, group=cp_group)

    gathered_input_seq = [torch.empty_like(local_seq_embeddings) for _ in range(world_size)]
    dist.all_gather(gathered_input_seq, q_seq)

    ########################################################################
    # Gather outputs and reassemble on rank 0
    ########################################################################
    gathered_doc = [torch.empty_like(out_doc) for _ in range(world_size)]
    gathered_seq = [torch.empty_like(out_seq) for _ in range(world_size)]
    dist.all_gather(gathered_doc, out_doc)
    dist.all_gather(gathered_seq, out_seq)

    if rank == 0:
        # Map outputs
        global_out_doc = map_local_to_global(doc_lens_doc, gathered_doc)
        global_out_seq = map_local_to_global_seq_custom(doc_lens_seq, gathered_seq)

        # Map gradients
        global_dq_doc = map_local_to_global(doc_lens_doc, gathered_dq_doc)
        global_dk_doc = map_local_to_global(doc_lens_doc, gathered_dk_doc)
        global_dv_doc = map_local_to_global(doc_lens_doc, gathered_dv_doc)
        global_dq_seq = map_local_to_global(doc_lens_seq, gathered_dq_seq)
        global_dk_seq = map_local_to_global(doc_lens_seq, gathered_dk_seq)
        global_dv_seq = map_local_to_global(doc_lens_seq, gathered_dv_seq)

        print(f"[Rank 0] Global doc output shape: {global_out_doc.shape}")
        print(f"[Rank 0] Global seq output shape: {global_out_seq.shape}")

        diff_doc_seq = torch.norm(global_out_doc.float() - global_out_seq.float())
        print(f"[Rank 0] Global output L-2 diff (per-doc vs per-seq): {diff_doc_seq.item()}", flush=True)
        max_abs_diff = (global_out_doc.float() - global_out_seq.float()).abs().max()
        print(f"[Rank 0] Global output max abs diff (per-doc vs per-seq): {max_abs_diff.item()}", flush=True)

        # Compare backward results
        max_abs_diff = (global_dq_doc.float() - global_dq_seq.float()).abs().max()
        print(f"[Rank 0] Global dq doc vs seq max abs diff: {max_abs_diff.item()}", flush=True)
        max_abs_diff = (global_dk_doc.float() - global_dk_seq.float()).abs().max()
        print(f"[Rank 0] Global dk doc vs seq max abs diff: {max_abs_diff.item()}", flush=True)
        max_abs_diff = (global_dv_doc.float() - global_dv_seq.float()).abs().max()
        print(f"[Rank 0] Global dv doc vs seq max abs diff: {max_abs_diff.item()}", flush=True)
        print(f"Diff tensor for dv_doc: {global_dv_doc.float() - global_dv_seq.float()}", flush=True)

    dist.destroy_process_group()

    ########################################################################
    # Compare with Standard Flash Attention
    ########################################################################
    # if rank == 0 and _flash_attn_fwd is not None:
    #     # We'll reuse the same global_embeddings as Q, K, V for a single doc
    #     global_q_std = global_embeddings.clone()
    #     global_k_std = global_embeddings.clone()
    #     global_v_std = global_embeddings.clone()

    #     # print(f"[Rank 0] Global Q shape: {global_q_std.shape}, tensor: {global_q_std.squeeze(2)}", flush=True)

    #     # print(f"All input tensors are same: {torch.equal(global_q_std, global_embeddings)}", flush=True)

    #     global_out_std = _flash_attn_fwd(
    #         q=global_q_std,
    #         k=global_k_std,
    #         v=global_v_std,
    #         dropout_p=0.0,
    #         softmax_scale=global_q_std.shape[-1] ** -0.5,
    #         causal=True,
    #         window_size=(0, 0),
    #         alibi_slopes=None,
    #         return_softmax=False
    #     )
    #     global_out_std_tensor = global_out_std[0]  # [B, T, nHeads, headDim]

    #     # print(global_out_std_tensor - global_out_seq)

    #     diff_seq_std = torch.norm(global_out_seq.float() - global_out_std_tensor.float())
    #     print(f"[Rank 0] Global output L-2 diff (per-seq vs standard): {diff_seq_std.item()}", flush=True)
    #     max_abs_diff_std = (global_out_seq.float() - global_out_std_tensor.float()).abs().max()
    #     print(f"[Rank 0] Global output max abs diff (per-seq vs standard): {max_abs_diff_std.item()}", flush=True)

    #     if 'global_out_doc' in locals():
    #         diff_doc_std = torch.norm(global_out_doc.float() - global_out_std_tensor.float())
    #         print(f"[Rank 0] Global output L-2 diff (per-doc vs standard): {diff_doc_std.item()}", flush=True)
    #         max_abs_diff_doc_std = (global_out_doc.float() - global_out_std_tensor.float()).abs().max()
    #         print(f"[Rank 0] Global output max abs diff (per-doc vs standard): {max_abs_diff_doc_std.item()}", flush=True)

    # elif rank == 0:
    #     print("[Rank 0] Warning: flash_attn is not available; skipping standard attention comparison.")

if __name__ == "__main__":
    world_size = 2
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)
