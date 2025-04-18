#!/usr/bin/env python
import sys
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from itertools import accumulate
import argparse
import random

parser = argparse.ArgumentParser(description='arguments')
parser.add_argument('--context_length', type=int, default=16) # n * 1024
parser.add_argument('--batch_size', type=int, default=1) 
parser.add_argument('--num_heads', type=int, default=32)
parser.add_argument('--head_dim', type=int, default=128)
parser.add_argument('--avg_doc_len', type=float, default=0.25)
parser.add_argument('--std_doc_len', type=float, default=0.5)
parser.add_argument('--cp_size', type=int, default=2)


# Add project root to path 
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Per-sequence forward pass.
from transformer_engine.pytorch.attention import AttnFuncWithCPAndKVAllGather
# Per-document forward pass.
from per_document_cp_sharding.per_document_cp_sharding import AttnFuncWithAllGatherPerDocSharding


try:
    from flash_attn.flash_attn_interface import _flash_attn_forward as _flash_attn_fwd
    from flash_attn.flash_attn_interface import flash_attn_varlen_func
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
      cp_size: the context-parallel group size.

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

def run(rank, world_size, scenario_doc, scenario_seq, scenario_name):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    
    # Initialize NCCL process group.
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    cp_group = dist.new_group(ranks=list(range(world_size)))
    
    # Use scenario parameters passed from main.
    doc_lens_doc = scenario_doc
    doc_lens_seq = scenario_seq

    B = 1
    n_heads = 8
    head_dim = 64
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

    if rank == 0:
        print("=" * 80, flush=True)
        print(f"Testing {scenario_name}", flush=True)

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
    q_doc = local_doc_embeddings
    k_doc = local_doc_embeddings.clone()
    v_doc = local_doc_embeddings.clone()

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

    ########################################################################
    # Per-Sequence CP
    ########################################################################
    local_seq_embeddings = global_to_local_seq_embeddings(
        global_embeddings,
        [doc_lens_seq],
        rank,
        world_size
    )
    q_seq = local_seq_embeddings.clone()
    k_seq = q_seq.clone()
    v_seq = q_seq.clone()

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
    # if rank == 0:
    #     print(out_seq)
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
        # Print scenario header
        global_out_doc = map_local_to_global(doc_lens_doc, gathered_doc)
        global_out_seq = map_local_to_global_seq_custom(doc_lens_seq, gathered_seq)

        print(f"[Rank 0] Global doc input shape: {global_embeddings.shape}", flush=True)
        print(f"[Rank 0] Global seq input shape: {global_embeddings.shape}", flush=True)

        # Commented out L2 diff print (as requested)
        # diff_doc_seq = torch.norm(global_out_doc.float() - global_out_seq.float())
        # print(f"[Rank 0] Global output L-2 diff (per-doc vs per-seq): {diff_doc_seq.item()}", flush=True)
        max_abs_diff = (global_out_doc.float() - global_out_seq.float()).abs().max()
        print(f"[Rank 0] Global output max abs diff (per-doc vs per-seq): {max_abs_diff.item()}", flush=True)

    dist.destroy_process_group()

    ########################################################################
    # Compare with Standard Flash Attention
    ########################################################################
    if rank == 0 and _flash_attn_fwd is not None:
        # We'll reuse the same global_embeddings as Q, K, V for a single doc
        global_q_std = global_embeddings.clone()
        global_k_std = global_embeddings.clone()
        global_v_std = global_embeddings.clone()

        global_out_std = _flash_attn_fwd(
            q=global_q_std,
            k=global_k_std,
            v=global_v_std,
            dropout_p=0.0,
            softmax_scale=global_q_std.shape[-1] ** -0.5,
            causal=True,
            window_size=(0, 0),
            alibi_slopes=None,
            return_softmax=False
        )
        global_out_std_tensor = global_out_std[0]  # [B, T, nHeads, headDim]

        # diff_seq_std = torch.norm(global_out_seq.float() - global_out_std_tensor.float())
        # print(f"[Rank 0] Global output L-2 diff (per-seq vs standard): {diff_seq_std.item()}", flush=True)
        max_abs_diff_std = (global_out_seq.float() - global_out_std_tensor.float()).abs().max()
        print(f"[Rank 0] Global output max abs diff (per-seq vs standard): {max_abs_diff_std.item()}", flush=True)

        if 'global_out_doc' in locals():
            # diff_doc_std = torch.norm(global_out_doc.float() - global_out_std_tensor.float())
            # print(f"[Rank 0] Global output L-2 diff (per-doc vs standard): {diff_doc_std.item()}", flush=True)
            max_abs_diff_doc_std = (global_out_doc.float() - global_out_std_tensor.float()).abs().max()
            print(f"[Rank 0] Global output max abs diff (per-doc vs standard): {max_abs_diff_doc_std.item()}", flush=True)

    elif rank == 0:
        print("[Rank 0] Warning: flash_attn is not available; skipping standard attention comparison.")


def compute_global_fwd_result( 
    q, k, v, doc_lens
):
    max_len = torch.tensor([max(doc_lens)], dtype=torch.int32).to(q.device)
    max_len_k = max_len.clone()
    cu_seqlens = torch.tensor([0] + list(accumulate(doc_lens)), dtype=torch.int32).to(q.device)
    cu_seqlens_k = cu_seqlens.clone()
    softmax_scale = q.shape[-1] ** -0.5

    output = flash_attn_varlen_func(
        q=q,
        k=k,
        v=v,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_len,
        max_seqlen_k=max_len_k,
        dropout_p=0.0,
        softmax_scale=softmax_scale,
        causal=True,
        return_attn_probs=False
    )

    return output

def per_seq_get_q(q_tensor, cp_size, rank):
    """
    Get the query tensor for per-sequence CP.
    """
    chunk_q = q_tensor.chunk(2 * cp_size, dim=0)

    chunk_1 = rank
    chunk_2 = 2 * cp_size - 1 - rank
    chunk_q = [chunk_q[chunk_1], chunk_q[chunk_2]]
    local_q_tensor = torch.cat(chunk_q, dim=0)

    assert local_q_tensor.shape[0] == q_tensor.shape[0] // cp_size, "local_q_tensor shape mismatch"

    return local_q_tensor

def per_seq_kv_shuffle(k_tensor, v_tensor, cp_size):
    chunk_k = k_tensor.chunk(2 * cp_size, dim=0)
    chunk_v = v_tensor.chunk(2 * cp_size, dim=0)

    new_k_list = []
    new_v_list = []
    for rank in range(cp_size):
        chunk_1 = rank
        chunk_2 = 2 * cp_size - 1 - rank
        new_k_list.append(chunk_k[chunk_1])
        new_k_list.append(chunk_k[chunk_2])
        new_v_list.append(chunk_v[chunk_1])
        new_v_list.append(chunk_v[chunk_2])

    new_k_tensor = torch.cat(new_k_list, dim=0)
    new_v_tensor = torch.cat(new_v_list, dim=0)

    return new_k_tensor, new_v_tensor

def generate_doc_lens(avg_doc_len, std_doc_len, context_length, divide_cp=1):
    """
    Generate a list of document lengths based on average and standard deviation.
    """
    doc_lens = []
    cur_len = 0
    while cur_len <= context_length:
        doc_len = int(torch.normal(avg_doc_len, std_doc_len, size=(1,1)).item() * context_length)

        # Ensure doc_len is a multiple of cp_size
        if divide_cp > 1:
            doc_len = (doc_len // divide_cp) * divide_cp

        if doc_len <= 0:
            continue
        else:
            doc_lens.append(doc_len)
            cur_len += doc_len
    
    # Ensure the last document length does not exceed the context length
    if cur_len > context_length:
        doc_lens[-1] = context_length - sum(doc_lens[:-1])
    if doc_lens[-1] == 0:
        doc_lens = doc_lens[:-1]
    
    assert sum(doc_lens) == context_length, f"Total length {sum(doc_lens)} must equals context length {context_length}."
    for doc_len in doc_lens:
        assert doc_len % divide_cp == 0, f"Document length {doc_len} must be divisible by {divide_cp}."

    return doc_lens


if __name__ == "__main__":
    # Fix the random seed for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    args = parser.parse_args()

    device= torch.device("cuda:0")
    context_length = args.context_length * 1024
    batch_size = args.batch_size
    num_heads = args.num_heads
    head_dim = args.head_dim
    cp_size = args.cp_size

    assert batch_size == 1, "Batch size must be 1 for this test."

    # Generate document lengths
    # doc_lens = generate_doc_lens(args.avg_doc_len, args.std_doc_len, context_length, divide_cp=cp_size * 2)
    # print(f"Generated document lengths: {doc_lens}")
    doc_lens = [16384]

    # Generate token embeddings
    q_tensor = torch.randn((context_length * batch_size, num_heads, head_dim), dtype=torch.bfloat16, device=device) # (S, nHeads, headDim)
    k_tensor = torch.randn((context_length * batch_size, num_heads, head_dim), dtype=torch.bfloat16, device=device)
    v_tensor = torch.randn((context_length * batch_size, num_heads, head_dim), dtype=torch.bfloat16, device=device)

    # simulate cp attention 
    ref_result = compute_global_fwd_result(q_tensor, k_tensor, v_tensor, doc_lens)

    # compute per_seq cp attention
    # local_q_tensor = per_seq_get_q(q_tensor, cp_size, rank=0)
    # local_k_tensor, local_v_tensor = per_seq_kv_shuffle(k_tensor, v_tensor, cp_size) # each rank shares the same k and v

    # get last 16 q token:
    local_q_tensor = q_tensor[-16:, :, :]
    local_ref_result = ref_result[-16:, :, :]

    print(ref_result.shape)
    print(local_ref_result.shape)
    print(local_q_tensor.shape)


    max_seqlen_q = torch.tensor([16], dtype=torch.int32).to(q_tensor.device)
    max_seqlen_k = torch.tensor([16384], dtype=torch.int32).to(q_tensor.device) # last doc
    cu_seqlens_q = torch.tensor([0, 16], dtype=torch.int32).to(q_tensor.device)
    cu_seqlens_k = torch.tensor([0, 16384], dtype=torch.int32).to(q_tensor.device)
    softmax_scale = local_q_tensor.shape[-1] ** -0.5
    local_result = flash_attn_varlen_func(
        q=local_q_tensor,
        k=k_tensor,
        v=v_tensor,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        dropout_p=0.0,
        softmax_scale=softmax_scale,
        causal=True,
        return_attn_probs=False
    )

    torch.testing.assert_close(local_ref_result, local_result)




    # world_size = 8
    # # Define the test scenarios.
    # # Scenario 1: One single document that fills the whole context (128k tokens, using 1024*128)
    # scenario1_doc = [1024 * 128]
    # scenario1_seq = [1024 * 128]
    # scenario1_name = "Scenario 1: One single document of 128k length"

    # # Scenario 2: 128 short documents of 1k length (using 1024 tokens each for exact 128*1024)
    # scenario2_doc = [1024] * 128
    # scenario2_seq = [1024 * 128]
    # scenario2_name = "Scenario 2: 128 short documents of 1k length"

    # # Scenario 3: Mixed - one long document of 100k tokens, then 28 short documents of 1k tokens
    # scenario3_doc = [100 * 1024] + [1024] * 28
    # scenario3_seq = [100 * 1024 + 1024 * 28]
    # scenario3_name = "Scenario 3: Mixed: one long document of 100k tokens, then 28 documents of 1k length"

    # # Scenario 4: Alternating - alternate between long documents of 32k tokens and short documents of 1k tokens.
    # scenario4_doc = [31 * 1024, 1024, 31 * 1024, 1024, 31 * 1024, 1024, 31 * 1024, 1024]
    # scenario4_seq = [31 * 1024 + 1024 + 31 * 1024 + 1024 + 31 * 1024 + 1024 + 31 * 1024 + 1024]
    # scenario4_name = "Scenario 4: Alternating: Long docs of 32k and short docs of 1k tokens"

    # scenarios = [
    #     (scenario1_name, scenario1_doc, scenario1_seq),
    #     # (scenario2_name, scenario2_doc, scenario2_seq),
    #     # (scenario3_name, scenario3_doc, scenario3_seq),
    #     # (scenario4_name, scenario4_doc, scenario4_seq)
    # ]

    # for name, doc_lens, seq_lens in scenarios:
    #     # Spawn the processes for this scenario, passing the parameters explicitly.
    #     mp.spawn(run, args=(world_size, doc_lens, seq_lens, name), nprocs=world_size, join=True)
