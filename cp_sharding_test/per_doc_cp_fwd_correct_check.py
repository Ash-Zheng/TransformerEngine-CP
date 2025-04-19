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
parser.add_argument('--fix_seed', type=int, default=1)


# Add project root to path 
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Per-sequence forward pass.
from transformer_engine.pytorch.attention import AttnFuncWithCPAndKVAllGather
# Per-document forward pass.
from per_document_cp_sharding.per_document_cp_sharding import AttnFuncWithAllGatherPerDocSharding


from flash_attn.flash_attn_interface import (
    _flash_attn_varlen_forward,
    _flash_attn_varlen_backward,
)
from cp_utils import (
    generate_doc_lens,
    compute_per_doc_cp_shard_doc_len,
    compute_per_doc_metadate,
    get_per_doc_local_result,
    kv_shuffle_for_per_doc_cp,
)


def compute_global_fwd_result( 
    q, k, v, doc_lens, softmax_scale=None
):
    max_len = torch.tensor([max(doc_lens)], dtype=torch.int32).to(q.device)
    max_len_k = max_len.clone()
    cu_seqlens = torch.tensor([0] + list(accumulate(doc_lens)), dtype=torch.int32).to(q.device)
    cu_seqlens_k = cu_seqlens.clone()
    softmax_scale = q.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale

    out, lse, _, _, _, _, _, _ = _flash_attn_varlen_forward(
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
        window_size=(0, 0),
        return_softmax=False,
        alibi_slopes=None,
    )

    return out, lse



if __name__ == "__main__":
    args = parser.parse_args()
    
    if args.fix_seed:
        # Fix the random seed for reproducibility
        random.seed(42)
        torch.manual_seed(42)

    device= torch.device("cuda:0")
    context_length = args.context_length * 1024
    batch_size = args.batch_size
    num_heads = args.num_heads
    head_dim = args.head_dim
    cp_size = args.cp_size
    softmax_scale = head_dim ** -0.5

    assert batch_size == 1, "Batch size must be 1 for this test."

    # Generate document lengths
    doc_lens = generate_doc_lens(args.avg_doc_len, args.std_doc_len, context_length, divide_cp=cp_size * 2)
    print(f"Generated document lengths: {doc_lens}")

    # Generate token embeddings
    q_tensor = torch.randn((context_length * batch_size, num_heads, head_dim), dtype=torch.bfloat16, device=device) # (S, nHeads, headDim)
    k_tensor = torch.randn((context_length * batch_size, num_heads, head_dim), dtype=torch.bfloat16, device=device)
    v_tensor = torch.randn((context_length * batch_size, num_heads, head_dim), dtype=torch.bfloat16, device=device)

    # simulate cp attention 
    ref_out, ref_lse = compute_global_fwd_result(q_tensor, k_tensor, v_tensor, doc_lens, softmax_scale)
    # to save local out and lse
    rank_chunk_fwd_state = [[None, None] for _ in range(cp_size)]

    doc_shards = compute_per_doc_cp_shard_doc_len(doc_lens, context_length, cp_size)

    # compute per_seq cp attention
    for test_rank_id in range(cp_size):
        for test_chunk_id in range(2):
            print(f"Compute Rank {test_rank_id}, Chunk {test_chunk_id}:")
            local_q, local_k, local_v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k = compute_per_doc_metadate(
                context_length, 
                q_tensor, 
                k_tensor, 
                v_tensor, 
                doc_lens, 
                doc_shards, 
                cp_size, 
                rank=test_rank_id, 
                chunk_id=test_chunk_id
            )

            local_out, local_lse, _, _, _, _, _, _ = _flash_attn_varlen_forward(
                q=local_q,
                k=local_k,
                v=local_v,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                dropout_p=0.0,
                softmax_scale=softmax_scale,
                causal=True,
                window_size=(0, 0),
                alibi_slopes=None,
                return_softmax=False,
            )

            local_ref = get_per_doc_local_result(context_length, ref_out, doc_lens, doc_shards, cp_size, rank=test_rank_id, chunk_id=test_chunk_id)
            torch.testing.assert_close(local_ref, local_out)
            print(f"Rank {test_rank_id}, Chunk {test_chunk_id}: Forward Pass Passed Tests", flush=True)

    
