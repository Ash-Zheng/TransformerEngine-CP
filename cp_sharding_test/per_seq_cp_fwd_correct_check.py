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
    flash_attn_varlen_func,
)


def compute_global_fwd_result( 
    q, k, v, doc_lens, softmax_scale=None
):
    max_len = torch.tensor([max(doc_lens)], dtype=torch.int32).to(q.device)
    max_len_k = max_len.clone()
    cu_seqlens = torch.tensor([0] + list(accumulate(doc_lens)), dtype=torch.int32).to(q.device)
    cu_seqlens_k = cu_seqlens.clone()
    softmax_scale = q.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale

    output, lse, _ = flash_attn_varlen_func(
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
        return_attn_probs=True
    )

    return output, lse # return out, lse

def compute_per_seq_metadate(context_length, q, k, v, doc_lens, cp_size, rank, chunk_id):
    """
    Compute the cumulative sequence lengths for per-sequence CP.
    """
    # ============== Split doc lens for sequence sharding =================
    chunk_size = context_length // (2 * cp_size)

    split_doc_lens = []
    prefix_lens = []
    cur_length = 0
    for i, doc_len in enumerate(doc_lens):
        if cur_length + doc_len <= chunk_size: 
            split_doc_lens.append(doc_len)
            prefix_lens.append(0)
            cur_length += doc_len
        else: # split the document
            split_doc_lens.append(chunk_size - cur_length)
            prefix_lens.append(0)
            cu_prefix = chunk_size - cur_length
            remained_length = doc_len - (chunk_size - cur_length)
            while remained_length > chunk_size:
                split_doc_lens.append(chunk_size)
                prefix_lens.append(cu_prefix)
                cu_prefix += chunk_size
                remained_length -= chunk_size
            if remained_length > 0:
                split_doc_lens.append(remained_length)
                prefix_lens.append(cu_prefix)
                cur_length = remained_length
            else:
                cur_length = 0
        
        if cur_length == chunk_size:
            cur_length = 0
    # print(f"split_doc_lens: {split_doc_lens}")
    assert sum(split_doc_lens) == context_length, f"Total length {sum(split_doc_lens)} must equals context length {context_length}."
    
    cur_offset = 0
    doc_idx_list = [0] # to record the document index for each chunk
    for i, doc_len in enumerate(split_doc_lens):
        cur_length += doc_len
        if cur_length == chunk_size:
            doc_idx_list.append(i + 1)
            cur_length = 0
        elif cur_length > chunk_size:
            assert False, "cur_length > chunk_size, this should not happen."
        
    for i in range(len(doc_idx_list)-1):
        assert sum(split_doc_lens[doc_idx_list[i]:doc_idx_list[i+1]]) == chunk_size, f"error doc per chunk"
    
    # ============== Compute metadata =================
    if chunk_id == 0:
        chunk_index = rank
    else:
        chunk_index = 2 * cp_size - 1 - rank
    
    this_chunk_docs = split_doc_lens[doc_idx_list[chunk_index]:doc_idx_list[chunk_index+1]]
    local_q = q_tensor.chunk(2 * cp_size, dim=0)[chunk_index]
    cu_seqlens_q = torch.tensor([0] + list(accumulate(this_chunk_docs)), dtype=torch.int32).to(q.device)
    max_seqlen_q = torch.tensor([max(this_chunk_docs)], dtype=torch.int32).to(q.device)

    # check if the first doc is splitted
    k_offset = chunk_index * chunk_size
    doc_id_split = doc_idx_list[chunk_index]
    if prefix_lens[doc_id_split] > 0:
        k_offset -= prefix_lens[doc_id_split]
        this_chunk_docs[0] += prefix_lens[doc_id_split]
        assert k_offset >= 0, f"error k_offset {k_offset} < 0"
    local_k = k_tensor[k_offset:(chunk_index+1) * chunk_size, :, :]
    local_v = v_tensor[k_offset:(chunk_index+1) * chunk_size, :, :]
    cu_seqlens_k = torch.tensor([0] + list(accumulate(this_chunk_docs)), dtype=torch.int32).to(q.device)
    max_seqlen_k = torch.tensor([max(this_chunk_docs)], dtype=torch.int32).to(q.device)
   
    return local_q, local_k, local_v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, k_offset


def get_local_ref_result(global_ref_result, cp_size, rank, chunk_id):
    """
    Get the local reference result for the current rank and chunk id.
    """
    chunk_size = global_ref_result.shape[0] // (2 * cp_size)
    if chunk_id == 0:
        chunk_index = rank
    else:
        chunk_index = 2 * cp_size - 1 - rank

    local_ref_result = global_ref_result[chunk_index * chunk_size:(chunk_index + 1) * chunk_size, :, :]
    return local_ref_result

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

    # to save forward data
    rank_chunk_fwd_state = [[None, None] for _ in range(cp_size)]

    # compute per_seq cp attention
    for test_rank_id in range(cp_size):
        for test_chunk_id in range(2):
            local_q, local_k, local_v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, k_offset = compute_per_seq_metadate(
                context_length, 
                q_tensor, 
                k_tensor, 
                v_tensor, 
                doc_lens, 
                cp_size, 
                rank=test_rank_id, 
                chunk_id=test_chunk_id
            )
            if test_chunk_id == 0:
                chunk_index = test_rank_id
            else:
                chunk_index = 2 * cp_size - 1 - test_rank_id
            q_offset = chunk_index * (context_length // (2 * cp_size))
            local_q_len = local_q.shape[0]
            local_k_len = local_k.shape[0]

            local_out, local_lse, _ = flash_attn_varlen_func(
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
                return_attn_probs=True
            )

            rank_chunk_fwd_state[test_rank_id][test_chunk_id] = {
                "local_q": local_q,
                "local_k": local_k,
                "local_v": local_v,
                "cu_seqlens_q": cu_seqlens_q,
                "cu_seqlens_k": cu_seqlens_k,
                "max_seqlen_q": max_seqlen_q,
                "max_seqlen_k": max_seqlen_k,
                "q_offset": q_offset,
                "k_offset": k_offset,
                "q_len": local_q_len,
                "k_len": local_k_len,
                "local_out": local_out,
                "local_lse": local_lse,
            }

            # compare result
            local_ref_result = get_local_ref_result(ref_out, cp_size, rank=test_rank_id, chunk_id=test_chunk_id)
            torch.testing.assert_close(local_ref_result, local_out)
            print(f"Rank {test_rank_id}, Chunk {test_chunk_id}: Forward Pass Passed Tests", flush=True)


    # backward pass test
    max_len    = torch.tensor([max(doc_lens)], dtype=torch.int32, device=q_tensor.device)
    cu_seqlens = torch.tensor([0] + list(accumulate(doc_lens)), dtype=torch.int32, device=q_tensor.device)
    d_out = torch.randn_like(ref_out, dtype=torch.bfloat16, device=q_tensor.device)

    dq_ref = torch.zeros_like(q_tensor)
    dk_ref = torch.zeros_like(k_tensor)
    dv_ref = torch.zeros_like(v_tensor)
    
    # compute reference global backward
    _flash_attn_varlen_backward(
        d_out,                      # dout
        q_tensor, k_tensor, v_tensor,
        ref_out,                    # out
        ref_lse,                    # softmax_lse
        dq_ref, dk_ref, dv_ref,
        cu_seqlens, cu_seqlens,     # cu_seqlens_q / cu_seqlens_k
        int(max_len), int(max_len), # max_seqlen_q / max_seqlen_k
        0.0,                        # dropout_p
        softmax_scale,
        True,                       # causal
        (0, 0),                     # window_size
        None,                       # alibi_slopes
        False,                      # deterministic
        None,                       # rng_state
    )
    
    # compute per-sequence backward
    chunk_size = context_length // (2 * cp_size)

    # to store grad results
    dq_computed = torch.zeros_like(q_tensor)
    dk_computed = torch.zeros_like(k_tensor)
    dv_computed = torch.zeros_like(v_tensor)

    for test_rank_id in range(cp_size):
        for test_chunk_id in range(2):
            info = rank_chunk_fwd_state[test_rank_id][test_chunk_id]

            local_q = info["local_q"]
            local_k = info["local_k"]
            local_v = info["local_v"]
            local_out = info["local_out"]
            local_lse = info["local_lse"]
            cu_seqlens_q = info["cu_seqlens_q"]
            cu_seqlens_k = info["cu_seqlens_k"]
            max_seqlen_q = info["max_seqlen_q"]
            max_seqlen_k = info["max_seqlen_k"]

            q_offset = info["q_offset"]
            k_offset = info["k_offset"]
            q_len    = info["q_len"]
            k_len    = info["k_len"]

            # slicing
            d_out_chunk = d_out[q_offset : q_offset + q_len]

            # allocate grad buffers
            dq_local = torch.zeros_like(local_q)
            dk_local = torch.zeros_like(local_k)
            dv_local = torch.zeros_like(local_v)

            # backward kernel
            _flash_attn_varlen_backward(
                d_out_chunk,
                local_q, local_k, local_v,
                local_out,
                local_lse,
                dq_local, dk_local, dv_local,
                cu_seqlens_q, cu_seqlens_k,
                int(max_seqlen_q), int(max_seqlen_k),
                0.0,
                softmax_scale,
                True,
                (0, 0),
                None,
                False,
                None,
            )

            # accumulate gradients
            dq_computed [q_offset : q_offset + q_len] += dq_local
            dk_computed [k_offset : k_offset + k_len] += dk_local
            dv_computed [k_offset : k_offset + k_len] += dv_local

    # compare results
    torch.testing.assert_close(dq_ref, dq_computed)
    torch.testing.assert_close(dk_ref, dk_computed)
    torch.testing.assert_close(dv_ref, dv_computed)

    print(f"Backward Pass Passed Tests", flush=True)
    print("All tests passed!", flush=True)

