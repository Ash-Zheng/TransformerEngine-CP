import torch
from itertools import accumulate

from transformer_engine.pytorch.distributed import (
    gather_along_first_dim,
    reduce_scatter_along_first_dim, 
    get_distributed_world_size,
    get_distributed_rank,
)
from transformer_engine.pytorch.utils import (
    nvtx_range_push,
    nvtx_range_pop,
)

from flash_attn.flash_attn_interface import flash_attn_varlen_func, _flash_attn_varlen_backward


def per_seq_kv_shuffle(k_tensor, v_tensor, cp_size):
    chunk_k = k_tensor.chunk(2 * cp_size, dim=0)
    chunk_v = v_tensor.chunk(2 * cp_size, dim=0)

    new_k, new_v = [], []
    for r in range(cp_size):
        new_k.append(chunk_k[r])
        new_k.append(chunk_k[2 * cp_size - 1 - r])
        new_v.append(chunk_v[r])
        new_v.append(chunk_v[2 * cp_size - 1 - r])
    return torch.cat(new_k, dim=0), torch.cat(new_v, dim=0)

def per_seq_kv_unshuffle(k_tensor, v_tensor, cp_size):
     # Split into 2·cp_size equal chunks along dim-0
    k_chunks = k_tensor.chunk(2 * cp_size, dim=0)
    v_chunks = v_tensor.chunk(2 * cp_size, dim=0)

    orig_k, orig_v = [None] * (2 * cp_size), [None] * (2 * cp_size)

    # Shuffle produced:  [ L0, L(2p-1), L1, L(2p-2), L2, … ]
    # We invert that mapping here.
    for r in range(cp_size):
        even_idx  = 2 * r           # position of L r     in shuffle
        odd_idx   = 2 * r + 1       # position of L(2p-1-r)

        orig_k[r]                  = k_chunks[even_idx]
        orig_k[2 * cp_size - 1 - r] = k_chunks[odd_idx]

        orig_v[r]                  = v_chunks[even_idx]
        orig_v[2 * cp_size - 1 - r] = v_chunks[odd_idx]

    return torch.cat(orig_k, dim=0), torch.cat(orig_v, dim=0)

    

class PerSequenceCPAttention(torch.autograd.Function):
    """
    Attention with per‑sequence context parallelism.
    
    Forward path:
      1. allgather local K / V across pipeline ranks
      2. shuffle KV back to global order with `per_seq_kv_shuffle`
      3. for chunk id 0 and 1 : slice KV for this chunk
         and call `flash_attn_varlen_func`
      4. concatenate outputs and store everything for backward
    """

    @staticmethod
    def forward(
        ctx,
        local_q, local_k, local_v,
        cu_seqlens_q_list, cu_seqlens_kv_list,
        max_seqlen_q_list, max_seqlen_kv_list,
        doc_lens,
        context_length,
        dropout_p,
        softmax_scale,
        attn_mask_type,
        cp_group,
        cp_stream
    ):
        nvtx_range_push("PerSequenceCPAttention.fwd")
        assert attn_mask_type == "causal", "Only causal attention is supported"
        assert cp_group is not None, "cp_group must be provided"
        assert cp_stream is not None, "cp_stream must be provided"

        cp_size = get_distributed_world_size(cp_group)
        rank = get_distributed_rank(cp_group)

        # compute prefix_lens, doc_idx_list
        chunk_size = context_length // (2 * cp_size)

        split_doc_lens, prefix_lens = [], []
        cur_len = 0
        for doc_len in doc_lens:
            if cur_len + doc_len <= chunk_size:
                split_doc_lens.append(doc_len)
                prefix_lens.append(0)
                cur_len += doc_len
            else:                                   # need to split the doc
                split_doc_lens.append(chunk_size - cur_len)
                prefix_lens.append(0)

                cu_prefix  = chunk_size - cur_len   # how many tokens already taken
                remain_len = doc_len - cu_prefix
                while remain_len > chunk_size:
                    split_doc_lens.append(chunk_size)
                    prefix_lens.append(cu_prefix)
                    cu_prefix  += chunk_size
                    remain_len -= chunk_size
                if remain_len > 0:
                    split_doc_lens.append(remain_len)
                    prefix_lens.append(cu_prefix)
                    cur_len = remain_len
                else:
                    cur_len = 0
            if cur_len == chunk_size:
                cur_len = 0

        # indices where each chunk starts inside split_doc_lens
        doc_idx_list, acc = [0], 0
        for i, dlen in enumerate(split_doc_lens):
            acc += dlen
            if acc == chunk_size:
                doc_idx_list.append(i + 1)
                acc = 0

        # allgather kv, then shuffle back to global order
        k_global, _ = gather_along_first_dim(local_k, cp_group)
        v_global, _ = gather_along_first_dim(local_v, cp_group)
        k_global, v_global = per_seq_kv_shuffle(k_global, v_global, cp_size)
        print(f"rank {rank} k_global shape: {k_global.shape}, v_global shape: {v_global.shape}")

        # compute forward pass
        outputs, lses = [], []
        q_chunks = local_q.chunk(2, dim=0)
        k_offsets = [] # list of k offsets for each chunk
        for chunk_id in range(2):  # 0, 1
            if chunk_id == 0:
                chunk_index = rank
            else:
                chunk_index = 2 * cp_size - 1 - rank

            this_chunk_docs = split_doc_lens[
                doc_idx_list[chunk_index] : doc_idx_list[chunk_index + 1]
            ]

            # slice kv for this chunk
            k_offset = chunk_index * chunk_size
            doc_id_split = doc_idx_list[chunk_index]
            if prefix_lens[doc_id_split] > 0:
                k_offset -= prefix_lens[doc_id_split]
                this_chunk_docs[0] += prefix_lens[doc_id_split]
                assert k_offset >= 0, f"k_offset < 0 for chunk {chunk_id}"

            k_start = k_offset
            k_end   = k_start + chunk_size
            local_k_slice = k_global[k_start:k_end]
            local_v_slice = v_global[k_start:k_end]

            out, lse, _ = flash_attn_varlen_func(
                q                 = q_chunks[chunk_id],
                k                 = local_k_slice,
                v                 = local_v_slice,
                cu_seqlens_q      = cu_seqlens_q_list [chunk_id],
                cu_seqlens_k      = cu_seqlens_kv_list[chunk_id],
                max_seqlen_q      = max_seqlen_q_list [chunk_id],
                max_seqlen_k      = max_seqlen_kv_list[chunk_id],
                dropout_p         = 0.0,
                softmax_scale     = softmax_scale,
                causal            = True,
                return_attn_probs = True,
            )
            outputs.append(out)
            lses.append(lse)
            k_offsets.append(k_offset)
            print(f"rank {rank} out shape: {out.shape} for chunk {chunk_id}")

        # concatenate chunk-results
        final_out = torch.cat(outputs, dim=0)

        ctx.save_for_backward(
            local_q,
            k_global, v_global,
            *outputs,             
            *lses,
            *cu_seqlens_q_list, *cu_seqlens_kv_list,
            *max_seqlen_q_list, *max_seqlen_kv_list,
        )
        ctx.k_offsets      = k_offsets
        ctx.q_chunk_sizes  = [c.shape[0] for c in q_chunks]
        ctx.dropout_p      = dropout_p
        ctx.softmax_scale  = softmax_scale
        ctx.attn_mask_type = attn_mask_type
        ctx.cp_group       = cp_group
        ctx.cp_stream      = cp_stream
        
        nvtx_range_pop()
        return final_out
    
    @staticmethod
    def backward(ctx, d_out_cat, d_lse_cat):
        """
        Backward pass for PerSequenceCPAttention.
        """
        nvtx_range_push("PerSequenceCPAttention.bwd")

        (
            local_q,
            gathered_k, gathered_v,
            out_L, out_R, 
            lse_L, lse_R,
            cu_q_L, cu_q_R, cu_k_L, cu_k_R,
            maxq_L, maxq_R, maxk_L, maxk_R,
        ) = ctx.saved_tensors

        cp_group   = ctx.cp_group
        k_offsets  = ctx.k_offsets
        (qlen_L, qlen_R) = ctx.q_chunk_sizes
        world_size = get_distributed_world_size(cp_group)
        rank    = get_distributed_rank(cp_group)

        # split grad_out into two chunks
        dq_local = torch.zeros_like(local_q)
        dk_global = torch.zeros_like(gathered_k)
        dv_global = torch.zeros_like(gathered_v)
        
        # split grad-out
        d_out_L, d_out_R   = d_out_cat.split([qlen_L, qlen_R], dim=0)

        # compute dq and dk/dv for each chunk
        for i, (d_out, q_len, out, lse, cu_q, cu_k, max_q, max_k) in enumerate([
            (d_out_L, qlen_L, out_L, lse_L, cu_q_L, cu_k_L, maxq_L, maxk_L),
            (d_out_R, qlen_R, out_R, lse_R, cu_q_R, cu_k_R, maxq_R, maxk_R),
        ]):
            k_start = k_offsets[i]
            k_len   = cu_k[-1].item()
            kv_k    = gathered_k[k_start : k_start + k_len]
            kv_v    = gathered_v[k_start : k_start + k_len]

            dq_chunk = torch.zeros_like(local_q[:q_len])
            dk_chunk = torch.zeros_like(kv_k)
            dv_chunk = torch.zeros_like(kv_v)

            _ = _flash_attn_varlen_backward(
                d_out,
                local_q[ sum(ctx.q_chunk_sizes[:i]) : sum(ctx.q_chunk_sizes[:i+1]) ],
                kv_k, kv_v,
                out,
                lse,
                dq_chunk, dk_chunk, dv_chunk,
                cu_q, cu_k, int(max_q), int(max_k),
                0.0, ctx.softmax_scale, True, (0,0), None, False, None
            )

            dq_local[ sum(ctx.q_chunk_sizes[:i]) : sum(ctx.q_chunk_sizes[:i+1]) ] = dq_chunk
            dk_global[k_start : k_start+k_len] += dk_chunk
            dv_global[k_start : k_start+k_len] += dv_chunk

        # invert the shuffle for dk/dv
        dk_global, dv_global = per_seq_kv_unshuffle(dk_global, dv_global, world_size)

        # now do reduce_scatter for dk/dv
        dk_local, _ = reduce_scatter_along_first_dim(dk_global, cp_group)
        dv_local, _ = reduce_scatter_along_first_dim(dv_global, cp_group)

        nvtx_range_pop()
        return (
            dq_local,            # grad w.r.t. local_q
            dk_local,            # grad w.r.t. local_k
            dv_local,            # grad w.r.t. local_v
            None, None, None, None, None, None, None, None, None
        )

