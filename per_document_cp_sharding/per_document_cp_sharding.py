import torch
from torch.autograd import Function
import math
import torch.distributed as dist

from transformer_engine.pytorch.distributed import (
    get_distributed_world_size,
    get_distributed_rank,
)
from transformer_engine.pytorch.utils import (
    nvtx_range_push,
    nvtx_range_pop,
)

DEBUG = True
def debug_log(msg):
    if DEBUG:
        print(msg)

class AttnFuncWithCPAndPerDocKVAllGather(Function):
    """
    Per-document CP sharding implementation with enhanced logging for debugging.
    
    In the forward pass, we pad each document (if needed), compute balanced boundaries,
    and reshape the padded tensor into N equal chunks. We store these padded tensors
    and boundaries for use in the backward pass.
    
    The backward pass then uses these stored tensors and boundaries to reconstruct the
    same views as in the forward pass, ensuring that gradient slices are computed on the
    exact same “chunked” tensors.
    """
    @staticmethod
    def forward(ctx,
                is_training,
                q,         # [B, S, num_heads, head_dim]
                k,         # [B, S, num_heads, head_dim]
                v,         # [B, S, num_heads, head_dim]
                doc_lens,  # list or 1D tensor; sum(doc_lens)==S
                dropout_p,
                softmax_scale,
                qkv_format,
                attn_mask_type, 
                attn_bias_type, 
                attn_bias,
                deterministic,
                use_fused_attention,  # must be False
                window_size,          # unused here
                cp_group,
                cp_stream):
        nvtx_range_push("AttnFuncWithCPAndPerDocKVAllGather.forward")
        orig_shape = q.shape  # [B, S, num_heads, head_dim]
        B, S, num_heads, head_dim = orig_shape
        D = num_heads * head_dim
        ctx.orig_shape = orig_shape
        ctx.num_heads = num_heads
        ctx.head_dim = head_dim

        # merge head dimensions
        q = q.view(B, S, D)
        k = k.view(B, S, D)
        v = v.view(B, S, D)
        if softmax_scale is None:
            softmax_scale = D ** (-0.5)
        rank = get_distributed_rank(cp_group)
        cp_size = get_distributed_world_size(cp_group)
        causal = "causal" in attn_mask_type
        assert attn_bias_type == "no_bias", f"{attn_bias_type} bias type not supported!"
        assert D % 8 == 0, "Merged hidden dimension must be a multiple of 8!"
        assert not use_fused_attention, "Flash attention not supported in per-document CP sharding."

        # compute document boundaries
        if isinstance(doc_lens, torch.Tensor):
            doc_lens = doc_lens.tolist()
        boundaries = [0]
        for l in doc_lens:
            boundaries.append(boundaries[-1] + int(l))
        num_docs = len(doc_lens)
        ctx.boundaries = boundaries

        # helper: compute balanced (padded) boundaries for a document of length L split into N chunks
        def balanced_boundaries(L, N):
            seg_len = L // N
            rem = L % N
            bnds = [0]
            for j in range(N):
                bnds.append(bnds[-1] + seg_len + (1 if j < rem else 0))
            return bnds

        # lists to store padded tensors and boundaries for each document
        ctx.doc_padded = []   # list of dict per document
        ctx.shard_info = []   # list of dict per document
        ctx.attn_probs = []   # list of tuple (attn_probs0, attn_probs1) per document

        output = q.new_zeros(B, S, D)
        N = 2 * cp_size

        for i in range(num_docs):
            doc_start = boundaries[i]
            doc_end = boundaries[i+1]
            L_doc = doc_end - doc_start

            bnds = balanced_boundaries(L_doc, N)
            debug_log(f"[Doc {i}] L_doc: {L_doc}, balanced boundaries: {bnds}")

            # compute effective indices for this rank
            seg0 = (bnds[rank], bnds[rank+1])
            seg1 = (bnds[N - rank - 1], bnds[N - rank])
            debug_log(f"[Doc {i}] Rank {rank} effective seg0: {seg0}, seg1: {seg1}")

            # determine if padding is needed
            pad = 0
            if L_doc % N != 0:
                pad = N - (L_doc % N)
                debug_log(f"[Doc {i}] Padding required: {pad} tokens")

            # create padded tensors
            doc_q_pad = torch.cat([q[:, doc_start:doc_end, :],
                                   q.new_zeros(B, pad, D)], dim=1)
            doc_k_pad = torch.cat([k[:, doc_start:doc_end, :],
                                   k.new_zeros(B, pad, D)], dim=1)
            doc_v_pad = torch.cat([v[:, doc_start:doc_end, :],
                                   v.new_zeros(B, pad, D)], dim=1)
            L_pad = L_doc + pad
            L_chunk = L_pad // N
            debug_log(f"[Doc {i}] L_pad: {L_pad}, L_chunk: {L_chunk}")

            # store the padded tensors and boundaries
            ctx.doc_padded.append({
                'q': doc_q_pad,
                'k': doc_k_pad,
                'v': doc_v_pad,
                'bnds': bnds,
                'pad': pad,
                'L_chunk': L_chunk
            })
            ctx.shard_info.append({
                'doc_start': doc_start,
                'doc_end': doc_end,
                'seg0': seg0,
                'seg1': seg1,
            })

            # reshape padded tensors
            doc_q_reshaped = doc_q_pad.view(B, N, L_chunk, D)
            doc_k_reshaped = doc_k_pad.view(B, N, L_chunk, D)
            doc_v_reshaped = doc_v_pad.view(B, N, L_chunk, D)
            debug_log(f"[Doc {i}] doc_q_reshaped shape: {doc_q_reshaped.shape}")

            # for this rank, select the two shards
            idx0 = rank
            idx1 = N - rank - 1
            q_shard0 = doc_q_reshaped[:, idx0, :, :]
            q_shard1 = doc_q_reshaped[:, idx1, :, :]
            local_k0 = doc_k_reshaped[:, idx0, :, :]
            local_v0 = doc_v_reshaped[:, idx0, :, :]
            local_k1 = doc_k_reshaped[:, idx1, :, :]
            local_v1 = doc_v_reshaped[:, idx1, :, :]

            # compute effective lengths
            eff_len0 = seg0[1] - seg0[0]
            eff_len1 = seg1[1] - seg1[0]

            # compute attention on each shard
            def compute_attn(q_shard, k_full, v_full):
                attn_scores = torch.matmul(q_shard, k_full.transpose(-2, -1)) * softmax_scale
                if causal:
                    B_, Lq = q_shard.shape[:2]
                    L_full = k_full.size(1)
                    key_idx = torch.arange(0, L_full, device=q.device).unsqueeze(0)
                    query_idx = torch.arange(0, Lq, device=q.device).unsqueeze(1)
                    mask = key_idx <= query_idx
                    attn_scores.masked_fill_(~mask, float('-inf'))
                attn_probs = torch.softmax(attn_scores, dim=-1)
                if is_training and dropout_p > 0:
                    attn_probs = torch.nn.functional.dropout(attn_probs, p=dropout_p, training=True)
                out = torch.matmul(attn_probs, v_full)
                return out, attn_probs

            out_seg0, attn_probs0 = compute_attn(q_shard0, local_k0, local_v0)
            out_seg1, attn_probs1 = compute_attn(q_shard1, local_k1, local_v1)
            ctx.attn_probs.append((attn_probs0, attn_probs1))
            debug_log(f"[Doc {i}] Shard0 out shape: {out_seg0.shape}, Shard1 out shape: {out_seg1.shape}")

            # crop the outputs to the effective lengths
            out0 = out_seg0[:, :eff_len0, :]
            out1 = out_seg1[:, :eff_len1, :]
            doc_out = q.new_zeros(B, L_doc, D)
            doc_out[:, seg0[0]:seg0[1], :].copy_(out0)
            doc_out[:, seg1[0]:seg1[1], :].copy_(out1)
            output[:, doc_start:doc_end, :] = doc_out

        dist.all_reduce(output, group=cp_group)
        ctx.save_for_backward(q, k, v, torch.tensor(doc_lens, device=q.device, dtype=torch.int32))
        ctx.cp_group = cp_group
        ctx.cp_stream = cp_stream
        ctx.softmax_scale = softmax_scale
        ctx.dropout_p = dropout_p
        ctx.input_shape = q.shape
        nvtx_range_pop("AttnFuncWithCPAndPerDocKVAllGather.forward")
        return output

    @staticmethod
    def backward(ctx, dout):
        nvtx_range_push("AttnFuncWithCPAndPerDocKVAllGather.backward")
        q, k, v, doc_lens_tensor = ctx.saved_tensors[:4]
        boundaries = ctx.boundaries
        shard_info = ctx.shard_info
        softmax_scale = ctx.softmax_scale
        dropout_p = ctx.dropout_p
        B, S, D = ctx.input_shape
        num_docs = len(shard_info)

        grad_q = torch.zeros_like(q)
        grad_k = torch.zeros_like(k)
        grad_v = torch.zeros_like(v)

        def attention_backward(dout_shard, attn_probs, q_shard, local_k, local_v):
            dV = torch.matmul(attn_probs.transpose(-2, -1), dout_shard)
            dA = torch.matmul(dout_shard, local_v.transpose(-2, -1))
            dScores = attn_probs * (dA - (dA * attn_probs).sum(dim=-1, keepdim=True)) * softmax_scale
            dQ = torch.matmul(dScores, local_k)
            dK = torch.matmul(dScores.transpose(-2, -1), q_shard)
            return dQ, dK, dV

        for i in range(num_docs):
            doc_start = boundaries[i]
            doc_end = boundaries[i+1]
            L_doc = doc_end - doc_start

            info = shard_info[i]
            seg0 = info['seg0']
            seg1 = info['seg1']

            # retrieve stored padded tensors and info
            padded = ctx.doc_padded[i]
            doc_q_pad = padded['q']
            doc_k_pad = padded['k']
            doc_v_pad = padded['v']
            L_pad = doc_q_pad.shape[1]
            N = 2 * get_distributed_world_size(ctx.cp_group)
            L_chunk = padded['L_chunk']
            debug_log(f"[Backward Doc {i}] L_doc: {L_doc}, pad: {padded['pad']}, L_pad: {L_pad}, L_chunk: {L_chunk}")

            # reshape stored padded tensors
            doc_q_reshaped = doc_q_pad.view(B, N, L_chunk, D)
            doc_k_reshaped = doc_k_pad.view(B, N, L_chunk, D)
            doc_v_reshaped = doc_v_pad.view(B, N, L_chunk, D)
            debug_log(f"[Backward Doc {i}] doc_q_reshaped shape: {doc_q_reshaped.shape}")

            rank_local = get_distributed_rank(ctx.cp_group)
            idx0 = rank_local
            idx1 = N - rank_local - 1

            eff_len0 = seg0[1] - seg0[0]
            eff_len1 = seg1[1] - seg1[0]

            q_shard0 = doc_q_reshaped[:, idx0, :eff_len0, :].view(B, -1, D)
            q_shard1 = doc_q_reshaped[:, idx1, :eff_len1, :].view(B, -1, D)
            local_k0 = doc_k_reshaped[:, idx0, :eff_len0, :]
            local_v0 = doc_v_reshaped[:, idx0, :eff_len0, :]
            local_k1 = doc_k_reshaped[:, idx1, :eff_len1, :]
            local_v1 = doc_v_reshaped[:, idx1, :eff_len1, :]

            dout_doc = dout[:, doc_start:doc_end, :]
            if L_doc < L_pad:
                dout_doc_pad = torch.cat([dout_doc, dout_doc.new_zeros(B, L_pad - L_doc, D)], dim=1)
            else:
                dout_doc_pad = dout_doc
            dout_reshaped = dout_doc_pad.view(B, N, L_chunk, D)
            dout_seg0 = dout_reshaped[:, idx0, :eff_len0, :]
            dout_seg1 = dout_reshaped[:, idx1, :eff_len1, :]

            # retrieve saved attention probabilities and crop them
            saved_attn_probs0, saved_attn_probs1 = ctx.attn_probs[i]
            saved_attn_probs0 = saved_attn_probs0[:, :eff_len0, :eff_len0]
            saved_attn_probs1 = saved_attn_probs1[:, :eff_len1, :eff_len1]

            dQ0, dK0, dV0 = attention_backward(dout_seg0, saved_attn_probs0, q_shard0, local_k0, local_v0)
            dQ1, dK1, dV1 = attention_backward(dout_seg1, saved_attn_probs1, q_shard1, local_k1, local_v1)

            grad_q[:, doc_start+seg0[0]:doc_start+seg0[1], :].copy_(dQ0)
            grad_q[:, doc_start+seg1[0]:doc_start+seg1[1], :].copy_(dQ1)
            grad_k[:, doc_start+seg0[0]:doc_start+seg0[1], :].add_(dK0)
            grad_k[:, doc_start+seg1[0]:doc_start+seg1[1], :].add_(dK1)
            grad_v[:, doc_start+seg0[0]:doc_start+seg0[1], :].add_(dV0)
            grad_v[:, doc_start+seg1[0]:doc_start+seg1[1], :].add_(dV1)

            debug_log(f"[Backward Doc {i}] dQ0 shape: {dQ0.shape}, dQ1 shape: {dQ1.shape}")
            debug_log(f"[Backward Doc {i}] dK0 shape: {dK0.shape}, dK1 shape: {dK1.shape}")

        dist.all_reduce(grad_q, group=ctx.cp_group)
        dist.all_reduce(grad_k, group=ctx.cp_group)
        dist.all_reduce(grad_v, group=ctx.cp_group)
        grad_q = grad_q.view(ctx.orig_shape)
        grad_k = grad_k.view(ctx.orig_shape)
        grad_v = grad_v.view(ctx.orig_shape)
        nvtx_range_pop("AttnFuncWithCPAndPerDocKVAllGather.backward")
        return (None, grad_q, grad_k, grad_v, None, None, None, None, None, None, None, None, None, None, None, None, None, None)

