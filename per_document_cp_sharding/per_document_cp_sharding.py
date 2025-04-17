import torch
import torch.distributed as dist
from torch.autograd import Function

try:
    from flash_attn.flash_attn_interface import _flash_attn_varlen_forward as fa_varlen_fwd
    from flash_attn.flash_attn_interface import _flash_attn_varlen_backward as fa_varlen_bwd
except ImportError:
    fa_varlen_fwd = None
    fa_varlen_bwd = None

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

class AttnFuncWithAllGatherPerDocSharding(Function):
    """
    Per-document CP attention.
    """

    @staticmethod
    def forward(
        ctx,
        is_training,
        q,            # [B, T_local, nHeads, headDim] local Q
        k,
        v,
        doc_lens,
        dropout_p,
        softmax_scale,
        qkv_format,
        attn_mask_type,
        attn_bias_type,
        attn_bias,
        deterministic,
        use_fused_attention,
        window_size,   # (window_left, window_right)
        cp_group,
        cp_stream,
    ):
        nvtx_range_push("AttnFuncWithAllGatherPerDocSharding.forward")
        cp_size = get_distributed_world_size(cp_group)
        rank = get_distributed_rank(cp_group)

        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** -0.5
        causal = ("causal" in attn_mask_type)
        assert "padding" not in attn_mask_type, f"{attn_mask_type} not supported!"
        assert attn_bias_type == "no_bias", f"{attn_bias_type} not supported!"

        B, T_local, nHeads, headDim = q.shape
        # 1) All-Gather k and v.
        k_4d = k.movedim(1, 0).contiguous()
        v_4d = v.movedim(1, 0).contiguous()
        k_ag, _ = gather_along_first_dim(k_4d, cp_group)
        v_ag, _ = gather_along_first_dim(v_4d, cp_group)

        # 2) Work with the true GLOBAL document lengths
        raw = doc_lens[0]
        if sum(raw) == T_local:
            doc_lens_for_b = [L * cp_size for L in raw]   # recover globals
        else:
            doc_lens_for_b = raw

        local_front_sizes = []
        local_back_sizes  = []

        for L_global in doc_lens_for_b:
            num_chunks = 2 * cp_size
            base       = L_global // num_chunks
            rem        = L_global %  num_chunks

            # size of each of the 2*cp_size chunks
            chunk_sizes = [base + (1 if i < rem else 0) for i in range(num_chunks)]

            front_size = chunk_sizes[rank]
            back_size  = chunk_sizes[num_chunks - 1 - rank]

            local_front_sizes.append(front_size)
            local_back_sizes.append(back_size)

        s_front = sum(local_front_sizes)
        s_back  = sum(local_back_sizes)
        
        # 3) Split q into front/back
        q_front_list = []
        q_back_list = []
        offset_q = 0
        for fsz, bsz in zip(local_front_sizes, local_back_sizes):
            q_front_list.append(q[0, offset_q : offset_q+fsz])
            q_back_list.append(q[0, offset_q+fsz : offset_q+fsz+bsz])
            offset_q += (fsz + bsz)
        q_front = torch.cat(q_front_list, dim=0).unsqueeze(0)
        q_back  = torch.cat(q_back_list, dim=0).unsqueeze(0)

         # 4) Build the front/back shards for k_ag, v_ag using doc-lens offsets
        offset_doc = rank * T_local
        global_front_k = []
        global_back_k  = []
        global_front_v = []
        global_back_v  = []

        i_doc = 0
        for fsz, bsz in zip(local_front_sizes, local_back_sizes):
            # doc's total length
            doc_length = doc_lens_for_b[i_doc]
            L_local = fsz + bsz  # local chunk size

            # front chunk => [offset_doc : offset_doc+fsz]
            global_front_k.append(k_ag[offset_doc : offset_doc + fsz])
            global_front_v.append(v_ag[offset_doc : offset_doc + fsz])

            # back chunk => [offset_doc + (L_local-bsz) : offset_doc+L_local]
            global_back_k.append(k_ag[offset_doc + (L_local - bsz) : offset_doc + L_local])
            global_back_v.append(v_ag[offset_doc + (L_local - bsz) : offset_doc + L_local])

            offset_doc += L_local
            i_doc      += 1
        
        # Combine all front/back shards
        k_front = torch.cat(global_front_k, dim=0).unsqueeze(1)  # => [s_front, B, nHeads, headDim]
        k_back  = torch.cat(global_back_k, dim=0).unsqueeze(1)
        v_front = torch.cat(global_front_v, dim=0).unsqueeze(1)
        v_back  = torch.cat(global_back_v, dim=0).unsqueeze(1)

        # 5) Flatten and call fa_varlen_fwd
        cu_seqlens_front = torch.tensor([0, s_front], device=q.device, dtype=torch.int32)
        cu_seqlens_back  = torch.tensor([0, s_back],  device=q.device, dtype=torch.int32)

        q_front_2d = q_front.reshape(-1, nHeads, headDim)
        k_front_2d = k_front.reshape(-1, nHeads, headDim)
        v_front_2d = v_front.reshape(-1, nHeads, headDim)

        q_back_2d  = q_back.reshape(-1, nHeads, headDim)
        k_back_2d  = k_back.reshape(-1, nHeads, headDim)
        v_back_2d  = v_back.reshape(-1, nHeads, headDim)

        # We'll do front chunk on default stream, back chunk on cp_stream
        with torch.cuda.stream(torch.cuda.current_stream()):
            out_front_2d, softmax_lse_front, _, _, _, _, _, _ = fa_varlen_fwd(
                q_front_2d, k_front_2d, v_front_2d,
                cu_seqlens_front, cu_seqlens_front,
                s_front, s_front,
                dropout_p,
                softmax_scale,
                causal,
                window_size,
                None,
                False
            )

        with torch.cuda.stream(cp_stream):
            out_back_2d, softmax_lse_back, _, _, _, _, _, _ = fa_varlen_fwd(
                q_back_2d, k_back_2d, v_back_2d,
                cu_seqlens_back, cu_seqlens_back,
                s_back, s_back,
                dropout_p,
                softmax_scale,
                causal,
                window_size,
                None,
                False
            )

        torch.cuda.current_stream().wait_stream(cp_stream)


        out_front_4d = out_front_2d.view(s_front, B, nHeads, headDim).transpose(0,1).contiguous()
        out_back_4d  = out_back_2d.view(s_back,  B, nHeads, headDim).transpose(0,1).contiguous()

        # 6) Reassemble outputs
        outputs = []
        front_idx = 0
        back_idx  = 0
        for fsz, bsz in zip(local_front_sizes, local_back_sizes):
            doc_front = out_front_4d[:, front_idx : front_idx + fsz]
            doc_back  = out_back_4d[:, back_idx  : back_idx  + bsz]
            doc_out   = torch.cat([doc_front, doc_back], dim=1)
            outputs.append(doc_out)
            front_idx += fsz
            back_idx  += bsz

        out = torch.cat(outputs, dim=1)  # => [B, T_local, nHeads, headDim]

        # Convert doc_lens_for_b to a tensor so we can store in ctx
        doc_lens_for_b_t = torch.tensor(doc_lens_for_b, device=q.device, dtype=torch.int32)

        # Save for backward
        ctx.save_for_backward(
            q, k, v,
            q_front, q_back,
            q_front_2d, q_back_2d,
            k_front_2d, k_back_2d,
            v_front_2d, v_back_2d,
            torch.tensor(local_front_sizes, device=q.device, dtype=torch.int32),
            torch.tensor(local_back_sizes,  device=q.device, dtype=torch.int32),
            torch.tensor([T_local],         device=q.device, dtype=torch.int32),
            out_front_2d, out_back_2d,
            softmax_lse_front, softmax_lse_back,
            doc_lens_for_b_t
        )
        ctx.cp_group = cp_group
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.s_front = s_front
        ctx.s_back  = s_back

        nvtx_range_pop("AttnFuncWithAllGatherPerDocSharding.forward")
        return out.to(torch.bfloat16)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for per-document CP attention.

        Key Fix:
        - We remove the symmetrical reorder lines (chunk_ids_for_kv_ag, etc.)
            that apply only to the per-sequence approach.
        - We still do a reduce_scatter_along_first_dim on [cp_size*T_local, ...],
            but do NOT reorder or reshape to [2*cp_size, s//2, ...].
        - Each rank's doc tokens are disjoint, so the reduce-scatter is effectively
            distributing each rank's piece without overlap.
        """
        (
            q, k, v,
            q_front, q_back,
            q_front_2d, q_back_2d,
            k_front_2d, k_back_2d,
            v_front_2d, v_back_2d,
            local_front_sizes, local_back_sizes,
            t_local_tensor,
            out_front_2d, out_back_2d,
            softmax_lse_front, softmax_lse_back,
            doc_lens_for_b_t
        ) = ctx.saved_tensors

        dropout_p    = ctx.dropout_p
        softmax_scale= ctx.softmax_scale
        causal       = ctx.causal
        window_size  = ctx.window_size
        s_front      = ctx.s_front
        s_back       = ctx.s_back
        T_local      = t_local_tensor.item()
        B            = q.shape[0]
        nHeads       = q.shape[2]
        headDim      = q.shape[3]
        doc_lens_for_b = doc_lens_for_b_t.tolist()
        cp_group     = ctx.cp_group
        rank         = get_distributed_rank(cp_group)
        cp_size      = get_distributed_world_size(cp_group)

        # 1) Re-split grad_output into front/back doc chunks
        local_front_sizes = local_front_sizes.tolist()
        local_back_sizes  = local_back_sizes.tolist()

        grad_front_list = []
        grad_back_list  = []
        offset_q = 0
        for fsz, bsz in zip(local_front_sizes, local_back_sizes):
            grad_front_list.append(grad_output[:, offset_q : offset_q + fsz])
            grad_back_list.append(grad_output[:, offset_q + fsz : offset_q + fsz + bsz])
            offset_q += (fsz + bsz)

        grad_front_4d = torch.cat(grad_front_list, dim=1)  # [B, sum(front), nHeads, headDim]
        grad_back_4d  = torch.cat(grad_back_list, dim=1)   # [B, sum(back),  nHeads, headDim]

        # Flatten for varlen_bwd
        grad_front_2d = grad_front_4d.transpose(0,1).reshape(-1, nHeads, headDim)
        grad_back_2d  = grad_back_4d.transpose(0,1).reshape(-1, nHeads, headDim)

        # 2) varlen_bwd calls for front/back
        cu_seqlens_front = torch.tensor([0, s_front], dtype=torch.int32, device=q.device)
        cu_seqlens_back  = torch.tensor([0, s_back],  dtype=torch.int32, device=q.device)

        dq_front_2d = torch.empty_like(q_front_2d)
        dk_front_2d = torch.empty_like(k_front_2d)
        dv_front_2d = torch.empty_like(v_front_2d)

        dq_back_2d  = torch.empty_like(q_back_2d)
        dk_back_2d  = torch.empty_like(k_back_2d)
        dv_back_2d  = torch.empty_like(v_back_2d)

        # front chunk backward
        fa_varlen_bwd(
            grad_front_2d,
            q_front_2d,
            k_front_2d,
            v_front_2d,
            out_front_2d,
            softmax_lse_front,
            dq_front_2d,
            dk_front_2d,
            dv_front_2d,
            cu_seqlens_front,
            cu_seqlens_front,
            s_front,
            s_front,
            dropout_p,
            softmax_scale,
            causal,
            window_size,
            None,  # alibi_slopes
            False, # deterministic
            None   # rng_state
        )
        # back chunk backward
        fa_varlen_bwd(
            grad_back_2d,
            q_back_2d,
            k_back_2d,
            v_back_2d,
            out_back_2d,
            softmax_lse_back,
            dq_back_2d,
            dk_back_2d,
            dv_back_2d,
            cu_seqlens_back,
            cu_seqlens_back,
            s_back,
            s_back,
            dropout_p,
            softmax_scale,
            causal,
            window_size,
            None,
            False,
            None
        )

        # 3) Reassemble local dQ
        dq_front_4d = dq_front_2d.view(s_front, B, nHeads, headDim).transpose(0,1)
        dq_back_4d  = dq_back_2d.view(s_back,  B, nHeads, headDim).transpose(0,1)

        dq_docs = []
        front_idx = 0
        back_idx  = 0
        for fsz, bsz in zip(local_front_sizes, local_back_sizes):
            df = dq_front_4d[:, front_idx : front_idx + fsz]
            db = dq_back_4d[:, back_idx  : back_idx  + bsz]
            dq_doc = torch.cat([df, db], dim=1)  # [B, fsz+bsz, nHeads, headDim]
            dq_docs.append(dq_doc)
            front_idx += fsz
            back_idx  += bsz

        dq_local = torch.cat(dq_docs, dim=1)  # => [B, T_local, nHeads, headDim]

        # 4) Reassemble global dK, dV => shape [cp_size*T_local, B, nHeads, headDim]
        dK_global = torch.zeros(cp_size * T_local, B, nHeads, headDim,
                                device=q.device, dtype=dq_local.dtype)
        dV_global = torch.zeros_like(dK_global)

        # place front/back chunk grads at correct offsets
        dk_front_4d = dk_front_2d.view(s_front, B, nHeads, headDim)
        dk_back_4d  = dk_back_2d.view(s_back,  B, nHeads, headDim)
        dv_front_4d = dv_front_2d.view(s_front, B, nHeads, headDim)
        dv_back_4d  = dv_back_2d.view(s_back,  B, nHeads, headDim)

        offset_doc = rank * T_local
        front_idx = 0
        back_idx  = 0
        i_doc = 0 # doc index
        for fsz, bsz in zip(local_front_sizes, local_back_sizes):
            doc_length = doc_lens_for_b[i_doc]  
            L_local = fsz + bsz

            # front chunk => [offset_doc : offset_doc+fsz]
            dK_global[offset_doc : offset_doc + fsz] += dk_front_4d[front_idx : front_idx + fsz]
            dV_global[offset_doc + fsz : offset_doc + fsz + bsz] += dv_back_4d[back_idx : back_idx + bsz]
            # back chunk => [offset_doc + (L_local-bsz) : offset_doc+L_local]
            dK_global[offset_doc + (L_local - bsz) : offset_doc + L_local] += dk_back_4d[back_idx : back_idx + bsz]
            dV_global[offset_doc + (L_local - bsz) : offset_doc + L_local] += dv_back_4d[back_idx : back_idx + bsz]

            offset_doc += L_local

            front_idx += fsz
            back_idx  += bsz
            i_doc     += 1

        dK_global, _ = reduce_scatter_along_first_dim(dK_global, cp_group)
        dV_global, _ = reduce_scatter_along_first_dim(dV_global, cp_group)

        # reorder => local shape [B, T_local, nHeads, headDim]
        dK_local = dK_global.movedim(0,1).contiguous()
        dV_local = dV_global.movedim(0,1).contiguous()

        return (
            None,        # is_training
            dq_local,    # dQ
            dK_local,    # dK
            dV_local,    # dV
            None,        # doc_lens
            None,        # dropout_p
            None,        # softmax_scale
            None,        # qkv_format
            None,        # attn_mask_type
            None,        # attn_bias_type
            None,        # attn_bias
            None,        # deterministic
            None,        # use_fused_attention
            None,        # window_size
            None,        # cp_group
            None         # cp_stream
        )