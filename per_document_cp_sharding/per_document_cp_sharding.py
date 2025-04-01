import torch
import torch.distributed as dist
from torch.autograd import Function

try:
    from flash_attn.flash_attn_interface import _flash_attn_varlen_forward as fa_varlen_fwd
except ImportError:
    fa_varlen_fwd = None

from transformer_engine.pytorch.distributed import (
    gather_along_first_dim,
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
        doc_lens,     # e.g. [[4,8]] if B=1
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

        qkv_dtype = q.dtype
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** -0.5
        causal = ("causal" in attn_mask_type)
        assert "padding" not in attn_mask_type, f"{attn_mask_type} not supported!"
        assert attn_bias_type == "no_bias", f"{attn_bias_type} not supported!"

        B, T_local, nHeads, headDim = q.shape
        # e.g. if global doc lengths were [8,16] and cp_size=2 => local doc_lens = [4,8].
        # We'll compute front/back shards for each doc in doc_lens[0].

        # 1) All-Gather local K, V => shape [cp_size*T_local, B, nHeads, headDim]
        k_4d = k.movedim(1, 0).contiguous()  # => [T_local, B, nHeads, headDim]
        v_4d = v.movedim(1, 0).contiguous()
        k_ag, _ = gather_along_first_dim(k_4d, cp_group)  # => [cp_size*T_local, B, nHeads, headDim]
        v_ag, _ = gather_along_first_dim(v_4d, cp_group)

        # 2) Flatten local Q => we'll eventually do 2 calls (front/back).
        # But first we need to define front_size/back_size for each doc.
        local_front_sizes = []
        local_back_sizes  = []
        for L_local in doc_lens[0]:
            base = L_local // 2
            rem  = L_local % 2
            front_size = base + (1 if rank < rem else 0)
            back_size  = base + (1 if (1 - rank) < rem else 0)
            local_front_sizes.append(front_size)
            local_back_sizes.append(back_size)

        s_front = sum(local_front_sizes)
        s_back  = sum(local_back_sizes)

        # 3) Split local Q => front half + back half (per doc).
        q_front_list = []
        q_back_list  = []
        offset_q = 0
        for fsz, bsz in zip(local_front_sizes, local_back_sizes):
            q_front_list.append(q[0, offset_q : offset_q+fsz])
            q_back_list.append(q[0, offset_q+fsz : offset_q+fsz+bsz])
            offset_q += (fsz + bsz)
        q_front = torch.cat(q_front_list, dim=0).unsqueeze(0)  # => [B, s_front, nHeads, headDim]
        q_back  = torch.cat(q_back_list, dim=0).unsqueeze(0)

        # 4) Each rank's slice in k_ag, v_ag is from rank*T_local .. (rank+1)*T_local -1
        offset_for_rank = rank * T_local

        # build the same doc-based front/back shards from k_global, v_global,
        # but each doc is within [offset_for_rank .. offset_for_rank+L_local].
        global_front_k = []
        global_back_k  = []
        global_front_v = []
        global_back_v  = []

        offset_doc = offset_for_rank
        for i, (fsz, bsz) in enumerate(zip(local_front_sizes, local_back_sizes)):
            L_local = fsz + bsz  # local doc length
            # front = [offset_doc : offset_doc+fsz]
            # back  = [offset_doc + (L_local - bsz) : offset_doc+L_local]
            global_front_k.append(k_ag[offset_doc : offset_doc + fsz])
            global_back_k.append(k_ag[offset_doc + (L_local - bsz) : offset_doc + L_local])
            global_front_v.append(v_ag[offset_doc : offset_doc + fsz])
            global_back_v.append(v_ag[offset_doc + (L_local - bsz) : offset_doc + L_local])
            offset_doc += L_local

        k_front = torch.cat(global_front_k, dim=0).unsqueeze(1)  # => [s_front, B, nHeads, headDim]
        k_back  = torch.cat(global_back_k, dim=0).unsqueeze(1)
        v_front = torch.cat(global_front_v, dim=0).unsqueeze(1)
        v_back  = torch.cat(global_back_v, dim=0).unsqueeze(1)

        # 5) Flatten + varlen calls
        cu_seqlens_front = torch.tensor([0, s_front], dtype=torch.int32, device=q.device)
        cu_seqlens_back  = torch.tensor([0, s_back],  dtype=torch.int32, device=q.device)

        # Flatten Q
        q_front_2d = q_front.reshape(-1, nHeads, headDim)
        k_front_2d = k_front.reshape(-1, nHeads, headDim)
        v_front_2d = v_front.reshape(-1, nHeads, headDim)
        q_back_2d  = q_back.reshape(-1, nHeads, headDim)
        k_back_2d  = k_back.reshape(-1, nHeads, headDim)
        v_back_2d  = v_back.reshape(-1, nHeads, headDim)

        # Two calls
        with torch.cuda.stream(torch.cuda.current_stream()):
            out_front_2d, _, _, _, _, _, _, _ = fa_varlen_fwd(
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
            out_back_2d, _, _, _, _, _, _, _  = fa_varlen_fwd(
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

        out_front_4d = out_front_2d.view(s_front, B, nHeads, headDim).transpose(0,1).contiguous()  # => [B, s_front, nHeads, headDim]
        out_back_4d  = out_back_2d.view(s_back,  B, nHeads, headDim).transpose(0,1).contiguous()

        # 6) Reassemble doc by doc in order: front followed by back
        #   so total => [B, T_local, nHeads, headDim]
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

        nvtx_range_pop("AttnFuncWithAllGatherPerDocSharding.forward")
        return out.to(torch.bfloat16)
    
    @staticmethod
    def backward():
        # TODO: implement the backwards pass
        return None
