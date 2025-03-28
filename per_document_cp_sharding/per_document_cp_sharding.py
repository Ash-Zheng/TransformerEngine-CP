import torch
from torch.autograd import Function
import torch.distributed as dist

from transformer_engine.pytorch.distributed import (
    get_distributed_world_size,
    get_distributed_rank,
)
from transformer_engine.pytorch.utils import (
    nvtx_range_push,
    nvtx_range_pop,
)

###############################################################################
# Utility functions
###############################################################################

def gather_along_first_dim(tensor: torch.Tensor, group):
    world_size = dist.get_world_size(group)
    chunks = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(chunks, tensor, group=group)
    return torch.cat(chunks, dim=0)

def reduce_scatter_along_first_dim(tensor: torch.Tensor, group):
    world_size = dist.get_world_size(group)
    chunk_size = tensor.shape[0] // world_size
    splits = list(torch.split(tensor, chunk_size, dim=0))
    out = torch.zeros_like(splits[0])
    dist.reduce_scatter(out, splits, group=group)
    return out

def compute_divisible_and_leftover(L, cp_size):
    D = (L // (2*cp_size)) * (2*cp_size)
    R = L - D
    return D, R

def forward_attn_chunk(q_chunk, k_chunk, v_chunk, softmax_scale, causal, is_training, dropout_p):
    attn_scores = torch.matmul(q_chunk, k_chunk.transpose(-2, -1)) * softmax_scale
    if causal:
        Lq = q_chunk.size(1)
        Lk = k_chunk.size(1)
        idx_q = torch.arange(Lq, device=q_chunk.device).view(Lq, 1)
        idx_k = torch.arange(Lk, device=q_chunk.device).view(1, Lk)
        mask = (idx_k <= idx_q)
        attn_scores.masked_fill_(~mask, float('-inf'))
    attn_probs = torch.softmax(attn_scores, dim=-1)
    if is_training and dropout_p > 0:
        attn_probs = torch.nn.functional.dropout(attn_probs, p=dropout_p, training=True)
    out_chunk = torch.matmul(attn_probs, v_chunk)
    return out_chunk, attn_probs

def backward_attn_chunk(dout, attn_probs, q_chunk, k_chunk, v_chunk, softmax_scale):
    dV = torch.matmul(attn_probs.transpose(-2, -1), dout)
    dA = torch.matmul(dout, v_chunk.transpose(-2, -1))
    dScores = attn_probs * (dA - (dA * attn_probs).sum(dim=-1, keepdim=True))
    dScores *= softmax_scale
    dQ = torch.matmul(dScores, k_chunk)
    dK = torch.matmul(dScores.transpose(-2, -1), q_chunk)
    return dQ, dK, dV

###############################################################################
# Main Class: Per-document CP sharding with round-robin distribution.
###############################################################################

class AttnFuncWithPerDocRoundRobinSharding(Function):
    r"""
    Per-document CP sharding with round-robin distribution.
    
    This implementation splits each document into a “divisible” portion (of length D_i,
    a multiple of 2*cp_size) and a leftover portion. For each portion, it gathers K,V 
    from all ranks, reshapes them into a 4D tensor of shape [2*cp_size, chunk_len, B, D],
    and then processes the local Q in chunks. In the case where the local portion length
    is less than 2*cp_size (i.e. cannot be evenly split), we simply process the entire tensor
    with vanilla attention. The final output is a flat tensor of shape [B, S, D] (with D = nHeads*headDim).
    
    Remember to call dist.destroy_process_group() when done to avoid NCCL warnings.
    """

    @staticmethod
    def forward(ctx,
                is_training,
                q, k, v,      # [B, S, nHeads, headDim]
                doc_lens,     # list (or tensor) of document lengths (sum = S)
                dropout_p,
                softmax_scale,
                qkv_format,
                attn_mask_type,
                attn_bias_type,
                attn_bias,
                deterministic,
                use_fused_attention,
                window_size,
                cp_group,
                cp_stream):

        nvtx_range_push("AttnFuncWithPerDocRoundRobinSharding.forward")

        B, S, nHeads, headDim = q.shape
        D = nHeads * headDim
        # Flatten Q, K, V to [B, S, D]
        q_2d = q.view(B, S, D)
        k_2d = k.view(B, S, D)
        v_2d = v.view(B, S, D)

        if softmax_scale is None:
            softmax_scale = 1.0 / (headDim ** 0.5)

        cp_size = dist.get_world_size(cp_group)
        rank = dist.get_rank(cp_group)
        causal = ("causal" in attn_mask_type)

        out_2d = torch.zeros_like(q_2d)

        if isinstance(doc_lens, torch.Tensor):
            doc_lens = doc_lens.tolist()

        # Compute document boundaries.
        boundaries = [0]
        for l in doc_lens:
            boundaries.append(boundaries[-1] + int(l))

        saved_divisible = []
        saved_leftover = []
        local_seq_chunk_ids = [rank, (2 * cp_size - rank - 1)]

        # Helper for forward on one chunk.
        def sym_chunk_fwd(q_loc, k_loc, v_loc):
            print(f"[sym_chunk_fwd] q_loc shape: {q_loc.shape}, k_loc shape: {k_loc.shape}")
            local_len = q_loc.shape[0]
            # If local portion is too small to split among 2*cp_size, process whole.
            if local_len < 2 * cp_size:
                print(f"[sym_chunk_fwd] local_len {local_len} < 2*cp_size {2*cp_size}, processing as single chunk")
                q_loc_2d = q_loc.movedim(0, 1)
                k_loc_2d = k_loc.movedim(0, 1)
                v_loc_2d = v_loc.movedim(0, 1)
                out_chunk, attn_probs_ = forward_attn_chunk(q_loc_2d, k_loc_2d, v_loc_2d, softmax_scale, causal, is_training, dropout_p)
                step_saves = {"q_sub_0": q_loc, "k_sub_0": k_loc, "v_sub_0": v_loc}
                attn_probs_steps = [attn_probs_, None]
                return out_chunk.movedim(1, 0), gather_along_first_dim(k_loc, cp_group), gather_along_first_dim(v_loc, cp_group), step_saves, attn_probs_steps

            chunk_len = local_len // 2
            print(f"[sym_chunk_fwd] local_len: {local_len}, chunk_len: {chunk_len}")
            gather_k = gather_along_first_dim(k_loc, cp_group)
            gather_v = gather_along_first_dim(v_loc, cp_group)
            print(f"[sym_chunk_fwd] gather_k shape: {gather_k.shape}")
            # Reshape gathered k,v to [2*cp_size, chunk_len, B, D]
            k_4d = gather_k.view(2 * cp_size, chunk_len, B, D)
            v_4d = gather_v.view(2 * cp_size, chunk_len, B, D)
            out_loc = torch.zeros_like(q_loc)
            step_saves = {}
            attn_probs_steps = [None, None]
            for step_i in range(len(local_seq_chunk_ids) + 1):
                if step_i < len(local_seq_chunk_ids):
                    c_id = local_seq_chunk_ids[step_i]
                    row_start = c_id * chunk_len
                    row_end = row_start + chunk_len
                    rs = max(0, min(row_start, local_len))
                    re = max(0, min(row_end, local_len))
                    L_sub = re - rs
                    if L_sub <= 0:
                        continue
                    q_sub = q_loc[rs:re]
                    # Safety check on c_id
                    if c_id < 0 or c_id >= 2 * cp_size:
                        continue
                    k_sub = k_4d[c_id]
                    v_sub = v_4d[c_id]
                    if L_sub < k_sub.shape[0]:
                        k_sub = k_sub[:L_sub]
                        v_sub = v_sub[:L_sub]
                    q_sub_2d = q_sub.movedim(0, 1)
                    k_sub_2d = k_sub.movedim(0, 1)
                    v_sub_2d = v_sub.movedim(0, 1)
                    out_chunk_2d, attn_probs_ = forward_attn_chunk(q_sub_2d, k_sub_2d, v_sub_2d, softmax_scale, causal, is_training, dropout_p)
                    attn_probs_steps[step_i] = attn_probs_
                    step_saves[f"q_sub_{step_i}"] = q_sub
                    step_saves[f"k_sub_{step_i}"] = k_sub
                    step_saves[f"v_sub_{step_i}"] = v_sub
                if step_i > 0:
                    prev = step_i - 1
                    pc_id = local_seq_chunk_ids[prev]
                    ps = pc_id * chunk_len
                    pe = ps + chunk_len
                    ps_c = max(0, min(ps, local_len))
                    pe_c = max(0, min(pe, local_len))
                    l_out = pe_c - ps_c
                    if l_out > 0:
                        out_loc[ps_c:pe_c] = out_chunk_2d.movedim(1, 0)[:l_out]
            return out_loc, gather_k, gather_v, step_saves, attn_probs_steps

        # Process divisible portions.
        for i in range(len(doc_lens)):
            ds = boundaries[i]
            de = boundaries[i+1]
            L_i = de - ds
            D_i, R_i = compute_divisible_and_leftover(L_i, cp_size)
            if D_i > 0:
                q_loc = q_2d[:, ds:ds+D_i, :].movedim(1, 0).contiguous()
                k_loc = k_2d[:, ds:ds+D_i, :].movedim(1, 0).contiguous()
                v_loc = v_2d[:, ds:ds+D_i, :].movedim(1, 0).contiguous()
                out_loc, gk, gv, step_saves, attn_probs_steps = sym_chunk_fwd(q_loc, k_loc, v_loc)
                out_loc_2d = out_loc.movedim(0, 1)
                out_2d[:, ds:ds+D_i, :] += out_loc_2d
                saved_divisible.append({
                    'ds': ds,
                    'D_i': D_i,
                    'q_loc': q_loc,
                    'gather_k': gk,
                    'gather_v': gv,
                    'step_saves': step_saves,
                    'attn_probs': attn_probs_steps
                })

        # Process leftover portions.
        leftover_saves = []
        for i in range(len(doc_lens)):
            ds = boundaries[i]
            de = boundaries[i+1]
            L_i = de - ds
            D_i, R_i = compute_divisible_and_leftover(L_i, cp_size)
            if R_i == 0:
                leftover_saves.append(None)
                continue
            st = ds + D_i
            ed = de
            R_len = ed - st
            q_left = q_2d[:, st:ed, :].movedim(1, 0).contiguous()
            k_left = k_2d[:, st:ed, :].movedim(1, 0).contiguous()
            v_left = v_2d[:, st:ed, :].movedim(1, 0).contiguous()
            N = 2 * cp_size
            leftover_idx = []
            for c in range(N):
                leftover_idx += [j for j in range(R_len) if (j % N) == c]
            leftover_idx_t = torch.tensor(leftover_idx, device=q.device, dtype=torch.long)
            inv_idx_t = torch.empty_like(leftover_idx_t)
            for pos, val in enumerate(leftover_idx):
                inv_idx_t[val] = pos
            q_left_rr = q_left.index_select(0, leftover_idx_t)
            k_left_rr = k_left.index_select(0, leftover_idx_t)
            v_left_rr = v_left.index_select(0, leftover_idx_t)
            out_loc, gk, gv, step_saves, attn_probs_steps = sym_chunk_fwd(q_left_rr, k_left_rr, v_left_rr)
            out_loc_inv = out_loc.index_select(0, inv_idx_t)
            out_loc_2d = out_loc_inv.movedim(0, 1)
            out_2d[:, st:ed, :] += out_loc_2d
            leftover_saves.append({
                'ds_left': st,
                'R_len': R_len,
                'q_left_rr': q_left_rr,
                'gather_k': gk,
                'gather_v': gv,
                'left_idx': leftover_idx_t,
                'inv_idx': inv_idx_t,
                'step_saves': step_saves,
                'attn_probs': attn_probs_steps
            })

        ctx.saved_divisible = saved_divisible
        ctx.saved_leftover = leftover_saves
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.attn_mask_type = attn_mask_type
        ctx.attn_bias_type = attn_bias_type
        ctx.cp_group = cp_group
        ctx.cp_stream = cp_stream
        ctx.orig_shape = q.shape  # [B, S, nHeads, headDim]
        ctx.is_training = is_training

        nvtx_range_pop("AttnFuncWithPerDocRoundRobinSharding.forward")
        return out_2d

    @staticmethod
    def backward(ctx, dout):
        nvtx_range_push("AttnFuncWithPerDocRoundRobinSharding.backward")
        B, S, nHeads, headDim = ctx.orig_shape
        D = nHeads * headDim
        dout_2d = dout.view(B, S, D)
        dq_2d = torch.zeros_like(dout_2d)
        dk_2d = torch.zeros_like(dout_2d)
        dv_2d = torch.zeros_like(dout_2d)

        cp_group = ctx.cp_group
        rank = dist.get_rank(cp_group)
        cp_size = dist.get_world_size(cp_group)
        local_seq_chunk_ids = [rank, (2 * cp_size - rank - 1)]
        dropout_p = ctx.dropout_p
        softmax_scale = ctx.softmax_scale
        causal = ("causal" in ctx.attn_mask_type)
        is_training = ctx.is_training

        def sym_chunk_bwd(dout_loc, q_loc, gather_k, gather_v, step_saves, attn_probs):
            dq_loc = torch.zeros_like(q_loc)
            dk_loc = torch.zeros_like(q_loc)
            dv_loc = torch.zeros_like(q_loc)
            local_len = q_loc.shape[0]
            # If local_len is too small to split evenly, process whole.
            if local_len < 2 * cp_size:
                dQ_, dK_, dV_ = backward_attn_chunk(
                    dout_loc.movedim(0, 1),
                    attn_probs[0],
                    q_loc.movedim(0, 1),
                    gather_k.movedim(0, 1),
                    gather_v.movedim(0, 1),
                    softmax_scale,
                )
                dq_loc = dQ_.movedim(1, 0)
                dk_loc = dK_.movedim(1, 0)
                dv_loc = dV_.movedim(1, 0)
                dk_ag = gather_along_first_dim(dk_loc, cp_group)
                dv_ag = gather_along_first_dim(dv_loc, cp_group)
                dk_rs = reduce_scatter_along_first_dim(dk_ag, cp_group)
                dv_rs = reduce_scatter_along_first_dim(dv_ag, cp_group)
                return dq_loc, dk_rs, dv_rs

            chunk_len = local_len // 2
            for step_i in range(len(local_seq_chunk_ids) + 1):
                if step_i < len(local_seq_chunk_ids):
                    c_id = local_seq_chunk_ids[step_i]
                    row_start = c_id * chunk_len
                    row_end = row_start + chunk_len
                    rs = max(0, min(row_start, local_len))
                    re = max(0, min(row_end, local_len))
                    L_sub = re - rs
                    if L_sub <= 0:
                        continue
                    q_sub = step_saves[f"q_sub_{step_i}"]
                    k_sub = step_saves[f"k_sub_{step_i}"]
                    v_sub = step_saves[f"v_sub_{step_i}"]
                    attn_probs_ = attn_probs[step_i]
                    dout_sub = dout_loc[rs:re]
                    if L_sub < k_sub.shape[0]:
                        k_sub = k_sub[:L_sub]
                        v_sub = v_sub[:L_sub]
                    dout_sub_2d = dout_sub.movedim(0, 1)
                    q_sub_2d = q_sub.movedim(0, 1)
                    k_sub_2d = k_sub.movedim(0, 1)
                    v_sub_2d = v_sub.movedim(0, 1)
                    dQ_, dK_, dV_ = backward_attn_chunk(
                        dout_sub_2d, attn_probs_, q_sub_2d, k_sub_2d, v_sub_2d, softmax_scale
                    )
                    dq_loc[rs:re] += dQ_.movedim(1, 0)
                    dk_loc[rs:re] += dK_.movedim(1, 0)
                    dv_loc[rs:re] += dV_.movedim(1, 0)
            dk_ag = gather_along_first_dim(dk_loc, cp_group)
            dv_ag = gather_along_first_dim(dv_loc, cp_group)
            dk_rs = reduce_scatter_along_first_dim(dk_ag, cp_group)
            dv_rs = reduce_scatter_along_first_dim(dv_ag, cp_group)
            return dq_loc, dk_rs, dv_rs

        for rec in ctx.saved_divisible:
            ds = rec["ds"]
            D_i = rec["D_i"]
            q_loc = rec["q_loc"]
            gk = rec["gather_k"]
            gv = rec["gather_v"]
            step_saves = rec["step_saves"]
            attn_probs = rec["attn_probs"]

            dout_loc = dout_2d[:, ds:ds+D_i, :].movedim(1, 0).contiguous()
            dq_loc, dk_rs, dv_rs = sym_chunk_bwd(dout_loc, q_loc, gk, gv, step_saves, attn_probs)
            dq_2d[:, ds:ds+D_i, :] += dq_loc.movedim(0, 1)
            dk_2d[:, ds:ds+D_i, :] += dk_rs.movedim(0, 1)
            dv_2d[:, ds:ds+D_i, :] += dv_rs.movedim(0, 1)

        for rec in ctx.saved_leftover:
            if rec is None:
                continue
            ds_left = rec["ds_left"]
            R_len = rec["R_len"]
            q_left_rr = rec["q_left_rr"]
            gk = rec["gather_k"]
            gv = rec["gather_v"]
            leftover_idx = rec["left_idx"]
            inv_idx = rec["inv_idx"]
            step_saves = rec["step_saves"]
            attn_probs = rec["attn_probs"]

            dout_loc = dout_2d[:, ds_left:ds_left+R_len, :].movedim(1, 0).contiguous()
            dout_rr = dout_loc.index_select(0, leftover_idx)
            dq_loc, dk_rs, dv_rs = sym_chunk_bwd(dout_rr, q_left_rr, gk, gv, step_saves, attn_probs)
            dq_loc_inv = dq_loc.index_select(0, inv_idx)
            dq_2d[:, ds_left:ds_left+R_len, :] += dq_loc_inv.movedim(0, 1)
            dk_inv = dk_rs.index_select(0, inv_idx)
            dv_inv = dv_rs.index_select(0, inv_idx)
            dk_2d[:, ds_left:ds_left+R_len, :] += dk_inv.movedim(0, 1)
            dv_2d[:, ds_left:ds_left+R_len, :] += dv_inv.movedim(0, 1)

        dq = dq_2d.view(B, S, nHeads, headDim)
        dk = dk_2d.view(B, S, nHeads, headDim)
        dv = dv_2d.view(B, S, nHeads, headDim)

        nvtx_range_pop("AttnFuncWithPerDocRoundRobinSharding.backward")
        return (None, dq, dk, dv, None, None, None, None, None, None, None, None, None, None, None, None, None, None)

