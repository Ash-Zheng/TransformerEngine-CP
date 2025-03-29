import torch
from torch.autograd import Function
import torch.distributed as dist

from transformer_engine.pytorch.distributed import (
    gather_along_first_dim,
    reduce_scatter_along_first_dim,
)
from flash_attn.flash_attn_interface import (
    _flash_attn_varlen_forward as fa_varlen_fwd,
    _flash_attn_varlen_backward as fa_varlen_bwd,
)

def compute_equal_chunks(total_tokens, cp_size):
    """
    Divide total_tokens into 2*cp_size contiguous segments.
    Each segment gets a base length of total_tokens // (2*cp_size) tokens,
    with any remaining tokens distributed such that no segment differs by more than one token.
    Returns a list of tuples (start_offset, chunk_size) for each of the 2*cp_size segments.
    """
    N = 2 * cp_size
    base_len = total_tokens // N
    leftover = total_tokens % N
    chunk_sizes = [base_len + (1 if i < leftover else 0) for i in range(N)]
    chunk_ranges = []
    offset = 0
    for c_id in range(N):
        c_size = chunk_sizes[c_id]
        chunk_ranges.append((offset, c_size))
        offset += c_size
    return chunk_ranges

class AttnFuncWithPerDocRoundRobinSharding(Function):
    """
    Implements per-document CP sharding using exactly two variable-length attention calls per rank.
    The process is as follows:
      1) merge document lengths to get the total token count T.
      2) split the range [0, T) into 2*cp_size contiguous segments
      3) each rank processes two segments: one at index equal to its rank and one at index (N - rank - 1)
      4) extract the corresponding sub-range from the query tensor and perform a variable-length forward call,
         yielding an output of shape [chunk_size, nHeads, headDim]
      5) use partial indexing on the gathered key tensor (of shape [cp_size*T, B, nHeads, headDim])
      6) in total, exactly two forward calls are executed per rank
    """

    @staticmethod
    def forward(
        ctx,
        is_training,
        q,        # shape: [B, T, nHeads, headDim], where T = sum(doc_lens)
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
        window_size,
        cp_group,
        cp_stream,
    ):
        B, T, nHeads, headDim = q.shape
        cp_size = dist.get_world_size(cp_group)
        rank    = dist.get_rank(cp_group)
        N       = 2 * cp_size

        # compute the chunk ranges for the 2*cp_size segments.
        total_tokens = T
        chunk_ranges = compute_equal_chunks(total_tokens, cp_size)

        # determine the two chunk indices that this rank will process
        local_chunk_ids = [rank, (N - rank - 1)]

        # rearrange and gather the key (K) and value (V) tensors.
        # k_local and v_local are reshaped to have the token dimension first
        k_local = k.movedim(1, 0)  # shape becomes [T, B, nHeads, headDim]
        v_local = v.movedim(1, 0)
        k_ag, _ = gather_along_first_dim(k_local, cp_group)  # resulting shape: [cp_size*T, B, nHeads, headDim]
        v_ag, _ = gather_along_first_dim(v_local, cp_group)

        # perform two forward calls using the chunk indices specified in local_chunk_ids
        out_4d = torch.zeros_like(q)  # initialize output tensor with shape [B, T, nHeads, headDim]
        causal_bool = ("causal" in attn_mask_type)

        # Save necessary data for the backward pass
        fwd_data = {}
        for c_id in local_chunk_ids:
            c_start, c_size = chunk_ranges[c_id]
            if c_size == 0:
                fwd_data[c_id] = None
                continue
            # extract the sub-range from the query tensor corresponding to this chunk (assumes B = 1)
            q_chunk = q[0, c_start : c_start + c_size]  # resulting shape: [c_size, nHeads, headDim]
            # create a cumulative sequence lengths tensor for the query sub-range: [0, c_size]
            cu_q = torch.tensor([0, c_size], device=q.device, dtype=torch.int32)

            # extract the corresponding sub-range from the gathered key and value tensors.
            k_chunk = k_ag[c_start : c_start + c_size]
            v_chunk = v_ag[c_start : c_start + c_size]
            k_chunk_3d = k_chunk[:, 0]  # shape: [c_size, nHeads, headDim]
            v_chunk_3d = v_chunk[:, 0]  # shape: [c_size, nHeads, headDim]

            # set up the cumulative sequence lengths and maximum sequence lengths for the variable-length forward call
            cu_k = torch.tensor([0, c_size], device=k_ag.device, dtype=torch.int32)
            max_seqlen_q = c_size
            max_seqlen_k = c_size

            # execute the variable-length forward attention function
            out_chunk, q_new, k_new, v_new, out_pad, softmax_lse, s_dmask, rng_state = fa_varlen_fwd(
                q_chunk,
                k_chunk_3d,
                v_chunk_3d,
                cu_q, 
                cu_k,
                max_seqlen_q,
                max_seqlen_k,
                float(dropout_p),
                float(softmax_scale if softmax_scale else 1.0),
                bool(causal_bool),
                window_size,
                None, 
                False
            )
            # Insert the computed output chunk into the correct location in the output tensor.
            out_4d[0, c_start : c_start + c_size] = out_chunk

            # save the forward pass data for this chunk for use in the backward pass.
            fwd_data[c_id] = (
                out_chunk,      # output from the forward pass
                softmax_lse,    # log-sum-exp values for softmax
                q_chunk, k_chunk_3d, v_chunk_3d,
                cu_q, cu_k, max_seqlen_q, max_seqlen_k,
                rng_state
            )

        # Reshape the output tensor to 2D: [B, T, nHeads * headDim]
        out_2d = out_4d.view(B, T, nHeads * headDim)

        # Save tensors and context variables needed for the backward pass.
        ctx.save_for_backward(q, k_ag, v_ag)
        ctx.fwd_data       = fwd_data
        ctx.chunk_ranges   = chunk_ranges
        ctx.local_chunk_ids = local_chunk_ids
        ctx.cp_group       = cp_group
        ctx.nHeads         = nHeads
        ctx.headDim        = headDim
        ctx.dropout_p      = dropout_p
        ctx.softmax_scale  = softmax_scale
        ctx.causal_bool    = causal_bool
        ctx.window_size    = window_size
        ctx.T              = T
        return out_2d

    @staticmethod
    def backward(ctx, dout):
        (q, k_ag, v_ag) = ctx.saved_tensors
        fwd_data        = ctx.fwd_data
        chunk_ranges    = ctx.chunk_ranges
        local_chunk_ids = ctx.local_chunk_ids
        cp_group        = ctx.cp_group
        nHeads          = ctx.nHeads
        headDim         = ctx.headDim
        dropout_p       = ctx.dropout_p
        softmax_scale   = ctx.softmax_scale
        causal_bool     = ctx.causal_bool
        window_size     = ctx.window_size
        T               = ctx.T

        B, _, _, _ = q.shape
        # Reshape dout to its 4D form with shape [B, T, nHeads, headDim]
        dout_4d = dout.view(B, T, nHeads, headDim)
        dQ_4d = torch.zeros_like(q)

        # Extract the gathered key and value tensors by removing the batch dimension
        k_ag_3d = k_ag[:, 0]  # shape: [cp_size*T, nHeads, headDim]
        v_ag_3d = v_ag[:, 0]
        dK_ag_3d = torch.zeros_like(k_ag_3d)
        dV_ag_3d = torch.zeros_like(v_ag_3d)

        def partial_accum_dout(c_start, c_size):
            # Extract the gradient corresponding to a specific chunk
            # Returns a tensor of shape [c_size, nHeads, headDim]
            return dout_4d[0, c_start : c_start + c_size]

        for c_id in local_chunk_ids:
            chunk_info = fwd_data.get(c_id, None)
            if chunk_info is None:
                continue
            (out_chunk, softmax_lse, q_chunk, k_chunk_3d, v_chunk_3d,
             cu_q, cu_k, maxQ, maxK, rng_state) = chunk_info
            c_start, c_size = chunk_ranges[c_id]
            dout_chunk_3d = partial_accum_dout(c_start, c_size)

            # Compute gradients for the current chunk using the variable-length backward function
            dq_chunk, dk_chunk, dv_chunk, softmax_d = fa_varlen_bwd(
                dout_chunk_3d,
                q_chunk,
                k_chunk_3d,
                v_chunk_3d,
                out_chunk,
                softmax_lse,
                None,
                None,
                None,
                cu_q, cu_k,
                maxQ, maxK,
                float(dropout_p),
                float(softmax_scale if softmax_scale else 1.0),
                bool(causal_bool),
                window_size,
                None,
                True,
                rng_state
            )
            # accumulate the gradient for q into its corresponding slice
            dQ_4d[0, c_start : c_start + c_size] += dq_chunk

            # accumulate the gradients for k and v into the corresponding slices
            dK_ag_3d[c_start : c_start + c_size].add_(dk_chunk)
            dV_ag_3d[c_start : c_start + c_size].add_(dv_chunk)

        # perform reduce-scatter on the accumulated gradients for k and v
        dK_local_3d, _ = reduce_scatter_along_first_dim(dK_ag_3d, cp_group)
        dV_local_3d, _ = reduce_scatter_along_first_dim(dV_ag_3d, cp_group)
        # reshape the local gradients to match the original tensor shapes: [B, T, nHeads, headDim]
        dK_4d = dK_local_3d.unsqueeze(1).movedim(0, 1).contiguous()
        dV_4d = dV_local_3d.unsqueeze(1).movedim(0, 1).contiguous()

        return (
            None,
            dQ_4d,   # gradient for q with shape [B, T, nHeads, headDim]
            dK_4d,   # gradient for k
            dV_4d,   # gradient for v
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None
        )
