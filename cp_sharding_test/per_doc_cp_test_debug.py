#!/usr/bin/env python
import sys
import os
import random
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# add project root to path if needed.
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from transformer_engine.pytorch.attention import AttnFuncWithCPAndKVAllGather
from per_document_cp_sharding.per_document_cp_sharding import AttnFuncWithPerDocRoundRobinSharding
from transformer_engine.pytorch.distributed import get_distributed_world_size, get_distributed_rank
from transformer_engine.pytorch.utils import nvtx_range_push, nvtx_range_pop

# global parameters
CONTEXT_LENGTH = 128 * 1024  # 128k tokens
NUM_HEADS = 8
HEAD_DIM = 64              # must be a multiple of 8
D = NUM_HEADS * HEAD_DIM   # merged token dimension

def print_tensor_stats(tensor, name):
    tensor = tensor.float()  # ensure float for printing stats
    print(f"{name}: mean={tensor.mean().item():.4f}, std={tensor.std().item():.4f}, min={tensor.min().item():.4f}, max={tensor.max().item():.4f}")

def print_stats(ref, test, boundaries, tensor_name="Output"):
    for i in range(len(boundaries)-1):
        start = boundaries[i]
        end = boundaries[i+1]
        ref_slice = ref[:, start:end, :]
        test_slice = test[:, start:end, :]
        diff = (ref_slice - test_slice).abs()
        print(f"Document {i} (tokens {start}-{end}, length={end-start}):")
        print(f"  {tensor_name} diff: mean={diff.mean().item():.4f}, std={diff.std().item():.4f}, min={diff.min().item():.4f}, max={diff.max().item():.4f}")
        print("  Ref stats:")
        print_tensor_stats(ref_slice, "    ")
        print("  Test stats:")
        print_tensor_stats(test_slice, "    ")

def get_single_document():
    # return a single document spanning the full context length.
    return [CONTEXT_LENGTH]

def get_eight_documents():
    # generate 8 random document lengths that sum to CONTEXT_LENGTH.
    # sample 7 random breakpoints within [1, CONTEXT_LENGTH-1]
    # then compute the differences.
    pts = sorted(random.sample(range(1, CONTEXT_LENGTH), 7))
    docs = [pts[0]] + [pts[i] - pts[i-1] for i in range(1, len(pts))] + [CONTEXT_LENGTH - pts[-1]]
    # to reduce variability in logging, print the document lengths.
    print("Eight document lengths:", docs)
    return docs

def run(rank, world_size, doc_lens):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    cp_group = dist.new_group(ranks=list(range(world_size)))
    cp_stream = torch.cuda.Stream(device=device)

    total_tokens = sum(doc_lens)
    assert total_tokens == CONTEXT_LENGTH, f"Total tokens {total_tokens} != {CONTEXT_LENGTH}"

    # compute document boundaries based on doc_lens.
    boundaries = [0]
    for l in doc_lens:
        boundaries.append(boundaries[-1] + l)
    print(f"[Rank {rank}] Document boundaries: {boundaries}")

    # print expected shard boundaries for each document.
    def compute_expected_shard_boundaries(doc_len, cp_size, rank):
        N = 2 * cp_size
        seg_len_base = doc_len // N
        rem = doc_len % N
        extra0 = 1 if rank < rem else 0
        seg0_start = rank * seg_len_base + min(rank, rem)
        seg0_end = seg0_start + seg_len_base + extra0
        r2 = N - rank - 1
        extra1 = 1 if r2 < rem else 0
        seg1_start = r2 * seg_len_base + min(r2, rem)
        seg1_end = seg1_start + seg_len_base + extra1
        return seg0_start, seg0_end, seg1_start, seg1_end

    cp_size = world_size  # for testing
    print(f"[Rank {rank}] Expected shard boundaries per document:")
    for i, l in enumerate(doc_lens):
        s0, e0, s1, e1 = compute_expected_shard_boundaries(l, cp_size, rank)
        print(f" Document {i}: length={l}, seg0: [{s0}, {e0}), seg1: [{s1}, {e1})")

    # Create identical random inputs in bshd layout.
    # Force values to be in the range (-0.5, +0.5) by using torch.rand and shifting.
    q_seq = (torch.rand((1, CONTEXT_LENGTH, NUM_HEADS, HEAD_DIM), device=device, dtype=torch.bfloat16) - 0.5).requires_grad_()
    k_seq = (torch.rand((1, CONTEXT_LENGTH, NUM_HEADS, HEAD_DIM), device=device, dtype=torch.bfloat16) - 0.5).requires_grad_()
    v_seq = (torch.rand((1, CONTEXT_LENGTH, NUM_HEADS, HEAD_DIM), device=device, dtype=torch.bfloat16) - 0.5).requires_grad_()
    q_doc = q_seq.clone().detach().requires_grad_()
    k_doc = k_seq.clone().detach().requires_grad_()
    v_doc = v_seq.clone().detach().requires_grad_()

    # common parameters
    is_training = True
    dropout_p = 0.0
    softmax_scale = None
    qkv_format = "bshd"
    attn_mask_type = "causal"
    attn_bias_type = "no_bias"
    attn_bias = None
    deterministic = False
    use_fused_attention = False
    window_size = (128, 128)
    cu_seqlens_q = torch.tensor([0, CONTEXT_LENGTH], device=device, dtype=torch.int32)
    cu_seqlens_q_padded = cu_seqlens_q.clone()

    # run per-sequence CP sharding
    out_seq = AttnFuncWithCPAndKVAllGather.apply(
        is_training,
        q_seq,
        k_seq,
        v_seq,
        cu_seqlens_q,
        CONTEXT_LENGTH,
        CONTEXT_LENGTH,
        cu_seqlens_q_padded,
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
        cp_stream
    )
    # reshape to [1, CONTEXT_LENGTH, D]
    out_seq = out_seq.view(1, CONTEXT_LENGTH, D)

    # run per-document CP sharding
    out_doc = AttnFuncWithPerDocRoundRobinSharding.apply(
        is_training,
        q_doc,
        k_doc,
        v_doc,
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
        cp_stream
    )
    # out_doc is expected to be [1, CONTEXT_LENGTH, D]

    torch.cuda.synchronize(device)

    # intermediate logging: print overall output tensor statistics
    print(f"[Rank {rank}] Per-sequence output stats:")
    print_tensor_stats(out_seq.float(), "  ")
    print(f"[Rank {rank}] Per-document output stats:")
    print_tensor_stats(out_doc.float(), "  ")

    full_diff = (out_seq.float() - out_doc.float()).abs().max().item()
    print(f"[Rank {rank}] Full output max diff = {full_diff:.4f}")
    if full_diff > 1e-2:
        print(f"[Rank {rank}] Full outputs differ!")
    else:
        print(f"[Rank {rank}] Full outputs match.")

    # log per-document differences
    print("\nPer-document output differences:")
    print_stats(out_seq.float(), out_doc.float(), boundaries, tensor_name="Output")

    # backward pass
    loss_seq = out_seq.sum()
    loss_seq.backward()
    grad_q_seq = q_seq.grad.clone()

    loss_doc = out_doc.sum()
    loss_doc.backward()
    grad_q_doc = q_doc.grad.clone()

    print("\nGradient statistics for q (per-sequence):")
    print_tensor_stats(grad_q_seq.view(1, CONTEXT_LENGTH, D).float(), "  ")
    print("\nGradient statistics for q (per-document):")
    print_tensor_stats(grad_q_doc.view(1, CONTEXT_LENGTH, D).float(), "  ")

    full_grad_diff = (grad_q_seq.view(1, CONTEXT_LENGTH, D).float() - grad_q_doc.view(1, CONTEXT_LENGTH, D).float()).abs().max().item()
    print(f"[Rank {rank}] Full q grad max diff = {full_grad_diff:.4f}")
    if full_grad_diff > 1e-2:
        print(f"[Rank {rank}] Full q gradients differ!")
    else:
        print(f"[Rank {rank}] Full q gradients match.")

    print("\nPer-document q grad differences:")
    print_stats(grad_q_seq.view(1, CONTEXT_LENGTH, D).float(), grad_q_doc.view(1, CONTEXT_LENGTH, D).float(), boundaries, tensor_name="q grad")

    dist.destroy_process_group()

if __name__ == "__main__":
    # select test case via command-line argument: "single" or "multiple"
    test_case = sys.argv[1] if len(sys.argv) > 1 else "single"
    if test_case == "single":
        print("Running single-document test case.")
        doc_lens = get_single_document()
    elif test_case == "multiple":
        print("Running multiple-document test case (8 documents).")
        doc_lens = get_eight_documents()
    else:
        print("Unknown test case. Use 'single' or 'multiple'.")
        sys.exit(1)

    world_size = 2
    mp.spawn(run, args=(world_size, doc_lens), nprocs=world_size)
