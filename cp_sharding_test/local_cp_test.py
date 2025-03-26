from transformer_engine.pytorch.attention import (
    AttnFuncWithCPAndKVAllGather,
)
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import os
import random

# global parameters
CONTEXT_LENGTH = 128 * 1024  # 128k context length (# tokens)
NUM_HEADS = 8
HEAD_DIM = 64  # must be a multiple of 8

# helper: given a list of document lengths, ensure they sum exactly to CONTEXT_LENGTH
def adjust_docs(docs):
    total = sum(docs)
    if total != CONTEXT_LENGTH:
        # adjust the last document so that the sum matches CONTEXT_LENGTH
        docs[-1] += CONTEXT_LENGTH - total
    return docs

# define benchmarks: each benchmark is a list of document lengths that sum to CONTEXT_LENGTH
def get_benchmarks():
    benchmarks = {}

    # 1. single Long Document: one document fills the whole context
    benchmarks["single_long"] = adjust_docs([CONTEXT_LENGTH])

    # 2. many Short Documents: many documents of 1k tokens
    num_docs = CONTEXT_LENGTH // 1024
    benchmarks["many_short"] = adjust_docs([1024] * num_docs)

    # 3. mixed: one long doc (100k tokens) then many short docs (1k tokens) filling the rest
    long_doc = 100 * 1024
    num_short = (CONTEXT_LENGTH - long_doc) // 1024
    benchmarks["mixed"] = adjust_docs([long_doc] + [1024] * num_short)

    # 4. alternating Long and Short: alternate between a moderately long doc (32k tokens) and a short doc (1k tokens)
    docs_alternating = []
    remaining = CONTEXT_LENGTH
    toggle = True
    while remaining > 0:
        if toggle:
            doc_len = min(32000, remaining)
        else:
            doc_len = min(1024, remaining)
        docs_alternating.append(doc_len)
        remaining -= doc_len
        toggle = not toggle
    benchmarks["alternating"] = adjust_docs(docs_alternating)

    # 5. randomized: random document lengths between 512 and 16k tokens
    random.seed(42)
    docs_random = []
    remaining = CONTEXT_LENGTH
    while remaining > 0:
        doc_len = random.randint(512, 16 * 1024)
        doc_len = min(doc_len, remaining)
        docs_random.append(doc_len)
        remaining -= doc_len
    benchmarks["random"] = adjust_docs(docs_random)

    # 6. edge-case: fixed size documents of 4096 tokens, with minimal leftover
    fixed_size = 4096
    num_full = CONTEXT_LENGTH // fixed_size
    docs_edge = [fixed_size] * num_full
    leftover = CONTEXT_LENGTH - sum(docs_edge)
    if leftover > 0:
        docs_edge.append(leftover)
    benchmarks["edge_case"] = adjust_docs(docs_edge)

    return benchmarks

def run(rank, world_size):
    # set environment variables for torch
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # each process sets its GPU device
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)

    # initialize the distributed process group
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    # create a new group and cuda streams for cp
    cp_group = dist.new_group(ranks=list(range(world_size)))
    cp_stream = torch.cuda.Stream(device=device)

    # create a dummy cu_seqlens tensor
    cu_seqlens_q = torch.tensor([0, CONTEXT_LENGTH], device=device, dtype=torch.int32)
    cu_seqlens_q_padded = cu_seqlens_q.clone()

    # set other parameterrs for the attention function
    is_training = True
    max_seqlen_q = CONTEXT_LENGTH
    max_seqlen_kv = CONTEXT_LENGTH
    dropout_p = 0.0
    softmax_scale = None
    qkv_format = "bshd"  # batch, sequence, heads, hidden dimension
    attn_mask_type = "causal"
    attn_bias_type = "no_bias"
    attn_bias = None
    deterministic = False
    use_fused_attention = False  # for this test, use the flash attention path
    # update window_size to be a tuple, e.g., (128, 128) rather than an int
    window_size = (128, 128)

    # define a mapping from benchmark name to a constant value to fill the tensors
    value_map = {
        "single_long": 1.0,
        "many_short": 2.0,
        "mixed": 3.0,
        "alternating": 4.0,
        "random": 5.0,
        "edge_case": 6.0,
    }

    benchmarks = get_benchmarks()

    # loop over the benchmarks
    for bench_name, docs in benchmarks.items():
        # for verification, ensure the sum of document lengths equals CONTEXT_LENGTH
        total_tokens = sum(docs)
        assert total_tokens == CONTEXT_LENGTH, f"Benchmark {bench_name} total tokens {total_tokens} != {CONTEXT_LENGTH}"

        if rank == 0:
            print(f"\nBenchmark: {bench_name}")
            print(f"Document lengths: {docs[:5]}{' ...' if len(docs)>5 else ''} (total {len(docs)} docs)")

        # create q, k, v tensors with layout [B, S, num_heads, head_dim]
        B = 1
        S = CONTEXT_LENGTH
        np_heads = NUM_HEADS
        h_dim = HEAD_DIM
        # use random tokens for a more realistic test
        q = torch.randn((B, S, np_heads, h_dim), device=device, dtype=torch.float16)
        k = torch.randn((B, S, np_heads, h_dim), device=device, dtype=torch.float16)
        v = torch.randn((B, S, np_heads, h_dim), device=device, dtype=torch.float16)

        # enable gradient tracking
        q.requires_grad = True
        k.requires_grad = True
        v.requires_grad = True

        # forward pass
        # timing the forward pass
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        out = AttnFuncWithCPAndKVAllGather.apply(
            is_training,
            q,
            k,
            v,
            cu_seqlens_q,
            max_seqlen_q,
            max_seqlen_kv,
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
            cp_stream,
        )
        end_event.record()
        torch.cuda.synchronize(device)
        time_ms = start_event.elapsed_time(end_event)
        # record FLOPs performance: FLOPs = 4 * B * S^2 * HEAD_DIM * NUM_HEADS
        flops = 4 * B * (S ** 2) * HEAD_DIM * NUM_HEADS
        time_sec = time_ms / 1000.0
        tflops = flops / (time_sec * 1e12)
        print(f"Rank {rank} Benchmark {bench_name}: TFLOPS: {tflops:.2f}")

        # in the non-fused branch with "bshd", the output is flattened to [S, num_heads, head_dim]
        if not use_fused_attention and qkv_format == "bshd":
            expected_shape = (S, np_heads, h_dim)
        else:
            expected_shape = q.shape

        if out.shape != expected_shape:
            print(f"Rank {rank} Benchmark {bench_name}: Output shape {out.shape} does not match expected {expected_shape}")
        else:
            print(f"Rank {rank} Benchmark {bench_name}: Output shape OK: {out.shape}")

        # backward pass
        loss = out.sum()
        loss.backward()
        q_grad_norm = q.grad.norm().item() if q.grad is not None else 0.0
        k_grad_norm = k.grad.norm().item() if k.grad is not None else 0.0
        v_grad_norm = v.grad.norm().item() if v.grad is not None else 0.0
        print(f"Rank {rank} Benchmark {bench_name}: Gradients norm: q: {q_grad_norm:.4f}, k: {k_grad_norm:.4f}, v: {v_grad_norm:.4f}")

    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 4
    mp.spawn(run, args=(world_size,), nprocs=world_size)
