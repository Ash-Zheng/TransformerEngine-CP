#!/usr/bin/env python
import sys
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# add project root to path 
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Per-document forward pass.
from per_document_cp_sharding.per_document_cp_sharding import AttnFuncWithAllGatherPerDocSharding
# Per-sequence forward pass.
from transformer_engine.pytorch.attention import AttnFuncWithCPAndKVAllGather

try:
    from flash_attn.flash_attn_interface import _flash_attn_forward as _flash_attn_fwd
except ImportError:
    _flash_attn_fwd = None

#########################################
# Utility Functions
#########################################

def generate_global_tokens():
    """
    Generate a global token sequence.
    In each document, the last token is -1 (eos).
    """
    tokens = []
    doc_lens = [1000, 100, 100, 100, 100, 100, 100, 100, 100]
    total_tokens = 0
    for i in range(len(doc_lens)):
        # generate doc tokens, last token is -1 (eos)
        doc_tokens = list(range(total_tokens, total_tokens + doc_lens[i] - 1))
        doc_tokens.append(-1)
        tokens.extend(doc_tokens)
        total_tokens += doc_lens[i]
    return torch.tensor([tokens], dtype=torch.int64)

def embed_tokens(tokens):
    """
    Embeds tokens into vectors of size 8x64.
    Each token is mapped to a vector of size 8x64 with all entries equal to the token id (as float).
    The output shape is [B, T, 8, 64] and is converted to bfloat16.
    """
    B, T = tokens.shape
    emb = tokens.float().unsqueeze(-1).unsqueeze(-1).expand(B, T, 8, 64)
    return emb.to(torch.bfloat16)

#########################################
# Per-document Mapping Helpers
#########################################

def global_to_local_doc(global_tokens, global_doc_lens, rank, cp_size):
    """
    Maps a global token tensor (shape [B, T]) to local tokens for per-document CP.
    global_doc_lens is a list (for B=1) of document lengths, e.g. [1000, 100, 100, ...].
    For cp_size=2 (num_chunks=4), each document is split into 4 equal contiguous chunks.
      - For rank 0: local tokens = concatenation of chunk0 and chunk3.
      - For rank 1: local tokens = concatenation of chunk1 and chunk2.
    Returns a tensor of shape [B, T_local] where T_local = sum_d (L_d // cp_size).
    """
    B, T = global_tokens.shape
    num_chunks = 2 * cp_size  # 4
    local_tokens_list = []
    for b in range(B):
        tokens = global_tokens[b]
        local_doc_list = []
        start = 0
        for L in global_doc_lens[b]:
            doc = tokens[start:start+L]
            base = L // num_chunks  # assumes divisible
            chunks = [doc[j*base:(j+1)*base] for j in range(num_chunks)]
            if rank == 0:
                local_doc = torch.cat([chunks[0], chunks[num_chunks - 1]], dim=0)
            else:
                local_doc = torch.cat([chunks[rank], chunks[num_chunks - 1 - rank]], dim=0)
            local_doc_list.append(local_doc)
            start += L
        local_tokens_list.append(torch.cat(local_doc_list, dim=0).unsqueeze(0))
    return torch.cat(local_tokens_list, dim=0)

def compute_local_doc_lens(global_doc_lens, cp_size, rank):
    """
    For per-document CP, each document's local length = global_length // cp_size.
    For example, if global_doc_lens = [1000, 100, 100, ...] and cp_size=2,
    then local_doc_lens = [500, 50, 50, ...].
    """
    return [L // cp_size for L in global_doc_lens]

def map_local_to_global_doc(doc_lens, gathered_outputs):
    """
    Reassemble the global output from gathered local outputs for per-document CP.
    
    Args:
      doc_lens: a list of global document lengths,
      gathered_outputs: list of length cp_size, where gathered_outputs[r] is of shape [1, local_sum, nHeads, headDim],
          with local_sum = sum_d (L_d // cp_size).
    
    For each document d, each worker’s local output is split into two equal halves (front and back),
    where each half has length L_d // (2 * cp_size). The global output for document d is assembled by:
       [front from worker 0, front from worker 1, …, front from worker (R-1),
        back from worker (R-1), …, back from worker 0].
    """
    cp_size = len(gathered_outputs)
    shard_len_list = [L // (2 * cp_size) for L in doc_lens]
    boundaries = [0]
    for L in doc_lens:
        boundaries.append(boundaries[-1] + (L // cp_size))
    reassembled_docs = []
    for d in range(len(doc_lens)):
        front_shard_len = shard_len_list[d]
        doc_shards = []
        for r in range(cp_size):
            local_out = gathered_outputs[r][0]  # shape: [local_sum, nHeads, headDim]
            doc_portion = local_out[boundaries[d]:boundaries[d+1], :, :]
            front = doc_portion[:front_shard_len, :, :]
            back  = doc_portion[front_shard_len:, :, :]
            doc_shards.append((front, back))
        front_parts = [doc_shards[r][0] for r in range(cp_size)]
        back_parts  = [doc_shards[r][1] for r in reversed(range(cp_size))]
        doc_global = torch.cat(front_parts + back_parts, dim=0)
        reassembled_docs.append(doc_global)
    global_out = torch.cat(reassembled_docs, dim=0).unsqueeze(0)
    return global_out

#########################################
# Per-sequence Mapping Helpers
#########################################

def global_to_local_seq_custom(global_tokens, global_doc_lens, rank, cp_size):
    """
    Implements the per-sequence global-to-local mapping with symmetric sharding.
    For each document in global_tokens with length L (from global_doc_lens), assume L is divisible by 2*cp_size.
    Split the document into num_chunks = 2*cp_size contiguous chunks (each of length L_chunk = L // (2*cp_size)).
    Then, for a CP worker with rank r, local tokens = concat( chunk[r], chunk[2*cp_size - 1 - r] ).
    For multiple documents (B=1), do this for each document and concatenate.
    Returns a tensor of shape [B, T_local] where T_local = sum_d (L_d // cp_size).
    """
    B, T = global_tokens.shape
    local_tokens_list = []
    for b in range(B):
        tokens = global_tokens[b]
        doc_tokens_list = []
        start = 0
        for L in global_doc_lens[b]:
            doc = tokens[start:start+L]
            num_chunks = 2 * cp_size
            L_chunk = L // num_chunks
            chunks = [doc[i*L_chunk:(i+1)*L_chunk] for i in range(num_chunks)]
            local_doc = torch.cat([chunks[rank], chunks[num_chunks - 1 - rank]], dim=0)
            doc_tokens_list.append(local_doc)
            start += L
        local_tokens_list.append(torch.cat(doc_tokens_list, dim=0).unsqueeze(0))
    return torch.cat(local_tokens_list, dim=0)

def map_local_to_global_seq_custom(doc_lens, gathered_outputs):
    """
    Reassemble the global output from gathered local outputs for per-sequence CP using symmetric reordering.
    
    Args:
      doc_lens: a list of global document lengths [L0, L1, ..., L_{D-1}] for B=1.
                Each L is assumed divisible by (2 * cp_size).
      gathered_outputs: list of length cp_size, where each tensor is of shape [T_local, nHeads, headDim],
          with T_local = sum_d (L_d // cp_size).
    
    For each document d, each worker’s local slice (length = L_d_local = L_d // cp_size) is split evenly into:
         front: first half (length = L_d // (2*cp_size))
         back:  second half (length = L_d // (2*cp_size))
    The global order for document d is:
         [front from worker 0, front from worker 1, …, front from worker (R-1),
          back from worker (R-1), …, back from worker 0]
    Returns a tensor of shape [1, sum_d L_d, nHeads, headDim].
    """
    cp_size = len(gathered_outputs)
    shard_len_list = [L // (2 * cp_size) for L in doc_lens]
    boundaries = [0]
    for L in doc_lens:
        boundaries.append(boundaries[-1] + (L // cp_size))
    reassembled_docs = []
    for d in range(len(doc_lens)):
        front_shard_len = shard_len_list[d]
        doc_shards = []
        for r in range(cp_size):
            local_out = gathered_outputs[r]  # shape: [T_local, nHeads, headDim]
            doc_portion = local_out[boundaries[d]:boundaries[d+1], :, :]
            front = doc_portion[:front_shard_len, :, :]
            back  = doc_portion[front_shard_len:, :, :]
            doc_shards.append((front, back))
        front_parts = [doc_shards[r][0] for r in range(cp_size)]
        back_parts  = [doc_shards[r][1] for r in reversed(range(cp_size))]
        doc_global = torch.cat(front_parts + back_parts, dim=0)
        reassembled_docs.append(doc_global)
    global_out = torch.cat(reassembled_docs, dim=0).unsqueeze(0)
    return global_out

#########################################
# Distributed Run Function
#########################################

def run(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    
    # Initialize NCCL distributed process group.
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    cp_group = dist.new_group(ranks=list(range(world_size)))
    
    ###############################
    # Per-document Forward Pass
    ###############################
    # Use global_doc_lens_doc consistent with generate_global_tokens().
    global_doc_lens_doc = [1000, 100, 100, 100, 100, 100, 100, 100, 100]
    global_tokens_doc = generate_global_tokens() 
    global_tokens_doc = global_tokens_doc.cuda()  # shape [1, 1800]
    
    # Compute local tokens for per-document CP.
    local_tokens_doc = global_to_local_doc(global_tokens_doc, [global_doc_lens_doc], rank, world_size)
    q_doc = embed_tokens(local_tokens_doc)  # shape [B, T_local, 8, 64]
    k_doc = q_doc.clone()
    v_doc = q_doc.clone()
    local_doc_lens = compute_local_doc_lens(global_doc_lens_doc, world_size, rank) 
    local_doc_lens_nested = [local_doc_lens]  # Now a list of lengths per document
    out_doc = AttnFuncWithAllGatherPerDocSharding.apply(
        True,         # is_training
        q_doc,        # [B, T_local, 8, 64]
        k_doc,
        v_doc,
        local_doc_lens_nested,
        0.0,
        None,
        "bshd",
        "causal",
        "no_bias",
        None,
        False,
        False,
        (0, 0),
        cp_group,
        torch.cuda.current_stream()
    )
    
    ###############################
    # Per-sequence Forward Pass
    ###############################
    global_doc_lens_seq = [1000 + 100 + 100 + 100 + 100 + 100 + 100 + 100 + 100]
    global_tokens_seq = generate_global_tokens() 
    global_tokens_seq = global_tokens_seq.cuda()  
    local_tokens_seq = global_to_local_seq_custom(global_tokens_seq, [global_doc_lens_seq], rank, world_size)
    q_seq = embed_tokens(local_tokens_seq)  # shape [B, T_local, 8, 64]
    k_seq = q_seq.clone()
    v_seq = q_seq.clone()
    T_local_seq = q_seq.shape[1]
    max_seqlen_q = T_local_seq * world_size
    max_seqlen_kv = T_local_seq * world_size
    cu_seqlens_q = torch.tensor([0, max_seqlen_q], device=q_seq.device, dtype=torch.int32)
    cu_seqlens_q_padded = None
    out_seq = AttnFuncWithCPAndKVAllGather.apply(
        True,
        q_seq,
        k_seq,
        v_seq,
        cu_seqlens_q,
        max_seqlen_q,
        max_seqlen_kv,
        cu_seqlens_q_padded,
        0.0,
        None,
        "bshd",
        "causal",
        "no_bias",
        None,
        False,
        False,
        (0, 0),
        cp_group,
        torch.cuda.current_stream()
    )
    
    # Gather outputs.
    gathered_doc = [torch.empty_like(out_doc) for _ in range(world_size)]
    gathered_seq = [torch.empty_like(out_seq) for _ in range(world_size)]
    dist.all_gather(gathered_doc, out_doc)
    dist.all_gather(gathered_seq, out_seq)
    if rank == 0:
        global_out_doc = map_local_to_global_doc(global_doc_lens_doc, gathered_doc)
        global_out_seq = map_local_to_global_seq_custom(global_doc_lens_seq, gathered_seq)
        diff_doc_seq = torch.norm(global_out_doc.float() - global_out_seq.float())
        print(f"[Rank {rank}] Global output difference (per-doc vs per-seq): {diff_doc_seq.item()}", flush=True)
    
    dist.destroy_process_group()
    
    #####################################
    # Extra Section: Standard Flash-Attn
    #####################################
    # Outside distributed setting, standard flash attention forward uses the full global embedding.
    global_tokens_std = generate_global_tokens().cuda()  # shape [1, 1800]
    global_q_std = embed_tokens(global_tokens_std)  # shape [1, 1800, 8, 64]
    global_k_std = global_q_std.clone()
    global_v_std = global_q_std.clone()
    global_out_std = _flash_attn_fwd(
        q=global_q_std,
        k=global_k_std,
        v=global_v_std,
        dropout_p=0.0,
        softmax_scale=global_q_std.shape[-1] ** -0.5,
        causal=True,
        window_size=(0, 0),
        alibi_slopes=None,
        return_softmax=False
    )
    global_out_std_tensor = global_out_std[0]
    if rank == 0:
        diff_seq_std = torch.norm(global_out_seq.float() - global_out_std_tensor.float())
        diff_doc_std = torch.norm(global_out_doc.float() - global_out_std_tensor.float())
        print(f"[Rank {rank}] Global output difference (per-seq vs standard): {diff_seq_std.item()}", flush=True)
        print(f"[Rank {rank}] Global output difference (per-doc vs standard): {diff_doc_std.item()}", flush=True)

if __name__ == "__main__":
    world_size = 2
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)
