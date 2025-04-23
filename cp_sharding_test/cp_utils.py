import torch
from itertools import accumulate

class doc_shard:
    def __init__(self, shard_len, shard_id, doc_id, doc_len, prefix_len):
        """
        Initialize the doc_shard object with shard length, shard ID, prefix length, and document length.
        """
        self.shard_len = shard_len
        self.shard_id = shard_id
        self.doc_id = doc_id
        self.doc_len = doc_len
        self.prefix_len = prefix_len
    
    def __repr__(self):
        """
        String representation of the doc_shard object.
        """
        return f"doc_shard(shard_len={self.shard_len}, shard_id={self.shard_id}, doc_id={self.doc_id}, doc_len={self.doc_len}, prefix_len={self.prefix_len})"


def compute_per_doc_cp_shard_doc_len(doc_lens, context_length, cp_size):
    """
    Compute the per-document sharding for CP (Column Parallel) sharding.
    Each document is divided into chunks of 2 * cp_size.
    """
    n_doc = len(doc_lens)
    doc_shards = [[] for _ in range(2 * cp_size)] # (2 * cp_size, <=n_doc)
    remainder_idx = 0
    for doc_id, doc in enumerate(doc_lens):
        chunk_size = doc // (2 * cp_size)
        tmp_length = [chunk_size] * (2 * cp_size)
        ramainder = doc - chunk_size * (2 * cp_size)
        while ramainder > 0:
            tmp_length[remainder_idx] += 1
            remainder_idx += 1
            remainder_idx = remainder_idx % (2 * cp_size)

        assert sum(tmp_length) == doc, f"Total length {sum(tmp_length)} must equals document length {doc}."
    
        # construct the doc_shard
        prefix_len = 0
        for i in range(2 * cp_size):
            if tmp_length[i] == 0:
                doc_shards[i].append(None)
            else:
                doc_shard_i = doc_shard(tmp_length[i], i, doc_id, doc, prefix_len)
                doc_shards[i].append(doc_shard_i)
                prefix_len += tmp_length[i]

    return doc_shards


def compute_per_doc_metadate(context_length, q, k, v, doc_lens, doc_shards, cp_size, rank, chunk_id):
    """
    Compute the metadata (e.g., cumulative sequence lengths) for per-document CP.
    """
    # ============== Compute metadata =================
    chunk_size = context_length // (2 * cp_size)
    if chunk_id == 0:
        chunk_index = rank
    else:
        chunk_index = 2 * cp_size - 1 - rank

    global_cu_lens =  [0] + list(accumulate(doc_lens))

    this_doc_shards = doc_shards[chunk_index]
    this_chunk_docs = []

    local_q_list = []
    local_k_list = []
    local_v_list = []
    kv_len_list = []

    for doc_shard_i in this_doc_shards:
        # print("doc_shard_i:", doc_shard_i)
        if doc_shard_i is None:
            continue
        else:
            this_chunk_docs.append(doc_shard_i.shard_len)
            q_chunk_start = global_cu_lens[doc_shard_i.doc_id] + doc_shard_i.prefix_len
            q_chunk_end = q_chunk_start + doc_shard_i.shard_len
            local_q_list.append(q[q_chunk_start:q_chunk_end, :, :])

            k_chunk_start = global_cu_lens[doc_shard_i.doc_id]
            k_chunk_end = k_chunk_start + doc_shard_i.prefix_len + doc_shard_i.shard_len
            local_k_list.append(k[k_chunk_start:k_chunk_end, :, :])
            local_v_list.append(v[k_chunk_start:k_chunk_end, :, :])
            kv_len_list.append(doc_shard_i.prefix_len + doc_shard_i.shard_len)
    
    assert sum(this_chunk_docs) == chunk_size, f"Total length {sum(this_chunk_docs)} must equals chunk_size {chunk_size}."

    # print("kv_len_list:", kv_len_list)
    
    local_q = torch.cat(local_q_list, dim=0)
    local_k = torch.cat(local_k_list, dim=0)
    local_v = torch.cat(local_v_list, dim=0)
    cu_seqlens_q = torch.tensor([0] + list(accumulate(this_chunk_docs)), dtype=torch.int32).to(q.device)
    max_seqlen_q = torch.tensor([max(this_chunk_docs)], dtype=torch.int32).to(q.device)

    cu_seqlens_k = torch.tensor([0] + list(accumulate(kv_len_list)), dtype=torch.int32).to(q.device)
    max_seqlen_k = torch.tensor([max(kv_len_list)], dtype=torch.int32).to(q.device)

    # print("cu_seqlens_q:", cu_seqlens_q, "max_seqlen_q:", max_seqlen_q, "cu_seqlens_k:", cu_seqlens_k, "max_seqlen_k:", max_seqlen_k)

    return local_q, local_k, local_v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k


def get_per_doc_local_result(context_length, global_result, doc_lens, doc_shards, cp_size, rank, chunk_id):
    """
    Get the local result for per-doc CP based on the global result.
    """
    chunk_size = context_length // (2 * cp_size)
    if chunk_id == 0:
        chunk_index = rank
    else:
        chunk_index = 2 * cp_size - 1 - rank

    global_cu_lens =  [0] + list(accumulate(doc_lens))

    this_doc_shards = doc_shards[chunk_index]
    this_chunk_docs = []

    local_out_list = []

    for doc_shard_i in this_doc_shards:
        if doc_shard_i is None:
            continue
        else:
            this_chunk_docs.append(doc_shard_i.shard_len)
            chunk_start = global_cu_lens[doc_shard_i.doc_id] + doc_shard_i.prefix_len
            chunk_end = chunk_start + doc_shard_i.shard_len
            local_out_list.append(global_result[chunk_start:chunk_end, :, :])
    
    local_result = torch.cat(local_out_list, dim=0)

    return local_result

def kv_shuffle_for_per_doc_cp(context_length, k_tensor_list, v_tensor_list, doc_lens, doc_shards, cp_size):
    """
    This function has two usages:
    * (1) Use the kv tensors gathered from all ranks and shuffle them to original order (order in global kv tensor).
    * (2) It can also used to shuffle the result on each rank to compare with the original result.
    """
    chunk_size = context_length // (2 * cp_size)
    global_cu_lens =  [0] + list(accumulate(doc_lens))
    global_k = [[] for _ in range(len(doc_lens))]
    global_v = [[] for _ in range(len(doc_lens))]
    for chunk_id in range(2):
        rank_range = range(cp_size) if chunk_id == 0 else range(cp_size - 1, -1, -1)
        for rank in rank_range:
            if chunk_id == 0:
                chunk_index = rank
            else:
                chunk_index = 2 * cp_size - 1 - rank

            k_tensor = k_tensor_list[rank][chunk_id * chunk_size:(chunk_id + 1) * chunk_size, :, :]
            v_tensor = v_tensor_list[rank][chunk_id * chunk_size:(chunk_id + 1) * chunk_size, :, :] if v_tensor_list is not None else None


            this_doc_shards = doc_shards[chunk_index]
            offset = 0
            for doc_shard_i in this_doc_shards:
                if doc_shard_i is not None:
                    this_doc_k = k_tensor[offset:offset + doc_shard_i.shard_len, :, :]
                    this_doc_v = v_tensor[offset:offset + doc_shard_i.shard_len, :, :] if v_tensor is not None else None
                    offset += doc_shard_i.shard_len

                    global_k[doc_shard_i.doc_id].append(this_doc_k)
                    if v_tensor is not None:
                        global_v[doc_shard_i.doc_id].append(this_doc_v)

    # Concatenate the tensors for each chunk
    flat_k = [k_chunk for sub in global_k for k_chunk in sub]
    flat_v = [v_chunk for sub in global_v for v_chunk in sub] if v_tensor_list is not None else None

    # Concatenate the tensors for each chunk
    shuffled_k_tensor = torch.cat(flat_k, dim=0)
    if flat_v is not None:
        shuffled_v_tensor = torch.cat(flat_v, dim=0)
    else:
        shuffled_v_tensor = None

    assert shuffled_k_tensor.shape[0] == context_length, f"shuffled_k_tensor shape {shuffled_k_tensor.shape[0]} must equals context length {context_length}."
                    
    return shuffled_k_tensor, shuffled_v_tensor


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