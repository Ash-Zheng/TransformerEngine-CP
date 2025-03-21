

from transformer_engine.pytorch.attention import (
    AttnFuncWithCPAndKVAllGather,
)

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

def run(rank, world_size):
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    dist_group = dist.new_group(ranks=list(range(world_size)))
    
    # ================== #
    # fuction to be tested
    # ================== #

    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = 4
    mp.spawn(run, args=(world_size,), nprocs=world_size)
