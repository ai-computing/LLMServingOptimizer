#!/usr/bin/env python3
"""Minimal NCCL transport probe: one all-reduce per world size.
Run with NCCL_DEBUG=INFO; the NCCL init log reveals the transport
(P2P/IPC over NVLink vs SHM) chosen for the given CUDA_VISIBLE_DEVICES set.
Usage: NCCL_DEBUG=INFO python3 nvlink_probe.py <world_size>
"""
import os, sys, torch, torch.distributed as dist, torch.multiprocessing as mp

def worker(rank, world):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29566")
    dist.init_process_group("nccl", rank=rank, world_size=world)
    torch.cuda.set_device(rank)
    x = torch.ones(1024 * 1024, dtype=torch.float16, device="cuda")  # 2 MB
    dist.all_reduce(x)
    torch.cuda.synchronize()
    dist.destroy_process_group()

if __name__ == "__main__":
    mp.spawn(worker, args=(int(sys.argv[1]),), nprocs=int(sys.argv[1]))
