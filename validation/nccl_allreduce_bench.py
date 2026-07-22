#!/usr/bin/env python3
"""Measure real NCCL all-reduce latency/busbw at world_size=4 (intra-socket GPUs 0-3)
vs world_size=8 (cross-socket), across message sizes spanning decode->prefill payloads.
Compare against the simulator's bandwidth-only model (bytes/bottleneck_bw, latency=0).

Run: python3 _nccl_allreduce_bench.py <world_size>
(CUDA_VISIBLE_DEVICES selects which GPUs; rank i -> visible device i)
"""
import os, sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

SIZES = [16*1024, 64*1024, 256*1024, 1024*1024, 4*1024*1024, 16*1024*1024, 64*1024*1024]
# sim model bottleneck bandwidth (GB/s): TP4 -> PCIe 24.5, TP8 -> QPI 21.0
SIM_BW = {4: 24.5, 8: 21.0}

def worker(rank, world):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29555")
    dist.init_process_group("nccl", rank=rank, world_size=world)
    torch.cuda.set_device(rank)
    out = []
    for nb in SIZES:
        x = torch.ones(nb // 2, dtype=torch.float16, device="cuda")  # fp16
        for _ in range(15):
            dist.all_reduce(x)
        torch.cuda.synchronize(); dist.barrier()
        reps = 50
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(reps):
            dist.all_reduce(x)
        e1.record(); torch.cuda.synchronize()
        ms = e0.elapsed_time(e1) / reps
        out.append((nb, ms))
    if rank == 0:
        bw = SIM_BW.get(world, 21.0)
        print(f"\n===== NCCL all-reduce  world_size={world} =====")
        print(f"{'bytes':>10} {'real ms':>9} {'busbw GB/s':>11} {'sim ms(bw-only)':>16} {'real/sim':>9}")
        for nb, ms in out:
            busbw = (2*(world-1)/world) * nb / (ms/1000) / 1e9
            sim_ms = ((2*(world-1)/world) * nb / (bw*1e9)) * 1000  # latency=0 model
            ratio = ms / sim_ms if sim_ms > 0 else float('inf')
            print(f"{nb:>10} {ms:>9.4f} {busbw:>11.1f} {sim_ms:>16.4f} {ratio:>8.1f}x")
    dist.destroy_process_group()

if __name__ == "__main__":
    world = int(sys.argv[1])
    mp.spawn(worker, args=(world,), nprocs=world)
