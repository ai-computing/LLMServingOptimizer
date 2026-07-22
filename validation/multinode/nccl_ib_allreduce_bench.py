#!/usr/bin/env python3
"""Cross-node NCCL all-reduce latency/busbw at world_size=16 (8 GPUs on s8 + 8 on s2),
spanning the InfiniBand tier. Mirrors validation/nccl_allreduce_bench.py (single-node
world_size=4/8) so the IB tier's effective busbw and fixed latency floor can be read off
and fed into the simulator's 4-tier link_bw[3] + collective_overhead node_floor_ns.

Launch via torchrun on BOTH nodes (inside the vllm_mn container, IB env already set):
  # head (s8), rank 0-7:
  torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 \
           --master_addr=192.168.210.108 --master_port=29555 nccl_ib_allreduce_bench.py
  # worker (s2), rank 8-15:
  torchrun --nnodes=2 --nproc_per_node=8 --node_rank=1 \
           --master_addr=192.168.210.108 --master_port=29555 nccl_ib_allreduce_bench.py
"""
import os
import torch
import torch.distributed as dist

SIZES = [16*1024, 64*1024, 256*1024, 1024*1024, 4*1024*1024, 16*1024*1024, 64*1024*1024]
SIM_BW_GBPS = 25.0  # 4-tier link_bw[3] starting estimate (HDR200 nominal); refine from busbw


def main():
    dist.init_process_group("nccl")
    world = dist.get_world_size()
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
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
        print(f"\n===== NCCL all-reduce  world_size={world} (cross-node, IB) =====")
        print(f"{'bytes':>10} {'real ms':>9} {'busbw GB/s':>11} {'sim ms(bw-only)':>16} {'real/sim':>9}")
        for nb, ms in out:
            busbw = (2*(world-1)/world) * nb / (ms/1000) / 1e9
            sim_ms = ((2*(world-1)/world) * nb / (SIM_BW_GBPS*1e9)) * 1000  # latency=0 model
            ratio = ms / sim_ms if sim_ms > 0 else float('inf')
            print(f"{nb:>10} {ms:>9.4f} {busbw:>11.1f} {sim_ms:>16.4f} {ratio:>8.1f}x")
        print("\n# IB tier read-off: floor ~= small-message real ms (16KB decode payload);")
        print("# effective busbw from the large-message rows feeds link_bw[3].")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
