#!/usr/bin/env python3
"""Live capacity-curve campaign runner (hw path of plan M6).

Boots a vLLM server on the given GPUs, sweeps per-instance concurrency with
``vllm bench serve`` (random dataset at chat-like lengths), samples GPU power
via nvidia-smi in parallel, and writes the measured-backend oracle YAML
through sim_backends.measured.campaign.capacity_campaign.run_campaign.

Usage (A5000 tp1 on GPU 0):
    python scripts/run_capacity_campaign.py --hw A5000 \
        --model meta-llama/Llama-3.1-8B --tp 1 --gpus 0 \
        --concurrencies 1,2,4,8,16,32,64

Requires: backends/upstream/.venv-vllm with vllm installed, model weights in
the HF cache, idle GPUs. Takes ~10-15 min per (hw, tp).
"""
from __future__ import annotations

import argparse
import datetime
import json
import signal
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from sim_backends.measured.campaign.capacity_campaign import (  # noqa: E402
    CampaignConfig,
    run_campaign,
)

VLLM = REPO_ROOT / "backends/upstream/.venv-vllm/bin/vllm"


class PowerSampler:
    """Samples total power (W) over the given GPUs once per second."""

    def __init__(self, gpus: str):
        self.gpus = gpus
        self.samples: list[float] = []
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None

    def start(self):
        self._proc = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=power.draw",
             "--format=csv,noheader,nounits", "-l", "1", "-i", self.gpus],
            stdout=subprocess.PIPE, text=True)
        n_gpus = len(self.gpus.split(","))
        buf: list[float] = []

        def pump():
            for line in self._proc.stdout:
                try:
                    buf.append(float(line.strip()))
                except ValueError:
                    continue
                if len(buf) == n_gpus:          # one reading per GPU per tick
                    self.samples.append(sum(buf))
                    buf.clear()

        self._thread = threading.Thread(target=pump, daemon=True)
        self._thread.start()

    def stop(self) -> float:
        if self._proc:
            self._proc.send_signal(signal.SIGTERM)
            self._proc.wait(timeout=10)
        if self._thread:
            self._thread.join(timeout=5)
        return statistics.mean(self.samples) if self.samples else 0.0


def wait_health(port: int, timeout_s: int = 600) -> None:
    deadline = time.time() + timeout_s
    url = f"http://localhost:{port}/health"
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return
        except Exception:
            time.sleep(3)
    raise TimeoutError(f"vLLM server on :{port} not healthy after {timeout_s}s")


def bench_once(args, concurrency: int, result_dir: Path) -> dict:
    """One bench run at the given concurrency; returns the saved metrics JSON."""
    num_prompts = max(16, min(4 * concurrency, 256))
    fname = f"bench_c{concurrency}.json"
    cmd = [str(VLLM), "bench", "serve",
           "--backend", "vllm",
           "--base-url", f"http://localhost:{args.port}",
           "--model", args.model,
           "--dataset-name", "random",
           "--random-input-len", str(args.input_len),
           "--random-output-len", str(args.output_len),
           "--num-prompts", str(num_prompts),
           "--max-concurrency", str(concurrency),
           "--ignore-eos",
           "--save-result", "--result-dir", str(result_dir),
           "--result-filename", fname]
    print(f"[campaign] c={concurrency}: {num_prompts} prompts ...", flush=True)
    subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=1800)
    return json.loads((result_dir / fname).read_text())


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--hw", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--gpus", default="0", help="CUDA_VISIBLE_DEVICES list")
    p.add_argument("--port", type=int, default=8100)
    p.add_argument("--concurrencies", default="1,2,4,8,16,32,64")
    p.add_argument("--input-len", type=int, default=256)
    p.add_argument("--output-len", type=int, default=256)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--out-root", default=str(REPO_ROOT / "profiles/measured"))
    args = p.parse_args()

    concurrencies = [int(c) for c in args.concurrencies.split(",")]
    workdir = REPO_ROOT / "output" / "campaign" / \
        f"{args.hw}_tp{args.tp}_{datetime.date.today().isoformat()}"
    workdir.mkdir(parents=True, exist_ok=True)

    import os
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = args.gpus
    env.setdefault("HF_HUB_OFFLINE", "1")   # weights are already cached

    print(f"[campaign] starting vLLM server (tp{args.tp}, GPUs {args.gpus}) ...",
          flush=True)
    server_log = open(workdir / "server.log", "w")
    server = subprocess.Popen(
        [str(VLLM), "serve", args.model,
         "--dtype", "bfloat16",
         "--tensor-parallel-size", str(args.tp),
         "--max-model-len", str(args.max_model_len),
         "--max-num-seqs", "128",
         "--max-num-batched-tokens", "2048",
         "--block-size", "16",
         "--port", str(args.port)],
        stdout=server_log, stderr=subprocess.STDOUT, env=env)
    try:
        wait_health(args.port)
        print("[campaign] server healthy; sampling idle power (15s) ...", flush=True)
        idle = PowerSampler(args.gpus)
        idle.start()
        time.sleep(15)
        idle_w = idle.stop()
        print(f"[campaign] idle_w = {idle_w:.1f} W", flush=True)

        def measure(c: int) -> dict:
            sampler = PowerSampler(args.gpus)
            sampler.start()
            try:
                d = bench_once(args, c, workdir)
            finally:
                avg_w = sampler.stop()
            point = {
                "thr_toks_s": float(d.get("output_throughput", 0.0)),
                "ttft_ms": float(d.get("mean_ttft_ms", 0.0)),
                "tpot_ms": float(d.get("mean_tpot_ms", 0.0)),
                "itl_p99_ms": float(d.get("p99_itl_ms")
                                    or d.get("p99_tpot_ms") or 0.0),
                "avg_w": avg_w,
            }
            print(f"[campaign] c={c}: {point}", flush=True)
            return point

        cfg = CampaignConfig(
            hw=args.hw, model=args.model, tp=args.tp,
            concurrencies=concurrencies,
            stack=f"vllm-{_vllm_version()}",
            measured_at=datetime.date.today().isoformat(),
            idle_w=round(idle_w, 1),
            out_root=args.out_root)
        path = run_campaign(cfg, measure_fn=measure)
        print(f"[campaign] wrote oracle: {path}", flush=True)
        return 0
    finally:
        server.send_signal(signal.SIGINT)
        try:
            server.wait(timeout=60)
        except subprocess.TimeoutExpired:
            server.kill()
        server_log.close()
        print("[campaign] server stopped", flush=True)


def _vllm_version() -> str:
    try:
        out = subprocess.run([str(VLLM), "--version"], capture_output=True,
                             text=True, timeout=60).stdout.strip()
        return out.split()[-1]
    except Exception:
        return "unknown"


if __name__ == "__main__":
    sys.exit(main())
