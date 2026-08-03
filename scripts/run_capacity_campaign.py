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
import re
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
    # 'auto' honours a quantized checkpoint's own config (fp8 / int8 / int4);
    # forcing bfloat16 would fight it
    p.add_argument("--dtype", default="auto")
    p.add_argument("--kv-cache-dtype", default=None,
                   help="e.g. fp8_e5m2 (storage only on Ampere)")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--quality-samples", type=int, default=0,
                   help="run the GSM8K check with N problems (0 = skip)")
    p.add_argument("--label", default=None, help="tag for the results record")
    p.add_argument("--results-json", default=None,
                   help="write the full record (config + points + quality) here")
    args = p.parse_args()

    concurrencies = [int(c) for c in args.concurrencies.split(",")]
    tag = args.label or args.model.split("/")[-1]
    workdir = REPO_ROOT / "output" / "campaign" / \
        f"{args.hw}_tp{args.tp}_{tag}_{datetime.date.today().isoformat()}"
    workdir.mkdir(parents=True, exist_ok=True)

    import os
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = args.gpus
    env.setdefault("HF_HUB_OFFLINE", "1")   # weights are already cached

    print(f"[campaign] starting vLLM server (tp{args.tp}, GPUs {args.gpus}) ...",
          flush=True)
    server_log = open(workdir / "server.log", "w")
    serve_cmd = [str(VLLM), "serve", args.model,
                 "--dtype", args.dtype,
                 "--tensor-parallel-size", str(args.tp),
                 "--max-model-len", str(args.max_model_len),
                 "--max-num-seqs", "128",
                 "--max-num-batched-tokens", "2048",
                 "--block-size", "16",
                 "--gpu-memory-utilization", str(args.gpu_memory_utilization),
                 "--port", str(args.port)]
    if args.kv_cache_dtype:
        serve_cmd += ["--kv-cache-dtype", args.kv_cache_dtype]
    server = subprocess.Popen(serve_cmd, stdout=server_log,
                              stderr=subprocess.STDOUT, env=env)
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

        quality = None
        if args.quality_samples > 0:
            from eval_gsm8k import evaluate_gsm8k
            print(f"[campaign] GSM8K check ({args.quality_samples} problems) ...",
                  flush=True)
            quality = evaluate_gsm8k(
                base_url=f"http://localhost:{args.port}", model=args.model,
                n=args.quality_samples, concurrency=16)
            print(f"[campaign] quality: {quality}", flush=True)

        if args.results_json:
            import yaml as _yaml
            oracle = _yaml.safe_load(Path(path).read_text())
            record = {
                "label": tag, "hw": args.hw, "model": args.model,
                "tp": args.tp, "gpus": args.gpus, "dtype": args.dtype,
                "kv_cache_dtype": args.kv_cache_dtype,
                "max_model_len": args.max_model_len,
                "gpu_memory_utilization": args.gpu_memory_utilization,
                "stack": f"vllm-{_vllm_version()}",
                "measured_at": datetime.datetime.now().isoformat(timespec="seconds"),
                "idle_w": round(idle_w, 1),
                "kv_cache_tokens": _kv_cache_tokens(workdir / "server.log"),
                "points": oracle["points"],
                "quality": quality,
                "oracle_path": str(path),
            }
            Path(args.results_json).parent.mkdir(parents=True, exist_ok=True)
            Path(args.results_json).write_text(json.dumps(record, indent=2))
            print(f"[campaign] wrote record: {args.results_json}", flush=True)
        return 0
    finally:
        server.send_signal(signal.SIGINT)
        try:
            server.wait(timeout=60)
        except subprocess.TimeoutExpired:
            server.kill()
        server_log.close()
        print("[campaign] server stopped", flush=True)


def _kv_cache_tokens(server_log: Path) -> int | None:
    """vLLM logs the KV cache it could allocate — the headroom quantization buys."""
    try:
        m = re.findall(r"GPU KV cache size:\s*([\d,]+)\s*tokens",
                       server_log.read_text(errors="replace"))
        return int(m[-1].replace(",", "")) if m else None
    except OSError:
        return None


def _vllm_version() -> str:
    try:
        out = subprocess.run([str(VLLM), "--version"], capture_output=True,
                             text=True, timeout=60).stdout.strip()
        return out.split()[-1]
    except Exception:
        return "unknown"


if __name__ == "__main__":
    sys.exit(main())
