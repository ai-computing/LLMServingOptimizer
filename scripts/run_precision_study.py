#!/usr/bin/env python3
"""Drive the fp16 / fp8 / int8 / int4 study on this host's 2x RTX A5000.

Each configuration boots vLLM once and, against that one server, runs the
concurrency sweep (throughput / TTFT / TPOT + nvidia-smi power) and the GSM8K
accuracy check, then writes one JSON record. Configurations run in separate
processes and are resumable: an existing record is skipped, a failure is
logged and the study continues.

    python scripts/run_precision_study.py            # whole matrix
    python scripts/run_precision_study.py llama8b-fp16 qwen14b-int4-tp1
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "output" / "precision_study"
CAMPAIGN = REPO_ROOT / "scripts" / "run_capacity_campaign.py"

LLAMA = {
    "fp16": "meta-llama/Llama-3.1-8B-Instruct",
    "fp8": "RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8",
    "int8": "RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w8a8",
    "int4": "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
}
QWEN = {
    "fp16": "Qwen/Qwen2.5-14B-Instruct",
    "fp8": "RedHatAI/Qwen2.5-14B-Instruct-FP8-dynamic",
    "int8": "RedHatAI/Qwen2.5-14B-Instruct-quantized.w8a8",
    "int4": "Qwen/Qwen2.5-14B-Instruct-AWQ",
}
SWEEP_8B = "1,2,4,8,16,32,64"
SWEEP_14B = "1,4,16,64"

#: label -> (model, tp, gpus, concurrencies)
CONFIGS: dict[str, tuple[str, int, str, str]] = {}
for prec, mid in LLAMA.items():                       # 8B: everything fits tp1
    CONFIGS[f"llama8b-{prec}"] = (mid, 1, "0", SWEEP_8B)
for prec, mid in QWEN.items():                        # 14B: fp16 needs tp2
    CONFIGS[f"qwen14b-{prec}-tp2"] = (mid, 2, "0,1", SWEEP_14B)
for prec in ("int8", "int4"):                         # quantization removes TP
    CONFIGS[f"qwen14b-{prec}-tp1"] = (QWEN[prec], 1, "0", SWEEP_14B)

QUALITY_SAMPLES = 200
MAX_MODEL_LEN = 4096


def model_cached(model_id: str) -> bool:
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(model_id, local_files_only=True,
                          ignore_patterns=["*.pt", "original/*"])
        return True
    except Exception:
        return False


def wait_for_model(model_id: str, timeout_s: int = 7200) -> bool:
    """Downloads run in the background; give them time before giving up."""
    deadline = time.time() + timeout_s
    announced = False
    while time.time() < deadline:
        if model_cached(model_id):
            return True
        if not announced:
            print(f"[study]   waiting for download: {model_id}", flush=True)
            announced = True
        time.sleep(60)
    return False


def wait_gpus_free(timeout_s: int = 180) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True).stdout.split()
        if all(int(x) < 1024 for x in out if x.isdigit()):
            return
        time.sleep(5)
    print("[study]   warning: GPUs still busy, continuing anyway", flush=True)


def run_one(label: str) -> str:
    model, tp, gpus, sweep = CONFIGS[label]
    rec = OUT / f"{label}.json"
    if rec.is_file():
        return "skipped (already have a record)"
    if not wait_for_model(model):
        return "FAILED: model never appeared in the HF cache"
    wait_gpus_free()

    cmd = [sys.executable, str(CAMPAIGN),
           "--hw", "A5000", "--model", model, "--tp", str(tp), "--gpus", gpus,
           "--concurrencies", sweep, "--max-model-len", str(MAX_MODEL_LEN),
           "--dtype", "auto", "--label", label,
           "--quality-samples", str(QUALITY_SAMPLES),
           "--results-json", str(rec)]
    log = OUT / f"{label}.log"
    print(f"[study] === {label}: {model} tp{tp} gpus={gpus}", flush=True)
    t0 = time.time()
    with open(log, "w") as f:
        rc = subprocess.run(cmd, cwd=str(REPO_ROOT), stdout=f,
                            stderr=subprocess.STDOUT).returncode
    dt = time.time() - t0
    if rc != 0 or not rec.is_file():
        tail = "\n".join(log.read_text(errors="replace").splitlines()[-6:])
        return f"FAILED (rc={rc}, {dt/60:.1f} min)\n{tail}"
    d = json.loads(rec.read_text())
    q = (d.get("quality") or {}).get("accuracy")
    best = max(d["points"], key=lambda p: p["thr_toks_s"])
    return (f"ok ({dt/60:.1f} min) peak {best['thr_toks_s']:.0f} tok/s @c"
            f"{best['concurrency']}, gsm8k {q}, kv {d.get('kv_cache_tokens')}")


def main(argv: list[str]) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    labels = argv or list(CONFIGS)
    unknown = [l for l in labels if l not in CONFIGS]
    if unknown:
        print(f"unknown labels: {unknown}\navailable: {list(CONFIGS)}")
        return 2
    summary = {}
    for label in labels:
        try:
            summary[label] = run_one(label)
        except Exception as e:                      # never abort the matrix
            summary[label] = f"FAILED: {type(e).__name__}: {e}"
        print(f"[study] {label}: {summary[label]}", flush=True)
    print("\n[study] ===== summary =====", flush=True)
    for k, v in summary.items():
        print(f"  {k:22s} {v.splitlines()[0]}", flush=True)
    (OUT / "study_summary.json").write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
