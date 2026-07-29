#!/usr/bin/env python3
"""Three-way backend comparison report (plan M7, P0.5 DoD).

Collects per-request results for the SAME (cluster, workload) from up to three
sources and emits a markdown diff table against the real-vLLM baseline:

* real vLLM record  — ``--vllm-jsonl`` (validation/ format: ttft_ns/tpot_ns per req)
* simulator backend — ``--sim-csv`` (backend result CSV) or live via planner
* measured backend  — ``--measured-config`` + ``--dataset`` (runs in-process, seconds)

Usage (fixtures + live measured run):
    python scripts/compare_backends.py \
        --vllm-jsonl validation/vllm_a40_tp1_results.jsonl \
        --sim-csv validation/sim_a40_tp1_nocache_results.csv --sim-label upstream \
        --measured-config cluster_config/measured/a40_tp1.json \
        --dataset dataset/sharegpt_req100_rate10_llama.jsonl \
        --num-reqs 100 --out compare_report.md
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

NS_PER_MS = 1e6
NS_PER_S = 1e9

METRICS = ["ttft_mean_ms", "ttft_p99_ms", "tpot_mean_ms", "tpot_p99_ms",
           "latency_mean_s", "throughput_toks_s"]


def _p(vals, q):
    if not vals:
        return float("nan")
    s = sorted(vals)
    return s[min(len(s) - 1, int(q * (len(s) - 1)))]


def aggregate(records: list[dict]) -> dict:
    """records: [{ttft_ns, tpot_ns, latency_ns, output, arrival_ns, end_time_ns}]"""
    if not records:
        raise ValueError("no records to aggregate")
    ttft = [r["ttft_ns"] / NS_PER_MS for r in records]
    tpot = [r["tpot_ns"] / NS_PER_MS for r in records if r["tpot_ns"] > 0]
    lat = [r["latency_ns"] / NS_PER_S for r in records]
    span = max(r["end_time_ns"] for r in records) - min(r["arrival_ns"] for r in records)
    total_out = sum(r["output"] for r in records)
    return {
        "ttft_mean_ms": sum(ttft) / len(ttft),
        "ttft_p99_ms": _p(ttft, 0.99),
        "tpot_mean_ms": sum(tpot) / len(tpot) if tpot else float("nan"),
        "tpot_p99_ms": _p(tpot, 0.99),
        "latency_mean_s": sum(lat) / len(lat),
        "throughput_toks_s": total_out / (span / NS_PER_S) if span > 0 else 0.0,
        "num_requests": len(records),
    }


def load_vllm_jsonl(path: str | Path) -> list[dict]:
    recs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            out_toks = int(d.get("actual_output_toks") or d["output_toks"])
            recs.append({
                "ttft_ns": int(d["ttft_ns"]),
                "tpot_ns": int(d["tpot_ns"]),
                "latency_ns": int(d["total_latency_ns"]),
                "output": out_toks,
                "arrival_ns": int(d["arrival_time_ns"]),
                "end_time_ns": int(d["arrival_time_ns"]) + int(d["total_latency_ns"]),
            })
    return recs


def load_sim_csv(path: str | Path, backend: str) -> list[dict]:
    from sim_backends import get_backend
    return get_backend(backend).parse_csv(str(path))


def run_measured(cluster_config: str, dataset: str, num_reqs: int,
                 out_csv: str | Path) -> list[dict]:
    from sim_backends import ScenarioSpec, get_backend
    b = get_backend("measured")
    proc = b.run(cluster_config, str(out_csv),
                 ScenarioSpec(dataset=dataset, num_reqs=num_reqs))
    if proc.returncode != 0:
        raise RuntimeError(f"measured backend failed: {proc.stderr}")
    return b.parse_csv(str(out_csv))


def diff_table(results: dict[str, dict], baseline: str = "vllm") -> str:
    """Markdown table: metric rows, per-source value + diff% vs baseline."""
    if baseline not in results:
        raise ValueError(f"baseline '{baseline}' not among {sorted(results)}")
    others = [k for k in results if k != baseline]
    lines = ["| metric | " + baseline + " | " +
             " | ".join(f"{o} | diff% " for o in others) + "|"]
    lines.append("|" + "---|" * (2 + 2 * len(others)))
    base = results[baseline]
    for m in METRICS:
        row = [m, f"{base.get(m, float('nan')):.2f}"]
        for o in others:
            v = results[o].get(m, float("nan"))
            b = base.get(m, float("nan"))
            diff = (v - b) / b * 100 if b else float("nan")
            row += [f"{v:.2f}", f"{diff:+.1f}%"]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def error_summary(results: dict[str, dict], baseline: str = "vllm") -> dict[str, float]:
    """Mean |diff%| across METRICS per source (for the P0.5 DoD check)."""
    base = results[baseline]
    out = {}
    for k, v in results.items():
        if k == baseline:
            continue
        errs = [abs((v[m] - base[m]) / base[m]) * 100
                for m in METRICS if base.get(m)]
        out[k] = sum(errs) / len(errs)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--vllm-jsonl", help="real vLLM per-request jsonl (baseline)")
    p.add_argument("--sim-csv", help="simulator result CSV (fixture)")
    p.add_argument("--sim-label", default="upstream",
                   choices=["upstream", "legacy"])
    p.add_argument("--measured-config", help="measured cluster config JSON")
    p.add_argument("--dataset", help="workload jsonl (for the measured run)")
    p.add_argument("--num-reqs", type=int, default=0)
    p.add_argument("--out", default="compare_backends_report.md")
    args = p.parse_args(argv)

    results: dict[str, dict] = {}
    if args.vllm_jsonl:
        results["vllm"] = aggregate(load_vllm_jsonl(args.vllm_jsonl))
    if args.sim_csv:
        results[args.sim_label] = aggregate(load_sim_csv(args.sim_csv, args.sim_label))
    if args.measured_config:
        if not args.dataset:
            p.error("--measured-config requires --dataset")
        out_csv = Path(args.out).with_suffix(".measured.csv")
        results["measured"] = aggregate(run_measured(
            args.measured_config, args.dataset, args.num_reqs, out_csv))
    if len(results) < 2:
        p.error("need at least two sources to compare")

    baseline = "vllm" if "vllm" in results else sorted(results)[0]
    md = ["# Backend comparison report", "",
          f"baseline: **{baseline}**", "",
          diff_table(results, baseline=baseline), ""]
    if baseline == "vllm":
        md.append("## Mean |diff%| vs real vLLM")
        for k, v in sorted(error_summary(results).items(), key=lambda kv: kv[1]):
            md.append(f"- {k}: **{v:.1f}%**")
        md.append("")
    Path(args.out).write_text("\n".join(md), encoding="utf-8")
    print(f"wrote {args.out}")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
