#!/usr/bin/env python3
"""Parse sweep logs + CSVs and write A6000_4_REPORT.md."""

import os
import re
import csv
import json
from pathlib import Path

REPO = Path(__file__).parent.parent
LOG_DIR = REPO / "output/a6000_4_sweep/full"
REPORT = REPO / "A6000_4_REPORT.md"

CONFIGS = [
    # (label, PP=npu_group, TP=npus_per_group, npu_num, DP=num_instances, pd_layout, phys_npus)
    ("01_tp1_pp1_dp1",   1, 1, 1, 1,  "—",        1),
    ("02_tp2_pp1_dp1",   1, 2, 2, 1,  "—",        2),
    ("03_tp1_pp2_dp1",   2, 1, 2, 1,  "—",        2),
    ("04_tp2_pp2_dp1",   2, 2, 4, 1,  "—",        4),
    ("05_tp1_pp4_dp1",   4, 1, 4, 1,  "—",        4),
    ("06_tp1_pp1_dp2",   1, 1, 1, 2,  "—",        2),
    ("07_tp2_pp1_dp2",   1, 2, 2, 2,  "—",        4),
    ("08_tp1_pp2_dp2",   2, 1, 2, 2,  "—",        4),
    ("09_tp1_pp1_dp4",   1, 1, 1, 4,  "—",        4),
    ("10_pd_1p1d_tp1",   1, 1, 1, 1, "1P+1D",    3),
    ("11_pd_1p2d_tp1",   1, 1, 1, 2, "1P+2D",    4),
    ("12_pd_1p1d_tp2d",  1, 2, 2, 1, "1P+1D(T2)", 4),
    ("13_pd_1p1d_pp2d",  2, 1, 2, 1, "1P+1D(P2)", 4),
]


def parse_log(log_path: Path) -> dict:
    """Extract key metrics from a simulation stdout log."""
    metrics = {}
    if not log_path.exists():
        return metrics
    text = log_path.read_text()

    patterns = {
        "total_latency_s":    r"Total latency \(s\):\s+([\d.]+)",
        "req_throughput":     r"Request throughput \(req/s\):\s+([\d.]+)",
        "prompt_throughput":  r"Average prompt throughput \(tok/s\):\s+([\d.]+)",
        "gen_throughput":     r"Average generation throughput \(tok/s\):\s+([\d.]+)",
        "total_token_tp":     r"Total token throughput \(tok/s\):\s+([\d.]+)",
        "total_requests":     r"Total requests:\s+(\d+)",
    }
    for key, pat in patterns.items():
        m = re.search(pat, text)
        if m:
            metrics[key] = float(m.group(1))

    # NPU memory util — look for last reported utilization across all instances
    npu_utils = re.findall(r"NPU Memory Usage [\d.]+ MB \(([\d.]+) % Used\)", text)
    if npu_utils:
        metrics["npu_util_pct"] = float(npu_utils[-1])

    return metrics


def parse_csv(csv_path: Path) -> dict:
    """Compute average TTFT, TPOT, ITL from per-request CSV."""
    stats = {}
    if not csv_path.exists():
        return stats
    ttfts, tpots, itls = [], [], []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ttfts.append(float(row["TTFT"]))
                tpots.append(float(row["TPOT"]))
                # ITL is a JSON list of inter-token latencies
                itl_list = json.loads(row["ITL"])
                if itl_list:
                    itls.extend(itl_list)
            except (KeyError, ValueError, json.JSONDecodeError):
                pass
    ns_to_ms = 1e-6
    if ttfts:
        stats["ttft_avg_ms"] = sum(ttfts) / len(ttfts) * ns_to_ms
        stats["ttft_p99_ms"] = sorted(ttfts)[int(len(ttfts) * 0.99)] * ns_to_ms
    if tpots:
        stats["tpot_avg_ms"] = sum(tpots) / len(tpots) * ns_to_ms
    if itls:
        stats["itl_avg_ms"] = sum(itls) / len(itls) * ns_to_ms
    return stats


def fmt(val, decimals=2, suffix=""):
    if val is None:
        return "—"
    return f"{val:.{decimals}f}{suffix}"


def main():
    rows = []
    for cfg in CONFIGS:
        label, pp, tp, tp_per_inst, dp, pd_layout, phys = cfg
        log_path = LOG_DIR / f"{label}.log"
        csv_path = LOG_DIR / f"{label}.csv"
        log_m = parse_log(log_path)
        csv_m = parse_csv(csv_path)
        rows.append({
            "label": label,
            "pp": pp,
            "tp": tp,
            "dp": dp,
            "pd": pd_layout,
            "phys_npus": phys,
            **log_m,
            **csv_m,
        })

    lines = []
    lines.append("# A6000 × 4 NPU — Llama-3.1-8B Parallelism Sweep Report\n")
    lines.append(f"**Model:** meta-llama/Llama-3.1-8B  ")
    lines.append(f"**Hardware:** NVIDIA A6000 (40 GB, 768 GB/s)  ")
    lines.append(f"**Dataset:** sharegpt_req100_rate10_llama.jsonl (100 requests)  ")
    lines.append(f"**Flags:** `--fp 16 --block-size 16 --num-req 100`  ")
    lines.append(f"**Simulator:** LLMServingSim v1.0.0  \n")

    lines.append("## Configuration Matrix\n")
    lines.append("| # | Config label | TP | PP | DP | P/D layout | Phys NPUs |")
    lines.append("|---|---|---|---|---|---|---|")
    for i, cfg in enumerate(CONFIGS, 1):
        label, pp, tp, _, dp, pd_layout, phys = cfg
        lines.append(f"| {i:2d} | `{label}` | {tp} | {pp} | {dp} | {pd_layout} | {phys} |")

    lines.append("\n## Results\n")
    lines.append("| # | Config | Sim latency (s) | Req throughput (req/s) | Prompt TP (tok/s) | Gen TP (tok/s) | Total TP (tok/s) | Avg TTFT (ms) | Avg TPOT (ms) | Avg ITL (ms) | NPU util (%) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for i, row in enumerate(rows, 1):
        lines.append(
            f"| {i:2d} | `{row['label']}` "
            f"| {fmt(row.get('total_latency_s'))} "
            f"| {fmt(row.get('req_throughput'))} "
            f"| {fmt(row.get('prompt_throughput'))} "
            f"| {fmt(row.get('gen_throughput'))} "
            f"| {fmt(row.get('total_token_tp'))} "
            f"| {fmt(row.get('ttft_avg_ms'))} "
            f"| {fmt(row.get('tpot_avg_ms'))} "
            f"| {fmt(row.get('itl_avg_ms'))} "
            f"| {fmt(row.get('npu_util_pct'))} |"
        )

    lines.append("\n## Per-axis Observations\n")
    lines.append("### Tensor Parallelism (TP=1 vs TP=2)\n")
    lines.append(
        "Configs 1 vs 2 (single instance, DP=1, PP=1): TP=2 splits each layer across 2 NPUs, "
        "reducing per-NPU compute time at the cost of ALLREDUCE communication after each "
        "attention and FFN block. For Llama-3.1-8B (8 attention heads per TP shard at TP=2), "
        "TP=2 is expected to reduce generation latency when the model is memory-bound but "
        "adds a synchronization overhead visible in TTFT."
    )
    lines.append("\n### Pipeline Parallelism (PP=1/2/4)\n")
    lines.append(
        "Configs 1, 3, 5 (DP=1, TP=1, PP=1/2/4): pipeline stages split the 32 Transformer "
        "layers across NPUs. Each stage runs independently and passes activations to the next. "
        "PP reduces per-stage memory but adds bubble overhead (inter-stage send/recv) that "
        "grows linearly with PP degree. At PP=4 on a single-instance Llama-3.1-8B each stage "
        "handles ~8 layers."
    )
    lines.append("\n### Data Parallelism (DP=1/2/4)\n")
    lines.append(
        "Configs 1, 6, 9 (TP=1, PP=1, DP=1/2/4): each instance serves an independent subset "
        "of requests via RR routing. DP scales throughput near-linearly because instances share "
        "no state. TTFT and TPOT per request should remain approximately constant while total "
        "system throughput multiplies with DP degree."
    )
    lines.append("\n### P/D Disaggregation\n")
    lines.append(
        "Configs 10–13 split prefill and decode into dedicated instances. The prefill instance "
        "processes prompt tokens and transmits KV cache to the decode instance. This removes "
        "head-of-line blocking between chunked-prefill and decode iterations. Note that in "
        "LLMServingSim a prefill instance occupies 2× its declared `npu_num` (one set for "
        "compute, one set for KV-cache-send), so the physical NPU budget must account for this."
    )

    lines.append("\n## Caveats\n")
    lines.append(
        "- **TP=4 excluded**: no profiled latency tables for A6000 + Llama-3.1-8B at TP=4 "
        "(`llm_profile/perf_models/A6000/meta-llama/Llama-3.1-8B/tp4/` absent). "
        "Re-run `llm_profile/profile_layers.sh` on real A6000 hardware to add TP=4 support.\n"
        "- **TTFT definition differs from vLLM**: LLMServingSim measures TTFT as the cycle "
        "when prefill computation completes, not when the client receives the first token. "
        "Reported values are therefore lower than vLLM-reported TTFT.\n"
        "- **Configs 1–3, 6, 10 use fewer than 4 physical NPUs**: included for scaling "
        "comparison. Throughput is not comparable on an NPU-count basis without normalization.\n"
        "- **PP modeling**: pipeline-parallel send/recv cost is modeled via link energy "
        "consumption (`npu_group - 1` inter-stage transfers) but bubble overhead is "
        "approximated. Real PP efficiency may differ."
    )

    REPORT.write_text("\n".join(lines) + "\n")
    print(f"Report written to {REPORT}")


if __name__ == "__main__":
    main()
