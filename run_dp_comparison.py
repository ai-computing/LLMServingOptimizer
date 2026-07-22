"""
DP Scaling Comparison: LLMServingSim vs. actual vLLM Docker measurements

Runs three LLMServingSim experiments and compares DP scaling factor against
real-world vLLM Docker benchmark results (A5000, Llama-3.1-8B-Instruct):
  Actual single GPU : 575.4 tok/s
  Actual DP=2       : 1009.0 tok/s  (1.75×, 87.7% efficiency)

Simulation modes compared:
  [A] Single GPU       — cluster_config/single_node_single_instance.json
  [B] Native DP=2      — cluster_config/single_node_multi_instance.json  (ASTRA-Sim serialized)
  [C] Partition DP=2   — Method 2: max(sim_0_time, sim_1_time)

Usage:
  python run_dp_comparison.py --num-req 100
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inference_serving.dp_partition_sim import run_dp_partition_sim

REPO_ROOT = Path(__file__).resolve().parent
FREQ      = 1_000_000_000

# ── Actual vLLM Docker measurements (A5000, Llama-3.1-8B-Instruct, 100 req) ──
ACTUAL_SINGLE_TOKS   = 575.4   # tok/s
ACTUAL_DP2_TOKS      = 1009.0  # tok/s
ACTUAL_SPEEDUP       = ACTUAL_DP2_TOKS / ACTUAL_SINGLE_TOKS


def run_main(config: str, dataset: str, num_req: int, output_csv: str,
             fp: int = 16, block_size: int = 16) -> dict:
    """Run main.py and parse stdout metrics."""
    cmd = [
        sys.executable, str(REPO_ROOT / "main.py"),
        "--cluster-config", config,
        "--dataset",        dataset,
        "--num-req",        str(num_req),
        "--output",         output_csv,
        "--fp",             str(fp),
        "--block-size",     str(block_size),
        "--log-interval",   "1.0",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO_ROOT))
    if result.returncode != 0:
        print(result.stderr[-2000:], file=sys.stderr)
        raise RuntimeError(f"main.py exited {result.returncode}")

    stdout = result.stdout

    def find(pattern, cast=float):
        m = re.search(pattern, stdout)
        return cast(m.group(1)) if m else None

    return {
        "total_latency_s":  find(r"Total latency \(s\):\s+([\d.]+)"),
        "total_gen_tokens": find(r"Total generated tokens:\s+(\d+)", int),
        "throughput_tok_s": find(r"Average generation throughput \(tok/s\):\s+([\d.]+)"),
        "ttft_mean_ms":     None,
        "tpot_mean_ms":     None,
        "output_csv":       output_csv,
    }


def _csv_latency_stats(csv_path: str) -> tuple[float | None, float | None]:
    """Return (mean_TTFT_ms, mean_TPOT_ms) from a simulation output CSV."""
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        ttft = df["TTFT"].dropna().mean() / 1e6 if "TTFT" in df.columns else None
        tpot = df["TPOT"].dropna().mean() / 1e6 if "TPOT" in df.columns else None
        return ttft, tpot
    except Exception:
        return None, None


def print_sep(title=""):
    w = 64
    if title:
        pad = (w - len(title) - 2) // 2
        print("=" * pad + f" {title} " + "=" * (w - pad - len(title) - 2))
    else:
        print("=" * w)


def main():
    parser = argparse.ArgumentParser(description="DP Scaling Comparison")
    parser.add_argument("--num-req",      type=int, default=100)
    parser.add_argument("--dataset",      default="dataset/sharegpt_req100_rate10_llama.jsonl")
    parser.add_argument("--single-config",  default="cluster_config/single_node_single_instance.json")
    parser.add_argument("--dp-config",      default="cluster_config/single_node_multi_instance.json")
    parser.add_argument("--fp",           type=int, default=16)
    parser.add_argument("--block-size",   type=int, default=16)
    parser.add_argument("--skip-native",  action="store_true",
                        help="Skip native DP run (slow due to ASTRA-Sim serialization)")
    args = parser.parse_args()

    extra_kw = dict(fp=args.fp, block_size=args.block_size)
    Path("output").mkdir(exist_ok=True)

    # ── [A] Single GPU ────────────────────────────────────────────────────────
    print_sep("A  Single GPU (sim)")
    print(f"  config : {args.single_config}")
    res_single = run_main(
        config=args.single_config,
        dataset=args.dataset,
        num_req=args.num_req,
        output_csv="output/dp_compare_single.csv",
        **extra_kw,
    )
    res_single["ttft_mean_ms"], res_single["tpot_mean_ms"] = \
        _csv_latency_stats(res_single["output_csv"])
    print(f"  Throughput  : {res_single['throughput_tok_s']:.1f} tok/s")
    print(f"  TTFT mean   : {res_single['ttft_mean_ms']:.1f} ms" if res_single['ttft_mean_ms'] else "  TTFT        : N/A")
    print(f"  TPOT mean   : {res_single['tpot_mean_ms']:.2f} ms" if res_single['tpot_mean_ms'] else "  TPOT        : N/A")

    # ── [B] Native DP=2 ───────────────────────────────────────────────────────
    res_native = None
    if not args.skip_native:
        print_sep("B  Native DP=2 (sim, ASTRA-Sim serialized)")
        print(f"  config : {args.dp_config}")
        res_native = run_main(
            config=args.dp_config,
            dataset=args.dataset,
            num_req=args.num_req,
            output_csv="output/dp_compare_native.csv",
            **extra_kw,
        )
        res_native["ttft_mean_ms"], res_native["tpot_mean_ms"] = \
            _csv_latency_stats(res_native["output_csv"])
        print(f"  Throughput  : {res_native['throughput_tok_s']:.1f} tok/s")
        print(f"  TTFT mean   : {res_native['ttft_mean_ms']:.1f} ms" if res_native['ttft_mean_ms'] else "  TTFT        : N/A")
        print(f"  TPOT mean   : {res_native['tpot_mean_ms']:.2f} ms" if res_native['tpot_mean_ms'] else "  TPOT        : N/A")
    else:
        print_sep("B  Native DP=2")
        print("  (skipped)")

    # ── [C] Partition DP=2 (Method 2) ─────────────────────────────────────────
    print_sep("C  Partition DP=2 (Method 2)")
    res_partition = run_dp_partition_sim(
        dp_config_path=args.dp_config,
        dp_count=2,
        dataset_path=args.dataset,
        num_req=args.num_req,
        output_csv="output/dp_compare_partition.csv",
        extra_args=[
            "--fp",           str(args.fp),
            "--block-size",   str(args.block_size),
            "--log-interval", "1.0",
        ],
    )

    # ── Summary table ─────────────────────────────────────────────────────────
    print_sep("Comparison Summary")

    single_sim = res_single["throughput_tok_s"] or 0.0

    rows = [
        ("Actual  Single GPU (vLLM A5000)",  ACTUAL_SINGLE_TOKS,  1.0,   "—"),
        ("Actual  DP=2      (vLLM A5000)",   ACTUAL_DP2_TOKS,     ACTUAL_SPEEDUP, f"{ACTUAL_SPEEDUP/2*100:.1f}%"),
    ]
    if res_native:
        native_tps = res_native["throughput_tok_s"] or 0.0
        speedup_n  = native_tps / single_sim if single_sim else 0.0
        rows.append(("Sim [B]  Native DP=2 (A6000)",   native_tps,  speedup_n,  f"{speedup_n/2*100:.1f}%"))
    part_tps    = res_partition["throughput_tok_s"]
    speedup_p   = part_tps / single_sim if single_sim else 0.0
    rows.append(("Sim [C]  Partition DP=2 (A6000)", part_tps,  speedup_p,  f"{speedup_p/2*100:.1f}%"))

    col_w = 42
    print(f"  {'Scenario':<{col_w}} {'Throughput':>12}  {'Speedup':>8}  {'Efficiency':>10}")
    print(f"  {'-'*col_w}  {'-'*12}  {'-'*8}  {'-'*10}")
    for label, tps, sp, eff in rows:
        print(f"  {label:<{col_w}} {tps:>11.1f}  {sp:>7.2f}×  {eff:>10}")

    # ── Latency comparison ────────────────────────────────────────────────────
    print()
    print(f"  {'Scenario':<{col_w}} {'TTFT mean':>10}  {'TPOT mean':>10}")
    print(f"  {'-'*col_w}  {'-'*10}  {'-'*10}")
    print(f"  {'Actual Single (vLLM A5000)':<{col_w}} {'4460.8 ms':>10}  {'76.1 ms':>10}")
    print(f"  {'Actual DP=2   (vLLM A5000)':<{col_w}} {'2367.0 ms':>10}  {'60.2 ms':>10}")
    def fmt(v): return f"{v:.1f} ms" if v is not None else "N/A"
    print(f"  {'Sim [A] Single (A6000)':<{col_w}} {fmt(res_single['ttft_mean_ms']):>10}  {fmt(res_single['tpot_mean_ms']):>10}")
    if res_native:
        print(f"  {'Sim [B] Native DP=2 (A6000)':<{col_w}} {fmt(res_native['ttft_mean_ms']):>10}  {fmt(res_native['tpot_mean_ms']):>10}")
    part_ttft = res_partition["ttft_ms"].get("mean") if res_partition["ttft_ms"] else None
    part_tpot = res_partition["tpot_ms"].get("mean") if res_partition["tpot_ms"] else None
    print(f"  {'Sim [C] Partition DP=2 (A6000)':<{col_w}} {fmt(part_ttft):>10}  {fmt(part_tpot):>10}")

    print_sep()
    print()
    print("  Notes:")
    print("  * Absolute throughput differs (A6000 sim vs A5000 actual) — compare SPEEDUP.")
    print("  * Native DP=2 speedup ≈ 1.0 because ASTRA-Sim processes instances serially.")
    print("  * Method 2 speedup approaches actual because each partition uses N/2 requests.")
    print("  * Actual speedup: 1.75× (A5000 vLLM), A100: 1.36×")


if __name__ == "__main__":
    main()
