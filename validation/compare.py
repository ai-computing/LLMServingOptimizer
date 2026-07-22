"""
시뮬레이터 vs vLLM 결과 비교 분석 스크립트.

Usage:
    python3 validation/compare.py --tp 1
    python3 validation/compare.py --tp 2
    python3 validation/compare.py --tp 1 2   # 두 설정 모두

출력:
    validation/results_summary.csv   — MAPE, p50/p99 비교 테이블
    validation/ttft_cdf_tp{n}.png    — TTFT CDF (matplotlib 있을 때)
    validation/tpot_cdf_tp{n}.png    — TPOT CDF
    validation/scatter_tp{n}.png     — per-request 산포도
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats as stats

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False
    print("[warn] matplotlib not available — skipping plots, text output only")

REPO_ROOT = Path(__file__).parent.parent
VAL_DIR = REPO_ROOT / "validation"
NS_TO_MS = 1e-6


def load_vllm(tp: int) -> pd.DataFrame:
    path = VAL_DIR / f"vllm_tp{tp}_results.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"vLLM results not found: {path}")
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    # drop errored requests
    df = df[df.get("error", pd.Series([None] * len(df))).isna()].copy() if "error" in df.columns else df
    df["ttft_ms"] = df["ttft_ns"] * NS_TO_MS
    df["tpot_ms"] = df["tpot_ns"] * NS_TO_MS
    df = df.dropna(subset=["ttft_ms", "tpot_ms"])
    return df.sort_values("req_idx").reset_index(drop=True)


def load_sim(tp: int) -> pd.DataFrame:
    path = VAL_DIR / f"sim_tp{tp}_results.csv"
    if not path.exists():
        raise FileNotFoundError(f"Simulator results not found: {path}")
    df = pd.read_csv(path)
    # Columns: instance id, request id, model, input, output,
    #          arrival, end_time, latency, queuing_delay, TTFT, TPOT, ITL
    df = df.rename(columns={
        "request id": "req_idx",
        "TTFT": "ttft_ns",
        "TPOT": "tpot_ns",
    })
    df["ttft_ms"] = df["ttft_ns"] * NS_TO_MS
    df["tpot_ms"] = df["tpot_ns"] * NS_TO_MS
    return df.sort_values("req_idx").reset_index(drop=True)


def load_power_log(label: str) -> dict | None:
    """Parse nvidia-smi power CSV → {idle_w, active_w, mean_w}"""
    path = VAL_DIR / f"{label}_power.csv"
    if not path.exists():
        return None
    rows = []
    for line in path.read_text().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            try:
                rows.append(float(parts[3]))   # power.draw
            except ValueError:
                continue
    if not rows:
        return None
    arr = np.array(rows)
    return {
        "mean_w": float(np.mean(arr)),
        "p95_w": float(np.percentile(arr, 95)),
        "idle_w": float(np.percentile(arr, 5)),
    }


def parse_sim_power(tp: int) -> float | None:
    """Extract average power from simulator stdout log."""
    path = VAL_DIR / f"sim_tp{tp}_stdout.txt"
    if not path.exists():
        return None
    powers = []
    for line in path.read_text().splitlines():
        if "Avg power consumption:" in line:
            try:
                w = float(line.split(":")[-1].strip().split()[0])
                powers.append(w)
            except (ValueError, IndexError):
                continue
    return float(np.mean(powers)) if powers else None


def mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    mask = actual > 0
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def percentile(arr: np.ndarray, p: float) -> float:
    return float(np.percentile(arr, p))


def analyze_tp(tp: int) -> dict:
    print(f"\n{'='*60}")
    print(f"  TP={tp} analysis")
    print(f"{'='*60}")

    vllm_df = load_vllm(tp)
    sim_df = load_sim(tp)

    n_vllm = len(vllm_df)
    n_sim = len(sim_df)
    n = min(n_vllm, n_sim)
    print(f"vLLM requests: {n_vllm}  |  Sim requests: {n_sim}  |  Matched: {n}")

    # Align by position (both sorted by req_idx from same dataset)
    v_ttft = vllm_df["ttft_ms"].values[:n]
    v_tpot = vllm_df["tpot_ms"].values[:n]
    s_ttft = sim_df["ttft_ms"].values[:n]
    s_tpot = sim_df["tpot_ms"].values[:n]

    # TTFT note: simulator measures computation-complete time (lower than vLLM client-receive time)
    ttft_mape = mape(v_ttft, s_ttft)
    tpot_mape = mape(v_tpot, s_tpot)
    ttft_r, _ = stats.pearsonr(v_ttft, s_ttft)
    tpot_r, _ = stats.pearsonr(v_tpot, s_tpot)
    ttft_rho, _ = stats.spearmanr(v_ttft, s_ttft)
    tpot_rho, _ = stats.spearmanr(v_tpot, s_tpot)

    # Throughput: total output tokens / total wall-clock time
    v_total_out = vllm_df["actual_output_toks"].sum() if "actual_output_toks" in vllm_df.columns else 0
    v_elapsed_s = (vllm_df["arrival_time_ns"].max() + vllm_df["total_latency_ns"].max()) / 1e9 \
        if "total_latency_ns" in vllm_df.columns else None
    v_throughput = v_total_out / v_elapsed_s if v_elapsed_s else None

    # Simulator throughput from stdout
    sim_throughput = None
    stdout_path = VAL_DIR / f"sim_tp{tp}_stdout.txt"
    if stdout_path.exists():
        for line in stdout_path.read_text().splitlines():
            if "generation throughput" in line.lower():
                try:
                    val = float(line.split(":")[-1].strip().split()[0])
                    sim_throughput = val
                except (ValueError, IndexError):
                    pass

    # Power
    vllm_power = load_power_log(f"vllm_tp{tp}")
    sim_power_w = parse_sim_power(tp)

    # Print table
    print(f"\n{'Metric':<22} {'vLLM':>12} {'Sim':>12} {'MAPE%':>8} {'Pearson r':>10} {'Spearman ρ':>11}")
    print("-" * 77)

    def row(name, v_arr, s_arr, unit="ms"):
        for p_label, p_val in [("p50", 50), ("p99", 99)]:
            v_p = percentile(v_arr, p_val)
            s_p = percentile(s_arr, p_val)
            diff = (s_p - v_p) / v_p * 100
            print(f"  {name} {p_label:<15} {v_p:>11.1f} {s_p:>11.1f} {diff:>+7.1f}%", end="")
            if p_val == 50:
                print()
            else:
                print()

    print(f"\nTTFT (ms):   MAPE={ttft_mape:.1f}%  Pearson r={ttft_r:.3f}  Spearman ρ={ttft_rho:.3f}")
    print(f"  vLLM  p50={percentile(v_ttft,50):.1f}  p99={percentile(v_ttft,99):.1f}")
    print(f"  Sim   p50={percentile(s_ttft,50):.1f}  p99={percentile(s_ttft,99):.1f}")

    print(f"\nTPOT (ms):   MAPE={tpot_mape:.1f}%  Pearson r={tpot_r:.3f}  Spearman ρ={tpot_rho:.3f}")
    print(f"  vLLM  p50={percentile(v_tpot,50):.1f}  p99={percentile(v_tpot,99):.1f}")
    print(f"  Sim   p50={percentile(s_tpot,50):.1f}  p99={percentile(s_tpot,99):.1f}")

    if v_throughput and sim_throughput:
        tp_mape = abs(v_throughput - sim_throughput) / v_throughput * 100
        print(f"\nThroughput (tok/s):  vLLM={v_throughput:.1f}  Sim={sim_throughput:.1f}  err={tp_mape:.1f}%")

    if vllm_power and sim_power_w:
        pwr_mape = abs(vllm_power["mean_w"] - sim_power_w) / vllm_power["mean_w"] * 100
        print(f"\nGPU Power (W):  vLLM mean={vllm_power['mean_w']:.1f}  Sim mean={sim_power_w:.1f}  err={pwr_mape:.1f}%")

    # Pass/fail against targets
    print(f"\n--- Accuracy targets ---")
    checks = [
        ("TTFT p99 MAPE", ttft_mape, 20.0),
        ("TPOT p99 MAPE", tpot_mape, 15.0),
    ]
    for label, val, target in checks:
        status = "PASS ✓" if val <= target else "FAIL ✗"
        print(f"  {label}: {val:.1f}% (target ≤{target}%)  {status}")

    # Plots
    if HAS_PLOT:
        _plot_cdf(v_ttft, s_ttft, f"TTFT (ms)", f"ttft_cdf_tp{tp}", tp)
        _plot_cdf(v_tpot, s_tpot, f"TPOT (ms)", f"tpot_cdf_tp{tp}", tp)
        _plot_scatter(v_ttft, s_ttft, v_tpot, s_tpot, tp)

    return {
        "tp": tp,
        "n_matched": n,
        "ttft_mape": ttft_mape,
        "tpot_mape": tpot_mape,
        "ttft_pearson_r": ttft_r,
        "tpot_pearson_r": tpot_r,
        "ttft_p50_vllm": percentile(v_ttft, 50),
        "ttft_p99_vllm": percentile(v_ttft, 99),
        "ttft_p50_sim": percentile(s_ttft, 50),
        "ttft_p99_sim": percentile(s_ttft, 99),
        "tpot_p50_vllm": percentile(v_tpot, 50),
        "tpot_p99_vllm": percentile(v_tpot, 99),
        "tpot_p50_sim": percentile(s_tpot, 50),
        "tpot_p99_sim": percentile(s_tpot, 99),
        "throughput_vllm": v_throughput,
        "throughput_sim": sim_throughput,
        "power_vllm_mean_w": vllm_power["mean_w"] if vllm_power else None,
        "power_sim_mean_w": sim_power_w,
    }


def _plot_cdf(v_arr, s_arr, xlabel, fname, tp):
    fig, ax = plt.subplots(figsize=(7, 5))
    for arr, label, color in [(v_arr, "vLLM (measured)", "#1f77b4"),
                               (s_arr, "Simulator", "#ff7f0e")]:
        sorted_arr = np.sort(arr)
        cdf = np.arange(1, len(sorted_arr) + 1) / len(sorted_arr)
        ax.plot(sorted_arr, cdf, label=label, color=color, linewidth=2)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("CDF", fontsize=12)
    ax.set_title(f"{xlabel} CDF — A5000 TP={tp}", fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    out = VAL_DIR / f"{fname}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def _plot_scatter(v_ttft, s_ttft, v_tpot, s_tpot, tp):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, v_arr, s_arr, label in [
        (axes[0], v_ttft, s_ttft, "TTFT (ms)"),
        (axes[1], v_tpot, s_tpot, "TPOT (ms)"),
    ]:
        ax.scatter(v_arr, s_arr, alpha=0.4, s=15, color="#1f77b4")
        lim = max(v_arr.max(), s_arr.max()) * 1.05
        ax.plot([0, lim], [0, lim], "k--", linewidth=1, label="y=x")
        ax.set_xlabel(f"vLLM {label}", fontsize=11)
        ax.set_ylabel(f"Simulator {label}", fontsize=11)
        ax.set_title(f"Per-request {label} — TP={tp}", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
    out = VAL_DIR / f"scatter_tp{tp}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tp", type=int, nargs="+", default=[1, 2], choices=[1, 2],
                   help="Which TP configs to analyze (default: 1 2)")
    args = p.parse_args()

    all_rows = []
    for tp in args.tp:
        try:
            row = analyze_tp(tp)
            all_rows.append(row)
        except FileNotFoundError as e:
            print(f"[skip TP={tp}] {e}")

    if all_rows:
        summary_df = pd.DataFrame(all_rows)
        out = VAL_DIR / "results_summary.csv"
        summary_df.to_csv(out, index=False, float_format="%.3f")
        print(f"\nSummary saved to: {out}")


if __name__ == "__main__":
    main()
