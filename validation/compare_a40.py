"""
A40 / Llama-3.1-8B  —  LLMServingSim vs vLLM 비교 (TP-parametric).

Usage: python3 validation/compare_a40.py [--tp N]   (default 1)

입력 (TP=N):
  validation/vllm_tp{N}_results.jsonl     (send_requests.py 출력)
  validation/sim_a40_tp{N}_results.csv    (main.py --output)
  validation/sim_a40_tp{N}_stdout.txt     (throughput + 전력 분해 파싱)
  validation/vllm_a40_tp{N}_power.csv     (nvidia-smi GPU 전력 로그, N개 GPU)

전력 비교는 모든 GPU/NPU 합산(시스템 전체 GPU 전력) 기준으로 공정 비교한다.
출력: 콘솔 표 + validation/compare_a40_tp{N}_summary.csv
"""
import argparse, csv, json, re
from collections import defaultdict
from pathlib import Path
import numpy as np

VAL = Path(__file__).parent
NS_MS = 1e-6


def pct(a, p): return float(np.percentile(a, p))
def mape(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = a > 0
    return float(np.mean(np.abs((a[m] - b[m]) / a[m])) * 100)


def load_vllm(tp):
    rows = [json.loads(l) for l in (VAL / f"vllm_a40_tp{tp}_results.jsonl").read_text().splitlines() if l.strip()]
    rows = [r for r in rows if "error" not in r and r.get("ttft_ns") and r.get("tpot_ns")]
    rows.sort(key=lambda r: r["req_idx"])
    return rows


def load_sim(tp):
    rows = list(csv.DictReader(open(VAL / f"sim_a40_tp{tp}_results.csv")))
    rows.sort(key=lambda r: int(r["request id"]))
    return rows


def parse_sim_stdout(tp):
    t = (VAL / f"sim_a40_tp{tp}_stdout.txt").read_text()
    g = lambda pat: (float(re.search(pat, t).group(1)) if re.search(pat, t) else None)
    return {
        "gen_tput": g(r"Average generation throughput \(tok/s\):\s+([\d.]+)"),
        "total_tput": g(r"Total token throughput \(tok/s\):\s+([\d.]+)"),
        "req_tput": g(r"Request throughput \(req/s\):\s+([\d.]+)"),
        "npu_j": g(r"NPU energy consumption \(J\):\s+([\d.]+)"),
        "node_kj": g(r"Node 0 total energy consumption \(kJ\):\s+([\d.]+)"),
    }


def load_vllm_power(tp):
    """Total-GPU power summed across all GPUs per sample cycle.
    nvidia-smi -i 0,1 logs each GPU with a slightly different timestamp, so we group by
    GPU index (preserving order) and sum element-wise across GPUs."""
    p = VAL / f"vllm_a40_tp{tp}_power.csv"
    if not p.exists():
        return None
    by_gpu = defaultdict(list)             # gpu index -> ordered power.draw series
    for line in p.read_text().splitlines():
        f = [x.strip() for x in line.split(",")]
        # timestamp,index,name,util,power.draw,temp
        if len(f) >= 6:
            try:
                by_gpu[int(f[1])].append(float(f[4]))
            except ValueError:
                pass
    if not by_gpu:
        return None
    gpus = sorted(by_gpu)
    m = min(len(by_gpu[g]) for g in gpus)
    tot = np.array([sum(by_gpu[g][i] for g in gpus) for i in range(m)])  # per-cycle total
    return {"mean_w": float(tot.mean()), "p95_w": pct(tot, 95), "max_w": float(tot.max()),
            "min_w": float(tot.min()), "n": len(tot), "ngpu": len(gpus)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tp", type=int, default=1)
    tp = ap.parse_args().tp

    v = load_vllm(tp); s = load_sim(tp); meta = parse_sim_stdout(tp); vp = load_vllm_power(tp)
    n = min(len(v), len(s))
    print(f"\n{'='*68}\n  A40 / Llama-3.1-8B  —  Sim vs vLLM  (TP={tp}, {n} matched reqs)\n{'='*68}")

    v_ttft = np.array([r["ttft_ns"] for r in v[:n]]) * NS_MS
    v_tpot = np.array([r["tpot_ns"] for r in v[:n]]) * NS_MS
    s_ttft = np.array([float(r["TTFT"]) for r in s[:n]]) * NS_MS
    s_tpot = np.array([float(r["TPOT"]) for r in s[:n]]) * NS_MS

    def block(name, vv, ss):
        print(f"\n{name} (ms)        {'vLLM':>10} {'Sim':>10} {'Δ%':>8}")
        for lbl, p in [("p50", 50), ("p99", 99), ("mean", None)]:
            a = vv.mean() if p is None else pct(vv, p)
            b = ss.mean() if p is None else pct(ss, p)
            print(f"  {lbl:<14} {a:>10.1f} {b:>10.1f} {(b-a)/a*100:>+7.1f}%")
        print(f"  MAPE={mape(vv,ss):.1f}%  Pearson r={np.corrcoef(vv,ss)[0,1]:.3f}")

    block("TTFT", v_ttft, s_ttft)
    print("   note: sim TTFT = compute-complete time (definitionally < vLLM client-receive TTFT)")
    block("TPOT", v_tpot, s_tpot)

    v_out = sum(r.get("actual_output_toks", 0) for r in v[:n])
    v_in = sum(int(r["input_toks"]) for r in v[:n])
    v_wall = max((r["arrival_time_ns"] + r["total_latency_ns"]) for r in v[:n]) / 1e9
    v_tput = v_out / v_wall
    v_total_tput = (v_out + v_in) / v_wall
    print(f"\nThroughput (tok/s)  {'vLLM':>10} {'Sim':>10} {'Δ%':>8}")
    print(f"  generation     {v_tput:>10.1f} {meta['gen_tput']:>10.1f} {(meta['gen_tput']-v_tput)/v_tput*100:>+7.1f}%")
    print(f"  total(pp+gen)  {v_total_tput:>10.1f} {meta['total_tput']:>10.1f} {(meta['total_tput']-v_total_tput)/v_total_tput*100:>+7.1f}%")
    print(f"  (vLLM out={v_out} toks / {v_wall:.1f}s wall;  sim req/s={meta['req_tput']} vs vLLM {n/v_wall:.2f})")

    print(f"\nGPU Power (W, all {tp} GPU(s) summed)  {'vLLM':>10} {'Sim':>10} {'Δ%':>8}")
    if vp:
        sim_end_s = max(int(r["end_time"]) for r in s) / 1e9
        sim_npu_w = meta["npu_j"] / sim_end_s            # total NPU power over sim makespan
        sim_node_w = meta["node_kj"] * 1000 / sim_end_s
        print(f"  GPU/NPU mean   {vp['mean_w']:>10.1f} {sim_npu_w:>10.1f} {(sim_npu_w-vp['mean_w'])/vp['mean_w']*100:>+7.1f}%")
        print(f"  GPU p95 / max  {vp['p95_w']:>10.1f} {vp['max_w']:>10.1f}  (measured)")
        print(f"  sim NODE total power = {sim_node_w:.1f} W (whole-system; not GPU-comparable)")
        print(f"  (vLLM nvidia-smi: {vp['n']} samples)")
    else:
        sim_npu_w = None
        print("  [vLLM power log missing]")

    out = VAL / f"compare_a40_tp{tp}_summary.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "vllm", "sim", "delta_pct"])
        for lbl, vv, ss in [("ttft_p50_ms", v_ttft, s_ttft), ("ttft_p99_ms", v_ttft, s_ttft),
                            ("tpot_p50_ms", v_tpot, s_tpot), ("tpot_p99_ms", v_tpot, s_tpot)]:
            q = 50 if "p50" in lbl else 99
            a, b = pct(vv, q), pct(ss, q)
            w.writerow([lbl, round(a, 2), round(b, 2), round((b-a)/a*100, 1)])
        w.writerow(["throughput_gen_tok_s", round(v_tput, 1), meta["gen_tput"], round((meta["gen_tput"]-v_tput)/v_tput*100, 1)])
        if vp:
            w.writerow(["gpu_power_w_total", round(vp["mean_w"], 1), round(sim_npu_w, 1), round((sim_npu_w-vp["mean_w"])/vp["mean_w"]*100, 1)])
    print(f"\nSaved: {out}\n")


if __name__ == "__main__":
    main()
