#!/usr/bin/env python3
"""Sweep node-tier IB collective-overhead constants for 70B TP16 to fit the real
vLLM ground truth (gen 67 tok/s, TPOT p50 890ms). Runs inside the servingsim container.
Base config: cluster_config/a40_16gpu_tp16_70b_4tier_cohd.json (keeps 4-tier + socket tier;
only node_floor_ns / node_per_token_ns are varied)."""
import json, subprocess, csv, re, os, sys

BASE = "cluster_config/a40_16gpu_tp16_70b_4tier_cohd.json"
DATASET = "dataset/sharegpt_req100_rate10_llama.jsonl"
REAL_GEN, REAL_TPOT = 67.0, 890.0

# (node_floor_ns, node_per_token_ns)
POINTS = [(105000, 36000), (105000, 40000), (105000, 44000), (150000, 40000)]

base = json.load(open(BASE))

def tpot_p50(csv_path):
    rows = list(csv.DictReader(open(csv_path)))
    tp = sorted(float(x["TPOT"]) / 1e6 for x in rows)
    return tp[min(int(len(tp) * 0.5), len(tp) - 1)]

print(f"{'floor_us':>9}{'per_tok_us':>11}{'gen':>8}{'total':>8}{'TPOT50':>9}  {'gen_err':>8}{'tpot_err':>9}")
for floor, per_tok in POINTS:
    cfg = json.loads(json.dumps(base))
    cfg["collective_overhead"]["node_floor_ns"] = floor
    cfg["collective_overhead"]["node_per_token_ns"] = per_tok
    cpath = f"cluster_config/_recal70b_{floor}_{per_tok}.json"
    json.dump(cfg, open(cpath, "w"), indent=2)
    out = f"output/_recal70b_{floor}_{per_tok}.csv"
    p = subprocess.run(
        ["python", "main.py", "--cluster-config", cpath, "--fp", "16",
         "--block-size", "16", "--dataset", DATASET, "--output", out,
         "--num-req", "100", "--log-interval", "1.0"],
        capture_output=True, text=True)
    t = p.stdout
    g = lambda pat: float(re.search(pat, t).group(1)) if re.search(pat, t) else -1
    gen = g(r"generation throughput \(tok/s\):\s+([\d.]+)")
    tot = g(r"Total token throughput \(tok/s\):\s+([\d.]+)")
    tp50 = tpot_p50(out)
    gerr = (gen - REAL_GEN) / REAL_GEN * 100
    terr = (tp50 - REAL_TPOT) / REAL_TPOT * 100
    print(f"{floor/1000:>9.0f}{per_tok/1000:>11.0f}{gen:>8.0f}{tot:>8.0f}{tp50:>9.1f}  {gerr:>+7.1f}%{terr:>+8.1f}%")
    sys.stdout.flush()
