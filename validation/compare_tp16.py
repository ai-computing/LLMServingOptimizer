#!/usr/bin/env python3
"""Compare TP=16 (cross-node, 2x8 A40 over IB) simulator predictions vs real vLLM.
Mirrors compare_tp8.py. sim variants: '<tag>_4tier' (bandwidth-only 4-tier, no overhead)
and '<tag>_cohd' (4-tier + node/IB collective-overhead). vLLM ground truth from
vllm_a40_tp16_<tag>_results.jsonl.

Usage: python3 validation/compare_tp16.py <tag>     # tag in {8b, 70b}
"""
import csv, re, json, sys

V = "validation"
TAG = sys.argv[1] if len(sys.argv) > 1 else "8b"


def simstat(name):
    r = list(csv.DictReader(open(f"{V}/sim_a40_tp16_{name}_results.csv")))
    t = open(f"{V}/sim_a40_tp16_{name}_stdout.txt").read()
    ttft = sorted(float(x["TTFT"]) / 1e6 for x in r)
    tpot = sorted(float(x["TPOT"]) / 1e6 for x in r)
    mk = (max(int(x["end_time"]) for x in r) - min(int(x["arrival"]) for x in r)) / 1e9
    g = lambda p: (re.search(p, t).group(1) if re.search(p, t) else "0")
    pf = lambda a, q: a[min(int(len(a) * q), len(a) - 1)]
    return dict(mk=mk, ttft50=pf(ttft, .5), tpot50=pf(tpot, .5),
                gen=float(g(r"generation throughput \(tok/s\):\s+([\d.]+)")),
                tot=float(g(r"Total token throughput \(tok/s\):\s+([\d.]+)")))


def vstat():
    rows = [json.loads(l) for l in open(f"{V}/vllm_a40_tp16_{TAG}_results.jsonl") if l.strip()]
    rows = [x for x in rows if "error" not in x and x.get("tpot_ns")]
    ttft = sorted(x["ttft_ns"] / 1e6 for x in rows)
    tpot = sorted(x["tpot_ns"] / 1e6 for x in rows)
    out = sum(x["actual_output_toks"] for x in rows)
    wall = max(x["arrival_time_ns"] + x["total_latency_ns"] for x in rows) / 1e9
    pf = lambda a, q: a[min(int(len(a) * q), len(a) - 1)]
    return dict(ttft50=pf(ttft, .5), tpot50=pf(tpot, .5), gen=out / wall, n=len(rows))


bw = simstat(f"{TAG}_4tier")
co = simstat(f"{TAG}_cohd")
v = vstat()
print("=" * 74)
print(f"  A40 {TAG.upper()} TP=16 cross-node (2x8 A40 over IB 200Gb/s), ShareGPT, vLLM n={v['n']}")
print("=" * 74)
print(f"{'metric':18}{'sim(bw-only)':>13}{'sim(+IB cohd)':>14}{'vLLM(real)':>12}")
print(f"{'gen tput(tok/s)':18}{bw['gen']:>13.0f}{co['gen']:>14.0f}{v['gen']:>12.0f}")
print(f"{'total tput(tok/s)':18}{bw['tot']:>13.0f}{co['tot']:>14.0f}{'-':>12}")
print(f"{'TTFT p50(ms)':18}{bw['ttft50']:>13.0f}{co['ttft50']:>14.0f}{v['ttft50']:>12.0f}")
print(f"{'TPOT p50(ms)':18}{bw['tpot50']:>13.1f}{co['tpot50']:>14.1f}{v['tpot50']:>12.1f}")
print(f"{'makespan(s)':18}{bw['mk']:>13.1f}{co['mk']:>14.1f}{'-':>12}")
print()
print(f"sim(bw-only) vs vLLM: gen {(bw['gen']-v['gen'])/v['gen']*100:+.1f}%, "
      f"TPOT {(bw['tpot50']-v['tpot50'])/v['tpot50']*100:+.1f}%")
print(f"sim(+IB cohd) vs vLLM: gen {(co['gen']-v['gen'])/v['gen']*100:+.1f}%, "
      f"TPOT {(co['tpot50']-v['tpot50'])/v['tpot50']*100:+.1f}%")
