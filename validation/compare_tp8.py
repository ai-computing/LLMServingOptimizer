import csv, re, json, collections
import numpy as np

V = "validation"

def simstat(name):
    r = list(csv.DictReader(open(f"{V}/sim_a40_tp8_{name}_results.csv")))
    t = open(f"{V}/sim_a40_tp8_{name}_stdout.txt").read()
    ttft = sorted(float(x["TTFT"]) / 1e6 for x in r)
    tpot = sorted(float(x["TPOT"]) / 1e6 for x in r)
    mk = (max(int(x["end_time"]) for x in r) - min(int(x["arrival"]) for x in r)) / 1e9
    g = lambda p: (re.search(p, t).group(1) if re.search(p, t) else "0")
    pf = lambda a, q: a[min(int(len(a) * q), len(a) - 1)]
    return dict(mk=mk, ttft50=pf(ttft, .5), tpot50=pf(tpot, .5),
                gen=float(g(r"generation throughput \(tok/s\):\s+([\d.]+)")),
                npu=float(g(r"NPU energy consumption \(J\):\s+([\d.]+)")))

def vstat():
    rows = [json.loads(l) for l in open(f"{V}/vllm_a40_tp8_results.jsonl") if l.strip()]
    rows = [x for x in rows if "error" not in x and x.get("tpot_ns")]
    ttft = sorted(x["ttft_ns"] / 1e6 for x in rows)
    tpot = sorted(x["tpot_ns"] / 1e6 for x in rows)
    out = sum(x["actual_output_toks"] for x in rows)
    wall = max(x["arrival_time_ns"] + x["total_latency_ns"] for x in rows) / 1e9
    pf = lambda a, q: a[min(int(len(a) * q), len(a) - 1)]
    return dict(ttft50=pf(ttft, .5), tpot50=pf(tpot, .5), gen=out / wall)

byg = collections.defaultdict(list)
for line in open(f"{V}/vllm_a40_tp8_power.csv"):
    f = [x.strip() for x in line.split(",")]
    if len(f) >= 6:
        try:
            byg[int(f[1])].append(float(f[4]))
        except ValueError:
            pass
m = min(len(v) for v in byg.values())
vpow = float(np.mean([sum(byg[g][i] for g in byg) for i in range(m)]))

meas, extr, v = simstat("measured"), simstat("extrap"), vstat()
print("=" * 72)
print("  A40 8B TP=8 extrapolation validity (300 req, 8x A40, link_bw=21 GB/s)")
print("=" * 72)
print(f"{'metric':17}{'sim(extrap)':>13}{'sim(measured)':>14}{'vLLM(real)':>12}")
print(f"{'makespan(s)':17}{extr['mk']:>13.1f}{meas['mk']:>14.1f}{'-':>12}")
print(f"{'gen tput(tok/s)':17}{extr['gen']:>13.0f}{meas['gen']:>14.0f}{v['gen']:>12.0f}")
print(f"{'TTFT p50(ms)':17}{extr['ttft50']:>13.0f}{meas['ttft50']:>14.0f}{v['ttft50']:>12.0f}")
print(f"{'TPOT p50(ms)':17}{extr['tpot50']:>13.1f}{meas['tpot50']:>14.1f}{v['tpot50']:>12.1f}")
print(f"{'GPU pow(W,8gpu)':17}{extr['npu']/extr['mk']:>13.0f}{meas['npu']/meas['mk']:>14.0f}{vpow:>12.0f}")
print()
print(f"extrap vs measured (sim end-to-end): gen {(extr['gen']-meas['gen'])/meas['gen']*100:+.1f}%, "
      f"TPOT {(extr['tpot50']-meas['tpot50'])/meas['tpot50']*100:+.1f}%, makespan {(extr['mk']-meas['mk'])/meas['mk']*100:+.1f}%")
print(f"sim(measured) vs vLLM: gen {(meas['gen']-v['gen'])/v['gen']*100:+.1f}%, "
      f"TPOT {(meas['tpot50']-v['tpot50'])/v['tpot50']*100:+.1f}%")
print(f"sim(extrap)   vs vLLM: gen {(extr['gen']-v['gen'])/v['gen']*100:+.1f}%, "
      f"TPOT {(extr['tpot50']-v['tpot50'])/v['tpot50']*100:+.1f}%")
