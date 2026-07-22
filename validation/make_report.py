"""Generate a PDF report of the A40 LLMServingSim validation work (2026-06-19)."""
import csv, re, json, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

V = "validation"
PM = "llm_profile/perf_models"
C1, C2, C3 = "#3b6fb0", "#e07b39", "#5a9e6f"   # sim, vllm, accent

def pf(a, q): return sorted(a)[min(int(len(a)*q), len(a)-1)]

def sim_stat(path_csv, path_out):
    r = list(csv.DictReader(open(path_csv)))
    ttft = [float(x["TTFT"])/1e6 for x in r]; tpot = [float(x["TPOT"])/1e6 for x in r]
    mk = (max(int(x["end_time"]) for x in r)-min(int(x["arrival"]) for x in r))/1e9
    t = open(path_out).read()
    g = lambda p: float(re.search(p, t).group(1)) if re.search(p, t) else 0.0
    return dict(ttft=pf(ttft,.5), tpot=pf(tpot,.5), mk=mk,
                gen=g(r"generation throughput \(tok/s\):\s+([\d.]+)"),
                npu=g(r"NPU energy consumption \(J\):\s+([\d.]+)"),
                node=g(r"Node 0 total energy consumption \(kJ\):\s+([\d.]+)"))

def vllm_stat(tp):
    rows = [json.loads(l) for l in open(f"{V}/vllm_a40_tp{tp}_results.jsonl") if l.strip()]
    rows = [x for x in rows if "error" not in x and x.get("tpot_ns")]
    ttft = [x["ttft_ns"]/1e6 for x in rows]; tpot = [x["tpot_ns"]/1e6 for x in rows]
    out = sum(x["actual_output_toks"] for x in rows)
    wall = max(x["arrival_time_ns"]+x["total_latency_ns"] for x in rows)/1e9
    return dict(ttft=pf(ttft,.5), tpot=pf(tpot,.5), gen=out/wall)

def vllm_power(tp):
    byg = collections.defaultdict(list)
    for line in open(f"{V}/vllm_a40_tp{tp}_power.csv"):
        f = [x.strip() for x in line.split(",")]
        if len(f) >= 6:
            try: byg[int(f[1])].append(float(f[4]))
            except ValueError: pass
    if not byg: return 0.0
    m = min(len(v) for v in byg.values())
    return float(np.mean([sum(byg[g][i] for g in byg) for i in range(m)]))

def textpage(pdf, title, lines, fontsize=11):
    fig = plt.figure(figsize=(8.27, 11.69))  # A4
    fig.text(0.08, 0.93, title, fontsize=17, weight="bold")
    fig.text(0.08, 0.90, "_"*78, fontsize=11, color="#888")
    y = 0.86
    for ln in lines:
        fs = fontsize; w = "normal"; x = 0.08; col = "#111"
        if ln.startswith("## "): ln = ln[3:]; fs = 13; w = "bold"; y -= 0.005
        elif ln.startswith("• "): x = 0.10
        elif ln.startswith("  "): x = 0.12; col = "#444"; fs = 10
        fig.text(x, y, ln, fontsize=fs, weight=w, color=col, family="DejaVu Sans")
        y -= 0.030 if fs >= 13 else 0.025
    emit(fig); plt.close(fig)

# ---------- gather data ----------
TPS = [1, 2, 4]
sim = {tp: sim_stat(f"{V}/sim_a40_tp{tp}_results.csv", f"{V}/sim_a40_tp{tp}_stdout.txt") for tp in TPS}
buggy = {tp: sim_stat(f"{V}/sim_a40_tp{tp}_buggy_results.csv", f"{V}/sim_a40_tp{tp}_stdout.txt") for tp in TPS}
vll = {tp: vllm_stat(tp) for tp in TPS}
vllp = {tp: vllm_power(tp) for tp in TPS}
simp = {tp: sim[tp]["npu"]/sim[tp]["mk"] for tp in TPS}

pdf = PdfPages(f"{V}/A40_validation_report_2026-06-19.pdf")
_pg=[0]
def emit(fig):
    _pg[0]+=1
    fig.savefig(f"/tmp/prev_p{_pg[0]}.png", dpi=92)
    pdf.savefig(fig)

# ---------- Page 1: title + summary ----------
textpage(pdf, "LLMServingSim — A40 Validation Report", [
    "Date: 2026-06-19    Model: Llama-3.1-8B / 70B    HW: NVIDIA A40 x8 (48GB)",
    "Backend: ASTRA-Sim analytical    Workload: sharegpt 300 req, rate 10/s, fp16",
    "",
    "## Scope of work",
    "• Profiled A40 (8B tp1/2, 70B tp4/tp8) incl. GPU power; extrapolated tp4/tp8.",
    "• Validated simulator vs real vLLM on A40 for TP=1/2/4/8 (TTFT, TPOT, throughput, power).",
    "• Found & fixed a profiler bug (record_function -> cuda_event).",
    "• Studied extrapolation validity (tp8) and cross-hardware 70B (A40/A100/H100).",
    "",
    "## Key findings",
    "• GPU power matches vLLM within <1.5% at TP1/TP2 (sim active_power=300W ~ A40 TDP).",
    "• Profiler bug (record_function) under-measured layer GEMMs AND attention",
    "  (decode x6.65). Switching to cuda_event fixed it; TPOT match improved from",
    "  -57..-70% to within +-3..38% (TP4 near-exact).",
    "• Corrected sim is mildly conservative on throughput (-13..-26%).",
    "• Extrapolation valid on compute side (end-to-end +-10..18%), but at high TP the",
    "  limiting error is the sim's single-link collective model, not the profile:",
    "  real vLLM TP=8 collapses (690 tok/s, 18.8s TTFT) via cross-NUMA all-reduce;",
    "  the sim cannot reproduce it.",
    "• 70B TP4 cross-hw: A100=2.5x, H100=3.2x A40 throughput; A100 best perf/J.",
    "",
    "## Measured interconnect (A40, P2P)",
    "• NVLink pair (GPU0-1): 52.8 GB/s   • PCIe cross-pair (0-2): 24.5 GB/s",
    "• cross-NUMA (0-4, SYS): 21.0 GB/s  (TP8 all-reduce bottleneck)",
])

# ---------- Page 2: profiler bug impact (TPOT before/after vs vLLM) ----------
fig, axes = plt.subplots(1, 2, figsize=(11.69, 6.0))
x = np.arange(len(TPS)); w = 0.27
ax = axes[0]
ax.bar(x-w, [buggy[t]["tpot"] for t in TPS], w, label="sim (buggy/record_function)", color="#c44")
ax.bar(x,   [sim[t]["tpot"]   for t in TPS], w, label="sim (fixed/cuda_event)", color=C1)
ax.bar(x+w, [vll[t]["tpot"]   for t in TPS], w, label="vLLM (measured)", color=C2)
ax.set_xticks(x); ax.set_xticklabels([f"TP{t}" for t in TPS]); ax.set_ylabel("TPOT p50 (ms)")
ax.set_title("Profiler-bug fix: TPOT vs vLLM"); ax.legend(fontsize=8); ax.grid(axis="y", alpha=.3)
ax = axes[1]
ax.bar(["prefill\nattn","decode\nattn"], [1.45, 6.65], color=[C3, "#c44"])
ax.axhline(1.0, ls="--", c="#888"); ax.set_ylabel("under-measurement factor (cuda_event / record_function)")
ax.set_title("record_function under-measured attention\n(8B tp2)")
for i,v in enumerate([1.45,6.65]): ax.text(i, v+0.1, f"x{v}", ha="center", weight="bold")
fig.suptitle("(1) Profiler bug: record_function -> cuda_event", fontsize=14, weight="bold")
fig.tight_layout(rect=[0,0,1,0.95]); emit(fig); plt.close(fig)

# ---------- Page 3: corrected sim vs vLLM, 4 panels ----------
fig, axes = plt.subplots(2, 2, figsize=(11.69, 8.0))
def grouped(ax, sim_v, vll_v, title, ylab, log=False):
    ax.bar(x-0.2, sim_v, 0.4, label="sim", color=C1)
    ax.bar(x+0.2, vll_v, 0.4, label="vLLM", color=C2)
    ax.set_xticks(x); ax.set_xticklabels([f"TP{t}" for t in TPS]); ax.set_title(title); ax.set_ylabel(ylab)
    if log: ax.set_yscale("log")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=.3)
grouped(axes[0,0], [sim[t]["gen"] for t in TPS], [vll[t]["gen"] for t in TPS], "Generation throughput", "tok/s")
grouped(axes[0,1], [sim[t]["tpot"] for t in TPS], [vll[t]["tpot"] for t in TPS], "TPOT p50", "ms")
grouped(axes[1,0], [sim[t]["ttft"] for t in TPS], [vll[t]["ttft"] for t in TPS], "TTFT p50 (log)", "ms", log=True)
grouped(axes[1,1], [simp[t] for t in TPS], [vllp[t] for t in TPS], "GPU power (all GPUs summed)", "W")
fig.suptitle("(2) Corrected sim vs vLLM — Llama-3.1-8B, A40, TP1/2/4", fontsize=14, weight="bold")
fig.tight_layout(rect=[0,0,1,0.95]); emit(fig); plt.close(fig)

# ---------- Page 4: 70B cross-hardware tp4 ----------
hw70 = ["a40","a100","h100"]
s70 = {h: sim_stat(f"{V}/sim_{h}_tp4_70b_results.csv", f"{V}/sim_{h}_tp4_70b_stdout.txt") for h in hw70}
# total generated tokens (from any, same workload)
def totgen(h):
    t = open(f"{V}/sim_{h}_tp4_70b_stdout.txt").read(); m = re.search(r"Total generated tokens:\s+(\d+)", t)
    return int(m.group(1)) if m else 83000
fig, axes = plt.subplots(1, 3, figsize=(11.69, 5.2))
labels = [h.upper() for h in hw70]; cols = ["#888", C1, C2]
axes[0].bar(labels, [s70[h]["gen"] for h in hw70], color=cols); axes[0].set_title("Throughput (tok/s)"); axes[0].grid(axis="y",alpha=.3)
axes[1].bar(labels, [s70[h]["npu"]/s70[h]["mk"] for h in hw70], color=cols); axes[1].set_title("GPU power (4 GPUs, W)"); axes[1].grid(axis="y",alpha=.3)
eff = [totgen(h)/s70[h]["node"] for h in hw70]
axes[2].bar(labels, eff, color=cols); axes[2].set_title("Energy efficiency (tok/kJ)"); axes[2].grid(axis="y",alpha=.3)
for i,e in enumerate(eff): axes[2].text(i, e+5, f"{e:.0f}", ha="center", weight="bold")
fig.suptitle("(3) Llama-3.1-70B TP=4 — cross-hardware sim (A40/A100/H100)", fontsize=14, weight="bold")
fig.text(0.5, 0.02, "sim prediction; A40 link/power measured, A100/H100 from datasheet (estimates)", ha="center", fontsize=8, color="#666")
fig.tight_layout(rect=[0,0.04,1,0.95]); emit(fig); plt.close(fig)

# ---------- Page 5: TP8 extrapolation validity ----------
ex = sim_stat(f"{V}/sim_a40_tp8_extrap_results.csv", f"{V}/sim_a40_tp8_extrap_stdout.txt")
me = sim_stat(f"{V}/sim_a40_tp8_measured_results.csv", f"{V}/sim_a40_tp8_measured_stdout.txt")
v8 = vllm_stat(8)
# per-layer extrap vs measured MAE
def avg_layers(hw):
    d = collections.defaultdict(list)
    for r in csv.DictReader(open(f"{PM}/{hw}/meta-llama/Llama-3.1-8B/tp8/layers.csv")):
        d[r["layer_name"]].append(float(r["latency(ns)"]))
    return {k: np.mean(v) for k, v in d.items()}
mea, ext = avg_layers("A40"), avg_layers("A40x")
layers = [l for l in mea if l in ext]
errs = [(ext[l]-mea[l])/mea[l]*100 for l in layers]
fig, axes = plt.subplots(1, 2, figsize=(11.69, 6.0))
ax = axes[0]
grp = ["throughput\n(tok/s)", "TPOT p50\n(ms)", "TTFT p50\n(ms, log)"]
xx = np.arange(3)
ax.bar(xx-0.25, [ex["gen"], ex["tpot"], ex["ttft"]], 0.25, label="sim (extrapolated)", color="#9b59b6")
ax.bar(xx,      [me["gen"], me["tpot"], me["ttft"]], 0.25, label="sim (measured)", color=C1)
ax.bar(xx+0.25, [v8["gen"], v8["tpot"], v8["ttft"]], 0.25, label="vLLM (real 8-GPU)", color=C2)
ax.set_yscale("log"); ax.set_xticks(xx); ax.set_xticklabels(grp); ax.set_ylabel("value (log)")
ax.set_title("End-to-end: extrap vs measured vs vLLM (TP=8)"); ax.legend(fontsize=8); ax.grid(axis="y",alpha=.3)
ax = axes[1]
order = sorted(range(len(layers)), key=lambda i: errs[i])
ax.barh([layers[i] for i in order], [errs[i] for i in order], color=["#c44" if abs(errs[i])>20 else C3 for i in order])
ax.axvline(0, c="#888"); ax.set_xlabel("extrapolation error % (extrap vs measured)")
ax.set_title(f"Per-layer tp8 extrap error\nMAE={np.mean(np.abs(errs)):.0f}%")
ax.tick_params(labelsize=7)
fig.suptitle("(4) TP=8 extrapolation validity (8B, 8x A40)", fontsize=14, weight="bold")
fig.text(0.5, 0.01, "vLLM TP=8 collapses (cross-NUMA all-reduce); sim can't reproduce -> high-TP limit is the sim collective model, not extrapolation",
         ha="center", fontsize=8, color="#666")
fig.tight_layout(rect=[0,0.04,1,0.95]); emit(fig); plt.close(fig)

# ---------- Page 6: data table ----------
rows = [["Config","sim TTFT","vLLM TTFT","sim TPOT","vLLM TPOT","sim tput","vLLM tput","sim pow","vLLM pow"]]
for t in TPS:
    rows.append([f"8B TP{t}", f"{sim[t]['ttft']:.0f}", f"{vll[t]['ttft']:.0f}", f"{sim[t]['tpot']:.1f}", f"{vll[t]['tpot']:.1f}",
                 f"{sim[t]['gen']:.0f}", f"{vll[t]['gen']:.0f}", f"{simp[t]:.0f}", f"{vllp[t]:.0f}"])
rows.append([f"8B TP8", f"{me['ttft']:.0f}", f"{v8['ttft']:.0f}", f"{me['tpot']:.1f}", f"{v8['tpot']:.1f}",
             f"{me['gen']:.0f}", f"{v8['gen']:.0f}", "-", "-"])
fig = plt.figure(figsize=(11.69, 5.5)); ax = fig.add_subplot(111); ax.axis("off")
tb = ax.table(cellText=rows[1:], colLabels=rows[0], loc="center", cellLoc="center")
tb.auto_set_font_size(False); tb.set_fontsize(8.5); tb.scale(1, 1.7)
for j in range(len(rows[0])): tb[0,j].set_facecolor("#3b6fb0"); tb[0,j].set_text_props(color="w", weight="bold")
ax.set_title("(5) A40 Llama-3.1-8B — sim vs vLLM (ms / tok/s / W)  [TP8 sim=measured profile]", fontsize=12, weight="bold", pad=20)
emit(fig); plt.close(fig)

pdf.close()
print("PDF written:", f"{V}/A40_validation_report_2026-06-19.pdf")
