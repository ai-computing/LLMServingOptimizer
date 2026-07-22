# LLMServingSim Web UI — Design Plan

Web-based parallelism sweep explorer for LLMServingSim.  
Users specify heterogeneous hardware scenarios; the app auto-enumerates all valid
TP/PP/DP/P-D configurations, runs simulations in parallel, and renders interactive plots.

---

## Architecture

```
Browser  ──── HTTP (REST + Server-Sent Events) ────►  FastAPI app (uvicorn 0.0.0.0:8000)
                                                            │
                                                    asyncio task pool  (K = min(4, cpu//2))
                                                            │
                                                  asyncio.create_subprocess_exec
                                                  python3 main.py --cluster-config …
                                                            │
                                              output/web_sweeps/<sweep_id>/
                                              ├─ scenario.json
                                              ├─ configs/<label>.json
                                              ├─ runs/<label>.{csv,log}
                                              ├─ status.json
                                              └─ metrics.json
```

---

## UI Flow (3 pages)

### 1 — Scenario Builder (`GET /`)

- Tab bar: multiple scenarios can be defined for cross-hardware comparison.
- Per-scenario form:
  - **Instance rows**: hardware (dropdown), model, npu_count, pd_role (combined/prefill/decode/auto).
  - **Axes** checkboxes: vary TP / vary PP / vary DP / include P/D split.
  - **Workload**: dataset (auto-filtered by model tokenizer), num_req slider.
  - **Phase**: smoke (10 req) / full (100 req) / both.
- **"Enumerate configs"** → `POST /api/enumerate` → preview table with count + ETA.
  - If count > 20: warning + per-config checkboxes (first 20 default-on).
- **"Run sweep"** → `POST /api/sweeps` → redirect to progress page.

### 2 — Progress (`GET /sweep/<id>`)

- Server-Sent Events from `GET /api/sweeps/<id>/events`.
  Each event: `{label, state, elapsed_s, last_log_line}`.
  States: `queued | running | done | failed | cancelled`.
- Table: one row per config, state badge, elapsed time, last log tail.
- **Cancel** button → `POST /api/sweeps/<id>/cancel` (SIGTERM to subprocesses).
- Results link activates when all configs reach a terminal state.

### 3 — Results (`GET /sweep/<id>/results`)

Four chart sections (Plotly.js via CDN, figures rendered server-side):

| Section | Chart type | X | Y / grouping |
|---|---|---|---|
| Bar charts | Grouped bars | Config label | Throughput / TTFT / TPOT / ITL |
| Pareto scatter | Scatter | Avg latency (selectable) | Total throughput; Pareto frontier line |
| Axis scaling | Line | TP / PP / DP value | Metric; separate trace per other-axis combination |
| Per-request CDF | Step CDF | Latency (ms) | Cumulative fraction; one trace per config |

- **Compare scenarios** toggle: overlay traces from multiple scenarios with distinct colors.
- Download buttons: `metrics.json`, `report.md`, zip of all CSVs/logs.

---

## Module Responsibilities

| Module | Responsibility |
|---|---|
| `webapp/hardware_catalog.py` | Scan `llm_profile/perf_models/<hw>/<model>/tp*` → `{(hw,model): set[tp]}` |
| `webapp/enumerate.py` | Scenario → list of `ConfigSpec` (label, axes, npu params); enforces all constraints |
| `webapp/cluster_builder.py` | `ConfigSpec` → cluster JSON dict the simulator accepts |
| `webapp/parser.py` | Log regex + CSV reader → `metrics` dict (ns→ms, Mean/P99 triplets) |
| `webapp/runner.py` | asyncio sweep orchestrator; `Semaphore(K)` concurrency gate; SSE event queue |
| `webapp/plots.py` | `metrics` list → Plotly JSON dicts (4 chart types) |
| `webapp/app.py` | FastAPI entry; route registry; startup catalog scan |
| `webapp/templates/` | Jinja2 HTML (base, index, progress, results, sweeps list) |
| `webapp/static/` | `app.css`, `app.js` (form handling, SSE client, Plotly init) |
| `script/serve_webapp.sh` | Env setup (LD_LIBRARY_PATH, PATH) + `uvicorn webapp.app:app …` |

---

## Key Constraints

**Sweep enumeration rules** (all enforced in `enumerate.py`):
- `npu_group ≤ npu_num` AND `npu_num % npu_group == 0` (`config_builder.py:252,411`)
- `npus_per_group = npu_num // npu_group` must be in the profiled TP set
- Prefill instances cost `2 × npu_num` physical NPUs (`config_builder.py:292-294`)
- P/D with decode `npu_num > 1` is skipped (known topology crash)

**Runtime constraints**:
- Simulator requires `LD_LIBRARY_PATH=/tmp/protobuf_prefix/usr/lib/x86_64-linux-gnu`
  and `PATH=$HOME/.local/bin:$PATH` (for the `python` symlink used by `graph_generator.py`)
- Each config runs as one Python parent + one AnalyticalAstra child subprocess
- Typical wall-clock per config: 60–180 s (100 reqs), 10–30 s (smoke / 10 reqs)

**Dataset–model tokenizer coupling** (not enforced by simulator):
- `*_llama.jsonl` → Llama-3.1-8B / Llama-3.1-70B
- `*_mixtral.jsonl` → Mixtral-8x7B
- `*_phi.jsonl` → Phi-mini-MoE

---

## Hardware Catalog (from `llm_profile/perf_models/`)

| Hardware | Model | Profiled TP |
|---|---|---|
| A6000 | Llama-3.1-8B | 1, 2 |
| A6000 | Phi-mini-MoE | 1, 2 |
| H100 | Llama-3.1-8B | 1, 2 |
| H100 | Llama-3.1-70B | 1, 2, 4 |
| H100 | Mixtral-8x7B | 1, 2, 4 |
| RTX3090 | Llama-3.1-8B | 1, 2 |
| RTX3090 | Phi-mini-MoE | 1, 2 |
| TPU-v6e-1 | Llama-3.1-8B | 1 |

---

## File Layout

```
webapp/
  __init__.py
  app.py
  config.py
  hardware_catalog.py
  enumerate.py
  cluster_builder.py
  runner.py
  parser.py
  plots.py
  templates/
    base.html
    index.html
    progress.html
    results.html
    sweeps_list.html
  static/
    app.css
    app.js
script/
  serve_webapp.sh
output/
  web_sweeps/           (created at runtime)
```

---

## Build Order

1. `hardware_catalog.py` → `parser.py` → `cluster_builder.py`  (standalone modules)
2. `enumerate.py`  (depends on hardware_catalog)
3. `runner.py`     (depends on cluster_builder, parser)
4. `plots.py`      (depends on parser output format)
5. `app.py` + templates + `app.js`  (ties everything together)
6. `script/serve_webapp.sh`

---

## Verification Checklist

- [ ] `enumerate.py` round-trip: `{A6000, 4 NPUs, Llama-3.1-8B}` → 11 configs (matching the known A6000-4 sweep, excluding the 2 crash-prone P/D variants)
- [ ] End-to-end smoke: scenario `{A6000, 2 NPUs, Llama-3.1-8B, 10 reqs}` completes in <60 s and all 4 chart types render
- [ ] Heterogeneous P/D: `{H100 prefill, A6000 decode}` produces valid cluster JSON
- [ ] Cross-scenario overlay: two scenarios on the results page, distinct colors
- [ ] Cancel: subprocesses reaped (`pgrep AnalyticalAstra` returns nothing after cancel)
- [ ] Soft-cap: scenario yielding >20 configs shows checkbox UI
- [ ] Reload-safe: progress page reconnects SSE after browser refresh
