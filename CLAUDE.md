# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

LLMServingOptimizer: our own tooling (webapp DSE UI, MILP/Max-Flow planner,
validation suite) layered on top of TWO LLMServingSim simulator backends kept
as git submodules under `backends/`:

- `backends/legacy` — ai-computing/LLMServingSim fork, branch `backend-legacy`
  (`main.py` + `inference_serving/`, old-format profiles in
  `llm_profile/perf_models/`, our core patches + A40/A5000/RNGD profiles)
- `backends/upstream` — casys-kaist/LLMServingSim v1.1.0+ (`python -m serving`
  + `serving/core/`, new vLLM-layerwise profiles in `profiler/perf/`)

NEVER edit files inside `backends/*` from this repo's tasks — they are
submodules with their own history; change them in their own checkouts/branches.

## Commands

```bash
./scripts/setup.sh                          # full environment setup (see Build/setup)
pip install -r requirements-planner.txt     # planner-only deps (ortools, networkx, pydantic, pandas)

# tests (two independent pytest suites)
pytest tests/                               # DSE subsystem (uses legacy profile data)
pytest planner/tests/                       # planner (MILP solver, renderer, mock evaluator)
pytest tests/dse/test_ranker.py -k pareto   # single file / single test

# webapp (FastAPI + uvicorn --reload, port 8000; env LLMSS_PORT/LLMSS_HOST)
./scripts/serve_webapp.sh                   # main UI at /, DSE UI at /dse/explore

# DSE from the CLI (no web server needed)
python -m webapp.dse.cli explore --spec examples/dse/spec_llama8b_smoke.yaml --job-name smoke
./scripts/demo_dse.sh                       # same smoke spec + prints Top-N

# planner
python -m planner.cli --spec planner/specs/example_hetero_8gpu.yaml --validate-only
python -m planner.cli --spec planner/specs/example_hetero_8gpu.yaml --dry-run   # Stage 1 only, no sim
python -m planner.cli --spec planner/specs/example_hetero_8gpu.yaml --out-dir planner_out/ --jobs 8
```

## The adapter layer (start here)

`sim_backends/` hides every backend difference. `get_backend("legacy"|"upstream")`
returns a `SimBackend` with `build_cluster_config`, `build_cli`, `run`,
`parse_csv` (normalizes the CSV `output` column: legacy stores input+output,
upstream stores pure output), `parse_stdout`, `list_hardware`.

Key differences the adapter absorbs (do not hand-roll these elsewhere):

| | legacy | upstream |
|---|---|---|
| entry/cwd | `python main.py`, cwd=backends/legacy | `.venv/bin/python -m serving`, cwd=backends/upstream |
| dtype / reqs | `--fp 16`, `--num-req` | `--dtype bfloat16`, `--num-reqs` |
| cluster schema | `npu_num`/`npu_group` (TP = npu_num//npu_group) | `num_npus`/`tp_size` |
| CLI paths | MUST be relative to backend root | absolute OK |
| scheduler defaults | prefix/chunked OFF, RR | prefix/chunked ON, LOAD |
| parallel isolation | webapp PID-tag cleanup | `--run-id` |

## Subsystem architecture

Both optimizers are non-invasive wrappers around the simulators: they only
produce simulator inputs (`cluster_config/*.json` + CLI args) and parse
simulator outputs (CSV + stdout).

- **webapp** (`webapp/`): FastAPI app (`app.py`), single-run + sweep UI.
  `runner.py` assembles commands via the adapter; `hardware_catalog.py` builds
  the per-backend hardware/model catalog; `sim_cache.py` dedupes runs.
- **webapp DSE** (`webapp/dse/`): pipeline in `core/` — `generator.py`
  (ResourcePool → candidate configs) → `stage1_filters.py` /
  `stage2_predictor.py` (analytical pre-filtering before simulation) →
  `config_builder.py` → `runner.py` (parallel sweep) → `ranker.py`
  (SLO filter + Pareto + weighted Top-N). Served under `/api/dse/*`
  (`server/routes.py`, SSE progress) or via `cli.py`. Job artifacts land in
  `output/dse_jobs/<timestamp>-<name>/` (all_candidates/top_n/pareto.json,
  configs/, runs/).
- **planner** (`planner/`): two stages — Stage 1 `milp_solver.py` (OR-Tools
  CP-SAT over `graph_model.py` topology graph) picks Top-K instance
  allocations; Stage 2 `sim_evaluator.py` renders each via
  `config_renderer.py`, simulates, and re-ranks (Pareto + weighted score).
  Spec = pydantic YAML (`spec_schema.py`, examples in `planner/specs/`).
  Outputs to `--out-dir`: `best_cluster_config.json`, `pareto.csv`,
  `report.md`, `configs/`, `sim_out/`, `cache/`.
- **validation** (`validation/`): scripts + archived results comparing sim vs
  real vLLM runs (A40/A5000, TP1–16); reports also in `docs/`. Root
  `run_dp_partition.py` / `run_dp_comparison.py` implement/validate the
  DP-partition method (DP=N as max over N single-instance sims).

## Integration points

- webapp: `workload["backend"]` flows from the UI dropdown (index.html →
  app.js `collectWorkload`) through `app.py` into `runner.py`, which delegates
  command assembly to the adapter. Catalog: `hardware_catalog.build_catalog(backend=)`.
- planner: `PlannerSpec.backend` (spec YAML `backend:` key) → orchestrator
  threads it into `config_renderer.render(backend=)` and
  `sim_evaluator.evaluate(backend=)`. Planner emits REPO_ROOT-relative paths;
  `sim_evaluator._rebase_path_args` re-anchors them per backend.
- DSE stage-2 predictor reads ONLY the legacy profile format
  (`backends/legacy/llm_profile/perf_models`).
- Our upstream-format profiles live in `profiles/upstream/<HW>/` and get
  symlinked into `backends/upstream/profiler/perf/` by `scripts/setup.sh`
  (the submodule stays clean).
- Cluster configs: `cluster_config/` holds legacy-format JSONs;
  `cluster_config/upstream/` holds the new format.

## Build / setup

`./scripts/setup.sh` — submodules (recursive), legacy chakra (user site,
protobuf 3.x) vs upstream `.venv` chakra (needs `pip install -U protobuf`
because the shipped gencode is 7.35 despite a `protobuf==6.*` pin), msgspec,
ASTRA-Sim analytical builds, `AnalyticalAstra` symlinks, profile symlinks.

Known quirks setup.sh already handles (don't re-fix by hand): upstream venv
needs msgspec (radix_tree AttributeError otherwise); both backends need the
`AnalyticalAstra` symlink after the ASTRA-Sim build. GPU profiling/benching
needs a separate vLLM venv (see setup.sh tail).

## Verified accuracy context (A5000, Llama-3.1-8B, ShareGPT 100req)

upstream sim vs real vLLM: TP1 TTFT -19%/TPOT -8% (good), TP2 TTFT -53%/TPOT -32%
(collective cost underestimated on PCIe clusters — profiler measures per-GPU
kernels only; allreduce comes from the ASTRA-Sim network model). legacy sim is
substantially less accurate at TP1 (TTFT ~65% MAPE). Details:
`docs/A5000_VALIDATION_REPORT.md`.
