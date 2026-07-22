# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

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

## Build / setup

`./scripts/setup.sh` — submodules (recursive), legacy chakra (user site,
protobuf 3.x) vs upstream `.venv` chakra (needs `pip install -U protobuf`
because the shipped gencode is 7.35 despite a `protobuf==6.*` pin), msgspec,
ASTRA-Sim analytical builds, `AnalyticalAstra` symlinks, profile symlinks.

## Tests

```bash
pytest tests/                 # DSE subsystem (uses legacy profile data)
```

## Verified accuracy context (A5000, Llama-3.1-8B, ShareGPT 100req)

upstream sim vs real vLLM: TP1 TTFT -19%/TPOT -8% (good), TP2 TTFT -53%/TPOT -32%
(collective cost underestimated on PCIe clusters — profiler measures per-GPU
kernels only; allreduce comes from the ASTRA-Sim network model). legacy sim is
substantially less accurate at TP1 (TTFT ~65% MAPE). Details:
`docs/A5000_VALIDATION_REPORT.md`.
