# planner — heterogeneous-cluster resource-allocation planner

An **offline planner** that, given a heterogeneous cluster (mixed GPUs/NPUs) and
user requirements (SLO + objectives), finds a good **resource allocation** and
emits a validated `cluster_config/*.json` for LLMServingSim.

It is a **non-invasive wrapper**: it never edits the simulator. It only produces
simulator inputs (`cluster_config` JSON + CLI args) and parses simulator outputs
(`output/*.csv`). See `../PLAN_MILP_MaxFlow.md` for the full design.

## Two-stage design

1. **Stage 1 — structural (MILP / OR-Tools CP-SAT)**: model the cluster as a
   graph, enumerate feasible instance templates `(hardware, tp, role)`, and pick
   instance counts that maximize a coarse throughput proxy under device-capacity,
   memory, and (optional) P/D-balance constraints → **Top-K candidates**.
2. **Stage 2 — simulation**: render each candidate to a `cluster_config`, run
   `main.py` as a subprocess, parse `output/*.csv` for TTFT/TPOT/ITL-p99/throughput
   (and energy from stdout when power modeling is on), check SLO constraints, and
   re-rank (Pareto + weighted score).

## Install

```bash
pip install -r ../requirements-planner.txt
```

## Usage

```bash
# validate a spec against the repo's profiles/model configs (no solving)
python -m planner.cli --spec planner/specs/example_hetero_8gpu.yaml --validate-only

# Stage-1 + rendering only (fast; no simulator)
python -m planner.cli --spec planner/specs/example_hetero_8gpu.yaml --dry-run

# full run (Stage 1 + Stage 2 simulation)
python -m planner.cli \
    --spec planner/specs/example_hetero_8gpu.yaml \
    --out-dir planner_out/ --jobs 8
```

Outputs land in `--out-dir`:

| File | Contents |
|---|---|
| `best_cluster_config.json` | winning rendered config (drop straight into `main.py`) |
| `best_run.json` | winning run's CLI args + metrics |
| `pareto.csv` | every candidate: metrics, pass/fail, Pareto membership |
| `report.md` | human-readable summary |
| `configs/` | all rendered candidate configs |
| `sim_out/` | per-candidate simulator CSVs |
| `cache/` | per-run result cache (dedupes repeated evaluations) |

## Modules

| Module | Responsibility |
|---|---|
| `spec_schema.py` | pydantic spec model + repo validation (profiles/model configs) |
| `graph_model.py` | topology → `networkx` graph; bandwidth/latency unit parsing |
| `milp_solver.py` | Stage 1: CP-SAT → Top-K `Allocation`s |
| `config_renderer.py` | `Allocation` → `cluster_config` JSON + CLI args |
| `sim_evaluator.py` | Stage 2: subprocess + CSV parsing + disk cache |
| `objective.py` | SLO constraints, weighted score, Pareto front |
| `search_orchestrator.py` | full pipeline (parallel Stage 2) |
| `report.py` | writes best config + `pareto.csv` + `report.md` |
| `cli.py` | entry point |
| `types.py` | shared dataclasses (`Device`/`Instance`/`Allocation`/`Metrics`) |
| `utils.py` | logging, hashing, profile-catalog scan, model-size estimator |

## Known limitations

- **PP not modeled**: the simulator has no pipeline-parallel config knob
  (`npu_group ≤ npu_num` only), so `pp_choices` must be `[1]`. TP maps to
  `npu_group`; `npu_num/npu_group` is the data-parallel replica count.
- **`toks_per_wh`** requires power modeling; the planner attaches a power block
  automatically when a `toks_per_wh` objective is present, and parses energy from
  the simulator's stdout (best-effort — see `PLAN_MILP_MaxFlow.md` §8).
- The Stage-1 throughput/power proxy uses coarse per-hardware constants; Stage-2
  simulation is the source of truth and re-ranks the candidates.

## Tests

```bash
pytest planner/tests/          # unit tests (no simulator required)
```
