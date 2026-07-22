# Planner specs

A spec is a single YAML file describing the cluster, workload, requirements, and
search space. It is validated by `planner/spec_schema.py` (pydantic) and against
the repo's profiles/model configs before any solving happens.

Validate a spec without running anything:

```bash
python -m planner.cli --spec planner/specs/example_hetero_8gpu.yaml --validate-only
```

## Sections

| Section | Meaning |
|---|---|
| `model` | `name` (HuggingFace id, must have `model_config/<name>.json`), `fp` |
| `workload` | `dataset` (repo-root-relative path), `num_req` |
| `topology` | `nodes[]` with `devices[] {name, count, mem_gb}`, inter-node `links[]`, optional `intra_node_bandwidth`, optional `tp_group_shape` |
| `profiles` | `perf_root` (default `llm_profile/perf_models`) |
| `requirements` | hard constraints `ttft_ms`/`tpot_ms`/`itl_p99_ms` + `objectives[]` |
| `search_space` | `tp_choices`, `pp_choices` (must be `[1]`), P/D ranges, `batch_tokens_choices` |
| `solver` | `top_k`, `time_limit_sec`, `pareto_epsilon_steps` |

## Rules & gotchas

- **Device names** must match a hardware profile directory under `perf_root`, and
  the `(hardware, model, tp)` combination must have a *complete* profile
  (`layers.csv` + attention predictions). Validation reports missing combos.
- **Bandwidth units**: uppercase `B` = bytes (`600GBps`), lowercase `b` = bits
  (`200Gbps`). Latency accepts `ns`/`us`/`ms`/`s`.
- **`pp_choices` must be `[1]`**: pipeline parallelism is not a `cluster_config`
  knob in this simulator (only `npu_num`/`npu_group`). TP → `npu_group`.
- **Objective metrics** are `throughput` and `toks_per_wh` only; latency targets
  belong in the hard-constraint block. `toks_per_wh` triggers automatic power-block
  emission (and best-effort energy parsing from the sim's stdout).

## Files

- `example_hetero_8gpu.yaml` — H100 + A6000 + A5000 running Llama-3.1-8B; all
  combos have complete profiles, so it validates and can run end-to-end.
