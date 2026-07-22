"""DSE CLI entry point — runs an exploration from a YAML/JSON spec file.

Usage:
    python -m webapp.dse.cli explore --spec examples/dse/spec.yaml --out results/

Pipeline:
  1. Load spec → JobSpec
  2. Generate candidates
  3. Build per-candidate cluster JSONs
  4. Run sweep (parallel subprocesses)
  5. Rank → write top_n.json + all_candidates.json + pareto.json
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from webapp.config import REPO_ROOT
from webapp.hardware_catalog import build_catalog

from .core.config_builder import write_candidate_cluster_json
from .core.generator import generate_candidates, load_metadata
from .core.ranker import rank_candidates
from .core.runner import run_dse_job_sync
from .core.schemas import JobSpec


def _now_id(name: str = "dse") -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{ts}-{name}"


def cmd_explore(args: argparse.Namespace) -> int:
    # 1. Load spec
    spec_path = Path(args.spec)
    if not spec_path.exists():
        print(f"ERROR: spec not found: {spec_path}", file=sys.stderr)
        return 1
    raw = (yaml.safe_load(spec_path.read_text())
           if spec_path.suffix in (".yaml", ".yml")
           else json.loads(spec_path.read_text()))
    try:
        spec = JobSpec.model_validate(raw)
    except Exception as e:
        print(f"ERROR: spec validation failed: {e}", file=sys.stderr)
        return 1
    print(f"[1/5] Loaded spec: {spec_path}")

    # 2. Generate candidates
    catalog = build_catalog()
    metadata = load_metadata()
    candidates = generate_candidates(spec, catalog, metadata)
    if not candidates:
        print("ERROR: 0 candidates after pre-filter. Check resource_pool / model memory.",
              file=sys.stderr)
        return 1
    print(f"[2/5] Generated {len(candidates)} candidates")

    # 3. Build per-candidate cluster JSONs
    job_id = _now_id(args.job_name or "dse")
    job_dir = Path(args.out or (REPO_ROOT / "output" / "dse_jobs")) / job_id
    configs_dir = job_dir / "configs"
    for cand in candidates:
        write_candidate_cluster_json(
            cand, configs_dir, metadata["hardware"], enable_power=True,
        )
    # Persist candidate metadata so the progress page can display hw / parallelism.
    # Matches what webapp/dse/server/routes.py:_execute_job writes for API jobs.
    cand_slim = [{
        "candidate_id": c.candidate_id,
        "label": c.label,
        "hw_distribution": c.hw_distribution,
        "parallelism": c.parallelism,
        "pd_layout": c.pd_layout,
    } for c in candidates]
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "candidates.json").write_text(json.dumps(cand_slim, indent=2))
    print(f"[3/5] Wrote {len(candidates)} cluster JSONs → {configs_dir}")

    # 4. Run simulations
    print(f"[4/5] Running simulations (job_dir={job_dir}) ...")
    results = run_dse_job_sync(job_id, candidates, spec, job_dir)
    done = sum(1 for r in results if r.state == "done")
    print(f"      done: {done}/{len(results)}")

    # 5. Rank
    # Reload from candidates.json — run_dse_job_sync may have extended the
    # list with retry candidates beyond the initially generated set.
    from types import SimpleNamespace as _NS
    _cand_slim = json.loads((job_dir / "candidates.json").read_text())
    cand_by_label = {
        c["label"]: _NS(hw_distribution=c["hw_distribution"])
        for c in _cand_slim
    }
    ranked = rank_candidates(
        results, spec.constraints, spec.weights, spec.top_n,
        candidates_by_label=cand_by_label, diversity=True,
    )

    # Persist
    (job_dir / "all_candidates.json").write_text(
        json.dumps([r.model_dump() for r in ranked.all_results], indent=2, default=str)
    )
    (job_dir / "top_n.json").write_text(
        json.dumps(
            [ranked.all_results[i].model_dump() for i in ranked.top_n_indices],
            indent=2, default=str,
        )
    )
    (job_dir / "pareto.json").write_text(
        json.dumps(
            [ranked.all_results[i].model_dump() for i in ranked.pareto_indices],
            indent=2, default=str,
        )
    )
    print(f"[5/5] Wrote rankings → {job_dir}")
    print()
    print("Top-N:")
    for rank, idx in enumerate(ranked.top_n_indices, 1):
        r = ranked.all_results[idx]
        hw = cand_by_label[r.label].hw_distribution
        print(f"  {rank}. {r.label:40s} hw={hw} score={r.score} "
              f"ttft_p99={r.metrics.get('p99_ttft_ms')} tp={r.metrics.get('total_token_tp')}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(prog="webapp.dse.cli", description="DSE CLI")
    sub = p.add_subparsers(dest="command", required=True)

    e = sub.add_parser("explore", help="Run a DSE job from a spec file")
    e.add_argument("--spec", required=True, help="YAML/JSON spec path")
    e.add_argument("--out", help="Output root dir (default: output/dse_jobs)")
    e.add_argument("--job-name", help="Job suffix (default: 'dse')")
    e.set_defaults(func=cmd_explore)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
