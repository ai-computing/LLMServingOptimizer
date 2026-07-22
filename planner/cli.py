"""Planner entry point: ``python -m planner.cli --spec <spec.yaml>``."""
from __future__ import annotations

import argparse
import sys

from . import report
from .search_orchestrator import run
from .spec_schema import load_spec
from .utils import get_logger


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="planner",
        description="Offline resource-allocation planner for heterogeneous LLM serving.",
    )
    p.add_argument("--spec", required=True, help="path to planner spec YAML")
    p.add_argument("--out-dir", default="planner_out", help="output directory")
    p.add_argument("--jobs", type=int, default=4, help="parallel Stage-2 simulations")
    p.add_argument("--dry-run", action="store_true",
                   help="Stage-1 + rendering only; skip simulation")
    p.add_argument("--validate-only", action="store_true",
                   help="validate the spec against the repo and exit")
    p.add_argument("--skip-repo-validation", action="store_true",
                   help="skip profile/model existence checks")
    p.add_argument("--timeout-sec", type=int, default=1800,
                   help="per-simulation timeout")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    log = get_logger("planner", level=args.log_level)

    if args.validate_only:
        spec = load_spec(args.spec)
        problems = spec.validate_against_repo()
        if problems:
            print("INVALID:")
            for pr in problems:
                print(f"  - {pr}")
            return 1
        print("OK: spec is valid against the repo.")
        return 0

    try:
        result = run(
            args.spec,
            out_dir=args.out_dir,
            jobs=args.jobs,
            dry_run=args.dry_run,
            skip_repo_validation=args.skip_repo_validation,
            timeout_sec=args.timeout_sec,
        )
    except (ValueError, FileNotFoundError) as e:
        log.error("%s", e)
        return 1

    paths = report.write_reports(result, args.out_dir)
    print("\n=== Planner finished ===")
    for k, v in paths.items():
        print(f"  {k}: {v}")
    if result.best is not None:
        print(f"  best run: {result.best.run_id}")
    elif not result.dry_run:
        print("  (no candidate satisfied the SLO constraints)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
