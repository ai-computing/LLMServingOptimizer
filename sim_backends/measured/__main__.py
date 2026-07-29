"""Subprocess entry point: ``python -m sim_backends.measured``.

Accepts the upstream-style CLI flags the planner's config_renderer emits
(unknown flags are tolerated and ignored) so sim_evaluator can shell out to
the measured backend exactly like it does for the simulators.
"""
from __future__ import annotations

import argparse
import sys

from .backend import run_from_files


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="sim_backends.measured")
    p.add_argument("--cluster-config", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--num-reqs", "--num-req", dest="num_reqs", type=int, default=0)
    p.add_argument("--request-routing-policy", default="RR")
    p.add_argument("--run-id", default=None)
    args, _unknown = p.parse_known_args(argv)

    routing = "WEIGHTED" if args.request_routing_policy.upper() == "CUSTOM" else "RR"
    stdout = run_from_files(args.cluster_config, args.dataset, args.output,
                            num_reqs=args.num_reqs, routing=routing)
    sys.stdout.write(stdout)
    return 0


if __name__ == "__main__":
    sys.exit(main())
