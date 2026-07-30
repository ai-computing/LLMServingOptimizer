"""Regression: per-backend CLI path anchoring in Stage-2 evaluation.

upstream's build_cluster_config unconditionally prepends "../" after its
astra-sim chdir (same as legacy), so BOTH simulator backends need paths
relative to their backend root; only the measured backend (root = repo root)
accepts absolute paths. Passing absolute paths to upstream produced
'..//home/...' FileNotFoundError (found via the service tab, 2026-07-30).
"""
from __future__ import annotations

from pathlib import Path

from planner.sim_evaluator import _rebase_path_args
from planner.utils import REPO_ROOT


ARGS = ["--cluster-config", "output/planner_stage/x/configs/c.json",
        "--dataset", "dataset/wl.jsonl",
        "--output", "output/planner_stage/x/sim_out/c.csv",
        "--num-reqs", "10"]


def test_relative_mode_anchors_at_backend_root():
    root = REPO_ROOT / "backends" / "upstream"
    out = _rebase_path_args(ARGS, root, absolute=False)
    cfg = out[out.index("--cluster-config") + 1]
    assert not Path(cfg).is_absolute()
    # relative path must resolve back to the original repo file
    assert (root / cfg).resolve() == (REPO_ROOT / ARGS[1]).resolve()
    assert out[out.index("--num-reqs") + 1] == "10"  # non-path args untouched


def test_absolute_mode_for_measured():
    out = _rebase_path_args(ARGS, REPO_ROOT, absolute=True)
    cfg = out[out.index("--cluster-config") + 1]
    assert Path(cfg).is_absolute()
    assert Path(cfg) == (REPO_ROOT / ARGS[1]).resolve()


def test_evaluate_uses_relative_for_upstream():
    """Guard the flag wiring itself: the evaluate() source must anchor
    upstream relatively (absolute only for measured)."""
    import inspect

    from planner import sim_evaluator
    src = inspect.getsource(sim_evaluator.evaluate)
    assert 'absolute=(backend == "measured")' in src
