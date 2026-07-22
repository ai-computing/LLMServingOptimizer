"""End-to-end pipeline: spec -> graph -> MILP(top_k) -> render -> sim -> rank.

Stage-2 simulations of the K candidates run in a thread pool (each is an external
subprocess, so the GIL is not a bottleneck). Setting ``dry_run=True`` stops after
rendering, which is useful for testing Stage-1 + rendering without the simulator.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Union

from . import config_renderer, milp_solver, objective
from .graph_model import build_graph
from .sim_evaluator import evaluate
from .spec_schema import PlannerSpec, load_spec
from .types import Allocation, Infeasible, Metrics
from .utils import get_logger, hash_obj

log = get_logger("planner.orchestrator")


@dataclass
class CandidateResult:
    run_id: str
    allocation: Allocation
    config_path: str
    cli_args: list[str]
    batch_tokens: int
    metrics: Optional[Metrics] = None
    infeasible_reason: Optional[str] = None
    passed: bool = False
    violations: list[str] = field(default_factory=list)
    score: float = float("-inf")


@dataclass
class PlannerResult:
    spec: PlannerSpec
    candidates: list[CandidateResult] = field(default_factory=list)
    pareto: list[str] = field(default_factory=list)  # run_ids on the Pareto front
    best: Optional[CandidateResult] = None
    dry_run: bool = False


def _expand(spec: PlannerSpec, allocations: list[Allocation], out_dir: Path):
    """Cartesian product of allocations x batch_tokens -> rendered candidates."""
    candidates: list[CandidateResult] = []
    for alloc in allocations:
        for bt in spec.search_space.batch_tokens_choices:
            base = hash_obj([alloc.signature(), bt])
            run_id = f"cand_{base}"
            rel_config, cli_args = config_renderer.render(
                alloc, spec, out_dir, run_id, batch_tokens=bt,
                backend=getattr(spec, "backend", "legacy"),
            )
            candidates.append(
                CandidateResult(
                    run_id=run_id, allocation=alloc, config_path=rel_config,
                    cli_args=cli_args, batch_tokens=bt,
                )
            )
    return candidates


def _hw_summary(c: CandidateResult) -> str:
    """Readable hardware layout for one candidate (for UI/event display)."""
    parts = []
    for i in c.allocation.instances:
        role = f"/{i.pd_type}" if i.pd_type else ""
        parts.append(f"{i.hardware}x{i.replicas}(tp{i.tp}{role})")
    return ", ".join(parts)


def run(
    spec_path: str | Path,
    out_dir: str | Path = "planner_out",
    jobs: int = 4,
    dry_run: bool = False,
    skip_repo_validation: bool = False,
    timeout_sec: int = 1800,
    on_event: Optional[Callable[[dict], None]] = None,
) -> PlannerResult:
    """Load a spec from disk and run the pipeline (see :func:`run_spec`)."""
    spec = load_spec(spec_path)
    if not skip_repo_validation:
        problems = spec.validate_against_repo()
        if problems:
            raise ValueError("spec validation failed:\n  - " + "\n  - ".join(problems))
    return run_spec(spec, out_dir=out_dir, jobs=jobs, dry_run=dry_run,
                    timeout_sec=timeout_sec, on_event=on_event)


def run_spec(
    spec: PlannerSpec,
    out_dir: str | Path = "planner_out",
    jobs: int = 4,
    dry_run: bool = False,
    timeout_sec: int = 1800,
    on_event: Optional[Callable[[dict], None]] = None,
) -> PlannerResult:
    """Run the two-stage pipeline on an already-loaded spec.

    ``on_event`` (optional) is called with progress dicts as work completes:
      {"type": "stage1", "candidates": [{run_id, batch_tokens, hw_summary}, ...]}
      {"type": "candidate", "run_id", "state": "done"|"infeasible",
                            "passed", "metrics", "reason"}
      {"type": "finished", "best_run_id", "pareto": [run_id, ...],
                           "num_passed", "num_candidates"}
    The callback may be invoked from worker threads, so it must be thread-safe.
    """
    def _emit(ev: dict) -> None:
        if on_event is not None:
            try:
                on_event(ev)
            except Exception:  # never let UI wiring break the run
                log.exception("on_event callback raised")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    graph = build_graph(spec)
    allocations = milp_solver.solve(spec, graph=graph)
    if not allocations:
        log.warning("Stage-1 found no feasible allocations")
        _emit({"type": "stage1", "candidates": []})
        _emit({"type": "finished", "best_run_id": None, "pareto": [],
               "num_passed": 0, "num_candidates": 0})
        return PlannerResult(spec=spec, dry_run=dry_run)

    candidates = _expand(spec, allocations, out_dir)
    result = PlannerResult(spec=spec, candidates=candidates, dry_run=dry_run)
    _emit({"type": "stage1", "candidates": [
        {"run_id": c.run_id, "batch_tokens": c.batch_tokens, "hw_summary": _hw_summary(c)}
        for c in candidates
    ]})
    if dry_run:
        log.info("dry run: rendered %d candidate config(s), skipping simulation", len(candidates))
        _emit({"type": "finished", "best_run_id": None, "pareto": [],
               "num_passed": 0, "num_candidates": len(candidates)})
        return result

    # Stage 2: evaluate in parallel
    def _eval(c: CandidateResult) -> CandidateResult:
        res: Union[Metrics, Infeasible] = evaluate(
            c.cli_args, c.run_id, out_dir, timeout_sec=timeout_sec,
            backend=getattr(spec, "backend", "legacy"),
        )
        if isinstance(res, Infeasible):
            c.infeasible_reason = res.reason
            _emit({"type": "candidate", "run_id": c.run_id, "state": "infeasible",
                   "passed": False, "reason": res.reason})
            return c
        c.metrics = res
        c.passed, c.violations = objective.check_constraints(res, spec.requirements)
        c.score = objective.score(res, spec.requirements)
        _emit({"type": "candidate", "run_id": c.run_id, "state": "done",
               "passed": c.passed, "metrics": res.as_row()})
        return c

    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        futures = {pool.submit(_eval, c): c for c in candidates}
        for fut in as_completed(futures):
            fut.result()  # populated in place

    # rank: passing candidates by score, then Pareto front
    passing = [c for c in candidates if c.passed and c.metrics is not None]
    pf_input = [(c.run_id, c.metrics) for c in passing]
    result.pareto = objective.pareto_front(pf_input, spec.requirements) if pf_input else []
    if passing:
        result.best = max(passing, key=lambda c: c.score)
    log.info(
        "Stage-2 done: %d candidates, %d passed SLO, %d on Pareto front",
        len(candidates), len(passing), len(result.pareto),
    )
    _emit({"type": "finished",
           "best_run_id": result.best.run_id if result.best else None,
           "pareto": list(result.pareto),
           "num_passed": len(passing), "num_candidates": len(candidates)})
    return result
