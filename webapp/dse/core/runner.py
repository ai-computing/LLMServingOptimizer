"""DSE simulation runner — wraps webapp.runner.run_sweep.

The existing webapp.runner already handles:
  - asyncio.Semaphore(MAX_CONCURRENT) for parallel subprocess execution
  - per-PID trace/workload isolation (PID_TAG) + cleanup
  - timeout (CONFIG_TIMEOUT_S or workload.timeout_s) with SIGTERM/SIGKILL
  - SSE event broadcasting
  - status.json atomic writes
  - log/CSV parsing on completion

DSE simply rephrases its candidate list as the sweep's ConfigSpec list and
hands off to run_sweep. After completion we parse status.json → SimulationResult.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from webapp.runner import finalize_sweep, is_cancelled, run_sweep
from webapp.parser import parse_run
from webapp.hardware_catalog import build_catalog

from .config_builder import write_candidate_cluster_json
from .generator import generate_candidates, load_metadata
from .schemas import CandidateConfig, JobSpec, SimulationResult

_MAX_RETRY_ROUNDS = 3


async def run_dse_job(
    job_id: str,
    candidates: list[CandidateConfig],
    spec: JobSpec,
    job_dir: Path,
) -> list[SimulationResult]:
    """Execute every candidate's simulation in parallel and collect results.

    Args:
        job_id: matches sweep_id used by webapp.runner — also the SSE channel
        candidates: list from generator.generate_candidates()
        spec: original JobSpec (for workload, timeout)
        job_dir: output/dse_jobs/<job_id>/
    """
    # webapp.runner expects a list[ConfigSpec]. Each candidate already has
    # config_spec attached.
    config_specs = [c.config_spec for c in candidates]

    # webapp.runner.run_sweep rewrites every cluster JSON itself (it doesn't
    # use the files we pre-wrote). To make sure power modeling stays on, we
    # build a UNION power template covering every hardware found across all
    # candidates and pass it through workload. config_builder.py only reads
    # power["npu"][<hw>] entries that actually appear in instances, so extra
    # entries are harmless.
    from .config_builder import build_power_template_from_catalog
    from .generator import load_metadata
    from .interconnect import load_fabrics, resolve_interconnect
    metadata = load_metadata()
    hw_meta = metadata["hardware"]
    union_hw: dict[str, int] = {}
    for c in candidates:
        for hw, cnt in c.hw_distribution.items():
            union_hw[hw] = union_hw.get(hw, 0) + cnt
    power_template = build_power_template_from_catalog(
        union_hw, hw_meta, enable_power=True,
    )

    # Per-candidate interconnect: fabric preset (if spec.fabric set) else the
    # per-hardware catalog interconnect_bw_gbs scalar. Shared dict object — the
    # retry loop appends to it and run_sweep re-reads it each merge round.
    fabric_def = load_fabrics().get(spec.fabric) if spec.fabric else None
    interconnect_by_label: dict[str, dict] = {}
    for c in candidates:
        ic = resolve_interconnect(c.config_spec, fabric_def, hw_meta)
        if ic:
            interconnect_by_label[c.label] = ic

    workload = {
        "dataset":   spec.workload.dataset,
        "num_req":   spec.workload.num_req,
        "phase":     "full",
        "timeout_s": spec.workload.timeout_s,
        "power_template": power_template,
        "interconnect_by_label": interconnect_by_label,
    }

    scenario_json = {
        "type": "dse_job",
        "job_id": job_id,
        "spec": spec.model_dump(),
    }

    await run_sweep(
        sweep_id=job_id,
        configs=config_specs,
        scenario_json=scenario_json,
        sweep_dir=job_dir,
        workload=workload,
        broadcast_final=False,  # finalize_sweep() called after all retry rounds
        max_concurrent=spec.max_concurrent,
    )

    # Retry loop: replace failed/cancelled candidates with fresh ones drawn
    # from the untried portion of the candidate space.  Each round:
    #   1. Count failures in status.json.
    #   2. Call generate_candidates(exclude_labels=tried_labels, override_max=N)
    #      to get exactly N replacements not yet attempted.
    #   3. Run them via run_sweep(merge=True) so status.json accumulates.
    # This continues up to _MAX_RETRY_ROUNDS or until no failures remain or
    # no untried candidates are left.
    catalog = build_catalog()
    tried_labels: set[str] = {c.label for c in candidates}
    all_candidates: list[CandidateConfig] = list(candidates)

    for round_num in range(_MAX_RETRY_ROUNDS):
        # Stop retrying if the user cancelled: the "cancelled" states below
        # would otherwise be treated as failures and trigger replacement runs,
        # so a cancel would never actually halt the sweep.
        if is_cancelled(job_id):
            break
        status_path = job_dir / "status.json"
        if not status_path.exists():
            break
        status = json.loads(status_path.read_text())
        failed_count = sum(
            1 for entry in status.get("configs", {}).values()
            if entry.get("state") in ("failed", "cancelled")
        )
        if failed_count == 0:
            break

        replacements = generate_candidates(
            spec, catalog, metadata,
            exclude_labels=tried_labels,
            override_max=failed_count,
        )
        if not replacements:
            break

        for cand in replacements:
            ic = resolve_interconnect(cand.config_spec, fabric_def, hw_meta)
            if ic:
                interconnect_by_label[cand.label] = ic
            write_candidate_cluster_json(
                cand, job_dir / "configs", hw_meta, enable_power=True,
                interconnect=ic or None,
            )

        all_candidates.extend(replacements)
        tried_labels.update(c.label for c in replacements)

        # Update candidates.json so the progress UI shows the new rows.
        cand_slim = [{
            "candidate_id": c.candidate_id,
            "label": c.label,
            "hw_distribution": c.hw_distribution,
            "parallelism": c.parallelism,
            "pd_layout": c.pd_layout,
        } for c in all_candidates]
        (job_dir / "candidates.json").write_text(json.dumps(cand_slim, indent=2))

        await run_sweep(
            sweep_id=job_id,
            configs=[c.config_spec for c in replacements],
            scenario_json={"type": "dse_job_retry", "job_id": job_id,
                           "round": round_num + 1},
            sweep_dir=job_dir,
            workload=workload,
            merge=True,
            broadcast_final=False,
            max_concurrent=spec.max_concurrent,
        )

    # Broadcast terminal sweep_state once — after all retry rounds complete.
    # This closes the SSE stream on the Progress page exactly once.
    await finalize_sweep(job_id, job_dir)

    # Read status.json → SimulationResult per candidate
    return _collect_results(all_candidates, job_dir)


def _collect_results(
    candidates: list[CandidateConfig], job_dir: Path,
) -> list[SimulationResult]:
    """Parse job_dir/status.json + per-config CSVs into SimulationResult list."""
    status = json.loads((job_dir / "status.json").read_text())
    configs = status.get("configs", {})
    out: list[SimulationResult] = []

    for cand in candidates:
        entry = configs.get(cand.label, {})
        state = entry.get("state", "failed")
        elapsed = float(entry.get("elapsed_s", 0))
        metrics = entry.get("metrics", {})

        # If status.json doesn't carry full metrics (older runs), re-parse
        log_path = job_dir / "runs" / f"{cand.label}.log"
        csv_path = job_dir / "runs" / f"{cand.label}.csv"
        if state == "done" and not metrics:
            try:
                metrics = parse_run(log_path, csv_path)
            except Exception:
                metrics = {}

        out.append(SimulationResult(
            candidate_id=cand.candidate_id,
            label=cand.label,
            state=state if state in ("done", "failed", "timeout", "cancelled") else "failed",
            elapsed_s=elapsed,
            metrics=metrics,
            cluster_config_path=str(job_dir / "configs" / f"{cand.label}.json"),
            raw_csv_path=str(csv_path) if csv_path.exists() else None,
            log_path=str(log_path) if log_path.exists() else None,
            error=entry.get("error"),
        ))
    return out


def run_dse_job_sync(
    job_id: str, candidates: list[CandidateConfig], spec: JobSpec, job_dir: Path,
) -> list[SimulationResult]:
    """Synchronous wrapper for CLI use (avoids asyncio.run boilerplate)."""
    return asyncio.run(run_dse_job(job_id, candidates, spec, job_dir))
