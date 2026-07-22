"""FastAPI routes for the MILP/Max-Flow planner tab. Mounted at /api/planner.

Storage (filesystem, mirrors the DSE tool):
  output/planner_jobs/<job_id>/
    ├── spec.json                 # the PlannerSpec used
    ├── status.json               # live progress (SSE polls this)
    ├── configs/<run_id>.json     # rendered candidate cluster configs
    ├── sim_out/<run_id>.csv      # per-candidate simulator output
    ├── cache/                    # sim result cache
    ├── pareto.csv                # report.py output
    ├── report.md
    └── best_cluster_config.json  # winning config (if any passed SLO)

Progress is streamed by polling status.json (few candidates, minute-scale sims),
which keeps us clear of the sync-planner / async-webapp thread boundary.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from webapp.config import OUTPUT_DIR
from webapp.hardware_catalog import build_catalog, list_hardware, list_models_for_hardware

from planner.report import write_reports
from planner.search_orchestrator import run_spec
from planner.spec_schema import PlannerSpec

planner_router = APIRouter()

PLANNER_ROOT = OUTPUT_DIR.parent / "planner_jobs"

# Typical accelerator memory (GB) to pre-fill the form; user can override.
_DEFAULT_MEM_GB = {
    "H100": 80, "A100": 80, "A6000": 48, "A40": 48, "A40x": 48,
    "A5000": 24, "RTX3090": 24, "RNGD": 48, "TPU-v6e-1": 32,
}

# per-job locks guarding status.json read-modify-write from worker threads
_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _lock_for(job_id: str) -> threading.Lock:
    with _locks_guard:
        return _locks.setdefault(job_id, threading.Lock())


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _job_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-planner")


def _job_dir(job_id: str) -> Path:
    return PLANNER_ROOT / job_id


def _status_path(job_id: str) -> Path:
    return _job_dir(job_id) / "status.json"


def _read_status(job_id: str) -> dict:
    p = _status_path(job_id)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return {}


def _write_status(job_id: str, status: dict) -> None:
    _status_path(job_id).write_text(json.dumps(status, indent=2, default=str))


def _mutate_status(job_id: str, fn) -> None:
    """Thread-safe read-modify-write of status.json."""
    with _lock_for(job_id):
        status = _read_status(job_id)
        fn(status)
        _write_status(job_id, status)


# ---------------------------------------------------------------------------
# Background job

def _make_on_event(job_id: str):
    def on_event(ev: dict) -> None:
        etype = ev.get("type")
        if etype == "stage1":
            def _f(s):
                cands = ev.get("candidates", [])
                s["stage1_candidates"] = len(cands)
                s["candidates"] = {
                    c["run_id"]: {
                        "batch_tokens": c["batch_tokens"],
                        "hw_summary": c["hw_summary"],
                        "state": "pending", "passed": None, "metrics": None,
                    } for c in cands
                }
                if s.get("state") == "queued":
                    s["state"] = "running"
            _mutate_status(job_id, _f)
        elif etype == "candidate":
            def _f(s):
                c = s.setdefault("candidates", {}).get(ev["run_id"])
                if c is not None:
                    c["state"] = ev.get("state")
                    c["passed"] = ev.get("passed")
                    c["metrics"] = ev.get("metrics")
                    c["reason"] = ev.get("reason")
            _mutate_status(job_id, _f)
        elif etype == "finished":
            def _f(s):
                s["best_run_id"] = ev.get("best_run_id")
                s["pareto"] = ev.get("pareto", [])
                s["num_passed"] = ev.get("num_passed", 0)
            _mutate_status(job_id, _f)
    return on_event


def _execute_job(job_id: str, spec_dict: dict, jobs: int, dry_run: bool,
                 timeout_sec: int) -> None:
    job_dir = _job_dir(job_id)
    try:
        spec = PlannerSpec.model_validate(spec_dict)
        result = run_spec(
            spec, out_dir=job_dir, jobs=jobs, dry_run=dry_run,
            timeout_sec=timeout_sec, on_event=_make_on_event(job_id),
        )
        write_reports(result, job_dir)
        _mutate_status(job_id, lambda s: s.update(
            state="done", finished_at=_now()))
    except Exception as e:  # noqa: BLE001
        (job_dir / "error.txt").write_text(f"{type(e).__name__}: {e}")
        _mutate_status(job_id, lambda s: s.update(
            state="failed", finished_at=_now(), error=f"{type(e).__name__}: {e}"))


# ---------------------------------------------------------------------------
# Routes

@planner_router.get("/catalog")
async def api_catalog() -> JSONResponse:
    """Hardware -> models -> available TP degrees, plus default memory."""
    catalog = build_catalog()
    out: dict[str, Any] = {}
    for hw in list_hardware(catalog):
        models = {
            model: sorted(catalog.get((hw, model), frozenset()))
            for model in list_models_for_hardware(catalog, hw)
            if catalog.get((hw, model))
        }
        out[hw] = {"models": models, "default_mem_gb": _DEFAULT_MEM_GB.get(hw, 48)}
    return JSONResponse({"hardware": out})


@planner_router.post("/validate")
async def api_validate(spec_dict: dict) -> JSONResponse:
    """Structural + repo validation without running anything."""
    try:
        spec = PlannerSpec.model_validate(spec_dict)
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"ok": False, "problems": [str(e)]}, status_code=400)
    problems = spec.validate_against_repo()
    return JSONResponse({"ok": not problems, "problems": problems})


@planner_router.post("/jobs")
async def api_create_job(body: dict, background: BackgroundTasks) -> JSONResponse:
    """Create + launch a planner job. body = {spec, jobs?, dry_run?, timeout_sec?}."""
    spec_dict = body.get("spec")
    if not spec_dict:
        raise HTTPException(status_code=400, detail="missing 'spec'")
    try:
        spec = PlannerSpec.model_validate(spec_dict)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"invalid spec: {e}")
    problems = spec.validate_against_repo()
    if problems:
        raise HTTPException(status_code=400, detail="; ".join(problems))

    job_id = _job_id()
    job_dir = _job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "spec.json").write_text(json.dumps(spec_dict, indent=2))
    _write_status(job_id, {
        "job_id": job_id, "state": "queued", "created_at": _now(),
        "dry_run": bool(body.get("dry_run", False)),
        "candidates": {}, "stage1_candidates": 0,
    })
    background.add_task(
        _execute_job, job_id, spec_dict,
        int(body.get("jobs", 4)), bool(body.get("dry_run", False)),
        int(body.get("timeout_sec", 1800)),
    )
    return JSONResponse({"job_id": job_id})


@planner_router.get("/jobs")
async def api_list_jobs() -> JSONResponse:
    PLANNER_ROOT.mkdir(parents=True, exist_ok=True)
    out = []
    for p in sorted(PLANNER_ROOT.iterdir(), reverse=True):
        if not p.is_dir():
            continue
        st = _read_status(p.name)
        if not st:
            continue
        out.append({
            "job_id": p.name, "state": st.get("state"),
            "created_at": st.get("created_at"),
            "num_candidates": len(st.get("candidates", {})),
            "num_passed": st.get("num_passed"),
        })
    return JSONResponse({"jobs": out})


@planner_router.get("/jobs/{job_id}")
async def api_get_job(job_id: str) -> JSONResponse:
    if not _job_dir(job_id).exists():
        raise HTTPException(status_code=404, detail="job not found")
    spec_p = _job_dir(job_id) / "spec.json"
    return JSONResponse({
        "job_id": job_id,
        "status": _read_status(job_id),
        "spec": json.loads(spec_p.read_text()) if spec_p.exists() else None,
    })


@planner_router.get("/jobs/{job_id}/results")
async def api_get_results(job_id: str) -> JSONResponse:
    import csv as _csv
    job_dir = _job_dir(job_id)
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="job not found")
    rows = []
    pareto_csv = job_dir / "pareto.csv"
    if pareto_csv.exists():
        with open(pareto_csv) as f:
            rows = list(_csv.DictReader(f))
    report_md = job_dir / "report.md"
    best_cfg = job_dir / "best_cluster_config.json"
    return JSONResponse({
        "job_id": job_id,
        "status": _read_status(job_id),
        "candidates": rows,
        "report_md": report_md.read_text() if report_md.exists() else "",
        "has_best_config": best_cfg.exists(),
    })


@planner_router.get("/jobs/{job_id}/download/config")
async def api_download_config(job_id: str) -> FileResponse:
    best = _job_dir(job_id) / "best_cluster_config.json"
    if not best.exists():
        raise HTTPException(status_code=404, detail="no best config (no candidate passed SLO)")
    return FileResponse(str(best), media_type="application/json",
                        filename=f"{job_id}_cluster_config.json")


@planner_router.get("/jobs/{job_id}/events")
async def api_events(job_id: str) -> StreamingResponse:
    """SSE by polling status.json (emit on change; close on terminal state)."""
    import asyncio

    async def stream():
        last = None
        # ~20 min ceiling (600 * 2s) guards against an orphaned browser tab
        for _ in range(600):
            st = _read_status(job_id)
            payload = json.dumps(st, default=str)
            if payload != last:
                yield f"data: {payload}\n\n"
                last = payload
            if st.get("state") in ("done", "failed"):
                return
            await asyncio.sleep(2)
        yield f"data: {json.dumps({'type': 'timeout'})}\n\n"

    return StreamingResponse(
        stream(), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
