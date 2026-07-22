"""FastAPI routes for the DSE tool. Mounted into webapp/app.py.

Storage model: filesystem-only (matches existing webapp sweep pattern).
  output/dse_jobs/<job_id>/
    ├── spec.json                # JobSpec
    ├── status.json              # webapp.runner state
    ├── candidates.json          # generated CandidateConfig list (slim)
    ├── all_candidates.json      # SimulationResult list (post-run)
    ├── top_n.json
    ├── pareto.json
    ├── configs/<label>.json     # webapp.runner writes these
    └── runs/<label>.{log,csv}

Job listing: dir scan on output/dse_jobs/.
Caching: spec hash → if a previous job with same hash exists in "done" state,
return its results immediately (no re-run).
"""
from __future__ import annotations

import asyncio
import hashlib
import io
import json
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from webapp.config import OUTPUT_DIR, compute_max_concurrent
from webapp.hardware_catalog import build_catalog, list_hardware, list_models_for_hardware
from webapp.parser import parse_run
from webapp.runner import cancel_sweep, subscribe_events, unsubscribe_events

from ..core.config_builder import write_candidate_cluster_json
from ..core.generator import dry_run_detail, generate_candidates, load_metadata
from ..core.ranker import rank_candidates
from ..core.runner import run_dse_job
from ..core.schemas import JobSpec, ObjectiveWeights, SimulationResult

dse_router = APIRouter()

DSE_ROOT = OUTPUT_DIR.parent / "dse_jobs"


# ---------------------------------------------------------------------------
# Helpers

def _now_id(name: str = "dse") -> str:
    return f"{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}-{name}"


def _spec_hash(spec: JobSpec) -> str:
    """Stable hash for cache lookup. Excludes top_n / weights (rerank-only)."""
    hashable = spec.model_dump()
    # weights & top_n can be re-applied without re-simulating
    hashable.pop("weights", None)
    hashable.pop("top_n", None)
    payload = json.dumps(hashable, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _job_dir(job_id: str) -> Path:
    return DSE_ROOT / job_id


def _list_jobs() -> list[dict]:
    """List all DSE jobs (dir scan). Most recent first."""
    DSE_ROOT.mkdir(parents=True, exist_ok=True)
    out = []
    for p in DSE_ROOT.iterdir():
        if not p.is_dir():
            continue
        status_path = p / "status.json"
        if not status_path.exists():
            continue
        try:
            status = json.loads(status_path.read_text())
        except json.JSONDecodeError:
            continue
        out.append({
            "job_id": p.name,
            "state": status.get("state"),
            "created_at": status.get("created_at"),
            "finished_at": status.get("finished_at"),
            "progress_done": len([c for c in status.get("configs", {}).values()
                                  if c.get("state") in ("done", "failed", "cancelled")]),
            "progress_total": len(status.get("configs", {})),
        })
    out.sort(key=lambda r: r.get("created_at") or "", reverse=True)
    return out


def _find_cached(spec_hash: str) -> str | None:
    """Return job_id of a completed job with this spec hash, or None."""
    DSE_ROOT.mkdir(parents=True, exist_ok=True)
    for p in DSE_ROOT.iterdir():
        if not p.is_dir():
            continue
        hash_marker = p / "spec_hash.txt"
        if not hash_marker.exists():
            continue
        if hash_marker.read_text().strip() != spec_hash:
            continue
        # Check the job finished successfully
        status_path = p / "status.json"
        if status_path.exists():
            try:
                if json.loads(status_path.read_text()).get("state") == "done":
                    return p.name
            except json.JSONDecodeError:
                pass
    return None


def _load_spec(job_dir: Path) -> JobSpec:
    return JobSpec.model_validate(json.loads((job_dir / "spec.json").read_text()))


def _load_candidates_meta(job_dir: Path) -> dict[str, dict]:
    """Label → {hw_distribution, parallelism, pd_layout, candidate_id}."""
    path = job_dir / "candidates.json"
    if not path.exists():
        return {}
    return {c["label"]: c for c in json.loads(path.read_text())}


def _results_from_status(job_dir: Path) -> list[SimulationResult]:
    """Reconstruct SimulationResult list from status.json + runs/ on disk.

    Used as a fallback when the ranked *.json files were never persisted — e.g.
    a cancelled job (ranking races _execute_job) or an interrupted run. Mirrors
    core.runner._collect_results but keyed off status.json alone (no candidate
    objects needed).
    """
    status_path = job_dir / "status.json"
    if not status_path.exists():
        return []
    try:
        status = json.loads(status_path.read_text())
    except json.JSONDecodeError:
        return []
    cand_meta = _load_candidates_meta(job_dir)
    out: list[SimulationResult] = []
    for label, entry in status.get("configs", {}).items():
        state = entry.get("state", "queued")
        if state not in ("done", "failed", "timeout", "cancelled"):
            continue  # still queued/running — not a result yet
        metrics = entry.get("metrics", {})
        log_path = job_dir / "runs" / f"{label}.log"
        csv_path = job_dir / "runs" / f"{label}.csv"
        if state == "done" and not metrics:
            try:
                metrics = parse_run(log_path, csv_path)
            except Exception:
                metrics = {}
        out.append(SimulationResult(
            candidate_id=cand_meta.get(label, {}).get("candidate_id", label),
            label=label,
            state=state,
            elapsed_s=float(entry.get("elapsed_s", 0)),
            metrics=metrics,
            cluster_config_path=str(job_dir / "configs" / f"{label}.json"),
            raw_csv_path=str(csv_path) if csv_path.exists() else None,
            log_path=str(log_path) if log_path.exists() else None,
            error=entry.get("error"),
        ))
    return out


# ---------------------------------------------------------------------------
# Background job runner

async def _execute_job(job_id: str, spec: JobSpec) -> None:
    """Generate → build → simulate → rank → persist. Runs as BackgroundTask."""
    job_dir = _job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)

    try:
        catalog = build_catalog()
        metadata = load_metadata()
        candidates = generate_candidates(spec, catalog, metadata)

        # Persist candidate list (slim — config_spec is non-serializable)
        cand_slim = [{
            "candidate_id": c.candidate_id,
            "label":        c.label,
            "hw_distribution": c.hw_distribution,
            "parallelism":  c.parallelism,
            "pd_layout":    c.pd_layout,
        } for c in candidates]
        (job_dir / "candidates.json").write_text(json.dumps(cand_slim, indent=2))

        if not candidates:
            # Write an empty-success status so the page doesn't hang
            (job_dir / "status.json").write_text(json.dumps({
                "sweep_id": job_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "state": "done",
                "configs": {},
            }, indent=2))
            return

        # Run sweep — webapp.runner.run_sweep handles MAX_CONCURRENT, SSE, status.json
        results = await run_dse_job(job_id, candidates, spec, job_dir)

        # Reload candidate meta from disk — run_dse_job may have extended
        # the list with retry candidates beyond the initial `candidates`.
        all_cand_meta = _load_candidates_meta(job_dir)
        cand_by_label = {
            lbl: SimpleNamespace(hw_distribution=m["hw_distribution"])
            for lbl, m in all_cand_meta.items()
        }
        ranked = rank_candidates(
            results, spec.constraints, spec.weights, spec.top_n,
            candidates_by_label=cand_by_label, diversity=True,
        )

        (job_dir / "all_candidates.json").write_text(
            json.dumps([r.model_dump() for r in ranked.all_results], indent=2, default=str)
        )
        (job_dir / "top_n.json").write_text(
            json.dumps([ranked.all_results[i].model_dump()
                        for i in ranked.top_n_indices], indent=2, default=str)
        )
        (job_dir / "pareto.json").write_text(
            json.dumps([ranked.all_results[i].model_dump()
                        for i in ranked.pareto_indices], indent=2, default=str)
        )
    except Exception as e:
        # Best-effort: record failure
        (job_dir / "error.txt").write_text(f"{type(e).__name__}: {e}")
        raise


# ---------------------------------------------------------------------------
# Routes — /api/dse/...

@dse_router.get("/concurrency")
async def api_concurrency() -> JSONResponse:
    """Return current auto-computed max concurrent simulation count."""
    return JSONResponse({"max_concurrent": compute_max_concurrent()})


@dse_router.get("/catalog")
async def api_catalog() -> JSONResponse:
    """Hardware + models + availability — merged from build_catalog() + 03_catalog.yaml."""
    catalog = build_catalog()
    metadata = load_metadata()
    hw_meta = metadata.get("hardware", {})
    model_meta = metadata.get("models", {})

    out: dict[str, Any] = {"hardware": {}, "models": {}, "fabrics": {}}
    for hw in list_hardware(catalog):
        out["hardware"][hw] = {
            **hw_meta.get(hw, {}),
            "available_models": {
                model: sorted(catalog.get((hw, model), frozenset()))
                for model in list_models_for_hardware(catalog, hw)
                if catalog.get((hw, model))
            },
        }
    for model_name, meta in model_meta.items():
        out["models"][model_name] = meta

    from ..core.interconnect import load_fabrics
    out["fabrics"] = load_fabrics()
    return JSONResponse(out)


@dse_router.post("/dry-run")
async def api_dry_run(spec: JobSpec) -> JSONResponse:
    """Estimate candidate count and return the full candidate list."""
    catalog = build_catalog()
    try:
        unique_count, simulated_count, candidates = dry_run_detail(spec, catalog)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return JSONResponse({
        "estimated_candidates": unique_count,
        "simulated_candidates": simulated_count,
        "candidates": candidates,
    })


@dse_router.post("/jobs")
async def api_create_job(spec: JobSpec, background: BackgroundTasks) -> JSONResponse:
    """Create a new DSE job. Returns job_id + estimated candidate count."""
    spec_hash = _spec_hash(spec)
    cached = _find_cached(spec_hash)
    if cached:
        return JSONResponse({"job_id": cached, "cached": True, "estimated_candidates": None})

    job_id = _now_id("dse")
    job_dir = _job_dir(job_id)
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "spec.json").write_text(json.dumps(spec.model_dump(), indent=2))
    (job_dir / "spec_hash.txt").write_text(spec_hash)

    catalog = build_catalog()
    _, simulated_count, _ = dry_run_detail(spec, catalog)

    background.add_task(_execute_job, job_id, spec)
    return JSONResponse({"job_id": job_id, "cached": False, "estimated_candidates": simulated_count})


@dse_router.get("/jobs")
async def api_list_jobs() -> JSONResponse:
    return JSONResponse({"jobs": _list_jobs()})


@dse_router.get("/jobs/history")
async def api_jobs_history() -> JSONResponse:
    """Return past DSE jobs with spec for the history dropdown."""
    DSE_ROOT.mkdir(parents=True, exist_ok=True)
    out = []
    for p in DSE_ROOT.iterdir():
        if not p.is_dir() or p.name.endswith(".deleted"):
            continue
        spec_path = p / "spec.json"
        if not spec_path.exists():
            continue
        try:
            spec = json.loads(spec_path.read_text())
            status_path = p / "status.json"
            status = json.loads(status_path.read_text()) if status_path.exists() else {}
            rp = spec.get("resource_pool", {})
            hw_summary = ", ".join(
                f"{it['hw']} x[{it['min']}-{it['max']}]"
                for it in rp.get("items", [])
            )
            out.append({
                "job_id": p.name,
                "created_at": status.get("created_at", ""),
                "state": status.get("state", "unknown"),
                "model_name": spec.get("model", {}).get("name", ""),
                "dataset": spec.get("workload", {}).get("dataset", ""),
                "hw_summary": hw_summary,
                "spec": spec,
            })
        except Exception:
            continue
    out.sort(key=lambda r: r.get("created_at") or "", reverse=True)
    return JSONResponse(out)


@dse_router.get("/jobs/{job_id}")
async def api_get_job(job_id: str) -> JSONResponse:
    job_dir = _job_dir(job_id)
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="job not found")
    status_path = job_dir / "status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {"state": "queued"}
    return JSONResponse({
        "job_id": job_id,
        "spec": json.loads((job_dir / "spec.json").read_text())
                if (job_dir / "spec.json").exists() else None,
        "status": status,
    })


@dse_router.get("/jobs/{job_id}/results")
async def api_get_results(job_id: str) -> JSONResponse:
    job_dir = _job_dir(job_id)
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="job not found")
    out: dict[str, Any] = {"job_id": job_id}
    for kind in ("all_candidates", "top_n", "pareto"):
        path = job_dir / f"{kind}.json"
        out[kind] = json.loads(path.read_text()) if path.exists() else []
    out["candidates_meta"] = _load_candidates_meta(job_dir)

    # Fallback: ranked files absent or empty (cancelled / interrupted job) but
    # some candidates already finished — rank them on the fly so Open Results
    # shows the completed ones instead of a blank page.
    if not out["all_candidates"]:
        results = _results_from_status(job_dir)
        if any(r.state == "done" for r in results):
            spec = _load_spec(job_dir)
            cand_meta = out["candidates_meta"]
            cand_by_label = {
                lbl: SimpleNamespace(hw_distribution=m["hw_distribution"])
                for lbl, m in cand_meta.items()
            }
            ranked = rank_candidates(
                results, spec.constraints, spec.weights, spec.top_n,
                candidates_by_label=cand_by_label, diversity=True,
            )
            out["all_candidates"] = [r.model_dump() for r in ranked.all_results]
            out["top_n"] = [ranked.all_results[i].model_dump()
                            for i in ranked.top_n_indices]
            out["pareto"] = [ranked.all_results[i].model_dump()
                             for i in ranked.pareto_indices]
    return JSONResponse(out)


@dse_router.post("/jobs/{job_id}/rerank")
async def api_rerank(job_id: str, body: dict) -> JSONResponse:
    """Apply new weights/top_n to existing all_candidates.json (no re-simulation)."""
    job_dir = _job_dir(job_id)
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="job not found")
    all_path = job_dir / "all_candidates.json"
    if not all_path.exists():
        raise HTTPException(status_code=400, detail="job not yet completed")

    spec = _load_spec(job_dir)
    weights = ObjectiveWeights(**body.get("weights", spec.weights.model_dump()))
    top_n_count = int(body.get("top_n", spec.top_n))

    results = [SimulationResult.model_validate(r)
               for r in json.loads(all_path.read_text())]
    # Reset on_pareto / score / meets_slo
    for r in results:
        r.on_pareto = False
        r.score = None
        r.meets_slo = True

    cand_meta = _load_candidates_meta(job_dir)
    # rank_candidates wants candidates_by_label objects with .hw_distribution
    cand_by_label = {
        lbl: SimpleNamespace(hw_distribution=m["hw_distribution"])
        for lbl, m in cand_meta.items()
    }

    ranked = rank_candidates(
        results, spec.constraints, weights, top_n_count,
        candidates_by_label=cand_by_label, diversity=True,
    )
    return JSONResponse({
        "top_n":       [ranked.all_results[i].model_dump() for i in ranked.top_n_indices],
        "pareto":      [ranked.all_results[i].model_dump() for i in ranked.pareto_indices],
        "all_results": [r.model_dump() for r in ranked.all_results],
        "weights_used": weights.model_dump(),
    })


@dse_router.post("/jobs/{job_id}/cancel")
async def api_cancel_job(job_id: str) -> JSONResponse:
    """Stop a running sweep but KEEP the job dir + partial results.

    Distinct from DELETE: cancel signals the runner to stop spawning work and
    SIGTERMs in-flight subprocesses, then lets _execute_job rank+persist the
    candidates that already finished so Open Results shows the completed ones.
    """
    job_dir = _job_dir(job_id)
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="job not found")
    try:
        await cancel_sweep(job_id)
    except Exception:
        pass
    return JSONResponse({"cancelled": True})


@dse_router.delete("/jobs/{job_id}")
async def api_delete_job(job_id: str) -> JSONResponse:
    job_dir = _job_dir(job_id)
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="job not found")
    # Best-effort cancel running subprocess (sweep_id == job_id)
    try:
        await cancel_sweep(job_id)
    except Exception:
        pass
    # Soft-delete: rename to <id>.deleted (kept for forensics)
    new_path = job_dir.with_name(job_dir.name + ".deleted")
    job_dir.rename(new_path)
    return JSONResponse({"deleted": True, "moved_to": str(new_path)})


@dse_router.get("/jobs/{job_id}/events")
async def api_events(job_id: str) -> StreamingResponse:
    """SSE — reuse webapp.runner.subscribe_events. sweep_id == job_id."""
    async def event_stream():
        # SSE event protocol:
        #   event: snapshot — full status.json payload sent once on connect
        #   data: <json>    — per-candidate progress events from webapp.runner
        #   data: {"type":"heartbeat"} — keepalive every ~5 s during idle
        # The stream closes when sweep_state reaches "done"/"failed"/"cancelled".
        queue = subscribe_events(job_id)
        try:
            # Initial snapshot so the progress page can render existing state
            # without waiting for the first queued event.
            job_dir = _job_dir(job_id)
            status_path = job_dir / "status.json"
            if status_path.exists():
                snapshot = json.loads(status_path.read_text())
                yield f"event: snapshot\ndata: {json.dumps(snapshot)}\n\n"
                # Job already finished before client connected — close stream immediately
                if snapshot.get("state") in ("done", "failed", "cancelled"):
                    yield f"data: {json.dumps({'sweep_state': snapshot['state']})}\n\n"
                    return
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=5)
                    yield f"data: {json.dumps(event)}\n\n"
                    if event.get("sweep_state") in ("done", "failed", "cancelled"):
                        break
                except asyncio.TimeoutError:
                    # No event in 5 s — send heartbeat so the browser knows
                    # the connection is still alive.
                    yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
        finally:
            unsubscribe_events(job_id, queue)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@dse_router.get("/jobs/{job_id}/download.zip")
async def api_download_zip(job_id: str) -> StreamingResponse:
    job_dir = _job_dir(job_id)
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail="job not found")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in job_dir.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(job_dir))
    buf.seek(0)
    return StreamingResponse(
        buf, media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{job_id}.zip"'},
    )
