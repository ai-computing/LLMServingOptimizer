"""Service API router (§4.5): submit → plan (async) → recommend → confirm/release.

Mountable into the existing webapp FastAPI app via ``app.include_router``.
The planning function is injectable (``ServiceState.planner_fn``) so the API
layer is unit-testable with a mock planner; the default implementation drives
the real pipeline: inventory snapshot → workload synthesis → fidelity routing
→ power-min planner → recommendation.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from ..fidelity_router import RoutingError, route
from ..inventory.ledger import ConflictError, Ledger
from ..inventory.registry import ClusterRegistry
from ..inventory.views import available_topology, pick_devices
from .models import (
    ConfirmOut,
    JobStatusOut,
    RecommendationOut,
    ServeRequestIn,
)


@dataclass
class Job:
    id: str
    tenant: str
    request: ServeRequestIn
    state: str = "pending"
    error: Optional[str] = None
    result: Optional[dict] = None          # RecommendationOut-shaped dict
    events: list[dict] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def emit(self, ev: dict) -> None:
        with self.lock:
            self.events.append(ev)


#: planner_fn(request, topology_dict, snapshot_ver, job) -> RecommendationOut-dict
PlannerFn = Callable[[ServeRequestIn, dict, int, Job], dict]


@dataclass
class ServiceState:
    registry: ClusterRegistry
    ledger: Ledger
    out_root: Path = Path("output/service_jobs")
    planner_fn: Optional[PlannerFn] = None
    jobs: dict[str, Job] = field(default_factory=dict)
    run_async: bool = True

    def planner(self) -> PlannerFn:
        return self.planner_fn or _default_planner


# ---------------------------------------------------------------------------
# default (real) planner pipeline
# ---------------------------------------------------------------------------
def _hw_tps_of_allocation(allocation) -> dict[str, list[int]]:
    out: dict[str, set[int]] = {}
    for inst in allocation.instances:
        out.setdefault(inst.hardware, set()).add(inst.tp)
    return {hw: sorted(tps) for hw, tps in out.items()}


def _candidate_out(c) -> dict:
    m = c.metrics
    parts = [f"{i.hardware}x{i.replicas}(tp{i.tp}"
             f"{'/' + i.pd_type if i.pd_type else ''})"
             for i in c.allocation.instances]
    return {
        "run_id": c.run_id,
        "hw_summary": ", ".join(parts),
        "passed": c.passed,
        "power_w": m.power_w if m else None,
        "power_source": (m.raw.get("power_source") if m else None),
        "metrics": m.as_row() if m else None,
        "violations": list(c.violations),
    }


def _default_planner(req: ServeRequestIn, topology: dict, snapshot_ver: int,
                     job: Job) -> dict:
    from planner.search_orchestrator import run_spec
    from planner.spec_schema import PlannerSpec
    from service.workload_synth import ScaleSpec, synthesize

    out_dir = Path("output/service_jobs") / job.id
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) synthesize the evaluation workload at the target request rate (§4.3:
    #    scale validation must run at the target rate, not a fixed trace)
    synth = synthesize(
        ScaleSpec(req_per_s=req.scale.req_per_s, duration_s=req.scale.duration_s,
                  preset=req.scale.preset, seed=0),
        out_dir / "workload.jsonl")
    job.emit({"type": "workload", "demand_toks_per_s": synth.demand_toks_per_s})

    # 2) filter excluded hardware from the available topology
    if req.exclude_hw:
        topology = json.loads(json.dumps(topology))
        for node in topology["nodes"]:
            node["devices"] = [d for d in node["devices"]
                               if d["name"] not in req.exclude_hw]
        topology["nodes"] = [n for n in topology["nodes"] if n["devices"]]
        if not topology["nodes"]:
            return {"best": None, "alternatives": [], "backend": "n/a",
                    "confidence": "n/a", "reason": "no hardware left after exclusions",
                    "snapshot_ver": snapshot_ver, "device_ids": [],
                    "infeasible": {"bottleneck": "availability",
                                   "detail": "exclude_hw removed every device",
                                   "suggestions": ["relax exclude_hw"]}}

    # 3) fidelity routing over the topology's hardware set; per-hardware TP
    #    candidates come from the measured-oracle catalog when available so a
    #    fully-covered cluster routes to the measured backend
    from sim_backends import get_backend
    oracle_cat = get_backend("measured").list_hardware()
    hw_tps = {}
    for node in topology["nodes"]:
        for dev in node["devices"]:
            hw_tps[dev["name"]] = (oracle_cat.get(dev["name"], {})
                                   .get(req.model) or [1, 2])
    try:
        decision = route(hw_tps, req.model, force_backend=req.force_backend)
    except RoutingError as e:
        return {"best": None, "alternatives": [], "backend": "n/a",
                "confidence": "n/a", "reason": str(e),
                "snapshot_ver": snapshot_ver, "device_ids": [],
                "infeasible": {"bottleneck": "evaluation",
                               "detail": str(e), "suggestions": [
                                   "run an oracle campaign for the NPU",
                                   "exclude the NPU hardware"]}}
    job.emit({"type": "routing", "backend": decision.backend,
              "confidence": decision.confidence})

    # 4) build the power-min spec and run the two-stage planner
    requirements: dict = {
        "demand": {"toks_per_s": synth.demand_toks_per_s},
        "objectives": [{"metric": "power_w", "direction": "min", "weight": 1.0}],
    }
    for k in ("ttft_ms", "tpot_ms", "itl_p99_ms"):
        v = getattr(req.slo, k)
        if v is not None:
            requirements[k] = {"constraint": "<=", "value": v}
    spec = PlannerSpec.model_validate({
        "model": {"name": req.model, "fp": req.fp},
        "workload": {"dataset": str(out_dir / "workload.jsonl"),
                     "num_req": req.num_req_eval},
        "topology": topology,
        "requirements": requirements,
        "search_space": {"tp_choices": sorted({tp for tps in hw_tps.values()
                                               for tp in tps})},
        "solver": {"top_k": 4, "time_limit_sec": 30, "pareto_epsilon_steps": 3},
        "backend": decision.backend,
    })
    result = run_spec(spec, out_dir=out_dir, jobs=4,
                      on_event=lambda ev: job.emit(ev))

    out: dict = {
        "backend": decision.backend, "confidence": decision.confidence,
        "reason": decision.reason, "snapshot_ver": snapshot_ver,
        "demand_toks_per_s": synth.demand_toks_per_s, "device_ids": [],
        "best": None, "alternatives": [],
        "infeasible": (result.infeasible_report.as_dict()
                       if result.infeasible_report else None),
    }
    if req.power_cap_w is not None:
        for c in result.candidates:
            if c.metrics and c.metrics.power_w and c.metrics.power_w > req.power_cap_w:
                c.passed = False
                c.violations.append(
                    f"power_w={c.metrics.power_w:.0f} exceeds cap {req.power_cap_w}")
        passing = [c for c in result.candidates if c.passed]
        result.best = min(passing, key=lambda c: c.metrics.power_w) if passing else None

    if result.best is not None:
        out["best"] = _candidate_out(result.best)
        out["alternatives"] = [_candidate_out(c) for c in result.candidates
                               if c is not result.best]
        need: dict[tuple[str, str], int] = {}
        for inst in result.best.allocation.instances:
            key = (inst.node_id, inst.hardware)
            need[key] = need.get(key, 0) + inst.npu_num
        out["_need"] = {f"{k[0]}|{k[1]}": v for k, v in need.items()}
    return out


# ---------------------------------------------------------------------------
# router
# ---------------------------------------------------------------------------
def _run_job(state: ServiceState, job: Job) -> None:
    job.state = "running"
    job.emit({"type": "state", "state": "running"})
    try:
        ver, free = state.ledger.snapshot()
        topology = available_topology(state.registry, free)
        if not topology["nodes"]:
            job.result = {"best": None, "alternatives": [], "backend": "n/a",
                          "confidence": "n/a", "reason": "no free devices",
                          "snapshot_ver": ver, "device_ids": [],
                          "infeasible": {"bottleneck": "availability",
                                         "detail": "all devices are reserved",
                                         "suggestions": ["release resources",
                                                         "wait for capacity"]}}
            job.state = "infeasible"
        else:
            result = state.planner()(job.request, topology, ver, job)
            # resolve concrete device ids for the winning allocation
            need = {tuple(k.split("|")): v
                    for k, v in (result.pop("_need", {}) or {}).items()}
            if result.get("best") and need:
                _, free_now = state.ledger.snapshot()
                result["device_ids"] = pick_devices(free_now, need)
            job.result = result
            job.state = "done" if result.get("best") else "infeasible"
    except Exception as e:  # noqa: BLE001 — job must never crash the server
        job.state = "failed"
        job.error = f"{type(e).__name__}: {e}"
    job.emit({"type": "state", "state": job.state})


def create_service_router(state: ServiceState) -> APIRouter:
    router = APIRouter(prefix="/api", tags=["service"])

    @router.get("/cluster")
    def get_cluster():
        ver, free = state.ledger.snapshot()
        reservations = [
            {"tenant": r.tenant, "request_id": r.request_id,
             "device_ids": r.device_ids, "state": r.state}
            for r in state.ledger.active_reservations()]
        return {"snapshot_ver": ver,
                "topology": state.registry.model_dump(),
                "free_devices": free,
                "reservations": reservations}

    @router.get("/models")
    def get_models():
        """Serviceable models: measured-oracle catalog + legacy profile catalog,
        filtered to hardware present in the registry."""
        from planner.utils import scan_profile_catalog
        from sim_backends import get_backend
        cluster_hw = {d.name for n in state.registry.nodes for d in n.devices}
        catalog: dict[str, dict] = {}
        for hw, models in get_backend("measured").list_hardware().items():
            if hw in cluster_hw:
                for model, tps in models.items():
                    catalog.setdefault(model, {})[hw] = {
                        "tps": tps, "source": "measured"}
        for (hw, model), tps in scan_profile_catalog().items():
            if hw in cluster_hw:
                catalog.setdefault(model, {}).setdefault(
                    hw, {"tps": sorted(tps), "source": "legacy"})
        return {"models": catalog}

    @router.post("/serve-requests", response_model=JobStatusOut)
    def submit(req: ServeRequestIn):
        job = Job(id=uuid.uuid4().hex[:12], tenant=req.tenant, request=req)
        state.jobs[job.id] = job
        if state.run_async:
            threading.Thread(target=_run_job, args=(state, job),
                             daemon=True).start()
        else:
            _run_job(state, job)
        return _status(job)

    def _status(job: Job) -> JobStatusOut:
        return JobStatusOut(
            id=job.id, tenant=job.tenant, state=job.state, error=job.error,
            result=RecommendationOut(**job.result) if job.result else None)

    def _get_job(job_id: str) -> Job:
        job = state.jobs.get(job_id)
        if job is None:
            raise HTTPException(404, f"unknown job '{job_id}'")
        return job

    @router.get("/serve-requests/{job_id}", response_model=JobStatusOut)
    def get_status(job_id: str):
        return _status(_get_job(job_id))

    @router.get("/serve-requests/{job_id}/events")
    def get_events(job_id: str):
        """SSE progress stream (replays past events, then follows)."""
        job = _get_job(job_id)

        def stream():
            sent = 0
            while True:
                with job.lock:
                    new = job.events[sent:]
                    sent += len(new)
                for ev in new:
                    yield f"data: {json.dumps(ev)}\n\n"
                if job.state in ("done", "infeasible", "failed",
                                 "confirmed", "released"):
                    yield f"data: {json.dumps({'type': 'end', 'state': job.state})}\n\n"
                    return
                time.sleep(0.2)

        return StreamingResponse(stream(), media_type="text/event-stream")

    @router.post("/serve-requests/{job_id}/confirm", response_model=ConfirmOut)
    def confirm(job_id: str):
        job = _get_job(job_id)
        if job.state not in ("done", "replanned"):
            raise HTTPException(409, f"job is '{job.state}', not confirmable")
        result = job.result or {}
        devices = result.get("device_ids") or []
        if not devices:
            raise HTTPException(409, "recommendation carries no devices")
        try:
            res = state.ledger.reserve(job.tenant, job.id, devices,
                                       result["snapshot_ver"])
        except ConflictError as e:
            # §6: automatic single re-plan against a fresh snapshot
            job.emit({"type": "conflict", "detail": str(e)})
            _run_job(state, job)
            if job.state == "done":
                job.state = "replanned"
            return ConfirmOut(state=job.state, replanned=True,
                              device_ids=(job.result or {}).get("device_ids", []),
                              detail=f"confirm conflict ({e}); re-planned — "
                                     "review the updated recommendation and "
                                     "confirm again")
        job.state = "confirmed"
        job.emit({"type": "state", "state": "confirmed"})
        return ConfirmOut(state="confirmed", reservation_id=res.id,
                          device_ids=res.device_ids)

    @router.post("/serve-requests/{job_id}/release", response_model=ConfirmOut)
    def release(job_id: str):
        job = _get_job(job_id)
        released = state.ledger.release(job.id)
        if released:
            job.state = "released"
            job.emit({"type": "state", "state": "released"})
        return ConfirmOut(state=job.state,
                          detail=None if released else "nothing to release")

    return router
