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
from ..inventory.views import pick_devices
from ..topology.views import available_topology as _topo_view
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
    # deployment layer (D2); None -> recommendation/reservation only
    deploy_manager: Optional[object] = None
    deploy_store: Optional[object] = None

    def planner(self) -> PlannerFn:
        return self.planner_fn or _default_planner

    def topology_graph(self):
        """Cached TopologyGraph. ``registry`` may be a v2 RegistryV2 (used as
        is) or a v1 ClusterRegistry (auto-promoted), so callers and tests can
        pass either."""
        if not hasattr(self, "_topo_graph"):
            from ..topology.graph import TopologyGraph
            from ..topology.schema import RegistryV2, promote_v1
            reg = (self.registry if isinstance(self.registry, RegistryV2)
                   else promote_v1(self.registry.model_dump()))
            self._topo_graph = TopologyGraph(reg)
        return self._topo_graph

    def docker_endpoints(self) -> dict[str, dict]:
        """{node_id: {endpoint, tls}} for the docker driver — so a deployment
        targeting a remote node never silently lands on the local daemon."""
        out: dict[str, dict] = {}
        for node in self.topology_graph().registry.nodes:
            if node.docker is not None:
                out[node.id] = {"endpoint": node.docker.endpoint,
                                "tls": node.docker.tls}
        return out

    def available_topology(self, free_device_ids: list[str]) -> dict:
        # topology.views output is golden-equal to the old inventory.views
        return _topo_view(self.topology_graph().registry, free_device_ids)


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


#: evaluation sources in fidelity-ladder order (§5.4)
_SOURCE_ORDER = ("measured", "upstream", "legacy")


def model_tp_catalog(model: str) -> dict[str, dict[str, list[int]]]:
    """{backend: {hardware: [tp, ...]}} — every source that can evaluate
    ``model``, from measured oracles and both simulator profile trees."""
    from sim_backends import get_backend

    out: dict[str, dict[str, list[int]]] = {}
    for name in _SOURCE_ORDER:
        try:
            cat = get_backend(name).list_hardware()
        except Exception:      # a backend that is not set up must not break planning
            continue
        out[name] = {hw: sorted(models[model]) for hw, models in cat.items()
                     if model in models}
    return out


def preferred_tp_options(catalog: dict, hardwares) -> tuple[dict[str, list[int]], list[str]]:
    """Per-hardware TP options from the highest-fidelity source that has any,
    plus the hardware with no profile/oracle at all for this model."""
    hw_tps: dict[str, list[int]] = {}
    unsupported: list[str] = []
    for hw in sorted(hardwares):
        for name in _SOURCE_ORDER:
            tps = catalog.get(name, {}).get(hw)
            if tps:
                hw_tps[hw] = list(tps)
                break
        else:
            unsupported.append(hw)
    return hw_tps, unsupported


def _drop_hardware(topology: dict, hardware: list[str]) -> dict:
    topo = json.loads(json.dumps(topology))
    for node in topo["nodes"]:
        node["devices"] = [d for d in node["devices"] if d["name"] not in hardware]
    topo["nodes"] = [n for n in topo["nodes"] if n["devices"]]
    return topo


def proxy_toks_per_unit(model: str) -> Optional[float]:
    """Stage-1's throughput proxy constant is calibrated for an 8B model; token
    rate scales roughly inversely with parameter count, so a 70B demand would
    otherwise be ~9x over-optimistic. Returns None when the model config is
    unavailable (planner then uses its own default)."""
    try:
        from planner.utils import estimate_weight_bytes, load_model_config
        params_b = estimate_weight_bytes(load_model_config(model), 16) / 2 / 1e9
        if params_b <= 0:
            return None
        return max(50.0, 1000.0 * 8.0 / params_b)
    except Exception:
        return None


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

    # 3) per-hardware TP options for THIS model, taken from the fidelity ladder
    #    (measured oracle > upstream profile > legacy profile). A hardcoded
    #    [1, 2] fallback used to make 70B look memory-infeasible: 141 GB of
    #    weights never fit at tp<=2, while A40 tp4/tp8 profiles exist. Hardware
    #    with no profile at all for the model is dropped from the topology.
    catalog = model_tp_catalog(req.model)
    hardwares = {d["name"] for n in topology["nodes"] for d in n["devices"]}
    hw_tps, unsupported = preferred_tp_options(catalog, hardwares)
    if unsupported:
        topology = _drop_hardware(topology, unsupported)
        job.emit({"type": "hw_excluded", "hardware": unsupported,
                  "detail": f"no profile/oracle for {req.model}"})
    if not hw_tps or not topology["nodes"]:
        return {"best": None, "alternatives": [], "backend": "n/a",
                "confidence": "n/a",
                "reason": f"no evaluation profile for '{req.model}' on any "
                          f"available hardware ({', '.join(sorted(hardwares))})",
                "snapshot_ver": snapshot_ver, "device_ids": [],
                "infeasible": {
                    "bottleneck": "profiles",
                    "detail": f"none of {sorted(hardwares)} has a measured "
                              f"oracle or simulator profile for {req.model}",
                    "suggestions": [
                        "profile the model on this hardware "
                        "(python -m profiler / scripts/run_capacity_campaign.py)",
                        "pick a model from GET /api/models",
                        "add hardware that has a profile for this model"]}}
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

    # the ROUTED backend defines the evaluable search space: keep only the
    # hardware/TPs it can actually run (the preferred source above may have
    # been a different rung of the ladder for some hardware)
    routed_cat = catalog.get(decision.backend, {})
    if routed_cat:
        hw_tps = {hw: tps for hw, tps in routed_cat.items() if hw in hw_tps}
        dropped = [hw for hw in hardwares if hw not in hw_tps]
        if dropped:
            topology = _drop_hardware(topology, dropped)
            job.emit({"type": "hw_excluded", "hardware": dropped,
                      "detail": f"{decision.backend} backend cannot evaluate them"})
    if not hw_tps or not topology["nodes"]:
        return {"best": None, "alternatives": [], "backend": decision.backend,
                "confidence": decision.confidence,
                "reason": f"{decision.backend} backend has no profile for "
                          f"'{req.model}' on the available hardware",
                "snapshot_ver": snapshot_ver, "device_ids": [],
                "infeasible": {"bottleneck": "profiles",
                               "detail": f"forced/routed backend "
                                         f"'{decision.backend}' cannot evaluate "
                                         f"{req.model} here",
                               "suggestions": ["drop force_backend",
                                               "profile the model for this backend"]}}

    # 4) build the power-min spec and run the two-stage planner
    demand: dict = {"toks_per_s": synth.demand_toks_per_s}
    scale = proxy_toks_per_unit(req.model)
    if scale is not None:
        demand["proxy_toks_per_unit"] = scale
    requirements: dict = {
        "demand": demand,
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
        # per-hardware TP restriction: Stage-1 must not propose (hw, tp)
        # combos the routed backend cannot evaluate (e.g. A40 tp2 with only a
        # tp1 oracle)
        "search_space": {"tp_choices": sorted({tp for tps in hw_tps.values()
                                               for tp in tps}),
                         "hw_tp_choices": hw_tps},
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
        groups: list[list] = []   # one (node, hw, tp) per instance replica
        for inst in result.best.allocation.instances:
            key = (inst.node_id, inst.hardware)
            need[key] = need.get(key, 0) + inst.npu_num
            groups.extend([[inst.node_id, inst.hardware, inst.tp]]
                          * (inst.npu_num // inst.tp))
        out["_need"] = {f"{k[0]}|{k[1]}": v for k, v in need.items()}
        out["_groups"] = groups
    return out


# ---------------------------------------------------------------------------
# router
# ---------------------------------------------------------------------------
def _run_job(state: ServiceState, job: Job) -> None:
    job.state = "running"
    job.emit({"type": "state", "state": "running"})
    try:
        ver, free = state.ledger.snapshot()
        topology = state.available_topology(free)
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
            # resolve concrete device ids for the winning allocation —
            # affinity-refined per TP group when group info is available (D1)
            need = {tuple(k.split("|")): v
                    for k, v in (result.pop("_need", {}) or {}).items()}
            groups = [tuple(g) for g in (result.pop("_groups", None) or [])]
            result["tp_groups"] = [list(g) for g in groups]  # kept for deploy
            if result.get("best") and (need or groups):
                _, free_now = state.ledger.snapshot()
                if groups:
                    from ..topology.placement import pick_devices_affinity
                    result["device_ids"] = pick_devices_affinity(
                        state.topology_graph(), free_now, groups)
                else:
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

    @router.get("/cluster/graph")
    def get_cluster_graph():
        """Topology graph for the cluster-view UI (vertices/edges + device
        states from the ledger). Works for v1 registries via auto-promotion."""
        graph = state.topology_graph()
        _, free = state.ledger.snapshot()
        free_set = set(free)
        device_states = {d: ("free" if d in free_set else "reserved")
                         for d in graph.registry.device_ids()}
        return graph.to_ui(device_states)

    @router.get("/models")
    def get_models():
        """Serviceable models per hardware present in the registry, with the TP
        options and source planning would actually use (measured oracle >
        upstream profile > legacy profile — the fidelity ladder)."""
        from sim_backends import get_backend
        # hardware present in the cluster (via the v2 graph: works for both
        # v1-promoted and native v2 registries)
        cluster_hw = {d.hw for n in state.topology_graph().registry.nodes
                      for d in n.devices}
        catalog: dict[str, dict] = {}
        for source in _SOURCE_ORDER:      # first source wins per (model, hw)
            try:
                per_hw = get_backend(source).list_hardware()
            except Exception:
                continue
            for hw, models in per_hw.items():
                if hw not in cluster_hw:
                    continue
                for model, tps in models.items():
                    catalog.setdefault(model, {}).setdefault(
                        hw, {"tps": sorted(tps), "source": source})
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
        dep_id = None
        if job.request.auto_deploy and getattr(state, "deploy_manager", None):
            from .deployment_routes import deploy_from_job
            try:
                dep_id = deploy_from_job(state, job)
            except Exception as e:  # deployment failure must not undo confirm
                job.emit({"type": "deploy_error",
                          "detail": f"{type(e).__name__}: {e}"})
        return ConfirmOut(state="confirmed", reservation_id=res.id,
                          device_ids=res.device_ids, deployment_id=dep_id)

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
