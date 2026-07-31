"""Deployment REST + SSE API (plan §3.5)."""
from __future__ import annotations

import json
import threading
import time
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ..deploy.state import DeploymentState as S
from ..deploy.store import DeploymentRow

_TERMINAL = (S.STOPPED, S.RELEASED, S.FAILED)


class TerminateIn(BaseModel):
    drain_timeout_s: float = Field(default=120.0, ge=0)


def _summary(d: DeploymentRow) -> dict:
    spec = d.spec or {}
    devices = [i for c in spec.get("containers", []) for i in c.get("device_ids", [])]
    return {"id": d.id, "tenant": d.tenant, "model": d.model,
            "state": d.state.value, "device_ids": devices,
            "endpoints": d.endpoints, "created_at": d.created_at,
            "ready_at": d.ready_at, "terminated_at": d.terminated_at,
            "last_error": d.last_error}


def create_deployment_router(state) -> APIRouter:
    router = APIRouter(prefix="/api/deployments", tags=["deployments"])

    def _store():
        if getattr(state, "deploy_store", None) is None:
            raise HTTPException(503, "deployment layer not configured")
        return state.deploy_store

    def _manager():
        if getattr(state, "deploy_manager", None) is None:
            raise HTTPException(503, "deployment layer not configured")
        return state.deploy_manager

    @router.get("")
    def list_deployments(tenant: Optional[str] = None,
                         dep_state: Optional[str] = None,
                         include_terminated: bool = False):
        states = None
        if dep_state:
            states = [S(dep_state)]
        elif not include_terminated:
            states = [s for s in S if s not in (S.RELEASED, S.FAILED)]
        return {"deployments": [_summary(d) for d in
                                _store().list(states=states, tenant=tenant)]}

    @router.get("/{dep_id}")
    def get_deployment(dep_id: str):
        try:
            d = _store().get(dep_id)
        except KeyError:
            raise HTTPException(404, f"unknown deployment '{dep_id}'")
        out = _summary(d)
        out["spec"] = d.spec
        out["slo"] = d.slo
        out["events"] = d.events
        return out

    def _runtime():
        rt = getattr(state, "monitor_runtime", None)
        if rt is None:
            raise HTTPException(503, "monitoring not configured")
        return rt

    @router.get("/{dep_id}/metrics")
    def metrics_sse(dep_id: str, interval_s: float = 5.0):
        """SSE: latest MetricSample + total power + SLO verdict every tick;
        closes when the deployment reaches a terminal state."""
        store, rt = _store(), _runtime()
        try:
            store.get(dep_id)
        except KeyError:
            raise HTTPException(404, f"unknown deployment '{dep_id}'")
        buf = rt.buf(dep_id)

        def stream():
            sent_ts = 0.0
            while True:
                row = store.get(dep_id)
                m = buf.metrics[-1] if buf.metrics else None
                p = buf.power[-1] if buf.power else None
                slo = getattr(buf, "last_slo", None)
                if m and m.ts > sent_ts:
                    sent_ts = m.ts
                    ev = {"type": "sample", "state": row.state.value,
                          "metrics": m.as_dict(),
                          "power_w": p[1] if p else None,
                          "energy_wh": round(buf.energy_wh, 3),
                          "slo": slo.as_dict() if slo else None}
                    yield f"data: {json.dumps(ev)}\n\n"
                if row.state in _TERMINAL:
                    yield f"data: {json.dumps({'type': 'end', 'state': row.state.value, 'summary': buf.summary()})}\n\n"
                    return
                time.sleep(min(1.0, interval_s))

        return StreamingResponse(stream(), media_type="text/event-stream")

    @router.get("/{dep_id}/logs")
    def logs_sse(dep_id: str, tail: int = 500, follow: bool = True):
        """SSE: replay the ring-buffer tail, then follow new lines."""
        store, rt = _store(), _runtime()
        try:
            store.get(dep_id)
        except KeyError:
            raise HTTPException(404, f"unknown deployment '{dep_id}'")
        buf = rt.buf(dep_id)

        def stream():
            lines = list(buf.logs)[-tail:]
            sent = len(buf.logs)
            for ln in lines:
                yield f"data: {json.dumps({'type': 'log', 'line': ln})}\n\n"
            while follow:
                row = store.get(dep_id)
                cur = list(buf.logs)
                for ln in cur[sent:]:
                    yield f"data: {json.dumps({'type': 'log', 'line': ln})}\n\n"
                sent = len(cur)
                if row.state in _TERMINAL:
                    yield f"data: {json.dumps({'type': 'end', 'state': row.state.value})}\n\n"
                    return
                time.sleep(0.5)
            yield f"data: {json.dumps({'type': 'end'})}\n\n"

        return StreamingResponse(stream(), media_type="text/event-stream")

    @router.post("/{dep_id}/terminate")
    def terminate(dep_id: str, body: TerminateIn = TerminateIn()):
        store, mgr = _store(), _manager()
        try:
            store.get(dep_id)
        except KeyError:
            raise HTTPException(404, f"unknown deployment '{dep_id}'")
        if mgr.run_async:
            threading.Thread(target=mgr.terminate,
                             args=(dep_id, body.drain_timeout_s),
                             daemon=True).start()
        else:
            mgr.terminate(dep_id, body.drain_timeout_s)
        return _summary(store.get(dep_id))

    return router


def make_on_stopped(state, history_path) -> callable:
    """on_stopped hook: detach monitoring and append a calibration record
    (predicted vs measured power) to the history JSONL (plan D5)."""
    from pathlib import Path

    def _on_stopped(dep_id: str) -> None:
        summary = state.monitor_runtime.detach(dep_id)
        try:
            row = state.deploy_store.get(dep_id)
            c = (row.spec.get("containers") or [{}])[0]
            hw = (c.get("device_ids") or ["?/?/0"])[0].rsplit("/", 2)[1]
            rec = {"dep_id": dep_id, "model": row.model, "hw": hw,
                   "tp": row.spec.get("engine_args", {}).get(
                       "tensor-parallel-size", 1),
                   "predicted_power_w": row.spec.get("predicted_power_w"),
                   "measured_avg_w": summary.get("avg_power_w"),
                   "energy_wh": summary.get("energy_wh"),
                   "span_s": summary.get("span_s")}
            p = Path(history_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception:
            pass  # calibration logging must never break termination

    return _on_stopped


def deploy_from_job(state, job) -> Optional[str]:
    """confirm hook: build + launch a deployment from a recommended job.

    P0 limit: exactly one TP group (one container). Multi-group/multi-node
    orchestration is D5. Returns the deployment id or None when skipped."""
    result = job.result or {}
    groups = result.get("tp_groups") or []
    devices = result.get("device_ids") or []
    if len(groups) != 1 or not devices:
        job.emit({"type": "deploy_skipped",
                  "detail": f"auto-deploy supports exactly 1 TP group (got "
                            f"{len(groups)}) in P0"})
        return None
    from ..deploy.vllm_launcher import build_spec, pick_port

    used = set()
    for d in state.deploy_store.list(states=[s for s in S if s not in
                                             (S.RELEASED, S.FAILED, S.STOPPED)]):
        for c in (d.spec or {}).get("containers", []):
            used.update(c.get("ports", {}).values())
    _node, _hw, tp = groups[0]
    spec = build_spec(job.id, job.request.model, devices, tp=int(tp),
                      port=pick_port(used), graph=state.topology_graph())
    best = result.get("best") or {}
    spec.predicted_power_w = best.get("power_w")   # D5 calibration anchor
    slo = job.request.slo.model_dump(exclude_none=True)
    res = state.ledger.get(job.id)
    dep_id = state.deploy_manager.create(res.id, job.tenant, spec, slo=slo)
    job.emit({"type": "deployment", "deployment_id": dep_id})
    return dep_id
