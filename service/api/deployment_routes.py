"""Deployment REST API (plan §3.5, D2 subset — SSE metrics/logs land in D3)."""
from __future__ import annotations

import threading
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..deploy.state import DeploymentState as S
from ..deploy.store import DeploymentRow


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
    slo = job.request.slo.model_dump(exclude_none=True)
    res = state.ledger.get(job.id)
    dep_id = state.deploy_manager.create(res.id, job.tenant, spec, slo=slo)
    job.emit({"type": "deployment", "deployment_id": dep_id})
    return dep_id
