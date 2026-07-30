"""Deployment persistence: deployments + deployment_events tables sharing the
ledger SQLite DB (plan §3.2). Short transactions only (ledger convention)."""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .state import DeploymentState, check_transition

_SCHEMA = """
CREATE TABLE IF NOT EXISTS deployments (
  id TEXT PRIMARY KEY,
  reservation_id INTEGER NOT NULL,
  tenant TEXT NOT NULL,
  model TEXT NOT NULL,
  state TEXT NOT NULL,
  spec_json TEXT NOT NULL,
  endpoints_json TEXT,
  slo_json TEXT,
  created_at REAL, ready_at REAL, terminated_at REAL,
  last_error TEXT
);
CREATE TABLE IF NOT EXISTS deployment_events (
  dep_id TEXT NOT NULL,
  ts REAL NOT NULL,
  from_state TEXT NOT NULL,
  to_state TEXT NOT NULL,
  detail TEXT
);
"""


@dataclass
class DeploymentRow:
    id: str
    reservation_id: int
    tenant: str
    model: str
    state: DeploymentState
    spec: dict
    endpoints: Optional[dict] = None
    slo: Optional[dict] = None
    created_at: float = 0.0
    ready_at: Optional[float] = None
    terminated_at: Optional[float] = None
    last_error: Optional[str] = None
    events: list[dict] = field(default_factory=list)


class DeployStore:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        with self._conn() as c:
            c.executescript(_SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    def create(self, reservation_id: int, tenant: str, model: str, spec: dict,
               slo: Optional[dict] = None) -> DeploymentRow:
        dep_id = "dep-" + uuid.uuid4().hex[:12]
        now = time.time()
        with self._conn() as c:
            c.execute(
                "INSERT INTO deployments (id, reservation_id, tenant, model, "
                "state, spec_json, slo_json, created_at) VALUES (?,?,?,?,?,?,?,?)",
                (dep_id, reservation_id, tenant, model,
                 DeploymentState.PENDING.value, json.dumps(spec),
                 json.dumps(slo) if slo else None, now))
            c.execute("INSERT INTO deployment_events VALUES (?,?,?,?,?)",
                      (dep_id, now, "-", DeploymentState.PENDING.value, "created"))
        return self.get(dep_id)

    def transition(self, dep_id: str, to: DeploymentState, detail: str = "",
                   error: Optional[str] = None) -> DeploymentRow:
        """Validated state transition; audit-logged. Idempotent no-op when the
        row is already in ``to``."""
        with self._conn() as c:
            row = c.execute("SELECT state FROM deployments WHERE id=?",
                            (dep_id,)).fetchone()
            if row is None:
                raise KeyError(f"unknown deployment {dep_id}")
            cur = DeploymentState(row["state"])
            if cur == to:
                return self.get(dep_id)
            check_transition(cur, to)
            now = time.time()
            sets, vals = ["state=?"], [to.value]
            if to == DeploymentState.READY:
                sets.append("ready_at=?"); vals.append(now)
            if to in (DeploymentState.RELEASED, DeploymentState.FAILED,
                      DeploymentState.STOPPED):
                sets.append("terminated_at=?"); vals.append(now)
            if error is not None:
                sets.append("last_error=?"); vals.append(error)
            vals.append(dep_id)
            c.execute(f"UPDATE deployments SET {', '.join(sets)} WHERE id=?", vals)
            c.execute("INSERT INTO deployment_events VALUES (?,?,?,?,?)",
                      (dep_id, now, cur.value, to.value, detail or error or ""))
        return self.get(dep_id)

    def set_endpoints(self, dep_id: str, endpoints: dict) -> None:
        with self._conn() as c:
            c.execute("UPDATE deployments SET endpoints_json=? WHERE id=?",
                      (json.dumps(endpoints), dep_id))

    def get(self, dep_id: str) -> DeploymentRow:
        with self._conn() as c:
            row = c.execute("SELECT * FROM deployments WHERE id=?",
                            (dep_id,)).fetchone()
            if row is None:
                raise KeyError(f"unknown deployment {dep_id}")
            events = [dict(r) for r in c.execute(
                "SELECT ts, from_state, to_state, detail FROM deployment_events "
                "WHERE dep_id=? ORDER BY ts, rowid", (dep_id,))]
        return DeploymentRow(
            id=row["id"], reservation_id=row["reservation_id"],
            tenant=row["tenant"], model=row["model"],
            state=DeploymentState(row["state"]),
            spec=json.loads(row["spec_json"]),
            endpoints=json.loads(row["endpoints_json"]) if row["endpoints_json"] else None,
            slo=json.loads(row["slo_json"]) if row["slo_json"] else None,
            created_at=row["created_at"], ready_at=row["ready_at"],
            terminated_at=row["terminated_at"], last_error=row["last_error"],
            events=events)

    def list(self, states: Optional[list[DeploymentState]] = None,
             tenant: Optional[str] = None) -> list[DeploymentRow]:
        q, vals = "SELECT id FROM deployments", []
        conds = []
        if states:
            conds.append(f"state IN ({','.join('?' * len(states))})")
            vals += [s.value for s in states]
        if tenant:
            conds.append("tenant=?"); vals.append(tenant)
        if conds:
            q += " WHERE " + " AND ".join(conds)
        q += " ORDER BY created_at"
        with self._conn() as c:
            ids = [r["id"] for r in c.execute(q, vals)]
        return [self.get(i) for i in ids]
