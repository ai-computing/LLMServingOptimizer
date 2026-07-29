"""Reservation ledger: SQLite + snapshot-version optimistic locking (§4.4).

Design decision (§6): planning never locks — it works on a snapshot; conflicts
are detected at confirm time by comparing the snapshot version the plan was
computed against with the current one, and double-booking of individual
devices is re-checked inside the same transaction.
"""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from .registry import ClusterRegistry


class ConflictError(Exception):
    """Snapshot went stale or a requested device is no longer free."""


@dataclass
class Reservation:
    id: int
    tenant: str
    request_id: str
    device_ids: list[str]
    state: str          # reserved | active | released
    snapshot_ver: int
    created_at: float


_SCHEMA = """
CREATE TABLE IF NOT EXISTS reservations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tenant TEXT NOT NULL,
    request_id TEXT NOT NULL UNIQUE,
    device_ids TEXT NOT NULL,
    state TEXT NOT NULL DEFAULT 'reserved',
    snapshot_ver INTEGER NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value INTEGER NOT NULL
);
INSERT OR IGNORE INTO meta (key, value) VALUES ('version', 0);
"""


class Ledger:
    def __init__(self, db_path: str | Path, registry: ClusterRegistry):
        self.db_path = str(db_path)
        self.registry = registry
        self._all_devices = set(registry.device_ids())
        with self._conn() as c:
            c.executescript(_SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        return conn

    # -- queries ------------------------------------------------------------
    def snapshot(self) -> tuple[int, list[str]]:
        """(version, free device ids). Free = registry − active reservations."""
        with self._conn() as c:
            ver = c.execute("SELECT value FROM meta WHERE key='version'").fetchone()[0]
            used: set[str] = set()
            for row in c.execute(
                    "SELECT device_ids FROM reservations WHERE state != 'released'"):
                used.update(json.loads(row["device_ids"]))
        return ver, sorted(self._all_devices - used)

    def active_reservations(self) -> list[Reservation]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM reservations WHERE state != 'released' "
                "ORDER BY id").fetchall()
        return [self._to_res(r) for r in rows]

    def get(self, request_id: str) -> Reservation | None:
        with self._conn() as c:
            row = c.execute("SELECT * FROM reservations WHERE request_id=?",
                            (request_id,)).fetchone()
        return self._to_res(row) if row else None

    @staticmethod
    def _to_res(row) -> Reservation:
        return Reservation(id=row["id"], tenant=row["tenant"],
                           request_id=row["request_id"],
                           device_ids=json.loads(row["device_ids"]),
                           state=row["state"],
                           snapshot_ver=row["snapshot_ver"],
                           created_at=row["created_at"])

    # -- mutations ----------------------------------------------------------
    def reserve(self, tenant: str, request_id: str, device_ids: list[str],
                snapshot_ver: int) -> Reservation:
        """Atomically reserve devices against the given snapshot version.

        Raises :class:`ConflictError` when the snapshot is stale OR any device
        is already held; both checks run inside one immediate transaction, so
        two racing confirms cannot double-book (one commits, the other sees
        the bumped version).
        """
        unknown = [d for d in device_ids if d not in self._all_devices]
        if unknown:
            raise ValueError(f"unknown devices: {unknown}")
        if not device_ids:
            raise ValueError("empty reservation")

        conn = self._conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            ver = conn.execute(
                "SELECT value FROM meta WHERE key='version'").fetchone()[0]
            if ver != snapshot_ver:
                raise ConflictError(
                    f"snapshot version {snapshot_ver} is stale (current {ver}); "
                    "re-plan against a fresh snapshot")
            used: set[str] = set()
            for row in conn.execute(
                    "SELECT device_ids FROM reservations WHERE state != 'released'"):
                used.update(json.loads(row[0]))
            clash = sorted(used.intersection(device_ids))
            if clash:
                raise ConflictError(f"devices already reserved: {clash}")
            cur = conn.execute(
                "INSERT INTO reservations (tenant, request_id, device_ids, state, "
                "snapshot_ver, created_at) VALUES (?,?,?,?,?,?)",
                (tenant, request_id, json.dumps(sorted(device_ids)), "reserved",
                 snapshot_ver, time.time()))
            conn.execute("UPDATE meta SET value = value + 1 WHERE key='version'")
            conn.commit()
            rid = cur.lastrowid
        except sqlite3.IntegrityError as e:
            conn.rollback()
            raise ConflictError(f"request_id '{request_id}' already has a "
                                f"reservation") from e
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        res = self.get(request_id)
        assert res is not None and res.id == rid
        return res

    def release(self, request_id: str) -> bool:
        """Release a reservation; idempotent (returns False when already
        released or unknown)."""
        conn = self._conn()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT state FROM reservations WHERE request_id=?",
                (request_id,)).fetchone()
            if row is None or row["state"] == "released":
                conn.rollback()
                return False
            conn.execute("UPDATE reservations SET state='released' "
                         "WHERE request_id=?", (request_id,))
            conn.execute("UPDATE meta SET value = value + 1 WHERE key='version'")
            conn.commit()
            return True
        finally:
            conn.close()
