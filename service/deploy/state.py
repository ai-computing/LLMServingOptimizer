"""Deployment lifecycle state machine (plan §3.3). Any transition outside the
table raises InvalidTransition; every applied transition is audit-logged by
the store."""
from __future__ import annotations

from enum import Enum


class DeploymentState(str, Enum):
    PENDING = "PENDING"
    PULLING = "PULLING"
    STARTING = "STARTING"
    HEALTH_CHECK = "HEALTH_CHECK"
    READY = "READY"
    DEGRADED = "DEGRADED"
    DRAINING = "DRAINING"
    STOPPED = "STOPPED"
    RELEASED = "RELEASED"
    FAILED = "FAILED"


S = DeploymentState
#: allowed transitions; FAILED is reachable from every non-terminal state
ALLOWED: dict[DeploymentState, set[DeploymentState]] = {
    S.PENDING: {S.PULLING},
    S.PULLING: {S.STARTING},
    S.STARTING: {S.HEALTH_CHECK},
    S.HEALTH_CHECK: {S.READY},
    S.READY: {S.DEGRADED, S.DRAINING},
    S.DEGRADED: {S.READY, S.DRAINING},
    S.DRAINING: {S.STOPPED},
    S.STOPPED: {S.RELEASED},
    S.RELEASED: set(),
    S.FAILED: set(),
}
_TERMINAL = {S.RELEASED, S.FAILED}


class InvalidTransition(Exception):
    pass


def check_transition(a: DeploymentState, b: DeploymentState) -> None:
    if b == S.FAILED and a not in _TERMINAL:
        return
    if b not in ALLOWED[a]:
        raise InvalidTransition(f"{a.value} -> {b.value} is not allowed")
