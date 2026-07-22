"""Backend registry: ``get_backend("legacy" | "upstream")``."""
from __future__ import annotations

from .base import (BACKENDS_DIR, REPO_ROOT, ClusterSpec, InstanceSpec,
                   NodeSpec, ScenarioSpec, SimBackend)
from .legacy import LegacyBackend
from .upstream_v1 import UpstreamBackend

_REGISTRY = {
    "legacy": LegacyBackend,
    "upstream": UpstreamBackend,
}

DEFAULT_BACKEND = "legacy"

_instances: dict[str, SimBackend] = {}


def get_backend(name: str = DEFAULT_BACKEND) -> SimBackend:
    if name not in _REGISTRY:
        raise KeyError(f"unknown backend {name!r}; choose from {sorted(_REGISTRY)}")
    if name not in _instances:
        _instances[name] = _REGISTRY[name]()
    return _instances[name]


def list_backends() -> list[str]:
    return sorted(_REGISTRY)


__all__ = ["get_backend", "list_backends", "SimBackend", "ScenarioSpec",
           "ClusterSpec", "NodeSpec", "InstanceSpec", "REPO_ROOT",
           "BACKENDS_DIR", "DEFAULT_BACKEND"]
