"""M10 tests: the webapp service tab renders and the service API mounts when a
cluster registry is configured (and degrades gracefully when it is not)."""
from __future__ import annotations

import importlib

import pytest

pytest.importorskip("fastapi")


def _client(monkeypatch, registry_path):
    if registry_path is None:
        monkeypatch.delenv("LLMSS_CLUSTER_REGISTRY", raising=False)
        monkeypatch.setenv("LLMSS_CLUSTER_REGISTRY", "/nonexistent/registry.yaml")
    else:
        monkeypatch.setenv("LLMSS_CLUSTER_REGISTRY", str(registry_path))
    import webapp.app as W
    importlib.reload(W)
    from fastapi.testclient import TestClient
    return TestClient(W.app)


def test_service_page_disabled_without_registry(monkeypatch):
    client = _client(monkeypatch, None)
    r = client.get("/service")
    assert r.status_code == 200
    assert "Service disabled" in r.text
    assert client.get("/api/cluster").status_code == 404  # router not mounted


def test_service_page_and_api_with_registry(monkeypatch):
    client = _client(monkeypatch, "service/cluster_registry.example.yaml")
    r = client.get("/service")
    assert r.status_code == 200 and "Recommend" in r.text
    cluster = client.get("/api/cluster")
    assert cluster.status_code == 200
    assert len(cluster.json()["free_devices"]) == 10  # 8xA40 + 2xA5000
    assert client.get("/api/models").status_code == 200
    # nav link present on other pages too
    assert 'href="/service"' in client.get("/service").text
