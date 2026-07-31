"""D4: the webapp service subtab pages render and the service API mounts when
a cluster registry is configured (and degrade gracefully when it is not)."""
from __future__ import annotations

import importlib

import pytest

pytest.importorskip("fastapi")


def _client(monkeypatch, registry_path, tmp_path=None):
    if registry_path is None:
        monkeypatch.setenv("LLMSS_CLUSTER_REGISTRY", "/nonexistent/registry.yaml")
    else:
        monkeypatch.setenv("LLMSS_CLUSTER_REGISTRY", str(registry_path))
    monkeypatch.setenv("LLMSS_FAKE_DOCKER", "1")
    if tmp_path is not None:
        monkeypatch.setenv("LLMSS_SERVICE_DB", str(tmp_path / "svc.sqlite"))
    import webapp.app as W
    importlib.reload(W)
    from fastapi.testclient import TestClient
    return TestClient(W.app)


def test_service_pages_disabled_without_registry(monkeypatch):
    client = _client(monkeypatch, None)
    r = client.get("/service", follow_redirects=True)
    assert r.status_code == 200 and "Service disabled" in r.text
    assert client.get("/api/cluster").status_code == 404  # router not mounted


def test_service_subtab_pages_render_with_registry(monkeypatch, tmp_path):
    client = _client(monkeypatch, "service/cluster_registry.example.yaml", tmp_path)
    r = client.get("/service", follow_redirects=True)
    assert r.status_code == 200 and 'id="svc-graph"' in r.text
    for path, marker in (("/service/cluster", "svc-graph"),
                         ("/service/request", "rq-form"),
                         ("/service/deployments", "dp-table")):
        page = client.get(path)
        assert page.status_code == 200, path
        assert marker in page.text
        assert "service.css" in page.text          # token stylesheet linked
        assert 'class="svc-subtabs"' in page.text  # subtab nav present
    # vendored libs referenced (no CDN)
    assert "vendor/d3.v7.min.js" in client.get("/service/cluster").text
    assert "vendor/chart.umd.min.js" in client.get("/service/deployments").text
    # APIs mounted
    assert client.get("/api/cluster").status_code == 200
    assert client.get("/api/cluster/graph").status_code == 200
    assert client.get("/api/deployments").status_code == 200
