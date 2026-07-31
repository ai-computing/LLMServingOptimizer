"""D4 Playwright E2E (ui marker): full service-tab flow on the fixture stack
(fake docker driver + fake /metrics, real planner with the measured backend).

submit -> stepper -> BEST card -> confirm modal -> deployment detail
auto-transition -> KPI updates -> terminate modal -> RELEASED -> list.
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.ui

REPO_ROOT = Path(__file__).resolve().parents[2]


def _playwright_available() -> bool:
    try:
        from playwright.sync_api import sync_playwright  # noqa: F401
        return True
    except Exception:
        return False


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _start_server(tmp, registry_path):
    port = _free_port()
    env = dict(os.environ,
               LLMSS_CLUSTER_REGISTRY=str(registry_path),
               LLMSS_SERVICE_DB=str(tmp / "svc.sqlite"),
               LLMSS_FAKE_DOCKER="1")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "webapp.app:app",
         "--port", str(port), "--host", "127.0.0.1"],
        cwd=str(REPO_ROOT), env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}"
    import urllib.request
    for _ in range(60):
        try:
            urllib.request.urlopen(base + "/service/cluster", timeout=1)
            break
        except Exception:
            time.sleep(0.5)
    else:
        proc.kill()
        raise TimeoutError("uvicorn did not come up")
    return base, proc


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("ui")
    registry = tmp / "registry.yaml"
    registry.write_text(
        "nodes:\n"
        "  - id: node0\n"
        "    host_base_w: 250\n"
        "    devices:\n"
        "      - {name: A40, count: 4, mem_gb: 48}\n")
    base, proc = _start_server(tmp, registry)
    yield base
    proc.terminate()
    proc.wait(timeout=10)


@pytest.fixture(scope="module")
def big_server(tmp_path_factory):
    """Service on the shipped 9-node heterogeneous example (66 devices)."""
    tmp = tmp_path_factory.mktemp("ui_big")
    base, proc = _start_server(
        tmp, REPO_ROOT / "service" / "cluster_registry.v2.example.yaml")
    yield base
    proc.terminate()
    proc.wait(timeout=10)


@pytest.mark.skipif(not _playwright_available(), reason="playwright not installed")
def test_full_flow_submit_confirm_deploy_terminate(server):
    from playwright.sync_api import expect, sync_playwright

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page()

        # -- cluster view renders the topology graph -------------------------
        page.goto(server + "/service/cluster")
        expect(page.locator("#svc-graph .dev-rect")).to_have_count(4, timeout=15000)
        expect(page.locator("#cv-summary")).to_contain_text("4")

        # -- submit a request --------------------------------------------------
        page.goto(server + "/service/request")
        page.wait_for_function(
            "document.getElementById('rq-model').options.length > 0")
        page.select_option("#rq-model", "meta-llama/Llama-3.1-8B")
        page.fill("#rq-rate", "2")
        page.fill("#rq-tpot", "200")
        page.click("details > summary")        # expand 고급 options
        page.fill("#rq-neval", "20")
        page.click("#rq-submit")

        # BEST card appears (measured backend: seconds)
        expect(page.locator("#rq-result")).to_be_visible(timeout=120000)
        expect(page.locator(".best-power")).to_contain_text("W")

        # -- confirm modal -> auto-deploy -> dashboard ---------------------------
        page.click("#rq-confirm")
        expect(page.locator(".svc-modal")).to_contain_text("스냅샷")
        page.click('.svc-modal [data-act="ok"]')
        page.wait_for_url("**/service/deployments#dep-*", timeout=30000)

        # detail auto-opens; deployment reaches READY and KPIs update via SSE
        expect(page.locator("#dd-state")).to_have_text("READY", timeout=30000)
        expect(page.locator("#k-thr")).not_to_have_text("—", timeout=30000)
        expect(page.locator("#dd-endpoint")).to_contain_text("http://")
        # events tab shows the lifecycle transitions
        page.click('[data-tab="events"]')
        expect(page.locator("#dd-events")).to_contain_text("HEALTH_CHECK → READY")

        # -- terminate flow ----------------------------------------------------------
        page.click("#dd-terminate")
        modal = page.locator(".svc-modal")
        expect(modal).to_contain_text("배포 종료")
        assert page.locator('.svc-modal [data-act="ok"]').is_disabled()
        page.check("#tm-ck")
        page.click('.svc-modal [data-act="ok"]')
        expect(page.locator("#dd-state")).to_have_text("RELEASED", timeout=30000)

        # back to the list: terminated deployment hidden by default
        page.click("#dd-back")
        expect(page.locator("#dp-table")).not_to_contain_text("dep-", timeout=10000)
        page.check("#dp-incl-term")
        expect(page.locator("#dp-table")).to_contain_text("RELEASED", timeout=10000)

        # cluster resources restored
        page.goto(server + "/service/cluster")
        page.wait_for_function(
            "fetch('/api/cluster').then(r=>r.json()).then(d=>d.free_devices.length===4)")
        browser.close()


@pytest.mark.skipif(not _playwright_available(), reason="playwright not installed")
def test_infeasible_shows_red_card(server):
    from playwright.sync_api import expect, sync_playwright

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page()
        page.goto(server + "/service/request")
        page.wait_for_function(
            "document.getElementById('rq-model').options.length > 0")
        page.fill("#rq-rate", "9999")            # demand far beyond capacity
        page.click("details > summary")
        page.fill("#rq-neval", "10")
        page.click("#rq-submit")
        card = page.locator(".infeasible-card")
        expect(card).to_be_visible(timeout=120000)
        expect(card).to_contain_text("달성 불가")
        expect(card.locator("li").first).to_be_visible()   # 완화 시나리오
        browser.close()


@pytest.mark.skipif(not _playwright_available(), reason="playwright not installed")
def test_cluster_view_layout_stays_bounded(big_server):
    """Regression: host boxes are fixed-size grid cells, not bounding boxes of
    force-scattered devices. The force layout grew hulls to 763-1369 px tall on
    a 520 px canvas at 66 devices (8 GPUs per host smeared vertically)."""
    from playwright.sync_api import expect, sync_playwright

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.goto(big_server + "/service/cluster")
        expect(page.locator("#svc-graph .dev-rect")).to_have_count(66, timeout=20000)
        page.wait_for_timeout(1200)

        geo = page.evaluate("""() => {
            const svg = document.getElementById('svc-graph');
            const [, , vbW, vbH] = svg.getAttribute('viewBox').split(' ').map(Number);
            const hulls = [...svg.querySelectorAll('.host-hull')].map(h => ({
                w: +h.getAttribute('width'), h: +h.getAttribute('height'),
                x: +h.getAttribute('x'), y: +h.getAttribute('y')}));
            const rects = [...svg.querySelectorAll('.dev-rect')].map(r => ({
                w: +r.getAttribute('width'), h: +r.getAttribute('height')}));
            return {vbW, vbH, hulls, rects,
                    hubs: svg.querySelectorAll('.fabric-hub').length};
        }""")

        # every host box identical and small; nothing taller than the canvas
        heights = {round(h["h"]) for h in geo["hulls"]}
        widths = {round(h["w"]) for h in geo["hulls"]}
        assert len(geo["hulls"]) == 9
        assert len(heights) == 1 and len(widths) == 1, (widths, heights)
        assert max(heights) < 200, heights
        # boxes stay inside the viewBox (no off-canvas hosts)
        for h in geo["hulls"]:
            assert h["x"] >= 0 and h["y"] >= 0
            assert h["x"] + h["w"] <= geo["vbW"] + 1
            assert h["y"] + h["h"] <= geo["vbH"] + 1
        # device rects stay modest and A40 (300 W) is not wildly bigger than
        # A5000 (230 W): active_w only scales the slot fill
        big = max(r["w"] for r in geo["rects"])
        small = min(r["w"] for r in geo["rects"])
        assert big <= 50 and big / small < 1.35, (small, big)
        # the all-to-all IB mesh is summarized as one labelled hub
        assert geo["hubs"] == 1
        hub = page.locator(".fabric-hub text")
        expect(hub).to_contain_text("full mesh")
        expect(hub).to_contain_text("200Gbps")

        # summary panel: per-hardware rows + host power bars
        summary = page.locator("#cv-summary")
        expect(summary).to_contain_text("A40")
        expect(summary).to_contain_text("64 / 64")
        assert page.locator("#cv-summary .host-bar").count() == 9

        # PCIe mesh is opt-in: switching to 전체 adds links, 없음 removes all
        n_default = page.locator("#svc-graph line").count()
        page.select_option("#cv-link-filter", "all")
        page.wait_for_timeout(800)
        assert page.locator("#svc-graph line").count() > n_default
        page.select_option("#cv-link-filter", "none")
        page.wait_for_timeout(800)
        assert page.locator("#svc-graph line").count() == 0
        browser.close()
