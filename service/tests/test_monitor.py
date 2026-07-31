"""D3 tests: prometheus parser + histogram p95 (hand-computed), rate
derivation, SLO 3-consecutive-violation semantics, energy integration, and a
fixture vLLM /metrics server polled by the collector."""
from __future__ import annotations

import pytest

from service.monitor.collector import (
    Collector,
    histogram_quantile,
    parse_prometheus,
)
from service.monitor.slo_checker import SLOChecker
from service.monitor.store import DeploymentBuffers

FIXTURE = """\
# HELP vllm:num_requests_running Number of requests currently running
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{model_name="m"} 3.0
vllm:num_requests_waiting{model_name="m"} 5.0
vllm:gpu_cache_usage_perc{model_name="m"} 0.42
vllm:prompt_tokens_total{model_name="m"} 1000.0
vllm:generation_tokens_total{model_name="m"} 2000.0
vllm:time_to_first_token_seconds_bucket{le="0.1",model_name="m"} 60.0
vllm:time_to_first_token_seconds_bucket{le="0.5",model_name="m"} 90.0
vllm:time_to_first_token_seconds_bucket{le="+Inf",model_name="m"} 100.0
vllm:time_per_output_token_seconds_bucket{le="0.05",model_name="m"} 80.0
vllm:time_per_output_token_seconds_bucket{le="0.1",model_name="m"} 100.0
vllm:time_per_output_token_seconds_bucket{le="+Inf",model_name="m"} 100.0
"""


def test_parser_and_hand_computed_p95():
    s = parse_prometheus(FIXTURE)
    assert s[("vllm:num_requests_running", (("model_name", "m"),))] == 3.0
    # ttft p95: target 95 of 100; bucket 0.1->60, 0.5->90, +Inf->100.
    # 95 falls in the +Inf bucket -> clamp to previous edge 0.5s
    assert histogram_quantile(s, "vllm:time_to_first_token_seconds", 0.95) == \
        pytest.approx(0.5)
    # ttft p50: target 50 within [0, 0.1] bucket: 0.1 * 50/60
    assert histogram_quantile(s, "vllm:time_to_first_token_seconds", 0.50) == \
        pytest.approx(0.1 * 50 / 60)
    # tpot p95: target 95 in (0.05, 0.1]: 0.05 + (95-80)/20 * 0.05 = 0.0875
    assert histogram_quantile(s, "vllm:time_per_output_token_seconds", 0.95) == \
        pytest.approx(0.0875)


def test_collector_sample_and_rates():
    c = Collector(dep_id="dep-1")
    m1 = c.sample(FIXTURE, ts=100.0)
    assert (m1.running, m1.waiting) == (3, 5)
    assert m1.kv_cache_usage == 0.42
    assert m1.gen_toks_per_s == 0.0            # no previous scrape yet
    assert m1.tpot_p95_ms == pytest.approx(87.5)
    assert not m1.unknown_layout
    text2 = FIXTURE.replace('generation_tokens_total{model_name="m"} 2000.0',
                            'generation_tokens_total{model_name="m"} 2500.0')
    m2 = c.sample(text2, ts=105.0)
    assert m2.gen_toks_per_s == pytest.approx(100.0)   # 500 toks / 5 s


def test_unknown_metric_layout_degrades_not_crashes():
    m = Collector(dep_id="d").sample("some_other_metric 1.0", ts=1.0)
    assert m.unknown_layout and m.running == 0


def _sample(ts, tpot_p95):
    from service.monitor.collector import MetricSample
    return MetricSample(ts=ts, dep_id="d", tpot_p95_ms=tpot_p95)


def test_slo_three_consecutive_violations_trigger_degraded_once():
    events = []
    chk = SLOChecker(dep_id="d", targets={"tpot_ms": 100},
                     on_degraded=lambda d: events.append("deg"),
                     on_recovered=lambda d: events.append("rec"))
    assert chk.push(_sample(1, 150)).verdict == "warn"
    assert chk.push(_sample(2, 150)).verdict == "warn"     # 2 in a row: still warn
    assert chk.push(_sample(3, 150)).verdict == "violated"  # 3rd -> degraded
    assert chk.push(_sample(4, 150)).verdict == "violated"  # stays, no re-fire
    assert events == ["deg"]
    # recovery: window must slide past the bad samples (window_s=30)
    assert chk.push(_sample(40, 50)).verdict == "ok"
    assert events == ["deg", "rec"]


def test_slo_two_violations_then_ok_never_degrades():
    events = []
    chk = SLOChecker(dep_id="d", targets={"tpot_ms": 100},
                     on_degraded=lambda d: events.append("deg"))
    chk.push(_sample(1, 150)); chk.push(_sample(2, 150))
    assert chk.push(_sample(40, 50)).verdict == "ok"
    assert events == []


def test_energy_integration_hand_computed():
    """Constant 200 W for 600 s -> 120000 J = 33.333 Wh (trapezoid exact)."""
    buf = DeploymentBuffers("d")
    for i in range(0, 601, 5):
        buf.push_power(float(i), 200.0)
    assert buf.energy_wh == pytest.approx(200 * 600 / 3600, rel=1e-9)
    s = buf.summary()
    assert s["avg_power_w"] == pytest.approx(200.0)
    assert s["span_s"] == 600.0


def test_collector_polls_fixture_vllm_server():
    """End-to-end against a live fake /metrics HTTP server (no vLLM)."""
    import http.server
    import threading
    import urllib.request

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            body = FIXTURE.encode()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{srv.server_port}/metrics"
        text = urllib.request.urlopen(url, timeout=5).read().decode()
        m = Collector(dep_id="d").sample(text)
        assert m.running == 3 and m.tpot_p95_ms == pytest.approx(87.5)
    finally:
        srv.shutdown()


def test_tpot_alias_drift_request_time_variant():
    """vLLM (this host's image) renamed the TPOT histogram to
    request_time_per_output_token_seconds — the alias table must catch it."""
    text = FIXTURE.replace("vllm:time_per_output_token_seconds",
                           "vllm:request_time_per_output_token_seconds")
    m = Collector(dep_id="d").sample(text, ts=1.0)
    assert m.tpot_p95_ms == pytest.approx(87.5)
