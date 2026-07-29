"""M5 tests: event simulator physics — conservation, M/D/1 queueing analytics,
saturation behaviour, weighted routing, power integration, envelope warnings.

All tests use synthetic in-memory oracles; no hardware, no simulator builds.
"""
from __future__ import annotations

import random

import pytest

from sim_backends.measured.event_sim import (
    ClusterModel,
    EventSimulator,
    HostCfg,
    IdleDeviceCfg,
    InstanceCfg,
    WorkloadReq,
)
from sim_backends.measured.oracles.base import Envelope, OracleMeta, SteadyPoint
from sim_backends.measured.oracles.capacity_curve import CapacityCurveOracle

NS = 1_000_000_000


def flat_oracle(ttft_ms=100.0, tpot_ms=10.0, max_c=64, active_w=200.0,
                idle_w=50.0, hw="X", model="m", tp=1):
    """Concurrency-independent oracle: service time is deterministic."""
    pts = [
        SteadyPoint(concurrency=1, thr_toks_s=1000 / tpot_ms, ttft_ms=ttft_ms,
                    tpot_ms=tpot_ms, itl_p99_ms=tpot_ms, avg_w=active_w),
        SteadyPoint(concurrency=max_c, thr_toks_s=max_c * 1000 / tpot_ms,
                    ttft_ms=ttft_ms, tpot_ms=tpot_ms, itl_p99_ms=tpot_ms,
                    avg_w=active_w),
    ]
    return CapacityCurveOracle(hw=hw, model=model, tp=tp, points=pts,
                               idle_w=idle_w, meta=OracleMeta(stack="synthetic"))


def poisson_workload(rate_per_s, n, in_toks=100, out_toks=10, seed=1):
    rng = random.Random(seed)
    t, reqs = 0.0, []
    for i in range(n):
        t += rng.expovariate(rate_per_s)
        reqs.append(WorkloadReq(rid=i, input_toks=in_toks, output_toks=out_toks,
                                arrival_ns=int(t * NS)))
    return reqs


# ---------------------------------------------------------------------------
def test_conservation_all_requests_complete():
    cluster = ClusterModel(instances=[
        InstanceCfg(inst_id=0, node_id="n0", oracle=flat_oracle())])
    wl = poisson_workload(rate_per_s=20, n=200, seed=3)
    out = EventSimulator(cluster).run(wl)
    assert out.summary.num_requests == 200
    assert {r.request_id for r in out.records} == set(range(200))
    for r in out.records:
        assert r.ttft_ns <= r.latency_ns
        assert r.end_time_ns == r.arrival_ns + r.latency_ns
        assert r.queuing_delay_ns >= 0


def test_md1_mean_wait_matches_analytic():
    """Single server (admit_cap=1), deterministic service D, Poisson arrivals:
    M/D/1 mean queueing delay Wq = rho/(2*mu*(1-rho)); simulated mean must be
    within 15%."""
    ttft_ms, tpot_ms, out_toks = 40.0, 5.0, 9        # D = 40 + 8*5 = 80 ms
    D = (ttft_ms + (out_toks - 1) * tpot_ms) / 1000  # 0.08 s
    mu = 1 / D                                       # 12.5 /s
    lam = 0.7 * mu
    rho = lam / mu
    wq_analytic = rho / (2 * mu * (1 - rho))          # seconds

    cluster = ClusterModel(instances=[InstanceCfg(
        inst_id=0, node_id="n0",
        oracle=flat_oracle(ttft_ms=ttft_ms, tpot_ms=tpot_ms), admit_cap=1)])
    wl = poisson_workload(rate_per_s=lam, n=4000, out_toks=out_toks, seed=7)
    out = EventSimulator(cluster).run(wl)
    wq_sim = sum(r.queuing_delay_ns for r in out.records) / len(out.records) / NS
    assert wq_sim == pytest.approx(wq_analytic, rel=0.15)


def test_saturation_queue_grows_and_latency_explodes():
    """lambda > mu: queue length grows monotonically and later arrivals wait
    far longer — the SLO violation must be visible."""
    cluster = ClusterModel(instances=[InstanceCfg(
        inst_id=0, node_id="n0",
        oracle=flat_oracle(ttft_ms=50, tpot_ms=10), admit_cap=4)])
    # service ~= 140 ms/req at cap 4 -> capacity ~28 req/s; drive at 60 req/s
    wl = poisson_workload(rate_per_s=60, n=600, out_toks=10, seed=11)
    out = EventSimulator(cluster).run(wl)
    assert out.summary.max_queue_len > 100
    recs = sorted(out.records, key=lambda r: r.arrival_ns)
    early = sum(r.ttft_ns for r in recs[:100]) / 100
    late = sum(r.ttft_ns for r in recs[-100:]) / 100
    assert late > 5 * early


def test_weighted_routing_beats_rr_with_heterogeneous_instances():
    """fast (2x) + slow instance: 2:1 flow-weighted routing yields lower p99
    latency than blind RR (miniature of the experiment-D straggler effect)."""
    def cluster(routing, w_fast, w_slow):
        return ClusterModel(routing=routing, instances=[
            InstanceCfg(inst_id=0, node_id="n0", weight=w_fast, admit_cap=4,
                        oracle=flat_oracle(ttft_ms=50, tpot_ms=5)),
            InstanceCfg(inst_id=1, node_id="n0", weight=w_slow, admit_cap=4,
                        oracle=flat_oracle(ttft_ms=100, tpot_ms=10)),
        ])
    wl = poisson_workload(rate_per_s=45, n=800, out_toks=10, seed=13)

    def p99(out):
        lat = sorted(r.latency_ns for r in out.records)
        return lat[int(0.99 * (len(lat) - 1))]

    p99_rr = p99(EventSimulator(cluster("RR", 1, 1)).run(wl))
    p99_w = p99(EventSimulator(cluster("WEIGHTED", 2, 1)).run(wl))
    assert p99_w < p99_rr


def test_energy_integration_hand_computed():
    """One request on instance A; instance B idle; host base power. Energy =
    P_active*T + idle_B*T + host*T where T is the request's completion time."""
    active_w, idle_w, base_w = 200.0, 50.0, 100.0
    ttft_ms, tpot_ms, out = 100.0, 10.0, 11          # T = 100 + 10*10 = 200 ms
    cluster = ClusterModel(
        routing="WEIGHTED",
        instances=[
            InstanceCfg(inst_id=0, node_id="n0", weight=1,
                        oracle=flat_oracle(ttft_ms=ttft_ms, tpot_ms=tpot_ms,
                                           active_w=active_w, idle_w=idle_w)),
            InstanceCfg(inst_id=1, node_id="n0", weight=0,   # never routed
                        oracle=flat_oracle(active_w=active_w, idle_w=idle_w)),
        ],
        hosts=[HostCfg(node_id="n0", base_w=base_w)],
    )
    wl = [WorkloadReq(rid=0, input_toks=64, output_toks=out, arrival_ns=0)]
    res = EventSimulator(cluster).run(wl)
    T = 0.2  # seconds
    assert res.summary.duration_s == pytest.approx(T, rel=1e-6)
    expected = active_w * T + idle_w * T + base_w * T
    assert res.summary.energy_j == pytest.approx(expected, rel=1e-6)


def test_idle_devices_add_energy():
    cluster = ClusterModel(
        instances=[InstanceCfg(inst_id=0, node_id="n0",
                               oracle=flat_oracle(ttft_ms=100, tpot_ms=10,
                                                  active_w=200, idle_w=50))],
        idle_devices=[IdleDeviceCfg(hw="X", count=3, idle_w=40.0)],
    )
    wl = [WorkloadReq(rid=0, input_toks=64, output_toks=11, arrival_ns=0)]
    res = EventSimulator(cluster).run(wl)
    assert res.summary.energy_j == pytest.approx((200 + 3 * 40) * 0.2, rel=1e-6)


def test_envelope_violation_warns():
    oracle = CapacityCurveOracle(
        hw="X", model="m", tp=1, idle_w=10,
        points=[SteadyPoint(concurrency=1, thr_toks_s=100, ttft_ms=50,
                            tpot_ms=10, avg_w=100),
                SteadyPoint(concurrency=4, thr_toks_s=300, ttft_ms=80,
                            tpot_ms=13, avg_w=130)])
    assert oracle.envelope == Envelope(max_concurrency=4)
    cluster = ClusterModel(instances=[
        InstanceCfg(inst_id=0, node_id="n0", oracle=oracle, admit_cap=8)])
    wl = poisson_workload(rate_per_s=200, n=30, out_toks=5, seed=5)
    with pytest.warns(Warning, match="above measured range"):
        EventSimulator(cluster).run(wl)


def test_pd_disaggregation_kv_transfer_delay():
    """P/D split: decode start is delayed by the KV transfer; end-to-end
    latency exceeds the no-split equivalent by exactly that delay."""
    in_toks, out_toks = 1000, 11
    kv_gbps, kv_per_tok = 1.0, 1e6   # 1000*1e6 B / 1 GB/s = 1.0 s transfer
    common = dict(ttft_ms=100, tpot_ms=10)
    split = ClusterModel(
        kv_link_gbps=kv_gbps, kv_bytes_per_token=kv_per_tok,
        instances=[
            InstanceCfg(inst_id=0, node_id="n0", role="prefill",
                        oracle=flat_oracle(**common)),
            InstanceCfg(inst_id=1, node_id="n1", role="decode",
                        oracle=flat_oracle(**common)),
        ])
    combined = ClusterModel(instances=[
        InstanceCfg(inst_id=0, node_id="n0", oracle=flat_oracle(**common))])
    wl = [WorkloadReq(rid=0, input_toks=in_toks, output_toks=out_toks, arrival_ns=0)]
    lat_split = EventSimulator(split).run(wl).records[0].latency_ns
    lat_comb = EventSimulator(combined).run(wl).records[0].latency_ns
    assert (lat_split - lat_comb) / NS == pytest.approx(1.0, rel=0.01)
