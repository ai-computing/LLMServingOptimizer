"""M6 tests: LENS bucket-oracle math — paper-formula reproduction by hand
calculation, batch expansion rules, and explicit infeasibility."""
from __future__ import annotations

import pytest

from sim_backends.measured.oracles.npu_bucket import (
    Bucket,
    InfeasibleError,
    NpuBucketOracle,
)


@pytest.fixture()
def oracle():
    return NpuBucketOracle(
        hw="RNGD", model="m", tp=1, idle_w=15,
        buckets=[Bucket(size=4096, ttft_ms=100, tbt_ms=10, avg_w=180),
                 Bucket(size=8192, ttft_ms=120, tbt_ms=14, avg_w=200)])


def test_lens_formula_hand_calculation(oracle):
    """l_in=3000, l_out=2000: decode token j runs at kv=3000+j, crossing the
    4096 boundary after j=1096. T = 100 + 1096*10 + 903*14 (1999 decode toks)."""
    t = oracle.request_latency_ms(3000, 2000)
    in_bucket1 = 4096 - 3000            # 1096 decode tokens at 10 ms
    in_bucket2 = (2000 - 1) - in_bucket1  # remaining 903 at 14 ms
    assert t == pytest.approx(100 + in_bucket1 * 10 + in_bucket2 * 14)


def test_single_bucket_no_split(oracle):
    # stays entirely inside bucket 4096: T = 100 + (out-1)*10
    assert oracle.request_latency_ms(1000, 501) == pytest.approx(100 + 500 * 10)


def test_ttft_uses_input_bucket(oracle):
    assert oracle.ttft_ms(3000) == 100
    assert oracle.ttft_ms(5000) == 120


def test_batch_decode_dominated_by_max_kv(oracle):
    # batch with kv positions in both buckets -> max position's bucket wins
    assert oracle.batch_decode_tbt_ms([1200, 3900, 6000]) == 14
    assert oracle.batch_decode_tbt_ms([1200, 3900]) == 10


def test_batch_prefill_additive(oracle):
    assert oracle.batch_prefill_ms([1000, 5000]) == pytest.approx(100 + 120)


def test_input_beyond_max_bucket_infeasible(oracle):
    with pytest.raises(InfeasibleError):
        oracle.ttft_ms(9000)
    with pytest.raises(InfeasibleError):
        oracle.request_latency_ms(8000, 500)  # decode crosses past 8192


def test_yaml_roundtrip(tmp_path):
    import yaml
    p = tmp_path / "tp1.yaml"
    p.write_text(yaml.safe_dump({
        "kind": "npu_bucket", "hw": "RNGD", "model": "m", "tp": 1,
        "idle_w": 15, "max_concurrency": 8,
        "buckets": [{"size": 4096, "ttft_ms": 100, "tbt_ms": 10, "avg_w": 180}],
        "meta": {"stack": "furiosa-sdk-1.x"}}))
    o = NpuBucketOracle.from_yaml(p)
    assert o.envelope.max_concurrency == 8
    assert o.meta.stack == "furiosa-sdk-1.x"
    assert o.request_latency_ms(100, 11) == pytest.approx(100 + 10 * 10)


def test_instance_oracle_protocol_compat(oracle):
    """Bucket oracle plugs into the event simulator via steady_state."""
    from sim_backends.measured.oracles.base import InstanceOracle
    assert isinstance(oracle, InstanceOracle)
    sp = oracle.steady_state(4)
    assert sp.tpot_ms > 0 and sp.thr_toks_s > 0
    assert oracle.power_w(0.0) == 15
    assert oracle.power_w(1.0) == 200
