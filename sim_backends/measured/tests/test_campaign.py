"""M6 tests: campaign runners generate correct oracle YAML from injected
(mock) measurements; dry-run emits complete commands. No hardware needed."""
from __future__ import annotations

import pytest

from sim_backends.measured.campaign.bucket_campaign import (
    BucketCampaignConfig,
    run_campaign as run_bucket_campaign,
)
from sim_backends.measured.campaign.capacity_campaign import (
    CampaignConfig,
    run_campaign as run_capacity_campaign,
)
from sim_backends.measured.oracles.capacity_curve import CapacityCurveOracle
from sim_backends.measured.oracles.npu_bucket import NpuBucketOracle


def test_capacity_campaign_writes_loadable_oracle(tmp_path):
    cfg = CampaignConfig(hw="A5000", model="meta-llama/Llama-3.1-8B", tp=1,
                         concurrencies=[1, 4], stack="vllm-0.19",
                         measured_at="2026-07-29", idle_w=20,
                         out_root=str(tmp_path))

    def fake_measure(c):
        return {"thr_toks_s": 50.0 * c, "ttft_ms": 80 + 10 * c,
                "tpot_ms": 20 + c, "itl_p99_ms": 25 + c, "avg_w": 180 + 5 * c}

    path = run_capacity_campaign(cfg, measure_fn=fake_measure)
    oracle = CapacityCurveOracle.from_yaml(path)
    assert oracle.hw == "A5000" and oracle.tp == 1
    assert oracle.meta.stack == "vllm-0.19"          # freshness anchor recorded
    assert oracle.envelope.max_concurrency == 4
    sp = oracle.steady_state(4)
    assert sp.thr_toks_s == pytest.approx(200.0)
    assert oracle.idle_w() == 20


def test_capacity_campaign_dry_run_commands():
    cfg = CampaignConfig(hw="A5000", model="m", tp=2, concurrencies=[1, 8])
    cmds = run_capacity_campaign(cfg, dry_run=True)
    assert len(cmds) == 2
    for cmd, c in zip(cmds, (1, 8)):
        s = " ".join(cmd)
        assert "--max-concurrency " + str(c) in s
        assert "--model m" in s and "--num-prompts" in s


def test_capacity_campaign_incomplete_measurement_rejected(tmp_path):
    cfg = CampaignConfig(hw="X", model="m", tp=1, concurrencies=[1],
                         out_root=str(tmp_path))
    with pytest.raises(ValueError, match="missing"):
        run_capacity_campaign(cfg, measure_fn=lambda c: {"thr_toks_s": 1.0})


def test_bucket_campaign_writes_loadable_oracle(tmp_path):
    cfg = BucketCampaignConfig(hw="RNGD", model="m", tp=1,
                               bucket_sizes=[4096, 8192], idle_w=15,
                               stack="furiosa-1.2", out_root=str(tmp_path))

    def fake_measure(size):
        return {"ttft_ms": 100 if size == 4096 else 120,
                "tbt_ms": 10 if size == 4096 else 14,
                "avg_w": 180}

    path = run_bucket_campaign(cfg, measure_fn=fake_measure)
    oracle = NpuBucketOracle.from_yaml(path)
    assert oracle.meta.stack == "furiosa-1.2"
    # the LENS hand-calc from test_npu_bucket must hold on the generated file
    assert oracle.request_latency_ms(3000, 2000) == pytest.approx(
        100 + 1096 * 10 + 903 * 14)


def test_bucket_campaign_dry_run_two_runs_per_bucket():
    cfg = BucketCampaignConfig(hw="RNGD", model="m", tp=1,
                               bucket_sizes=[2048, 4096])
    cmds = run_bucket_campaign(cfg, dry_run=True)
    assert len(cmds) == 4  # LENS: 2|B| measurements
    tags = [c[-1] for c in cmds]
    assert tags == ["ttft_b2048", "tbt_b2048", "ttft_b4096", "tbt_b4096"]
