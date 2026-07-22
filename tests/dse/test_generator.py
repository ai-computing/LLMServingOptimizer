"""Unit tests for webapp.dse.core.generator."""
import pytest

from webapp.dse.core.generator import (
    _aggregate_npu_mem_gb,
    _coarse_memory_prune as _meets_memory,
    _enumerate_hw_counts,
    _hw_counts_to_instance_groups,
    _sample,
    dry_run_detail,
    generate_candidates,
    load_metadata,
)
from webapp.dse.core.schemas import (
    HwAllocation,
    JobSpec,
    ModelSpec,
    ResourcePool,
    SearchConfig,
    WorkloadSpec,
)
from webapp.hardware_catalog import build_catalog


@pytest.fixture(scope="module")
def catalog():
    return build_catalog()


@pytest.fixture(scope="module")
def metadata():
    return load_metadata()


# ------- helpers (no I/O) ----------------------------------------------------

def test_enumerate_hw_counts_single_dim():
    spec = JobSpec(
        resource_pool=ResourcePool(items=[HwAllocation(hw="A6000", min=1, max=3)]),
        model=ModelSpec(name="meta-llama/Llama-3.1-8B"),
        workload=WorkloadSpec(dataset="x.jsonl"),
    )
    out = _enumerate_hw_counts(spec)
    assert {c["A6000"] for c in out} == {1, 2, 3}


def test_enumerate_hw_counts_skips_all_zero():
    spec = JobSpec(
        resource_pool=ResourcePool(items=[
            HwAllocation(hw="A6000", min=0, max=1),
            HwAllocation(hw="RNGD",  min=0, max=1),
        ]),
        model=ModelSpec(name="meta-llama/Llama-3.1-8B"),
        workload=WorkloadSpec(dataset="x.jsonl"),
    )
    out = _enumerate_hw_counts(spec)
    # 2x2=4 combos, minus (0,0). 3 remain.
    assert len(out) == 3
    assert {(c["A6000"], c["RNGD"]) for c in out} == {(0, 1), (1, 0), (1, 1)}


def test_enumerate_hw_counts_total_cap():
    spec = JobSpec(
        resource_pool=ResourcePool(
            items=[
                HwAllocation(hw="A6000", min=0, max=4),
                HwAllocation(hw="RNGD",  min=0, max=4),
            ],
            total_max_npus=4,
        ),
        model=ModelSpec(name="meta-llama/Llama-3.1-8B"),
        workload=WorkloadSpec(dataset="x.jsonl"),
    )
    out = _enumerate_hw_counts(spec)
    for combo in out:
        assert sum(combo.values()) <= 4
        assert sum(combo.values()) >= 1


def test_aggregate_npu_mem(metadata):
    hw_meta = metadata["hardware"]
    assert _aggregate_npu_mem_gb({"A6000": 2}, hw_meta) == 80
    assert _aggregate_npu_mem_gb({"A6000": 1, "RNGD": 1}, hw_meta) == 80
    assert _aggregate_npu_mem_gb({"Unknown": 5}, hw_meta) == 0  # unknown ignored


def test_meets_memory_fits(metadata):
    hw_meta = metadata["hardware"]
    model_meta = metadata["models"]["meta-llama/Llama-3.1-8B"]
    # 8B fp16 = 16GB, single A6000 = 40GB → fits
    assert _meets_memory({"A6000": 1}, model_meta, hw_meta, fp=16) is True


def test_meets_memory_too_small(metadata):
    hw_meta = metadata["hardware"]
    model_meta = metadata["models"]["meta-llama/Llama-3.1-70B"]
    # 70B fp16 = 141GB, single A6000 = 40GB → doesn't fit
    assert _meets_memory({"A6000": 1}, model_meta, hw_meta, fp=16) is False
    # 4 H100 80GB = 320GB → fits 141GB
    assert _meets_memory({"H100": 4}, model_meta, hw_meta, fp=16) is True


def test_instance_groups_format():
    g = _hw_counts_to_instance_groups({"A6000": 2, "RNGD": 0, "H100": 1},
                                      "meta-llama/Llama-3.1-8B", allow_pd=True)
    # Zero count entries dropped, others get one group each
    assert {x["hardware"] for x in g} == {"A6000", "H100"}
    a6000 = next(x for x in g if x["hardware"] == "A6000")
    assert a6000["npu_count"] == 2
    assert a6000["pd_role"] == "auto"
    assert a6000["model"] == "meta-llama/Llama-3.1-8B"


def test_sample_random_reproducible():
    cands = list(range(100))  # placeholder ints
    s1 = _sample(cands, 10, "random", seed=42)
    s2 = _sample(cands, 10, "random", seed=42)
    assert s1 == s2
    assert len(s1) == 10


def test_sample_grid_stride():
    cands = list(range(20))
    s = _sample(cands, 5, "grid", seed=0)
    # step = 20//5 = 4, so [0, 4, 8, 12, 16]
    assert s == [0, 4, 8, 12, 16]


# ------- full generator pipeline --------------------------------------------

def test_generate_basic_homogeneous(catalog, metadata):
    spec = JobSpec(
        resource_pool=ResourcePool(items=[HwAllocation(hw="A6000", min=1, max=1)]),
        model=ModelSpec(name="meta-llama/Llama-3.1-8B"),
        workload=WorkloadSpec(dataset="x.jsonl"),
    )
    cands = generate_candidates(spec, catalog, metadata)
    assert len(cands) >= 1
    # Single A6000 should produce at minimum tp1_pp1_dp1 (label is prefixed with hw)
    labels = {c.label for c in cands}
    assert any("tp1_pp1_dp1" in lbl for lbl in labels)


def test_generate_rejects_oversize_model(catalog, metadata):
    spec = JobSpec(
        resource_pool=ResourcePool(items=[HwAllocation(hw="A6000", min=1, max=1)]),
        model=ModelSpec(name="meta-llama/Llama-3.1-70B"),
        workload=WorkloadSpec(dataset="x.jsonl"),
    )
    cands = generate_candidates(spec, catalog, metadata)
    assert cands == []


def test_generate_respects_cap(catalog, metadata):
    spec = JobSpec(
        resource_pool=ResourcePool(items=[
            HwAllocation(hw="A6000", min=1, max=4),
            HwAllocation(hw="RNGD",  min=0, max=2),
        ]),
        model=ModelSpec(name="meta-llama/Llama-3.1-8B"),
        workload=WorkloadSpec(dataset="x.jsonl"),
        search=SearchConfig(max_combinations=5, random_seed=42),
    )
    cands = generate_candidates(spec, catalog, metadata)
    assert len(cands) <= 5


def test_dry_run_consistency(catalog, metadata):
    spec = JobSpec(
        resource_pool=ResourcePool(items=[HwAllocation(hw="A6000", min=1, max=2)]),
        model=ModelSpec(name="meta-llama/Llama-3.1-8B"),
        workload=WorkloadSpec(dataset="x.jsonl"),
    )
    # dry_run should match generate's unfiltered count when cap is huge
    unique_count, _simulated, _details = dry_run_detail(spec, catalog)
    spec_huge = spec.model_copy(update={"search": SearchConfig(max_combinations=10000)})
    cands = generate_candidates(spec_huge, catalog, metadata)
    assert unique_count == len(cands)
