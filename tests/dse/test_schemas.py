"""Unit tests for webapp.dse.core.schemas — Pydantic validation."""
import pytest
from pydantic import ValidationError

from webapp.dse.core.schemas import (
    HwAllocation,
    JobSpec,
    ModelSpec,
    ObjectiveWeights,
    ResourcePool,
    SearchConfig,
    WorkloadSpec,
)


def test_hw_allocation_max_lt_min_fails():
    with pytest.raises(ValidationError):
        HwAllocation(hw="A6000", min=5, max=2)


def test_resource_pool_empty_fails():
    with pytest.raises(ValidationError):
        ResourcePool(items=[])


def test_objective_weights_auto_normalize():
    w = ObjectiveWeights(ttft=2, tpot=2, throughput=2, power=2)
    # All 2's → each becomes 0.25
    assert w.ttft == 0.25 and w.tpot == 0.25
    assert abs(w.ttft + w.tpot + w.throughput + w.power - 1.0) < 1e-9


def test_objective_weights_zero_sum_fails():
    with pytest.raises(ValidationError):
        ObjectiveWeights(ttft=0, tpot=0, throughput=0, power=0)


def test_objective_weights_skewed():
    w = ObjectiveWeights(ttft=4, tpot=1, throughput=1, power=2)
    assert abs(w.ttft - 0.5) < 1e-9
    assert abs(w.power - 0.25) < 1e-9


def test_search_config_invalid_sampling():
    with pytest.raises(ValidationError):
        SearchConfig(sampling_strategy="bogus")


def test_job_spec_defaults():
    spec = JobSpec(
        resource_pool=ResourcePool(items=[HwAllocation(hw="A6000", max=2)]),
        model=ModelSpec(name="x"),
        workload=WorkloadSpec(dataset="d.jsonl"),
    )
    assert spec.top_n == 5
    assert spec.features.allow_pd_disagg is True
    assert spec.search.max_combinations == 20
    # weights default normalized
    assert abs(sum([spec.weights.ttft, spec.weights.tpot,
                    spec.weights.throughput, spec.weights.power]) - 1.0) < 1e-9


def test_job_spec_round_trip():
    """spec_json → JobSpec → model_dump → JobSpec should be stable."""
    src = {
        "resource_pool": {"items": [{"hw": "H100", "max": 4}]},
        "model": {"name": "meta-llama/Llama-3.1-8B"},
        "workload": {"dataset": "d.jsonl", "num_req": 50},
        "constraints": {"ttft_p99_ms": 500},
        "weights": {"ttft": 1, "tpot": 1, "throughput": 1, "power": 1},
    }
    spec = JobSpec.model_validate(src)
    dumped = spec.model_dump()
    spec2 = JobSpec.model_validate(dumped)
    assert spec2.model_dump() == dumped
