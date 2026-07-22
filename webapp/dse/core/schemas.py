"""Pydantic models for the DSE web tool input/output.

Mirrors the spec in PLAN_webapp_dse_detail.md §9. Used by:
  - cli.py / server routes for input validation
  - generator.py for combination enumeration
  - ranker.py for SLO + scoring
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator


class HwAllocation(BaseModel):
    """One hardware type's allowed count range in the resource pool."""
    hw: str                       # e.g. "H100", "RNGD" — must exist in catalog
    min: int = Field(0, ge=0)
    max: int = Field(..., ge=0)

    @model_validator(mode="after")
    def _check_range(self) -> "HwAllocation":
        if self.max < self.min:
            raise ValueError(f"hw={self.hw!r}: max ({self.max}) < min ({self.min})")
        return self


class ResourcePool(BaseModel):
    items: list[HwAllocation]
    total_max_npus: Optional[int] = Field(None, ge=1,
        description="Optional global cap on sum(count) across all hardware.")

    @model_validator(mode="after")
    def _non_empty(self) -> "ResourcePool":
        if not self.items:
            raise ValueError("resource_pool.items must be non-empty")
        return self


class ModelSpec(BaseModel):
    name: str                     # HuggingFace id, e.g. "meta-llama/Llama-3.1-8B"
    fp: Literal[8, 16, 32] = 16


class WorkloadSpec(BaseModel):
    dataset: Optional[str] = None
    num_req: int = Field(100, ge=1)
    timeout_s: Optional[int] = Field(None, ge=10,
        description="Per-candidate simulation timeout. None → use CONFIG_TIMEOUT_S.")
    avg_prompt_len: int = Field(512, ge=1,
        description="Average prompt length (tokens); used for TTFT roofline filter.")
    max_seq_len: int = Field(2048, ge=1,
        description="Max sequence length (prompt+output); used for KV cache memory check.")


class Constraints(BaseModel):
    """SLO and resource caps. None = unconstrained (skipped in filter)."""
    ttft_p99_ms: Optional[float] = Field(None, ge=0)
    tpot_p99_ms: Optional[float] = Field(None, ge=0)
    itl_p99_ms: Optional[float] = Field(None, ge=0)
    throughput_min_tok_s: Optional[float] = Field(None, ge=0)
    power_max_w: Optional[float] = Field(None, ge=0)
    energy_max_wh: Optional[float] = Field(None, ge=0)
    tokwh_min: Optional[float] = Field(None, ge=0,
        description="Minimum energy efficiency: tokens per Wh (higher = better).")


class FeatureFlags(BaseModel):
    allow_pd_disagg: bool = True
    allow_pim: bool = False
    allow_cxl: bool = False
    prefix_caching: bool = False
    attn_offloading: bool = False
    sub_batch_interleaving: bool = False


class SearchConfig(BaseModel):
    max_combinations: int = Field(20, ge=1)
    sampling_strategy: Literal["random", "grid"] = "random"
    random_seed: int = 0
    use_stage1: bool = Field(True,
        description="Apply Stage 1 analytical filters before simulation.")
    use_stage2: bool = Field(True,
        description="Apply Stage 2 profile-based pre-ranking before sampling.")


class ObjectiveWeights(BaseModel):
    """Weight per objective. Auto-normalized to sum=1 in validator.

    All directions are encoded *here*: latency/power are minimized, throughput
    and tokwh are maximized. The ranker applies the appropriate sign during scoring.
    """
    ttft: float = Field(0.25, ge=0)
    tpot: float = Field(0.25, ge=0)
    throughput: float = Field(0.25, ge=0)
    power: float = Field(0.25, ge=0)
    tokwh: float = Field(0.0, ge=0,
        description="Energy efficiency weight (Tokens/Wh, higher is better).")

    @model_validator(mode="after")
    def _normalize(self) -> "ObjectiveWeights":
        s = self.ttft + self.tpot + self.throughput + self.power + self.tokwh
        if s <= 0:
            raise ValueError("at least one weight must be > 0")
        self.ttft /= s
        self.tpot /= s
        self.throughput /= s
        self.power /= s
        self.tokwh /= s
        return self


class JobSpec(BaseModel):
    """Top-level DSE input. Loaded from YAML/JSON via POST /api/dse/jobs."""
    resource_pool: ResourcePool
    model: ModelSpec
    workload: WorkloadSpec
    fabric: Optional[str] = Field(None,
        description="Interconnect fabric preset name (docs/dse/fabrics.yaml). "
                    "None → per-hardware catalog interconnect_bw_gbs scalar fallback.")
    constraints: Constraints = Field(default_factory=Constraints)
    features: FeatureFlags = Field(default_factory=FeatureFlags)
    search: SearchConfig = Field(default_factory=SearchConfig)
    weights: ObjectiveWeights = Field(default_factory=ObjectiveWeights)
    top_n: int = Field(5, ge=1)
    max_concurrent: Optional[int] = Field(None, ge=1,
        description="Max simultaneous simulations. None → auto from CPU/RAM.")


# ----------------------------------------------------------------------------
# Generator / runner outputs

class CandidateConfig(BaseModel):
    """One enumerated candidate ready for cluster JSON build + simulation."""
    candidate_id: str             # e.g. "c042"
    config_spec: Any              # webapp.cluster_builder.ConfigSpec (dataclass, non-pydantic)
    hw_distribution: dict[str, int]  # {"H100": 2, "A6000": 4}
    parallelism: dict[str, int]   # {"tp": 2, "pp": 1, "dp": 3}
    pd_layout: str                # "—" or "1P+3D"
    label: str                    # ConfigSpec.label (used as filename)

    model_config = {"arbitrary_types_allowed": True}


class SimulationResult(BaseModel):
    """One candidate's outcome after runner.run_dse_job."""
    candidate_id: str
    label: str
    state: Literal["done", "failed", "timeout", "cancelled"]
    elapsed_s: float
    metrics: dict[str, Any] = Field(default_factory=dict)
    cluster_config_path: Optional[str] = None
    raw_csv_path: Optional[str] = None
    log_path: Optional[str] = None
    error: Optional[str] = None
    # populated by ranker:
    meets_slo: bool = True
    on_pareto: bool = False
    score: Optional[float] = None


class RankedResults(BaseModel):
    """Output of ranker.rank_candidates."""
    all_results: list[SimulationResult]
    pareto_indices: list[int]
    top_n_indices: list[int]
    weights_used: ObjectiveWeights
