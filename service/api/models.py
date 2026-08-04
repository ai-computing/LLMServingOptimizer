"""Pydantic request/response schemas for the serving-recommendation API (§4.5)."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class SLOSpec(BaseModel):
    ttft_ms: Optional[float] = Field(default=None, gt=0)
    tpot_ms: Optional[float] = Field(default=None, gt=0)
    itl_p99_ms: Optional[float] = Field(default=None, gt=0)


class ScaleSpecIn(BaseModel):
    req_per_s: float = Field(gt=0)
    preset: Optional[str] = None                    # chat | summarize | agentic
    len_dist: Optional[dict] = None                 # {input_mean, output_mean}
    duration_s: float = Field(default=60.0, gt=0)


class ServeRequestIn(BaseModel):
    model: str
    fp: int = 16
    tenant: str = "default"
    slo: SLOSpec = Field(default_factory=SLOSpec)
    scale: ScaleSpecIn
    power_cap_w: Optional[float] = Field(default=None, gt=0)
    exclude_hw: list[str] = Field(default_factory=list)
    force_backend: Optional[Literal["measured", "upstream", "legacy"]] = None
    num_req_eval: int = Field(default=50, ge=1)     # Stage-2 evaluation depth
    auto_deploy: bool = True                        # confirm -> deployment chain (D2)


class CandidateOut(BaseModel):
    run_id: str
    hw_summary: str
    passed: bool
    power_w: Optional[float] = None
    power_source: Optional[str] = None
    metrics: Optional[dict] = None
    violations: list[str] = Field(default_factory=list)


class RecommendationOut(BaseModel):
    best: Optional[CandidateOut] = None
    alternatives: list[CandidateOut] = Field(default_factory=list)
    backend: str
    confidence: str
    reason: str
    snapshot_ver: int
    device_ids: list[str] = Field(default_factory=list)
    demand_toks_per_s: Optional[float] = None
    # advisory: the run never generated at the demanded rate, but nothing backed
    # up either (objective.demand_shortfall) — not a rejection
    demand_note: Optional[str] = None
    infeasible: Optional[dict] = None               # InfeasibleReport.as_dict()


class JobStatusOut(BaseModel):
    id: str
    tenant: str
    state: Literal["pending", "running", "done", "infeasible", "failed",
                   "confirmed", "released", "replanned"]
    error: Optional[str] = None
    result: Optional[RecommendationOut] = None


class ConfirmOut(BaseModel):
    state: str
    reservation_id: Optional[int] = None
    deployment_id: Optional[str] = None             # set when auto_deploy fired
    device_ids: list[str] = Field(default_factory=list)
    # set when a confirm conflict triggered an automatic re-plan
    replanned: bool = False
    detail: Optional[str] = None
