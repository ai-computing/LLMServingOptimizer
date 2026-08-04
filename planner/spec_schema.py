"""Input spec (YAML) parsing and validation via pydantic.

The spec is the single user-facing entry point; it is converted internally into a
topology graph, a search space, and an objective. Validation catches missing
fields, unknown device/model profiles, and invalid metric names *before* any
solving or simulation happens.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field, model_validator

from .utils import LEGACY_ROOT, load_model_config, scan_profile_catalog

VALID_OBJECTIVE_METRICS = {"throughput", "toks_per_wh", "power_w"}
VALID_CONSTRAINT_METRICS = {"ttft_ms", "tpot_ms", "itl_p99_ms"}


class ModelSpec(BaseModel):
    name: str
    fp: int = 16


class WorkloadSpec(BaseModel):
    dataset: str
    num_req: int = 100


class DeviceSpec(BaseModel):
    name: str
    count: int = Field(ge=1)
    mem_gb: float = Field(gt=0)


class NodeSpec(BaseModel):
    id: str
    devices: list[DeviceSpec]
    # host-level base power (W) counted once when any device on this node is
    # used (power-min mode). None -> fall back to the power profiles'
    # host_overhead.base_w (max across this node's hardware types).
    host_base_w: Optional[float] = Field(default=None, ge=0)


class LinkSpec(BaseModel):
    src: str
    dst: str
    bandwidth: str  # e.g. "200Gbps"
    latency: str  # e.g. "0.0005ms"


class TopologySpec(BaseModel):
    nodes: list[NodeSpec]
    links: list[LinkSpec] = Field(default_factory=list)
    intra_node_bandwidth: Optional[str] = None
    # optional hierarchical TP fabric (serialized to cluster_config tp_group_shape)
    tp_group_shape: Optional[list[int]] = None


class ProfilesSpec(BaseModel):
    perf_root: str = "llm_profile/perf_models"


class Constraint(BaseModel):
    constraint: Literal["<=", "<", ">=", ">", "=="]
    value: float


class Objective(BaseModel):
    metric: str
    direction: Literal["max", "min"]
    weight: float = 1.0

    @model_validator(mode="after")
    def _check_metric(self):
        if self.metric not in VALID_OBJECTIVE_METRICS:
            raise ValueError(
                f"objective metric '{self.metric}' not in {sorted(VALID_OBJECTIVE_METRICS)}"
            )
        if self.metric == "power_w" and self.direction != "min":
            raise ValueError("objective 'power_w' only supports direction: min")
        return self


class DemandSpec(BaseModel):
    """Service-scale hard constraint (design doc §4.1).

    Either a direct token-rate demand (``toks_per_s``) or a request rate plus a
    length distribution (``req_per_s`` + ``preset``/``len_dist``, converted to
    toks_per_s by the workload synthesizer before solving).
    """
    toks_per_s: Optional[float] = Field(default=None, gt=0)
    req_per_s: Optional[float] = Field(default=None, gt=0)
    preset: Optional[str] = None          # chat | summarize | agentic (service.presets)
    len_dist: Optional[dict] = None       # {input_mean, output_mean, ...} explicit
    # calibration: tokens/s produced by 1.0 unit of the Stage-1 throughput proxy
    # (A6000 tp1 == 1.0 unit). None -> milp_solver._PROXY_TOKS_PER_UNIT.
    proxy_toks_per_unit: Optional[float] = Field(default=None, gt=0)
    # Stage-2 demand attainment (see objective.check_demand_attainment).
    # How far the demonstrated peak generation rate may fall below toks_per_s
    # before the candidate is failed; >= 1.0 disables that test.
    attainment_tolerance: float = Field(default=0.10, ge=0)
    # Queue-wait p95 (ms) above which requests are judged to be backing up.
    # 0 disables that test.
    max_queue_ms: float = Field(default=1000.0, ge=0)

    @model_validator(mode="after")
    def _check(self):
        if self.toks_per_s is None and self.req_per_s is None:
            raise ValueError("demand needs toks_per_s or req_per_s")
        if self.toks_per_s is None and self.preset is None and self.len_dist is None:
            raise ValueError(
                "demand.req_per_s needs a length distribution (preset or len_dist) "
                "to be convertible to toks_per_s"
            )
        return self

    def resolve_toks_per_s(self) -> float:
        """Fill ``toks_per_s`` from req_per_s x E[in+out] when not given directly.

        ``len_dist`` accepts explicit means ({input_mean, output_mean}); presets
        resolve through the service-layer synthesizer (lazy import — the planner
        core stays usable without the service package for direct toks_per_s specs).
        """
        if self.toks_per_s is not None:
            return self.toks_per_s
        if self.len_dist is not None:
            per_req = float(self.len_dist.get("input_mean", 0)) + \
                float(self.len_dist.get("output_mean", 0))
            if per_req <= 0:
                raise ValueError("demand.len_dist needs positive input_mean/output_mean")
            self.toks_per_s = self.req_per_s * per_req
            return self.toks_per_s
        from service.workload_synth import demand_toks_per_s  # lazy service import
        self.toks_per_s = demand_toks_per_s(self.req_per_s, preset=self.preset)
        return self.toks_per_s


class Requirements(BaseModel):
    # hard constraints (optional individually)
    ttft_ms: Optional[Constraint] = None
    tpot_ms: Optional[Constraint] = None
    itl_p99_ms: Optional[Constraint] = None
    demand: Optional[DemandSpec] = None
    objectives: list[Objective] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_power_min(self):
        if any(o.metric == "power_w" for o in self.objectives) and self.demand is None:
            raise ValueError(
                "objective power_w(min) requires requirements.demand — without a "
                "demand floor the minimum-power solution is degenerate"
            )
        return self


class SearchSpace(BaseModel):
    pd_disaggregation: bool = False
    tp_choices: list[int] = Field(default_factory=lambda: [1])
    # optional per-hardware TP restriction: {hardware: [tp, ...]}. A hardware
    # listed here only gets templates for those TPs (intersected with
    # tp_choices); unlisted hardware uses tp_choices as before. Lets the
    # service exclude (hw, tp) combos its evaluation backend cannot evaluate
    # (e.g. no measured oracle for A40 tp2).
    hw_tp_choices: Optional[dict[str, list[int]]] = None
    # optional measured capacity per (hardware, tp), in output toks/s for ONE
    # instance: {hardware: {tp: toks_per_s}}. When given it replaces the
    # analytical rel_throughput*tp proxy for that combo, which is blind to
    # quantization and to TP's sublinear payoff on PCIe. Same units as
    # requirements.demand.toks_per_s (a generation rate).
    hw_tp_capacity: Optional[dict[str, dict[int, float]]] = None
    pp_choices: list[int] = Field(default_factory=lambda: [1])  # PP not a config knob
    xpyd_prefill_range: list[int] = Field(default_factory=lambda: [1, 1])
    xpyd_decode_range: list[int] = Field(default_factory=lambda: [1, 1])
    batch_tokens_choices: list[int] = Field(default_factory=lambda: [2048])

    @model_validator(mode="after")
    def _check(self):
        if any(t < 1 for t in self.tp_choices):
            raise ValueError("tp_choices must be >= 1")
        if self.hw_tp_choices is not None:
            for hw, tps in self.hw_tp_choices.items():
                if not tps or any(t < 1 for t in tps):
                    raise ValueError(f"hw_tp_choices[{hw!r}] must be non-empty, all >= 1")
        if self.hw_tp_capacity is not None:
            for hw, per_tp in self.hw_tp_capacity.items():
                for tp, cap in per_tp.items():
                    if tp < 1 or cap <= 0:
                        raise ValueError(
                            f"hw_tp_capacity[{hw!r}][{tp}]={cap} must have tp >= 1 "
                            "and a positive toks/s")
        if self.pp_choices != [1]:
            # PP is not exposed as a cluster_config knob (npu_group <= npu_num only).
            raise ValueError(
                "pp_choices must be [1]: pipeline parallelism is not a cluster_config knob "
                "in this simulator version (see PLAN_MILP_MaxFlow.md §5.3)."
            )
        for rng, label in (
            (self.xpyd_prefill_range, "xpyd_prefill_range"),
            (self.xpyd_decode_range, "xpyd_decode_range"),
        ):
            if len(rng) != 2 or rng[0] < 0 or rng[1] < rng[0]:
                raise ValueError(f"{label} must be [lo, hi] with 0 <= lo <= hi")
        return self


class SolverSpec(BaseModel):
    top_k: int = 8
    time_limit_sec: int = 120
    pareto_epsilon_steps: int = 5


class PlannerSpec(BaseModel):
    model: ModelSpec
    workload: WorkloadSpec
    topology: TopologySpec
    profiles: ProfilesSpec = Field(default_factory=ProfilesSpec)
    requirements: Requirements = Field(default_factory=Requirements)
    search_space: SearchSpace = Field(default_factory=SearchSpace)
    solver: SolverSpec = Field(default_factory=SolverSpec)
    # Simulator backend for Stage-2 evaluation: "legacy" (fork) or "upstream".
    # Stage-1 MILP is backend-agnostic; profile validation below is
    # legacy-format only.
    backend: str = "legacy"

    # ---- cross-field validation against the actual repo (profiles/model config) ----
    def validate_against_repo(self) -> list[str]:
        """Return a list of human-readable problems (empty = OK).

        Checks: (1) model config exists, (2) each (hardware, model, tp) combo in
        the search space has a complete profile.
        """
        problems: list[str] = []
        try:
            load_model_config(self.model.name)
        except FileNotFoundError as e:
            problems.append(str(e))

        # relative perf_root is anchored at the legacy backend root (the profile
        # layout documented in ProfilesSpec), not the caller's CWD
        perf_root = Path(self.profiles.perf_root)
        if not perf_root.is_absolute():
            perf_root = LEGACY_ROOT / perf_root
        catalog = scan_profile_catalog(perf_root)
        hardwares = {d.name for n in self.topology.nodes for d in n.devices}
        for hw in sorted(hardwares):
            key = (hw, self.model.name)
            if key not in catalog:
                problems.append(
                    f"no complete profile for hardware '{hw}' + model "
                    f"'{self.model.name}' under {self.profiles.perf_root}"
                )
                continue
            available_tps = catalog[key]
            missing = [t for t in self.search_space.tp_choices if t not in available_tps]
            if missing:
                problems.append(
                    f"hardware '{hw}': missing profiles for tp={missing} "
                    f"(available: {sorted(available_tps)})"
                )
        return problems


def load_spec(path: str | Path) -> PlannerSpec:
    """Parse and structurally validate a spec YAML (does not touch the repo)."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return PlannerSpec.model_validate(data)
