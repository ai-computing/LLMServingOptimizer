"""Combination Generator — ResourcePool → list[CandidateConfig].

Approach (B) per PLAN_webapp_dse_detail.md §0 Q1: each hardware in the pool
becomes a single instance_group with npu_count = chosen count. The existing
`webapp.enumerate.enumerate_configs` handles parallelism (TP/PP/DP) + P/D
role assignment within each hw-count combination.

Pipeline:
  1. Cartesian product over hw counts in [min, max]
  2. Prune: zero-count hw skipped, total_max_npus respected
  3. Pre-filter: aggregate NPU memory ≥ model weight footprint
  4. For each surviving count combination, call enumerate_configs
  5. Wrap each ConfigSpec as CandidateConfig
  6. If total > search.max_combinations, sample (random with seed)
"""
from __future__ import annotations

import random
from dataclasses import replace as _dc_replace
from itertools import product
from pathlib import Path
from typing import Any

import yaml

from webapp.cluster_builder import ConfigSpec
from webapp.enumerate import enumerate_configs

from .schemas import CandidateConfig, JobSpec
from .stage1_filters import apply_stage1_filters, check_candidate


# Path to the metadata catalog (docs/dse/03_catalog.yaml)
_CATALOG_YAML = Path(__file__).resolve().parents[3] / "docs" / "dse" / "03_catalog.yaml"


def load_metadata() -> dict[str, Any]:
    """Load 03_catalog.yaml (hardware + models metadata + availability)."""
    with open(_CATALOG_YAML) as f:
        return yaml.safe_load(f)


def _aggregate_npu_mem_gb(hw_counts: dict[str, int], hw_meta: dict[str, Any]) -> float:
    """Total NPU memory across all NPUs in the candidate."""
    return sum(
        cnt * hw_meta[hw]["mem_size_gb"]
        for hw, cnt in hw_counts.items()
        if hw in hw_meta and cnt > 0
    )


def _coarse_memory_prune(hw_counts: dict[str, int], model_meta: dict[str, Any],
                         hw_meta: dict[str, Any], fp: int) -> bool:
    """Coarse aggregate memory pre-filter: skip if total NPU HBM < model weight.

    This is a fast early-out before enumerate_configs (which is slower).
    It does NOT enforce per-NPU shard fit — that stricter check runs in
    stage1_filters.filter_memory after parallelism is known.
    """
    weight_key = f"weight_size_fp{fp}_gb"
    weight_gb = model_meta.get(weight_key)
    if weight_gb is None:
        weight_gb = model_meta.get("params_b", 0) * (fp / 8)
    return _aggregate_npu_mem_gb(hw_counts, hw_meta) >= weight_gb


def _enumerate_hw_counts(spec: JobSpec) -> list[dict[str, int]]:
    """Cartesian product over each hw's [min, max] range."""
    items = spec.resource_pool.items
    if not items:
        return []
    ranges = [(item.hw, list(range(item.min, item.max + 1))) for item in items]
    out: list[dict[str, int]] = []
    for combo in product(*[r[1] for r in ranges]):
        d = {ranges[i][0]: combo[i] for i in range(len(ranges))}
        total = sum(d.values())
        if total == 0:
            continue  # at least one NPU required
        if spec.resource_pool.total_max_npus is not None and total > spec.resource_pool.total_max_npus:
            continue
        out.append(d)
    return out


def _hw_counts_to_instance_groups(
    hw_counts: dict[str, int], model_name: str, allow_pd: bool,
) -> list[dict]:
    """Approach (B): one instance_group per hardware with npu_count = total count.

    enumerate_configs then explores parallelism (TP/PP/DP) within each group.
    pd_role='auto' lets enumerate decide combined / prefill / decode.
    """
    return [
        {
            "hardware": hw,
            "model": model_name,
            "npu_count": cnt,
            "pd_role": "auto",
        }
        for hw, cnt in hw_counts.items()
        if cnt > 0
    ]


def _config_spec_to_candidate(
    spec_cs: ConfigSpec,
    candidate_id: str,
    hw_counts: dict[str, int],
) -> CandidateConfig:
    """Wrap a ConfigSpec with DSE-level metadata.

    For single-hardware scenarios, enumerate_configs() produces parallelism-only
    labels (e.g. tp1_pp1_dp1) that collide across different hardware types — so
    H100×1 and A6000×1 would both produce tp1_pp1_dp1 and the dedup would drop
    all but the first.  Prefix the label with the hw distribution so each
    hardware variant gets a unique identity.

    Multi-hardware scenarios already produce unique het_* labels from
    enumerate_configs, so those are left unchanged.
    """
    hw_distribution = {hw: cnt for hw, cnt in hw_counts.items() if cnt > 0}

    if not spec_cs.label.startswith("het_"):
        # Build prefix sorted for determinism: "A6000x2", "H100x1_A6000x1", etc.
        hw_prefix = "_".join(f"{hw}x{cnt}" for hw, cnt in sorted(hw_distribution.items()))
        spec_cs = _dc_replace(spec_cs, label=f"{hw_prefix}_{spec_cs.label}")

    return CandidateConfig(
        candidate_id=candidate_id,
        config_spec=spec_cs,
        hw_distribution=hw_distribution,
        parallelism={"tp": spec_cs.tp, "pp": spec_cs.pp, "dp": spec_cs.dp},
        pd_layout=spec_cs.pd_layout,
        label=spec_cs.label,
    )


def generate_candidates(
    spec: JobSpec,
    catalog: dict[tuple[str, str], frozenset[int]],
    metadata: dict[str, Any] | None = None,
    exclude_labels: set[str] | None = None,
    override_max: int | None = None,
) -> list[CandidateConfig]:
    """Generate all valid candidates for the given spec.

    Steps:
      1. cartesian over hw counts
      2. memory pre-filter
      3. enumerate_configs per surviving hw_counts
      4. wrap as CandidateConfig
      5. exclude already-tried labels (retry mode)
      6. sample if over cap (override_max or spec.search.max_combinations)

    Args:
        spec: JobSpec from API/YAML
        catalog: build_catalog() output (real perf_models scan)
        metadata: optional 03_catalog.yaml dict; loaded on demand
        exclude_labels: labels to skip (retry mode — already simulated)
        override_max: sampling cap override; use spec.search.max_combinations
            when None. Pass the number of desired replacements in retry mode.
    """
    if metadata is None:
        metadata = load_metadata()
    hw_meta = metadata["hardware"]
    model_meta = metadata["models"].get(spec.model.name, {})

    all_candidates: list[CandidateConfig] = []
    counter = 0

    for hw_counts in _enumerate_hw_counts(spec):
        # Skip if pre-filters reject
        if not _coarse_memory_prune(hw_counts, model_meta, hw_meta, spec.model.fp):
            continue

        # Build scenario for enumerate_configs
        scenario = {
            "instance_groups": _hw_counts_to_instance_groups(
                hw_counts, spec.model.name, spec.features.allow_pd_disagg,
            ),
            "axes": {
                "vary_tp": True,
                "vary_pp": True,
                "vary_dp": True,
                "include_pd": spec.features.allow_pd_disagg,
            },
        }

        # Enumerate parallelism + P/D combinations
        total_available = sum(hw_counts.values())
        try:
            specs = enumerate_configs(scenario, catalog)
        except Exception:
            # enumerate_configs raises on invalid scenario (e.g. unknown hw).
            # Skip silently — generator pre-filters should have caught it.
            continue

        # Only keep configs that use ALL available NPUs in this hw_count bucket.
        # enumerate_configs also generates under-utilised sub-configs
        # (e.g. budget=2, dp=1, npu_num=1 → 1 NPU used).  In DSE the
        # resource_pool min/max range already handles "try 1 card" vs "try 2
        # cards" via separate hw_count iterations, so each iteration must
        # produce only configs that fill its entire budget — otherwise
        # RNGDx2_tp1_pp1_dp1 and RNGDx1_tp1_pp1_dp1 become identical runs.
        specs = [cs for cs in specs
                 if sum(inst.npu_num for inst in cs.instances) == total_available]

        for cs in specs:
            counter += 1
            cid = f"c{counter:04d}"
            all_candidates.append(_config_spec_to_candidate(cs, cid, hw_counts))

    # Deduplicate by label — different hw_counts can enumerate identical
    # ConfigSpec labels (e.g. tp1_pp1_dp1 appears for A6000 x1, x2, and x4).
    # status.json uses label as dict key and cluster JSONs use label as filename,
    # so duplicates would silently overwrite each other.  Keep first occurrence.
    seen: set[str] = set()
    deduped: list[CandidateConfig] = []
    for c in all_candidates:
        if c.label not in seen:
            seen.add(c.label)
            deduped.append(c)
    all_candidates = deduped

    # Stage 1: analytical pre-filter — prune before sampling so the budget is
    # spent on physically feasible candidates only.
    if spec.search.use_stage1:
        all_candidates, _s1_rejections = apply_stage1_filters(all_candidates, spec, metadata)

    # Stage 2: profile-based pre-ranking — sort survivors by predicted TTFT so
    # that when sampling trims the list the best-predicted candidates are kept.
    if spec.search.use_stage2 and all_candidates:
        try:
            from .stage2_predictor import apply_stage2_predictions
            preds, _ = apply_stage2_predictions(all_candidates, spec, metadata)
            pred_rank = {p["candidate_id"]: i for i, p in enumerate(preds)}
            all_candidates.sort(
                key=lambda c: pred_rank.get(c.candidate_id, len(all_candidates))
            )
        except Exception:
            pass  # Stage 2 failure is non-fatal; keep original ordering

    # Remove already-tried labels (retry rounds)
    if exclude_labels:
        all_candidates = [c for c in all_candidates if c.label not in exclude_labels]

    # Sample if we exceeded the cap.
    # dry_run_detail() uses the same _sample() call with the same seed, so the
    # preview shown to the user (will_simulate=True) matches what actually runs.
    cap = override_max if override_max is not None else spec.search.max_combinations
    if len(all_candidates) > cap:
        all_candidates = _sample(
            all_candidates, cap,
            strategy=spec.search.sampling_strategy,
            seed=spec.search.random_seed,
        )

    return all_candidates


def _sample(
    candidates: list[CandidateConfig], k: int, strategy: str, seed: int,
) -> list[CandidateConfig]:
    """Sub-sample candidates to fit the search budget."""
    if strategy == "random":
        rng = random.Random(seed)
        return rng.sample(candidates, k)
    elif strategy == "grid":
        # Simple grid: take every (n/k)-th candidate after stable sort
        step = max(1, len(candidates) // k)
        return [candidates[i] for i in range(0, len(candidates), step)][:k]
    else:
        # Unknown strategy → fall back to random
        rng = random.Random(seed)
        return rng.sample(candidates, k)


def dry_run_detail(
    spec: JobSpec,
    catalog: dict,
) -> tuple[int, int, list[dict]]:
    """Estimate candidate counts and return per-candidate slim metadata.

    Generates all Stage-1-filtered candidates (no sampling cap), then marks
    which ones would actually be selected for simulation.

    Returns:
        (unique_count, simulated_count, candidate_details)
        candidate_details: list sorted by label, each entry:
          {label, hw_distribution, parallelism, pd_layout, will_simulate}
    """
    metadata = load_metadata()
    # Generate ALL stage-1-filtered candidates by using a very large cap.
    all_cands = generate_candidates(spec, catalog, metadata, override_max=10_000)
    unique_count = len(all_cands)
    simulated_count = min(unique_count, spec.search.max_combinations)

    # Determine which labels will survive sampling (same seed → deterministic).
    if unique_count <= spec.search.max_combinations:
        simulated_labels: set[str] = {c.label for c in all_cands}
    else:
        sampled = _sample(
            all_cands, spec.search.max_combinations,
            strategy=spec.search.sampling_strategy,
            seed=spec.search.random_seed,
        )
        simulated_labels = {c.label for c in sampled}

    details = [
        {
            "label": c.label,
            "hw_distribution": c.hw_distribution,
            "parallelism": c.parallelism,
            "pd_layout": c.pd_layout,
            "will_simulate": c.label in simulated_labels,
        }
        for c in all_cands
    ]
    return unique_count, simulated_count, details
