"""Enumerate valid (TP, PP, DP, P/D) configurations for a scenario.

A scenario is a dict from the API:
    {
      "instance_groups": [
          {"hardware": "A6000", "model": "meta-llama/Llama-3.1-8B",
           "npu_count": 4, "pd_role": "auto"}
      ],
      "axes": {"vary_tp": true, "vary_pp": true, "vary_dp": true,
               "include_pd": true},
    }

For a single homogeneous group, this enumerates combined-mode and P/D-mode
parallelism layouts that fit within `npu_count` total physical NPUs.

For multi-group heterogeneous scenarios, each group becomes one cluster
instance and per-group parallelism is enumerated independently and combined
via cartesian product.

The known A6000 x 4 sweep should produce these 11 configs:
  combined: tp1_pp1_dp1, tp2_pp1_dp1, tp1_pp2_dp1, tp2_pp2_dp1, tp1_pp4_dp1,
            tp1_pp1_dp2, tp2_pp1_dp2, tp1_pp2_dp2, tp1_pp1_dp4
  P/D:     pd_1p1d_tp1, pd_1p2d_tp1
The 1P + 1D(tp2/pp2) variants are skipped due to a known ASTRA-Sim topology
crash whenever decode npu_num > 1.
"""
from __future__ import annotations

from itertools import product

from .cluster_builder import ConfigSpec, InstanceSpec
from .hardware_catalog import get_tp_options


def _per_instance_layouts(
    catalog: dict[tuple[str, str], frozenset[int]],
    hardware: str,
    model: str,
    max_npu_num: int,
    vary_tp: bool,
    vary_pp: bool,
) -> list[tuple[int, int]]:
    """Return list of (npu_num, npu_group) layouts that fit within max_npu_num.

    npus_per_group (== npu_num // npu_group) must be in the profiled TP set.
    """
    tp_options = get_tp_options(catalog, hardware, model)
    if not tp_options:
        return []

    tps = tp_options if vary_tp else [min(tp_options)]
    layouts: set[tuple[int, int]] = set()

    for tp in tps:
        # Pipeline range: 1, 2, ..., max_npu_num // tp (so npu_num <= budget).
        max_pp = max_npu_num // tp
        if max_pp < 1:
            continue
        pps = range(1, max_pp + 1) if vary_pp else [1]
        for pp in pps:
            npu_num = tp * pp
            npu_group = pp
            if npu_num <= max_npu_num and npu_group <= npu_num and npu_num % npu_group == 0:
                layouts.add((npu_num, npu_group))

    return sorted(layouts)


def _label_combined(tp: int, pp: int, dp: int) -> str:
    return f"tp{tp}_pp{pp}_dp{dp}"


def _label_pd(num_p: int, num_d: int, tp: int, pp: int) -> str:
    suffix = f"_tp{tp}" if pp == 1 else f"_tp{tp}pp{pp}"
    return f"pd_{num_p}p{num_d}d{suffix}"


def _enum_single_group_combined(
    catalog: dict[tuple[str, str], frozenset[int]],
    hardware: str,
    model: str,
    budget: int,
    vary_tp: bool,
    vary_pp: bool,
    vary_dp: bool,
) -> list[ConfigSpec]:
    """Combined-mode (no P/D split) configs for a single homogeneous group."""
    out: list[ConfigSpec] = []
    seen: set[tuple] = set()

    for npu_num, npu_group in _per_instance_layouts(
        catalog, hardware, model, budget, vary_tp, vary_pp
    ):
        max_dp = budget // npu_num
        if max_dp < 1:
            continue
        dps = range(1, max_dp + 1) if vary_dp else [1]
        for dp in dps:
            phys = dp * npu_num
            if phys > budget:
                continue
            # Restrict to layouts whose total physical NPU count divides the
            # budget cleanly (matches the canonical A6000x4 sweep). This skips
            # awkward leftovers like dp=3 with budget=4.
            if budget % phys != 0:
                continue
            tp = npu_num // npu_group
            pp = npu_group
            key = (npu_num, npu_group, dp, "combined")
            if key in seen:
                continue
            seen.add(key)
            inst = InstanceSpec(
                hardware=hardware,
                model=model,
                npu_num=npu_num,
                npu_group=npu_group,
                pd_type=None,
            )
            out.append(ConfigSpec(
                label=_label_combined(tp, pp, dp),
                instances=[inst] * dp,
                tp=tp, pp=pp, dp=dp,
                pd_layout="—",
            ))
    return out


def _enum_single_group_pd(
    catalog: dict[tuple[str, str], frozenset[int]],
    hardware: str,
    model: str,
    budget: int,
    vary_tp: bool,
    vary_pp: bool,
) -> list[ConfigSpec]:
    """P/D split-mode configs.

    Prefill instances cost 2 * npu_num physical NPUs (config_builder.py:292-294).
    Decode instances cost npu_num physical NPUs.
    Skip any decode with npu_num > 1 (known ASTRA-Sim topology crash).
    """
    out: list[ConfigSpec] = []
    seen: set[tuple] = set()

    layouts = _per_instance_layouts(catalog, hardware, model, budget, vary_tp, vary_pp)
    if not layouts:
        return []

    for (p_npu, p_grp) in layouts:
        # Prefill costs 1× physical NPU per npu_num — same as decode/combined.
        # ASTRA-Sim's 2× doubling (in config_builder.py) adds virtual sender
        # NPUs for KV-transfer modeling and does not consume real hardware.
        # Reference: cluster_config/sim_matrix/a6000_2_pd.json (budget=2, 1P+1D
        # each npu=1) ran successfully in the reference codebase.
        prefill_cost = p_npu
        if prefill_cost > budget:
            continue
        # Prefill PP > 1 (p_grp > 1) causes a topology mismatch in
        # _create_network_config: total_npu = 2*p_npu + d_npu is not
        # divisible by effective_num_instances (3 for 1P+1D), so ASTRA-Sim
        # receives a truncated NPU count and deadlocks.
        if p_grp > 1:
            continue
        for (d_npu, d_grp) in layouts:
            # Known crash: P/D with decode npu_num > 1.
            if d_npu > 1:
                continue
            # Enumerate counts: 1P + ND, NP + 1D, ensuring fit.
            # 1 prefill + N decode
            num_p = 1
            remaining = budget - prefill_cost
            max_nd = remaining // d_npu if d_npu > 0 else 0
            for num_d in range(1, max_nd + 1):
                total = num_p * prefill_cost + num_d * d_npu
                if total > budget:
                    continue
                tp = p_npu // p_grp
                pp = p_grp
                key = (p_npu, p_grp, d_npu, d_grp, num_p, num_d, "pd")
                if key in seen:
                    continue
                seen.add(key)
                instances: list[InstanceSpec] = []
                for _ in range(num_p):
                    instances.append(InstanceSpec(
                        hardware=hardware, model=model,
                        npu_num=p_npu, npu_group=p_grp,
                        pd_type="prefill",
                    ))
                for _ in range(num_d):
                    instances.append(InstanceSpec(
                        hardware=hardware, model=model,
                        npu_num=d_npu, npu_group=d_grp,
                        pd_type="decode",
                    ))
                # Topology validity: ASTRA-Sim drops NPUs silently when the
                # grid math doesn't divide cleanly. Now that prefill_cost is
                # 1× (not 2×), more configs fit the budget — but some still
                # break topology (e.g. tp2 prefill, Np+1D with N>1).
                if not _topology_valid(instances):
                    continue
                out.append(ConfigSpec(
                    label=_label_pd(num_p, num_d, tp, pp),
                    instances=instances,
                    tp=tp, pp=pp, dp=num_d,
                    pd_layout=f"{num_p}P+{num_d}D",
                ))

            # N prefill + 1 decode (skip num_p==1 which we already covered above)
            for num_p in range(2, (budget // prefill_cost) + 1):
                num_d = 1
                total = num_p * prefill_cost + num_d * d_npu
                if total > budget:
                    continue
                tp = p_npu // p_grp
                pp = p_grp
                key = (p_npu, p_grp, d_npu, d_grp, num_p, num_d, "pd")
                if key in seen:
                    continue
                seen.add(key)
                instances = []
                for _ in range(num_p):
                    instances.append(InstanceSpec(
                        hardware=hardware, model=model,
                        npu_num=p_npu, npu_group=p_grp,
                        pd_type="prefill",
                    ))
                for _ in range(num_d):
                    instances.append(InstanceSpec(
                        hardware=hardware, model=model,
                        npu_num=d_npu, npu_group=d_grp,
                        pd_type="decode",
                    ))
                if not _topology_valid(instances):
                    continue
                out.append(ConfigSpec(
                    label=_label_pd(num_p, num_d, tp, pp),
                    instances=instances,
                    tp=tp, pp=pp, dp=num_d,
                    pd_layout=f"{num_p}P+{num_d}D",
                ))

    return out


def _enum_single_group(
    catalog: dict[tuple[str, str], frozenset[int]],
    group: dict,
    axes: dict,
) -> list[ConfigSpec]:
    """Enumerate configs for one instance group."""
    hardware = group["hardware"]
    model = group["model"]
    budget = int(group["npu_count"])
    pd_role = group.get("pd_role", "auto")

    vary_tp = bool(axes.get("vary_tp", True))
    vary_pp = bool(axes.get("vary_pp", True))
    vary_dp = bool(axes.get("vary_dp", True))
    include_pd = bool(axes.get("include_pd", True))

    configs: list[ConfigSpec] = []

    if pd_role in ("auto", "combined", None):
        configs.extend(_enum_single_group_combined(
            catalog, hardware, model, budget, vary_tp, vary_pp, vary_dp,
        ))

    if include_pd and pd_role in ("auto", "prefill", "decode", "pd"):
        configs.extend(_enum_single_group_pd(
            catalog, hardware, model, budget, vary_tp, vary_pp,
        ))

    return configs


def _per_group_layouts_with_role(
    catalog: dict[tuple[str, str], frozenset[int]],
    group: dict,
    axes: dict,
) -> list[InstanceSpec]:
    """For a multi-group scenario, enumerate per-group layout candidates.

    Returns a list of InstanceSpec candidates, one per (npu_num, npu_group)
    layout that fits the group's budget. The pd_type comes from group.pd_role.
    """
    hardware = group["hardware"]
    model = group["model"]
    budget = int(group["npu_count"])
    pd_role = group.get("pd_role", "auto")
    vary_tp = bool(axes.get("vary_tp", True))
    vary_pp = bool(axes.get("vary_pp", True))

    if pd_role == "prefill":
        # In multi-group mode each group is an independent physical instance,
        # so the full budget is available regardless of pd_type.
        # (The 2x ASTRA-Sim topology factor is handled by config_builder.py.)
        max_npu = budget
        pd_type = "prefill"
    elif pd_role == "decode":
        max_npu = budget
        pd_type = "decode"
    else:
        # "auto", "combined", None -> no P/D
        max_npu = budget
        pd_type = None

    candidates: list[InstanceSpec] = []
    for (npu_num, npu_group) in _per_instance_layouts(
        catalog, hardware, model, max_npu, vary_tp, vary_pp
    ):
        # Decode npu_num > 1 known crash.
        if pd_type == "decode" and npu_num > 1:
            continue
        # Prefill PP > 1 in multi-group P/D produces misleading "success":
        # ASTRA-Sim silently drops the extra PP-stage NPUs (no warning, no
        # exception) and reports clocks identical to the PP=1 baseline.
        # We saw this with het_p1x4_H100__d1x1_RNGD vs het_p1x1_H100__d1x1_RNGD
        # — same total clocks down to 4 significant digits. Block these to
        # avoid presenting fake metrics. Single-group P/D is already filtered
        # for the same reason in _enum_single_group_pd.
        if pd_type == "prefill" and npu_group > 1:
            continue
        candidates.append(InstanceSpec(
            hardware=hardware, model=model,
            npu_num=npu_num, npu_group=npu_group,
            pd_type=pd_type,
        ))
    return candidates


def _label_multi(instances: list[InstanceSpec]) -> str:
    parts = []
    for inst in instances:
        tp = inst.npu_num // inst.npu_group
        pp = inst.npu_group
        role = inst.pd_type or "c"
        prefix = {"prefill": "p", "decode": "d", "c": "c"}[role]
        parts.append(f"{prefix}{tp}x{pp}_{inst.hardware}")
    return "het_" + "__".join(parts)


def _topology_valid(instances: list[InstanceSpec]) -> bool:
    """Check whether the ASTRA-Sim topology math divides cleanly.

    Mirrors _create_network_config in inference_serving/config_builder.py:
    if total NPUs don't divide evenly into the chosen topology dimensions,
    NPUs get silently dropped via integer division and ASTRA-Sim deadlocks
    waiting for collectives that will never complete.

    The actual constraint behind "no heterogeneous P/D" is this divisibility,
    not the hardware-type mismatch. Heterogeneous P/D with npu_num=1 on every
    instance (e.g., 1×A6000 prefill + 1×RNGD decode in one node) works.
    """
    total_npu = sum(
        i.npu_num * 2 if i.pd_type == "prefill" else i.npu_num
        for i in instances
    )
    total_grp = sum(
        i.npu_group * 2 if i.pd_type == "prefill" else i.npu_group
        for i in instances
    )
    prefill_count = sum(1 for i in instances if i.pd_type == "prefill")
    eff_num_inst = len(instances) + prefill_count

    if total_npu == total_grp:
        # full-pipeline path: npus_per_group = total_npu // eff_num_inst
        return eff_num_inst > 0 and total_npu % eff_num_inst == 0
    # mixed path: npus_per_group = total_npu // total_grp
    return total_grp > 0 and total_npu % total_grp == 0


def _enum_multi_group(
    catalog: dict[tuple[str, str], frozenset[int]],
    groups: list[dict],
    axes: dict,
) -> list[ConfigSpec]:
    """Cartesian product across groups -- each group becomes one cluster instance.

    For groups with pd_role="auto" and include_pd=True, also tries assigning
    prefill/decode roles across groups so heterogeneous P/D splits are enumerated.
    """
    include_pd = bool(axes.get("include_pd", True))

    def _effective_roles(pd_role: str) -> list[str]:
        """Return the set of roles to try for this group."""
        if pd_role in ("prefill", "decode", "combined"):
            return [pd_role]
        # "auto" or unset: try combined; also try P/D roles when include_pd
        roles = ["combined"]
        if include_pd:
            roles += ["prefill", "decode"]
        return roles

    # Build per-group role lists
    group_role_options = [_effective_roles(g.get("pd_role", "auto")) for g in groups]

    out: list[ConfigSpec] = []
    seen: set[tuple] = set()

    for role_combo in product(*group_role_options):
        has_prefill = "prefill" in role_combo
        has_decode = "decode" in role_combo
        # P/D is only valid if BOTH prefill and decode are present.
        if has_prefill != has_decode:
            continue
        # Heterogeneous-hardware P/D is allowed when the topology math divides
        # cleanly (validated per-combo below via _topology_valid).

        # Heterogeneous combined-mode: every group runs as a standalone
        # instance with different hardware. ASTRA-Sim deadlocks here because
        # its uniform grid topology can't represent two independent compute
        # rates — the instances diverge in simulated time and never meet at
        # the collectives they're expected to share. Empirically all 8 such
        # configs in the 4xH100_4xRNGD sweep timed out.
        all_combined = all(r == "combined" for r in role_combo)
        if all_combined and len({g["hardware"] for g in groups}) > 1:
            continue

        # Heterogeneous P/D + many combined-mode bridges: ASTRA-Sim's
        # collective routing hangs when the topology is dominated by
        # combined instances of mixed hardware sitting between prefill and
        # decode. Empirical signature from the 4xH100_4xRNGD sweep: configs
        # with `combined_count > prefill_count + decode_count` made 0
        # progress before timing out (no progress lines in 120s). Block to
        # spare users wasted slots; misses ~4/12 borderline timeouts but
        # never blocks a successful run.
        if len({g["hardware"] for g in groups}) > 1:
            n_p = sum(1 for r in role_combo if r == "prefill")
            n_d = sum(1 for r in role_combo if r == "decode")
            n_c = sum(1 for r in role_combo if r == "combined")
            if n_c > 0 and n_p > 0 and n_d > 0 and n_c > n_p + n_d:
                continue

        # Build per-group layout candidates for this role assignment.
        per_group_layouts: list[list[InstanceSpec]] = []
        valid = True
        for g, role in zip(groups, role_combo):
            g_with_role = dict(g, pd_role=role)
            candidates = _per_group_layouts_with_role(catalog, g_with_role, axes)
            if not candidates:
                valid = False
                break
            per_group_layouts.append(candidates)
        if not valid:
            continue

        for combo in product(*per_group_layouts):
            instances = list(combo)
            # Skip combos whose ASTRA-Sim topology would silently drop NPUs
            # (see _create_network_config). This is the real constraint that
            # makes most "heterogeneous P/D" cases fail.
            if not _topology_valid(instances):
                continue
            # Normalize instance order: prefill first, combined/None next,
            # decode last. Reason: config_builder.py assigns NPU IDs in
            # instance-list order, and ASTRA-Sim's grid topology expects
            # prefill NPUs at lower IDs so KV-transfer collectives line up.
            # Putting decode before prefill (e.g. het_d1x1_H100__p1x4_RNGD)
            # would make the lone decode NPU share a topology group with
            # prefill NPUs and deadlock on within-group collectives.
            # Sorted is stable, so per-group order is preserved within roles.
            instances.sort(key=lambda i: (
                0 if i.pd_type == "prefill"
                else 2 if i.pd_type == "decode"
                else 1
            ))
            # Display values use the first instance as representative
            rep = instances[0]
            tp = rep.npu_num // rep.npu_group
            pp = rep.npu_group
            prefill_count = sum(1 for i in instances if i.pd_type == "prefill")
            decode_count = sum(1 for i in instances if i.pd_type == "decode")
            if prefill_count and decode_count:
                pd_layout = f"{prefill_count}P+{decode_count}D"
            else:
                pd_layout = "—"

            key = tuple(
                (i.hardware, i.model, i.npu_num, i.npu_group, i.pd_type)
                for i in instances
            )
            if key in seen:
                continue
            seen.add(key)

            out.append(ConfigSpec(
                label=_label_multi(instances),
                instances=instances,
                tp=tp, pp=pp,
                dp=max(decode_count, 1) if pd_layout != "—" else 1,
            pd_layout=pd_layout,
        ))

    return out


def enumerate_configs(
    scenario: dict,
    catalog: dict[tuple[str, str], frozenset[int]],
) -> list[ConfigSpec]:
    """Return all valid ConfigSpecs for the given scenario.

    Scenario shape:
        {"instance_groups": [...], "axes": {...}}

    For a single group, runs combined + P/D enumerations (homogeneous case).
    For multi-group, takes the cartesian product of per-group layouts.
    Deduplicates by (instance signature) tuple.
    """
    groups = scenario.get("instance_groups", []) or []
    axes = scenario.get("axes", {}) or {}

    if not groups:
        return []

    if len(groups) == 1:
        return _enum_single_group(catalog, groups[0], axes)

    return _enum_multi_group(catalog, groups, axes)
