"""Cluster JSON builder for DSE candidates.

Thin wrapper around webapp.cluster_builder.build_cluster_json that:
  - Auto-derives a power template from the catalog (no manual user input)
  - Writes to output/dse_jobs/<job_id>/configs/<candidate_id>.json
  - Validates by running inference_serving.config_builder.build_cluster_config()
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from webapp.cluster_builder import build_cluster_json
from webapp.config import CPU_MEM_DEFAULT, LINK_BW_DEFAULT, LINK_LATENCY_DEFAULT, REPO_ROOT

from .schemas import CandidateConfig


def build_power_template_from_catalog(
    hw_distribution: dict[str, int],
    hw_meta: dict[str, Any],
    enable_power: bool = True,
) -> dict[str, Any] | None:
    """Construct a power block covering every hardware in the candidate.

    Pulls idle/standby/active/standby_duration from 03_catalog.yaml's
    `hardware.<hw>` entries. CPU/DRAM/Link/NIC/Storage use reasonable host
    defaults — the user can override later via UI.
    """
    if not enable_power:
        return None
    if not all(hw in hw_meta for hw in hw_distribution):
        return None  # Missing catalog entry — disable power modeling

    npu_block: dict[str, dict] = {}
    for hw in hw_distribution:
        meta = hw_meta[hw]
        npu_block[hw] = {
            "idle_power":       meta["idle_power_w"],
            "standby_power":    meta["standby_power_w"],
            "active_power":     meta["tdp_w"],
            "standby_duration": meta["standby_duration_s"],
        }

    return {
        "base_node_power": 60,        # typical workstation chassis baseline
        "npu": npu_block,
        "cpu":     {"idle_power": 10, "active_power": 200, "util": 0.15},
        "dram":    {"dimm_size": 32, "idle_power": 2.0, "energy_per_bit": 6.0},
        "link":    {"num_links": 1, "idle_power": 5, "energy_per_bit": 4.0},
        "nic":     {"num_nics": 1, "idle_power": 20},
        "storage": {"num_devices": 2, "idle_power": 5},
    }


def write_candidate_cluster_json(
    candidate: CandidateConfig,
    output_dir: Path,
    hw_meta: dict[str, Any],
    enable_power: bool = True,
    link_bw: int = LINK_BW_DEFAULT,
    link_latency: int = LINK_LATENCY_DEFAULT,
    cpu_mem: dict | None = None,
    interconnect: dict | None = None,
) -> Path:
    """Build the cluster JSON for a single candidate and write to disk.

    Returns the path written. The DSE runner passes this path to main.py
    via --cluster-config.

    `interconnect` (from interconnect.resolve_interconnect) overrides the flat
    link_bw/link_latency with fabric- or catalog-derived values and may carry a
    tp_group_shape for hierarchical topologies. When None, the flat defaults are
    used (legacy behaviour).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    power = build_power_template_from_catalog(
        candidate.hw_distribution, hw_meta, enable_power=enable_power,
    )
    ic = interconnect or {}
    cluster_json = build_cluster_json(
        candidate.config_spec,
        cpu_mem=cpu_mem or CPU_MEM_DEFAULT,
        link_bw=ic.get("link_bw", link_bw),
        link_latency=ic.get("link_latency", link_latency),
        power_template=power,
        tp_group_shape=ic.get("tp_group_shape"),
        collective_overhead=ic.get("collective_overhead"),
    )
    out_path = output_dir / f"{candidate.label}.json"
    out_path.write_text(json.dumps(cluster_json, indent=2))
    return out_path


def dry_run_validate(cluster_json_path: Path) -> tuple[bool, str | None]:
    """Run inference_serving.config_builder.build_cluster_config dry-run.

    Returns (ok, error_message). On success error_message=None.
    The build mutates inputs/system.json — we restore the cwd to repo root
    after the call so the DSE job sees a clean state.
    """
    # build_cluster_config lives in the legacy backend submodule; make it
    # importable, then chdir into ITS astra-sim (it expects that cwd).
    # Defer import to avoid importing astra-sim deps during generator work.
    import sys
    legacy_root = REPO_ROOT / "backends" / "legacy"
    if str(legacy_root) not in sys.path:
        sys.path.insert(0, str(legacy_root))
    from inference_serving.config_builder import build_cluster_config

    cwd = os.getcwd()
    astra_sim = legacy_root / "astra-sim"
    try:
        os.chdir(astra_sim)
        # build_cluster_config wants a path relative to the LEGACY backend
        # root (it prepends "../" after the astra-sim chdir)
        rel = os.path.relpath(cluster_json_path, legacy_root)
        build_cluster_config(str(astra_sim), rel,
                              enable_local_offloading=False,
                              enable_attn_offloading=False)
        return True, None
    except (KeyError, ValueError, FileNotFoundError) as e:
        return False, str(e)
    finally:
        os.chdir(cwd)
