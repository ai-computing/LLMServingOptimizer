"""Configuration constants for the LLMServingSim webapp.

All paths are anchored to the repo root (parent of the webapp/ dir).
SIM_ENV mirrors the env vars used by script/run_a6000_4_sweep.sh so that
the AnalyticalAstra binary can find libprotobuf.so.23 and graph_generator.py
can locate the `python` symlink.
"""
import os
import pathlib

REPO_ROOT = pathlib.Path(__file__).parent.parent

# Simulator backends live under backends/ (git submodules); the adapter
# package sim_backends/ hides their CLI/config differences. Legacy-format
# profile paths below stay module constants because the DSE stage-2
# predictor and hardware catalog read the legacy CSV layout directly.
BACKENDS_DIR = REPO_ROOT / "backends"
LEGACY_ROOT = BACKENDS_DIR / "legacy"
LLM_PROFILE_DIR = LEGACY_ROOT / "llm_profile/perf_models"
CLUSTER_CONFIG_DIR = REPO_ROOT / "cluster_config"
DATASET_DIR = REPO_ROOT / "dataset"
OUTPUT_DIR = REPO_ROOT / "output/web_sweeps"
SIM_CACHE_ROOT = REPO_ROOT / "output" / "sim_cache"
MAIN_PY = LEGACY_ROOT / "main.py"  # legacy entrypoint (kept for back-compat)

# Simulator env -- required for AnalyticalAstra binary and `python` symlink
# Source: script/run_a6000_4_sweep.sh lines 11-13
SIM_ENV = {
    **os.environ,
    "LD_LIBRARY_PATH": "/tmp/protobuf_prefix/usr/lib/x86_64-linux-gnu:" + os.environ.get("LD_LIBRARY_PATH", ""),
    "PATH": os.path.expanduser("~/.local/bin") + ":" + os.environ.get("PATH", ""),
}

CONFIG_TIMEOUT_S = 600  # kill a config run after 10 min

# Dynamic concurrency tuning
MEM_PER_SIM_GB = 2.0   # estimated RAM per simulation subprocess
MEM_RESERVE_GB = 4.0   # headroom to leave for OS + webapp


def _available_mem_gb() -> float:
    """Read MemAvailable from /proc/meminfo; returns 8.0 on failure."""
    try:
        with open("/proc/meminfo") as _f:
            for _line in _f:
                if _line.startswith("MemAvailable:"):
                    return int(_line.split()[1]) / (1024 * 1024)
    except Exception:
        pass
    return 8.0


def compute_max_concurrent() -> int:
    """Return concurrency limit derived from available CPU cores and free RAM.

    cpu_limit  = os.cpu_count()   (one slot per logical core)
    mem_limit  = floor((MemAvailable - MEM_RESERVE_GB) / MEM_PER_SIM_GB)
    result     = max(1, min(cpu_limit, mem_limit))
    """
    cpu_limit = os.cpu_count() or 4
    usable_gb = max(0.0, _available_mem_gb() - MEM_RESERVE_GB)
    mem_limit = max(1, int(usable_gb / MEM_PER_SIM_GB))
    return max(1, min(cpu_limit, mem_limit))


# Static snapshot used for docstrings / comments only.
MAX_CONCURRENT = compute_max_concurrent()
SOFT_CAP = 20           # warn user if sweep exceeds this count

# Default hardware memory specs -- used when building cluster JSONs
# Read from cluster_config/single_node_*.json at startup; these are fallbacks
HW_DEFAULTS = {
    "A40":       {"mem_size": 45,  "mem_bw": 696,  "mem_latency": 0},
    "A6000":     {"mem_size": 40,  "mem_bw": 768,  "mem_latency": 0},
    "H100":      {"mem_size": 80,  "mem_bw": 3000, "mem_latency": 0},
    "RNGD":      {"mem_size": 40,  "mem_bw": 1500, "mem_latency": 0},
    "RTX3090":   {"mem_size": 24,  "mem_bw": 936,  "mem_latency": 0},
    "TPU-v6e-1": {"mem_size": 16,  "mem_bw": 1640, "mem_latency": 0},
    "A5000":     {"mem_size": 24,  "mem_bw": 768,  "mem_latency": 0},
}
CPU_MEM_DEFAULT = {"mem_size": 128, "mem_bw": 256, "mem_latency": 0}
LINK_BW_DEFAULT = 112
LINK_LATENCY_DEFAULT = 0
