"""Persistent simulation result cache.

Cache key = SHA256[:16](sorted JSON of cluster_json + workload params).
Only successful (state=done) runs are cached; failures and timeouts are not.

Layout:
  output/sim_cache/<16-hex-key>/
    metrics.json   — parsed metrics dict (atomic write)
    output.log     — copy of simulation log
    output.csv     — copy of per-request CSV
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path


def make_key(
    cluster_json: dict,
    dataset: str,
    num_req: int,
    fp: int = 16,
    block_size: int = 16,
) -> str:
    """Return a 16-char hex cache key deterministically derived from inputs."""
    payload = json.dumps(
        {
            "cluster": cluster_json,
            "dataset": str(Path(dataset)),
            "num_req": num_req,
            "fp": fp,
            "block_size": block_size,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def lookup(cache_root: Path, key: str) -> dict | None:
    """Return the cached metrics dict on hit, or None on miss."""
    entry = cache_root / key / "metrics.json"
    if not entry.exists():
        return None
    try:
        return json.loads(entry.read_text())
    except Exception:
        return None


def save(
    cache_root: Path,
    key: str,
    metrics: dict,
    log_src: Path | None,
    csv_src: Path | None,
) -> None:
    """Persist a successful simulation result.

    metrics.json is written atomically (tmp → rename).
    log and csv are copied best-effort; errors are silently ignored.
    """
    entry_dir = cache_root / key
    entry_dir.mkdir(parents=True, exist_ok=True)

    tmp = entry_dir / "metrics.json.tmp"
    tmp.write_text(json.dumps(metrics, indent=2))
    os.replace(tmp, entry_dir / "metrics.json")

    for src, dst_name in ((log_src, "output.log"), (csv_src, "output.csv")):
        if src is not None and Path(src).exists():
            try:
                shutil.copy2(src, entry_dir / dst_name)
            except OSError:
                pass


def get_cached_paths(cache_root: Path, key: str) -> tuple[Path, Path]:
    """Return (log_path, csv_path) inside the cache entry directory."""
    return cache_root / key / "output.log", cache_root / key / "output.csv"
