"""Cluster config file I/O for the web UI: list / load / save under cluster_config/."""
from __future__ import annotations

import json
import re
from pathlib import Path

from .config import CLUSTER_CONFIG_DIR

WEB_SUBDIR = "web"
_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")


def list_configs() -> list[dict]:
    """Return [{path, name, web}, ...] sorted, paths relative to repo root."""
    out: list[dict] = []
    if not CLUSTER_CONFIG_DIR.is_dir():
        return out
    for p in sorted(CLUSTER_CONFIG_DIR.rglob("*.json")):
        rel = p.relative_to(CLUSTER_CONFIG_DIR.parent)
        out.append({
            "path": str(rel).replace("\\", "/"),
            "name": p.stem,
            "web": p.parent.name == WEB_SUBDIR,
        })
    return out


def load_config(rel_path: str) -> dict:
    """Load a cluster_config-relative file; reject path traversal."""
    safe = Path(rel_path)
    if safe.is_absolute() or ".." in safe.parts or safe.parts[:1] != ("cluster_config",):
        raise ValueError(f"invalid path: {rel_path!r}")
    target = (CLUSTER_CONFIG_DIR.parent / safe).resolve()
    base = CLUSTER_CONFIG_DIR.resolve()
    if base not in target.parents and target != base:
        raise ValueError(f"path escapes cluster_config: {rel_path!r}")
    return json.loads(target.read_text())


def sanitize_filename(name: str) -> str:
    name = (name or "").strip()
    if name.endswith(".json"):
        name = name[:-5]
    if not _NAME_RE.fullmatch(name):
        raise ValueError("filename must match [A-Za-z0-9._-]+")
    if name in ("", ".", ".."):
        raise ValueError("invalid filename")
    return name


def save_config(name: str, cluster_json: dict) -> str:
    """Write cluster_config/web/<name>.json atomically. Returns the rel path."""
    safe = sanitize_filename(name)
    web_dir = CLUSTER_CONFIG_DIR / WEB_SUBDIR
    web_dir.mkdir(parents=True, exist_ok=True)
    target = web_dir / f"{safe}.json"
    tmp = web_dir / f".tmp.{safe}.json"
    tmp.write_text(json.dumps(cluster_json, indent=2))
    tmp.replace(target)
    return f"cluster_config/{WEB_SUBDIR}/{safe}.json"


def delete_config(rel_path: str) -> None:
    """Delete a cluster_config/web/*.json file.

    Restricted to the `web/` subdirectory so pre-existing reference configs
    in cluster_config/sim_matrix/, cluster_config/*.json (top-level), etc.
    can never be removed through the UI. Rejects path traversal.
    """
    safe = Path(rel_path)
    if safe.is_absolute() or ".." in safe.parts:
        raise ValueError(f"invalid path: {rel_path!r}")
    if safe.parts[:2] != ("cluster_config", WEB_SUBDIR):
        raise ValueError(f"only files under cluster_config/{WEB_SUBDIR}/ are deletable")
    target = (CLUSTER_CONFIG_DIR.parent / safe).resolve()
    web_base = (CLUSTER_CONFIG_DIR / WEB_SUBDIR).resolve()
    if web_base not in target.parents:
        raise ValueError(f"path escapes cluster_config/{WEB_SUBDIR}/: {rel_path!r}")
    if not target.is_file():
        raise FileNotFoundError(f"not found: {rel_path}")
    target.unlink()
