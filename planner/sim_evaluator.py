"""Stage 2: run LLMServingSim (``main.py``) on a rendered config and parse metrics.

Robustness contract: any failure (non-zero exit, timeout, crash, empty CSV)
returns an :class:`Infeasible` marker rather than raising, so the orchestrator
can drop the candidate and keep going. Results are cached on disk keyed by the
config content + CLI args.
"""
from __future__ import annotations

import ast
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from .types import Infeasible, Metrics
from .utils import REPO_ROOT, get_logger, hash_obj, stage_path

log = get_logger("planner.sim")

NS_PER_MS = 1e6
NS_PER_S = 1e9
J_PER_WH = 3600.0


def _sim_env() -> dict:
    """Subprocess env for running the simulator.

    Mirrors ``webapp/config.py:SIM_ENV`` (the tested setup) so the planner runs
    the simulator without the caller having to export LD_LIBRARY_PATH/PATH first.
    """
    import os

    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = (
        "/tmp/protobuf_prefix/usr/lib/x86_64-linux-gnu:" + env.get("LD_LIBRARY_PATH", "")
    )
    env["PATH"] = os.path.expanduser("~/.local/bin") + ":" + env.get("PATH", "")
    return env


# ---------------------------------------------------------------------------
# CSV parsing (pure; unit-tested with mock CSVs)
# ---------------------------------------------------------------------------
def _p99(series: pd.Series) -> float:
    return float(series.quantile(0.99)) if len(series) else float("nan")


#: sliding window for the demonstrated generation rate. One request's decode
#: span is seconds at typical TPOT, so a few seconds smooths per-request
#: boundaries without averaging the busy stretch away.
_GEN_WINDOW_S = 5.0


def _peak_generation_rate(df, window_s: float = _GEN_WINDOW_S) -> Optional[float]:
    """Highest sustained generation rate in the run (toks/s).

    Each request generates its ``output`` tokens over the decode span implied by
    its TPOT and ending at ``end_time``; spreading tokens over that span and
    sliding a window over the timeline recovers what the cluster was really
    producing per second. See ``Metrics.peak_gen_toks_s`` for why this and not
    ``throughput_toks_s`` answers "did it keep up with the demand?".
    """
    needed = {"output", "end_time", "TPOT"}
    if not needed.issubset(df.columns) or df.empty or window_s <= 0:
        return None
    buckets: dict[int, float] = {}
    for out_toks, end_ns, tpot_ns in zip(df["output"], df["end_time"], df["TPOT"]):
        try:
            out_toks, end_ns, tpot_ns = float(out_toks), float(end_ns), float(tpot_ns)
        except (TypeError, ValueError):
            continue
        if out_toks <= 0 or not math.isfinite(end_ns):
            continue
        span_ns = max(out_toks * max(tpot_ns, 0.0), float(NS_PER_S) / 1000.0)
        rate = out_toks / (span_ns / NS_PER_S)          # tokens per second
        start_s, end_s = (end_ns - span_ns) / NS_PER_S, end_ns / NS_PER_S
        for sec in range(int(start_s // 1), int(end_s // 1) + 1):
            overlap = min(end_s, sec + 1) - max(start_s, sec)
            if overlap > 0:
                buckets[sec] = buckets.get(sec, 0.0) + rate * overlap
    if not buckets:
        return None
    lo, hi = min(buckets), max(buckets)
    width = max(1, int(window_s))
    best = 0.0
    for start in range(lo, hi + 1):
        tokens = sum(buckets.get(s, 0.0) for s in range(start, start + width))
        best = max(best, tokens / width)
    return best


def parse_metrics_csv(csv_path: str | Path) -> Metrics:
    """Aggregate a per-request output CSV into SLO metrics.

    Expected columns: instance id, request id, model, input, output, arrival,
    end_time, latency, queuing_delay, TTFT, TPOT, ITL. Times are nanoseconds;
    ``ITL`` is a stringified list of per-token intervals.
    """
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"empty results CSV: {csv_path}")

    ttft_ms = float(df["TTFT"].mean()) / NS_PER_MS
    tpot_ms = float(df["TPOT"].mean()) / NS_PER_MS

    # ITL is a list-per-request; flatten all per-token intervals then take p99
    itls: list[float] = []
    for cell in df["ITL"].dropna():
        if isinstance(cell, str):
            try:
                vals = ast.literal_eval(cell)
            except (ValueError, SyntaxError):
                continue
            itls.extend(float(v) for v in vals)
        elif isinstance(cell, (int, float)):
            itls.append(float(cell))
    itl_p99_ms = (float(pd.Series(itls).quantile(0.99)) / NS_PER_MS) if itls else float("nan")

    # throughput = total generated tokens / wall-clock span
    total_out = float(df["output"].sum())
    span_ns = float(df["end_time"].max() - df["arrival"].min())
    throughput = (total_out / (span_ns / NS_PER_S)) if span_ns > 0 else 0.0

    # backlog: how long requests waited for a slot (see Metrics.queue_p95_ms)
    queue_p95 = queue_max = None
    if "queuing_delay" in df.columns:
        q = pd.to_numeric(df["queuing_delay"], errors="coerce").dropna()
        if not q.empty:
            queue_p95 = float(q.quantile(0.95)) / NS_PER_MS
            queue_max = float(q.max()) / NS_PER_MS

    return Metrics(
        ttft_ms=ttft_ms,
        tpot_ms=tpot_ms,
        itl_p99_ms=itl_p99_ms,
        throughput_toks_s=throughput,
        peak_gen_toks_s=_peak_generation_rate(df),
        queue_p95_ms=queue_p95,
        queue_max_ms=queue_max,
        num_requests=len(df),
        raw={"total_output_tokens": total_out, "span_ns": span_ns,
             "arrival_span_ns": float(df["arrival"].max() - df["arrival"].min())},
    )


def _parse_energy_from_stdout(stdout: str) -> Optional[float]:
    """Best-effort scan for the total energy (Joules) in the sim's stdout.

    main.py prints ``Total energy consumption (kJ): <value>`` (only when power
    modeling is configured). We parse that kJ value and convert to Joules. This
    is not a stable contract; see PLAN_MILP_MaxFlow.md §8 for the recommended
    CSV-column patch. Returns None when the line is absent.
    """
    import re

    m = re.search(
        r"Total\s+energy\s+consumption\s*\(kJ\)\s*:\s*([\d.]+)", stdout, re.IGNORECASE
    )
    if m:
        try:
            return float(m.group(1)) * 1000.0  # kJ -> J
        except ValueError:
            pass
    return None


# ---------------------------------------------------------------------------
# Subprocess evaluation
# ---------------------------------------------------------------------------
#: CLI flags whose value is a path (rendered REPO_ROOT-relative) that must be
#: re-anchored to the selected backend's root before the subprocess call.
_PATH_FLAGS = {"--cluster-config", "--dataset", "--output"}

# --- memory guard ----------------------------------------------------------
# A single Stage-2 candidate can blow up: an A40 70B (tp4 x2) upstream run was
# observed growing ~13 GB/min past 73 GB RSS, which would OOM-kill the host
# (the webapp, other candidates, anything). Each simulation therefore runs in
# its own session with an RSS ceiling; exceeding it kills that process tree and
# the candidate becomes Infeasible — the documented robustness contract —
# instead of taking the machine down. Override with LLMSS_SIM_MEM_LIMIT_GB
# (0 disables).
_DEFAULT_SIM_MEM_LIMIT_GB = 16.0
_MEM_POLL_SEC = 2.0


def sim_mem_limit_bytes() -> int:
    import os
    try:
        gb = float(os.environ.get("LLMSS_SIM_MEM_LIMIT_GB",
                                  _DEFAULT_SIM_MEM_LIMIT_GB))
    except ValueError:
        gb = _DEFAULT_SIM_MEM_LIMIT_GB
    return int(gb * (1024 ** 3)) if gb > 0 else 0


def _tree_rss_bytes(pid: int) -> int:
    """Summed RSS of a process and its descendants (Linux /proc)."""
    import os

    page = os.sysconf("SC_PAGE_SIZE")
    total, stack, seen = 0, [pid], set()
    while stack:
        p = stack.pop()
        if p in seen:
            continue
        seen.add(p)
        try:
            with open(f"/proc/{p}/statm") as f:
                total += int(f.read().split()[1]) * page
        except (OSError, IndexError, ValueError):
            continue
        try:                      # children of each thread (CONFIG_PROC_CHILDREN)
            for tid in os.listdir(f"/proc/{p}/task"):
                with open(f"/proc/{p}/task/{tid}/children") as f:
                    stack.extend(int(c) for c in f.read().split())
        except OSError:
            pass
    return total


def _kill_tree(proc) -> None:
    import os
    import signal

    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except OSError:
            try:
                proc.kill()
            except OSError:
                pass
        try:
            proc.wait(timeout=5)
            return
        except Exception:
            continue


def run_guarded(cmd: list[str], cwd: str, env: dict, timeout_sec: int,
                mem_limit_bytes: Optional[int] = None,
                poll_sec: float = _MEM_POLL_SEC):
    """subprocess.run() plus an RSS ceiling on the process tree.

    Returns a CompletedProcess (returncode 137 + a ``stderr`` explanation when
    the memory guard fired); raises TimeoutExpired like subprocess.run. Output
    goes to a temp file, so a chatty simulator can never fill a pipe buffer
    while we are polling.
    """
    import subprocess as sp
    import tempfile
    import time as _time

    limit = sim_mem_limit_bytes() if mem_limit_bytes is None else mem_limit_bytes
    with tempfile.TemporaryFile("w+", encoding="utf-8", errors="replace") as out:
        proc = sp.Popen(cmd, cwd=cwd, env=env, stdout=out, stderr=sp.STDOUT,
                        text=True, start_new_session=True)
        deadline = _time.time() + timeout_sec
        peak = 0
        while True:
            rc = proc.poll()
            if rc is not None:
                break
            if limit:
                rss = _tree_rss_bytes(proc.pid)
                peak = max(peak, rss)
                if rss > limit:
                    _kill_tree(proc)
                    out.seek(0)
                    log.warning("memory guard killed %s (%.1f GiB > %.1f GiB)",
                                cmd[-1], rss / 2 ** 30, limit / 2 ** 30)
                    return sp.CompletedProcess(
                        cmd, 137, out.read(),
                        f"memory guard: RSS {rss / 2 ** 30:.1f} GiB exceeded the "
                        f"{limit / 2 ** 30:.0f} GiB per-simulation limit "
                        f"(LLMSS_SIM_MEM_LIMIT_GB)")
            if _time.time() > deadline:
                _kill_tree(proc)
                raise sp.TimeoutExpired(cmd, timeout_sec)
            _time.sleep(poll_sec)
        out.seek(0)
        text = out.read()
    if peak:
        log.info("simulation peak RSS %.1f GiB", peak / 2 ** 30)
    return sp.CompletedProcess(cmd, rc, text, "")


def _rebase_path_args(args: list[str], backend_root: Path, absolute: bool) -> list[str]:
    """Convert REPO_ROOT-relative path values to what the backend CLI accepts:
    legacy wants paths relative to its root; upstream accepts absolute."""
    import os

    out = list(args)
    for i, tok in enumerate(out[:-1]):
        if tok in _PATH_FLAGS:
            p = Path(out[i + 1])
            if not p.is_absolute():
                p = (REPO_ROOT / p).resolve()
            out[i + 1] = str(p) if absolute else os.path.relpath(p, backend_root)
    return out


def evaluate(
    cli_args: list[str],
    run_id: str,
    out_dir: str | Path,
    timeout_sec: int = 1800,
    cache_dir: Optional[str | Path] = None,
    python_exe: Optional[str] = None,
    backend: str = "legacy",
) -> Union[Metrics, Infeasible]:
    """Run the selected simulator backend and return parsed Metrics or Infeasible."""
    from sim_backends import get_backend

    b = get_backend(backend)
    out_dir = Path(out_dir)
    csv_path, rel_out = stage_path(out_dir, f"sim_out/{run_id}.csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    full_args = list(cli_args) + ["--output", rel_out]

    # disk cache (backend name in the key: same config on the two backends
    # is a different run)
    cache_dir = Path(cache_dir) if cache_dir else out_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hash_obj([backend, *full_args])
    cache_file = cache_dir / f"{key}.json"
    if cache_file.is_file():
        log.info("cache hit for run %s (%s)", run_id, key)
        d = json.loads(cache_file.read_text())
        if d.get("infeasible"):
            return Infeasible(d["reason"])
        return Metrics(**{k: v for k, v in d.items() if k != "infeasible"})

    # legacy AND upstream both need paths relative to their backend root:
    # upstream's build_cluster_config unconditionally prepends "../" after its
    # astra-sim chdir (same quirk as legacy — see UpstreamBackend._path_for_cli).
    # Only the measured backend (root = repo root) takes absolute paths.
    run_args = _rebase_path_args(full_args, b.root, absolute=(backend == "measured"))
    entry = {"legacy": ["main.py"], "upstream": ["-m", "serving"],
             "measured": ["-m", "sim_backends.measured"]}[backend]
    cmd = [python_exe or b.python_exe(), *entry, *run_args]
    log.info("running: %s", " ".join(cmd))
    try:
        proc = run_guarded(cmd, cwd=str(b.root), env=b.env(),
                           timeout_sec=timeout_sec)
    except subprocess.TimeoutExpired:
        result = Infeasible(f"timeout after {timeout_sec}s")
        cache_file.write_text(json.dumps({"infeasible": True, "reason": result.reason}))
        return result

    if proc.returncode != 0:
        # output is merged into stdout; stderr carries the guard explanation
        tail = (proc.stderr or "") + (proc.stdout or "")[-500:]
        result = Infeasible(f"exit {proc.returncode}: {tail}")
        cache_file.write_text(json.dumps({"infeasible": True, "reason": result.reason}))
        return result

    abs_csv = REPO_ROOT / rel_out
    if not abs_csv.is_file():
        result = Infeasible("no output CSV produced")
        cache_file.write_text(json.dumps({"infeasible": True, "reason": result.reason}))
        return result

    try:
        metrics = parse_metrics_csv(abs_csv)
    except Exception as e:  # noqa: BLE001 - any parse error => infeasible
        result = Infeasible(f"parse error: {e}")
        cache_file.write_text(json.dumps({"infeasible": True, "reason": result.reason}))
        return result

    energy = _parse_energy_from_stdout(proc.stdout or "")
    if energy and energy > 0:
        metrics.energy_j = energy
        total_out = metrics.raw.get("total_output_tokens", 0.0)
        metrics.toks_per_wh = total_out / (energy / J_PER_WH) if energy else None
        metrics.toks_per_j = total_out / energy
        span_ns = metrics.raw.get("span_ns", 0.0)
        if span_ns > 0:
            metrics.power_w = energy / (span_ns / NS_PER_S)
            metrics.raw["power_source"] = "sim_energy"

    cache_file.write_text(json.dumps(metrics.as_row()))
    return metrics
