"""Parse simulator stdout logs and per-request CSVs into a metrics dict.

Logic ported from script/build_a6000_4_report.py:32-86 and extended to cover
the Median/P99 TTFT/TPOT/ITL fields plus per-request distribution lists used
by plots.py for CDFs.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

# ANSI color escapes appear throughout LLMServingSim's stdout (logger.py).
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")

PATTERNS = {
    "total_latency_s":   r"Total latency \(s\):\s+([\d.]+)",
    "req_throughput":    r"Request throughput \(req/s\):\s+([\d.]+)",
    "prompt_throughput": r"Average prompt throughput \(tok/s\):\s+([\d.]+)",
    "gen_throughput":    r"Average generation throughput \(tok/s\):\s+([\d.]+)",
    "total_token_tp":    r"Total token throughput \(tok/s\):\s+([\d.]+)",
    "total_requests":    r"Total requests:\s+(\d+)",
    "mean_ttft_ms":      r"Mean TTFT \(ms\):\s+([\d.]+)",
    "median_ttft_ms":    r"Median TTFT \(ms\):\s+([\d.]+)",
    "p99_ttft_ms":       r"P99 TTFT \(ms\):\s+([\d.]+)",
    "mean_tpot_ms":      r"Mean TPOT \(ms\):\s+([\d.]+)",
    "median_tpot_ms":    r"Median TPOT \(ms\):\s+([\d.]+)",
    "p99_tpot_ms":       r"P99 TPOT \(ms\):\s+([\d.]+)",
    "mean_itl_ms":       r"Mean ITL \(ms\):\s+([\d.]+)",
    "median_itl_ms":     r"Median ITL \(ms\):\s+([\d.]+)",
    "p99_itl_ms":        r"P99 ITL \(ms\):\s+([\d.]+)",
}

_INT_KEYS = {"total_requests"}

_NPU_UTIL_RE = re.compile(r"NPU Memory Usage [\d.]+ MB \(([\d.]+) % Used\)")
_SUCCESS_MARKER = "Simulation results"

# Power-model output (only present when the cluster config has a `power` block;
# see inference_serving/power_model.py:print_power_summary).
# The "Total energy consumption (kJ):" line is the system-wide total — match
# at start-of-line via MULTILINE so we don't confuse it with "Node N total
# energy consumption (kJ):".
_TOTAL_ENERGY_RE = re.compile(r"^Total energy consumption \(kJ\):\s+([\d.]+)", re.MULTILINE)
# Per-device totals printed as a tree, e.g. "├─ NPU energy consumption (J):  321.45".
_DEVICE_ENERGY_RE = re.compile(r"[├└]─\s+(Base Node|NPU|CPU|Memory|Link|NIC|Storage)\s+energy consumption \(J\):\s+([\d.]+)")
# Log label -> metric key (Wh, converted in parse_log). "Memory" is the
# printed label for the dram device.
_DEVICE_KEY_MAP = {
    "Base Node": "base_node_energy_wh",
    "NPU":       "npu_energy_wh",
    "CPU":       "cpu_energy_wh",
    "Memory":    "dram_energy_wh",
    "Link":      "link_energy_wh",
    "NIC":       "nic_energy_wh",
    "Storage":   "storage_energy_wh",
}


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def parse_log(log_path: Path) -> dict:
    """Extract scalar metrics from a simulator stdout log.

    Returns {} on missing file or parse failure.
    """
    if not log_path.exists():
        return {}
    try:
        text = log_path.read_text(errors="replace")
    except OSError:
        return {}
    text = _strip_ansi(text)

    metrics: dict = {}
    for key, pat in PATTERNS.items():
        m = re.search(pat, text)
        if m:
            try:
                if key in _INT_KEYS:
                    metrics[key] = int(m.group(1))
                else:
                    metrics[key] = float(m.group(1))
            except ValueError:
                pass

    npu_utils = _NPU_UTIL_RE.findall(text)
    if npu_utils:
        try:
            metrics["npu_util_pct"] = float(npu_utils[-1])
        except ValueError:
            pass

    # Power model totals — silently absent when the run had no `power` block.
    # Convert kJ→Wh (×1000/3600) so all downstream display uses watt-hours.
    m = _TOTAL_ENERGY_RE.search(text)
    if m:
        try:
            kj = float(m.group(1))
            metrics["total_energy_wh"] = kj * 1000.0 / 3600.0
        except ValueError:
            pass

    # Per-device energy sums (J) across all nodes, converted to Wh.
    device_j_sums: dict[str, float] = {}
    for label, val_str in _DEVICE_ENERGY_RE.findall(text):
        key = _DEVICE_KEY_MAP.get(label)
        if not key:
            continue
        try:
            device_j_sums[key] = device_j_sums.get(key, 0.0) + float(val_str)
        except ValueError:
            pass
    for key, j_total in device_j_sums.items():
        metrics[key] = j_total / 3600.0  # J → Wh

    # Derived: tokens per watt-hour — energy efficiency index.
    # = (tok/s) × simulation_time_s / energy_Wh
    tp = metrics.get("total_token_tp")
    lat = metrics.get("total_latency_s")
    ewh = metrics.get("total_energy_wh")
    if tp is not None and lat is not None and ewh and ewh > 0:
        metrics["tok_per_wh"] = tp * lat / ewh

    # Derived: average power = total_energy_wh × 3600 / total_latency_s
    # Used by ranker.py's power_max_w SLO check.
    if ewh and lat and lat > 0:
        metrics["avg_power_w"] = ewh * 3600.0 / lat

    return metrics


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Return the pct-percentile of an already-sorted list (matches old report)."""
    if not sorted_values:
        return 0.0
    idx = int(len(sorted_values) * pct)
    if idx >= len(sorted_values):
        idx = len(sorted_values) - 1
    return sorted_values[idx]


def parse_csv(csv_path: Path) -> dict:
    """Compute per-request stats from CSV (TTFT/TPOT/ITL).

    CSV columns: instance id,request id,model,input,output,arrival,end_time,
    latency,queuing_delay,TTFT,TPOT,ITL
    Native units are nanoseconds; ITL is a JSON list of inter-token latencies.
    Output values are in milliseconds.
    """
    stats: dict = {}
    if not csv_path.exists():
        return stats

    ttfts_ns: list[float] = []
    tpots_ns: list[float] = []
    itls_ns: list[float] = []
    request_count = 0

    try:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                request_count += 1
                try:
                    ttfts_ns.append(float(row["TTFT"]))
                except (KeyError, ValueError):
                    pass
                try:
                    tpots_ns.append(float(row["TPOT"]))
                except (KeyError, ValueError):
                    pass
                try:
                    itl_list = json.loads(row.get("ITL", "[]") or "[]")
                    if isinstance(itl_list, list):
                        for v in itl_list:
                            try:
                                itls_ns.append(float(v))
                            except (TypeError, ValueError):
                                pass
                except (json.JSONDecodeError, TypeError):
                    pass
    except OSError:
        return stats

    ns_to_ms = 1e-6
    stats["request_count"] = request_count

    if ttfts_ns:
        ttfts_ms = sorted(v * ns_to_ms for v in ttfts_ns)
        stats["ttft_avg_ms"] = sum(ttfts_ms) / len(ttfts_ms)
        stats["ttft_p99_ms"] = _percentile(ttfts_ms, 0.99)
        stats["ttft_values_ms"] = ttfts_ms

    if tpots_ns:
        tpots_ms = sorted(v * ns_to_ms for v in tpots_ns)
        stats["tpot_avg_ms"] = sum(tpots_ms) / len(tpots_ms)
        stats["tpot_p99_ms"] = _percentile(tpots_ms, 0.99)

    if itls_ns:
        itls_ms = sorted(v * ns_to_ms for v in itls_ns)
        stats["itl_avg_ms"] = sum(itls_ms) / len(itls_ms)
        stats["itl_p99_ms"] = _percentile(itls_ms, 0.99)
        stats["itl_values_ms"] = itls_ms

    return stats


def parse_run(log_path: Path, csv_path: Path) -> dict:
    """Combine log + CSV parsing into one metrics dict.

    Log uses p99_*_ms naming; CSV uses *_p99_ms naming.  Both forms are
    aliased here so downstream code (ranker, UI) works regardless of which
    source populated the value.
    """
    m = parse_log(log_path)
    m.update(parse_csv(csv_path))
    # Alias: prefer the already-present key, fill the missing form.
    for log_k, csv_k in [
        ("p99_ttft_ms", "ttft_p99_ms"),
        ("p99_tpot_ms", "tpot_p99_ms"),
        ("p99_itl_ms",  "itl_p99_ms"),
    ]:
        if log_k in m and csv_k not in m:
            m[csv_k] = m[log_k]
        elif csv_k in m and log_k not in m:
            m[log_k] = m[csv_k]
    return m


def is_successful(log_path: Path) -> bool:
    """True if the log contains the 'Simulation results' marker."""
    if not log_path.exists():
        return False
    try:
        text = log_path.read_text(errors="replace")
    except OSError:
        return False
    return _SUCCESS_MARKER in _strip_ansi(text)


# Python exception line, e.g. "FileNotFoundError: ..." or "TypeError: ...".
_EXC_LINE_RE = re.compile(r"^[A-Z][A-Za-z_]*(?:Error|Exception|Warning):\s+.+")
# Last-resort: lines that mention failure/error keywords.
_ERR_HINT_RE = re.compile(r"\b(error|exception|failed|fatal|abort|killed)\b", re.IGNORECASE)
_MAX_EXCERPT_LEN = 240
_TAIL_BYTES = 16 * 1024  # 16 KiB — enough to capture trailing tracebacks


def extract_error_excerpt(log_path: Path) -> str:
    """Best-effort one-line summary of why a failed run failed.

    Prefers the final Python exception line (e.g. ``TypeError: ...``) since
    that's what `runner.py` sees when a subprocess exits with returncode!=0.
    Falls back to the most recent error-hinting line, then the last non-empty
    line of the log. Returns ``""`` if nothing useful is found.
    """
    if not log_path.exists():
        return ""
    try:
        with log_path.open("rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - _TAIL_BYTES))
            blob = f.read()
    except OSError:
        return ""

    text = _strip_ansi(blob.decode(errors="replace"))
    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return ""

    # 1) Last Python exception line — typically the actual root cause.
    for ln in reversed(lines):
        if _EXC_LINE_RE.match(ln.strip()):
            return ln.strip()[:_MAX_EXCERPT_LEN]

    # 2) Last error-hinting line.
    for ln in reversed(lines):
        if _ERR_HINT_RE.search(ln):
            return ln.strip()[:_MAX_EXCERPT_LEN]

    # 3) Last non-empty line.
    return lines[-1].strip()[:_MAX_EXCERPT_LEN]
