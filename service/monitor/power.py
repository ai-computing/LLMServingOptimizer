"""Device power sampling (plan §4.3): nvidia-smi query (local; DCGM/remote and
furiosa-smi are D5 stubs). Parsing is a pure function for tests."""
from __future__ import annotations

import subprocess
import time
from typing import Optional

from .store import PowerSample

_QUERY = ("--query-gpu=index,power.draw,utilization.gpu,memory.used,"
          "temperature.gpu")


def parse_nvidia_csv(text: str, device_id_by_index: dict[int, str],
                     ts: Optional[float] = None) -> list[PowerSample]:
    """csv,noheader,nounits rows: index, power.draw, util, mem_used_MiB, temp."""
    ts = ts if ts is not None else time.time()
    out: list[PowerSample] = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            idx = int(parts[0])
        except ValueError:
            continue
        if idx not in device_id_by_index:
            continue

        def num(s: str) -> float:
            try:
                return float(s)
            except ValueError:   # "[N/A]" etc.
                return 0.0

        out.append(PowerSample(ts=ts, device_id=device_id_by_index[idx],
                               power_w=num(parts[1]), util=num(parts[2]),
                               mem_used_gb=num(parts[3]) / 1024.0,
                               temp_c=num(parts[4])))
    return out


def query_nvidia_power(device_id_by_index: dict[int, str]) -> list[PowerSample]:
    """One local nvidia-smi sample for the given host-local GPU indices."""
    if not device_id_by_index:
        return []
    idx = ",".join(str(i) for i in sorted(device_id_by_index))
    proc = subprocess.run(
        ["nvidia-smi", _QUERY, "--format=csv,noheader,nounits", "-i", idx],
        capture_output=True, text=True, timeout=10)
    if proc.returncode != 0:
        return []
    return parse_nvidia_csv(proc.stdout, device_id_by_index)
