#!/usr/bin/env python3
"""Pinpoint the accumulation site: main-thread stack + oversized locals.

Every INTERVAL the probe thread walks the main thread's frames and reports, per
frame, file:line:function plus any local variable holding a container with more
than 1e6 items (with an element sample). That names file, line and variable.
Self-caps RSS. The submodule is never modified.
"""
from __future__ import annotations

import os
import sys
import threading
import time

INTERVAL = float(os.environ.get("PROBE_INTERVAL", "15"))
CAP_GB = float(os.environ.get("PROBE_CAP_GB", "16"))
BIG = int(os.environ.get("PROBE_BIG", "1000000"))


def rss_gb() -> float:
    with open("/proc/self/statm") as f:
        return int(f.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1024 ** 3


def report() -> None:
    t0 = time.time()
    me = threading.get_ident()
    while True:
        time.sleep(INTERVAL)
        rss = rss_gb()
        print(f"\n[probe] t={time.time() - t0:.0f}s rss={rss:.2f}GB", flush=True)
        frames = sys._current_frames()
        for tid, frame in frames.items():
            if tid == me:
                continue
            stack = []
            f = frame
            depth = 0
            while f is not None and depth < 40:
                stack.append((f, f.f_code.co_filename.split("/")[-1],
                              f.f_lineno, f.f_code.co_name))
                f = f.f_back
                depth += 1
            print(f"[probe] thread {tid} stack (innermost first):", flush=True)
            for f, fn, ln, name in stack[:12]:
                print(f"[probe]     {fn}:{ln} {name}()", flush=True)
            # oversized locals anywhere on the stack
            for f, fn, ln, name in stack:
                for var, val in list(f.f_locals.items()):
                    try:
                        n = len(val)
                    except Exception:
                        continue
                    if isinstance(val, (list, dict, set, str, bytes)) and n > BIG:
                        sample = ""
                        try:
                            it = iter(val.values()) if isinstance(val, dict) else iter(val)
                            s = next(it)
                            sample = f"{type(s).__name__}={s!r:.60}"
                        except Exception:
                            pass
                        print(f"[probe]   >>> {fn}:{ln} {name}()  "
                              f"local '{var}' {type(val).__name__} len={n:,} "
                              f"first={sample}", flush=True)
        del frames
        if rss > CAP_GB:
            print(f"[probe] RSS cap {CAP_GB} GB -> exit", flush=True)
            os._exit(9)


if __name__ == "__main__":
    threading.Thread(target=report, daemon=True).start()
    sys.argv = ["serving"] + sys.argv[1:]
    import runpy
    runpy.run_module("serving", run_name="__main__")
