"""Stage-2 memory guard: a runaway simulation must die as an Infeasible
candidate, not OOM-kill the host.

Observed live: an A40 70B (tp4 x2) upstream run grew ~13 GB/min past 73 GB RSS
with 4 candidates in flight, leaving 10 GB of 93 GB free — one more minute and
the OOM killer would have taken out the webapp.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

from planner.sim_evaluator import (
    _tree_rss_bytes,
    run_guarded,
    sim_mem_limit_bytes,
)

_HOG = ("import time\n"
        "blocks = []\n"
        "while True:\n"
        "    blocks.append(bytearray(32 * 1024 * 1024))\n"
        "    time.sleep(0.01)\n")

_CHILD_HOG = ("import subprocess, sys, time\n"
              "subprocess.Popen([sys.executable, '-c', '''\n"
              "import time\n"
              "b = []\n"
              "while True:\n"
              "    b.append(bytearray(32 * 1024 * 1024)); time.sleep(0.01)\n"
              "'''])\n"
              "time.sleep(60)\n")


def _env():
    return dict(os.environ)


def test_tree_rss_counts_self_and_children():
    assert _tree_rss_bytes(os.getpid()) > 1_000_000        # this interpreter
    assert _tree_rss_bytes(999_999_99) == 0                # missing pid: no crash


def test_limit_from_env(monkeypatch):
    monkeypatch.setenv("LLMSS_SIM_MEM_LIMIT_GB", "3.5")
    assert sim_mem_limit_bytes() == int(3.5 * 1024 ** 3)
    monkeypatch.setenv("LLMSS_SIM_MEM_LIMIT_GB", "0")      # disabled
    assert sim_mem_limit_bytes() == 0
    monkeypatch.setenv("LLMSS_SIM_MEM_LIMIT_GB", "nonsense")
    assert sim_mem_limit_bytes() > 0                       # falls back to default


def test_normal_command_passes_through(tmp_path):
    res = run_guarded([sys.executable, "-c", "print('ok'); import sys; sys.exit(0)"],
                      cwd=str(tmp_path), env=_env(), timeout_sec=30,
                      mem_limit_bytes=512 * 1024 ** 2, poll_sec=0.05)
    assert res.returncode == 0 and "ok" in res.stdout


def test_nonzero_exit_keeps_output(tmp_path):
    res = run_guarded([sys.executable, "-c",
                       "import sys; print('boom'); sys.exit(3)"],
                      cwd=str(tmp_path), env=_env(), timeout_sec=30,
                      mem_limit_bytes=512 * 1024 ** 2, poll_sec=0.05)
    assert res.returncode == 3 and "boom" in res.stdout


def test_chatty_command_does_not_deadlock(tmp_path):
    """Output goes to a temp file, so a simulator printing far more than a pipe
    buffer can hold cannot block while we poll memory."""
    res = run_guarded([sys.executable, "-c",
                       "print('x' * 200000)\nprint('done')"],
                      cwd=str(tmp_path), env=_env(), timeout_sec=60,
                      mem_limit_bytes=512 * 1024 ** 2, poll_sec=0.05)
    assert res.returncode == 0 and res.stdout.rstrip().endswith("done")


def test_memory_guard_kills_runaway(tmp_path):
    res = run_guarded([sys.executable, "-c", _HOG], cwd=str(tmp_path),
                      env=_env(), timeout_sec=120,
                      mem_limit_bytes=250 * 1024 ** 2, poll_sec=0.05)
    assert res.returncode == 137
    assert "memory guard" in res.stderr and "GiB" in res.stderr


def test_memory_guard_counts_child_processes(tmp_path):
    """The upstream backend spawns AnalyticalAstra; the ceiling applies to the
    whole tree and the kill takes the group down."""
    res = run_guarded([sys.executable, "-c", _CHILD_HOG], cwd=str(tmp_path),
                      env=_env(), timeout_sec=120,
                      mem_limit_bytes=300 * 1024 ** 2, poll_sec=0.05)
    assert res.returncode == 137, res.stdout[-300:]
    assert "memory guard" in res.stderr


def test_guard_disabled_lets_process_finish(tmp_path):
    res = run_guarded([sys.executable, "-c", "print('unlimited')"],
                      cwd=str(tmp_path), env=_env(), timeout_sec=30,
                      mem_limit_bytes=0, poll_sec=0.05)
    assert res.returncode == 0 and "unlimited" in res.stdout


def test_timeout_still_raises_and_kills(tmp_path):
    with pytest.raises(subprocess.TimeoutExpired):
        run_guarded([sys.executable, "-c", "import time; time.sleep(30)"],
                    cwd=str(tmp_path), env=_env(), timeout_sec=1,
                    mem_limit_bytes=512 * 1024 ** 2, poll_sec=0.05)


def test_evaluate_reports_guard_kill_as_infeasible(tmp_path, monkeypatch):
    """The Stage-2 contract: a guard kill becomes Infeasible, so the planner
    drops that candidate and keeps going."""
    from planner import sim_evaluator
    from planner.types import Infeasible

    def fake_guarded(cmd, cwd, env, timeout_sec, **kw):
        return subprocess.CompletedProcess(
            cmd, 137, "", "memory guard: RSS 17.2 GiB exceeded the 16 GiB "
                          "per-simulation limit (LLMSS_SIM_MEM_LIMIT_GB)")

    monkeypatch.setattr(sim_evaluator, "run_guarded", fake_guarded)
    res = sim_evaluator.evaluate(
        ["--cluster-config", "x.json", "--dataset", "d.jsonl"],
        run_id="guard", out_dir=tmp_path, timeout_sec=10, backend="measured")
    assert isinstance(res, Infeasible)
    assert "memory guard" in res.reason and "137" in res.reason
