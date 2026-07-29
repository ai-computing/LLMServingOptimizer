"""Third evaluation backend ``measured``: measurement-based instance oracles
combined with a lightweight event-driven cluster simulator (design doc §5).

Implements the same ``SimBackend`` contract as the ``legacy``/``upstream``
adapters but runs in-process (no subprocess, no ASTRA-Sim dependency).
Filled in by milestones M5–M7; M0 only guarantees importability.
"""
