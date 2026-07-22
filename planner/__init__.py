"""Offline resource-allocation planner for heterogeneous LLM serving clusters.

Two-stage design (see PLAN_MILP_MaxFlow.md):

  Stage 1 (structural, exact-ish):  MILP / Max-Flow over a cluster graph produces
                                    Top-K candidate allocations.
  Stage 2 (simulation, ground-truth): each candidate is rendered to a
                                    ``cluster_config/*.json`` and evaluated by running
                                    LLMServingSim (``main.py``) as a subprocess; the
                                    resulting ``output/*.csv`` is parsed for SLO metrics
                                    and the candidates are re-ranked.

The planner is a *non-invasive wrapper*: it never edits the simulator source. It
only produces simulator inputs and parses simulator outputs.
"""

__all__ = ["__version__"]
__version__ = "0.1.0"
