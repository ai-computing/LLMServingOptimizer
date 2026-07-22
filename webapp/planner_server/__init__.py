"""Planner server routes mounted under /api/planner and /planner on the webapp.

Exposes the MILP/Max-Flow two-stage planner (top-level ``planner/`` package) as
a web tab: build a spec, run Stage-1 (CP-SAT epsilon-sweep) + Stage-2 (real
simulation), and inspect the Pareto-ranked candidates.
"""
from .routes import planner_router

__all__ = ["planner_router"]
