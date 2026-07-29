"""Power-optimal LLM serving resource recommendation service.

Service layer on top of the planner/DSE stack (PLAN_power_service.md):
workload synthesis (M3), inventory management (M8), fidelity routing and
the FastAPI service API (M9). Non-invasive: only produces simulator inputs
and consumes simulator outputs via ``sim_backends``.
"""
