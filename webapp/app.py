"""FastAPI entry point for the LLMServingSim Web UI.

Routes:
    GET  /                         -> scenario builder
    GET  /sweeps                   -> sweep history list
    GET  /sweep/{sweep_id}         -> live progress page
    GET  /sweep/{sweep_id}/results -> results page (server-rendered Plotly)

    POST /api/enumerate            -> per-scenario config preview
    POST /api/sweeps               -> create + launch sweep
    GET  /api/sweeps/{id}/status   -> status.json (polling fallback)
    GET  /api/sweeps/{id}/events   -> SSE event stream
    POST /api/sweeps/{id}/cancel   -> cancel sweep
    GET  /api/sweeps/{id}/pareto   -> Plotly JSON for pareto chart (param x_metric)
    GET  /api/sweeps/{id}/download/metrics -> metrics.json
    GET  /api/sweeps/{id}/download/zip     -> zip of runs/*.{csv,log}
    GET  /api/hardware             -> {hardware_name: [model, ...]}
    GET  /api/datasets             -> [{path, family, compatible_models}, ...]
"""
from __future__ import annotations

import asyncio
import io
import json
import re
import time
import zipfile
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    StreamingResponse,
    FileResponse,
    Response,
)
from fastapi.staticfiles import StaticFiles
from jinja2 import Environment, FileSystemLoader, select_autoescape

from . import runner
from .cluster_builder import ConfigSpec, InstanceSpec, build_cluster_json, validate_spec
from .cluster_io import delete_config, list_configs, load_config, save_config, sanitize_filename
from .config import (
    CPU_MEM_DEFAULT,
    DATASET_DIR,
    LINK_BW_DEFAULT,
    LINK_LATENCY_DEFAULT,
    OUTPUT_DIR,
    SOFT_CAP,
)
from .enumerate import enumerate_configs
from .hardware_catalog import (
    build_catalog,
    list_hardware,
    list_models_for_hardware,
)
from .parser import parse_run
from .plots import all_plots, assign_config_colors, pareto_scatter

# ---------------------------------------------------------------------------
# App + Jinja + static
# ---------------------------------------------------------------------------

WEBAPP_DIR = Path(__file__).parent
TEMPLATES_DIR = WEBAPP_DIR / "templates"
STATIC_DIR = WEBAPP_DIR / "static"

_jinja = Environment(
    loader=FileSystemLoader(str(TEMPLATES_DIR)),
    autoescape=select_autoescape(["html", "xml"]),
)

app = FastAPI(title="LLMServingSim Web UI")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _static_url(rel_path: str) -> str:
    """Build a /static/<rel_path> URL with a cache-busting ?v=<mtime> suffix.

    The version is the file's modification time, so the URL changes whenever the
    asset is edited — browsers fetch the new file instead of serving a stale
    cached copy (which previously left an old dse_progress.js calling the wrong
    cancel endpoint). Falls back to no suffix if the file is missing.
    """
    f = STATIC_DIR / rel_path
    try:
        return f"/static/{rel_path}?v={int(f.stat().st_mtime)}"
    except OSError:
        return f"/static/{rel_path}"


_jinja.globals["static"] = _static_url

# DSE (Design Space Exploration) routes. Lives under /api/dse/... so it's
# isolated from sweep routes but shares the same FastAPI process, catalog,
# and SSE infrastructure (webapp.runner). See webapp/dse/server/routes.py.
from .dse.server import dse_router  # noqa: E402
app.include_router(dse_router, prefix="/api/dse")

# Planner (MILP/Max-Flow two-stage planner) routes under /api/planner.
from .planner_server import planner_router  # noqa: E402
app.include_router(planner_router, prefix="/api/planner")

# Power-optimal serving-recommendation service (service/ package). Mounted
# only when a cluster registry exists; endpoints live under /api/... (see
# service/api/routes.py: /api/cluster, /api/models, /api/serve-requests).
import os  # noqa: E402

_REGISTRY_PATH = Path(os.environ.get(
    "LLMSS_CLUSTER_REGISTRY", str(WEBAPP_DIR.parent / "cluster_registry.yaml")))
SERVICE_ENABLED = _REGISTRY_PATH.is_file()
if SERVICE_ENABLED:
    from service.api.routes import ServiceState, create_service_router  # noqa: E402
    from service.inventory.ledger import Ledger  # noqa: E402
    from service.topology.schema import load_registry_any  # noqa: E402

    # registry v2 (v1 files are auto-promoted), so hierarchical intra-node
    # fabrics and per-node dockerd endpoints are honored
    _svc_registry = load_registry_any(_REGISTRY_PATH)
    _svc_ledger_path = Path(os.environ.get(
        "LLMSS_SERVICE_DB",
        str(WEBAPP_DIR.parent / "output" / "service_ledger.sqlite")))
    _svc_ledger_path.parent.mkdir(parents=True, exist_ok=True)
    service_state = ServiceState(registry=_svc_registry,
                                 ledger=Ledger(_svc_ledger_path, _svc_registry))
    # deployment layer (execution plan D2): same DB as the ledger; local/remote
    # dockerd per registry (v1 registries default to the local socket)
    from service.api.deployment_routes import create_deployment_router  # noqa: E402
    from service.deploy.docker_driver import DockerSdkDriver  # noqa: E402
    from service.deploy.manager import DeploymentManager  # noqa: E402
    from service.deploy.store import DeployStore  # noqa: E402
    from service.monitor.runtime import MonitorRuntime  # noqa: E402

    service_state.deploy_store = DeployStore(_svc_ledger_path)
    if os.environ.get("LLMSS_FAKE_DOCKER") == "1":
        # UI/E2E test mode: in-memory driver, instant health (plan D4)
        from service.deploy.docker_driver import FakeDriver  # noqa: E402
        _svc_driver = FakeDriver()
    else:
        # per-node dockerd endpoints from the registry: a deployment on a
        # remote node must never silently run on the local daemon
        _svc_driver = DockerSdkDriver(service_state.docker_endpoints())
    service_state.monitor_runtime = MonitorRuntime(
        service_state.deploy_store, driver=_svc_driver)
    _mgr_kw = {}
    if os.environ.get("LLMSS_FAKE_DOCKER") == "1":
        _mgr_kw = dict(health_fn=lambda url: True, sleep=lambda s: None)
        service_state.monitor_runtime.http_get = lambda url: (
            "vllm:num_requests_running 1\nvllm:num_requests_waiting 0\n"
            "vllm:gpu_cache_usage_perc 0.2\n"
            "vllm:generation_tokens_total 1000\nvllm:prompt_tokens_total 500\n")
        service_state.monitor_runtime.power_fn = lambda gpu_map: []
        service_state.monitor_runtime.interval_s = 1.0
    service_state.deploy_manager = DeploymentManager(
        service_state.deploy_store, _svc_driver, service_state.ledger,
        node_host=lambda node_id: "localhost",
        on_ready=service_state.monitor_runtime.attach,
        on_stopped=__import__("service.api.deployment_routes",
                              fromlist=["make_on_stopped"]).make_on_stopped(
            service_state,
            WEBAPP_DIR.parent / "output" / "power_calibration_history.jsonl"),
        **_mgr_kw)
    app.include_router(create_service_router(service_state))
    app.include_router(create_deployment_router(service_state))
    try:  # restart recovery (plan §4.4): mid-flight rollback + READY re-attach
        _recovered = service_state.deploy_manager.recover()
        for _dep_id in _recovered.get("reattached", []):
            service_state.monitor_runtime.attach(_dep_id)
    except Exception:
        pass

    def _reconcile_loop():  # plan §4.4 periodic ledger<->dockerd cross-check
        import time as _time
        node_ids = [n.id for n in _svc_registry.nodes]
        while True:
            _time.sleep(60)
            try:
                out = service_state.deploy_manager.reconcile(node_ids)
                if out["orphans_removed"] or out["missing_marked_failed"]:
                    print(f"[service] reconcile: {out}")
            except Exception:
                pass

    import threading as _threading
    _threading.Thread(target=_reconcile_loop, daemon=True).start()


# ---------------------------------------------------------------------------
# DSE — HTML pages (server-rendered Jinja templates)
# ---------------------------------------------------------------------------

@app.get("/dse/explore", response_class=HTMLResponse)
async def dse_explore_page(request: Request) -> HTMLResponse:
    return render("dse_explore.html", title="DSE — Explore")


@app.get("/dse/jobs/{job_id}", response_class=HTMLResponse)
async def dse_progress_page(job_id: str, request: Request) -> HTMLResponse:
    return render("dse_progress.html", title=f"DSE — {job_id}", job_id=job_id)


@app.get("/dse/jobs/{job_id}/results", response_class=HTMLResponse)
async def dse_results_page(job_id: str, request: Request) -> HTMLResponse:
    return render("dse_results.html", title=f"DSE Results — {job_id}", job_id=job_id)


# ---------------------------------------------------------------------------
# Planner (MILP/Max-Flow) — HTML pages
# ---------------------------------------------------------------------------

@app.get("/planner", response_class=HTMLResponse)
async def planner_explore_page(request: Request) -> HTMLResponse:
    return render("planner_explore.html", title="Planner")


@app.get("/planner/jobs/{job_id}", response_class=HTMLResponse)
async def planner_progress_page(job_id: str, request: Request) -> HTMLResponse:
    return render("planner_progress.html", title=f"Planner — {job_id}", job_id=job_id)


@app.get("/planner/jobs/{job_id}/results", response_class=HTMLResponse)
async def planner_results_page(job_id: str, request: Request) -> HTMLResponse:
    return render("planner_results.html", title=f"Planner Results — {job_id}", job_id=job_id)


@app.get("/service", response_class=HTMLResponse)
async def service_page(request: Request) -> HTMLResponse:
    from fastapi.responses import RedirectResponse
    return RedirectResponse("/service/cluster")


def _service_render(tpl: str, subtab: str, title: str) -> HTMLResponse:
    return render(tpl, title=title, subtab=subtab,
                  service_enabled=SERVICE_ENABLED,
                  registry_path=str(_REGISTRY_PATH))


@app.get("/service/cluster", response_class=HTMLResponse)
async def service_cluster_page(request: Request) -> HTMLResponse:
    return _service_render("service/cluster.html", "cluster", "Service — Cluster")


@app.get("/service/request", response_class=HTMLResponse)
async def service_request_page(request: Request) -> HTMLResponse:
    return _service_render("service/request.html", "request", "Service — Request")


@app.get("/service/deployments", response_class=HTMLResponse)
async def service_deployments_page(request: Request) -> HTMLResponse:
    return _service_render("service/deployments.html", "deployments",
                           "Service — Deployments")


@app.get("/favicon.ico", include_in_schema=False)
async def _favicon() -> Response:
    """Silence the 404s browsers generate by auto-requesting /favicon.ico.

    Serves a tiny inline SVG so the tab gets an icon and access logs stop
    complaining. Cached for 1 day to avoid repeat hits during reload work.
    """
    svg = (
        b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">'
        b'<rect width="64" height="64" rx="12" fill="#1f77b4"/>'
        b'<text x="32" y="44" font-family="monospace" font-size="36" '
        b'font-weight="bold" text-anchor="middle" fill="#fff">LS</text>'
        b'</svg>'
    )
    return Response(
        content=svg,
        media_type="image/svg+xml",
        headers={"Cache-Control": "public, max-age=86400"},
    )


_NAV_MAP = {
    "index.html":          "sweep",
    "sweeps_list.html":    "history",
    "progress.html":       "history",
    "results.html":        "history",
    "dse_explore.html":    "dse",
    "dse_progress.html":   "dse",
    "dse_results.html":    "dse",
    "dse_jobs_list.html":  "dse",
    "planner_explore.html":  "planner",
    "planner_progress.html": "planner",
    "planner_results.html":  "planner",
}


def render(template_name: str, **context: Any) -> HTMLResponse:
    context.setdefault("active_nav", _NAV_MAP.get(template_name, ""))
    tmpl = _jinja.get_template(template_name)
    return HTMLResponse(tmpl.render(**context))


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def _startup() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    app.state.catalog = build_catalog()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SLUG_RE = re.compile(r"[^a-zA-Z0-9._-]+")


def _slugify(text: str, max_len: int = 32) -> str:
    s = _SLUG_RE.sub("-", text or "").strip("-").lower()
    if not s:
        s = "sweep"
    return s[:max_len]


def _new_sweep_id(name: str) -> str:
    return f"{time.strftime('%Y%m%d-%H%M%S')}-{_slugify(name)}"


def _phase_estimate_s(phase: str) -> int:
    return {"smoke": 20, "full": 120, "both": 140}.get(phase or "full", 120)


def _phys_npus(spec: ConfigSpec) -> int:
    """Total physical NPUs used by a config.

    Each instance's npu_num is the count of physical compute NPUs. ASTRA-Sim
    internally adds virtual "sender" NPUs for prefill instances (see
    config_builder.py where npu_num doubles for prefill) but those are
    modeling artifacts for KV-transfer simulation, not real hardware — the
    reference's mixed_pd_ar.json runs 1 prefill (npu=1) + 1 decode (npu=1)
    on a 2-physical-NPU cluster successfully.
    """
    return sum(inst.npu_num for inst in spec.instances)


def _spec_to_preview(spec: ConfigSpec, phase: str) -> dict:
    return {
        "label": spec.label,
        "tp": spec.tp,
        "pp": spec.pp,
        "dp": spec.dp,
        "pd_layout": spec.pd_layout,
        "phys_npus": _phys_npus(spec),
        "estimated_s": _phase_estimate_s(phase),
        "instances": [
            {
                "hardware": inst.hardware,
                "model": inst.model,
                "npu_num": inst.npu_num,
                "npu_group": inst.npu_group,
                "pd_type": inst.pd_type,
            }
            for inst in spec.instances
        ],
    }


def _scenario_dict(scn: dict) -> dict:
    """Strip API-only fields and pass through to enumerate_configs."""
    return {
        "instance_groups": scn.get("instance_groups", []),
        "axes": scn.get("axes", {}),
    }


def _detect_dataset_family(filename: str) -> dict:
    name = filename.lower()
    if "llama" in name:
        return {
            "family": "llama",
            "compatible_models": [
                "meta-llama/Llama-3.1-8B",
                "meta-llama/Llama-3.1-70B",
            ],
        }
    if "mixtral" in name:
        return {
            "family": "mixtral",
            "compatible_models": ["mistralai/Mixtral-8x7B-v0.1"],
        }
    if "phi" in name:
        return {
            "family": "phi",
            "compatible_models": ["microsoft/Phi-mini-MoE-instruct"],
        }
    return {"family": "unknown", "compatible_models": []}


# ---------------------------------------------------------------------------
# HTML pages
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def page_index(request: Request) -> HTMLResponse:
    return render("index.html", title="New Sweep")


@app.get("/sweeps", response_class=HTMLResponse)
async def page_sweeps(request: Request) -> HTMLResponse:
    sweeps = runner.list_sweeps()
    return render("sweeps_list.html", title="Sweeps", sweeps=sweeps)


@app.get("/sweep/{sweep_id}", response_class=HTMLResponse)
async def page_progress(sweep_id: str, request: Request) -> HTMLResponse:
    status = runner.get_status(sweep_id)
    if not status:
        raise HTTPException(status_code=404, detail=f"sweep {sweep_id} not found")
    return render(
        "progress.html",
        title=f"Sweep {sweep_id}",
        sweep_id=sweep_id,
        status=status,
        status_json=json.dumps(status),
    )


@app.get("/sweep/{sweep_id}/results", response_class=HTMLResponse)
async def page_results(sweep_id: str, request: Request) -> HTMLResponse:
    sweep_dir = OUTPUT_DIR / sweep_id
    status = runner.get_status(sweep_id)
    if not status:
        raise HTTPException(status_code=404, detail=f"sweep {sweep_id} not found")

    cfgs = status.get("configs", {})
    results: list[dict] = []
    for label, entry in cfgs.items():
        if entry.get("state") != "done":
            continue
        log_path = sweep_dir / "runs" / f"{label}.log"
        csv_path = sweep_dir / "runs" / f"{label}.csv"
        # Re-parse to recover ttft_values_ms / itl_values_ms (dropped from status.json).
        metrics = parse_run(log_path, csv_path)
        cfg_path = sweep_dir / "configs" / f"{label}.json"
        tp = pp = dp = 1
        pd_layout = "—"
        if cfg_path.exists():
            try:
                cluster = json.loads(cfg_path.read_text())
                # Reconstruct display values from first instance.
                instances = cluster.get("nodes", [{}])[0].get("instances", [])
                if instances:
                    npu_num = int(instances[0].get("npu_num", 1))
                    npu_group = int(instances[0].get("npu_group", 1)) or 1
                    tp = npu_num // npu_group
                    pp = npu_group
                    p = sum(1 for i in instances if i.get("pd_type") == "prefill")
                    d = sum(1 for i in instances if i.get("pd_type") == "decode")
                    if p and d:
                        pd_layout = f"{p}P+{d}D"
                        dp = max(d, 1)
                    else:
                        dp = len(instances)
            except (OSError, json.JSONDecodeError):
                pass
        results.append({
            "label": label,
            "tp": tp,
            "pp": pp,
            "dp": dp,
            "pd_layout": pd_layout,
            **metrics,
        })

    plots_json = all_plots(results)

    # Persist a metrics.json snapshot for download.
    metrics_path = sweep_dir / "metrics.json"
    try:
        metrics_path.write_text(json.dumps(
            {r["label"]: {k: v for k, v in r.items() if k != "label"} for r in results},
            indent=2,
            default=str,
        ))
    except OSError:
        pass

    config_rows = [
        {
            "label": r["label"],
            "tp": r.get("tp"),
            "pp": r.get("pp"),
            "dp": r.get("dp"),
            "pd_layout": r.get("pd_layout"),
            "throughput": r.get("total_token_tp"),
            "ttft": r.get("mean_ttft_ms"),
            "tpot": r.get("mean_tpot_ms"),
            "itl": r.get("p99_itl_ms"),
            "energy_wh": r.get("total_energy_wh"),
            "energy_breakdown": {
                "Base Node": r.get("base_node_energy_wh"),
                "NPU":       r.get("npu_energy_wh"),
                "CPU":       r.get("cpu_energy_wh"),
                "Memory":    r.get("dram_energy_wh"),
                "Link":      r.get("link_energy_wh"),
                "NIC":       r.get("nic_energy_wh"),
                "Storage":   r.get("storage_energy_wh"),
            },
        }
        for r in results
    ]
    has_energy = any(r.get("energy_wh") is not None for r in config_rows)

    # Shared color mapping for the Config Legend card. The labels-order MUST
    # match what plots.py uses internally so the legend swatches line up with
    # the per-config colors in every chart.
    config_colors = assign_config_colors([r["label"] for r in results])

    return render(
        "results.html",
        title=f"Results — {sweep_id}",
        sweep_id=sweep_id,
        status=status,
        config_rows=config_rows,
        plots_json=plots_json,
        has_metrics=metrics_path.exists(),
        has_energy=has_energy,
        config_colors=config_colors,
    )


# ---------------------------------------------------------------------------
# API: catalog + datasets
# ---------------------------------------------------------------------------

@app.get("/api/hardware")
async def api_hardware() -> JSONResponse:
    catalog = build_catalog()  # rebuild each request — ~1ms, picks up newly profiled hardware without server restart
    out: dict[str, list[str]] = {}
    for hw in list_hardware(catalog):
        out[hw] = list_models_for_hardware(catalog, hw)
    return JSONResponse(out)


@app.get("/api/datasets")
async def api_datasets() -> JSONResponse:
    out: list[dict] = []
    if DATASET_DIR.is_dir():
        for f in sorted(DATASET_DIR.iterdir()):
            if not f.is_file() or f.suffix != ".jsonl":
                continue
            info = _detect_dataset_family(f.name)
            out.append({
                "path": f"dataset/{f.name}",
                "name": f.name,
                **info,
            })
    return JSONResponse(out)


# ---------------------------------------------------------------------------
# API: enumerate
# ---------------------------------------------------------------------------

@app.post("/api/enumerate")
async def api_enumerate(request: Request) -> JSONResponse:
    body = await request.json()
    scenarios_in = body.get("scenarios", []) or []
    workload = body.get("workload", {}) or {}
    phase = workload.get("phase", "full")

    catalog = build_catalog(backend=workload.get("backend", "legacy"))  # rebuilt per request; source follows selected backend
    out_scenarios: list[dict] = []
    for scn in scenarios_in:
        specs = enumerate_configs(_scenario_dict(scn), catalog)
        configs = [_spec_to_preview(s, phase) for s in specs]
        count = len(configs)
        total_s = sum(c["estimated_s"] for c in configs)
        out_scenarios.append({
            "name": scn.get("name", "Scenario"),
            "configs": configs,
            "count": count,
            "exceeds_soft_cap": count > SOFT_CAP,
            "soft_cap": SOFT_CAP,
            "estimated_total_s": total_s,
        })

    return JSONResponse({"scenarios": out_scenarios})


# ---------------------------------------------------------------------------
# API: sweeps
# ---------------------------------------------------------------------------

@app.post("/api/sweeps")
async def api_create_sweep(request: Request) -> JSONResponse:
    body = await request.json()
    scenarios_in = body.get("scenarios", []) or []
    workload = body.get("workload", {}) or {}
    selected_labels: list[list[str]] = body.get("selected_labels") or []

    if not scenarios_in:
        raise HTTPException(status_code=400, detail="no scenarios provided")
    if "dataset" not in workload:
        raise HTTPException(status_code=400, detail="workload.dataset required")

    catalog = build_catalog(backend=workload.get("backend", "legacy"))  # rebuilt per request; source follows selected backend

    sweep_name = scenarios_in[0].get("name", "sweep")
    sweep_id = _new_sweep_id(sweep_name)
    sweep_dir = OUTPUT_DIR / sweep_id
    sweep_dir.mkdir(parents=True, exist_ok=True)

    all_specs: list[ConfigSpec] = []
    seen_labels: set[str] = set()
    for idx, scn in enumerate(scenarios_in):
        specs = enumerate_configs(_scenario_dict(scn), catalog)
        # Apply per-scenario label filter, if provided.
        if idx < len(selected_labels) and selected_labels[idx]:
            allowed = set(selected_labels[idx])
            specs = [s for s in specs if s.label in allowed]
        # Prefix label with scenario index to avoid cross-scenario collisions.
        scenario_tag = _slugify(scn.get("name", f"s{idx}"), 16)
        for s in specs:
            new_label = f"{scenario_tag}__{s.label}" if len(scenarios_in) > 1 else s.label
            if new_label in seen_labels:
                continue
            seen_labels.add(new_label)
            all_specs.append(ConfigSpec(
                label=new_label,
                instances=s.instances,
                tp=s.tp, pp=s.pp, dp=s.dp,
                pd_layout=s.pd_layout,
            ))

    if not all_specs:
        raise HTTPException(status_code=400, detail="no configs after filtering")

    scenario_record = {
        "sweep_id": sweep_id,
        "scenarios": scenarios_in,
        "workload": workload,
        "selected_labels": selected_labels,
    }

    # Launch in background; status.json is created inside run_sweep.
    asyncio.create_task(runner.run_sweep(
        sweep_id=sweep_id,
        configs=all_specs,
        scenario_json=scenario_record,
        sweep_dir=sweep_dir,
        workload=workload,
    ))

    return JSONResponse({"sweep_id": sweep_id})


@app.get("/api/sweeps/{sweep_id}/status")
async def api_sweep_status(sweep_id: str) -> JSONResponse:
    status = runner.get_status(sweep_id)
    if not status:
        raise HTTPException(status_code=404, detail="sweep not found")
    return JSONResponse(status)


@app.post("/api/sweeps/{sweep_id}/cancel")
async def api_sweep_cancel(sweep_id: str) -> JSONResponse:
    await runner.cancel_sweep(sweep_id)
    return JSONResponse({"ok": True})


@app.get("/api/sweeps/{sweep_id}/events")
async def api_sweep_events(sweep_id: str, request: Request) -> StreamingResponse:
    """SSE event stream. Heartbeat every 5s; closes on sweep_done."""
    queue = runner.subscribe_events(sweep_id)

    async def event_generator():
        try:
            # Send current snapshot first so reconnects sync state immediately.
            snapshot = runner.get_status(sweep_id)
            if snapshot:
                yield f"event: snapshot\ndata: {json.dumps(snapshot)}\n\n"
            sweep_done = False
            while not sweep_done:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=5.0)
                    yield f"data: {json.dumps(event, default=str)}\n\n"
                    if event.get("sweep_state") in ("done", "failed", "cancelled"):
                        sweep_done = True
                except asyncio.TimeoutError:
                    yield "data: {\"type\": \"heartbeat\"}\n\n"
        finally:
            runner.unsubscribe_events(sweep_id, queue)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/sweeps/{sweep_id}/pareto")
async def api_sweep_pareto(sweep_id: str, x_metric: str = "mean_ttft_ms") -> JSONResponse:
    sweep_dir = OUTPUT_DIR / sweep_id
    status = runner.get_status(sweep_id)
    if not status:
        raise HTTPException(status_code=404, detail="sweep not found")

    cfgs = status.get("configs", {})
    results: list[dict] = []
    for label, entry in cfgs.items():
        if entry.get("state") != "done":
            continue
        log_path = sweep_dir / "runs" / f"{label}.log"
        csv_path = sweep_dir / "runs" / f"{label}.csv"
        m = parse_run(log_path, csv_path)
        cfg_path = sweep_dir / "configs" / f"{label}.json"
        tp = pp = dp = 1
        pd_layout = "—"
        if cfg_path.exists():
            try:
                cluster = json.loads(cfg_path.read_text())
                instances = cluster.get("nodes", [{}])[0].get("instances", [])
                if instances:
                    npu_num = int(instances[0].get("npu_num", 1))
                    npu_group = int(instances[0].get("npu_group", 1)) or 1
                    tp = npu_num // npu_group
                    pp = npu_group
                    p = sum(1 for i in instances if i.get("pd_type") == "prefill")
                    d = sum(1 for i in instances if i.get("pd_type") == "decode")
                    if p and d:
                        pd_layout = f"{p}P+{d}D"
                        dp = max(d, 1)
                    else:
                        dp = len(instances)
            except (OSError, json.JSONDecodeError):
                pass
        results.append({
            "label": label,
            "tp": tp, "pp": pp, "dp": dp, "pd_layout": pd_layout,
            **m,
        })

    fig_json = pareto_scatter(results, x_metric=x_metric)
    return JSONResponse({"figure": json.loads(fig_json)})


@app.get("/api/sweeps/{sweep_id}/download/metrics")
async def api_download_metrics(sweep_id: str) -> FileResponse:
    sweep_dir = OUTPUT_DIR / sweep_id
    metrics_path = sweep_dir / "metrics.json"
    if not metrics_path.exists():
        # Trigger a recompute by walking the results page logic, then write.
        await page_results(sweep_id, None)  # type: ignore[arg-type]
    if not metrics_path.exists():
        raise HTTPException(status_code=404, detail="metrics.json not available")
    return FileResponse(
        path=str(metrics_path),
        media_type="application/json",
        filename=f"{sweep_id}-metrics.json",
    )


@app.get("/api/sweeps/{sweep_id}/download/zip")
async def api_download_zip(sweep_id: str) -> StreamingResponse:
    sweep_dir = OUTPUT_DIR / sweep_id
    if not sweep_dir.is_dir():
        raise HTTPException(status_code=404, detail="sweep not found")

    runs_dir = sweep_dir / "runs"
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        if runs_dir.is_dir():
            for f in sorted(runs_dir.iterdir()):
                if f.is_file() and f.suffix in (".csv", ".log"):
                    zf.write(f, arcname=f"runs/{f.name}")
        # Bundle scenario + status + metrics for context.
        for extra in ("scenario.json", "status.json", "metrics.json"):
            ep = sweep_dir / extra
            if ep.is_file():
                zf.write(ep, arcname=extra)
    buf.seek(0)

    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{sweep_id}.zip"',
        },
    )


# ---------------------------------------------------------------------------
# API: cluster config builder (list / load / save)
# ---------------------------------------------------------------------------

@app.get("/api/cluster-configs")
async def api_list_cluster_configs() -> JSONResponse:
    return JSONResponse({"configs": list_configs()})


@app.get("/api/cluster-configs/load")
async def api_load_cluster_config(path: str) -> JSONResponse:
    """path = 'cluster_config/foo.json'. Returns raw JSON dict for the form."""
    try:
        data = load_config(path)
    except (ValueError, OSError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    return JSONResponse({"path": path, "config": data})


@app.post("/api/cluster-configs")
async def api_save_cluster_config(request: Request) -> JSONResponse:
    """
    body = {
      "name": "my_config",
      "link_bw": 112, "link_latency": 0,
      "nodes": [
        {
          "cpu_mem": {"mem_size":128,"mem_bw":256,"mem_latency":0},
          "instances": [
            {"hardware":"A6000","model":"meta-llama/Llama-3.1-8B",
             "npu_num":1,"npu_group":1,"pd_type":null,"npu_mem":{}}
          ]
        }
      ]
    }
    """
    body = await request.json()
    try:
        name = sanitize_filename(body.get("name", ""))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    nodes_in = body.get("nodes") or []
    if not nodes_in:
        raise HTTPException(status_code=400, detail="at least one node required")

    catalog = build_catalog()  # rebuild each request — ~1ms, picks up newly profiled hardware without server restart
    instances_per_node: list[list[InstanceSpec]] = []
    cpu_mem_per_node: list[dict] = []
    flat_specs: list[InstanceSpec] = []

    for ni, node in enumerate(nodes_in):
        inst_list: list[InstanceSpec] = []
        for ii, inst in enumerate(node.get("instances") or []):
            try:
                inst_spec = InstanceSpec(
                    hardware=str(inst["hardware"]),
                    model=str(inst["model"]),
                    npu_num=int(inst["npu_num"]),
                    npu_group=int(inst["npu_group"]),
                    pd_type=inst.get("pd_type") or None,
                    npu_mem=inst.get("npu_mem") or {},
                )
            except (KeyError, TypeError, ValueError) as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"node#{ni} instance#{ii}: {e}",
                )
            inst_list.append(inst_spec)
            flat_specs.append(inst_spec)
        if not inst_list:
            raise HTTPException(status_code=400, detail=f"node#{ni} has no instances")
        instances_per_node.append(inst_list)
        cpu_mem_per_node.append({**CPU_MEM_DEFAULT, **(node.get("cpu_mem") or {})})

    dummy = ConfigSpec(
        label=name, instances=flat_specs,
        tp=0, pp=0, dp=0, pd_layout="—",
    )
    errors = validate_spec(dummy, catalog)
    if errors:
        raise HTTPException(status_code=400, detail={"errors": errors})

    cluster_json = build_cluster_json(
        dummy,
        cpu_mem=CPU_MEM_DEFAULT,
        link_bw=int(body.get("link_bw", LINK_BW_DEFAULT)),
        link_latency=int(body.get("link_latency", LINK_LATENCY_DEFAULT)),
        num_nodes=len(nodes_in),
        cpu_mem_per_node=cpu_mem_per_node,
        instances_per_node=instances_per_node,
    )
    try:
        rel = save_config(name, cluster_json)
    except (ValueError, OSError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    return JSONResponse({"path": rel, "saved": True})


@app.delete("/api/cluster-configs")
async def api_delete_cluster_config(path: str) -> JSONResponse:
    """Delete a user-saved config under cluster_config/web/.

    Reference configs outside cluster_config/web/ are rejected by
    delete_config() so the UI can't remove example/reference JSONs.
    """
    try:
        delete_config(path)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except (ValueError, OSError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    return JSONResponse({"path": path, "deleted": True})
