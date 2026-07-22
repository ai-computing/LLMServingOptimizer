"""Build Plotly figures for sweep results.

Each function returns a JSON string (figure.to_json()) suitable for embedding
in an HTML page that uses Plotly.js.

Result records (input to every function) are flat dicts with keys including:
    label, tp, pp, dp, pd_layout, total_token_tp, mean_ttft_ms,
    mean_tpot_ms, mean_itl_ms, ttft_values_ms (list), itl_values_ms (list)
"""
from __future__ import annotations

import plotly.graph_objects as go
import plotly.express as px

# parallelism family colors — used only for line/pareto-frontier accents
# and as a fallback. Per-config colors are assigned from CONFIG_PALETTE.
FAMILY_COLORS = {
    "tp":       "#1f77b4",
    "pp":       "#ff7f0e",
    "dp":       "#2ca02c",
    "mixed":    "#9467bd",
    "pd":       "#d62728",
    "baseline": "#7f7f7f",
}

# 48-color qualitative palette so each config in a sweep gets a distinct
# color across all charts. We concatenate Dark24 + Light24 — together they
# cover dark and light shades while staying perceptually distinguishable.
CONFIG_PALETTE: list[str] = list(px.colors.qualitative.Dark24) + list(px.colors.qualitative.Light24)


def assign_config_colors(labels: list[str]) -> dict[str, str]:
    """Map each unique label to a stable color from CONFIG_PALETTE.

    Order of first appearance determines the assignment, so the same config
    receives the same color across bar charts, scatter, CDFs, etc.
    """
    out: dict[str, str] = {}
    for lbl in labels:
        if lbl not in out:
            out[lbl] = CONFIG_PALETTE[len(out) % len(CONFIG_PALETTE)]
    return out

# Metrics that appear as bar charts, with display titles + y-axis units.
_BAR_METRICS = {
    "total_token_tp":  ("Total token throughput", "tok/s"),
    "mean_ttft_ms":    ("Mean TTFT",              "ms"),
    "mean_tpot_ms":    ("Mean TPOT",              "ms"),
    "mean_itl_ms":     ("Mean ITL",               "ms"),
    "total_energy_wh": ("Total energy",           "Wh"),
}


def classify_family(axes: dict) -> str:
    """Classify a result by which parallelism axis dominates.

    `axes` is a dict-like with keys tp, pp, dp, pd_layout.
    """
    tp = int(axes.get("tp", 1) or 1)
    pp = int(axes.get("pp", 1) or 1)
    dp = int(axes.get("dp", 1) or 1)
    pd_layout = axes.get("pd_layout") or "—"

    if pd_layout and pd_layout != "—":
        return "pd"
    nontrivial = sum(1 for v in (tp, pp, dp) if v > 1)
    if nontrivial == 0:
        return "baseline"
    if nontrivial > 1:
        return "mixed"
    if tp > 1:
        return "tp"
    if pp > 1:
        return "pp"
    if dp > 1:
        return "dp"
    return "baseline"


def _family_for_record(rec: dict) -> str:
    return classify_family({
        "tp": rec.get("tp", 1),
        "pp": rec.get("pp", 1),
        "dp": rec.get("dp", 1),
        "pd_layout": rec.get("pd_layout", "—"),
    })


def _empty_fig(title: str) -> str:
    fig = go.Figure()
    fig.update_layout(title=title, annotations=[
        {"text": "No data", "showarrow": False,
         "xref": "paper", "yref": "paper", "x": 0.5, "y": 0.5}
    ])
    return fig.to_json()


def bar_charts(results: list[dict]) -> dict[str, str]:
    """Per-metric bar charts (one per key in _BAR_METRICS).

    Returns dict of metric_name -> Plotly JSON string.
    """
    out: dict[str, str] = {}
    if not results:
        for metric in _BAR_METRICS:
            out[metric] = _empty_fig(_BAR_METRICS[metric][0])
        return out

    labels = [r.get("label", "") for r in results]
    color_map = assign_config_colors(labels)
    bar_colors = [color_map[lbl] for lbl in labels]

    for metric, (title, unit) in _BAR_METRICS.items():
        fig = go.Figure()
        # Single trace with per-bar colors so every config is visually distinct.
        # X-axis already labels each bar, so the legend is redundant.
        fig.add_trace(go.Bar(
            x=labels,
            y=[r.get(metric) for r in results],
            marker_color=bar_colors,
            showlegend=False,
        ))
        fig.update_layout(
            title=f"{title} ({unit})",
            xaxis_title="Configuration",
            yaxis_title=f"{title} ({unit})",
            # Long heterogeneous labels (e.g. het_p1x1_A6000__c1x1_RNGD)
            # squash the plot when rendered on the x-axis. Hide tick labels
            # and rely on the shared Config Legend card for color → label.
            xaxis=dict(showticklabels=False),
        )
        out[metric] = fig.to_json()
    return out


def _pareto_frontier(points: list[tuple[float, float]]) -> list[int]:
    """Return indices of Pareto-optimal points (lower x, higher y).

    A point is dominated if some other point has x' <= x AND y' >= y AND
    (x' < x OR y' > y).
    """
    n = len(points)
    keep: list[int] = []
    for i, (xi, yi) in enumerate(points):
        dominated = False
        for j, (xj, yj) in enumerate(points):
            if i == j:
                continue
            if xj <= xi and yj >= yi and (xj < xi or yj > yi):
                dominated = True
                break
        if not dominated:
            keep.append(i)
    return keep


def pareto_scatter(results: list[dict], x_metric: str = "mean_ttft_ms") -> str:
    """Scatter of x_metric vs total_token_tp with Pareto frontier line."""
    if not results:
        return _empty_fig("Pareto scatter")

    pts: list[tuple[float, float]] = []
    labels: list[str] = []
    families: list[str] = []
    valid: list[int] = []
    for i, r in enumerate(results):
        x = r.get(x_metric)
        y = r.get("total_token_tp")
        if x is None or y is None:
            continue
        pts.append((float(x), float(y)))
        labels.append(r.get("label", ""))
        families.append(_family_for_record(r))
        valid.append(i)

    if not pts:
        return _empty_fig("Pareto scatter")

    fig = go.Figure()
    color_map = assign_config_colors(labels)
    # One trace per config — color identifies the config (matched in the
    # shared Config Legend card). Drop the in-plot text labels and hide
    # per-config legend entries; hover still reveals the name.
    for i, lbl in enumerate(labels):
        fig.add_trace(go.Scatter(
            x=[pts[i][0]], y=[pts[i][1]],
            mode="markers",
            marker=dict(size=10, color=color_map[lbl]),
            name=lbl,
            showlegend=False,
        ))

    keep = _pareto_frontier(pts)
    if keep:
        # Order frontier by x ascending for a clean line.
        keep_sorted = sorted(keep, key=lambda i: pts[i][0])
        fx = [pts[i][0] for i in keep_sorted]
        fy = [pts[i][1] for i in keep_sorted]
        fig.add_trace(go.Scatter(
            x=fx, y=fy, mode="lines",
            line=dict(color="black", dash="dash"),
            name="Pareto frontier",
            showlegend=True,
        ))

    fig.update_layout(
        title=f"Throughput vs {x_metric}",
        xaxis_title=x_metric,
        yaxis_title="Total token throughput (tok/s)",
    )
    return fig.to_json()


def _mode(values: list) -> object:
    """Most common value in `values` (first wins on ties)."""
    counts: dict = {}
    order: list = []
    for v in values:
        if v not in counts:
            counts[v] = 0
            order.append(v)
        counts[v] += 1
    if not order:
        return None
    best = order[0]
    for v in order:
        if counts[v] > counts[best]:
            best = v
    return best


def axis_line_charts(results: list[dict]) -> dict[str, str]:
    """One line chart per axis (tp, pp, dp).

    For each axis: hold the other two axes at their most common value, then
    plot total_token_tp vs the varying axis.
    """
    out: dict[str, str] = {}
    axes = ["tp", "pp", "dp"]
    if not results:
        for ax in axes:
            out[ax] = _empty_fig(f"Throughput vs {ax}")
        return out

    for ax in axes:
        others = [a for a in axes if a != ax]
        # Filter to records that have non-PD layout (P/D mixes axis semantics).
        candidates = [
            r for r in results
            if (r.get("pd_layout") in (None, "—", ""))
            and r.get("total_token_tp") is not None
        ]
        if not candidates:
            out[ax] = _empty_fig(f"Throughput vs {ax}")
            continue

        # Anchor other axes at their mode across the candidate set.
        anchor = {a: _mode([c.get(a, 1) for c in candidates]) for a in others}
        line_pts = [
            r for r in candidates
            if all(r.get(a) == anchor[a] for a in others)
        ]
        line_pts.sort(key=lambda r: int(r.get(ax, 1) or 1))

        fig = go.Figure()
        if line_pts:
            xs = [int(r.get(ax, 1) or 1) for r in line_pts]
            ys = [float(r.get("total_token_tp")) for r in line_pts]
            ls = [r.get("label", "") for r in line_pts]
            # Drop in-plot text labels — config names live in the shared
            # legend. Hover still shows them via customdata.
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines+markers",
                customdata=ls,
                hovertemplate="%{customdata}<br>%{xaxis.title.text}=%{x}<br>tok/s=%{y:.2f}<extra></extra>",
                marker_color=FAMILY_COLORS.get(ax, "#888"),
                name=f"{ax} sweep ({', '.join(f'{a}={anchor[a]}' for a in others)})",
            ))
        else:
            fig.add_annotation(text="No data", showarrow=False,
                               xref="paper", yref="paper", x=0.5, y=0.5)

        fig.update_layout(
            title=f"Throughput vs {ax} (other axes held at mode)",
            xaxis_title=ax,
            yaxis_title="Total token throughput (tok/s)",
        )
        out[ax] = fig.to_json()
    return out


def _cdf(values: list[float]) -> tuple[list[float], list[float]]:
    if not values:
        return [], []
    sv = sorted(values)
    n = len(sv)
    ys = [(i + 1) / n for i in range(n)]
    return sv, ys


def cdf_charts(results: list[dict]) -> dict[str, str]:
    """CDFs for ttft_values_ms and itl_values_ms (one trace per config)."""
    out: dict[str, str] = {}
    labels = [r.get("label", "") for r in results]
    color_map = assign_config_colors(labels)
    for kind, key in (("ttft", "ttft_values_ms"), ("itl", "itl_values_ms")):
        traces = []
        for r in results:
            vals = r.get(key)
            if not vals:
                continue
            xs, ys = _cdf(list(vals))
            if not xs:
                continue
            lbl = r.get("label", "")
            traces.append(go.Scatter(
                x=xs, y=ys, mode="lines",
                name=lbl,
                line=dict(color=color_map[lbl]),
                showlegend=False,  # use shared Config Legend card
            ))

        if not traces:
            out[kind] = _empty_fig(f"{kind.upper()} CDF")
            continue

        fig = go.Figure(data=traces)
        fig.update_layout(
            title=f"{kind.upper()} CDF",
            xaxis_title=f"{kind.upper()} (ms)",
            yaxis_title="Cumulative fraction",
        )
        out[kind] = fig.to_json()
    return out


def all_plots(results: list[dict]) -> dict[str, str]:
    """Return every plot JSON keyed for direct template consumption."""
    bars = bar_charts(results)
    return {
        **bars,
        "pareto": pareto_scatter(results),
        **{f"line_{k}": v for k, v in axis_line_charts(results).items()},
        **{f"cdf_{k}": v for k, v in cdf_charts(results).items()},
    }
