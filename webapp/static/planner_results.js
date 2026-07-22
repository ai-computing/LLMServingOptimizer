// Planner — results: Pareto chart (Plotly) + candidate table + report.
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const JOB = window.PLANNER_JOB_ID;
  const num = (s) => { const v = parseFloat(s); return Number.isFinite(v) ? v : null; };
  const truthy = (s) => String(s).toLowerCase() === "true";

  async function load() {
    const r = await fetch(`/api/planner/jobs/${JOB}/results`);
    const data = await r.json();
    if (!data.has_best_config) $("dl-config").style.display = "none";
    renderTable(data.candidates || []);
    renderPlot(data.candidates || []);
    $("report-md").textContent = data.report_md || "(no report)";
  }

  function renderTable(rows) {
    $("results-body").innerHTML = rows.map((c) => {
      const ok = c.status === "ok";
      const highlight = truthy(c.on_pareto) ? "row-highlighted" : "";
      return `<tr class="${highlight}">
        <td>${(c.run_id || "").replace("cand_", "").slice(0, 8)}</td>
        <td>${c.batch_tokens || ""}</td>
        <td>${truthy(c.on_pareto) ? "★" : ""}</td>
        <td>${truthy(c.passed) ? "✅" : (ok ? "❌" : "—")}</td>
        <td>${c.ttft_ms || "—"}</td>
        <td>${c.tpot_ms || "—"}</td>
        <td>${c.itl_p99_ms || "—"}</td>
        <td>${c.throughput_toks_s || "—"}</td>
        <td>${c.toks_per_wh || "—"}</td>
        <td>${c.score || "—"}</td>
      </tr>`;
    }).join("") || `<tr><td colspan="10" class="hint">no candidates</td></tr>`;
  }

  function renderPlot(rows) {
    const ok = rows.filter((c) => c.status === "ok" && num(c.throughput_toks_s) !== null);
    if (!ok.length) {
      $("plot-note").textContent = "No successfully-simulated candidates to plot.";
      return;
    }
    const haveEnergy = ok.some((c) => num(c.toks_per_wh) !== null);
    const yKey = haveEnergy ? "toks_per_wh" : "itl_p99_ms";
    const yLabel = haveEnergy ? "Energy efficiency (toks/Wh, higher better)"
                              : "ITL p99 (ms, lower better)";

    const groups = [
      { name: "Pareto front", filter: (c) => truthy(c.on_pareto), color: "#C44E52", size: 15, symbol: "star" },
      { name: "passed SLO",   filter: (c) => truthy(c.passed) && !truthy(c.on_pareto), color: "#4C72B0", size: 10, symbol: "circle" },
      { name: "failed SLO",   filter: (c) => !truthy(c.passed), color: "#999999", size: 9, symbol: "x" },
    ];
    const traces = groups.map((g) => {
      const pts = ok.filter(g.filter);
      return {
        x: pts.map((c) => num(c.throughput_toks_s)),
        y: pts.map((c) => num(c[yKey])),
        text: pts.map((c) => (c.run_id || "").replace("cand_", "").slice(0, 8) + ` (bt=${c.batch_tokens})`),
        mode: "markers", type: "scatter", name: g.name,
        marker: { color: g.color, size: g.size, symbol: g.symbol,
                  line: { width: 1, color: "#333" } },
      };
    }).filter((t) => t.x.length);

    Plotly.newPlot("pareto-plot", traces, {
      xaxis: { title: "Throughput (tok/s, higher better)" },
      yaxis: { title: yLabel },
      margin: { t: 20, r: 20, b: 50, l: 70 },
      legend: { orientation: "h", y: -0.2 },
      hovermode: "closest",
    }, { responsive: true, displayModeBar: false });
    $("plot-note").textContent = haveEnergy ? ""
      : "Toks/Wh unavailable (no energy objective) — showing ITL p99 on the Y axis.";
  }

  document.addEventListener("DOMContentLoaded", load);
})();
