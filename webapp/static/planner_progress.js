// Planner — live progress via SSE polling of status.json.
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  const JOB = window.PLANNER_JOB_ID;

  const fmt = (v) => (v === null || v === undefined || v === "") ? "—"
    : (typeof v === "number" ? v.toFixed(v < 100 ? 2 : 1) : v);

  function stateBadge(state) {
    const cls = { queued: "badge-queued", running: "badge-running",
      done: "badge-done", failed: "badge-failed" }[state] || "badge-unknown";
    return `badge ${cls}`;
  }

  function render(st) {
    const stateEl = $("state");
    stateEl.textContent = st.state || "queued";
    stateEl.className = stateBadge(st.state);

    const cands = st.candidates || {};
    const ids = Object.keys(cands);
    const done = ids.filter((k) => cands[k].state !== "pending").length;
    $("counts").textContent =
      `${ids.length} candidate(s), ${done} evaluated` +
      (st.num_passed != null ? `, ${st.num_passed} passed SLO` : "");

    const body = $("cand-body");
    if (!ids.length) return;
    const paretoSet = new Set(st.pareto || []);
    body.innerHTML = ids.map((id) => {
      const c = cands[id];
      const m = c.metrics || {};
      const stateCell = c.state === "pending"
        ? '<span class="badge badge-running">running</span>'
        : (c.state === "infeasible"
          ? `<span class="badge badge-failed" title="${c.reason || ""}">infeasible</span>`
          : '<span class="badge badge-done">done</span>');
      const slo = c.passed === true ? "✅" : (c.passed === false && c.state === "done" ? "❌" : "");
      const star = paretoSet.has(id) ? " ★" : "";
      return `<tr class="${st.best_run_id === id ? 'best-cell' : ''}">
        <td>${id.replace('cand_', '').slice(0, 8)}${star}</td>
        <td>${c.hw_summary || ""}</td>
        <td>${c.batch_tokens || ""}</td>
        <td>${stateCell}</td>
        <td>${fmt(m.ttft_ms)}</td>
        <td>${fmt(m.tpot_ms)}</td>
        <td>${fmt(m.itl_p99_ms)}</td>
        <td>${fmt(m.throughput_toks_s)}</td>
        <td>${fmt(m.toks_per_wh)}</td>
        <td>${slo}</td>
      </tr>`;
    }).join("");
  }

  function connect() {
    const es = new EventSource(`/api/planner/jobs/${JOB}/events`);
    es.onmessage = (e) => {
      let st;
      try { st = JSON.parse(e.data); } catch (_) { return; }
      if (st.type === "timeout") { es.close(); return; }
      render(st);
      if (st.state === "done" || st.state === "failed") {
        es.close();
        if (st.state === "done") $("results-link").style.display = "inline-block";
      }
    };
    es.onerror = () => { es.close(); setTimeout(connect, 3000); };
  }

  document.addEventListener("DOMContentLoaded", connect);
})();
