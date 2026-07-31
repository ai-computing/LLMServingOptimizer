/* Deployment list + detail dashboard (plan §5.4): KPI strip, Chart.js time
   series with SLO target lines, log viewer with follow, terminate flow with
   drain progress. */
(function () {
  const $ = (id) => document.getElementById(id);
  let dep = null, stream = null, charts = {}, logFollow = true;

  function uptime(d) {
    if (!d.ready_at) return "—";
    const end = d.terminated_at || Date.now() / 1000;
    const s = Math.max(0, end - d.ready_at);
    return s > 3600 ? (s / 3600).toFixed(1) + " h" : Math.round(s / 60) + " min";
  }

  async function loadList() {
    const incl = $("dp-incl-term").checked;
    const r = await fetch("/api/deployments?include_terminated=" + incl);
    const rows = (await r.json()).deployments;
    $("dp-updated").textContent = "갱신 " + new Date().toLocaleTimeString();
    const tbl = $("dp-table");
    tbl.querySelectorAll("tr:not(:first-child)").forEach(x => x.remove());
    rows.forEach(d => {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td><span class="badge ${d.state}">${d.state}</span></td>
        <td><a href="#${d.id}">${svc.esc(d.id)}</a><br><span class="muted">${svc.esc(d.model)}</span></td>
        <td>${svc.esc(d.tenant)}</td><td>${d.device_ids.map(svc.esc).join("<br>")}</td>
        <td class="svc-num" data-power>—</td><td>${uptime(d)}</td>
        <td><button class="btn" data-open="${svc.esc(d.id)}">상세</button></td>`;
      tbl.appendChild(tr);
    });
    tbl.querySelectorAll("[data-open]").forEach(b =>
      b.addEventListener("click", () => openDetail(b.dataset.open)));
  }

  function mkChart(canvas, datasets, sloLine) {
    const cfg = {
      type: "line",
      data: { labels: [], datasets },
      options: { animation: false, scales: { x: { display: false } },
                 plugins: { legend: { labels: { boxWidth: 12 } } } } };
    if (sloLine != null) {
      cfg.data.datasets.push({ label: "SLO", data: [], borderDash: [6, 4],
        borderColor: getComputedStyle(document.documentElement)
          .getPropertyValue("--slo-violated"), pointRadius: 0 });
    }
    return new Chart(canvas, cfg);
  }

  async function openDetail(depId) {
    const r = await fetch("/api/deployments/" + depId);
    if (!r.ok) { svc.toast("배포를 찾을 수 없음: " + depId); return; }
    dep = await r.json();
    location.hash = depId;
    $("dp-list-card").style.display = "none";
    $("dp-detail").style.display = "";
    $("dd-id").textContent = dep.id;
    $("dd-model").textContent = dep.model;
    setState(dep.state);
    $("dd-endpoint").textContent = dep.endpoints ? dep.endpoints.openai_url : "—";
    $("dd-config").textContent = JSON.stringify(dep.spec, null, 2);
    renderEvents(dep.events);
    if (dep.state === "FAILED") {
      $("dd-logs").innerHTML = `<span class="err">last_error: ${svc.esc(dep.last_error || "")}</span>\n`;
      document.querySelector('[data-tab="logs"]').click();
    }
    charts.thr && charts.thr.destroy(); charts.lat && charts.lat.destroy();
    charts.thr = mkChart($("dd-chart-thr"), [
      { label: "gen tok/s", data: [], borderColor: "#1E88E5", pointRadius: 0 },
      { label: "대기열", data: [], borderColor: "#FB8C00", pointRadius: 0 }]);
    const sloTpot = dep.slo && dep.slo.tpot_ms;
    charts.lat = mkChart($("dd-chart-lat"), [
      { label: "TTFT p95 (ms)", data: [], borderColor: "#7b1fa2", pointRadius: 0 },
      { label: "TPOT p95 (ms)", data: [], borderColor: "#2E7D32", pointRadius: 0 }],
      sloTpot);
    stream && stream.stop();
    stream = svc.sse(`/api/deployments/${dep.id}/metrics`, onSample, onEndStream);
    svc.sse(`/api/deployments/${dep.id}/logs?tail=500`, onLog, () => {});
  }

  function setState(s) {
    const el = $("dd-state");
    el.className = "badge " + s; el.textContent = s;
  }

  function onSample(ev) {
    const m = ev.metrics;
    setState(ev.state);
    $("k-thr").textContent = svc.fmtTok(m.gen_toks_per_s);
    $("k-ttft").textContent = svc.fmtMs(m.ttft_p95_ms);
    $("k-tpot").textContent = svc.fmtMs(m.tpot_p95_ms);
    $("k-power").textContent = svc.fmtW(ev.power_w);
    $("dd-energy").textContent = svc.fmtWh(ev.energy_wh);
    $("dd-kv").value = m.kv_cache_usage;
    $("dd-kv-t").textContent = Math.round(m.kv_cache_usage * 100) + "%";
    $("dd-wait").textContent = m.waiting; $("dd-run").textContent = m.running;
    if (dep.slo) {
      if (dep.slo.tpot_ms) $("k-tpot-sub").textContent = "목표 ≤ " + dep.slo.tpot_ms;
      if (dep.slo.ttft_ms) $("k-ttft-sub").textContent = "목표 ≤ " + dep.slo.ttft_ms;
    }
    const best = dep.spec && dep.spec._predicted_power_w;
    if (best && ev.power_w)
      $("k-power-card").classList.toggle("warn",
        Math.abs(ev.power_w - best) / best > 0.10);
    if (ev.slo) {
      $("dd-slo").className = "badge " +
        ({ ok: "READY", warn: "DEGRADED", violated: "FAILED" }[ev.slo.verdict] || "");
      $("dd-slo").textContent = ev.slo.verdict;
      $("dd-slo-detail").textContent = (ev.slo.breaches || []).join("; ");
    }
    const t = new Date(m.ts * 1000).toLocaleTimeString();
    push(charts.thr, t, [m.gen_toks_per_s, m.waiting]);
    push(charts.lat, t, [m.ttft_p95_ms, m.tpot_p95_ms,
                         dep.slo && dep.slo.tpot_ms]);
  }

  function push(chart, label, vals) {
    chart.data.labels.push(label);
    vals.forEach((v, i) => chart.data.datasets[i] &&
      chart.data.datasets[i].data.push(v == null ? null : v));
    if (chart.data.labels.length > 720) {
      chart.data.labels.shift();
      chart.data.datasets.forEach(d => d.data.shift());
    }
    chart.update("none");
  }

  function onEndStream(ev) {
    setState(ev.state);
    if (ev.summary)
      svc.toast(`종료 · 총 에너지 ${svc.fmtWh(ev.summary.energy_wh)} · 평균 ${svc.fmtW(ev.summary.avg_power_w)}`);
    loadList();
  }

  function onLog(ev) {
    const view = $("dd-logs");
    const filter = $("dd-log-filter").value;
    if (filter && !ev.line.includes(filter)) return;
    const cls = /ERROR|Traceback/.test(ev.line) ? "err"
      : /WARN/i.test(ev.line) ? "warn" : "";
    view.insertAdjacentHTML("beforeend",
      `<span class="${cls}">${svc.esc(ev.line)}</span>\n`);
    while (view.childNodes.length > 5000) view.removeChild(view.firstChild);
    if (logFollow) view.scrollTop = view.scrollHeight;
  }

  $("dd-logs") && $("dd-logs").addEventListener("scroll", () => {
    const v = $("dd-logs");
    const atBottom = v.scrollHeight - v.scrollTop - v.clientHeight < 20;
    if (!atBottom) { logFollow = false; $("dd-follow").checked = false; }
  });
  $("dd-follow").addEventListener("change", e => logFollow = e.target.checked);

  function renderEvents(events) {
    $("dd-events").innerHTML = "<tr><th>시각</th><th>전이</th><th>detail</th></tr>" +
      (events || []).map(e =>
        `<tr><td>${new Date(e.ts * 1000).toLocaleTimeString()}</td>
         <td>${svc.esc(e.from_state)} → ${svc.esc(e.to_state)}</td>
         <td>${svc.esc(e.detail || "")}</td></tr>`).join("");
  }

  document.querySelectorAll(".dd-tab").forEach(a =>
    a.addEventListener("click", (e) => {
      e.preventDefault();
      document.querySelectorAll(".dd-tab").forEach(x => x.classList.remove("active"));
      a.classList.add("active");
      ["logs", "events", "config"].forEach(t =>
        $("dd-pane-" + t).style.display = t === a.dataset.tab ? "" : "none");
    }));

  $("dd-back").addEventListener("click", () => {
    stream && stream.stop();
    location.hash = "";
    $("dp-detail").style.display = "none";
    $("dp-list-card").style.display = "";
    loadList();
  });
  $("dd-copy").addEventListener("click", () =>
    navigator.clipboard.writeText($("dd-endpoint").textContent));

  $("dd-terminate").addEventListener("click", () => {
    const inflight = +$("dd-run").textContent + +$("dd-wait").textContent;
    const m = svc.modal(`<h3>배포 종료 — ${svc.esc(dep.id)}</h3>
      <p>in-flight 요청: <strong>${inflight}</strong></p>
      <p><label>drain 제한 (s) <input id="tm-drain" type="number" value="120" min="0"></label></p>
      <p class="muted">P0 한계: 게이트웨이가 없어 신규 유입 차단은 불가하며,
         대기열 소진을 제한 시간까지 기다립니다.</p>
      <p><label><input type="checkbox" id="tm-ck"> <strong>${svc.esc(dep.id)}</strong> 를 종료합니다</label></p>`,
      async (back) => {
        const drain = +document.querySelector("#tm-drain")?.value || 120;
        await fetch(`/api/deployments/${dep.id}/terminate`, {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ drain_timeout_s: drain }) });
        setState("DRAINING");
        svc.toast("DRAINING 시작…");
      }, { okLabel: "종료", confirmDisabled: true });
    m.querySelector("#tm-ck").addEventListener("change", (e) =>
      m.querySelector('[data-act="ok"]').disabled = !e.target.checked);
  });

  $("dp-incl-term").addEventListener("change", loadList);
  loadList();
  setInterval(() => { if ($("dp-detail").style.display === "none") loadList(); }, 10000);
  if (location.hash.length > 1) openDetail(location.hash.slice(1));
})();
