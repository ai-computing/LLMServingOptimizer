/* Serving-request screen: form validation, progress stepper (SSE), BEST card
   + alternatives table, confirm modal with auto-deploy (plan §5.3). */
(function () {
  const $ = (id) => document.getElementById(id);
  let currentJob = null;

  fetch("/api/models").then(r => r.json()).then(d => {
    const models = Object.keys(d.models);
    $("rq-model").innerHTML = models.map(m => `<option>${svc.esc(m)}</option>`).join("");
    const hws = new Set();
    models.forEach(m => Object.keys(d.models[m]).forEach(h => hws.add(h)));
    $("rq-exclude").innerHTML = [...hws].map(h => `<option>${svc.esc(h)}</option>`).join("");
  });

  function step(n, note) {
    document.querySelectorAll(".step").forEach(el => {
      const s = +el.dataset.s;
      el.classList.toggle("done", s < n);
      el.classList.toggle("active", s === n);
    });
    if (note) $("rq-progress-note").textContent = note;
  }

  $("rq-form").addEventListener("submit", async (e) => {
    e.preventDefault();
    const slo = {};
    if ($("rq-ttft").value) slo.ttft_ms = +$("rq-ttft").value;
    if ($("rq-tpot").value) slo.tpot_ms = +$("rq-tpot").value;
    if ($("rq-itl").value) slo.itl_p99_ms = +$("rq-itl").value;
    const body = {
      model: $("rq-model").value, fp: +$("rq-fp").value,
      tenant: $("rq-tenant").value || "default", slo,
      scale: { req_per_s: +$("rq-rate").value, preset: $("rq-preset").value,
               duration_s: +$("rq-dur").value },
      num_req_eval: +$("rq-neval").value,
      auto_deploy: $("rq-autodeploy").checked,
      exclude_hw: [...$("rq-exclude").selectedOptions].map(o => o.value),
    };
    if ($("rq-cap").value) body.power_cap_w = +$("rq-cap").value;
    if ($("rq-force").value) body.force_backend = $("rq-force").value;

    $("rq-idle").style.display = "none";
    $("rq-result").style.display = "none";
    $("rq-progress-card").style.display = "";
    step(1, "제출 중…");
    const r = await fetch("/api/serve-requests", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body) });
    if (!r.ok) {
      const err = await r.json().catch(() => ({}));
      step(1, "검증 실패: " + JSON.stringify(err.detail || r.status));
      return;
    }
    const job = await r.json();
    svc.sse(`/api/serve-requests/${job.id}/events`, (ev) => {
      if (ev.type === "workload") step(3, `demand ${Math.round(ev.demand_toks_per_s)} toks/s`);
      if (ev.type === "routing") step(3, `백엔드: ${ev.backend} (${ev.confidence})`);
      if (ev.type === "stage1") step(4, `Stage-2 후보 ${ev.candidates ? ev.candidates.length : "?"}개 평가 중`);
      if (ev.type === "candidate") $("rq-progress-note").textContent += " ●";
      if (ev.type === "finished") step(5, "완료");
    }, () => poll(job.id));
  });

  async function poll(jobId) {
    const job = await (await fetch(`/api/serve-requests/${jobId}`)).json();
    if (job.state === "running" || job.state === "pending")
      return setTimeout(() => poll(jobId), 800);
    currentJob = job;
    step(5, "");
    render(job);
  }

  function gauge(label, val, limit) {
    if (val == null) return "";
    const cls = limit == null ? "" : (val <= limit ? "slo-ok" : "slo-violated");
    const margin = limit ? ` / SLO ${limit} (여유 ${Math.round((1 - val / limit) * 100)}%)` : "";
    return `<div>${label}: <strong class="${cls}">${svc.fmtMs(val)}</strong>${margin}</div>`;
  }

  function candRow(c, res) {
    const m = c.metrics || {};
    return `<tr data-run="${svc.esc(c.run_id)}">
      <td>${svc.esc(c.hw_summary)}</td>
      <td class="svc-num">${svc.fmtW(c.power_w)}</td>
      <td class="svc-num">${svc.fmtMs(m.ttft_ms)}</td>
      <td class="svc-num">${svc.fmtMs(m.tpot_ms)}</td>
      <td>${c.passed ? '<span class="slo-ok">통과</span>'
                     : '<span class="slo-violated">위반</span>'}</td></tr>`;
  }

  function render(job) {
    const res = job.result || {};
    const el = $("rq-result");
    el.style.display = "";
    $("rq-progress-card").style.display = "none";
    if (!res.best) {
      const inf = res.infeasible || {};
      el.innerHTML = `<div class="infeasible-card">
        <h3>달성 불가 — ${svc.esc(inf.bottleneck || "unknown")}</h3>
        <p>${svc.esc(inf.detail || job.error || "")}</p>
        <ul>${(inf.suggestions || []).map(s => `<li>${svc.esc(s)}</li>`).join("")}</ul></div>`;
      return;
    }
    const b = res.best, m = b.metrics || {};
    const slo = currentJob ? currentJob : {};
    const conf = { high: "HIGH", medium: "MED", low: "LOW" }[res.confidence] || res.confidence;
    el.innerHTML = `
      <div class="best-card">
        <div>
          <div class="best-power">${svc.fmtW(b.power_w)}</div>
          <span class="badge READY">${conf}</span>
          <div class="sub muted">${svc.esc(res.backend)} · ${svc.esc(b.power_source || "")}</div>
        </div>
        <div>
          <h3 style="margin:0 0 8px">${svc.esc(b.hw_summary)}</h3>
          ${gauge("TTFT", m.ttft_ms, null)} ${gauge("TPOT", m.tpot_ms, null)}
          <div class="muted">devices: ${res.device_ids.map(svc.esc).join(", ")}</div>
        </div>
        <div style="text-align:center">
          <button class="btn primary" id="rq-confirm">확정${$("rq-autodeploy").checked ? " + 배포" : ""}</button>
        </div>
      </div>
      ${res.alternatives.length ? `<h4>대안 (${res.alternatives.length})</h4>
      <table class="svc-table"><tr><th>구성</th><th>전력</th><th>TTFT</th><th>TPOT</th><th>SLO</th></tr>
      ${res.alternatives.map(c => candRow(c, res)).join("")}</table>` : ""}
      <p class="muted" style="font-size:.85em">이유: ${svc.esc(res.reason)}</p>`;
    $("rq-confirm").addEventListener("click", confirmFlow);
  }

  function confirmFlow() {
    const res = currentJob.result;
    svc.modal(`<h3>확정 및 예약</h3>
      <p>잠글 디바이스: <code>${res.device_ids.map(svc.esc).join(", ")}</code></p>
      <p>예상 전력: <strong>${svc.fmtW(res.best.power_w)}</strong> ·
         스냅샷 v${res.snapshot_ver}</p>`,
      async () => {
        const r = await fetch(`/api/serve-requests/${currentJob.id}/confirm`, { method: "POST" });
        const d = await r.json();
        if (d.replanned) {
          svc.toast("자원 충돌 — 재플래닝됨. 갱신안을 확인 후 다시 확정하세요.");
          poll(currentJob.id);
          return;
        }
        if (d.state === "confirmed") {
          svc.toast("예약 완료" + (d.deployment_id ? " · 배포 시작: " + d.deployment_id : ""));
          if (d.deployment_id)
            location.href = `/service/deployments#${d.deployment_id}`;
        } else {
          svc.toast("확정 실패: " + (d.detail || d.state));
        }
      }, { okLabel: "확정" });
  }
})();
