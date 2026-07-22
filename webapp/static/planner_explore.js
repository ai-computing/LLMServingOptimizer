// Planner — scenario builder. Builds a PlannerSpec from the form and launches a job.
(function () {
  "use strict";
  const $ = (id) => document.getElementById(id);
  let CATALOG = {};       // {hw: {models:{model:[tp...]}, default_mem_gb}}

  async function loadCatalog() {
    const r = await fetch("/api/planner/catalog");
    const data = await r.json();
    CATALOG = data.hardware || {};
    // model union across hardware
    const models = new Set();
    Object.values(CATALOG).forEach((hw) =>
      Object.keys(hw.models || {}).forEach((m) => models.add(m)));
    const sel = $("model-select");
    sel.innerHTML = "";
    [...models].sort().forEach((m) => sel.add(new Option(m, m)));
    addDeviceRow();  // seed one row now that hardware is known
  }

  async function loadDatasets() {
    try {
      const r = await fetch("/api/datasets");
      const data = await r.json();
      const sel = $("dataset-select");
      sel.innerHTML = "";
      (data || []).forEach((d) => {
        const path = d.path || d;
        sel.add(new Option(path, path));
      });
    } catch (e) { /* datasets optional */ }
  }

  function hwOptions() {
    return Object.keys(CATALOG).sort();
  }

  function addDeviceRow() {
    const body = $("device-body");
    const tr = document.createElement("tr");
    const hwSel = document.createElement("select");
    hwOptions().forEach((hw) => hwSel.add(new Option(hw, hw)));
    hwSel.addEventListener("change", () => { memInput.value = CATALOG[hwSel.value]?.default_mem_gb || 48; });

    const nodeInput = Object.assign(document.createElement("input"), { type: "text", value: "node0" });
    const countInput = Object.assign(document.createElement("input"), { type: "number", min: "1", value: "2" });
    const memInput = Object.assign(document.createElement("input"), { type: "number", min: "1", value: CATALOG[hwSel.value]?.default_mem_gb || 48 });
    const rm = Object.assign(document.createElement("button"), { textContent: "✕", className: "btn-secondary", type: "button" });
    rm.addEventListener("click", () => tr.remove());

    [hwSel, nodeInput, countInput, memInput, rm].forEach((el) => {
      const td = document.createElement("td");
      td.appendChild(el);
      tr.appendChild(td);
    });
    body.appendChild(tr);
  }

  function readDevices() {
    const rows = [...$("device-body").querySelectorAll("tr")];
    return rows.map((tr) => {
      const [hw, node, count, mem] = tr.querySelectorAll("select,input");
      return { name: hw.value, node: node.value.trim() || "node0",
               count: parseInt(count.value, 10), mem_gb: parseFloat(mem.value) };
    });
  }

  function checkedValues(cls) {
    return [...document.querySelectorAll("." + cls)]
      .filter((c) => c.checked).map((c) => parseInt(c.value, 10));
  }

  function parseRange(s, fallback) {
    const parts = String(s).split(",").map((x) => parseInt(x.trim(), 10));
    return parts.length === 2 && parts.every(Number.isFinite) ? parts : fallback;
  }

  function buildSpec() {
    const devices = readDevices();
    // group by node
    const byNode = {};
    devices.forEach((d) => {
      (byNode[d.node] = byNode[d.node] || []).push({ name: d.name, count: d.count, mem_gb: d.mem_gb });
    });
    const nodeIds = Object.keys(byNode);
    const nodes = nodeIds.map((id) => ({ id, devices: byNode[id] }));

    // full-mesh links between distinct nodes (enables Max-Flow constraint)
    const links = [];
    for (let i = 0; i < nodeIds.length; i++)
      for (let j = i + 1; j < nodeIds.length; j++)
        links.push({ src: nodeIds[i], dst: nodeIds[j], bandwidth: $("link-bw").value, latency: "0.0005ms" });

    const req = { objectives: [] };
    if ($("c-ttft").checked) req.ttft_ms = { constraint: "<=", value: parseFloat($("v-ttft").value) };
    if ($("c-tpot").checked) req.tpot_ms = { constraint: "<=", value: parseFloat($("v-tpot").value) };
    if ($("c-itl").checked)  req.itl_p99_ms = { constraint: "<=", value: parseFloat($("v-itl").value) };
    if ($("o-thr").checked) req.objectives.push({ metric: "throughput", direction: "max", weight: parseFloat($("w-thr").value) });
    if ($("o-tpw").checked) req.objectives.push({ metric: "toks_per_wh", direction: "max", weight: parseFloat($("w-tpw").value) });

    const pd = $("pd-toggle").checked;
    return {
      model: { name: $("model-select").value, fp: parseInt($("fp-select").value, 10) },
      workload: { dataset: $("dataset-select").value, num_req: parseInt($("num-req").value, 10) },
      topology: { nodes, links, intra_node_bandwidth: $("intra-bw").value },
      requirements: req,
      search_space: {
        pd_disaggregation: pd,
        tp_choices: checkedValues("tp-choice"),
        pp_choices: [1],
        xpyd_prefill_range: parseRange($("xpyd-p").value, [1, 4]),
        xpyd_decode_range: parseRange($("xpyd-d").value, [1, 6]),
        batch_tokens_choices: checkedValues("bt-choice"),
      },
      solver: {
        top_k: parseInt($("top-k").value, 10),
        time_limit_sec: parseInt($("time-limit").value, 10),
        pareto_epsilon_steps: parseInt($("eps-steps").value, 10),
      },
    };
  }

  function showProblems(list) {
    const box = $("problems");
    if (!list || !list.length) { box.style.display = "none"; return; }
    box.style.display = "block";
    box.innerHTML = "<strong>Validation:</strong><ul>" +
      list.map((p) => `<li>${p}</li>`).join("") + "</ul>";
  }

  async function validate() {
    showProblems(null);
    const r = await fetch("/api/planner/validate", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(buildSpec()),
    });
    const data = await r.json();
    if (data.ok) showProblems(["✓ spec is valid against the repo."]);
    else showProblems(data.problems || ["invalid spec"]);
    return data.ok;
  }

  async function run() {
    showProblems(null);
    $("btn-run").disabled = true;
    try {
      const body = {
        spec: buildSpec(),
        jobs: parseInt($("jobs").value, 10),
        dry_run: $("dry-run").checked,
      };
      const r = await fetch("/api/planner/jobs", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await r.json();
      if (!r.ok) { showProblems([data.detail || "failed to create job"]); return; }
      window.location.href = `/planner/jobs/${data.job_id}`;
    } finally {
      $("btn-run").disabled = false;
    }
  }

  document.addEventListener("DOMContentLoaded", () => {
    $("btn-add-device").addEventListener("click", addDeviceRow);
    $("btn-validate").addEventListener("click", validate);
    $("btn-run").addEventListener("click", run);
    $("pd-toggle").addEventListener("change", (e) => {
      $("xpyd-row").style.display = e.target.checked ? "flex" : "none";
    });
    loadCatalog();
    loadDatasets();
  });
})();
