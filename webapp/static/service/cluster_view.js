/* Cluster view: deterministic rack-style topology renderer (plan §5.2).

   Layout: hosts on a grid, devices on a compact grid inside each host box.
   (A free d3-force layout with a per-host X hint smeared each host's devices
   vertically, so the host hull — a bounding box of its members — grew to
   1300+px on a 520px canvas at 66 devices. Positions are computed instead;
   d3 is still used for zoom/drag/selection.)

   Link density is filtered: an all-to-all NIC mesh renders as one labelled
   fabric hub instead of C(n,2) crossing lines, and the intra-node PCIe mesh
   is hidden unless asked for.  Highlight mode: #highlight=id1,id2 */
(function () {
  const svgEl = document.getElementById("svc-graph");
  const svg = d3.select(svgEl);
  const tip = document.createElement("div");
  tip.className = "svc-tooltip";
  tip.style.display = "none";
  document.body.appendChild(tip);

  const highlight = new Set(
    (location.hash.match(/highlight=([^&]+)/) || [, ""])[1]
      .split(",").filter(Boolean).map(decodeURIComponent));

  // slot geometry — gx leaves room for the intra-pair (NVLink) connector to show
  const DEV = { w: 46, h: 30, gx: 11, gy: 7 };
  const PAD = { x: 12, y: 10, title: 18 };
  const HOST_GAP = { x: 20, y: 34 };   // x is the MINIMUM gap (see layout)
  const MARGIN = 16;
  const NIC_R = 7;

  const POS_KEY = "svc-graph-pos";
  let overrides = {};
  try { overrides = JSON.parse(localStorage.getItem(POS_KEY) || "{}"); } catch (e) { }

  function saveOverrides() {
    localStorage.setItem(POS_KEY, JSON.stringify(overrides));
  }

  // ---- layout ---------------------------------------------------------------
  function layout(hosts, devices) {
    const byHost = d3.group(devices, (d) => d.host);
    const cells = hosts
      .filter((h) => byHost.has(h.id))
      .map((h) => {
        const devs = byHost.get(h.id);
        const cols = Math.min(4, devs.length);
        const rows = Math.ceil(devs.length / cols);
        return {
          host: h, devs, cols, rows,
          w: Math.max(150, cols * DEV.w + (cols - 1) * DEV.gx + 2 * PAD.x),
          h: rows * DEV.h + (rows - 1) * DEV.gy + 2 * PAD.y + PAD.title,
        };
      });
    if (!cells.length) return { cells: [], width: 200, height: 120 };

    const cellW = d3.max(cells, (c) => c.w);
    const cellH = d3.max(cells, (c) => c.h);
    const avail = (svgEl.clientWidth || 900) - 2 * MARGIN;
    // fit as many host columns as the panel allows at the minimum gap, then
    // spread the leftover space evenly (a fixed gap wasted a whole column)
    const hcols = Math.max(1, Math.min(cells.length,
      Math.floor((avail + HOST_GAP.x) / (cellW + HOST_GAP.x))));
    const gapX = hcols > 1
      ? Math.max(HOST_GAP.x, (avail - hcols * cellW) / (hcols - 1))
      : HOST_GAP.x;

    cells.forEach((c, i) => {
      const col = i % hcols, row = Math.floor(i / hcols);
      c.x = MARGIN + col * (cellW + gapX);
      c.y = MARGIN + row * (cellH + HOST_GAP.y);
      c.devs.forEach((d, k) => {
        const cx = k % c.cols, cy = Math.floor(k / c.cols);
        d.x = c.x + PAD.x + cx * (DEV.w + DEV.gx) + DEV.w / 2;
        d.y = c.y + PAD.title + PAD.y + cy * (DEV.h + DEV.gy) + DEV.h / 2;
        if (overrides[d.id]) { d.x = overrides[d.id][0]; d.y = overrides[d.id][1]; }
      });
      c.nic = { x: c.x + cellW / 2, y: c.y + cellH + 12 };
    });
    const rows = Math.ceil(cells.length / hcols);
    return {
      cells, hcols,
      width: 2 * MARGIN + hcols * cellW + (hcols - 1) * gapX,
      height: 2 * MARGIN + rows * cellH + (rows - 1) * HOST_GAP.y + 24,
    };
  }

  /** Complete NIC mesh (same kind+bandwidth) -> one fabric hub + spokes. */
  function fabricOf(interEdges) {
    if (interEdges.length < 3) return null;
    const nics = new Set();
    interEdges.forEach((e) => { nics.add(e.a); nics.add(e.b); });
    const n = nics.size;
    if (interEdges.length !== (n * (n - 1)) / 2) return null;
    const kinds = new Set(interEdges.map((e) => e.kind + "|" + e.bandwidth));
    if (kinds.size !== 1) return null;
    const e0 = interEdges[0];
    return { nics: [...nics], kind: e0.kind, bandwidth: e0.bandwidth,
             gbps: e0.gbps };
  }

  // ---- render ----------------------------------------------------------------
  function render(g) {
    document.getElementById("cv-updated").textContent =
      "갱신 " + new Date().toLocaleTimeString();
    const stateFilter = document.getElementById("cv-state-filter").value;
    const linkMode = document.getElementById("cv-link-filter").value;
    const powerOverlay = document.getElementById("cv-power-overlay").checked;

    const devices = g.vertices.filter(
      (v) => v.type === "device" && (!stateFilter || v.state === stateFilter));
    const shown = new Set(devices.map((v) => v.id));
    const nicOf = {};   // host -> nic vertex id
    g.vertices.filter((v) => v.type === "switch" && v.kind === "nic")
      .forEach((v) => { nicOf[v.host] = v.id; });

    const lay = layout(g.hosts, devices);
    svgEl.style.height = Math.max(320, lay.height) + "px";
    svg.attr("viewBox", `0 0 ${lay.width} ${Math.max(320, lay.height)}`);
    svg.selectAll("*").remove();
    const root = svg.append("g");
    svg.call(d3.zoom().scaleExtent([0.3, 4])
      .on("zoom", (ev) => root.attr("transform", ev.transform)));

    // z-order layers: inter-node links sit BEHIND host boxes (fabric spokes
    // would otherwise cut across them), intra-node links in front.
    const interLayer = root.append("g");
    const hostLayer = root.append("g");
    const intraLayer = root.append("g");
    const nicLayer = root.append("g");
    const devLayer = root.append("g");

    // host boxes
    const hostG = hostLayer.selectAll("g").data(lay.cells, (c) => c.host.id)
      .join("g");
    hostG.append("rect").attr("class", "host-hull")
      .attr("x", (c) => c.x).attr("y", (c) => c.y)
      .attr("width", d3.max(lay.cells, (c) => c.w))
      .attr("height", d3.max(lay.cells, (c) => c.h));
    hostG.append("text").attr("class", "host-label")
      .attr("x", (c) => c.x + 8).attr("y", (c) => c.y + 14)
      .text((c) => c.host.id + (c.host.host_base_w
        ? ` · base ${c.host.host_base_w}W · ${c.devs.length} dev` : ""));

    // links
    const pos = {};
    lay.cells.forEach((c) => {
      c.devs.forEach((d) => { pos[d.id] = [d.x, d.y]; });
      if (nicOf[c.host.id]) pos[nicOf[c.host.id]] = [c.nic.x, c.nic.y];
    });
    const intra = g.edges.filter((e) => !e.inter && shown.has(e.a) && shown.has(e.b));
    const inter = g.edges.filter((e) => e.inter && pos[e.a] && pos[e.b]);
    const fabric = linkMode === "fabric" ? fabricOf(inter) : null;
    const stroke = (gbps) => Math.max(0.8, Math.log2(1 + gbps) / 2);

    function drawLinks(layer, edges) {
      layer.selectAll("line").data(edges).join("line")
        .attr("class", (d) => "link-" + d.kind)
        .attr("stroke-width", (d) => stroke(d.gbps))
        .attr("x1", (d) => pos[d.a][0]).attr("y1", (d) => pos[d.a][1])
        .attr("x2", (d) => pos[d.b][0]).attr("y2", (d) => pos[d.b][1])
        .on("mousemove", (ev, d) => showTip(ev,
          `${d.kind} · ${d.bandwidth}${d.inter ? " · inter-node" : ""}`))
        .on("mouseout", hideTip);
    }

    if (linkMode !== "none") {
      drawLinks(intraLayer, linkMode === "all"
        ? intra : intra.filter((e) => e.kind === "nvlink" || e.kind === "xgmi"));
      if (!fabric) drawLinks(interLayer, inter);

      if (fabric) {   // one labelled hub instead of C(n,2) crossing lines
        const hub = { x: lay.width / 2, y: lay.height - 10 };
        interLayer.append("g").selectAll("line")
          .data(fabric.nics.filter((n) => pos[n])).join("line")
          .attr("class", "link-" + fabric.kind)
          .attr("stroke-width", stroke(fabric.gbps))
          .attr("x1", (n) => pos[n][0]).attr("y1", (n) => pos[n][1])
          .attr("x2", hub.x).attr("y2", hub.y);
        const hg = nicLayer.append("g").attr("class", "fabric-hub")
          .on("mousemove", (ev) => showTip(ev,
            `${fabric.kind} full mesh · ${fabric.bandwidth} · ` +
            `${fabric.nics.length} nodes · ${(fabric.nics.length *
              (fabric.nics.length - 1)) / 2} links`))
          .on("mouseout", hideTip);
        hg.append("rect").attr("x", hub.x - 92).attr("y", hub.y - 12)
          .attr("width", 184).attr("height", 24).attr("rx", 12);
        hg.append("text").attr("x", hub.x).attr("y", hub.y + 4)
          .attr("text-anchor", "middle")
          .text(`${fabric.kind} ${fabric.bandwidth} full mesh`);
      }
      nicLayer.selectAll("circle")
        .data(lay.cells.filter((c) => nicOf[c.host.id])).join("circle")
        .attr("class", "nic-dot").attr("r", NIC_R)
        .attr("cx", (c) => c.nic.x).attr("cy", (c) => c.nic.y)
        .on("mousemove", (ev, c) => showTip(ev, nicOf[c.host.id] + " (nic)"))
        .on("mouseout", hideTip);
    }

    // devices
    const devG = devLayer.selectAll("g").data(devices, (d) => d.id).join("g")
      .attr("transform", (d) => `translate(${d.x},${d.y})`)
      .call(d3.drag().on("drag", function (ev, d) {
        d.x = ev.x; d.y = ev.y;
        d3.select(this).attr("transform", `translate(${d.x},${d.y})`);
        overrides[d.id] = [d.x, d.y];
      }).on("end", () => { saveOverrides(); load(); }))
      .on("dblclick", (ev, d) => { delete overrides[d.id]; saveOverrides(); load(); })
      .on("mousemove", (ev, d) => showTip(ev,
        `${d.id}\n${d.hw} · ${d.mem_gb}GB · ${d.state}` +
        (d.active_w ? ` · active ${d.active_w}W` : "")))
      .on("mouseout", hideTip)
      .on("click", (ev, d) => detail(d));

    devG.append("rect")
      .attr("class", (d) => "dev-rect " + d.state +
        (highlight.size && !highlight.has(d.id) ? " dim" : ""))
      // uniform grid slots; active_w only nudges the fill size (120-300W)
      .each(function (d) {
        // 0.75-0.90 of the slot: keeps a gap so links stay visible (spec's
        // "size proportional to active_w over 120-300W" as the fill ratio)
        const s = 0.75 + 0.15 * Math.min(1, Math.max(0,
          ((d.active_w || 200) - 120) / 180));
        const w = DEV.w * s, h = DEV.h * s;
        d3.select(this).attr("width", w).attr("height", h)
          .attr("x", -w / 2).attr("y", -h / 2)
          .attr("rx", d.kind === "npu" ? 8 : 3);
      });
    devG.append("text").attr("class", "dev-label")
      .attr("text-anchor", "middle").attr("dy", 3)
      .text((d) => d.id.split("/").slice(1).join("/"));
    if (powerOverlay) {
      devG.append("text").attr("class", "dev-power")
        .attr("text-anchor", "middle").attr("dy", DEV.h / 2 + 9)
        .text((d) => (d.state === "free" ? d.idle_w : d.active_w) + "W");
    }

    summary(g);
  }

  // ---- summary panel ---------------------------------------------------------
  function summary(g) {
    const devs = g.vertices.filter((v) => v.type === "device");
    const byState = {}, byHw = {};
    devs.forEach((d) => {
      byState[d.state] = (byState[d.state] || 0) + 1;
      byHw[d.hw] = byHw[d.hw] || { n: 0, free: 0 };
      byHw[d.hw].n += 1;
      if (d.state === "free") byHw[d.hw].free += 1;
    });
    const wOf = (d) => (d.state === "free" ? (d.idle_w || 0) : (d.active_w || 0));
    const perHost = {};
    devs.forEach((d) => { perHost[d.host] = (perHost[d.host] || 0) + wOf(d); });
    g.hosts.forEach((h) => {
      if (h.host_base_w && perHost[h.id] !== undefined)
        perHost[h.id] += h.host_base_w;
    });
    const total = Object.values(perHost).reduce((a, b) => a + b, 0);
    const maxHost = Math.max(1, ...Object.values(perHost));

    document.getElementById("cv-summary").innerHTML =
      `<p>총 디바이스 <strong>${devs.length}</strong> · 노드 ${g.hosts.length}</p>` +
      "<table class='svc-table'><tr><th>hw</th><th>free / total</th></tr>" +
      Object.entries(byHw).map(([hw, v]) =>
        `<tr><td><strong>${svc.esc(hw)}</strong></td>` +
        `<td class="svc-num">${v.free} / ${v.n}</td></tr>`).join("") +
      "</table><p>" +
      Object.entries(byState).map(([s, n]) =>
        `<span class="badge ${s}">${s} ${n}</span>`).join(" ") +
      `</p><p>추정 총 전력 <strong>${svc.fmtW(total)}</strong></p>` +
      "<h4 style='margin:10px 0 4px'>호스트별 전력</h4>" +
      Object.entries(perHost).map(([h, w]) =>
        `<div class="host-bar"><span>${svc.esc(h)}</span>` +
        `<span class="bar"><i style="width:${(w / maxHost * 100).toFixed(0)}%"></i></span>` +
        `<span class="svc-num">${svc.fmtW(w)}</span></div>`).join("");
  }

  function detail(d) {
    const bar = document.getElementById("svc-detail-bar");
    bar.style.display = "block";
    bar.innerHTML =
      `<strong>${svc.esc(d.id)}</strong> · <span class="badge ${d.state}">${d.state}</span>` +
      ` · ${svc.esc(d.hw)} · ${d.mem_gb} GB · active ${svc.fmtW(d.active_w)}` +
      ` · idle ${svc.fmtW(d.idle_w)}` +
      (d.state !== "free"
        ? ` · <a href="/service/deployments" style="color:#8ecae6">배포 상세로 이동 →</a>`
        : "");
  }

  function showTip(ev, text) {
    tip.style.display = "block";
    tip.textContent = text;
    tip.style.left = ev.pageX + 12 + "px";
    tip.style.top = ev.pageY + 12 + "px";
  }
  function hideTip() { tip.style.display = "none"; }

  // ---- load / poll ------------------------------------------------------------
  function load() {
    fetch("/api/cluster/graph").then((r) => {
      if (!r.ok) throw new Error(r.status);
      return r.json();
    }).then(render).catch(() => {
      document.getElementById("cv-fallback").style.display = "";
      fetch("/api/cluster").then((r) => r.json()).then((d) => {
        document.getElementById("cv-table").innerHTML =
          "<table class='svc-table'><tr><th>free devices</th></tr>" +
          d.free_devices.map((x) => `<tr><td>${svc.esc(x)}</td></tr>`).join("") +
          "</table>";
      });
    });
  }

  ["cv-state-filter", "cv-link-filter", "cv-power-overlay"].forEach((id) =>
    document.getElementById(id).addEventListener("change", load));
  load();
  setInterval(load, 10000);   // spec: 10s poll
})();
