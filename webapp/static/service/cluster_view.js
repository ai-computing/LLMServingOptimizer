/* Cluster view: d3-force topology renderer (plan §5.2). Devices are squares
   colored by state class; hosts are hull rectangles; link style by kind.
   Highlight mode: /service/cluster#highlight=id1,id2 dims everything else. */
(function () {
  const svg = d3.select("#svc-graph");
  const W = document.getElementById("svc-graph").clientWidth || 900, H = 520;
  svg.attr("viewBox", `0 0 ${W} ${H}`);
  const tip = document.createElement("div");
  tip.className = "svc-tooltip"; tip.style.display = "none";
  document.body.appendChild(tip);

  const highlight = new Set(
    (location.hash.match(/highlight=([^&]+)/) || [, ""])[1]
      .split(",").filter(Boolean).map(decodeURIComponent));

  let sim = null;
  const posKey = "svc-graph-pos";
  const saved = JSON.parse(localStorage.getItem(posKey) || "{}");

  function load() {
    fetch("/api/cluster/graph").then(r => {
      if (!r.ok) throw new Error(r.status);
      return r.json();
    }).then(render).catch(() => {
      document.getElementById("cv-fallback").style.display = "";
      fetch("/api/cluster").then(r => r.json()).then(d => {
        document.getElementById("cv-table").innerHTML =
          "<table class='svc-table'><tr><th>free devices</th></tr>" +
          d.free_devices.map(x => `<tr><td>${svc.esc(x)}</td></tr>`).join("") +
          "</table>";
      });
    });
  }

  function render(g) {
    document.getElementById("cv-updated").textContent =
      "갱신 " + new Date().toLocaleTimeString();
    const filter = document.getElementById("cv-state-filter").value;
    const devs = g.vertices.filter(v => v.type === "device");
    const shown = new Set(devs.filter(v => !filter || v.state === filter).map(v => v.id));
    // switches shown only when kind === nic (spec default)
    const nodes = g.vertices.filter(v =>
      (v.type === "device" && shown.has(v.id)) ||
      (v.type === "switch" && v.kind === "nic"));
    const idset = new Set(nodes.map(n => n.id));
    const links = g.edges.filter(e => idset.has(e.a) && idset.has(e.b))
      .map(e => ({ source: e.a, target: e.b, ...e }));
    nodes.forEach(n => {
      if (saved[n.id]) { n.fx = saved[n.id][0]; n.fy = saved[n.id][1]; }
    });

    svg.selectAll("*").remove();
    const root = svg.append("g");
    svg.call(d3.zoom().scaleExtent([0.4, 3])
      .on("zoom", (ev) => root.attr("transform", ev.transform)));

    const hullLayer = root.append("g");
    const linkSel = root.append("g").selectAll("line").data(links).join("line")
      .attr("class", d => "link-" + d.kind)
      .attr("stroke-width", d => Math.max(1, Math.log2(1 + d.gbps)))
      .on("mousemove", (ev, d) => showTip(ev,
        `${d.kind} · ${d.bandwidth}${d.inter ? " · inter-node" : ""}`))
      .on("mouseout", hideTip);

    const nodeSel = root.append("g").selectAll("g").data(nodes).join("g")
      .call(d3.drag()
        .on("drag", (ev, d) => { d.fx = ev.x; d.fy = ev.y; sim.alpha(0.3).restart(); })
        .on("end", (ev, d) => { saved[d.id] = [d.fx, d.fy]; localStorage.setItem(posKey, JSON.stringify(saved)); }))
      .on("dblclick", (ev, d) => { d.fx = d.fy = null; delete saved[d.id]; localStorage.setItem(posKey, JSON.stringify(saved)); })
      .on("mousemove", (ev, d) => showTip(ev, d.type === "device"
        ? `${d.id} · ${d.hw} · ${d.mem_gb}GB · ${d.state}` : `${d.id} (${d.kind})`))
      .on("mouseout", hideTip)
      .on("click", (ev, d) => detail(d));

    nodeSel.each(function (d) {
      const el = d3.select(this);
      if (d.type === "switch") {
        el.append("circle").attr("r", 8).attr("fill", "#90a4ae");
      } else {
        const size = d.active_w ? 14 + 10 * Math.min(1, Math.max(0, (d.active_w - 120) / 180)) : 18;
        d._size = size;
        el.append("rect").attr("class", "dev-rect " + d.state +
            (highlight.size && !highlight.has(d.id) ? " dim" : ""))
          .attr("width", size * 2).attr("height", size * 1.4)
          .attr("x", -size).attr("y", -size * 0.7).attr("rx", d.kind === "npu" ? 8 : 3);
        el.append("text").text(d.id.split("/").slice(1).join("/"))
          .attr("text-anchor", "middle").attr("dy", 4).attr("font-size", 9)
          .attr("fill", d.state === "free" ? "#333" : "#fff");
        if (document.getElementById("cv-power-overlay").checked)
          el.append("text").attr("class", "pw").attr("dy", size * 0.7 + 11)
            .attr("text-anchor", "middle").attr("font-size", 8)
            .text(d.active_w ? d.active_w + "W" : "");
      }
    });

    sim = d3.forceSimulation(nodes)
      .force("link", d3.forceLink(links).id(d => d.id)
        .distance(d => d.inter ? 180 : 46).strength(d => d.inter ? 0.2 : 0.6))
      .force("charge", d3.forceManyBody().strength(-120))
      .force("center", d3.forceCenter(W / 2, H / 2))
      .force("host", hostForce(nodes))
      .on("tick", () => {
        linkSel.attr("x1", d => d.source.x).attr("y1", d => d.source.y)
               .attr("x2", d => d.target.x).attr("y2", d => d.target.y);
        nodeSel.attr("transform", d => `translate(${d.x},${d.y})`);
        drawHulls(hullLayer, nodes, g.hosts);
      });
    summary(g);
  }

  function hostForce(nodes) {
    const hosts = [...new Set(nodes.map(n => n.host))];
    const cx = {}; hosts.forEach((h, i) => {
      cx[h] = [(W / (hosts.length + 1)) * (i + 1), H / 2];
    });
    return d3.forceX(d => cx[d.host][0]).strength(0.25);
  }

  function drawHulls(layer, nodes, hosts) {
    const byHost = d3.group(nodes, d => d.host);
    const data = [...byHost.entries()].map(([h, ns]) => {
      const xs = ns.map(n => n.x), ys = ns.map(n => n.y);
      const host = hosts.find(x => x.id === h) || {};
      return { h, host, x0: Math.min(...xs) - 30, y0: Math.min(...ys) - 34,
               x1: Math.max(...xs) + 30, y1: Math.max(...ys) + 26 };
    });
    const sel = layer.selectAll("g.hull").data(data, d => d.h);
    const ent = sel.enter().append("g").attr("class", "hull");
    ent.append("rect").attr("class", "host-hull").attr("rx", 12);
    ent.append("text").attr("font-size", 11).attr("fill", "#5c6bc0");
    const all = ent.merge(sel);
    all.select("rect").attr("x", d => d.x0).attr("y", d => d.y0)
      .attr("width", d => d.x1 - d.x0).attr("height", d => d.y1 - d.y0);
    all.select("text").attr("x", d => d.x0 + 6).attr("y", d => d.y0 + 14)
      .text(d => d.h + (d.host.host_base_w ? ` · base ${d.host.host_base_w}W` : ""));
    sel.exit().remove();
  }

  function summary(g) {
    const devs = g.vertices.filter(v => v.type === "device");
    const byState = {};
    devs.forEach(d => byState[d.state] = (byState[d.state] || 0) + 1);
    const totalW = devs.reduce((s, d) =>
      s + (d.state === "free" ? (d.idle_w || 0) : (d.active_w || 0)), 0) +
      g.hosts.reduce((s, h) => s + (h.host_base_w || 0), 0);
    document.getElementById("cv-summary").innerHTML =
      `<p>총 디바이스 <strong>${devs.length}</strong></p>` +
      Object.entries(byState).map(([s, n]) =>
        `<p><span class="badge ${s}">${s}</span> <span class="svc-num">${n}</span></p>`).join("") +
      `<p>추정 총 전력 <strong>${svc.fmtW(totalW)}</strong></p>`;
  }

  function detail(d) {
    const bar = document.getElementById("svc-detail-bar");
    bar.style.display = "block";
    bar.innerHTML = d.type === "device"
      ? `<strong>${svc.esc(d.id)}</strong> · <span class="badge ${d.state}">${d.state}</span>` +
        ` · ${d.hw} · ${d.mem_gb} GB · active ${svc.fmtW(d.active_w)}` +
        (d.state !== "free" ? ` · <a href="/service/deployments" style="color:#8ecae6">배포 상세로 이동 →</a>` : "")
      : `<strong>${svc.esc(d.id)}</strong> (${d.kind})`;
  }

  function showTip(ev, text) {
    tip.style.display = "block"; tip.textContent = text;
    tip.style.left = (ev.pageX + 12) + "px"; tip.style.top = (ev.pageY + 12) + "px";
  }
  function hideTip() { tip.style.display = "none"; }

  document.getElementById("cv-state-filter").addEventListener("change", load);
  document.getElementById("cv-power-overlay").addEventListener("change", load);
  load();
  setInterval(load, 10000);   // spec: 10s poll
})();
