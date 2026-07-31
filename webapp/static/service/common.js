/* Service tab shared utilities: SSE client with exponential backoff, number
   formatters, toast, and modal helpers (plan §5.1). State strings and colors
   come from the server / CSS tokens — never hardcoded here. */
window.svc = (function () {
  function fmtW(w) { return w == null ? "—" : (w >= 1000 ? (w / 1000).toFixed(1) + " kW" : Math.round(w) + " W"); }
  function fmtMs(v) { return v == null ? "—" : v.toFixed(v < 10 ? 1 : 0) + " ms"; }
  function fmtTok(v) { return v == null ? "—" : Math.round(v) + " tok/s"; }
  function fmtWh(v) { return v == null ? "—" : v.toFixed(1) + " Wh"; }
  function esc(s) { const d = document.createElement("div"); d.textContent = String(s); return d.innerHTML; }

  function toast(msg, ms) {
    const host = document.getElementById("toast-host") || document.body;
    const t = document.createElement("div");
    t.className = "toast"; t.textContent = msg;
    t.style.cssText = "background:#263238;color:#fff;padding:10px 16px;border-radius:8px;margin:6px;";
    host.appendChild(t);
    setTimeout(() => t.remove(), ms || 5000);
  }

  function modal(html, onConfirm, opts) {
    opts = opts || {};
    const back = document.createElement("div");
    back.className = "svc-modal-backdrop";
    back.innerHTML = `<div class="svc-modal">${html}
      <div style="margin-top:14px;text-align:right">
        <button class="btn" data-act="cancel">취소</button>
        <button class="btn primary" data-act="ok" ${opts.confirmDisabled ? "disabled" : ""}>${esc(opts.okLabel || "확인")}</button>
      </div></div>`;
    back.addEventListener("click", (e) => {
      if (e.target === back || e.target.dataset.act === "cancel") back.remove();
      if (e.target.dataset.act === "ok" && !e.target.disabled) { back.remove(); onConfirm && onConfirm(back); }
    });
    document.body.appendChild(back);
    return back;
  }

  /* SSE with exponential backoff + header status indicator */
  function sse(url, onEvent, onEnd) {
    let stopped = false, delay = 1000;
    const status = document.getElementById("sse-status");
    function setStatus(ok) {
      if (!status) return;
      status.classList.toggle("lost", !ok);
      status.textContent = ok ? "SSE 연결됨" :
        "SSE 유실 — 재연결 중 (마지막 수신 " + new Date().toLocaleTimeString() + ")";
    }
    function connect() {
      if (stopped) return;
      const es = new EventSource(url);
      es.onopen = () => { setStatus(true); delay = 1000; };
      es.onmessage = (e) => {
        const d = JSON.parse(e.data);
        if (d.type === "end") { es.close(); stopped = true; setStatus(true); onEnd && onEnd(d); return; }
        onEvent(d);
      };
      es.onerror = () => {
        es.close(); setStatus(false);
        if (!stopped) { setTimeout(connect, delay); delay = Math.min(delay * 2, 15000); }
      };
    }
    connect();
    return { stop() { stopped = true; } };
  }

  return { fmtW, fmtMs, fmtTok, fmtWh, esc, toast, modal, sse };
})();
