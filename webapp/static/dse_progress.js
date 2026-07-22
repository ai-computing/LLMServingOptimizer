/* DSE progress page — SSE stream + live scatter. */
(function () {
    'use strict';

    // label → {ttft, throughput, energy} — drives both live plots
    const livePoints = {};
    // label → {hw_distribution, parallelism, pd_layout} — from candidates.json
    let candidatesMeta = {};

    // Stable per-candidate color assignment — same palette as webapp/plots.py
    const CONFIG_PALETTE = [
        // Plotly Dark24
        '#2E91E5','#E15F99','#1CA71C','#FB0D0D','#DA16FF','#222A2A','#B68100',
        '#750D86','#EB663B','#511CFB','#00A08B','#FB00D1','#FC0080','#B2828D',
        '#6C7C32','#778AAE','#862A16','#A777F1','#620042','#1616A7','#DA60CA',
        '#6C4516','#0D2A63','#AF0038',
        // Plotly Light24
        '#FD3216','#00FE35','#6A76FC','#FED4C4','#FE00CE','#0DF9FF','#F6F926',
        '#FF9616','#479B55','#EEA6FB','#DC587D','#D626FF','#6E899C','#00B5F7',
        '#B68E00','#C9FBE5','#FF0092','#22FFA7','#E3EE9E','#86CE00','#BC7196',
        '#7E7DCD','#FC6955','#E48F72',
    ];
    const _colorMap = {};
    let _colorIdx = 0;

    function getColor(label) {
        if (!_colorMap[label]) {
            _colorMap[label] = CONFIG_PALETTE[_colorIdx % CONFIG_PALETTE.length];
            _colorIdx++;
        }
        return _colorMap[label];
    }

    document.addEventListener('DOMContentLoaded', async () => {
        await loadCandidatesMeta();
        await loadInitial();
        initPlot();
        initTableSort();
        const es = new EventSource(`/api/dse/jobs/${JOB_ID}/events`);
        es.onmessage = (e) => {
            try {
                const ev = JSON.parse(e.data);
                if (ev.type === 'heartbeat') return;
                if (ev.sweep_state) {
                    if (['done', 'failed', 'cancelled'].includes(ev.sweep_state)) {
                        document.getElementById('btn-results').classList.remove('disabled');
                        es.close();
                    }
                    return;
                }
                if (ev.label) {
                    applyUpdate(ev);
                }
            } catch (err) { console.error('SSE parse', err); }
        };
        es.addEventListener('snapshot', (e) => {
            try {
                const snap = JSON.parse(e.data);
                for (const [label, entry] of Object.entries(snap.configs || {})) {
                    applyUpdate({label, ...entry});
                }
                // If the job was already finished when we connected, enable button now
                if (['done', 'failed', 'cancelled'].includes(snap.state)) {
                    document.getElementById('btn-results').classList.remove('disabled');
                    es.close();
                }
            } catch (e) {}
        });
        document.getElementById('btn-cancel').addEventListener('click', async () => {
            // Cancel (not delete): stops the run but keeps the job + partial
            // results so Open Results can show the candidates that finished.
            await fetch(`/api/dse/jobs/${JOB_ID}/cancel`, {method: 'POST'});
        });
    });

    async function loadCandidatesMeta() {
        // candidates.json is written before any simulations run, so
        // /api/dse/jobs/{id}/results already carries candidates_meta during
        // the live progress phase even if `all_candidates` etc. are empty.
        try {
            const r = await fetch(`/api/dse/jobs/${JOB_ID}/results`);
            const j = await r.json();
            candidatesMeta = j.candidates_meta || {};
        } catch (e) { candidatesMeta = {}; }
    }

    function formatHw(label) {
        const m = candidatesMeta[label];
        if (!m || !m.hw_distribution) return '—';
        return Object.entries(m.hw_distribution)
            .filter(([, c]) => c > 0)
            .map(([hw, c]) => `${c}×${hw}`)
            .join(' + ');
    }

    async function loadInitial() {
        // Pull spec to know total candidate count target
        try {
            const r = await fetch(`/api/dse/jobs/${JOB_ID}`);
            const j = await r.json();
            const status = j.status || {};
            for (const [label, entry] of Object.entries(status.configs || {})) {
                applyUpdate({label, ...entry});
            }
        } catch (e) {}
    }

    // Debounce flag so a burst of retry-candidate events triggers only one
    // candidatesMeta re-fetch instead of one per arriving label.
    let _metaRefreshPending = false;

    function applyUpdate(ev) {
        // If this label isn't in candidatesMeta yet (retry candidate arrived
        // after initial load), schedule a single re-fetch so the Hardware
        // column fills in as soon as the data is available.
        if (ev.label && !candidatesMeta[ev.label] && !_metaRefreshPending) {
            _metaRefreshPending = true;
            setTimeout(async () => {
                await loadCandidatesMeta();
                _metaRefreshPending = false;
                // Backfill any hw cells that are still "—"
                document.querySelectorAll('#dse-runs-body tr').forEach(tr => {
                    const hwCell = tr.querySelector('.dse-hw');
                    if (hwCell && hwCell.textContent === '—') {
                        const hw = formatHw(tr.dataset.label);
                        if (hw !== '—') { hwCell.textContent = hw; hwCell.title = hw; }
                    }
                });
            }, 300);
        }

        // Ensure row in candidates table
        let row = document.querySelector(`tr[data-label="${cssEscape(ev.label)}"]`);
        if (!row) {
            row = document.createElement('tr');
            row.dataset.label = ev.label;
            const hwText = formatHw(ev.label);
            row.innerHTML = `
                <td title="${ev.label}">${ev.label}</td>
                <td class="dse-hw" title="${hwText}">${hwText}</td>
                <td class="dse-state">${ev.state || 'queued'}</td>
                <td class="dse-ttft">—</td>
                <td class="dse-tpot">—</td>
                <td class="dse-tp">—</td>
                <td class="dse-energy">—</td>
                <td class="dse-tokwh">—</td>
                <td class="dse-elapsed">—</td>`;
            row.style.cursor = 'pointer';
            row.addEventListener('click', () => togglePersistentHighlight(ev.label));
            document.getElementById('dse-runs-body').appendChild(row);
        } else {
            // If candidatesMeta arrived late, fill hw cell on subsequent updates
            const hwCell = row.querySelector('.dse-hw');
            if (hwCell && hwCell.textContent === '—') {
                const hwText = formatHw(ev.label);
                if (hwText !== '—') {
                    hwCell.textContent = hwText;
                    hwCell.title = hwText;
                }
            }
        }
        if (ev.state) {
            const stateCell = row.querySelector('.dse-state');
            stateCell.textContent = ev.state;
            if (ev.cached) {
                const badge = document.createElement('span');
                badge.className = 'badge-cached';
                badge.textContent = 'cached';
                stateCell.appendChild(document.createTextNode(' '));
                stateCell.appendChild(badge);
            }
        }
        if (ev.elapsed_s != null) row.querySelector('.dse-elapsed').textContent = ev.elapsed_s.toFixed(1);
        // Defensive: failed/cancelled rows must never display metrics, even if
        // legacy status.json has stale fields from a runner race (see invariant
        // enforcement in webapp/runner.py).
        const failed = ['failed', 'cancelled'].includes(ev.state);
        const m = failed ? {} : (ev.metrics || {});
        const setCell = (cls, v, fmt = (x) => x.toFixed(2)) => {
            const cell = row.querySelector(cls);
            if (failed) { cell.textContent = '—'; return; }
            if (v != null) cell.textContent = fmt(v);
        };
        setCell('.dse-ttft', m.p99_ttft_ms);
        setCell('.dse-tpot', m.tpot_p99_ms);
        setCell('.dse-tp',   m.total_token_tp);
        setCell('.dse-energy', m.total_energy_wh, x => x.toFixed(4));
        const tokPerWh = m.tok_per_wh != null ? m.tok_per_wh
            : (m.total_token_tp != null && m.total_latency_s != null && m.total_energy_wh > 0
                ? m.total_token_tp * m.total_latency_s / m.total_energy_wh
                : null);
        setCell('.dse-tokwh', tokPerWh, x => x.toFixed(0));

        // Re-apply the active sort whenever a row arrives/changes so SSE
        // updates don't bump rows out of order.
        applyTableSort();

        // Add to live plots if metrics complete. Both charts share the same
        // candidate set; the axis mapping differs per chart.
        if (ev.state === 'done' && m.p99_ttft_ms != null && m.total_token_tp != null) {
            const tokPerWh2 = m.tok_per_wh != null ? m.tok_per_wh
                : (m.total_token_tp != null && m.total_latency_s != null && m.total_energy_wh > 0
                    ? m.total_token_tp * m.total_latency_s / m.total_energy_wh
                    : null);
            livePoints[ev.label] = {
                ttft: m.p99_ttft_ms,
                tpot: m.tpot_p99_ms != null ? m.tpot_p99_ms : null,
                throughput: m.total_token_tp,
                energy: m.total_energy_wh != null ? m.total_energy_wh : null,
                tokwh: tokPerWh2,
                label: ev.label,
            };
            redrawPlot();
        }
        updateProgressBar();
    }

    function initPlot() {
        // Chart 1: TTFT vs Throughput — per-candidate color
        Plotly.newPlot('dse-live-plot', [{
            x: [], y: [], mode: 'markers',
            marker: {size: 10, color: []},
            text: [],
            type: 'scatter',
        }], {
            xaxis: {title: 'TTFT p99 (ms)'},
            yaxis: {title: 'Throughput (tok/s)'},
            margin: {l: 60, r: 30, t: 20, b: 50},
            paper_bgcolor: 'rgba(0,0,0,0)',
        }, {responsive: true});

        // Chart 2: Energy vs Throughput — same per-candidate color
        Plotly.newPlot('dse-energy-plot', [{
            x: [], y: [], mode: 'markers',
            marker: {size: 10, color: []},
            text: [],
            type: 'scatter',
        }], {
            xaxis: {title: 'Energy (Wh)'},
            yaxis: {title: 'Throughput (tok/s)'},
            margin: {l: 60, r: 30, t: 20, b: 50},
            paper_bgcolor: 'rgba(0,0,0,0)',
        }, {responsive: true});

        // Chart 3: TPOT p99 vs Throughput — same per-candidate color
        Plotly.newPlot('dse-tpot-plot', [{
            x: [], y: [], mode: 'markers',
            marker: {size: 10, color: []},
            text: [],
            type: 'scatter',
        }], {
            xaxis: {title: 'TPOT p99 (ms)'},
            yaxis: {title: 'Throughput (tok/s)'},
            margin: {l: 60, r: 30, t: 20, b: 50},
            paper_bgcolor: 'rgba(0,0,0,0)',
        }, {responsive: true});

        // Chart 4: Tokens/Wh vs Throughput — same per-candidate color
        Plotly.newPlot('dse-tokwh-plot', [{
            x: [], y: [], mode: 'markers',
            marker: {size: 10, color: []},
            text: [],
            type: 'scatter',
        }], {
            xaxis: {title: 'Tokens/Wh'},
            yaxis: {title: 'Throughput (tok/s)'},
            margin: {l: 60, r: 30, t: 20, b: 50},
            paper_bgcolor: 'rgba(0,0,0,0)',
        }, {responsive: true});

        // Bind identical click/hover handlers to all charts so a click in
        // any one highlights the same row.
        const wirePlot = (id) => {
            const div = document.getElementById(id);
            div.on('plotly_click', (event) => {
                if (!event.points || !event.points[0]) return;
                const label = event.points[0].text;
                if (label) togglePersistentHighlight(label);
            });
            div.on('plotly_hover', (event) => {
                if (!event.points || !event.points[0]) return;
                const label = event.points[0].text;
                if (label) setHoverPreview(label);
            });
            div.on('plotly_unhover', () => clearHoverPreview());
        };
        wirePlot('dse-live-plot');
        wirePlot('dse-energy-plot');
        wirePlot('dse-tpot-plot');
        wirePlot('dse-tokwh-plot');
    }

    // Module state for click selection (so we can toggle the same dot off)
    let _selectedLabel = null;

    function togglePersistentHighlight(label) {
        if (_selectedLabel === label) {
            _selectedLabel = null;
            document.querySelectorAll('#dse-runs-body tr.row-highlighted')
                .forEach(tr => tr.classList.remove('row-highlighted'));
            redrawPlot();
            return;
        }
        _selectedLabel = label;
        let target = null;
        document.querySelectorAll('#dse-runs-body tr').forEach(tr => {
            if (tr.dataset.label === label) {
                tr.classList.add('row-highlighted');
                target = tr;
            } else {
                tr.classList.remove('row-highlighted');
            }
        });
        if (target) target.scrollIntoView({behavior: 'smooth', block: 'nearest'});
        redrawPlot();
    }

    function setHoverPreview(label) {
        document.querySelectorAll('#dse-runs-body tr').forEach(tr => {
            tr.classList.toggle('row-hover-preview', tr.dataset.label === label);
        });
    }

    function clearHoverPreview() {
        document.querySelectorAll('#dse-runs-body tr.row-hover-preview')
            .forEach(tr => tr.classList.remove('row-hover-preview'));
    }
    function redrawPlot() {
        const arr = Object.values(livePoints);
        const sel = _selectedLabel;
        const hasSel = sel !== null && arr.some(p => p.label === sel);

        const markerStyle = (points) => ({
            size:    points.map(p => hasSel ? (p.label === sel ? 14 : 8)   : 10),
            color:   points.map(p => getColor(p.label)),
            opacity: points.map(p => hasSel ? (p.label === sel ? 1.0 : 0.3) : 1.0),
        });

        // Chart 1: TTFT vs Throughput — color by candidate identity
        Plotly.react('dse-live-plot', [{
            x: arr.map(p => p.ttft), y: arr.map(p => p.throughput),
            mode: 'markers',
            marker: markerStyle(arr),
            text: arr.map(p => p.label),
            hovertemplate: '%{text}<br>TTFT=%{x:.2f}<br>Tok/s=%{y:.2f}<extra></extra>',
            type: 'scatter',
        }], {
            xaxis: {title: 'TTFT p99 (ms)'},
            yaxis: {title: 'Throughput (tok/s)'},
            margin: {l: 60, r: 30, t: 20, b: 50},
            paper_bgcolor: 'rgba(0,0,0,0)',
        }, {responsive: true});

        // Chart 2: Energy vs Throughput — same per-candidate color.
        // Skip points missing the energy metric (simulations without a power block).
        const energyPoints = arr.filter(p => p.energy != null);
        Plotly.react('dse-energy-plot', [{
            x: energyPoints.map(p => p.energy),
            y: energyPoints.map(p => p.throughput),
            mode: 'markers',
            marker: markerStyle(energyPoints),
            text: energyPoints.map(p => p.label),
            hovertemplate: '%{text}<br>Energy=%{x:.4f} Wh<br>Tok/s=%{y:.2f}<extra></extra>',
            type: 'scatter',
        }], {
            xaxis: {title: 'Energy (Wh)'},
            yaxis: {title: 'Throughput (tok/s)'},
            margin: {l: 60, r: 30, t: 20, b: 50},
            paper_bgcolor: 'rgba(0,0,0,0)',
        }, {responsive: true});

        // Chart 3: TPOT p99 vs Throughput — skip points missing tpot.
        const tpotPoints = arr.filter(p => p.tpot != null);
        Plotly.react('dse-tpot-plot', [{
            x: tpotPoints.map(p => p.tpot),
            y: tpotPoints.map(p => p.throughput),
            mode: 'markers',
            marker: markerStyle(tpotPoints),
            text: tpotPoints.map(p => p.label),
            hovertemplate: '%{text}<br>TPOT=%{x:.2f} ms<br>Tok/s=%{y:.2f}<extra></extra>',
            type: 'scatter',
        }], {
            xaxis: {title: 'TPOT p99 (ms)'},
            yaxis: {title: 'Throughput (tok/s)'},
            margin: {l: 60, r: 30, t: 20, b: 50},
            paper_bgcolor: 'rgba(0,0,0,0)',
        }, {responsive: true});

        // Chart 4: Tokens/Wh vs Throughput — skip points missing tokwh.
        const tokwhPoints = arr.filter(p => p.tokwh != null);
        Plotly.react('dse-tokwh-plot', [{
            x: tokwhPoints.map(p => p.tokwh),
            y: tokwhPoints.map(p => p.throughput),
            mode: 'markers',
            marker: markerStyle(tokwhPoints),
            text: tokwhPoints.map(p => p.label),
            hovertemplate: '%{text}<br>Tok/Wh=%{x:.0f}<br>Tok/s=%{y:.2f}<extra></extra>',
            type: 'scatter',
        }], {
            xaxis: {title: 'Tokens/Wh'},
            yaxis: {title: 'Throughput (tok/s)'},
            margin: {l: 60, r: 30, t: 20, b: 50},
            paper_bgcolor: 'rgba(0,0,0,0)',
        }, {responsive: true});
    }

    function updateProgressBar() {
        const rows = document.querySelectorAll('#dse-runs-body tr');
        let done = 0;
        rows.forEach(r => {
            const s = r.querySelector('.dse-state').textContent;
            if (['done', 'failed', 'cancelled'].includes(s)) done++;
        });
        const total = rows.length;
        document.getElementById('dse-progress-summary').textContent = `Progress: ${done}/${total} done`;
        const pct = total ? Math.round((done / total) * 100) : 0;
        document.getElementById('progress-fill').style.width = `${pct}%`;
    }

    function cssEscape(s) {
        if (window.CSS && CSS.escape) return CSS.escape(s);
        return String(s).replace(/[^a-zA-Z0-9_-]/g, c => '\\' + c);
    }

    // -------------------------------------------------------------------
    // Sortable Candidates table
    // Click header → sort asc. Click same header again → toggle to desc.
    // Missing values ("—") sink to the bottom regardless of direction.
    // -------------------------------------------------------------------
    let _sortCol = null;   // column index (0-based)
    let _sortDir = null;   // 'asc' | 'desc'

    function initTableSort() {
        const headers = document.querySelectorAll('#dse-runs thead th');
        headers.forEach((th, idx) => {
            th.style.cursor = 'pointer';
            th.dataset.sortIdx = String(idx);
            th.addEventListener('click', () => onHeaderClick(idx));
        });
    }

    function onHeaderClick(idx) {
        if (_sortCol === idx) {
            _sortDir = _sortDir === 'asc' ? 'desc' : 'asc';
        } else {
            _sortCol = idx;
            _sortDir = 'asc';
        }
        updateHeaderIndicator();
        applyTableSort();
    }

    function updateHeaderIndicator() {
        const headers = document.querySelectorAll('#dse-runs thead th');
        headers.forEach((th, idx) => {
            // Strip any prior arrow suffix
            th.textContent = th.textContent.replace(/\s*[▲▼]$/u, '');
            if (idx === _sortCol) {
                th.textContent += _sortDir === 'asc' ? ' ▲' : ' ▼';
            }
        });
    }

    function applyTableSort() {
        if (_sortCol === null) return;
        const tbody = document.getElementById('dse-runs-body');
        const rows = Array.from(tbody.querySelectorAll('tr'));
        const dir = _sortDir === 'asc' ? 1 : -1;

        rows.sort((a, b) => {
            const aText = (a.cells[_sortCol] || {}).textContent || '';
            const bText = (b.cells[_sortCol] || {}).textContent || '';
            const aNum = parseFloat(aText);
            const bNum = parseFloat(bText);
            const aMissing = aText.trim() === '—' || aText.trim() === '';
            const bMissing = bText.trim() === '—' || bText.trim() === '';

            // Missing values always sort to the bottom (regardless of dir)
            if (aMissing && bMissing) return 0;
            if (aMissing) return 1;
            if (bMissing) return -1;

            // Both numeric → numeric compare
            if (!isNaN(aNum) && !isNaN(bNum)) {
                return (aNum - bNum) * dir;
            }
            // Otherwise lexical
            return aText.localeCompare(bText) * dir;
        });
        rows.forEach(r => tbody.appendChild(r));  // re-append in sorted order
    }
})();
