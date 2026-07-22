/* DSE results page — Top-N table, Pareto plot, radar, rerank, SLO filter. */
(function () {
    'use strict';

    let allResults = [];
    let candidatesMeta = {};
    let currentTopN = [];
    let currentPareto = [];
    let _selectedLabel = null;
    let _displayedRows = [];   // same set shown in Candidate Ranking table

    // SLO filter: thresholds applied client-side to allResults
    // dir='max' → pass if metric <= threshold; dir='min' → pass if metric >= threshold
    const SLO_DEFS = [
        { sliderId: 'slo-ttft',   numId: 'slo-ttft-num',   key: 'p99_ttft_ms',     dir: 'max' },
        { sliderId: 'slo-tpot',   numId: 'slo-tpot-num',   key: 'p99_tpot_ms',     dir: 'max' },
        { sliderId: 'slo-tp',     numId: 'slo-tp-num',     key: 'total_token_tp',   dir: 'min' },
        { sliderId: 'slo-tokwh',  numId: 'slo-tokwh-num',  key: 'tok_per_wh',       dir: 'min' },
    ];
    // Per-def initial "no constraint" value — set in initSloSliders
    const sloInit = {};

    // Metric definitions: weight key → axis key, direction, label
    const METRIC_DEFS = [
        { weightKey: 'ttft',       axisKey: 'p99_ttft_ms',    dir: 'min', label: 'TTFT p99 (ms)' },
        { weightKey: 'tpot',       axisKey: 'p99_tpot_ms',    dir: 'min', label: 'TPOT p99 (ms)' },
        { weightKey: 'throughput', axisKey: 'total_token_tp', dir: 'max', label: 'Throughput (tok/s)' },
        { weightKey: 'tokwh',      axisKey: 'tok_per_wh',     dir: 'max', label: 'Tok/Wh' },
        { weightKey: 'power',      axisKey: 'total_energy_wh',dir: 'min', label: 'Energy (Wh)' },
    ];

    document.addEventListener('DOMContentLoaded', async () => {
        await loadResults();
        initSloSliders();
        initParetoInteraction();
        wireSliders();
        document.getElementById('btn-rerank').addEventListener('click', rerank);
        document.getElementById('btn-slo-reset').addEventListener('click', resetSlo);
        // Manual axis change clears the auto badge
        ['pareto-x', 'pareto-y'].forEach(id => {
            document.getElementById(id).addEventListener('change', () => {
                document.getElementById('pareto-auto-badge').style.display = 'none';
                drawPareto();
            });
        });
    });

    async function loadResults() {
        const r = await fetch(`/api/dse/jobs/${JOB_ID}/results`);
        const j = await r.json();
        allResults = j.all_candidates || [];
        currentTopN = j.top_n || [];
        currentPareto = j.pareto || [];
        candidatesMeta = j.candidates_meta || {};
        _displayedRows = currentTopN;
        renderTopN(currentTopN);
        drawPareto();
        drawRadar(_displayedRows);
    }

    // -------------------------------------------------------------------------
    // SLO Filter

    function initSloSliders() {
        const done = allResults.filter(r => r.state === 'done' && r.metrics);
        if (!done.length) return;

        for (const d of SLO_DEFS) {
            const vals = done.map(r => getMetricVal(r.metrics || {}, d.key)).filter(v => v != null);
            if (!vals.length) continue;

            const lo = Math.min(...vals);
            const hi = Math.max(...vals);
            // Add headroom so the default covers every candidate
            const sliderMin = lo > 0 ? parseFloat((lo * 0.9).toPrecision(4)) : 0;
            const sliderMax = parseFloat((hi * 1.1).toPrecision(4));
            const step = parseFloat(((sliderMax - sliderMin) / 200).toPrecision(3)) || 0.01;

            const slider = document.getElementById(d.sliderId);
            const num    = document.getElementById(d.numId);
            slider.min = sliderMin; slider.max = sliderMax; slider.step = step;
            num.min    = sliderMin; num.max    = sliderMax; num.step    = step;

            // Default: no constraint (max for upper-bound defs, min for lower-bound)
            const initVal = d.dir === 'max' ? sliderMax : sliderMin;
            slider.value = initVal;
            num.value    = initVal;
            sloInit[d.sliderId] = initVal;

            slider.addEventListener('input', () => {
                num.value = parseFloat(slider.value).toPrecision(5);
                applyFilter();
            });
            num.addEventListener('change', () => {
                slider.value = num.value;
                applyFilter();
            });
        }
        // Initial count
        applyFilter();
    }

    function getSloThresholds() {
        const t = {};
        for (const d of SLO_DEFS) {
            const num = document.getElementById(d.numId);
            t[d.key] = { val: parseFloat(num.value), dir: d.dir };
        }
        return t;
    }

    function getMetricVal(metrics, key) {
        if (metrics[key] != null) return metrics[key];
        // tok_per_wh may be absent — compute on the fly
        if (key === 'tok_per_wh') {
            const { total_token_tp: tp, total_latency_s: lat, total_energy_wh: e } = metrics;
            if (tp != null && lat != null && e > 0) return tp * lat / e;
        }
        return null;
    }

    function passesFilter(result, thresholds) {
        if (result.state !== 'done') return false;
        const m = result.metrics || {};
        for (const [key, { val, dir }] of Object.entries(thresholds)) {
            const v = getMetricVal(m, key);
            if (v == null) continue;   // missing metric → don't penalise
            if (dir === 'max' && v > val) return false;
            if (dir === 'min' && v < val) return false;
        }
        return true;
    }

    function isFilterActive(thresholds) {
        for (const d of SLO_DEFS) {
            const cur = thresholds[d.key]?.val;
            if (cur == null) continue;
            if (d.dir === 'max' && cur < sloInit[d.sliderId]) return true;
            if (d.dir === 'min' && cur > sloInit[d.sliderId]) return true;
        }
        return false;
    }

    function applyFilter() {
        const thresholds = getSloThresholds();
        const total = allResults.filter(r => r.state === 'done').length;
        const passing = allResults.filter(r => passesFilter(r, thresholds));
        const count = passing.length;

        // Update count badge
        const countEl = document.getElementById('slo-count');
        const labelEl = document.getElementById('top-n-label');
        countEl.textContent = `${count} / ${total} candidates pass`;
        countEl.style.color = count === 0 ? '#c0392b' : '';

        // Always show all passing candidates sorted by score (desc), then throughput.
        // When no filter is active this equals all done candidates.
        const sorted = [...passing].sort((a, b) => {
            const sa = a.score ?? -Infinity;
            const sb = b.score ?? -Infinity;
            if (sb !== sa) return sb - sa;
            return (b.metrics?.total_token_tp ?? 0) - (a.metrics?.total_token_tp ?? 0);
        });
        labelEl.textContent = isFilterActive(thresholds)
            ? `— SLO filter active (${count} / ${total})`
            : `(${count} candidates)`;
        _displayedRows = sorted;
        renderTopN(sorted);
        drawRadar(sorted);

        drawPareto(thresholds, isFilterActive(thresholds));
    }

    function resetSlo() {
        for (const d of SLO_DEFS) {
            const slider = document.getElementById(d.sliderId);
            const num    = document.getElementById(d.numId);
            if (sloInit[d.sliderId] == null) continue;
            slider.value = sloInit[d.sliderId];
            num.value    = sloInit[d.sliderId];
        }
        applyFilter();
    }

    // -------------------------------------------------------------------------
    // Selection: bidirectional table ↔ Pareto chart highlight

    function selectCandidate(label) {
        _selectedLabel = _selectedLabel === label ? null : label;
        document.querySelectorAll('#top-n-body tr').forEach(tr => {
            tr.classList.toggle('row-highlighted', tr.dataset.label === _selectedLabel);
        });
        if (_selectedLabel) {
            const sel = document.querySelector(`#top-n-body tr[data-label="${CSS.escape(_selectedLabel)}"]`);
            if (sel) sel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        }
        drawPareto();
        drawRadar(_displayedRows);
    }

    function initParetoInteraction() {
        // Pareto dot click → select candidate
        const paretoDiv = document.getElementById('pareto-plot');
        paretoDiv.on('plotly_click', (event) => {
            if (!event.points || !event.points[0]) return;
            const label = event.points[0].text;
            const result = allResults.find(r => r.label === label);
            if (!result) return;
            if (!passesFilter(result, getSloThresholds())) return;
            selectCandidate(label);
        });

        // Radar legend click → select candidate (return false prevents trace hide)
        const radarDiv = document.getElementById('radar-plot');
        radarDiv.on('plotly_legendclick', (event) => {
            const label = event.data?.[event.curveNumber]?.name;
            if (!label) return false;
            selectCandidate(label);
            return false; // prevent Plotly's default toggle-visibility behaviour
        });
    }

    // -------------------------------------------------------------------------
    // Rendering

    function fmtHw(label) {
        const m = candidatesMeta[label];
        if (!m) return '—';
        return Object.entries(m.hw_distribution || {}).map(([h, c]) => `${c}×${h}`).join(' + ');
    }

    function fmtPar(label) {
        const m = candidatesMeta[label];
        if (!m) return '—';
        const p = m.parallelism || {};
        return `tp${p.tp}_pp${p.pp}_dp${p.dp}`;
    }

    function renderTopN(rows, showAll = false) {
        const tbody = document.getElementById('top-n-body');
        tbody.innerHTML = '';
        rows.forEach((r, i) => {
            const m = r.metrics || {};
            const tr = document.createElement('tr');
            tr.dataset.label = r.label;
            tr.style.cursor = 'pointer';
            if (r.label === _selectedLabel) tr.classList.add('row-highlighted');
            tr.innerHTML = `
                <td>${i + 1}</td>
                <td title="${r.label}">${r.label}</td>
                <td>${fmtHw(r.label)}</td>
                <td>${fmtPar(r.label)}</td>
                <td>${candidatesMeta[r.label]?.pd_layout || '—'}</td>
                <td>${fmt(m.p99_ttft_ms)}</td>
                <td>${fmt(m.p99_tpot_ms)}</td>
                <td>${fmt(m.total_token_tp)}</td>
                <td>${fmt(getMetricVal(m, 'tok_per_wh'), 0)}</td>
                <td>${r.score != null ? r.score.toFixed(3) : '—'}</td>`;
            tr.addEventListener('click', () => selectCandidate(r.label));
            tbody.appendChild(tr);
        });
    }
    function fmt(v, d=2) { return v == null ? '—' : Number(v).toFixed(d); }

    function drawPareto(thresholds, filterActive) {
        const xKey = document.getElementById('pareto-x').value;
        const yKey = document.getElementById('pareto-y').value;
        const paretoLabels = new Set(currentPareto.map(r => r.label));

        // Determine threshold state (may be called without args from event listeners)
        if (!thresholds) {
            thresholds = getSloThresholds();
            filterActive = isFilterActive(thresholds);
        }

        const pts = allResults
            .filter(r => r.state === 'done' &&
                getMetricVal(r.metrics || {}, xKey) != null &&
                getMetricVal(r.metrics || {}, yKey) != null)
            .map(r => ({
                label:  r.label,
                x:      getMetricVal(r.metrics, xKey),
                y:      getMetricVal(r.metrics, yKey),
                pareto: paretoLabels.has(r.label),
                passes: passesFilter(r, thresholds),
            }));

        // Always colour by SLO-pass status so the Candidate Ranking membership
        // is visible in the chart regardless of whether a filter is active.
        const traces = [];

        // Grey: does not pass current SLO filter
        const filtered = pts.filter(p => !p.passes);
        if (filtered.length) traces.push({
            x: filtered.map(p => p.x), y: filtered.map(p => p.y), mode: 'markers',
            marker: { size: 7, color: '#ccc' }, text: filtered.map(p => p.label),
            hovertemplate: '%{text}<br>%{x:.2f}, %{y:.2f}<extra>Filtered out</extra>',
            name: 'Filtered out',
        });

        // Blue: passes SLO, not pareto-optimal
        const passOnly = pts.filter(p => p.passes && !p.pareto);
        if (passOnly.length) traces.push({
            x: passOnly.map(p => p.x), y: passOnly.map(p => p.y), mode: 'markers',
            marker: { size: 9, color: '#2E91E5' }, text: passOnly.map(p => p.label),
            hovertemplate: '%{text}<br>%{x:.2f}, %{y:.2f}<extra>In ranking</extra>',
            name: 'In ranking',
        });

        // Red star: passes SLO + pareto-optimal
        const passBest = pts.filter(p => p.passes && p.pareto);
        if (passBest.length) traces.push({
            x: passBest.map(p => p.x), y: passBest.map(p => p.y), mode: 'markers',
            marker: { size: 14, color: '#d62728', symbol: 'star' },
            text: passBest.map(p => p.label),
            hovertemplate: '%{text}<br>%{x:.2f}, %{y:.2f}<extra>Pareto-optimal</extra>',
            name: 'Pareto-optimal',
        });

        // Selected candidate: hollow orange ring drawn on top
        if (_selectedLabel) {
            const selPt = pts.find(p => p.label === _selectedLabel && p.passes);
            if (selPt) traces.push({
                x: [selPt.x], y: [selPt.y], mode: 'markers',
                marker: {
                    size: selPt.pareto ? 26 : 20,
                    color: 'rgba(0,0,0,0)',
                    line: { color: '#ff7700', width: 3 },
                    symbol: selPt.pareto ? 'star' : 'circle',
                },
                text: [selPt.label],
                hovertemplate: '%{text}<extra>Selected</extra>',
                name: 'Selected', showlegend: false,
            });
        }

        // Use react() instead of newPlot() so plotly_click handlers survive redraws
        Plotly.react('pareto-plot', traces, {
            xaxis: { title: xKey }, yaxis: { title: yKey },
            margin: { l: 60, r: 30, t: 20, b: 50 }, paper_bgcolor: 'rgba(0,0,0,0)',
        }, { responsive: true });
    }

    function drawRadar(rows) {
        if (!rows.length) { Plotly.purge('radar-plot'); return; }
        const metrics = [
            ['ttft',  'p99_ttft_ms',    'min'],
            ['tpot',  'p99_tpot_ms',    'min'],
            ['tp',    'total_token_tp', 'max'],
            ['power', 'total_energy_wh','min'],
            ['itl',   'p99_itl_ms',     'min'],
        ];
        const norms = {};
        for (const [name, key, dir] of metrics) {
            const vals = rows.map(r => r.metrics?.[key]).filter(v => v != null);
            if (!vals.length) { norms[name] = null; continue; }
            const lo = Math.min(...vals), hi = Math.max(...vals);
            const span = hi - lo;
            norms[name] = rows.map(r => {
                const v = r.metrics?.[key];
                if (v == null) return 0;
                if (span === 0) return 1;
                return dir === 'min' ? (hi - v) / span : (v - lo) / span;
            });
        }
        const selInRows = _selectedLabel !== null && rows.some(r => r.label === _selectedLabel);
        const traces = rows.map((r, i) => {
            const isSel = r.label === _selectedLabel;
            return {
                type: 'scatterpolar',
                r: metrics.map(([name]) => norms[name] ? norms[name][i] : 0),
                theta: metrics.map(([name]) => name),
                fill: 'toself', name: r.label,
                opacity: selInRows ? (isSel ? 1.0 : 0.15) : 0.7,
                line: { width: selInRows && isSel ? 3 : 1 },
            };
        });
        Plotly.react('radar-plot', traces, {
            polar: { radialaxis: { visible: true, range: [0, 1] } },
            paper_bgcolor: 'rgba(0,0,0,0)',
            margin: { l: 40, r: 40, t: 40, b: 40 },
        }, { responsive: true });
    }

    function wireSliders() {
        ['ttft','tpot','tp','power','tokwh'].forEach(id => {
            const s = document.getElementById('rw-' + id);
            const v = document.getElementById('rw-' + id + '-v');
            s.addEventListener('input', () => v.textContent = s.value);
        });
    }

    function autoSetParetoAxes(weights) {
        // Pick top-2 active metrics by weight
        const active = METRIC_DEFS
            .map(d => ({ ...d, w: weights[d.weightKey] || 0 }))
            .filter(d => d.w > 0)
            .sort((a, b) => b.w - a.w);
        if (active.length < 2) return;

        const top2 = active.slice(0, 2);
        const minOnes = top2.filter(d => d.dir === 'min');
        const maxOnes = top2.filter(d => d.dir === 'max');

        let xMetric, yMetric;
        if (minOnes.length === 1 && maxOnes.length === 1) {
            // One lower-is-better → X, one higher-is-better → Y
            xMetric = minOnes[0];
            yMetric = maxOnes[0];
        } else {
            // Both same direction: higher weight → Y, lower weight → X
            xMetric = top2[1];
            yMetric = top2[0];
        }

        const selX = document.getElementById('pareto-x');
        const selY = document.getElementById('pareto-y');
        if (selX.querySelector(`option[value="${xMetric.axisKey}"]`)) selX.value = xMetric.axisKey;
        if (selY.querySelector(`option[value="${yMetric.axisKey}"]`)) selY.value = yMetric.axisKey;
        document.getElementById('pareto-auto-badge').style.display = '';
    }

    async function rerank() {
        const weights = {
            ttft:       parseFloat(document.getElementById('rw-ttft').value),
            tpot:       parseFloat(document.getElementById('rw-tpot').value),
            throughput: parseFloat(document.getElementById('rw-tp').value),
            power:      parseFloat(document.getElementById('rw-power').value),
            tokwh:      parseFloat(document.getElementById('rw-tokwh').value),
        };
        const r = await fetch(`/api/dse/jobs/${JOB_ID}/rerank`, {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ weights, top_n: allResults.length || 999 }),
        });
        const j = await r.json();
        if (!r.ok) { alert('rerank failed: ' + (j.detail || 'error')); return; }
        currentTopN = j.top_n;
        currentPareto = j.pareto;
        if (j.all_results) allResults = j.all_results;
        autoSetParetoAxes(weights);
        applyFilter();
    }
})();
