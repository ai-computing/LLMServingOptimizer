/* DSE explore page — gather spec, hit /api/dse/dry-run + /api/dse/jobs. */
(function () {
    'use strict';

    let catalog = null;     // /api/dse/catalog response
    let datasets = [];
    let historyJobs = [];   // /api/dse/jobs/history response

    document.addEventListener('DOMContentLoaded', async () => {
        await loadCatalog();
        await loadDatasets();
        await loadHistory();
        await loadConcurrency();
        // Seed one resource pool row and sync model dropdown
        addHwRow();
        updateModelDropdown();
        document.getElementById('btn-add-hw').addEventListener('click', () => { addHwRow(); updateModelDropdown(); });
        document.getElementById('btn-dry-run').addEventListener('click', dryRun);
        document.getElementById('btn-start').addEventListener('click', startJob);
        document.getElementById('history-select').addEventListener('change', () => {
            document.getElementById('btn-apply-history').disabled =
                !document.getElementById('history-select').value;
        });
        document.getElementById('btn-apply-history').addEventListener('click', applyHistorySpec);
        wirePriorityRows();
    });

    async function loadCatalog() {
        const r = await fetch('/api/dse/catalog');
        catalog = await r.json();
        populateFabrics();
        // Initial model dropdown will be set by updateModelDropdown() after addHwRow()
    }

    function populateFabrics() {
        const sel = document.getElementById('fabric-select');
        if (!sel) return;
        const saved = sel.value;
        sel.innerHTML = '';
        const def = document.createElement('option');
        def.value = ''; def.textContent = 'Catalog default (per-HW)';
        sel.appendChild(def);
        for (const [name, f] of Object.entries(catalog.fabrics || {})) {
            const opt = document.createElement('option');
            opt.value = name;
            opt.textContent = name;
            opt.dataset.desc = (f && f.description) || '';
            sel.appendChild(opt);
        }
        sel.value = saved || '';
        sel.addEventListener('change', updateFabricDesc);
        updateFabricDesc();
    }

    function updateFabricDesc() {
        const sel = document.getElementById('fabric-select');
        const desc = document.getElementById('fabric-desc');
        if (!sel || !desc) return;
        const opt = sel.options[sel.selectedIndex];
        desc.textContent = (opt && opt.dataset.desc) || '';
    }

    function updateModelDropdown() {
        if (!catalog) return;
        const sel = document.getElementById('model-select');
        const saved = sel.value;

        // Collect hardware types currently in the resource pool
        const selectedHws = [...document.querySelectorAll('#resource-pool-body select')].map(s => s.value);

        // Compute intersection of available models for all selected hardware
        let compatible = null;
        for (const hw of selectedHws) {
            const hwModels = new Set(Object.keys(catalog.hardware[hw]?.available_models || {}));
            if (compatible === null) {
                compatible = hwModels;
            } else {
                compatible = new Set([...compatible].filter(m => hwModels.has(m)));
            }
        }
        // Fall back to all catalog models if no hardware selected or intersection is empty
        const allModels = Object.keys(catalog.models);
        const show = (compatible && compatible.size > 0) ? [...compatible] : allModels;

        sel.innerHTML = '';
        for (const m of show) {
            const opt = document.createElement('option');
            opt.value = m; opt.textContent = m;
            sel.appendChild(opt);
        }
        // Restore prior selection if still compatible, else pick first available
        sel.value = (saved && show.includes(saved)) ? saved : (show[0] || '');
    }

    async function loadDatasets() {
        // /api/datasets returns a list of {path, name, family, compatible_models}
        // (NOT wrapped in {datasets: ...}). Use `path` as the value (workload
        // spec expects the relative path) and `name` as the display text,
        // with family annotation when known.
        try {
            const r = await fetch('/api/datasets');
            datasets = await r.json();
        } catch (e) { datasets = []; }
        const sel = document.getElementById('dataset-select');
        for (const ds of datasets) {
            const opt = document.createElement('option');
            opt.value = ds.path;
            opt.textContent = ds.family && ds.family !== 'unknown'
                ? `${ds.name}  [${ds.family}]`
                : ds.name;
            sel.appendChild(opt);
        }
    }

    async function loadConcurrency() {
        try {
            const r = await fetch('/api/dse/concurrency');
            const j = await r.json();
            const el = document.getElementById('max-concurrent');
            el.value = j.max_concurrent;
            el.max = j.max_concurrent;
            el.title += ` (자동값: ${j.max_concurrent})`;
        } catch (e) {
            console.warn('Failed to load concurrency:', e);
        }
    }

    async function loadHistory() {
        try {
            const r = await fetch('/api/dse/jobs/history');
            if (!r.ok) return;
            historyJobs = await r.json();
            const sel = document.getElementById('history-select');
            for (const job of historyJobs) {
                const opt = document.createElement('option');
                opt.value = job.job_id;
                const date = job.created_at
                    ? job.created_at.substring(0, 16).replace('T', ' ')
                    : '?';
                const model = job.model_name
                    ? job.model_name.split('/').pop()
                    : '?';
                opt.textContent = `${date} | ${model} | ${job.hw_summary || '?'} [${job.state}]`;
                sel.appendChild(opt);
            }
        } catch (e) {
            console.warn('Failed to load DSE history:', e);
        }
    }

    function applyHistorySpec() {
        const sel = document.getElementById('history-select');
        const job = historyJobs.find(j => j.job_id === sel.value);
        if (!job || !job.spec) return;
        applySpec(job.spec);
    }

    function applySpec(spec) {
        // 1. Resource Pool
        const tbody = document.getElementById('resource-pool-body');
        tbody.innerHTML = '';
        for (const item of (spec.resource_pool?.items || [])) {
            addHwRow(item.hw, item.min, item.max);
        }
        const totalMaxEl = document.getElementById('total-max');
        if (spec.resource_pool?.total_max_npus != null) {
            totalMaxEl.value = spec.resource_pool.total_max_npus;
        } else {
            totalMaxEl.value = '';
        }

        // 2. Model & Workload — refresh dropdown first so the restored model is valid
        updateModelDropdown();
        if (spec.model?.name) {
            document.getElementById('model-select').value = spec.model.name;
        }
        if (spec.model?.fp != null) {
            document.getElementById('fp-select').value = String(spec.model.fp);
        }
        const fabricSel = document.getElementById('fabric-select');
        if (fabricSel) {
            fabricSel.value = spec.fabric || '';
            updateFabricDesc();
        }
        if (spec.workload?.dataset) {
            document.getElementById('dataset-select').value = spec.workload.dataset;
        }
        if (spec.workload?.num_req != null) {
            document.getElementById('num-req').value = spec.workload.num_req;
        }
        if (spec.workload?.timeout_s != null) {
            document.getElementById('timeout-s').value = spec.workload.timeout_s;
        }

        // 3. Features
        if (spec.features) {
            document.getElementById('feat-pd').checked = !!spec.features.allow_pd_disagg;
            document.getElementById('feat-prefix').checked = !!spec.features.prefix_caching;
            document.getElementById('feat-attn-off').checked = !!spec.features.attn_offloading;
        }

        // 4. Search
        if (spec.search) {
            if (spec.search.max_combinations != null)
                document.getElementById('search-max').value = spec.search.max_combinations;
            if (spec.search.sampling_strategy)
                document.getElementById('search-sampling').value = spec.search.sampling_strategy;
            if (spec.search.random_seed != null)
                document.getElementById('search-seed').value = spec.search.random_seed;
            if (spec.search.use_stage1 != null)
                document.getElementById('use-stage1').checked = spec.search.use_stage1;
            if (spec.search.use_stage2 != null)
                document.getElementById('use-stage2').checked = spec.search.use_stage2;
        }
        if (spec.top_n != null)
            document.getElementById('top-n').value = spec.top_n;
        if (spec.max_concurrent != null)
            document.getElementById('max-concurrent').value = spec.max_concurrent;

        // 5. Weights / objectives — set radios first, then checkboxes, then fire change
        if (spec.weights) {
            // spec key → checkbox id / radio name (note: throughput → obj-tp, pri-tp)
            const OBJ_MAP = [
                ['ttft',       'ttft',  'ttft'],
                ['tpot',       'tpot',  'tpot'],
                ['throughput', 'tp',    'tp'],
                ['power',      'power', 'power'],
                ['tokwh',      'tokwh', 'tokwh'],
            ];
            for (const [specKey, cbId, priName] of OBJ_MAP) {
                const w = spec.weights[specKey];
                if (w == null) continue;
                const radioVal = w >= 9 ? '9' : w >= 3 ? '3' : '1';
                const radio = document.querySelector(`input[name="pri-${priName}"][value="${radioVal}"]`);
                if (radio) radio.checked = true;
                const cb = document.getElementById(`obj-${cbId}`);
                if (cb) cb.checked = w > 0;
            }
        }
        // Trigger visual update for priority rows
        document.querySelectorAll('.obj-row input[type="checkbox"]').forEach(
            cb => cb.dispatchEvent(new Event('change'))
        );
    }

    function addHwRow(hwValue, minValue, maxValue) {
        if (!catalog) return;
        const tbody = document.getElementById('resource-pool-body');
        const tr = document.createElement('tr');
        const select = document.createElement('select');
        for (const hw of Object.keys(catalog.hardware)) {
            const opt = document.createElement('option');
            opt.value = hw; opt.textContent = hw;
            select.appendChild(opt);
        }
        if (hwValue != null) select.value = hwValue;
        select.addEventListener('change', updateModelDropdown);
        const minI = mkInput('number', minValue != null ? String(minValue) : '0', 0);
        const maxI = mkInput('number', maxValue != null ? String(maxValue) : '2', 0);
        const rm = document.createElement('button');
        rm.type = 'button'; rm.textContent = '✕';
        rm.className = 'btn-secondary'; rm.style.padding = '4px 10px';
        rm.addEventListener('click', () => { tr.remove(); updateModelDropdown(); });

        tr.appendChild(td(select));
        tr.appendChild(td(minI));
        tr.appendChild(td(maxI));
        tr.appendChild(td(rm));
        tbody.appendChild(tr);
    }
    function mkInput(type, value, min) {
        const i = document.createElement('input');
        i.type = type; i.value = value;
        if (min !== undefined) i.min = String(min);
        i.style.maxWidth = '80px';
        return i;
    }
    function td(el) { const t = document.createElement('td'); t.appendChild(el); return t; }

    function collectSpec() {
        const items = [...document.querySelectorAll('#resource-pool-body tr')].map(tr => {
            const cells = tr.cells;
            return {
                hw: cells[0].querySelector('select').value,
                min: parseInt(cells[1].querySelector('input').value, 10),
                max: parseInt(cells[2].querySelector('input').value, 10),
            };
        });
        const totalMax = parseInt(document.getElementById('total-max').value, 10);
        const constraints = {};

        // Derive weights from objective checkboxes + Low/Med/High priority.
        // Unchecked objectives get weight 0 (excluded from scoring).
        // Priority values: Low=1, Medium=3, High=9.
        const priVal = (name) => {
            const sel = document.querySelector(`input[name="${name}"]:checked`);
            return sel ? Number(sel.value) : 3;
        };
        const weights = {
            ttft:       document.getElementById('obj-ttft').checked  ? priVal('pri-ttft')  : 0,
            tpot:       document.getElementById('obj-tpot').checked  ? priVal('pri-tpot')  : 0,
            throughput: document.getElementById('obj-tp').checked    ? priVal('pri-tp')    : 0,
            power:      document.getElementById('obj-power').checked ? priVal('pri-power') : 0,
            tokwh:      document.getElementById('obj-tokwh').checked ? priVal('pri-tokwh') : 0,
        };
        // Ensure at least one weight > 0 (fall back to equal if nothing checked)
        if (Object.values(weights).every(v => v === 0)) {
            weights.ttft = weights.tpot = weights.throughput = weights.power = weights.tokwh = 1;
        }

        return {
            resource_pool: {
                items: items,
                total_max_npus: Number.isFinite(totalMax) ? totalMax : null,
            },
            model: {
                name: document.getElementById('model-select').value,
                fp: parseInt(document.getElementById('fp-select').value, 10),
            },
            fabric: document.getElementById('fabric-select').value || null,
            workload: {
                dataset: document.getElementById('dataset-select').value,
                num_req: parseInt(document.getElementById('num-req').value, 10) || 100,
                timeout_s: parseInt(document.getElementById('timeout-s').value, 10) || 120,
            },
            constraints: constraints,
            features: {
                allow_pd_disagg: document.getElementById('feat-pd').checked,
                prefix_caching: document.getElementById('feat-prefix').checked,
                attn_offloading: document.getElementById('feat-attn-off').checked,
            },
            search: {
                max_combinations: parseInt(document.getElementById('search-max').value, 10) || 20,
                sampling_strategy: document.getElementById('search-sampling').value,
                random_seed: parseInt(document.getElementById('search-seed').value, 10) || 0,
                use_stage1: document.getElementById('use-stage1').checked,
                use_stage2: document.getElementById('use-stage2').checked,
            },
            weights: weights,
            top_n: parseInt(document.getElementById('top-n').value, 10) || 5,
            max_concurrent: parseInt(document.getElementById('max-concurrent').value, 10) || null,
        };
    }

    async function dryRun() {
        const spec = collectSpec();
        const status = document.getElementById('dse-status');
        status.textContent = 'Estimating…';
        try {
            const r = await fetch('/api/dse/dry-run', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(spec),
            });
            const j = await r.json();
            if (!r.ok) {
                status.textContent = '❌ ' + (j.detail || 'error');
                document.getElementById('dry-run-results').style.display = 'none';
                return;
            }
            const unique = j.estimated_candidates;
            const sim = j.simulated_candidates ?? Math.min(unique, spec.search.max_combinations);
            if (unique === 0) {
                status.textContent = '⚠️ 0 candidates — the selected model may exceed GPU memory, or no TP profile exists for this hardware+model combination.';
                document.getElementById('dry-run-results').style.display = 'none';
            } else {
                status.textContent = sim < unique
                    ? `≈ ${unique} candidates found → ${sim} will be simulated (sampled, cap=${spec.search.max_combinations})`
                    : `≈ ${sim} candidates will be simulated (all found)`;
                renderDryRunList(j.candidates || [], unique, sim);
            }
        } catch (e) {
            status.textContent = '❌ ' + e.message;
            document.getElementById('dry-run-results').style.display = 'none';
        }
    }

    function renderDryRunList(candidates, unique, sim) {
        const section = document.getElementById('dry-run-results');
        const label = document.getElementById('dry-run-count-label');
        const tbody = document.getElementById('dry-run-list-body');

        label.textContent = `(${unique} total, ${sim} to simulate)`;
        tbody.innerHTML = '';

        for (const c of candidates) {
            const hw = Object.entries(c.hw_distribution || {})
                .filter(([, n]) => n > 0)
                .map(([hw, n]) => `${n}×${hw}`)
                .join(' + ') || '—';
            const par = c.parallelism || {};
            const willSim = c.will_simulate;
            const tr = document.createElement('tr');
            if (!willSim) tr.style.opacity = '0.45';
            tr.innerHTML = `
                <td>${c.label}</td>
                <td>${hw}</td>
                <td>${par.tp ?? '—'}</td>
                <td>${par.pp ?? '—'}</td>
                <td>${par.dp ?? '—'}</td>
                <td>${c.pd_layout || '—'}</td>
                <td>${willSim ? '✓' : '—'}</td>`;
            tbody.appendChild(tr);
        }
        section.style.display = candidates.length ? '' : 'none';
    }

    async function startJob() {
        const spec = collectSpec();
        const status = document.getElementById('dse-status');
        const btn = document.getElementById('btn-start');
        btn.disabled = true;
        status.textContent = 'Starting job…';
        try {
            const r = await fetch('/api/dse/jobs', {
                method: 'POST', headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(spec),
            });
            const j = await r.json();
            if (!r.ok) {
                status.textContent = '❌ ' + (j.detail || 'create failed');
                btn.disabled = false;
                return;
            }
            window.location = `/dse/jobs/${j.job_id}`;
        } catch (e) {
            status.textContent = '❌ ' + e.message;
            btn.disabled = false;
        }
    }

    function wirePriorityRows() {
        document.querySelectorAll('.obj-row').forEach(row => {
            const cb = row.querySelector('input[type="checkbox"]');
            const priSpan = row.querySelector('.obj-pri');
            if (!cb || !priSpan) return;
            const update = () => {
                priSpan.style.opacity = cb.checked ? '1' : '0.3';
                priSpan.querySelectorAll('input').forEach(i => i.disabled = !cb.checked);
            };
            cb.addEventListener('change', update);
            update();
        });
    }
})();
