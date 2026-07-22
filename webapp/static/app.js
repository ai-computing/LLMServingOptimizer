/* LLMServingSim Web UI — vanilla JS for scenario builder + progress + results */
(function () {
    'use strict';

    const LLMSS = {};
    window.LLMSS = LLMSS;

    // -------------------------------------------------------------------
    // Toast helper
    // -------------------------------------------------------------------
    function toast(msg, kind = 'info', timeoutMs = 4000) {
        const host = document.getElementById('toast-host');
        if (!host) return;
        const el = document.createElement('div');
        el.className = `toast toast-${kind}`;
        el.textContent = msg;
        host.appendChild(el);
        setTimeout(() => el.remove(), timeoutMs);
    }
    LLMSS.toast = toast;

    // -------------------------------------------------------------------
    // Catalog/dataset cache
    // -------------------------------------------------------------------
    let _hardware = null;   // {hwName: [model, ...]}
    let _datasets = null;   // [{path, name, family, compatible_models}]

    async function loadHardware() {
        if (_hardware) return _hardware;
        const r = await fetch('/api/hardware');
        if (!r.ok) throw new Error('failed to fetch hardware');
        _hardware = await r.json();
        return _hardware;
    }

    async function loadDatasets() {
        if (_datasets) return _datasets;
        const r = await fetch('/api/datasets');
        if (!r.ok) throw new Error('failed to fetch datasets');
        _datasets = await r.json();
        return _datasets;
    }

    // -------------------------------------------------------------------
    // Index page (scenario builder)
    // -------------------------------------------------------------------
    LLMSS.initIndexPage = async function () {
        try {
            await loadHardware();
            await loadDatasets();
        } catch (e) {
            toast('Failed to load catalog: ' + e.message, 'error', 8000);
            return;
        }

        populateDatasetSelect();

        // Bind controls
        document.getElementById('btn-add-scenario').addEventListener('click', () => {
            addScenarioForm();
        });
        document.getElementById('btn-enumerate').addEventListener('click', enumerateAll);
        document.getElementById('btn-run').addEventListener('click', runSweep);
        const numReq = document.getElementById('num-req');
        const numReqVal = document.getElementById('num-req-value');
        numReq.addEventListener('input', () => { numReqVal.textContent = numReq.value; });

        // Start with one scenario form.
        addScenarioForm();

        // Cluster Config Builder (lower section of the same page).
        initClusterBuilder();
    };

    function populateDatasetSelect(modelHint) {
        const sel = document.getElementById('dataset-select');
        if (!sel) return;
        const prev = sel.value;
        sel.innerHTML = '';
        const datasets = _datasets || [];
        for (const ds of datasets) {
            // If a model hint is set, only show datasets whose family matches.
            if (modelHint && ds.family !== 'unknown' && ds.compatible_models.length) {
                if (!ds.compatible_models.includes(modelHint)) continue;
            }
            const opt = document.createElement('option');
            opt.value = ds.path;
            opt.textContent = `${ds.name}  [${ds.family}]`;
            sel.appendChild(opt);
        }
        if (prev && [...sel.options].some(o => o.value === prev)) sel.value = prev;
    }

    // ----- Scenario form management -----

    let _scenarioCounter = 0;
    let _activeScenarioIdx = -1;

    function addScenarioForm() {
        _scenarioCounter += 1;
        const idx = _scenarioCounter;

        const tplForm = document.getElementById('scenario-form-template');
        const formNode = tplForm.content.firstElementChild.cloneNode(true);
        formNode.dataset.scenarioIdx = String(idx);
        formNode.querySelector('.scenario-title').textContent = `Scenario ${idx}`;
        const nameInput = formNode.querySelector('.scenario-name');
        nameInput.value = `Scenario ${idx}`;
        document.getElementById('scenario-forms').appendChild(formNode);

        // Add tab button.
        const tab = document.createElement('button');
        tab.className = 'tab-btn';
        tab.type = 'button';
        tab.dataset.scenarioIdx = String(idx);
        tab.textContent = nameInput.value;
        const close = document.createElement('span');
        close.className = 'tab-close';
        close.textContent = '×';
        close.title = 'Remove scenario';
        close.addEventListener('click', (ev) => {
            ev.stopPropagation();
            removeScenarioForm(idx);
        });
        tab.appendChild(close);
        tab.addEventListener('click', () => switchTab(idx));
        document.getElementById('scenario-tabs').appendChild(tab);

        nameInput.addEventListener('input', () => {
            tab.firstChild.nodeValue = nameInput.value || `Scenario ${idx}`;
        });

        // Add-IG button
        formNode.querySelector('.btn-add-ig').addEventListener('click', () => {
            addInstanceGroupRow(formNode);
        });
        // Seed with one row.
        addInstanceGroupRow(formNode);

        switchTab(idx);
    }

    function removeScenarioForm(idx) {
        const form = document.querySelector(`.scenario-form[data-scenario-idx="${idx}"]`);
        const tab = document.querySelector(`.tab-btn[data-scenario-idx="${idx}"]`);
        if (form) form.remove();
        if (tab) tab.remove();
        // Re-pick an active tab.
        const remaining = document.querySelectorAll('.tab-btn');
        if (remaining.length) {
            switchTab(parseInt(remaining[0].dataset.scenarioIdx, 10));
        } else {
            _activeScenarioIdx = -1;
        }
    }

    function switchTab(idx) {
        _activeScenarioIdx = idx;
        document.querySelectorAll('.scenario-form').forEach(f => {
            f.hidden = (f.dataset.scenarioIdx !== String(idx));
        });
        document.querySelectorAll('.tab-btn').forEach(t => {
            t.classList.toggle('active', t.dataset.scenarioIdx === String(idx));
        });
    }

    function addInstanceGroupRow(formNode, preset) {
        const tbody = formNode.querySelector('.ig-body');
        const tpl = document.getElementById('instance-group-row-template');
        const row = tpl.content.firstElementChild.cloneNode(true);

        // Hardware select
        const hwSel = row.querySelector('.ig-hardware');
        const hwNames = Object.keys(_hardware || {}).sort();
        for (const hw of hwNames) {
            const opt = document.createElement('option');
            opt.value = hw;
            opt.textContent = hw;
            hwSel.appendChild(opt);
        }

        const modelSel = row.querySelector('.ig-model');

        function refreshModels() {
            modelSel.innerHTML = '';
            const models = (_hardware[hwSel.value] || []);
            for (const m of models) {
                const opt = document.createElement('option');
                opt.value = m;
                opt.textContent = m;
                modelSel.appendChild(opt);
            }
            // Filter dataset list to match the chosen model if a single scenario.
            populateDatasetSelect(modelSel.value);
        }

        hwSel.addEventListener('change', refreshModels);
        modelSel.addEventListener('change', () => populateDatasetSelect(modelSel.value));

        if (preset && hwNames.includes(preset.hardware)) {
            hwSel.value = preset.hardware;
            refreshModels();
            if (preset.model && [...modelSel.options].some(o => o.value === preset.model))
                modelSel.value = preset.model;
            if (preset.npu_count)
                row.querySelector('.ig-npu-count').value = preset.npu_count;
            const pdSel = row.querySelector('.ig-pd-role');
            const pdVal = preset.pd_role || 'auto';
            if ([...pdSel.options].some(o => o.value === pdVal)) pdSel.value = pdVal;
        } else if (hwNames.length) {
            hwSel.value = hwNames[0];
            refreshModels();
        }

        row.querySelector('.btn-remove-ig').addEventListener('click', () => {
            row.remove();
            renumberRows(tbody);
        });

        tbody.appendChild(row);
        renumberRows(tbody);
    }

    function syncScenarioFromConfig(cfg) {
        // Collect all instances across nodes from the loaded cluster config.
        const instances = (cfg.nodes || []).flatMap(n => n.instances || []);
        if (!instances.length) return;

        // Apply to the currently active scenario form (fallback: first form).
        let formNode = document.querySelector(
            `.scenario-form[data-scenario-idx="${_activeScenarioIdx}"]`
        );
        if (!formNode) formNode = document.querySelector('.scenario-form');
        if (!formNode) return;

        console.log('[sync] updating scenario form', _activeScenarioIdx, 'with', instances.length, 'instances');

        const tbody = formNode.querySelector('.ig-body');
        tbody.innerHTML = '';

        for (const inst of instances) {
            const pdType = inst.pd_type || null;
            addInstanceGroupRow(formNode, {
                hardware:  inst.hardware,
                model:     inst.model_name || inst.model || '',
                npu_count: inst.npu_num || 1,
                pd_role:   pdType === 'prefill' ? 'prefill'
                         : pdType === 'decode'  ? 'decode'
                         : 'auto',
            });
        }
    }

    function renumberRows(tbody) {
        const rows = tbody.querySelectorAll('.ig-row');
        rows.forEach((r, i) => {
            r.querySelector('.ig-idx').textContent = String(i + 1);
        });
    }

    // ----- Form-state collection -----

    function collectScenarios() {
        const forms = document.querySelectorAll('.scenario-form');
        const out = [];
        forms.forEach(form => {
            const name = form.querySelector('.scenario-name').value || 'Scenario';
            const groups = [];
            form.querySelectorAll('.ig-row').forEach(row => {
                groups.push({
                    hardware: row.querySelector('.ig-hardware').value,
                    model: row.querySelector('.ig-model').value,
                    npu_count: parseInt(row.querySelector('.ig-npu-count').value, 10) || 1,
                    pd_role: row.querySelector('.ig-pd-role').value,
                });
            });
            out.push({
                name,
                instance_groups: groups,
                axes: {
                    vary_tp: form.querySelector('.ax-tp').checked,
                    vary_pp: form.querySelector('.ax-pp').checked,
                    vary_dp: form.querySelector('.ax-dp').checked,
                    include_pd: form.querySelector('.ax-pd').checked,
                },
            });
        });
        return out;
    }

    // Power profile captured from the most recently loaded cluster config
    // (set in ccLoad). Forwarded via workload so each sweep variant runs
    // power simulation. Null until the user loads a config with a `power`
    // block; cleared by ccClearLoadedPower (e.g. when starting a fresh config).
    let _loadedPowerTemplate = null;

    function collectWorkload() {
        // Per-sweep timeout override. Falls back to the CONFIG_TIMEOUT_S
        // default in webapp/config.py when blank or invalid. Lower bound 10s
        // mirrors the HTML min= attribute.
        const timeoutRaw = parseInt(document.getElementById('timeout-s').value, 10);
        const timeoutS = (Number.isFinite(timeoutRaw) && timeoutRaw >= 10) ? timeoutRaw : null;
        return {
            backend: (document.getElementById('backend-select') || {value: 'legacy'}).value,
            dataset: document.getElementById('dataset-select').value,
            num_req: parseInt(document.getElementById('num-req').value, 10) || 100,
            phase: (document.querySelector('input[name="phase"]:checked') || {}).value || 'full',
            power_template: _loadedPowerTemplate,
            timeout_s: timeoutS,
        };
    }

    // ----- Enumerate -----

    let _lastEnum = null; // {scenarios: [{name, configs:[...]}], workload}

    async function enumerateAll() {
        const scenarios = collectScenarios();
        const workload = collectWorkload();
        if (!scenarios.length) {
            toast('Define at least one scenario', 'error');
            return;
        }
        if (!workload.dataset) {
            toast('Select a dataset', 'error');
            return;
        }

        const btn = document.getElementById('btn-enumerate');
        const orig = btn.textContent;
        btn.disabled = true;
        btn.innerHTML = '<span class="spinner"></span>Enumerating…';

        try {
            const r = await fetch('/api/enumerate', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({scenarios, workload}),
            });
            if (!r.ok) {
                const err = await r.text();
                throw new Error(`enumerate failed (${r.status}): ${err}`);
            }
            const data = await r.json();
            _lastEnum = {scenarios: data.scenarios, scenarioInputs: scenarios, workload};
            renderEnumPreview(data);
            document.getElementById('btn-run').disabled = false;
        } catch (e) {
            toast('Enumerate error: ' + e.message, 'error', 8000);
        } finally {
            btn.disabled = false;
            btn.textContent = orig;
        }
    }

    function renderEnumPreview(data) {
        const host = document.getElementById('enum-results');
        const summary = document.getElementById('enum-summary');
        host.innerHTML = '';

        let totalCount = 0, totalSec = 0;
        data.scenarios.forEach((scn, scnIdx) => {
            totalCount += scn.count;
            totalSec += scn.estimated_total_s;

            const section = document.createElement('div');
            section.className = 'card preview-section';
            section.dataset.scenarioIdx = String(scnIdx);

            const heading = document.createElement('h3');
            heading.textContent = `${scn.name} — ${scn.count} configs (~${formatDur(scn.estimated_total_s)})`;
            section.appendChild(heading);

            if (scn.exceeds_soft_cap) {
                const banner = document.createElement('div');
                banner.className = 'warning-banner';
                banner.textContent = `${scn.count} configs exceed the soft cap of ${scn.soft_cap}. ` +
                    `By default only the first ${scn.soft_cap} are selected. Uncheck to drop, recheck to include.`;
                section.appendChild(banner);
            }

            if (!scn.configs.length) {
                const empty = document.createElement('p');
                empty.className = 'hint';
                empty.textContent = 'No valid configs for this scenario.';
                section.appendChild(empty);
                host.appendChild(section);
                return;
            }

            const tbl = document.createElement('table');
            tbl.className = 'matrix-table';
            tbl.innerHTML = `
                <thead>
                    <tr>
                        <th><input type="checkbox" class="select-all" checked></th>
                        <th>#</th><th>Label</th><th>TP</th><th>PP</th><th>DP</th>
                        <th>P/D</th><th>Phys NPUs</th><th>Est. time (s)</th>
                    </tr>
                </thead>
                <tbody></tbody>
            `;
            const tbody = tbl.querySelector('tbody');
            scn.configs.forEach((cfg, i) => {
                const tr = document.createElement('tr');
                const overCap = scn.exceeds_soft_cap && i >= scn.soft_cap;
                tr.innerHTML = `
                    <td><input type="checkbox" class="cfg-cb" data-label="${cfg.label}" ${overCap ? '' : 'checked'}></td>
                    <td>${i + 1}</td>
                    <td><code>${cfg.label}</code></td>
                    <td>${cfg.tp}</td>
                    <td>${cfg.pp}</td>
                    <td>${cfg.dp}</td>
                    <td>${cfg.pd_layout}</td>
                    <td>${cfg.phys_npus}</td>
                    <td>${cfg.estimated_s}</td>
                `;
                tbody.appendChild(tr);
            });
            section.appendChild(tbl);

            // Wire select-all
            const selAll = tbl.querySelector('.select-all');
            selAll.addEventListener('change', () => {
                tbl.querySelectorAll('.cfg-cb').forEach(cb => { cb.checked = selAll.checked; });
            });

            host.appendChild(section);
        });

        summary.textContent = `Total: ${totalCount} configs (~${formatDur(totalSec)} sequential, ` +
            `~${formatDur(Math.ceil(totalSec / 4))} with 4-way concurrency)`;
    }

    function formatDur(seconds) {
        if (seconds < 60) return `${seconds}s`;
        const m = Math.floor(seconds / 60);
        const s = seconds % 60;
        return s ? `${m}m${s}s` : `${m}m`;
    }

    // ----- Run -----

    async function runSweep() {
        if (!_lastEnum) {
            toast('Enumerate first', 'error');
            return;
        }
        // Collect selected labels per scenario in same order.
        const selected = [];
        document.querySelectorAll('#enum-results .preview-section').forEach(section => {
            const labels = [...section.querySelectorAll('.cfg-cb')]
                .filter(cb => cb.checked)
                .map(cb => cb.dataset.label);
            selected.push(labels);
        });
        if (!selected.some(arr => arr.length)) {
            toast('Select at least one config', 'error');
            return;
        }

        const btn = document.getElementById('btn-run');
        const orig = btn.textContent;
        btn.disabled = true;
        btn.innerHTML = '<span class="spinner"></span>Launching…';

        try {
            const r = await fetch('/api/sweeps', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({
                    scenarios: _lastEnum.scenarioInputs,
                    workload: _lastEnum.workload,
                    selected_labels: selected,
                }),
            });
            if (!r.ok) {
                const err = await r.text();
                throw new Error(`run failed (${r.status}): ${err}`);
            }
            const data = await r.json();
            window.location = `/sweep/${data.sweep_id}`;
        } catch (e) {
            toast('Run error: ' + e.message, 'error', 8000);
            btn.disabled = false;
            btn.textContent = orig;
        }
    }

    // -------------------------------------------------------------------
    // Cluster Config Builder
    // -------------------------------------------------------------------

    function initClusterBuilder() {
        document.getElementById('cc-add-node').addEventListener('click', () => addCcNodeRow());
        document.getElementById('cc-load-btn').addEventListener('click', async () => {
            const path = document.getElementById('cc-load-select').value;
            if (!path) return;
            await ccLoad(path);
        });
        document.getElementById('cc-save-btn').addEventListener('click', ccSave);
        document.getElementById('cc-delete-btn').addEventListener('click', ccDelete);
        populateCcLoadDropdown();
        addCcNodeRow();  // default: one empty node
    }

    async function ccDelete() {
        const sel = document.getElementById('cc-load-select');
        const path = sel.value;
        if (!path) {
            setCcStatus('❌ Select a config from the dropdown first', 'error');
            return;
        }
        // Server enforces the cluster_config/web/ restriction, but warn the
        // user up-front for reference configs so they don't think it's broken.
        if (!path.startsWith('cluster_config/web/')) {
            setCcStatus('❌ Only configs under cluster_config/web/ can be deleted', 'error');
            return;
        }
        if (!confirm(`Delete ${path}?`)) return;

        const btn = document.getElementById('cc-delete-btn');
        const orig = btn.textContent;
        btn.disabled = true;
        btn.innerHTML = '<span class="spinner"></span>Deleting…';
        try {
            const r = await fetch(`/api/cluster-configs?path=${encodeURIComponent(path)}`, {
                method: 'DELETE',
            });
            const j = await r.json().catch(() => ({}));
            if (!r.ok) {
                setCcStatus('❌ ' + (j.detail || 'delete failed'), 'error');
                return;
            }
            setCcStatus('🗑️ Deleted ' + path, 'ok');
            await populateCcLoadDropdown();
        } catch (e) {
            setCcStatus('❌ ' + e.message, 'error');
        } finally {
            btn.disabled = false;
            btn.textContent = orig;
        }
    }

    async function populateCcLoadDropdown() {
        const sel = document.getElementById('cc-load-select');
        const prev = sel.value;
        sel.innerHTML = '<option value="">(new config)</option>';
        try {
            const r = await fetch('/api/cluster-configs');
            if (!r.ok) return;
            const data = await r.json();
            for (const cfg of data.configs || []) {
                const opt = document.createElement('option');
                opt.value = cfg.path;
                opt.textContent = cfg.web ? `[web] ${cfg.path}` : cfg.path;
                sel.appendChild(opt);
            }
            if (prev && [...sel.options].some(o => o.value === prev)) sel.value = prev;
        } catch (e) {
            console.warn('could not fetch cluster-configs', e);
        }
    }

    function addCcNodeRow(nodeData) {
        const container = document.getElementById('cc-nodes');
        const tpl = document.getElementById('cc-node-template');
        const nodeEl = tpl.content.firstElementChild.cloneNode(true);

        if (nodeData) {
            const cm = nodeData.cpu_mem || {};
            nodeEl.querySelector('.cc-cpu-size').value = cm.mem_size  ?? 128;
            nodeEl.querySelector('.cc-cpu-bw').value   = cm.mem_bw    ?? 256;
            nodeEl.querySelector('.cc-cpu-lat').value  = cm.mem_latency ?? 0;
        }

        nodeEl.querySelector('.cc-rm-node').addEventListener('click', () => {
            nodeEl.remove();
            renumberCcNodes();
        });
        nodeEl.querySelector('.cc-add-inst').addEventListener('click', () => {
            addCcInstRow(nodeEl.querySelector('.cc-instances'));
        });

        container.appendChild(nodeEl);
        renumberCcNodes();

        const instContainer = nodeEl.querySelector('.cc-instances');
        const instances = (nodeData && nodeData.instances) || [null];
        for (const inst of instances) addCcInstRow(instContainer, inst);
    }

    function renumberCcNodes() {
        document.querySelectorAll('#cc-nodes .cc-node').forEach((n, i) => {
            const span = n.querySelector('.cc-node-idx');
            if (span) span.textContent = String(i + 1);
        });
    }

    function addCcInstRow(container, instData) {
        const tpl = document.getElementById('cc-inst-template');
        const instEl = tpl.content.firstElementChild.cloneNode(true);
        const hwSel    = instEl.querySelector('.cc-hw');
        const modelSel = instEl.querySelector('.cc-model');

        const hwNames = Object.keys(_hardware || {}).sort();
        for (const hw of hwNames) {
            const opt = document.createElement('option');
            opt.value = hw; opt.textContent = hw;
            hwSel.appendChild(opt);
        }

        function refreshCcModels() {
            const prev = modelSel.value;
            modelSel.innerHTML = '';
            for (const m of (_hardware[hwSel.value] || [])) {
                const opt = document.createElement('option');
                opt.value = m; opt.textContent = m;
                modelSel.appendChild(opt);
            }
            if (prev && [...modelSel.options].some(o => o.value === prev)) modelSel.value = prev;
        }
        hwSel.addEventListener('change', refreshCcModels);

        if (instData) {
            if (hwNames.includes(instData.hardware)) hwSel.value = instData.hardware;
            refreshCcModels();
            const modelVal = instData.model_name || instData.model || '';
            if (modelVal && [...modelSel.options].some(o => o.value === modelVal))
                modelSel.value = modelVal;
            instEl.querySelector('.cc-npu-num').value   = instData.npu_num   ?? 1;
            instEl.querySelector('.cc-npu-group').value = instData.npu_group ?? 1;
            const pdVal = instData.pd_type || '';
            if ([...instEl.querySelector('.cc-pd').options].some(o => o.value === pdVal))
                instEl.querySelector('.cc-pd').value = pdVal;
            const nm = instData.npu_mem || {};
            if (nm.mem_size    != null) instEl.querySelector('.cc-npu-size').value = nm.mem_size;
            if (nm.mem_bw      != null) instEl.querySelector('.cc-npu-bw').value   = nm.mem_bw;
            if (nm.mem_latency != null) instEl.querySelector('.cc-npu-lat').value  = nm.mem_latency;
        } else if (hwNames.length) {
            hwSel.value = hwNames[0];
            refreshCcModels();
        }

        instEl.querySelector('.cc-rm-inst').addEventListener('click', () => instEl.remove());
        container.appendChild(instEl);
    }

    function ccCollect() {
        return {
            name:         document.getElementById('cc-name').value,
            link_bw:      Number(document.getElementById('cc-link-bw').value)      || 112,
            link_latency: Number(document.getElementById('cc-link-latency').value) || 0,
            nodes: [...document.querySelectorAll('#cc-nodes .cc-node')].map(nodeEl => ({
                cpu_mem: {
                    mem_size:    Number(nodeEl.querySelector('.cc-cpu-size').value) || 128,
                    mem_bw:      Number(nodeEl.querySelector('.cc-cpu-bw').value)   || 256,
                    mem_latency: Number(nodeEl.querySelector('.cc-cpu-lat').value)  || 0,
                },
                instances: [...nodeEl.querySelectorAll('.cc-inst')].map(instEl => {
                    const memOv = {};
                    const s = instEl.querySelector('.cc-npu-size').value;
                    const b = instEl.querySelector('.cc-npu-bw').value;
                    const l = instEl.querySelector('.cc-npu-lat').value;
                    if (s !== '') memOv.mem_size    = Number(s);
                    if (b !== '') memOv.mem_bw      = Number(b);
                    if (l !== '') memOv.mem_latency = Number(l);
                    return {
                        hardware:  instEl.querySelector('.cc-hw').value,
                        model:     instEl.querySelector('.cc-model').value,
                        npu_num:   Number(instEl.querySelector('.cc-npu-num').value)   || 1,
                        npu_group: Number(instEl.querySelector('.cc-npu-group').value) || 1,
                        pd_type:   instEl.querySelector('.cc-pd').value || null,
                        npu_mem:   memOv,
                    };
                }),
            })),
        };
    }

    async function ccLoad(path) {
        console.log('[ccLoad] called with path=', path);
        try {
            const r = await fetch(`/api/cluster-configs/load?path=${encodeURIComponent(path)}`);
            const j = await r.json().catch(() => ({}));
            if (!r.ok) { toast('Load failed: ' + (j.detail || r.status), 'error'); return; }
            const cfg = j.config;
            document.getElementById('cc-name').value         = path.split('/').pop().replace(/\.json$/, '');
            document.getElementById('cc-link-bw').value      = cfg.link_bw      ?? 112;
            document.getElementById('cc-link-latency').value = cfg.link_latency ?? 0;
            document.getElementById('cc-nodes').innerHTML    = '';
            for (const node of cfg.nodes || []) addCcNodeRow(node);
            // Capture node-level `power` block for the sweep runner. The cc-*
            // form has no power inputs yet, so this is the only way an existing
            // power profile gets carried through enumerate → run.
            const firstPower = (cfg.nodes || []).map(n => n && n.power).find(p => p);
            _loadedPowerTemplate = firstPower || null;
            setCcStatus(firstPower
                ? `Loaded (power modeling enabled from ${path})`
                : '');
            syncScenarioFromConfig(cfg);
        } catch (e) {
            toast('Load error: ' + e.message, 'error');
        }
    }

    async function ccSave() {
        const payload = ccCollect();
        if (!payload.name) { setCcStatus('❌ Filename is required', 'error'); return; }
        if (!payload.nodes.length) { setCcStatus('❌ Add at least one node', 'error'); return; }
        const btn = document.getElementById('cc-save-btn');
        const orig = btn.textContent;
        btn.disabled = true;
        btn.innerHTML = '<span class="spinner"></span>Saving…';
        try {
            const r = await fetch('/api/cluster-configs', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify(payload),
            });
            const j = await r.json().catch(() => ({}));
            if (!r.ok) {
                const msg = (j.detail && j.detail.errors)
                    ? j.detail.errors.join('\n')
                    : (j.detail || 'save failed');
                setCcStatus('❌ ' + msg, 'error');
                return;
            }
            setCcStatus('✅ Saved to ' + j.path, 'ok');
            await populateCcLoadDropdown();
            // Reflect the just-saved config into the active scenario form so
            // section 2 picks up the new hardware/model/npu_count/pd_role.
            syncScenarioFromConfig(payload);
        } catch (e) {
            setCcStatus('❌ ' + e.message, 'error');
        } finally {
            btn.disabled = false;
            btn.textContent = orig;
        }
    }

    function setCcStatus(msg, kind) {
        const el = document.getElementById('cc-status');
        el.textContent = msg;
        el.className = 'cc-status' + (kind ? ' cc-status-' + kind : '');
    }

    // -------------------------------------------------------------------
    // Progress page
    // -------------------------------------------------------------------
    let _progressClock = null;
    let _progressStartMs = null;
    let _progressFrozenMs = null;  // set when sweep terminates → freeze elapsed

    LLMSS.initProgressPage = function (sweepId, status) {
        const cancelBtn = document.getElementById('btn-cancel');
        cancelBtn.addEventListener('click', async () => {
            cancelBtn.disabled = true;
            try {
                await fetch(`/api/sweeps/${sweepId}/cancel`, {method: 'POST'});
                toast('Cancellation requested', 'warn');
            } catch (e) {
                toast('Cancel failed: ' + e.message, 'error');
            }
        });

        // Pre-render existing state.
        if (status && status.configs) {
            for (const [label, entry] of Object.entries(status.configs)) {
                applyRowUpdate(label, entry);
            }
            updateProgressBar();
            if (['done', 'failed', 'cancelled'].includes(status.state)) {
                document.getElementById('btn-results').classList.remove('disabled');
            }
        }

        // Initialize the elapsed-time clock from the server timestamps.
        if (status && status.created_at) {
            _progressStartMs = Date.parse(status.created_at);
            if (status.finished_at) {
                _progressFrozenMs = Date.parse(status.finished_at) - _progressStartMs;
            } else if (['done', 'failed', 'cancelled'].includes(status.state)) {
                // Legacy sweep (predates finished_at). Approximate the wall-
                // clock duration as max config.elapsed_s — configs run in
                // parallel so the slowest one bounds the total.
                let maxElapsedSec = 0;
                for (const entry of Object.values(status.configs || {})) {
                    if (typeof entry.elapsed_s === 'number') {
                        maxElapsedSec = Math.max(maxElapsedSec, entry.elapsed_s);
                    }
                }
                _progressFrozenMs = maxElapsedSec * 1000;
            }
            updateProgressBar();
            // Tick only if not already frozen.
            if (_progressFrozenMs == null && _progressClock == null) {
                _progressClock = setInterval(updateProgressBar, 1000);
            }
        }

        const es = new EventSource(`/api/sweeps/${sweepId}/events`);
        es.onmessage = (e) => {
            try {
                const event = JSON.parse(e.data);
                if (event.type === 'heartbeat') return;
                if (event.sweep_state) {
                    if (['done', 'failed', 'cancelled'].includes(event.sweep_state)) {
                        document.getElementById('btn-results').classList.remove('disabled');
                        // Freeze elapsed using the server-provided finished_at.
                        if (event.finished_at && _progressStartMs != null) {
                            _progressFrozenMs = Date.parse(event.finished_at) - _progressStartMs;
                        } else if (_progressStartMs != null) {
                            _progressFrozenMs = Date.now() - _progressStartMs;
                        }
                        if (_progressClock != null) {
                            clearInterval(_progressClock);
                            _progressClock = null;
                        }
                        updateProgressBar();
                        toast(`Sweep ${event.sweep_state}`,
                            event.sweep_state === 'done' ? 'success' : 'warn');
                        es.close();
                    }
                    return;
                }
                if (event.label) {
                    applyRowUpdate(event.label, event);
                    updateProgressBar();
                }
            } catch (err) {
                console.error('parse SSE error', err);
            }
        };
        es.addEventListener('snapshot', (e) => {
            try {
                const snap = JSON.parse(e.data);
                if (snap && snap.configs) {
                    for (const [label, entry] of Object.entries(snap.configs)) {
                        applyRowUpdate(label, entry);
                    }
                    updateProgressBar();
                    if (['done', 'failed', 'cancelled'].includes(snap.state)) {
                        document.getElementById('btn-results').classList.remove('disabled');
                    }
                }
            } catch (err) { /* ignore */ }
        });
        es.onerror = () => {
            // Browser auto-reconnects; just log.
            console.warn('SSE error');
        };
    };

    function applyRowUpdate(label, update) {
        const row = document.querySelector(`#run-table tr[data-label="${cssEscape(label)}"]`);
        if (!row) return;
        if (update.state) {
            const badge = row.querySelector('.badge');
            if (badge) {
                badge.className = `badge badge-${update.state}`;
                badge.textContent = update.state;
            }
        }
        if (update.elapsed_s != null) {
            row.querySelector('.cfg-elapsed').textContent =
                `${update.elapsed_s.toFixed(1)}s`;
        }
        if (update.last_log_line) {
            row.querySelector('.cfg-log').textContent = update.last_log_line;
        }
        if (update.error) {
            row.querySelector('.cfg-log').textContent = `error: ${update.error}`;
        }
    }

    function _formatElapsed(ms) {
        // Whole seconds. Switch to mm:ss past 60s for readability.
        const total = Math.max(0, Math.floor(ms / 1000));
        if (total < 60) return `${total} sec`;
        const m = Math.floor(total / 60);
        const s = total % 60;
        return `${m}m ${s}s`;
    }

    function updateProgressBar() {
        const rows = document.querySelectorAll('#run-table tbody tr');
        let done = 0;
        rows.forEach(r => {
            const b = r.querySelector('.badge');
            if (b && ['badge-done', 'badge-failed', 'badge-cancelled'].some(c => b.classList.contains(c))) {
                done += 1;
            }
        });
        const total = rows.length;
        let summary = `Progress: ${done}/${total} done`;
        if (_progressStartMs != null) {
            const ms = _progressFrozenMs != null
                ? _progressFrozenMs
                : (Date.now() - _progressStartMs);
            summary += ` (${_formatElapsed(ms)} elapsed)`;
        }
        document.getElementById('progress-summary').textContent = summary;
        const pct = total ? Math.round((done / total) * 100) : 0;
        document.getElementById('progress-fill').style.width = `${pct}%`;
    }

    function cssEscape(s) {
        if (window.CSS && CSS.escape) return CSS.escape(s);
        return String(s).replace(/[^a-zA-Z0-9_-]/g, c => '\\' + c);
    }

    // -------------------------------------------------------------------
    // Results page
    // -------------------------------------------------------------------
    LLMSS.initResultsPage = function (sweepId, plots) {
        const layoutTweak = {
            margin: {l: 50, r: 30, t: 50, b: 60},
            paper_bgcolor: 'rgba(0,0,0,0)',
        };
        const config = {responsive: true, displayModeBar: false};

        const drawPlot = (divId, key) => {
            const el = document.getElementById(divId);
            if (!el) return;
            const json = plots[key];
            if (!json) {
                el.innerHTML = '<p class="hint" style="padding:16px">No data</p>';
                return;
            }
            const fig = JSON.parse(json);
            const layout = Object.assign({}, fig.layout || {}, layoutTweak);
            Plotly.newPlot(el, fig.data || [], layout, config);
        };

        drawPlot('plot-total_token_tp', 'total_token_tp');
        drawPlot('plot-mean_ttft_ms',   'mean_ttft_ms');
        drawPlot('plot-mean_tpot_ms',   'mean_tpot_ms');
        drawPlot('plot-mean_itl_ms',    'mean_itl_ms');
        drawPlot('plot-pareto',         'pareto');
        drawPlot('plot-line_tp',        'line_tp');
        drawPlot('plot-line_pp',        'line_pp');
        drawPlot('plot-line_dp',        'line_dp');
        drawPlot('plot-cdf_ttft',       'cdf_ttft');
        drawPlot('plot-cdf_itl',        'cdf_itl');

        // Pareto x-metric dropdown
        const paretoSel = document.getElementById('pareto-x-metric');
        if (paretoSel) {
            paretoSel.addEventListener('change', async () => {
                try {
                    const r = await fetch(
                        `/api/sweeps/${sweepId}/pareto?x_metric=${encodeURIComponent(paretoSel.value)}`
                    );
                    if (!r.ok) throw new Error(`HTTP ${r.status}`);
                    const data = await r.json();
                    const fig = data.figure;
                    const layout = Object.assign({}, fig.layout || {}, layoutTweak);
                    Plotly.react('plot-pareto', fig.data || [], layout, config);
                } catch (e) {
                    toast('Pareto reload failed: ' + e.message, 'error');
                }
            });
        }
    };

})();
