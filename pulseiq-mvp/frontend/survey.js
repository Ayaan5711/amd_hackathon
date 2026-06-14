/* Aegis Survey Analytics - frontend.
 * Flow: upload or demo -> investigate -> SSE progress -> dashboard/report/chat.
 */

let currentRunId = null;
let eventSource = null;
let chatProcessing = false;
let reportSections = {};

const API_BASE = 'api/survey';

const STEP_ORDER = ['triage', 'orchestrator', 'specialist_dispatch', 'risk_scoring', 'dashboard', 'report'];

const SEVERITY_ORDER = ['critical', 'high', 'medium', 'low'];

const CATEGORY_LABELS_FALLBACK = {
    pii: 'PII in open-text responses',
    security: 'Security signals',
    compliance: 'Outlier responses flagged for review',
    hallucination: 'Hallucination signals',
};

const REPORT_LABELS = {
    executive_summary: 'Executive Summary',
    segment_analysis: 'Segment Analysis',
    trends_analysis: 'Trends',
    themes_and_sentiment: 'Themes & Sentiment',
    anomalies_and_quality: 'Anomalies & Quality',
    recommendations: 'Recommendations',
};

const REPORT_ORDER = [
    'executive_summary',
    'segment_analysis',
    'trends_analysis',
    'themes_and_sentiment',
    'anomalies_and_quality',
    'recommendations',
];

// DOM references
const uploadView = document.getElementById('upload-view');
const progressView = document.getElementById('progress-view');
const resultsView = document.getElementById('results-view');

const uploadZone = document.getElementById('upload-zone');
const fileInput = document.getElementById('file-input');
const demoBtn = document.getElementById('demo-btn');
const uploadError = document.getElementById('upload-error');
const errorText = document.getElementById('error-text');

const progressFilename = document.getElementById('progress-filename');
const progressRowcount = document.getElementById('progress-rowcount');
const progressError = document.getElementById('progress-error');

const metricsStrip = document.getElementById('metrics-strip');
const newRunBtn = document.getElementById('new-run-btn');

const loadingOverlay = document.getElementById('loading-overlay');
const loadingText = document.getElementById('loading-text');

const chatMessages = document.getElementById('chat-messages');
const chatInput = document.getElementById('chat-input');
const chatSendBtn = document.getElementById('chat-send-btn');

function init() {
    setupUpload();
    setupTabs();
    setupChat();
    newRunBtn.addEventListener('click', resetToUpload);
}

// ---------------------------------------------------------------------------
// Upload / Demo
// ---------------------------------------------------------------------------
function setupUpload() {
    uploadZone.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', (e) => {
        const file = e.target.files[0];
        if (file) handleFile(file);
    });
    demoBtn.addEventListener('click', loadDemo);

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach((name) => {
        uploadZone.addEventListener(name, (e) => {
            e.preventDefault();
            e.stopPropagation();
        });
    });
    ['dragenter', 'dragover'].forEach((name) =>
        uploadZone.addEventListener(name, () => uploadZone.classList.add('dragover'))
    );
    ['dragleave', 'drop'].forEach((name) =>
        uploadZone.addEventListener(name, () => uploadZone.classList.remove('dragover'))
    );
    uploadZone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files.length > 0) handleFile(files[0]);
    });
}

async function handleFile(file) {
    if (!file.name.endsWith('.csv')) {
        showUploadError('Please upload a CSV file');
        return;
    }
    hideUploadError();
    showLoading('Uploading survey data...');

    try {
        const formData = new FormData();
        formData.append('file', file);
        const response = await fetch(`${API_BASE}/upload`, { method: 'POST', body: formData });
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Upload failed');
        }
        const data = await response.json();
        await startInvestigation(data);
    } catch (error) {
        showUploadError(error.message);
        hideLoading();
    }
}

async function loadDemo() {
    hideUploadError();
    showLoading('Loading demo dataset...');

    try {
        const response = await fetch(`${API_BASE}/demo`, { method: 'POST' });
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to load demo dataset');
        }
        const data = await response.json();
        await startInvestigation(data);
    } catch (error) {
        showUploadError(error.message);
        hideLoading();
    }
}

async function startInvestigation(uploadData) {
    try {
        const response = await fetch(`${API_BASE}/investigate/${uploadData.session_id}`, { method: 'POST' });
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to start analysis');
        }
        const data = await response.json();
        currentRunId = data.run_id;

        showProgressView(uploadData);
        streamProgress(currentRunId);
    } catch (error) {
        showUploadError(error.message);
    } finally {
        hideLoading();
    }
}

// ---------------------------------------------------------------------------
// Progress
// ---------------------------------------------------------------------------
function showProgressView(uploadData) {
    uploadView.style.display = 'none';
    progressView.style.display = 'flex';
    resultsView.style.display = 'none';
    newRunBtn.style.display = 'none';
    progressError.style.display = 'none';

    progressFilename.textContent = uploadData.filename;
    progressRowcount.textContent = uploadData.row_count.toLocaleString();

    document.querySelectorAll('#progress-steps .step').forEach((step) => {
        step.classList.remove('active', 'done');
        const msg = step.querySelector('.step-message');
        if (msg) msg.textContent = msg.dataset.default;
    });
}

function streamProgress(runId) {
    if (eventSource) eventSource.close();
    eventSource = new EventSource(`${API_BASE}/stream/${runId}`);

    eventSource.addEventListener('progress', (e) => {
        const event = JSON.parse(e.data);
        applyProgressEvent(event);
    });

    eventSource.addEventListener('complete', () => {
        eventSource.close();
        markAllStepsDone();
        finishInvestigation(runId);
    });

    eventSource.addEventListener('error', (e) => {
        let serverError = null;
        try {
            const parsed = JSON.parse(e.data);
            if (parsed && parsed.error) serverError = parsed.error;
        } catch (_) {
            // Native connection-level error event has no `.data`.
        }
        eventSource.close();
        if (serverError) {
            showProgressError(serverError);
            return;
        }
        // The stream dropped without a `complete`/`error` event (e.g. a proxy/tunnel
        // timeout during a long-running investigation). The background task keeps
        // running independently of this connection, so poll /status instead of
        // declaring failure immediately.
        pollInvestigationStatus(runId);
    });
}

async function pollInvestigationStatus(runId) {
    const POLL_INTERVAL_MS = 2000;
    const MAX_ATTEMPTS = 150; // ~5 minutes

    for (let attempt = 0; attempt < MAX_ATTEMPTS; attempt++) {
        await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS));
        try {
            const response = await fetch(`${API_BASE}/status/${runId}`);
            if (!response.ok) continue;
            const status = await response.json();
            (status.progress || []).forEach(applyProgressEvent);
            if (status.status === 'complete') {
                markAllStepsDone();
                finishInvestigation(runId);
                return;
            }
            if (status.status === 'error') {
                showProgressError(status.error || 'Analysis failed.');
                return;
            }
        } catch (_) {
            // Network hiccup - keep polling.
        }
    }
    showProgressError('Analysis failed.');
}

function applyProgressEvent(event) {
    const idx = STEP_ORDER.indexOf(event.step);
    if (idx === -1) return;

    STEP_ORDER.forEach((name, i) => {
        const el = document.querySelector(`#progress-steps .step[data-step="${name}"]`);
        if (!el) return;
        if (i < idx) {
            el.classList.add('done');
            el.classList.remove('active');
        } else if (i === idx) {
            el.classList.add('active');
            el.classList.remove('done');
        }
    });

    const step = document.querySelector(`#progress-steps .step[data-step="${event.step}"]`);
    const msg = step && step.querySelector('.step-message');
    if (msg && event.message) msg.textContent = event.message;
}

function markAllStepsDone() {
    document.querySelectorAll('#progress-steps .step').forEach((step) => {
        step.classList.remove('active');
        step.classList.add('done');
    });
}

function showProgressError(message) {
    progressError.textContent = message;
    progressError.style.display = 'flex';
}

// ---------------------------------------------------------------------------
// Results
// ---------------------------------------------------------------------------
async function finishInvestigation(runId) {
    showLoading('Loading results...');
    try {
        const [dashboard, report, metrics] = await Promise.all([
            fetchJSON(`${API_BASE}/dashboard/${runId}`),
            fetchJSON(`${API_BASE}/report/${runId}`),
            fetchJSON(`${API_BASE}/metrics/${runId}`),
        ]);

        renderMetricsStrip(metrics);
        renderDashboard(dashboard);
        renderReport(report);
        resetChat();

        progressView.style.display = 'none';
        resultsView.style.display = 'flex';
        newRunBtn.style.display = 'flex';
        switchTab('dashboard');
    } catch (error) {
        showProgressError(error.message);
    } finally {
        hideLoading();
    }
}

async function fetchJSON(url, options) {
    const response = await fetch(url, options);
    if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        throw new Error(error.detail || `Request to ${url} failed`);
    }
    return response.json();
}

function gpuMetricCard(gpu) {
    if (!gpu.gpu_available) {
        return { label: 'GPU', value: 'CPU (mock)' };
    }
    const value = gpu.gpu_utilization_pct != null ? `${Math.round(gpu.gpu_utilization_pct)}%` : 'AMD ROCm';
    let label = gpu.gpu_name ? `${gpu.gpu_name} util.` : 'GPU (AMD ROCm)';
    if (gpu.vram_used_gb != null && gpu.vram_total_gb != null) {
        label += ` · ${gpu.vram_used_gb}/${gpu.vram_total_gb} GB VRAM`;
    }
    return { label, value };
}

function renderMetricsStrip(metrics) {
    const eff = metrics.efficiency || {};
    const gpu = metrics.gpu || {};
    const cards = [
        { label: 'LLM Calls', value: metrics.total_calls },
        { label: 'Tokens', value: (metrics.total_tokens || 0).toLocaleString() },
        { label: 'Wall Clock', value: `${metrics.wall_clock_seconds}s` },
        { label: 'Calls Saved', value: eff.reduction_pct != null ? `${eff.reduction_pct}%` : 'N/A' },
        gpuMetricCard(gpu),
    ];
    metricsStrip.innerHTML = cards
        .map(
            (c) => `
        <div class="metric-card">
            <div class="metric-value">${c.value}</div>
            <div class="metric-label">${c.label}</div>
        </div>`
        )
        .join('');
}

function renderDashboard(dashboard) {
    const dist = dashboard.risk_distribution || {};
    const totalScored = Object.values(dist).reduce((a, b) => a + b, 0) || 1;

    const riskBars = SEVERITY_ORDER.map((sev) => {
        const count = dist[sev] || 0;
        const pct = (count / totalScored) * 100;
        return `
            <div class="bar-row severity-${sev}">
                <span class="bar-label">${sev}</span>
                <div class="bar-track"><div class="bar-fill" style="width:${pct}%"></div></div>
                <span class="bar-count">${count}</span>
            </div>`;
    }).join('');

    const categoryLabels = dashboard.category_labels || CATEGORY_LABELS_FALLBACK;
    const categoryBars = Object.entries(dashboard.findings_by_category || {})
        .filter(([, stats]) => stats.total > 0)
        .map(([cat, stats]) => {
            const pct = stats.total ? (stats.flagged / stats.total) * 100 : 0;
            return `
            <div class="bar-row category-${cat}">
                <span class="bar-label">${categoryLabels[cat] || cat}</span>
                <div class="bar-track"><div class="bar-fill" style="width:${pct}%"></div></div>
                <span class="bar-count">${stats.flagged}/${stats.total}</span>
            </div>`;
        })
        .join('');

    const crosstabCharts = dashboard.crosstabs ? renderCrosstabCharts(dashboard.crosstabs) : '';

    const topFindings = dashboard.top_findings || [];
    const topFindingsRows = topFindings
        .map(
            (f) => `
        <tr>
            <td><button class="log-link" data-log-id="${f.log_id}">#${f.log_id}</button></td>
            <td><span class="severity-badge badge-${f.severity}">${f.severity}</span></td>
            <td>${f.score}</td>
            <td>${(f.contributors || []).map((c) => `<span class="contributor-chip">${c}</span>`).join('')}</td>
        </tr>`
        )
        .join('');

    document.getElementById('tab-dashboard').innerHTML = `
        <div class="dashboard-grid">
            <div class="dashboard-card overall-score">
                <h3>Overall Review Score</h3>
                <div class="score-value">${dashboard.overall_risk_score}</div>
                <div class="score-label">${dashboard.total_flagged} of ${dashboard.total_entries} responses flagged for review</div>
            </div>
            <div class="dashboard-card">
                <h3>Review Priority</h3>
                ${riskBars}
            </div>
            ${categoryBars ? `<div class="dashboard-card"><h3>Findings by Category</h3>${categoryBars}</div>` : ''}
        </div>
        <div class="dashboard-grid">
            ${renderMetricSummaryChart(dashboard.metric_summary)}
            ${renderSegmentBreakdownChart(dashboard.segment_breakdown)}
            ${renderDemographicSummaryChart(dashboard.demographic_summary)}
            ${renderResponseSummaryChart(dashboard.response_summary)}
        </div>
        ${crosstabCharts ? `<div class="dashboard-grid">${crosstabCharts}</div>` : ''}
        <div class="dashboard-card">
            <h3>Responses Flagged for Review (High &amp; Critical)</h3>
            ${
                topFindingsRows
                    ? `<table class="findings-table">
                <thead><tr><th>Response</th><th>Priority</th><th>Score</th><th>Why</th></tr></thead>
                <tbody>${topFindingsRows}</tbody>
            </table>`
                    : '<p class="empty-note">No high-priority responses in this run.</p>'
            }
        </div>
    `;

    document.querySelectorAll('#tab-dashboard .log-link').forEach((btn) => {
        btn.addEventListener('click', () => {
            switchTab('chat');
            chatInput.value = `Why was response ${btn.dataset.logId} flagged?`;
            sendChatMessage();
        });
    });
}

function renderMetricSummaryChart(metricSummary) {
    if (!metricSummary || metricSummary.length === 0) return '';
    const rows = metricSummary
        .map((m) => {
            const pct = m.max > 0 ? Math.max(0, Math.min(100, (m.mean / m.max) * 100)) : 0;
            return `
            <div class="bar-row metric-row">
                <span class="bar-label">${m.column}</span>
                <div class="bar-track"><div class="bar-fill metric-fill" style="width:${pct}%"></div></div>
                <span class="bar-count">${m.mean}<span class="bar-range"> / ${m.max}</span></span>
            </div>`;
        })
        .join('');
    return `
        <div class="dashboard-card">
            <h3>Average Scores</h3>
            ${rows}
        </div>`;
}

function renderSegmentBreakdownChart(seg) {
    if (!seg || !seg.success || !seg.segments || seg.segments.length === 0) return '';
    const maxVal = Math.max(...seg.segments.map((s) => s.max), 1);
    const rows = seg.segments
        .map((s) => {
            const pct = Math.max(0, Math.min(100, (s.mean / maxVal) * 100));
            let cls = 'segment-row';
            if (s.segment === seg.best_segment) cls += ' segment-best';
            else if (s.segment === seg.worst_segment) cls += ' segment-worst';
            return `
            <div class="bar-row ${cls}">
                <span class="bar-label">${s.segment}</span>
                <div class="bar-track"><div class="bar-fill segment-fill" style="width:${pct}%"></div></div>
                <span class="bar-count">${s.mean}<span class="bar-range"> (n=${s.count})</span></span>
            </div>`;
        })
        .join('');
    return `
        <div class="dashboard-card">
            <h3>${seg.metric_column} by ${seg.segment_column}</h3>
            ${rows}
            <p class="empty-note">Best: <strong>${seg.best_segment}</strong> &middot; Worst: <strong>${seg.worst_segment}</strong> &middot; Gap: ${seg.gap}</p>
        </div>`;
}

// Demographic Profile: one donut chart + legend per demographic column, showing
// the FULL breakdown (every value), not just the dominant one.
function renderDemographicSummaryChart(demo) {
    if (!demo || !demo.success || !demo.profiles || demo.profiles.length === 0) return '';
    const blocks = demo.profiles
        .map((p) => {
            const stops = conicGradientStops(p.distribution);
            const legend = renderChartLegend(
                p.distribution.map((d, i) => ({ label: d.value, percent: d.percent, color: chartColor(i) }))
            );
            return `
            <div class="donut-block">
                <h4 class="chart-subtitle">${escapeHtml(p.column)}</h4>
                <div class="donut-row">
                    <div class="donut-chart" style="background: conic-gradient(${stops})">
                        <div class="donut-hole">
                            <span class="donut-value">${p.top_percent}%</span>
                            <span class="donut-caption">${escapeHtml(p.top_value)}</span>
                        </div>
                    </div>
                    ${legend}
                </div>
            </div>`;
        })
        .join('');
    return `
        <div class="dashboard-card">
            <h3>Demographic Profile</h3>
            ${blocks}
        </div>`;
}

// Response Distribution: one 100%-stacked bar per Likert-style question, showing
// the FULL option breakdown, plus a shared legend (options that recur across
// questions, e.g. the same 5-point Likert scale, get the same color).
function renderResponseSummaryChart(resp) {
    if (!resp || !resp.success || !resp.questions || resp.questions.length === 0) return '';

    const allOptions = [];
    resp.questions.forEach((q) => {
        q.options.forEach((opt) => {
            if (!allOptions.includes(opt)) allOptions.push(opt);
        });
    });
    const colorMap = buildColorMap(allOptions);

    const rows = resp.questions
        .map(
            (q) => `
            <div class="stacked-bar-row">
                <span class="bar-label">${escapeHtml(q.column)}</span>
                ${renderStackedBar(q.distribution, colorMap)}
                <span class="bar-count">${escapeHtml(q.dominant_value)} <span class="bar-range">(${q.dominant_percent}%)</span></span>
            </div>`
        )
        .join('');

    const legend = renderChartLegend(allOptions.map((opt) => ({ label: opt, color: colorMap[opt] })));

    return `
        <div class="dashboard-card">
            <h3>Response Distribution</h3>
            ${rows}
            ${legend}
        </div>`;
}

// Full Demographic Analysis: for each cross-tab (e.g. "Outlook_General by Gender"),
// one 100%-stacked bar per segment value, segmented by response option, plus a
// shared legend.
function renderCrosstabCharts(crosstabs) {
    if (!crosstabs || crosstabs.length === 0) return '';
    return crosstabs
        .map((ct) => {
            const colorMap = buildColorMap(ct.options);
            const rows = ct.segments
                .map(
                    (seg) => `
                <div class="stacked-bar-row">
                    <span class="bar-label">${escapeHtml(seg.segment)}</span>
                    ${renderStackedBar(seg.distribution, colorMap)}
                    <span class="bar-count">${escapeHtml(seg.dominant_value)} <span class="bar-range">(${seg.dominant_percent}%)</span></span>
                </div>`
                )
                .join('');
            const legend = renderChartLegend(ct.options.map((opt) => ({ label: opt, color: colorMap[opt] })));
            return `
            <div class="dashboard-card crosstab-card">
                <h3>${escapeHtml(ct.response_column)} by ${escapeHtml(ct.segment_column)}</h3>
                ${rows}
                ${legend}
            </div>`;
        })
        .join('');
}

function renderReport(report) {
    reportSections = report;
    const tabsEl = document.getElementById('report-tabs');
    const availableSections = REPORT_ORDER.filter((k) => report[k]);

    tabsEl.innerHTML = availableSections
        .map(
            (k, i) => `<button class="report-tab-btn${i === 0 ? ' active' : ''}" data-section="${k}">${REPORT_LABELS[k] || k}</button>`
        )
        .join('');

    tabsEl.querySelectorAll('.report-tab-btn').forEach((btn) => {
        btn.addEventListener('click', () => {
            tabsEl.querySelectorAll('.report-tab-btn').forEach((b) => b.classList.remove('active'));
            btn.classList.add('active');
            showReportSection(btn.dataset.section);
        });
    });

    if (availableSections.length > 0) showReportSection(availableSections[0]);
}

function showReportSection(key) {
    document.getElementById('report-content').innerHTML = renderMarkdown(reportSections[key] || '');
}

// Minimal markdown -> HTML: headings, bold, inline code, bullet/numbered lists, tables, paragraphs.
function renderMarkdown(text) {
    const lines = text.split('\n');
    let html = '';
    let listType = null; // 'ul' | 'ol' | null

    const closeList = () => {
        if (listType) {
            html += listType === 'ol' ? '</ol>' : '</ul>';
            listType = null;
        }
    };

    let i = 0;
    while (i < lines.length) {
        const line = lines[i].trim();

        if (!line) {
            closeList();
            i++;
            continue;
        }

        if (line.startsWith('|') && i + 1 < lines.length && isTableSeparator(lines[i + 1])) {
            closeList();
            i = renderTableInto(lines, i, (chunk) => {
                html += chunk;
            });
            continue;
        }

        const heading = line.match(/^(#{1,3})\s+(.*)$/);
        if (heading) {
            closeList();
            const level = heading[1].length + 1; // # -> h2, ## -> h3, ### -> h4
            html += `<h${level}>${inlineMarkdown(heading[2])}</h${level}>`;
            i++;
            continue;
        }

        const bullet = line.match(/^[-*]\s+(.*)$/);
        if (bullet) {
            if (listType !== 'ul') {
                closeList();
                html += '<ul>';
                listType = 'ul';
            }
            html += `<li>${inlineMarkdown(bullet[1])}</li>`;
            i++;
            continue;
        }

        const numbered = line.match(/^\d+\.\s+(.*)$/);
        if (numbered) {
            if (listType !== 'ol') {
                closeList();
                html += '<ol>';
                listType = 'ol';
            }
            html += `<li>${inlineMarkdown(numbered[1])}</li>`;
            i++;
            continue;
        }

        closeList();
        html += `<p>${inlineMarkdown(line)}</p>`;
        i++;
    }
    closeList();
    return html;
}

function isTableSeparator(line) {
    return /^\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)+\|?$/.test(line.trim());
}

function tableRowCells(line) {
    return line.trim().replace(/^\|/, '').replace(/\|$/, '').split('|').map((c) => c.trim());
}

function renderTableInto(lines, startIdx, emit) {
    emit('<table class="markdown-table"><thead><tr>');
    tableRowCells(lines[startIdx]).forEach((c) => emit(`<th>${inlineMarkdown(c)}</th>`));
    emit('</tr></thead><tbody>');

    let i = startIdx + 2; // skip header row + separator row
    while (i < lines.length && lines[i].trim().startsWith('|')) {
        emit('<tr>');
        tableRowCells(lines[i]).forEach((c) => emit(`<td>${inlineMarkdown(c)}</td>`));
        emit('</tr>');
        i++;
    }
    emit('</tbody></table>');
    return i;
}

function inlineMarkdown(text) {
    return text.replace(/`([^`]+)`/g, '<code>$1</code>').replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
}

// ---------------------------------------------------------------------------
// Tabs
// ---------------------------------------------------------------------------
function setupTabs() {
    document.querySelectorAll('.results-tabs .tab-btn').forEach((btn) => {
        btn.addEventListener('click', () => switchTab(btn.dataset.tab));
    });

    const printBtn = document.getElementById('print-report-btn');
    if (printBtn) {
        printBtn.addEventListener('click', () => {
            switchTab('report');
            window.print();
        });
    }
}

function switchTab(tabName) {
    document.querySelectorAll('.results-tabs .tab-btn').forEach((btn) => {
        btn.classList.toggle('active', btn.dataset.tab === tabName);
    });
    document.querySelectorAll('.tab-panel').forEach((panel) => {
        panel.style.display = panel.id === `tab-${tabName}` ? 'block' : 'none';
    });
}

// ---------------------------------------------------------------------------
// Chat
// ---------------------------------------------------------------------------
function setupChat() {
    chatSendBtn.addEventListener('click', sendChatMessage);
    chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendChatMessage();
        }
    });
    chatInput.addEventListener('input', () => {
        chatInput.style.height = 'auto';
        chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + 'px';
    });
}

function resetChat() {
    chatMessages.innerHTML = `
        <div class="welcome-card">
            <h2 class="welcome-title">Talk to your survey data</h2>
            <p class="welcome-text">Ask about segment comparisons, trends over time, themes in comments, outliers, or recommended actions.</p>
        </div>
        <div class="suggestion-chips">
            <button class="chip" data-query="Which department has the highest satisfaction?">
                <span class="chip-icon"></span>
                Compare departments
            </button>
            <button class="chip" data-query="Show me satisfaction trends by quarter">
                <span class="chip-icon"></span>
                View trends
            </button>
            <button class="chip" data-query="What are the main themes in the comments?">
                <span class="chip-icon"></span>
                Extract themes
            </button>
            <button class="chip" data-query="Are there any outliers in the data?">
                <span class="chip-icon"></span>
                Find anomalies
            </button>
            <button class="chip" data-query="What actions should we take?">
                <span class="chip-icon"></span>
                Get recommendations
            </button>
            <button class="chip" data-query="Give me an overview of flagged responses">
                <span class="chip-icon"></span>
                Flagged overview
            </button>
        </div>
    `;
    chatMessages.querySelectorAll('.chip').forEach((chip) => {
        chip.addEventListener('click', () => {
            chatInput.value = chip.dataset.query;
            sendChatMessage();
        });
    });
}

async function sendChatMessage() {
    const message = chatInput.value.trim();
    if (!message || chatProcessing || !currentRunId) return;

    addChatMessage('user', message);
    chatInput.value = '';
    chatInput.style.height = 'auto';
    chatProcessing = true;
    chatSendBtn.disabled = true;
    showTypingIndicator();

    try {
        const data = await fetchJSON(`${API_BASE}/chat/${currentRunId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message }),
        });
        removeTypingIndicator();
        addChatMessage('assistant', data.response, {
            followUps: data.follow_up_suggestions,
            toolCalls: data.tool_calls,
            evidence: data.evidence,
        });
    } catch (error) {
        removeTypingIndicator();
        addChatMessage('assistant', `Sorry, I encountered an error: ${error.message}`);
    } finally {
        chatProcessing = false;
        chatSendBtn.disabled = false;
        chatInput.focus();
    }
}

function addChatMessage(role, content, options = {}) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = role === 'user' ? 'You' : 'AI';

    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';

    if (options.toolCalls && options.toolCalls.length > 0) {
        const toolsDiv = document.createElement('div');
        toolsDiv.className = 'tool-badges';
        options.toolCalls.forEach((tool) => {
            const badge = document.createElement('span');
            badge.className = 'tool-badge';
            badge.textContent = formatToolName(tool.tool_name);
            toolsDiv.appendChild(badge);
        });
        contentDiv.appendChild(toolsDiv);
    }

    const textDiv = document.createElement('div');
    textDiv.innerHTML = escapeHtml(content).replace(/\n/g, '<br>');
    contentDiv.appendChild(textDiv);

    if (options.followUps && options.followUps.length > 0) {
        const followUpsDiv = document.createElement('div');
        followUpsDiv.className = 'follow-ups';
        options.followUps.forEach((suggestion) => {
            const btn = document.createElement('button');
            btn.className = 'follow-up-btn';
            btn.textContent = suggestion;
            btn.addEventListener('click', () => {
                chatInput.value = suggestion;
                sendChatMessage();
            });
            followUpsDiv.appendChild(btn);
        });
        contentDiv.appendChild(followUpsDiv);
    }

    if (options.evidence && Object.keys(options.evidence).length > 0) {
        const chartHtml = renderEvidenceChart(options.evidence.chart_data || []);
        const evidenceToShow = { ...options.evidence };
        if (chartHtml) {
            const chartDiv = document.createElement('div');
            chartDiv.className = 'chat-chart';
            chartDiv.innerHTML = chartHtml;
            contentDiv.appendChild(chartDiv);
            delete evidenceToShow.chart_data;
        }

        if (Object.keys(evidenceToShow).length > 0) {
            const evidenceDiv = document.createElement('details');
            evidenceDiv.className = 'evidence-panel';
            evidenceDiv.innerHTML = `<summary>View evidence</summary><pre>${escapeHtml(
                JSON.stringify(evidenceToShow, null, 2)
            )}</pre>`;
            contentDiv.appendChild(evidenceDiv);
        }
    }

    messageDiv.appendChild(avatar);
    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatToolName(name) {
    return name
        .split('_')
        .map((w) => w.charAt(0).toUpperCase() + w.slice(1))
        .join(' ');
}

function showTypingIndicator() {
    const indicator = document.createElement('div');
    indicator.className = 'message assistant';
    indicator.id = 'chat-typing-indicator';
    indicator.innerHTML = `
        <div class="message-avatar">AI</div>
        <div class="message-content">
            <div class="typing-indicator"><span></span><span></span><span></span></div>
        </div>`;
    chatMessages.appendChild(indicator);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function removeTypingIndicator() {
    const indicator = document.getElementById('chat-typing-indicator');
    if (indicator) indicator.remove();
}

// ---------------------------------------------------------------------------
// View / overlay helpers
// ---------------------------------------------------------------------------
function resetToUpload() {
    if (eventSource) {
        eventSource.close();
        eventSource = null;
    }
    currentRunId = null;

    uploadView.style.display = 'flex';
    progressView.style.display = 'none';
    resultsView.style.display = 'none';
    newRunBtn.style.display = 'none';
    fileInput.value = '';
}

function showLoading(text = 'Processing...') {
    loadingText.textContent = text;
    loadingOverlay.style.display = 'flex';
}

function hideLoading() {
    loadingOverlay.style.display = 'none';
}

function showUploadError(message) {
    errorText.textContent = message;
    uploadError.style.display = 'flex';
}

function hideUploadError() {
    uploadError.style.display = 'none';
}

document.addEventListener('DOMContentLoaded', init);
