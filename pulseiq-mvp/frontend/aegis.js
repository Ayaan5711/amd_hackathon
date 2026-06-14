/* Aegis - Governance/Audit pack frontend.
 * Flow: upload or demo -> investigate -> SSE progress -> dashboard/report/chat.
 */

let currentRunId = null;
let eventSource = null;
let chatProcessing = false;
let reportSections = {};

const STEP_ORDER = ['triage', 'orchestrator', 'specialist_dispatch', 'risk_scoring', 'dashboard', 'report'];

const CATEGORY_LABELS = {
    pii: 'PII Exposure',
    security: 'Security / Injection',
    compliance: 'Compliance',
    hallucination: 'Hallucination',
};

const SEVERITY_ORDER = ['critical', 'high', 'medium', 'low'];

const REPORT_LABELS = {
    executive_summary: 'Executive Summary',
    detailed_findings: 'Detailed Findings',
    remediation_plan: 'Remediation Plan',
    incident_notifications: 'Incident Notifications',
    monitoring_recommendations: 'Monitoring Recommendations',
};

const REPORT_ORDER = [
    'executive_summary',
    'detailed_findings',
    'remediation_plan',
    'incident_notifications',
    'monitoring_recommendations',
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
    if (!file.name.endsWith('.csv') && !file.name.endsWith('.json')) {
        showUploadError('Please upload a CSV or JSON log batch');
        return;
    }
    hideUploadError();
    showLoading('Uploading log batch...');

    try {
        const formData = new FormData();
        formData.append('file', file);
        const response = await fetch('api/governance/upload', { method: 'POST', body: formData });
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
        const response = await fetch('api/governance/demo', { method: 'POST' });
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
        const response = await fetch(`api/governance/investigate/${uploadData.session_id}`, { method: 'POST' });
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to start investigation');
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
    eventSource = new EventSource(`api/governance/stream/${runId}`);

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
            const response = await fetch(`api/governance/status/${runId}`);
            if (!response.ok) continue;
            const status = await response.json();
            (status.progress || []).forEach(applyProgressEvent);
            if (status.status === 'complete') {
                markAllStepsDone();
                finishInvestigation(runId);
                return;
            }
            if (status.status === 'error') {
                showProgressError(status.error || 'Investigation failed.');
                return;
            }
        } catch (_) {
            // Network hiccup - keep polling.
        }
    }
    showProgressError('Investigation failed.');
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
            fetchJSON(`api/governance/dashboard/${runId}`),
            fetchJSON(`api/governance/report/${runId}`),
            fetchJSON(`api/governance/metrics/${runId}`),
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
        {
            label: 'Avg Latency',
            value: metrics.total_calls ? `${Math.round(metrics.total_latency_ms / metrics.total_calls)}ms` : 'N/A',
        },
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

    const categoryBars = Object.entries(dashboard.findings_by_category || {})
        .map(([cat, stats]) => {
            const pct = stats.total ? (stats.flagged / stats.total) * 100 : 0;
            return `
            <div class="bar-row category-${cat}">
                <span class="bar-label">${CATEGORY_LABELS[cat] || cat}</span>
                <div class="bar-track"><div class="bar-fill" style="width:${pct}%"></div></div>
                <span class="bar-count">${stats.flagged}/${stats.total}</span>
            </div>`;
        })
        .join('');

    const topFindings = dashboard.top_findings || [];
    const topFindingsRows = topFindings
        .map(
            (f) => `
        <tr>
            <td><button class="log-link" data-log-id="${f.log_id}">${f.log_id}</button></td>
            <td><span class="severity-badge badge-${f.severity}">${f.severity}</span></td>
            <td>${f.score}</td>
            <td>${(f.contributors || []).map((c) => `<span class="contributor-chip">${c}</span>`).join('')}</td>
        </tr>`
        )
        .join('');

    document.getElementById('tab-dashboard').innerHTML = `
        <div class="dashboard-grid">
            <div class="dashboard-card overall-score">
                <h3>Overall Risk Score</h3>
                <div class="score-value">${dashboard.overall_risk_score}</div>
                <div class="score-label">${dashboard.total_flagged} of ${dashboard.total_entries} entries flagged</div>
            </div>
            <div class="dashboard-card">
                <h3>Risk Distribution</h3>
                ${riskBars}
            </div>
            <div class="dashboard-card">
                <h3>Findings by Category</h3>
                ${categoryBars}
            </div>
        </div>
        <div class="dashboard-card">
            <h3>Top Findings (High &amp; Critical)</h3>
            ${
                topFindingsRows
                    ? `<table class="findings-table">
                <thead><tr><th>Log ID</th><th>Severity</th><th>Score</th><th>Contributors</th></tr></thead>
                <tbody>${topFindingsRows}</tbody>
            </table>`
                    : '<p class="empty-note">No high or critical severity entries in this run.</p>'
            }
        </div>
    `;

    document.querySelectorAll('#tab-dashboard .log-link').forEach((btn) => {
        btn.addEventListener('click', () => {
            switchTab('chat');
            chatInput.value = `Why was ${btn.dataset.logId} flagged?`;
            sendChatMessage();
        });
    });
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

// Minimal markdown -> HTML: headings, bold, inline code, bullet lists, paragraphs.
function renderMarkdown(text) {
    const lines = text.split('\n');
    let html = '';
    let inList = false;

    for (const rawLine of lines) {
        const line = rawLine.trim();
        if (!line) {
            if (inList) {
                html += '</ul>';
                inList = false;
            }
            continue;
        }

        const heading = line.match(/^(#{1,3})\s+(.*)$/);
        if (heading) {
            if (inList) {
                html += '</ul>';
                inList = false;
            }
            const level = heading[1].length + 1; // # -> h2, ## -> h3, ### -> h4
            html += `<h${level}>${inlineMarkdown(heading[2])}</h${level}>`;
            continue;
        }

        const bullet = line.match(/^[-*]\s+(.*)$/);
        if (bullet) {
            if (!inList) {
                html += '<ul>';
                inList = true;
            }
            html += `<li>${inlineMarkdown(bullet[1])}</li>`;
            continue;
        }

        if (inList) {
            html += '</ul>';
            inList = false;
        }
        html += `<p>${inlineMarkdown(line)}</p>`;
    }
    if (inList) html += '</ul>';
    return html;
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
            <h2 class="welcome-title">Talk to the results</h2>
            <p class="welcome-text">Ask about specific findings, risk distribution, or efficiency for this run.</p>
        </div>
        <div class="suggestion-chips">
            <button class="chip" data-query="What's the overall risk distribution?">
                <span class="chip-icon"></span>
                Risk distribution
            </button>
            <button class="chip" data-query="Which category has the most findings?">
                <span class="chip-icon"></span>
                Compare categories
            </button>
            <button class="chip" data-query="What PII issues were found?">
                <span class="chip-icon"></span>
                PII findings
            </button>
            <button class="chip" data-query="How efficient was this run compared to a naive approach?">
                <span class="chip-icon"></span>
                Efficiency
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
    const thinkingEl = showThinkingMessage();

    try {
        const finalData = await streamChatTurn(
            `api/governance/chat_stream/${currentRunId}`,
            message,
            (delta) => updateThinkingMessage(thinkingEl, delta)
        );
        removeThinkingMessage(thinkingEl);
        addChatMessage('assistant', finalData.narrative, {
            followUps: finalData.follow_up_suggestions,
            toolCalls: finalData.tool_calls,
            evidence: finalData.evidence,
            thinking: thinkingEl.dataset.thinking,
        });
    } catch (error) {
        removeThinkingMessage(thinkingEl);
        addChatMessage('assistant', `Sorry, I encountered an error: ${error.message}`);
    } finally {
        chatProcessing = false;
        chatSendBtn.disabled = false;
        chatInput.focus();
    }
}

// Reads an SSE chat_stream response (POST body, so EventSource can't be used):
// each `thinking` event's delta is forwarded to `onThinking` and accumulated on
// `thinkingEl.dataset.thinking`; resolves with the `complete` event's data.
async function streamChatTurn(url, message, onThinking) {
    const response = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message }),
    });
    if (!response.ok) {
        const error = await response.json().catch(() => ({}));
        throw new Error(error.detail || `Request to ${url} failed`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let finalData = null;

    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        let sepIndex;
        while ((sepIndex = buffer.indexOf('\n\n')) !== -1) {
            const rawEvent = buffer.slice(0, sepIndex);
            buffer = buffer.slice(sepIndex + 2);

            let eventType = 'message';
            let dataStr = '';
            for (const line of rawEvent.split('\n')) {
                if (line.startsWith('event: ')) eventType = line.slice(7).trim();
                else if (line.startsWith('data: ')) dataStr += line.slice(6);
            }
            if (!dataStr) continue;
            const data = JSON.parse(dataStr);

            if (eventType === 'thinking') {
                onThinking(data.delta);
            } else if (eventType === 'complete') {
                finalData = data;
            } else if (eventType === 'error') {
                throw new Error(data.error || 'Chat stream error');
            }
        }
    }

    if (!finalData) throw new Error('Chat stream ended without a response');
    return finalData;
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

    if (options.thinking) {
        const thinkingDiv = document.createElement('details');
        thinkingDiv.className = 'thinking-trace';
        thinkingDiv.innerHTML = `<summary> Agent's reasoning</summary><div class="thinking-text">${escapeHtml(
            options.thinking
        )}</div>`;
        contentDiv.appendChild(thinkingDiv);
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

// Live "agent is thinking" bubble for chat_stream: shows Qwen3's <think> trace
// streaming in word-by-word while the synthesis call is in flight.
function showThinkingMessage() {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'message assistant thinking-message';
    messageDiv.dataset.thinking = '';
    messageDiv.innerHTML = `
        <div class="message-avatar">AI</div>
        <div class="message-content">
            <div class="thinking-trace-live">
                <span class="thinking-label"> Thinking...</span>
                <span class="thinking-text"></span>
            </div>
        </div>`;
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
    return messageDiv;
}

function updateThinkingMessage(el, delta) {
    if (!el || !delta) return;
    el.dataset.thinking += delta;
    const textEl = el.querySelector('.thinking-text');
    if (textEl) textEl.textContent = el.dataset.thinking;
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function removeThinkingMessage(el) {
    if (el) el.remove();
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
