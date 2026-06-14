/* Aegis - shared chart-rendering helpers.
 * Loaded before aegis.js/survey.js: provides the color/legend/stacked-bar
 * primitives their dashboard renderers depend on, plus generic
 * renderBarChart/renderDonutChart/renderStackedBarChart/renderEvidenceChart
 * used to turn chat "evidence.chart_data" into visuals.
 */

// Shared color palette for donut / stacked-bar / bar charts. Cycled by
// option/value index so the same option (e.g. "More than current") gets the
// same color across charts.
const CHART_COLORS = ['#60a5fa', '#c084fc', '#34d399', '#fbbf24', '#f87171', '#22d3ee', '#f472b6', '#a3e635'];

function chartColor(index) {
    return CHART_COLORS[index % CHART_COLORS.length];
}

// Builds `conic-gradient()` stops from a `[{value, percent}, ...]` distribution,
// clamping the final stop to 100% to avoid gaps from rounding.
function conicGradientStops(distribution) {
    let acc = 0;
    return distribution
        .map((d, i) => {
            const start = acc;
            acc += d.percent;
            const end = i === distribution.length - 1 ? 100 : acc;
            return `${chartColor(i)} ${start}% ${end}%`;
        })
        .join(', ');
}

// Builds a `value -> color` map for a list of option strings, preserving order.
function buildColorMap(options) {
    const map = {};
    options.forEach((opt, i) => {
        map[opt] = chartColor(i);
    });
    return map;
}

function renderChartLegend(items) {
    return `<ul class="chart-legend">${items
        .map(
            (item) => `
        <li class="legend-item">
            <span class="legend-swatch" style="background:${item.color}"></span>
            ${escapeHtml(item.label)}${item.percent != null ? ` <span class="legend-pct">${item.percent}%</span>` : ''}
        </li>`
        )
        .join('')}</ul>`;
}

function renderStackedBar(distribution, colorMap) {
    return `<div class="stacked-bar">${distribution
        .map((d) => `<div class="stacked-segment" style="width:${d.percent}%; background:${colorMap[d.value] || '#888'}" title="${escapeHtml(d.value)}: ${d.percent}%"></div>`)
        .join('')}</div>`;
}

// ---------------------------------------------------------------------------
// Generic chart renderers - turn chat "evidence.chart_data" tool results into
// the same .dashboard-card / .bar-row / .donut-* / .stacked-bar markup the
// dashboards use, reusing the existing CSS.
// ---------------------------------------------------------------------------

// rows: [{label, pct, count, colorClass?}] -> a .dashboard-card of .bar-rows.
function renderBarChart(title, rows) {
    if (!rows || rows.length === 0) return '';
    const body = rows
        .map((r) => {
            const pct = Math.max(0, Math.min(100, r.pct || 0));
            return `
            <div class="bar-row${r.colorClass ? ` ${r.colorClass}` : ''}">
                <span class="bar-label" title="${escapeHtml(String(r.label))}">${escapeHtml(String(r.label))}</span>
                <div class="bar-track"><div class="bar-fill" style="width:${pct}%"></div></div>
                <span class="bar-count">${r.count != null ? escapeHtml(String(r.count)) : ''}</span>
            </div>`;
        })
        .join('');
    return `
        <div class="dashboard-card">
            <h3>${escapeHtml(title)}</h3>
            ${body}
        </div>`;
}

// distribution: [{value, percent, count?}] -> donut + legend, or a bar chart
// for high-cardinality columns (matches the dashboard's own convention).
function renderDonutChart(title, distribution) {
    if (!distribution || distribution.length === 0) return '';
    if (distribution.length > 6) {
        return renderBarChart(
            title,
            distribution.map((d) => ({ label: d.value, pct: d.percent, count: `${d.percent}%` }))
        );
    }
    const stops = conicGradientStops(distribution);
    const top = distribution[0];
    const colorMap = buildColorMap(distribution.map((d) => d.value));
    const legend = renderChartLegend(
        distribution.map((d) => ({ label: d.value, color: colorMap[d.value], percent: d.percent }))
    );
    return `
        <div class="dashboard-card">
            <h3>${escapeHtml(title)}</h3>
            <div class="donut-row">
                <div class="donut-chart" style="background: conic-gradient(${stops})">
                    <div class="donut-hole">
                        <span class="donut-value">${top.percent}%</span>
                        <span class="donut-caption">${escapeHtml(String(top.value))}</span>
                    </div>
                </div>
                ${legend}
            </div>
        </div>`;
}

// segments: [{segment, distribution, dominant_value, dominant_percent}], options: string[]
function renderStackedBarChart(title, options, segments) {
    if (!segments || segments.length === 0 || !options || options.length === 0) return '';
    const colorMap = buildColorMap(options);
    const rows = segments
        .map(
            (seg) => `
            <div class="stacked-bar-row">
                <span class="bar-label" title="${escapeHtml(seg.segment)}">${escapeHtml(seg.segment)}</span>
                ${renderStackedBar(seg.distribution, colorMap)}
                <span class="bar-count">${escapeHtml(String(seg.dominant_value))} <span class="bar-range">(${seg.dominant_percent}%)</span></span>
            </div>`
        )
        .join('');
    const legend = renderChartLegend(options.map((opt) => ({ label: opt, color: colorMap[opt] })));
    return `
        <div class="dashboard-card crosstab-card">
            <h3>${escapeHtml(title)}</h3>
            ${rows}
            ${legend}
        </div>`;
}

const _THRESHOLD_OP_SYMBOLS = { ge: '>=', gt: '>', le: '<=', lt: '<', eq: '=' };

// chartData: [{tool_name, result}] from chat evidence.chart_data -> combined HTML
// (or '' if nothing in chartData maps to a chart - the JSON evidence panel
// remains the fallback in that case).
function renderEvidenceChart(chartData) {
    if (!chartData || chartData.length === 0) return '';
    return chartData.map(renderOneEvidenceChart).filter(Boolean).join('');
}

function renderOneEvidenceChart(entry) {
    const { tool_name: toolName, result } = entry || {};
    if (!result) return '';

    switch (toolName) {
        case 'get_risk_distribution': {
            const dist = result.risk_distribution || {};
            const total = Object.values(dist).reduce((a, b) => a + b, 0) || 1;
            const rows = ['critical', 'high', 'medium', 'low']
                .filter((sev) => dist[sev] != null)
                .map((sev) => ({
                    label: sev,
                    pct: (dist[sev] / total) * 100,
                    count: dist[sev],
                    colorClass: `severity-${sev}`,
                }));
            return renderBarChart('Risk Distribution', rows);
        }
        case 'compare_categories': {
            const comparison = result.comparison || {};
            const rows = Object.entries(comparison).map(([cat, stats]) => ({
                label: cat,
                pct: stats.total_considered ? (stats.flagged / stats.total_considered) * 100 : 0,
                count: `${stats.flagged}/${stats.total_considered}`,
                colorClass: `category-${cat}`,
            }));
            return renderBarChart('Category Comparison', rows);
        }
        case 'get_value_distribution':
            return renderDonutChart(result.column || 'Distribution', result.distribution || []);
        case 'get_segment_stats': {
            const segments = result.segments || [];
            if (segments.length === 0) return '';
            const maxVal = Math.max(...segments.map((s) => s.mean), 1);
            const rows = segments.map((s) => ({
                label: s.segment,
                pct: (s.mean / maxVal) * 100,
                count: `${s.mean} (n=${s.count})`,
                colorClass:
                    s.segment === result.best_segment
                        ? 'segment-best'
                        : s.segment === result.worst_segment
                          ? 'segment-worst'
                          : 'segment-fill',
            }));
            return renderBarChart(`${result.metric_column || 'Metric'} by ${result.segment_column || 'Segment'}`, rows);
        }
        case 'get_response_by_segment':
            return renderStackedBarChart(
                `${result.response_column || 'Response'} by ${result.segment_column || 'Segment'}`,
                result.options || [],
                result.segments || []
            );
        case 'find_top_segment_for_value': {
            const ranking = result.ranking || [];
            const rows = ranking.map((r) => ({
                label: r.segment,
                pct: r.percent,
                count: `${r.percent}%`,
                colorClass: r.segment === result.top_segment ? 'segment-best' : 'segment-fill',
            }));
            return renderBarChart(`"${result.value}" (${result.response_column}) by ${result.segment_column}`, rows);
        }
        case 'find_top_segment_for_numeric_threshold': {
            const ranking = result.ranking || [];
            const rows = ranking.map((r) => ({
                label: r.segment,
                pct: r.percent,
                count: `${r.percent}%`,
                colorClass: r.segment === result.top_segment ? 'segment-best' : 'segment-fill',
            }));
            const op = _THRESHOLD_OP_SYMBOLS[result.op] || result.op;
            return renderBarChart(`${result.value_column} ${op} ${result.threshold} by ${result.segment_column}`, rows);
        }
        case 'get_accuracy_metrics': {
            const calls = result.calls_by_agent || {};
            const total = Object.values(calls).reduce((a, b) => a + b, 0) || 1;
            const rows = Object.entries(calls).map(([agent, count]) => ({
                label: agent,
                pct: (count / total) * 100,
                count,
            }));
            const avgLatency = result.avg_latency_ms != null ? ` (avg ${result.avg_latency_ms}ms/call)` : '';
            return renderBarChart(`LLM Calls by Agent${avgLatency}`, rows);
        }
        default:
            return '';
    }
}
