/**
 * Persistent platform strip - shows the AMD ROCm/vLLM stack (and live GPU
 * stats when available) that powers this app. Self-contained: loaded on
 * every page (including index.html, which loads no other script files).
 */
(function () {
    const PLATFORM_POLL_MS = 10000;

    function _escapeHtml(str) {
        const div = document.createElement('div');
        div.textContent = String(str);
        return div.innerHTML;
    }

    function formatPlatformLine(info) {
        const gpu = info.gpu || {};
        const models = info.models || [];
        const modelLabel = models.length ? _escapeHtml(models.join(', ')) : 'Qwen3-8B';

        if (info.llm_mode === 'vllm') {
            if (gpu.gpu_available) {
                const parts = [
                    `<strong>${gpu.gpu_name ? _escapeHtml(gpu.gpu_name) : 'AMD Instinct GPU'}</strong>`,
                    'ROCm',
                    `vLLM &middot; ${modelLabel}`,
                ];
                if (gpu.gpu_utilization_pct != null) {
                    parts.push(`GPU ${Math.round(gpu.gpu_utilization_pct)}%`);
                }
                if (gpu.vram_used_gb != null && gpu.vram_total_gb != null) {
                    parts.push(`VRAM ${gpu.vram_used_gb}/${gpu.vram_total_gb} GB`);
                }
                return '⚡ ' + parts.join(' &middot; ');
            }
            return `⚡ AMD ROCm &middot; vLLM serving <strong>${modelLabel}</strong>`;
        }

        return ' Mock mode (<code>LLM_MODE=mock</code>) &mdash; no live GPU/LLM calls';
    }

    async function refreshPlatformStrip(stripEl) {
        try {
            const res = await fetch('api/platform/info');
            if (!res.ok) return;
            const info = await res.json();
            stripEl.innerHTML = formatPlatformLine(info);
            stripEl.title = info.vllm_base_url ? `vLLM endpoint: ${info.vllm_base_url}` : '';
        } catch (e) {
            // Platform strip is informational only - leave previous content on error.
        }
    }

    function initPlatformStrip() {
        const strip = document.createElement('div');
        strip.id = 'platform-strip';
        strip.className = 'platform-strip';
        strip.textContent = 'Loading platform info…';
        document.body.insertBefore(strip, document.body.firstChild);

        function applyOffset() {
            const height = strip.getBoundingClientRect().height;
            const nav = document.querySelector('.nav');
            if (nav) {
                nav.style.top = `${height}px`;
            }
            const main = document.querySelector('.main');
            if (main) {
                if (main.dataset.basePaddingTop === undefined) {
                    main.dataset.basePaddingTop = getComputedStyle(main).paddingTop;
                }
                const base = parseFloat(main.dataset.basePaddingTop) || 0;
                main.style.paddingTop = `${base + height}px`;
            }
        }

        refreshPlatformStrip(strip).then(applyOffset);
        window.addEventListener('resize', applyOffset);
        setInterval(() => refreshPlatformStrip(strip).then(applyOffset), PLATFORM_POLL_MS);
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initPlatformStrip);
    } else {
        initPlatformStrip();
    }
})();
