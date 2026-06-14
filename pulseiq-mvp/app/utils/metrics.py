"""Per-run metrics: LLM call cost/latency, token-efficiency, and GPU stats."""

import json
import logging
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Specialist agents that are gated by triage (only run on flagged entries).
# Used to compute the "naive vs. actual" token-efficiency comparison.
GATED_SPECIALIST_AGENTS: tuple[str, ...] = ("security", "compliance", "hallucination")


@dataclass
class LLMCallRecord:
    agent: str
    model: str
    tokens_in: int
    tokens_out: int
    latency_ms: float
    enable_thinking: bool = False
    response_schema: str | None = None


@dataclass
class MetricsCollector:
    """Collects LLM call records for a single investigation run."""

    calls: list[LLMCallRecord] = field(default_factory=list)
    started_at: float = field(default_factory=time.time)

    def record_call(
        self,
        agent: str,
        model: str,
        tokens_in: int,
        tokens_out: int,
        latency_ms: float,
        enable_thinking: bool = False,
        response_schema: str | None = None,
    ) -> None:
        self.calls.append(
            LLMCallRecord(
                agent=agent,
                model=model,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                latency_ms=latency_ms,
                enable_thinking=enable_thinking,
                response_schema=response_schema,
            )
        )

    @property
    def total_calls(self) -> int:
        return len(self.calls)

    @property
    def total_tokens(self) -> int:
        return sum(c.tokens_in + c.tokens_out for c in self.calls)

    @property
    def total_latency_ms(self) -> float:
        return sum(c.latency_ms for c in self.calls)

    def calls_by_agent(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for c in self.calls:
            counts[c.agent] = counts.get(c.agent, 0) + 1
        return counts

    def efficiency(self, num_entries: int, agents_per_entry_naive: int = len(GATED_SPECIALIST_AGENTS)) -> dict[str, Any]:
        """
        Token-efficiency story: naive baseline = every gated specialist agent
        runs on every entry. Actual = only the entries the orchestrator
        dispatched to each specialist (triage-gated).
        """
        naive_calls = num_entries * agents_per_entry_naive
        actual_calls = sum(1 for c in self.calls if c.agent in GATED_SPECIALIST_AGENTS)
        reduction_pct = 0.0
        if naive_calls > 0:
            reduction_pct = round((1 - actual_calls / naive_calls) * 100, 1)
        return {
            "naive_llm_calls": naive_calls,
            "actual_llm_calls": actual_calls,
            "reduction_pct": reduction_pct,
        }

    def summary(self, num_entries: int = 0) -> dict[str, Any]:
        return {
            "total_calls": self.total_calls,
            "total_tokens": self.total_tokens,
            "total_latency_ms": round(self.total_latency_ms, 1),
            "wall_clock_seconds": round(time.time() - self.started_at, 2),
            "calls_by_agent": self.calls_by_agent(),
            "efficiency": self.efficiency(num_entries) if num_entries else None,
            "gpu": gpu_stats(),
        }


def _to_float(value: Any) -> float | None:
    try:
        return float(str(value).rstrip("%").strip())
    except (TypeError, ValueError):
        return None


def _bytes_to_gb(value: Any) -> float | None:
    raw_bytes = _to_float(value)
    if raw_bytes is None:
        return None
    return round(raw_bytes / (1024**3), 2)


def _parse_gpu_summary(raw: dict[str, Any]) -> dict[str, Any]:
    """Best-effort extraction of GPU name/utilization/VRAM from `rocm-smi --json`
    output. Field names vary across ROCm versions, so every value defaults to
    `None` if it can't be confidently found - `raw` stays available either way
    for debugging/fallback display."""
    summary: dict[str, Any] = {
        "gpu_name": None,
        "gpu_utilization_pct": None,
        "vram_used_gb": None,
        "vram_total_gb": None,
    }

    card = next((v for k, v in raw.items() if k.startswith("card") and isinstance(v, dict)), None)
    if not card:
        return summary

    for key, value in card.items():
        key_lower = key.lower()
        if any(s in key_lower for s in ("card series", "card model", "product name", "device name")):
            summary["gpu_name"] = value
        elif "gpu use" in key_lower or "gpu busy" in key_lower:
            summary["gpu_utilization_pct"] = _to_float(value)
        elif "vram total used memory" in key_lower:
            summary["vram_used_gb"] = _bytes_to_gb(value)
        elif "vram total memory" in key_lower:
            summary["vram_total_gb"] = _bytes_to_gb(value)

    return summary


def gpu_stats() -> dict[str, Any]:
    """Best-effort AMD GPU stats via rocm-smi. Returns {'gpu_available': False} off-AMD/Windows."""
    try:
        result = subprocess.run(
            ["rocm-smi", "--showuse", "--showmeminfo", "vram", "--showproductname", "--json"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return {"gpu_available": False}
        raw = json.loads(result.stdout)
        return {"gpu_available": True, "raw": raw, **_parse_gpu_summary(raw)}
    except (FileNotFoundError, subprocess.TimeoutExpired, json.JSONDecodeError, OSError) as e:
        logger.debug(f"rocm-smi unavailable: {e}")
        return {"gpu_available": False}
