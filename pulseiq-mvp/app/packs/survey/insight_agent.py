"""Insight Agent - LLM review of statistically anomalous survey responses.

Named `compliance_specialist` / agent="compliance" (not "insight") so the
existing risk_scoring/dashboard/dispatch wiring and
MetricsCollector.GATED_SPECIALIST_AGENTS - all keyed on "security"/
"compliance"/"hallucination" - apply to the Survey pack unchanged.
"""

from __future__ import annotations

import json
import re
from typing import Any

from app.agent.state import LogEntry, SpecialistFinding
from app.config import MAX_TOKENS_SPECIALIST, VLLM_MODEL_SPECIALIST
from app.packs.governance.llm_utils import parse_json_response
from app.utils.llm_client import call_llm_async
from app.utils.metrics import MetricsCollector

INSIGHT_VERDICT_PROMPT = """You are reviewing one survey response that was flagged because one or more \
of its numeric answers are statistical outliers compared to the rest of the dataset.

Survey response:
{response_json}

Flagged fields (value, dataset mean, z-score):
{outlier_json}

In one sentence, decide whether this response likely reflects a genuine, actionable signal \
(e.g. a strongly dissatisfied or disengaged respondent worth following up with) or is more \
likely noise (e.g. a data-entry error).

Respond with JSON only, no other text:
{{"needs_review": true or false, "reason": "<one sentence>"}}
"""


def _mock_insight_verdict(messages: list[dict[str, str]]) -> dict[str, Any]:
    """Content-aware mock verdict for response_schema="insight_verdict"."""
    prompt_text = messages[0].get("content", "") if messages else ""
    match = re.search(r"mean, z-score\):\n(.*?)\n\nIn one sentence", prompt_text, re.S)
    outlier_text = match.group(1).strip() if match else "{}"
    try:
        outliers = json.loads(outlier_text)
    except json.JSONDecodeError:
        outliers = {}

    fields = ", ".join(outliers.keys()) or "a numeric field"
    return {
        "needs_review": True,
        "reason": f"Response is a statistical outlier on {fields} relative to the rest of the dataset.",
    }


async def compliance_specialist(entry: LogEntry, context: dict[str, Any]) -> SpecialistFinding:
    """LLM review for survey responses flagged `compliance_suspect` (statistical outliers)."""
    metrics: MetricsCollector | None = context.get("metrics")
    response_json = entry.get("ai_response") or "{}"
    outlier_json = entry.get("retrieved_context") or "{}"

    prompt = INSIGHT_VERDICT_PROMPT.format(response_json=response_json, outlier_json=outlier_json)
    raw = await call_llm_async(
        messages=[{"role": "user", "content": prompt}],
        model=VLLM_MODEL_SPECIALIST,
        max_tokens=MAX_TOKENS_SPECIALIST,
        json_mode=True,
        enable_thinking=False,
        response_schema="insight_verdict",
        agent="compliance",
        metrics=metrics,
        mock_fabricator=_mock_insight_verdict,
    )
    verdict = parse_json_response(raw)
    flagged = bool(verdict.get("needs_review", False))
    reason = verdict.get("reason", "")
    return SpecialistFinding(
        log_id=entry["log_id"],
        agent="compliance",
        flagged=flagged,
        severity="medium" if flagged else "low",
        summary=reason or "No follow-up needed.",
        evidence={"verdict": verdict, "outliers": json.loads(outlier_json)},
    )
