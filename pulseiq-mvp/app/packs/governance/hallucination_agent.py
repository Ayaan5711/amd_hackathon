"""Hallucination / Groundedness Agent.

Triage: entries with a non-empty `retrieved_context` are candidates
(`has_context=True`).
Specialist: an LLM groundedness judge compares the AI response against the
retrieved context and flags unsupported claims.
"""

from __future__ import annotations

import re
from typing import Any

from app.agent.state import LogEntry, SpecialistFinding
from app.config import MAX_TOKENS_SPECIALIST, VLLM_MODEL_SPECIALIST
from app.packs.governance.llm_utils import parse_json_response
from app.utils.llm_client import call_llm_async
from app.utils.metrics import MetricsCollector

_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")

# Absolutist/exaggeration phrases that show up in fabricated answers but not
# in the (truthful) retrieved policy context.
_RED_FLAG_PHRASES = [
    "indefinitely", "no cap", "no deadline", "any time", "automatically",
    "no minimum", "no approval", "reduced rate", "no question",
]


def _unsupported_claims(ai_response: str, retrieved_context: str) -> list[str]:
    """Heuristic groundedness check: numbers and absolutist phrases in the
    response that don't appear anywhere in the retrieved context."""
    claims: list[str] = []

    resp_nums = set(_NUMBER_RE.findall(ai_response))
    ctx_nums = set(_NUMBER_RE.findall(retrieved_context))
    extra_nums = sorted(resp_nums - ctx_nums)
    if extra_nums:
        claims.append(f"Figures not found in retrieved context: {', '.join(extra_nums)}")

    response_lower = ai_response.lower()
    context_lower = retrieved_context.lower()
    extra_phrases = [
        p for p in _RED_FLAG_PHRASES if p in response_lower and p not in context_lower
    ]
    if extra_phrases:
        claims.append(f"Unsupported claims: {', '.join(extra_phrases)}")

    return claims


HALLUCINATION_VERDICT_PROMPT = """You are a groundedness reviewer for an AI assistant that answers \
questions using retrieved policy context.

Retrieved context:
{retrieved_context}

AI response:
{ai_response}

Determine whether every factual claim in the AI response (numbers, dates, \
percentages, deadlines, eligibility rules) is directly supported by the \
retrieved context. List any claims that are NOT supported.

Respond with JSON only, no other text:
{{"grounded": true or false, "unsupported_claims": ["..."], "severity": "low" or "medium" or "high"}}
"""


def _mock_hallucination_verdict(messages: list[dict[str, str]]) -> dict[str, Any]:
    """Content-aware mock verdict for response_schema="hallucination_verdict"."""
    prompt_text = messages[0].get("content", "") if messages else ""
    ctx_match = re.search(r"Retrieved context:\n(.*?)\n\nAI response:", prompt_text, re.S)
    resp_match = re.search(r"AI response:\n(.*?)\n\nDetermine", prompt_text, re.S)
    retrieved_context = ctx_match.group(1).strip() if ctx_match else ""
    ai_response = resp_match.group(1).strip() if resp_match else ""

    claims = _unsupported_claims(ai_response, retrieved_context)
    if claims:
        return {"grounded": False, "unsupported_claims": claims, "severity": "medium"}
    return {"grounded": True, "unsupported_claims": [], "severity": "low"}


async def hallucination_specialist(entry: LogEntry, context: dict[str, Any]) -> SpecialistFinding:
    """LLM-backed groundedness check for entries with retrieved_context."""
    metrics: MetricsCollector | None = context.get("metrics")
    retrieved_context = entry.get("retrieved_context") or ""
    prompt = HALLUCINATION_VERDICT_PROMPT.format(
        retrieved_context=retrieved_context,
        ai_response=entry.get("ai_response", ""),
    )
    raw = await call_llm_async(
        messages=[{"role": "user", "content": prompt}],
        model=VLLM_MODEL_SPECIALIST,
        max_tokens=MAX_TOKENS_SPECIALIST,
        json_mode=True,
        enable_thinking=False,
        response_schema="hallucination_verdict",
        agent="hallucination",
        metrics=metrics,
        mock_fabricator=_mock_hallucination_verdict,
    )
    verdict = parse_json_response(raw)
    grounded = bool(verdict.get("grounded", True))
    flagged = not grounded
    unsupported = verdict.get("unsupported_claims", [])
    summary = (
        f"Response contains claims not supported by retrieved context: {'; '.join(unsupported)}"
        if flagged
        else "Response is grounded in the retrieved context."
    )
    return SpecialistFinding(
        log_id=entry["log_id"],
        agent="hallucination",
        flagged=flagged,
        severity="medium" if flagged else "low",
        summary=summary,
        evidence={"verdict": verdict},
    )
