"""Governance pack chat tools - "talk to results" over a completed InvestigationState.

MCP-shaped tool registry + implementations (mirrors app/tools/registry.py's shape
for the Survey pack), but every tool reads from the investigation results
(triage_results, specialist_findings, risk_scores, dashboard, entries) instead
of a raw DataFrame.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from app.agent.state import InvestigationState
from app.packs.governance.pii_agent import pii_finding_for_entry

CATEGORIES: tuple[str, ...] = ("pii", "security", "compliance", "hallucination")

_CONTRIBUTOR_EXPLANATIONS: dict[str, str] = {
    "pii_critical": "Contains high-sensitivity PII (SSN, credit card, bank number, or IBAN).",
    "pii_other": "Contains other PII (e.g. name, email, phone number, or location).",
    "injection": "Confirmed prompt-injection attempt in the user prompt.",
    "compliance": "AI response violates a financial or medical advice policy.",
    "hallucination": "AI response makes claims unsupported by the retrieved context.",
}


GOVERNANCE_TOOL_REGISTRY: list[dict[str, Any]] = [
    {
        "name": "get_findings_by_category",
        "description": (
            "Get the count and list of flagged log entries for one finding category. "
            "Use this when the user asks how many entries had PII, prompt injection, "
            "compliance violations, or hallucinations."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "enum": list(CATEGORIES),
                    "description": "The finding category to look up.",
                }
            },
            "required": ["category"],
        },
    },
    {
        "name": "get_entry_detail",
        "description": (
            "Get the full detail for one log entry by log_id: prompt, response, triage "
            "signals, specialist findings, and risk score. Use this when the user asks "
            "about a specific log entry."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "log_id": {"type": "string", "description": "The log entry ID, e.g. 'LOG-0042'."}
            },
            "required": ["log_id"],
        },
    },
    {
        "name": "get_risk_distribution",
        "description": (
            "Get the dataset-level risk severity distribution (counts of low/medium/high/"
            "critical) and the overall average risk score. Use this for questions about "
            "overall risk or how risky this batch of logs is."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "explain_finding",
        "description": (
            "Get a plain-language explanation of why one log entry received its risk score, "
            "including every contributing factor. Use this when the user asks 'why' a "
            "specific entry was flagged."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "log_id": {"type": "string", "description": "The log entry ID, e.g. 'LOG-0042'."}
            },
            "required": ["log_id"],
        },
    },
    {
        "name": "compare_categories",
        "description": (
            "Compare flagged-entry counts across all four finding categories (PII, security, "
            "compliance, hallucination). Use this when the user asks which category has the "
            "most issues, or wants a side-by-side comparison."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_accuracy_metrics",
        "description": (
            "Get the token-efficiency metrics for this run: naive vs. actual LLM call counts, "
            "the reduction percentage, and the LLM call breakdown by agent. Use this for "
            "questions about efficiency, cost, or how many LLM calls were made."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
]


def _index(investigation: InvestigationState) -> tuple[
    dict[str, Any], dict[str, Any], dict[str, list[dict[str, Any]]], dict[str, Any]
]:
    """Build lookup indexes by log_id: entries, triage results, specialist findings, risk scores."""
    entries_by_id = {e["log_id"]: e for e in investigation.get("entries", [])}
    triage_by_id = {t["log_id"]: t for t in investigation.get("triage_results", [])}
    findings_by_id: dict[str, list[dict[str, Any]]] = {}
    for finding in investigation.get("specialist_findings", []):
        findings_by_id.setdefault(finding["log_id"], []).append(finding)
    risk_by_id = investigation.get("risk_scores", {}).get("by_log_id", {})
    return entries_by_id, triage_by_id, findings_by_id, risk_by_id


def get_findings_by_category(investigation: InvestigationState, category: str, **_: Any) -> dict[str, Any]:
    if category not in CATEGORIES:
        return {"success": False, "error": f"Unknown category '{category}'. Valid: {list(CATEGORIES)}"}

    findings: list[dict[str, Any]] = []
    if category == "pii":
        for triage in investigation.get("triage_results", []):
            if triage["has_pii"]:
                finding = pii_finding_for_entry(triage["log_id"], triage["pii_findings"])
                if finding:
                    findings.append({"log_id": finding["log_id"], "summary": finding["summary"]})
    else:
        for finding in investigation.get("specialist_findings", []):
            if finding["agent"] == category and finding["flagged"]:
                findings.append({"log_id": finding["log_id"], "summary": finding["summary"]})

    return {
        "success": True,
        "category": category,
        "flagged_count": len(findings),
        "findings": findings,
    }


def get_entry_detail(investigation: InvestigationState, log_id: str, **_: Any) -> dict[str, Any]:
    entries_by_id, triage_by_id, findings_by_id, risk_by_id = _index(investigation)

    entry = entries_by_id.get(log_id)
    if entry is None:
        return {"success": False, "error": f"No log entry found with log_id '{log_id}'."}

    triage = triage_by_id.get(log_id, {})
    return {
        "success": True,
        "log_id": log_id,
        "timestamp": entry.get("timestamp"),
        "user_prompt": entry.get("user_prompt"),
        "ai_response": entry.get("ai_response"),
        "retrieved_context": entry.get("retrieved_context"),
        "triage": {
            "has_pii": triage.get("has_pii", False),
            "pii_findings": triage.get("pii_findings", []),
            "injection_suspect": triage.get("injection_suspect", False),
            "compliance_suspect": triage.get("compliance_suspect", False),
            "has_context": triage.get("has_context", False),
        },
        "specialist_findings": findings_by_id.get(log_id, []),
        "risk": risk_by_id.get(log_id, {}),
    }


def get_risk_distribution(investigation: InvestigationState, **_: Any) -> dict[str, Any]:
    risk_scores = investigation.get("risk_scores", {})
    return {
        "success": True,
        "risk_distribution": risk_scores.get("risk_distribution", {}),
        "overall_risk_score": risk_scores.get("overall_risk_score", 0.0),
        "total_entries": len(investigation.get("entries", [])),
    }


def explain_finding(investigation: InvestigationState, log_id: str, **_: Any) -> dict[str, Any]:
    entries_by_id, triage_by_id, findings_by_id, risk_by_id = _index(investigation)

    if log_id not in entries_by_id:
        return {"success": False, "error": f"No log entry found with log_id '{log_id}'."}

    risk = risk_by_id.get(log_id, {"score": 0, "severity": "low", "contributors": []})
    explanations = [
        {"contributor": c, "explanation": _CONTRIBUTOR_EXPLANATIONS.get(c, c)} for c in risk.get("contributors", [])
    ]
    summaries = [f["summary"] for f in findings_by_id.get(log_id, []) if f["flagged"]]
    triage = triage_by_id.get(log_id, {})
    if triage.get("has_pii"):
        pii_finding = pii_finding_for_entry(log_id, triage["pii_findings"])
        if pii_finding:
            summaries.insert(0, pii_finding["summary"])

    return {
        "success": True,
        "log_id": log_id,
        "score": risk.get("score", 0),
        "severity": risk.get("severity", "low"),
        "contributors": explanations,
        "finding_summaries": summaries,
    }


def compare_categories(investigation: InvestigationState, **_: Any) -> dict[str, Any]:
    findings_by_category = investigation.get("dashboard", {}).get("findings_by_category", {})
    comparison = {
        category: {
            "flagged": stats.get("flagged", 0),
            "total_considered": stats.get("total", 0),
        }
        for category, stats in findings_by_category.items()
    }
    ranked = sorted(comparison.items(), key=lambda kv: kv[1]["flagged"], reverse=True)
    return {
        "success": True,
        "comparison": comparison,
        "most_flagged_category": ranked[0][0] if ranked and ranked[0][1]["flagged"] > 0 else None,
    }


def get_accuracy_metrics(investigation: InvestigationState, **_: Any) -> dict[str, Any]:
    metrics = investigation.get("metrics", {})
    return {
        "success": True,
        "total_calls": metrics.get("total_calls", 0),
        "total_tokens": metrics.get("total_tokens", 0),
        "calls_by_agent": metrics.get("calls_by_agent", {}),
        "efficiency": metrics.get("efficiency"),
    }


GOVERNANCE_TOOL_FUNCTIONS: dict[str, Callable[..., dict[str, Any]]] = {
    "get_findings_by_category": get_findings_by_category,
    "get_entry_detail": get_entry_detail,
    "get_risk_distribution": get_risk_distribution,
    "explain_finding": explain_finding,
    "compare_categories": compare_categories,
    "get_accuracy_metrics": get_accuracy_metrics,
}
