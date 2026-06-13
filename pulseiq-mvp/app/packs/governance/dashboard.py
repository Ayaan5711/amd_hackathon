"""Dashboard summary - aggregates triage/specialist/risk results for the UI."""

from __future__ import annotations

from typing import Any

from app.agent.state import LogEntry, SpecialistFinding, TriageResult


def dashboard_fn(
    entries: list[LogEntry],
    triage_results: list[TriageResult],
    specialist_findings: list[SpecialistFinding],
    risk_scores: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    entries_by_id = {e["log_id"]: e for e in entries}

    findings_by_category: dict[str, dict[str, int]] = {
        "pii": {"flagged": 0, "total": len(triage_results)},
        "security": {"flagged": 0, "total": 0},
        "compliance": {"flagged": 0, "total": 0},
        "hallucination": {"flagged": 0, "total": 0},
    }

    for triage in triage_results:
        if triage["has_pii"]:
            findings_by_category["pii"]["flagged"] += 1
        if triage["injection_suspect"]:
            findings_by_category["security"]["total"] += 1
        if triage["compliance_suspect"]:
            findings_by_category["compliance"]["total"] += 1
        if triage["has_context"]:
            findings_by_category["hallucination"]["total"] += 1

    for finding in specialist_findings:
        agent = finding["agent"]
        if agent in findings_by_category and finding["flagged"]:
            findings_by_category[agent]["flagged"] += 1

    by_log_id: dict[str, dict[str, Any]] = risk_scores.get("by_log_id", {})
    top_findings = sorted(
        (
            {
                "log_id": log_id,
                "score": score["score"],
                "severity": score["severity"],
                "contributors": score["contributors"],
                "timestamp": entries_by_id.get(log_id, {}).get("timestamp", ""),
            }
            for log_id, score in by_log_id.items()
            if score["severity"] in ("high", "critical")
        ),
        key=lambda r: r["score"],
        reverse=True,
    )

    return {
        "total_entries": len(entries),
        "total_flagged": sum(1 for s in by_log_id.values() if s["severity"] != "low"),
        "findings_by_category": findings_by_category,
        "risk_distribution": risk_scores.get("risk_distribution", {}),
        "overall_risk_score": risk_scores.get("overall_risk_score", 0.0),
        "top_findings": top_findings,
        "metrics": metrics,
    }
