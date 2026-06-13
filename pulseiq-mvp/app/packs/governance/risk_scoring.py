"""Risk Scoring - pure aggregation (no LLM) over triage + specialist findings.

Mirrors `compute_severity()` in scripts/generate_synthetic_logs.py so that,
on the seeded dataset, a run with perfect detection reproduces
ground_truth.csv's severity column exactly.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from app.agent.state import SpecialistFinding, TriageResult
from app.config import PII_CRITICAL_ENTITIES, RISK_SEVERITY_DEFAULT, RISK_SEVERITY_THRESHOLDS, RISK_WEIGHTS


def _severity_for_score(score: int) -> str:
    for severity, threshold in RISK_SEVERITY_THRESHOLDS:
        if score >= threshold:
            return severity
    return RISK_SEVERITY_DEFAULT


def risk_scoring_fn(
    triage_results: list[TriageResult],
    specialist_findings: list[SpecialistFinding],
) -> dict[str, Any]:
    """Aggregate per-entry risk scores/severity and dataset-level distribution."""
    findings_by_log: dict[str, list[SpecialistFinding]] = defaultdict(list)
    for finding in specialist_findings:
        findings_by_log[finding["log_id"]].append(finding)

    by_log_id: dict[str, dict[str, Any]] = {}
    severity_counts: dict[str, int] = {"low": 0, "medium": 0, "high": 0, "critical": 0}

    for triage in triage_results:
        log_id = triage["log_id"]
        score = 0
        contributors: list[str] = []

        if triage["has_pii"]:
            entity_types = {f["entity_type"] for f in triage["pii_findings"]}
            if entity_types & PII_CRITICAL_ENTITIES:
                score += RISK_WEIGHTS["pii_critical"]
                contributors.append("pii_critical")
            else:
                score += RISK_WEIGHTS["pii_other"]
                contributors.append("pii_other")

        for finding in findings_by_log.get(log_id, []):
            if not finding["flagged"]:
                continue
            if finding["agent"] == "security":
                score += RISK_WEIGHTS["injection"]
                contributors.append("injection")
            elif finding["agent"] == "compliance":
                score += RISK_WEIGHTS["compliance"]
                contributors.append("compliance")
            elif finding["agent"] == "hallucination":
                score += RISK_WEIGHTS["hallucination"]
                contributors.append("hallucination")

        severity = _severity_for_score(score)
        severity_counts[severity] += 1
        by_log_id[log_id] = {"score": score, "severity": severity, "contributors": contributors}

    overall_risk_score = (
        sum(v["score"] for v in by_log_id.values()) / len(by_log_id) if by_log_id else 0.0
    )

    return {
        "by_log_id": by_log_id,
        "risk_distribution": severity_counts,
        "overall_risk_score": round(overall_risk_score, 2),
    }
