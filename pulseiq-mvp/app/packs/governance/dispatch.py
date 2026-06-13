"""Dispatch planning - decides which specialist(s) examine which entries.

Deterministic given triage results. The orchestrator_node (investigation
graph) uses this as the basis for - and to validate - the LLM-authored
investigation plan.
"""

from __future__ import annotations

from typing import Any

from app.agent.state import TriageResult


def dispatch_plan_fn(triage_results: list[TriageResult], context: dict[str, Any]) -> list[dict[str, str]]:
    """Build the [{"log_id": ..., "agent": ...}, ...] dispatch plan for flagged entries."""
    plan: list[dict[str, str]] = []
    for triage in triage_results:
        if triage["injection_suspect"]:
            plan.append({"log_id": triage["log_id"], "agent": "security"})
        if triage["compliance_suspect"]:
            plan.append({"log_id": triage["log_id"], "agent": "compliance"})
        if triage["has_context"]:
            plan.append({"log_id": triage["log_id"], "agent": "hallucination"})
    return plan
