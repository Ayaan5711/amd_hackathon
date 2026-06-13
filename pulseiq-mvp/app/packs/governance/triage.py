"""Triage - the cheap, no-LLM signal computed for every entry.

Combines Presidio PII scanning with the Security/Compliance/Hallucination
heuristic prefilters into one TriageResult per log entry.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.agent.state import TriageResult
from app.packs.governance import pii_agent
from app.packs.governance.compliance_agent import detect_compliance_category
from app.packs.governance.security_agent import detect_injection
from app.utils.csv_loader import df_to_log_entries


def triage_fn(df: pd.DataFrame, context: dict[str, Any]) -> list[TriageResult]:
    """Compute the cheap triage signal for every entry in the log batch."""
    results: list[TriageResult] = []
    for entry in df_to_log_entries(df):
        pii_findings = pii_agent.scan_entry(entry)
        injection_suspect, _ = detect_injection(entry.get("user_prompt"))
        compliance_suspect = detect_compliance_category(entry.get("ai_response")) is not None
        has_context = bool(entry.get("retrieved_context"))

        results.append(TriageResult(
            log_id=entry["log_id"],
            pii_findings=pii_findings,
            has_pii=bool(pii_findings),
            injection_suspect=injection_suspect,
            compliance_suspect=compliance_suspect,
            has_context=has_context,
        ))
    return results
