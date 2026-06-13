"""Triage for the Survey Analytics pack.

Reuses the governance Presidio PII scan over the JSON-serialized row in
`ai_response`, and repurposes `compliance_suspect` to mean "this row has at
least one statistically anomalous numeric answer" (computed by
`survey_entries_fn` and surfaced via `retrieved_context`).

`injection_suspect` and `has_context` are always False - the Survey pack has
no security/hallucination specialists, so `dispatch_plan_fn` never routes to
them and `specialists` (which only has "compliance") never sees them.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from app.agent.state import TriageResult
from app.packs.governance import pii_agent
from app.packs.survey.entries import survey_entries_fn


def triage_fn(df: pd.DataFrame, context: dict[str, Any]) -> list[TriageResult]:
    """Compute the cheap triage signal for every survey response row."""
    results: list[TriageResult] = []
    for entry in survey_entries_fn(df):
        pii_findings = pii_agent.scan_entry(entry)
        results.append(TriageResult(
            log_id=entry["log_id"],
            pii_findings=pii_findings,
            has_pii=bool(pii_findings),
            injection_suspect=False,
            compliance_suspect=bool(entry.get("retrieved_context")),
            has_context=False,
        ))
    return results
