"""PII Agent - Presidio-based PII detection.

This is the cheap, no-LLM triage signal: it runs on every log entry inside
`triage_fn` and produces `pii_findings` / `has_pii` directly on the
`TriageResult`. Because PII detection costs nothing extra, its result also
becomes the "pii" row in `specialist_findings` (see `pii_finding_for_entry`)
without going through the LLM specialist-dispatch step.
"""

from __future__ import annotations

from typing import Any

from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider

from app.agent.state import LogEntry, SpecialistFinding
from app.config import PII_CRITICAL_ENTITIES, PII_ENTITIES, PII_SCORE_THRESHOLD, PRESIDIO_SPACY_MODEL

_analyzer: AnalyzerEngine | None = None


def _get_analyzer() -> AnalyzerEngine:
    """Lazily build a Presidio AnalyzerEngine pinned to PRESIDIO_SPACY_MODEL.

    Presidio's default AnalyzerEngine() pulls en_core_web_lg (~400MB); we pin
    to en_core_web_sm to keep the footprint small for the AMD notebook pull.
    """
    global _analyzer
    if _analyzer is None:
        nlp_configuration = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": "en", "model_name": PRESIDIO_SPACY_MODEL}],
        }
        nlp_engine = NlpEngineProvider(nlp_configuration=nlp_configuration).create_engine()
        _analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])
    return _analyzer


def scan_text(text: str | None) -> list[dict[str, Any]]:
    """Scan a single text field for PII entities above PII_SCORE_THRESHOLD."""
    if not text:
        return []
    analyzer = _get_analyzer()
    results = analyzer.analyze(text=text, language="en", entities=PII_ENTITIES)
    findings: list[dict[str, Any]] = []
    for r in results:
        if r.score < PII_SCORE_THRESHOLD:
            continue
        findings.append({
            "entity_type": r.entity_type,
            "score": round(r.score, 2),
            "snippet": text[r.start : r.end],
        })
    return findings


def scan_entry(entry: LogEntry) -> list[dict[str, Any]]:
    """Scan user_prompt and ai_response of a log entry for PII."""
    findings: list[dict[str, Any]] = []
    for field in ("user_prompt", "ai_response"):
        for f in scan_text(entry.get(field)):  # type: ignore[arg-type]
            findings.append({**f, "field": field})
    return findings


def pii_finding_for_entry(log_id: str, pii_findings: list[dict[str, Any]]) -> SpecialistFinding | None:
    """Build a SpecialistFinding-shaped record for the "pii" category.

    Returns None when no PII was found (no finding to report). Severity here
    reflects only the PII category in isolation - the overall per-entry
    severity (combining all categories) is computed by risk_scoring.
    """
    if not pii_findings:
        return None
    entity_types = sorted({f["entity_type"] for f in pii_findings})
    critical = sorted(set(entity_types) & PII_CRITICAL_ENTITIES)
    summary = f"Detected {', '.join(entity_types)} in this interaction."
    if critical:
        summary += f" Includes high-sensitivity entities: {', '.join(critical)}."
    return SpecialistFinding(
        log_id=log_id,
        agent="pii",
        flagged=True,
        severity="medium",
        summary=summary,
        evidence={"pii_findings": pii_findings},
    )
