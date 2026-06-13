"""Report Generation - 5 audit-report sections authored by an LLM from the
completed investigation's findings, risk scores, and dashboard summary.

Each section's prompt builder returns `(prompt, mock_fabricator)`. The mock
fabricator closes over `report_context` and fabricates realistic, run-specific
markdown - the same content-aware mock pattern used by the specialist agents -
so LLM_MODE=mock produces a genuinely useful demo report.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from app.packs.governance.pii_agent import pii_finding_for_entry
from app.utils.llm_client import MockFabricator

REPORT_SECTION_PROMPT = """You are a governance/compliance analyst authoring a section of an AI-usage \
audit report for the "{section_title}" section.

INVESTIGATION SUMMARY:
{summary}

{extra}

Write the "{section_title}" section as concise markdown (use headings and bullet points, and \
reference specific log_ids as evidence where relevant).

Respond with JSON only, no other text:
{{"content": "<markdown text for this section>"}}
"""

_CATEGORY_TITLES: dict[str, str] = {
    "pii": "PII Exposure",
    "security": "Prompt Injection",
    "compliance": "Policy Compliance",
    "hallucination": "Hallucination / Groundedness",
}


def _summary_block(ctx: dict[str, Any]) -> str:
    dashboard = ctx["dashboard"]
    eff = ctx["metrics"].get("efficiency") or {}
    lines = [
        f"- Total entries reviewed: {ctx['total_entries']}",
        f"- Entries dispatched to specialist review: {ctx['total_flagged']}",
        f"- Risk distribution: {dashboard['risk_distribution']}",
        f"- Overall risk score (0-100 avg): {dashboard['overall_risk_score']}",
        f"- Findings by category: {dashboard['findings_by_category']}",
    ]
    if eff:
        lines.append(
            f"- Token efficiency: {eff['actual_llm_calls']} specialist calls vs. "
            f"{eff['naive_llm_calls']} naive ({eff['reduction_pct']}% reduction)"
        )
    return "\n".join(lines)


def _findings_by_category(ctx: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    """log_id/summary pairs for every flagged finding, grouped by category."""
    grouped: dict[str, list[dict[str, str]]] = {cat: [] for cat in _CATEGORY_TITLES}
    for triage in ctx["triage_results"]:
        if triage["has_pii"]:
            finding = pii_finding_for_entry(triage["log_id"], triage["pii_findings"])
            if finding:
                grouped["pii"].append({"log_id": finding["log_id"], "summary": finding["summary"]})
    for finding in ctx["specialist_findings"]:
        if finding["flagged"] and finding["agent"] in grouped:
            grouped[finding["agent"]].append({"log_id": finding["log_id"], "summary": finding["summary"]})
    return grouped


def _top_findings(ctx: dict[str, Any], limit: int = 5) -> list[dict[str, Any]]:
    return ctx["dashboard"]["top_findings"][:limit]


# =============================================================================
# 1. Executive Audit Summary
# =============================================================================

def _mock_executive_summary(ctx: dict[str, Any]) -> MockFabricator:
    def fabricator(_messages: list[dict[str, str]]) -> dict[str, Any]:
        dashboard = ctx["dashboard"]
        dist = dashboard["risk_distribution"]
        eff = ctx["metrics"].get("efficiency") or {}
        lines = [
            "# Executive Audit Summary",
            "",
            f"This audit reviewed **{ctx['total_entries']}** AI-assistant interaction log entries. "
            f"**{ctx['total_flagged']}** entries were dispatched for specialist review based on "
            f"automated triage (PII scan + heuristic prefilters).",
            "",
            "## Risk Distribution",
            f"- Critical: {dist.get('critical', 0)}",
            f"- High: {dist.get('high', 0)}",
            f"- Medium: {dist.get('medium', 0)}",
            f"- Low: {dist.get('low', 0)}",
            "",
            f"Overall risk score (0-100 average across all entries): **{dashboard['overall_risk_score']}**.",
        ]
        top = _top_findings(ctx, limit=3)
        lines.append("")
        if top:
            lines.append("## Top Findings")
            for f in top:
                lines.append(
                    f"- `{f['log_id']}` - severity **{f['severity']}** (score {f['score']}), "
                    f"contributors: {', '.join(f['contributors'])}"
                )
        else:
            lines.append("No high or critical severity findings in this run.")
        if eff:
            lines.append("")
            lines.append("## Efficiency")
            lines.append(
                f"Triage-gated dispatch made {eff['actual_llm_calls']} specialist LLM calls versus "
                f"{eff['naive_llm_calls']} for a naive \"run every agent on every entry\" approach - "
                f"a **{eff['reduction_pct']}%** reduction."
            )
        return {"content": "\n".join(lines)}

    return fabricator


def build_executive_summary_prompt(ctx: dict[str, Any]) -> tuple[str, MockFabricator]:
    prompt = REPORT_SECTION_PROMPT.format(
        section_title="Executive Audit Summary",
        summary=_summary_block(ctx),
        extra="Audience: company leadership. Open with the headline numbers, then the top 2-3 "
        "findings, then close with the token-efficiency callout.",
    )
    return prompt, _mock_executive_summary(ctx)


# =============================================================================
# 2. Detailed Findings Report
# =============================================================================

def _mock_detailed_findings(ctx: dict[str, Any]) -> MockFabricator:
    def fabricator(_messages: list[dict[str, str]]) -> dict[str, Any]:
        grouped = _findings_by_category(ctx)
        lines = ["# Detailed Findings Report", ""]
        for category, title in _CATEGORY_TITLES.items():
            items = grouped[category]
            lines.append(f"## {title} ({len(items)} flagged)")
            if items:
                for item in items:
                    lines.append(f"- `{item['log_id']}`: {item['summary']}")
            else:
                lines.append("No findings in this category.")
            lines.append("")
        return {"content": "\n".join(lines)}

    return fabricator


def build_detailed_findings_prompt(ctx: dict[str, Any]) -> tuple[str, MockFabricator]:
    grouped = _findings_by_category(ctx)
    lines: list[str] = []
    for category, title in _CATEGORY_TITLES.items():
        items = grouped[category]
        lines.append(f"{title}: {len(items)} flagged")
        for item in items[:10]:
            lines.append(f"  - {item['log_id']}: {item['summary']}")
    prompt = REPORT_SECTION_PROMPT.format(
        section_title="Detailed Findings Report",
        summary=_summary_block(ctx),
        extra="Findings by category (log_id: summary):\n" + "\n".join(lines),
    )
    return prompt, _mock_detailed_findings(ctx)


# =============================================================================
# 3. Remediation Plan (30/60/90)
# =============================================================================

def _mock_remediation_plan(ctx: dict[str, Any]) -> MockFabricator:
    def fabricator(_messages: list[dict[str, str]]) -> dict[str, Any]:
        grouped = _findings_by_category(ctx)
        dist = ctx["dashboard"]["risk_distribution"]
        lines = ["# Remediation Plan (30/60/90 Days)", "", "## Next 30 Days"]
        if dist.get("critical", 0) or grouped["pii"]:
            lines.append(
                f"- Triage and remediate all {dist.get('critical', 0)} critical-severity entries "
                f"and {len(grouped['pii'])} PII exposure(s) identified in this run."
            )
        if grouped["security"]:
            lines.append(
                f"- Review and patch the system prompt / input filtering that allowed "
                f"{len(grouped['security'])} prompt-injection attempt(s) to reach the model."
            )
        if not (dist.get("critical", 0) or grouped["pii"] or grouped["security"]):
            lines.append("- No urgent items; proceed with the standard monitoring cadence.")
        lines += ["", "## Days 31-60"]
        lines.append(
            f"- Address {dist.get('high', 0)} high-severity finding(s) and "
            f"{len(grouped['compliance'])} compliance flag(s) via policy/RAG corpus updates."
        )
        lines.append(
            f"- Investigate {len(grouped['hallucination'])} groundedness finding(s) and tune "
            f"retrieval/grounding prompts accordingly."
        )
        lines += [
            "",
            "## Days 61-90",
            "- Roll the continuous-monitoring rules (see below) into production alerting.",
            "- Re-run this audit on a fresh log batch and compare risk distributions.",
        ]
        return {"content": "\n".join(lines)}

    return fabricator


def build_remediation_plan_prompt(ctx: dict[str, Any]) -> tuple[str, MockFabricator]:
    prompt = REPORT_SECTION_PROMPT.format(
        section_title="Remediation Plan (30/60/90 Days)",
        summary=_summary_block(ctx),
        extra="Group prioritized action items into '0-30 days', '31-60 days', and '61-90 days' "
        "horizons, ordered by the severity of the underlying findings.",
    )
    return prompt, _mock_remediation_plan(ctx)


# =============================================================================
# 4. Incident Notification Drafts
# =============================================================================

def _mock_incident_notifications(ctx: dict[str, Any]) -> MockFabricator:
    def fabricator(_messages: list[dict[str, str]]) -> dict[str, Any]:
        top = _top_findings(ctx, limit=5)
        if not top:
            return {
                "content": "# Incident Notification Drafts\n\nNo high or critical severity "
                "findings in this run - no incident notifications required."
            }
        entries_by_id = ctx["entries_by_id"]
        lines = ["# Incident Notification Drafts", ""]
        for f in top:
            entry = entries_by_id.get(f["log_id"], {})
            lines += [
                f"## Incident: `{f['log_id']}` (severity: {f['severity']})",
                f"**To:** Data Protection Officer / Security Team",
                f"**Subject:** AI interaction log {f['log_id']} flagged for review ({f['severity']})",
                "",
                f"During an automated audit, log entry `{f['log_id']}` "
                f"(timestamp {entry.get('timestamp', 'unknown')}) was flagged with risk score "
                f"{f['score']} - contributors: {', '.join(f['contributors'])}. Recommended action: "
                f"review the entry, confirm whether remediation (redaction, model/policy update, or "
                f"user notification) is required, and log the outcome.",
                "",
            ]
        return {"content": "\n".join(lines)}

    return fabricator


def build_incident_notifications_prompt(ctx: dict[str, Any]) -> tuple[str, MockFabricator]:
    top = _top_findings(ctx, limit=5)
    if top:
        lines = [
            f"- {f['log_id']}: severity {f['severity']}, score {f['score']}, "
            f"contributors {f['contributors']}"
            for f in top
        ]
        findings_block = "\n".join(lines)
    else:
        findings_block = "(none)"
    prompt = REPORT_SECTION_PROMPT.format(
        section_title="Incident Notification Drafts",
        summary=_summary_block(ctx),
        extra="High/critical findings requiring a notification draft:\n" + findings_block + "\n\n"
        "For each, draft a short internal notification (to the Data Protection Officer / Security "
        "team) naming the log_id, severity, and recommended next step. If there are no high/critical "
        "findings, state that no notifications are required.",
    )
    return prompt, _mock_incident_notifications(ctx)


# =============================================================================
# 5. Continuous Monitoring Recommendations
# =============================================================================

def _mock_monitoring_recommendations(ctx: dict[str, Any]) -> MockFabricator:
    def fabricator(_messages: list[dict[str, str]]) -> dict[str, Any]:
        grouped = _findings_by_category(ctx)
        eff = ctx["metrics"].get("efficiency") or {}
        lines = ["# Continuous Monitoring Recommendations", ""]
        if grouped["pii"]:
            lines.append(
                "- **PII**: Keep the Presidio PII scan in the triage path on every entry; alert "
                "when CREDIT_CARD, US_SSN, US_BANK_NUMBER, or IBAN_CODE entities are detected "
                "(critical severity)."
            )
        if grouped["security"]:
            lines.append(
                "- **Prompt injection**: Alert whenever the heuristic injection prefilter fires, "
                "and route those sessions to the security specialist regardless of confidence."
            )
        if grouped["compliance"]:
            lines.append(
                "- **Compliance**: Refresh the policy RAG corpus quarterly and re-run the "
                "compliance specialist on any response touching financial or medical keywords."
            )
        if grouped["hallucination"]:
            lines.append(
                "- **Groundedness**: For any response generated with retrieved_context, run the "
                "hallucination specialist and alert when unsupported claims are detected."
            )
        if not any(grouped.values()):
            lines.append(
                "- No findings in this run; continue running the existing triage rules on every "
                "batch and re-evaluate thresholds periodically."
            )
        if eff:
            lines += [
                "",
                f"- **Efficiency**: This triage-gated approach made {eff['actual_llm_calls']} "
                f"specialist calls vs. {eff['naive_llm_calls']} naive "
                f"({eff['reduction_pct']}% reduction) - keep the triage prefilters as the gating "
                f"mechanism in production.",
            ]
        return {"content": "\n".join(lines)}

    return fabricator


def build_monitoring_recommendations_prompt(ctx: dict[str, Any]) -> tuple[str, MockFabricator]:
    prompt = REPORT_SECTION_PROMPT.format(
        section_title="Continuous Monitoring Recommendations",
        summary=_summary_block(ctx),
        extra="Propose concrete, production-ready monitoring rules/alerts derived from the "
        "categories that had findings in this run. If a category had zero findings, you may "
        "omit a rule for it.",
    )
    return prompt, _mock_monitoring_recommendations(ctx)


# Ordered: section_name -> prompt builder. Iteration order is report order.
REPORT_SECTIONS: dict[str, Callable[[dict[str, Any]], tuple[str, MockFabricator]]] = {
    "executive_summary": build_executive_summary_prompt,
    "detailed_findings": build_detailed_findings_prompt,
    "remediation_plan": build_remediation_plan_prompt,
    "incident_notifications": build_incident_notifications_prompt,
    "monitoring_recommendations": build_monitoring_recommendations_prompt,
}
