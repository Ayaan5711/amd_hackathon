"""Report section(s) for the Survey Analytics pack.

Reconstructs a DataFrame from the investigation's entries (each entry's
`ai_response` is the JSON-serialized survey row), re-derives a schema via
`detect_column_type`, and runs the existing segment/anomaly survey tools to
produce genuine, data-grounded markdown - the same content-aware mock pattern
used by the governance report sections.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from app.tools.anomaly_flag import flag_anomalies
from app.tools.segment_stats import get_segment_stats
from app.utils.csv_loader import detect_column_type, get_categorical_columns, get_numeric_columns
from app.utils.llm_client import MockFabricator

SURVEY_REPORT_PROMPT = """You are a people-analytics consultant authoring the "{section_title}" \
section of a survey insights report.

SURVEY OVERVIEW:
{summary}

{extra}

Write the "{section_title}" section as concise markdown (use headings and bullet points, and \
reference specific segments/columns as evidence where relevant).

Respond with JSON only, no other text:
{{"content": "<markdown text for this section>"}}
"""


def _reconstruct_df_and_schema(ctx: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Rebuild the survey DataFrame + a minimal schema from the investigation's entries."""
    rows = [json.loads(e["ai_response"]) for e in ctx["entries"]]
    df = pd.DataFrame(rows)
    schema = {
        col: {"type": detect_column_type(df[col]), "n_unique": int(df[col].nunique())}
        for col in df.columns
    }
    return df, schema


def _segment_highlight(df: pd.DataFrame, schema: dict[str, Any]) -> dict[str, Any] | None:
    """Best-effort get_segment_stats over the first usable categorical/numeric pair."""
    for seg_col in get_categorical_columns(schema):
        if schema[seg_col]["n_unique"] < 2:
            continue
        for metric_col in get_numeric_columns(schema):
            result = get_segment_stats(df, schema, segment_column=seg_col, metric_column=metric_col)
            if result["success"]:
                return result
    return None


def _anomaly_highlight(df: pd.DataFrame, schema: dict[str, Any]) -> dict[str, Any] | None:
    numeric_cols = get_numeric_columns(schema)
    if not numeric_cols:
        return None
    result = flag_anomalies(df, schema, columns=numeric_cols)
    return result if result["success"] else None


def _mock_survey_insights(ctx: dict[str, Any]) -> MockFabricator:
    def fabricator(_messages: list[dict[str, str]]) -> dict[str, Any]:
        df, schema = _reconstruct_df_and_schema(ctx)
        dashboard = ctx["dashboard"]
        dist = dashboard["risk_distribution"]

        lines = [
            "# Survey Insights Summary",
            "",
            f"This run analyzed **{ctx['total_entries']}** survey responses. "
            f"**{ctx['total_flagged']}** were dispatched for follow-up review based on automated "
            f"triage (PII scan + statistical outlier detection).",
            "",
            "## Risk Distribution",
            f"- Critical: {dist.get('critical', 0)}",
            f"- High: {dist.get('high', 0)}",
            f"- Medium: {dist.get('medium', 0)}",
            f"- Low: {dist.get('low', 0)}",
            "",
        ]

        segment = _segment_highlight(df, schema)
        if segment:
            lines.append("## Segment Highlight")
            lines.append(
                f"`{segment['metric_column']}` by `{segment['segment_column']}`: best segment "
                f"**{segment['best_segment']}** vs. worst **{segment['worst_segment']}** "
                f"(gap of {segment['gap']})."
            )
        else:
            lines.append("## Segment Highlight")
            lines.append("No segment with enough responses for a reliable comparison.")
        lines.append("")

        anomalies = _anomaly_highlight(df, schema)
        if anomalies and anomalies["columns_with_anomalies"]:
            lines.append("## Outliers")
            lines.append(anomalies["summary"])
        else:
            lines.append("## Outliers")
            lines.append("No statistically significant outliers detected.")
        lines.append("")

        pii_flagged = dashboard["findings_by_category"]["pii"]["flagged"]
        if pii_flagged:
            lines.append("## PII")
            lines.append(
                f"{pii_flagged} response(s) contain detectable PII in open-text fields - "
                f"review before sharing this dataset externally."
            )
        else:
            lines.append("## PII")
            lines.append("No PII detected in open-text fields.")

        return {"content": "\n".join(lines)}

    return fabricator


def build_survey_insights_prompt(ctx: dict[str, Any]) -> tuple[str, MockFabricator]:
    df, schema = _reconstruct_df_and_schema(ctx)
    dashboard = ctx["dashboard"]
    segment = _segment_highlight(df, schema)
    anomalies = _anomaly_highlight(df, schema)

    summary_lines = [
        f"- Total responses: {ctx['total_entries']}",
        f"- Flagged for follow-up: {ctx['total_flagged']}",
        f"- Risk distribution: {dashboard['risk_distribution']}",
        f"- PII findings: {dashboard['findings_by_category']['pii']['flagged']}",
    ]
    extra_lines = []
    if segment:
        extra_lines.append(
            f"Segment stats: {segment['metric_column']} by {segment['segment_column']} - "
            f"best={segment['best_segment']}, worst={segment['worst_segment']}, gap={segment['gap']}"
        )
    if anomalies:
        extra_lines.append(f"Anomaly summary: {anomalies['summary']}")

    prompt = SURVEY_REPORT_PROMPT.format(
        section_title="Survey Insights Summary",
        summary="\n".join(summary_lines),
        extra="\n".join(extra_lines) if extra_lines else "(no additional segment/outlier data available)",
    )
    return prompt, _mock_survey_insights(ctx)


# Ordered: section_name -> prompt builder. Iteration order is report order.
SURVEY_REPORT_SECTIONS: dict[str, Any] = {
    "survey_insights_summary": build_survey_insights_prompt,
}
