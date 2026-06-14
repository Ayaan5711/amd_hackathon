"""Dashboard for the Survey Analytics pack.

Wraps `app/packs/governance/dashboard.py::dashboard_fn` (risk distribution,
findings-by-category, top findings, metrics - all reused unchanged) and adds
chart-ready blocks the frontend renders as bar charts:

- `metric_summary`: average/min/max for every numeric column ("Average Scores" chart).
- `segment_breakdown`: a `get_segment_stats` result for the primary segment column ×
  the highest-signal numeric metric ("Segment Comparison" chart).
- `demographic_summary`: top value + share per demographic column (Gender, Age Band,
  City, ...), or `None` if the dataset has no demographic columns.
  `{success, total_responses, profiles: [{column, top_value, top_percent,
  distribution: [{value, count, percent}, ...]}, ...]}`.
- `response_summary`: distribution + dominant value/% per Likert-style "response"
  column, or `None` if the dataset has none.
  `{success, total_responses, questions: [{column, options, distribution,
  dominant_value, dominant_percent}, ...]}`.

`category_labels` gives the frontend survey-flavored copy for the governance
categories (`pii` -> PII in open-text responses, `compliance` -> outlier responses
flagged for review); categories with `total == 0` (security/hallucination) are left
for the frontend to hide.
"""

from __future__ import annotations

from typing import Any

from app.agent.state import LogEntry, SpecialistFinding, TriageResult
from app.packs.governance.dashboard import dashboard_fn
from app.packs.survey.categorical import (
    build_demographic_profile,
    build_response_summary,
    split_demographic_and_response_columns,
)
from app.packs.survey.common import (
    pick_dimension_column,
    pick_excluded_columns,
    pick_metric_columns,
    pick_segment_column,
    reconstruct_df_and_schema,
)
from app.tools.segment_stats import get_segment_stats
from app.utils.csv_loader import get_categorical_columns, get_numeric_columns

SURVEY_CATEGORY_LABELS: dict[str, str] = {
    "pii": "PII in open-text responses",
    "security": "Security signals",
    "compliance": "Outlier responses flagged for review",
    "hallucination": "Hallucination signals",
}


def _metric_summary(df: Any, schema: dict[str, Any]) -> list[dict[str, Any]]:
    summary = []
    for col in get_numeric_columns(schema):
        series = df[col].dropna()
        if series.empty:
            continue
        summary.append({
            "column": col,
            "mean": round(float(series.mean()), 2),
            "min": round(float(series.min()), 2),
            "max": round(float(series.max()), 2),
            "count": int(series.count()),
        })
    return summary


def _segment_breakdown(df: Any, schema: dict[str, Any]) -> dict[str, Any] | None:
    categorical_cols = get_categorical_columns(schema)
    numeric_cols = get_numeric_columns(schema)
    dimension_col = pick_dimension_column(categorical_cols)
    segment_col = pick_segment_column(categorical_cols, exclude=dimension_col)
    if not segment_col:
        return None

    for metric_col in pick_metric_columns(numeric_cols):
        result = get_segment_stats(df, schema, segment_column=segment_col, metric_column=metric_col)
        if result["success"]:
            return result
    return None


def _demographic_summary(df: Any, schema: dict[str, Any]) -> dict[str, Any] | None:
    categorical_cols = get_categorical_columns(schema)
    exclude = pick_excluded_columns(categorical_cols)
    demographic_cols, _ = split_demographic_and_response_columns(df, schema, categorical_cols, exclude=exclude)
    if not demographic_cols:
        return None
    result = build_demographic_profile(df, schema, demographic_cols)
    return result if result["success"] else None


def _response_summary(df: Any, schema: dict[str, Any]) -> dict[str, Any] | None:
    categorical_cols = get_categorical_columns(schema)
    exclude = pick_excluded_columns(categorical_cols)
    _, response_cols = split_demographic_and_response_columns(df, schema, categorical_cols, exclude=exclude)
    if not response_cols:
        return None
    result = build_response_summary(df, schema, response_cols)
    return result if result["success"] else None


def survey_dashboard_fn(
    entries: list[LogEntry],
    triage_results: list[TriageResult],
    specialist_findings: list[SpecialistFinding],
    risk_scores: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    dashboard = dashboard_fn(entries, triage_results, specialist_findings, risk_scores, metrics)
    df, schema = reconstruct_df_and_schema(entries)

    dashboard["category_labels"] = SURVEY_CATEGORY_LABELS
    dashboard["metric_summary"] = _metric_summary(df, schema)
    dashboard["segment_breakdown"] = _segment_breakdown(df, schema)
    dashboard["demographic_summary"] = _demographic_summary(df, schema)
    dashboard["response_summary"] = _response_summary(df, schema)
    return dashboard
