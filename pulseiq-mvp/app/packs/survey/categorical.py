"""Demographic + Likert "response" analysis for the Survey Analytics pack.

The existing `get_segment_stats`/`_segment_table` path groups a NUMERIC metric column
(e.g. Satisfaction_Score, NPS) by a categorical segment column (e.g. Department,
Quarter) and compares means. Many real survey datasets (e.g. a TCS-style "Inflation
Expectations Survey") instead carry:

- demographic columns (Gender, Age Band, Income Group, State, City, ...): the
  interesting output is the % share of each value ("Male share: 51.3%").
- small-cardinality categorical "response"/Likert columns (e.g. "More than current" /
  "Similar to current" / "Less than current" / "No change" / "Decline"): the
  interesting output is the % choosing each option, the "dominant" option, and how
  that breaks down per demographic segment (a cross-tab).

This module is purely additive: it classifies categorical columns into
demographic/response buckets (`split_demographic_and_response_columns`) and provides
pandas-only primitives that mirror the `{"success": bool, ...}` shape used by
`app/tools/*`, so they can be wrapped as chat tools identically to
`get_segment_stats`.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

import pandas as pd

from app.config import MIN_SEGMENT_SIZE

_DEMOGRAPHIC_PATTERN = re.compile(
    r"gender|sex|age|age.?band|age.?group|income|education|occupation|"
    r"\bstate\b|\bcity\b|region|location|respondent.?category|marital|"
    r"qualification",
    re.IGNORECASE,
)

# Cardinality bounds for a "response"/Likert-style categorical column.
RESPONSE_MIN_UNIQUE = 2
RESPONSE_MAX_UNIQUE = 8


def split_demographic_and_response_columns(
    df: pd.DataFrame,
    schema: dict[str, Any],
    categorical_cols: list[str],
    exclude: set[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Split `categorical_cols` into (demographic_cols, response_cols), order preserved.

    - demographic_cols: column name matches `_DEMOGRAPHIC_PATTERN` (Gender, Age Band,
      Income Group, State, City, Region, Education, Occupation, ...), regardless of
      cardinality.
    - response_cols: not a demographic column, not in `exclude`, and
      `RESPONSE_MIN_UNIQUE <= n_unique <= RESPONSE_MAX_UNIQUE`.
    - everything else (incl. columns in `exclude`) is skipped entirely.
    """
    exclude = exclude or set()
    demographic_cols: list[str] = []
    response_cols: list[str] = []
    for col in categorical_cols:
        if _DEMOGRAPHIC_PATTERN.search(col):
            demographic_cols.append(col)
        elif col in exclude:
            continue
        else:
            n_unique = schema.get(col, {}).get("n_unique", 0)
            if RESPONSE_MIN_UNIQUE <= n_unique <= RESPONSE_MAX_UNIQUE:
                response_cols.append(col)
    return demographic_cols, response_cols


def get_value_distribution(df: pd.DataFrame, schema: dict[str, Any], column: str) -> dict[str, Any]:
    """% breakdown + dominant value for one categorical column."""
    if column not in df.columns:
        return {"success": False, "error": f"Column '{column}' not found.", "available_columns": list(df.columns)}

    series = df[column].dropna()
    if series.empty:
        return {"success": False, "error": f"Column '{column}' has no non-null values."}

    counts = series.value_counts()
    total = int(counts.sum())
    distribution = [
        {"value": str(value), "count": int(count), "percent": round(count / total * 100, 1)}
        for value, count in counts.items()
    ]
    top = distribution[0]
    return {
        "success": True,
        "column": column,
        "total": total,
        "distribution": distribution,
        "dominant_value": top["value"],
        "dominant_percent": top["percent"],
    }


def build_demographic_profile(df: pd.DataFrame, schema: dict[str, Any], demographic_cols: list[str]) -> dict[str, Any]:
    """Top-value + share per demographic column, plus full distributions for charts."""
    if not demographic_cols:
        return {"success": False, "total_responses": len(df), "profiles": []}

    profiles = []
    for col in demographic_cols:
        result = get_value_distribution(df, schema, col)
        if not result["success"]:
            continue
        profiles.append({
            "column": col,
            "top_value": result["dominant_value"],
            "top_percent": result["dominant_percent"],
            "distribution": result["distribution"],
        })

    return {"success": bool(profiles), "total_responses": len(df), "profiles": profiles}


def build_response_summary(df: pd.DataFrame, schema: dict[str, Any], response_cols: list[str]) -> dict[str, Any]:
    """Distribution + dominant value/percent for each Likert-style "response" column."""
    if not response_cols:
        return {"success": False, "total_responses": len(df), "questions": []}

    questions = []
    for col in response_cols:
        result = get_value_distribution(df, schema, col)
        if not result["success"]:
            continue
        questions.append({
            "column": col,
            "options": [d["value"] for d in result["distribution"]],
            "distribution": result["distribution"],
            "dominant_value": result["dominant_value"],
            "dominant_percent": result["dominant_percent"],
        })

    return {"success": bool(questions), "total_responses": len(df), "questions": questions}


def build_segment_response_crosstab(
    df: pd.DataFrame, schema: dict[str, Any], segment_column: str, response_column: str
) -> dict[str, Any]:
    """% choosing each option of `response_column`, broken out by `segment_column`
    (the "Full demographic analysis" cross-tab). Segments with fewer than
    `MIN_SEGMENT_SIZE` responses are dropped."""
    if segment_column not in df.columns:
        return {
            "success": False,
            "error": f"Segment column '{segment_column}' not found.",
            "available_columns": list(df.columns),
        }
    if response_column not in df.columns:
        return {
            "success": False,
            "error": f"Response column '{response_column}' not found.",
            "available_columns": list(df.columns),
        }

    overall = get_value_distribution(df, schema, response_column)
    options = [d["value"] for d in overall.get("distribution", [])]

    segments = []
    for segment_value, group in df.groupby(segment_column):
        responses = group[response_column].dropna()
        if len(responses) < MIN_SEGMENT_SIZE:
            continue
        counts = responses.value_counts()
        total = int(counts.sum())
        distribution = [
            {"value": str(value), "count": int(count), "percent": round(count / total * 100, 1)}
            for value, count in counts.items()
        ]
        top = distribution[0]
        segments.append({
            "segment": str(segment_value),
            "count": total,
            "distribution": distribution,
            "dominant_value": top["value"],
            "dominant_percent": top["percent"],
        })

    if not segments:
        return {
            "success": False,
            "error": f"No segments of '{segment_column}' have at least {MIN_SEGMENT_SIZE} responses for '{response_column}'.",
        }

    return {
        "success": True,
        "segment_column": segment_column,
        "response_column": response_column,
        "options": options,
        "segments": segments,
    }


def find_top_segment_for_value(
    df: pd.DataFrame, schema: dict[str, Any], segment_column: str, response_column: str, value: str
) -> dict[str, Any]:
    """Which value of `segment_column` has the highest % choosing `value` for
    `response_column`? Unlike `build_segment_response_crosstab`, small segments are
    NOT dropped (a `small_sample_warning` is added instead) so "which city/state has
    the highest..." rankings stay complete."""
    if segment_column not in df.columns:
        return {
            "success": False,
            "error": f"Segment column '{segment_column}' not found.",
            "available_columns": list(df.columns),
        }
    if response_column not in df.columns:
        return {
            "success": False,
            "error": f"Response column '{response_column}' not found.",
            "available_columns": list(df.columns),
        }

    available_values = [str(v) for v in df[response_column].dropna().unique()]
    if value not in available_values:
        return {
            "success": False,
            "error": f"Value '{value}' not found in column '{response_column}'.",
            "available_values": available_values,
        }

    ranking = []
    small_sample = False
    for segment_value, group in df.groupby(segment_column):
        responses = group[response_column].dropna()
        total = len(responses)
        if total == 0:
            continue
        count = int((responses == value).sum())
        ranking.append({"segment": str(segment_value), "count": count, "percent": round(count / total * 100, 1)})
        if total < MIN_SEGMENT_SIZE:
            small_sample = True

    if not ranking:
        return {"success": False, "error": f"No responses found for '{response_column}' grouped by '{segment_column}'."}

    ranking.sort(key=lambda r: r["percent"], reverse=True)
    top = ranking[0]
    result: dict[str, Any] = {
        "success": True,
        "segment_column": segment_column,
        "response_column": response_column,
        "value": value,
        "ranking": ranking,
        "top_segment": top["segment"],
        "top_percent": top["percent"],
        "top_count": top["count"],
    }
    if small_sample:
        result["small_sample_warning"] = (
            f"Some segments have fewer than {MIN_SEGMENT_SIZE} responses; treat their rankings with caution."
        )
    return result


_THRESHOLD_OPS: dict[str, Callable[[pd.Series, float], pd.Series]] = {
    "ge": lambda s, t: s >= t,
    "gt": lambda s, t: s > t,
    "le": lambda s, t: s <= t,
    "lt": lambda s, t: s < t,
    "eq": lambda s, t: s == t,
}


def count_numeric_threshold(df: pd.DataFrame, schema: dict[str, Any], column: str, op: str = "ge", threshold: float = 0) -> dict[str, Any]:
    """How many respondents have `column` <op> `threshold`? e.g. "How many respondents
    selected >=16% in Current Inflation Rate?" (op="ge", threshold=16)."""
    if column not in df.columns:
        return {"success": False, "error": f"Column '{column}' not found.", "available_columns": list(df.columns)}
    if schema.get(column, {}).get("type") not in ("numeric_scale", "numeric_score"):
        return {"success": False, "error": f"Column '{column}' is not numeric."}
    if op not in _THRESHOLD_OPS:
        return {"success": False, "error": f"Unsupported op '{op}'. Use one of {list(_THRESHOLD_OPS)}."}

    series = df[column].dropna()
    total = len(series)
    if total == 0:
        return {"success": False, "error": f"Column '{column}' has no non-null values."}

    count = int(_THRESHOLD_OPS[op](series, threshold).sum())
    return {
        "success": True,
        "column": column,
        "op": op,
        "threshold": threshold,
        "count": count,
        "total": total,
        "percent": round(count / total * 100, 1),
    }


def find_top_segment_for_numeric_threshold(
    df: pd.DataFrame, schema: dict[str, Any], segment_column: str, value_column: str, op: str = "ge", threshold: float = 0
) -> dict[str, Any]:
    """Which value of `segment_column` is most likely to have `value_column` <op>
    `threshold`? e.g. "Which respondent segments are most likely to select >=16% for
    Inflation Rate after 3 Months?" Per-segment version of `count_numeric_threshold`."""
    if segment_column not in df.columns:
        return {
            "success": False,
            "error": f"Segment column '{segment_column}' not found.",
            "available_columns": list(df.columns),
        }
    if value_column not in df.columns:
        return {
            "success": False,
            "error": f"Value column '{value_column}' not found.",
            "available_columns": list(df.columns),
        }
    if schema.get(value_column, {}).get("type") not in ("numeric_scale", "numeric_score"):
        return {"success": False, "error": f"Column '{value_column}' is not numeric."}
    if op not in _THRESHOLD_OPS:
        return {"success": False, "error": f"Unsupported op '{op}'. Use one of {list(_THRESHOLD_OPS)}."}

    ranking = []
    small_sample = False
    for segment_value, group in df.groupby(segment_column):
        series = group[value_column].dropna()
        total = len(series)
        if total == 0:
            continue
        count = int(_THRESHOLD_OPS[op](series, threshold).sum())
        ranking.append({"segment": str(segment_value), "count": count, "total": total, "percent": round(count / total * 100, 1)})
        if total < MIN_SEGMENT_SIZE:
            small_sample = True

    if not ranking:
        return {"success": False, "error": f"No responses found for '{value_column}' grouped by '{segment_column}'."}

    ranking.sort(key=lambda r: r["percent"], reverse=True)
    top = ranking[0]
    result: dict[str, Any] = {
        "success": True,
        "segment_column": segment_column,
        "value_column": value_column,
        "op": op,
        "threshold": threshold,
        "ranking": ranking,
        "top_segment": top["segment"],
        "top_percent": top["percent"],
        "top_count": top["count"],
    }
    if small_sample:
        result["small_sample_warning"] = (
            f"Some segments have fewer than {MIN_SEGMENT_SIZE} responses; treat their rankings with caution."
        )
    return result
