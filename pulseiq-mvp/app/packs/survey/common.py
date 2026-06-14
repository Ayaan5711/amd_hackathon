"""Shared helpers for the Survey Analytics pack.

Every report section, chat tool, and the dashboard need to go back from the
investigation's `entries` (each entry's `ai_response` is the JSON-serialized
survey row) to a pandas DataFrame + a minimal schema compatible with
`app/utils/csv_loader.py`'s `get_numeric_columns` / `get_categorical_columns`
and the existing survey tools (`app/tools/*`).
"""

from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd

from app.agent.state import LogEntry
from app.packs.survey.categorical import _DEMOGRAPHIC_PATTERN
from app.utils.csv_loader import detect_column_type


def reconstruct_df_and_schema(entries: list[LogEntry]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Rebuild the survey DataFrame + a minimal schema from investigation entries."""
    rows = [json.loads(e["ai_response"]) for e in entries]
    df = pd.DataFrame(rows)
    schema = {
        col: {"type": detect_column_type(df[col]), "n_unique": int(df[col].nunique())}
        for col in df.columns
    }
    return df, schema


_TIME_DIMENSION_PATTERN = re.compile(r"quarter|period|year|month|week|date|time", re.IGNORECASE)
_PREFERRED_METRIC_PATTERN = re.compile(r"satisf|nps|engage|score|rating", re.IGNORECASE)


def pick_dimension_column(categorical_cols: list[str]) -> str | None:
    """Pick the first categorical column that looks like a time/period dimension."""
    for col in categorical_cols:
        if _TIME_DIMENSION_PATTERN.search(col):
            return col
    return None


def pick_metric_columns(numeric_cols: list[str]) -> list[str]:
    """Prefer satisfaction/score/rating-style metrics; fall back to all numeric columns."""
    preferred = [c for c in numeric_cols if _PREFERRED_METRIC_PATTERN.search(c)]
    return preferred or numeric_cols


def pick_segment_column(categorical_cols: list[str], exclude: str | None = None) -> str | None:
    """Pick the first categorical column that isn't the time dimension."""
    for col in categorical_cols:
        if col != exclude:
            return col
    return None


def pick_excluded_columns(categorical_cols: list[str]) -> set[str]:
    """Columns claimed by the existing numeric-segment path (Department/Quarter-style)
    that must NOT be reclassified as demographic/response columns.

    - `pick_dimension_column`'s result is always excluded.
    - `pick_segment_column`'s result (excluding the dimension column) is excluded
      only if it does NOT look like a demographic column (`_DEMOGRAPHIC_PATTERN`).
      This keeps Department/Quarter-style columns out of the new classification,
      while letting a demographic column (e.g. Gender) stay classified as
      demographic even if it would otherwise be picked as the segment column.
    """
    dimension_col = pick_dimension_column(categorical_cols)
    segment_col = pick_segment_column(categorical_cols, exclude=dimension_col)
    excluded = {dimension_col} if dimension_col else set()
    if segment_col and not _DEMOGRAPHIC_PATTERN.search(segment_col):
        excluded.add(segment_col)
    return excluded
