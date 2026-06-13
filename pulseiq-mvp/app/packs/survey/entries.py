"""Survey row -> LogEntry mapping for the Survey Analytics pack.

Each response row becomes one LogEntry: the full row is serialized as JSON
into `ai_response` so the existing Presidio PII scanner (which scans
`user_prompt`/`ai_response`) can flag PII leaking into open-text survey
fields (e.g. a respondent naming a colleague in a comment). Any numeric
columns that are statistical outliers for this row are recorded as JSON in
`retrieved_context` - repurposed here as "why this row might need review"
rather than RAG context.
"""

from __future__ import annotations

import json

import pandas as pd

from app.agent.state import LogEntry
from app.config import ANOMALY_Z_THRESHOLD


def _row_outliers(df: pd.DataFrame) -> list[dict[str, dict[str, float]]]:
    """Per-row {column: {value, mean, z}} for numeric columns where |z| > ANOMALY_Z_THRESHOLD."""
    numeric_cols = df.select_dtypes(include="number").columns
    means = df[numeric_cols].mean()
    stds = df[numeric_cols].std(ddof=0)

    outliers: list[dict[str, dict[str, float]]] = []
    for _, row in df.iterrows():
        row_outliers: dict[str, dict[str, float]] = {}
        for col in numeric_cols:
            std = stds[col]
            if not std or pd.isna(row[col]):
                continue
            z = (row[col] - means[col]) / std
            if abs(z) > ANOMALY_Z_THRESHOLD:
                row_outliers[col] = {
                    "value": float(row[col]),
                    "mean": round(float(means[col]), 2),
                    "z": round(float(z), 2),
                }
        outliers.append(row_outliers)
    return outliers


def survey_entries_fn(df: pd.DataFrame) -> list[LogEntry]:
    """Convert a survey response DataFrame into LogEntry-shaped dicts.

    `log_id` is the row's positional index (as a string).
    """
    outliers = _row_outliers(df)
    entries: list[LogEntry] = []
    for i, record in enumerate(df.to_dict(orient="records")):
        clean_record = {k: (None if pd.isna(v) else v) for k, v in record.items()}
        entries.append({
            "log_id": str(i),
            "timestamp": "",
            "user_prompt": "",
            "ai_response": json.dumps(clean_record, default=str),
            "retrieved_context": json.dumps(outliers[i]) if outliers[i] else None,
            "model_name": None,
        })
    return entries
