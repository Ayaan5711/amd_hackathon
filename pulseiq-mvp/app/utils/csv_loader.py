"""CSV loading and schema detection utilities."""

import json
import logging
from io import BytesIO
from typing import Any

import chardet
import pandas as pd

from app.config import (
    INFERRED_TYPES,
    LOG_OPTIONAL_COLUMNS,
    LOG_REQUIRED_COLUMNS,
    MAX_CSV_MB,
    MAX_CSV_ROWS,
)

logger = logging.getLogger(__name__)


def detect_column_type(series: pd.Series) -> str:
    """
    Infer survey column type from values.
    
    Detects:
    - numeric_scale: Likert/rating scale (1-5 or 1-10)
    - numeric_score: Continuous score
    - categorical: Category/group column
    - open_text: Free-text response
    - boolean: Yes/No or True/False
    - datetime: Date or timestamp
    """
    # Check if object/string column is actually numeric stored as string.
    # pandas >= 3.0 infers a dedicated StringDtype (repr "str") for string
    # columns instead of "object" - is_string_dtype() catches both.
    if series.dtype == object or pd.api.types.is_string_dtype(series):
        coerced = pd.to_numeric(series.dropna(), errors="coerce")
        if coerced.notna().sum() / max(series.dropna().shape[0], 1) > 0.85:
            series = coerced
        else:
            unique_ratio = series.nunique() / max(len(series.dropna()), 1)
            avg_len = series.dropna().str.len().mean() if len(series.dropna()) > 0 else 0
            
            # Long text or high uniqueness suggests open text
            if avg_len > 30 or unique_ratio > 0.6:
                return "open_text"
            
            # Check for boolean values
            lower_vals = series.dropna().str.lower().unique()
            if set(lower_vals).issubset({"yes", "no", "true", "false", "y", "n", "1", "0"}):
                return "boolean"
            
            return "categorical"

    # Datetime detection
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    # Numeric detection
    if pd.api.types.is_numeric_dtype(series):
        mn, mx = series.min(), series.max()
        unique_count = series.nunique()
        # Likert scale detection: small range, integers, within typical scale bounds
        if unique_count <= 10 and 1 <= mn and mx <= 10:
            return "numeric_scale"
        return "numeric_score"

    return "categorical"


def load_csv(file_bytes: bytes, filename: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Load CSV bytes into DataFrame with schema detection.
    
    Args:
        file_bytes: Raw bytes of the CSV file
        filename: Original filename for error messages
        
    Returns:
        Tuple of (DataFrame, schema_dict)
        schema_dict: { col_name: { type, sample_values, n_unique, null_pct } }
        
    Raises:
        ValueError: If file is too large, has too many rows, or invalid format
    """
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_CSV_MB:
        raise ValueError(
            f"File too large: {size_mb:.1f}MB (max {MAX_CSV_MB}MB). "
            "Please upload a smaller file or reduce the number of rows."
        )

    # Detect encoding
    detection = chardet.detect(file_bytes[:10000])
    encoding = detection.get("encoding", "utf-8") or "utf-8"
    confidence = detection.get("confidence", 0)
    logger.info(f"Detected encoding: {encoding} (confidence: {confidence:.2f})")

    try:
        df = pd.read_csv(BytesIO(file_bytes), encoding=encoding, low_memory=False)
    except UnicodeDecodeError:
        # Fallback to utf-8 with error handling
        logger.warning("Encoding detection failed, trying utf-8 with replacement")
        df = pd.read_csv(BytesIO(file_bytes), encoding="utf-8", errors="replace", low_memory=False)
    except Exception as e:
        raise ValueError(f"Failed to parse CSV: {str(e)}")

    if len(df) > MAX_CSV_ROWS:
        raise ValueError(
            f"Too many rows: {len(df)} (max {MAX_CSV_ROWS}). "
            "Please upload a smaller file."
        )

    if len(df) == 0:
        raise ValueError("CSV file is empty")

    if len(df.columns) < 2:
        raise ValueError("CSV must have at least 2 columns")

    # Strip whitespace from column names
    df.columns = [str(c).strip() for c in df.columns]
    
    # Remove completely empty rows and columns
    df = df.dropna(how="all")
    df = df.loc[:, df.columns.notna()]

    # Build schema
    schema: dict[str, Any] = {}
    for col in df.columns:
        col_type = detect_column_type(df[col])
        sample = df[col].dropna().head(5).tolist()
        schema[col] = {
            "type": col_type,
            "type_description": INFERRED_TYPES.get(col_type, col_type),
            "n_unique": int(df[col].nunique()),
            "null_pct": round(df[col].isnull().mean() * 100, 1),
            "sample_values": [str(v) for v in sample],
        }

    logger.info(f"Loaded CSV '{filename}': {len(df)} rows, {len(df.columns)} columns")
    schema_summary = [f"{k}({v['type']})" for k, v in schema.items()]
    logger.info(f"Schema detected: {schema_summary}")

    return df, schema


def get_column_by_type(schema: dict[str, Any], col_type: str) -> list[str]:
    """Get column names matching a specific type."""
    return [col for col, info in schema.items() if info.get("type") == col_type]


def get_numeric_columns(schema: dict[str, Any]) -> list[str]:
    """Get all numeric column names (both scale and score types)."""
    return [
        col for col, info in schema.items()
        if info.get("type") in ("numeric_scale", "numeric_score")
    ]


def get_categorical_columns(schema: dict[str, Any]) -> list[str]:
    """Get categorical column names suitable for segmentation."""
    return [
        col for col, info in schema.items()
        if info.get("type") in ("categorical", "boolean")
        and info.get("n_unique", 0) <= 50
    ]


def load_log_batch(file_bytes: bytes, filename: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Load an AI-interaction-log batch (CSV or JSON records) for the Governance pack.

    Required columns: log_id, timestamp, user_prompt, ai_response
    Optional columns: retrieved_context, model_name (added as all-null if absent)

    Args:
        file_bytes: Raw bytes of the uploaded file
        filename: Original filename (used to pick CSV vs JSON parsing)

    Returns:
        Tuple of (DataFrame, schema_dict)

    Raises:
        ValueError: If the file is too large/empty or missing required columns
    """
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > MAX_CSV_MB:
        raise ValueError(f"File too large: {size_mb:.1f}MB (max {MAX_CSV_MB}MB).")

    if filename.lower().endswith(".json"):
        try:
            records = json.loads(file_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise ValueError(f"Failed to parse JSON log batch: {e}")

        if isinstance(records, dict):
            records = records.get("logs") or records.get("entries") or [records]
        if not isinstance(records, list):
            raise ValueError("JSON log batch must be a list of records (or {'logs': [...]})")

        df = pd.DataFrame.from_records(records)
    else:
        detection = chardet.detect(file_bytes[:10000])
        encoding = detection.get("encoding", "utf-8") or "utf-8"
        try:
            df = pd.read_csv(BytesIO(file_bytes), encoding=encoding, low_memory=False)
        except UnicodeDecodeError:
            df = pd.read_csv(BytesIO(file_bytes), encoding="utf-8", encoding_errors="replace", low_memory=False)
        except Exception as e:
            raise ValueError(f"Failed to parse CSV log batch: {e}")

    if len(df) == 0:
        raise ValueError("Log batch is empty")
    if len(df) > MAX_CSV_ROWS:
        raise ValueError(f"Too many rows: {len(df)} (max {MAX_CSV_ROWS}).")

    df.columns = [str(c).strip() for c in df.columns]

    missing = [c for c in LOG_REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Log batch is missing required columns: {', '.join(missing)}. "
            f"Required: {', '.join(LOG_REQUIRED_COLUMNS)}"
        )

    for col in LOG_OPTIONAL_COLUMNS:
        if col not in df.columns:
            df[col] = None

    df["log_id"] = df["log_id"].astype(str)

    schema: dict[str, Any] = {
        "columns": list(df.columns),
        "row_count": len(df),
        "has_retrieved_context": bool(df["retrieved_context"].notna().any()),
    }

    logger.info(f"Loaded log batch '{filename}': {len(df)} entries, columns={list(df.columns)}")
    return df, schema


def df_to_log_entries(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a log-batch DataFrame (from load_log_batch) into LogEntry-shaped dicts."""
    entries: list[dict[str, Any]] = []
    for record in df.to_dict(orient="records"):
        entries.append({
            "log_id": str(record.get("log_id")),
            "timestamp": str(record.get("timestamp")) if pd.notna(record.get("timestamp")) else "",
            "user_prompt": record.get("user_prompt") or "",
            "ai_response": record.get("ai_response") or "",
            "retrieved_context": record.get("retrieved_context") if pd.notna(record.get("retrieved_context")) else None,
            "model_name": record.get("model_name") if pd.notna(record.get("model_name")) else None,
        })
    return entries
