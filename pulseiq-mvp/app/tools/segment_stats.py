"""Segment statistics tool - compute averages by segment."""

import logging
from typing import Any

import pandas as pd

from app.config import MIN_SEGMENT_SIZE
from app.utils.csv_loader import get_categorical_columns, get_numeric_columns

logger = logging.getLogger(__name__)


def get_segment_stats(
    df: pd.DataFrame,
    schema: dict[str, Any],
    segment_column: str,
    metric_column: str
) -> dict[str, Any]:
    """
    Compute average scores and statistics grouped by segment.
    
    Args:
        df: Survey DataFrame
        schema: Column schema
        segment_column: Column to group by (e.g., 'Department')
        metric_column: Numeric column to aggregate (e.g., 'Satisfaction_Score')
        
    Returns:
        Dictionary with segment statistics
    """
    logger.info(f"Computing segment stats: {metric_column} by {segment_column}")
    
    # Validate columns exist
    if segment_column not in df.columns:
        return {
            "success": False,
            "error": f"Segment column '{segment_column}' not found in data",
            "available_segments": get_categorical_columns(schema)
        }
    
    if metric_column not in df.columns:
        return {
            "success": False,
            "error": f"Metric column '{metric_column}' not found in data",
            "available_metrics": get_numeric_columns(schema)
        }
    
    # Validate column types
    segment_info = schema.get(segment_column, {})
    metric_info = schema.get(metric_column, {})
    
    if metric_info.get("type") not in ("numeric_scale", "numeric_score"):
        return {
            "success": False,
            "error": f"Column '{metric_column}' is not numeric (type: {metric_info.get('type')})"
        }
    
    # Compute statistics
    try:
        # Group by segment
        grouped = df.groupby(segment_column, observed=True)[metric_column].agg([
            "count", "mean", "std", "min", "max"
        ]).reset_index()
        
        # Filter small segments
        grouped = grouped[grouped["count"] >= MIN_SEGMENT_SIZE]
        
        if len(grouped) == 0:
            return {
                "success": False,
                "error": f"No segments with at least {MIN_SEGMENT_SIZE} responses"
            }
        
        # Sort by mean (descending)
        grouped = grouped.sort_values("mean", ascending=False)
        
        # Calculate overall stats
        overall_mean = df[metric_column].mean()
        overall_std = df[metric_column].std()
        
        # Format results
        segments = []
        for _, row in grouped.iterrows():
            segments.append({
                "segment": str(row[segment_column]),
                "count": int(row["count"]),
                "mean": round(float(row["mean"]), 2),
                "std": round(float(row["std"]), 2) if pd.notna(row["std"]) else 0,
                "min": round(float(row["min"]), 2),
                "max": round(float(row["max"]), 2),
                "vs_overall": round(float(row["mean"] - overall_mean), 2)
            })
        
        # Find best and worst
        best = max(segments, key=lambda x: x["mean"])
        worst = min(segments, key=lambda x: x["mean"])
        
        result = {
            "success": True,
            "segment_column": segment_column,
            "metric_column": metric_column,
            "overall_mean": round(float(overall_mean), 2),
            "overall_std": round(float(overall_std), 2) if pd.notna(overall_std) else 0,
            "segment_count": len(segments),
            "total_responses": int(df[segment_column].notna().sum()),
            "best_segment": best["segment"],
            "worst_segment": worst["segment"],
            "gap": round(best["mean"] - worst["mean"], 2),
            "segments": segments
        }
        
        logger.info(f"Segment stats computed: {len(segments)} segments, gap={result['gap']}")
        return result
        
    except Exception as e:
        logger.error(f"Error computing segment stats: {e}")
        return {
            "success": False,
            "error": f"Failed to compute statistics: {str(e)}"
        }
