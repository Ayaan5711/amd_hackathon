"""Anomaly detection tool - identify statistical outliers."""

import logging
from typing import Any

import numpy as np
import pandas as pd

from app.config import ANOMALY_Z_THRESHOLD
from app.utils.csv_loader import get_numeric_columns

logger = logging.getLogger(__name__)


def flag_anomalies(
    df: pd.DataFrame,
    schema: dict[str, Any],
    columns: list[str],
    z_threshold: float = ANOMALY_Z_THRESHOLD
) -> dict[str, Any]:
    """
    Identify statistical outliers and anomalies in numeric columns.
    
    Args:
        df: Survey DataFrame
        schema: Column schema
        columns: List of numeric columns to check
        z_threshold: Z-score threshold for flagging (default: 2.0)
        
    Returns:
        Dictionary with anomaly detection results
    """
    logger.info(f"Flagging anomalies in columns: {columns} (threshold: {z_threshold})")
    
    # Validate columns
    valid_columns = []
    invalid_columns = []
    
    for col in columns:
        if col in df.columns:
            col_type = schema.get(col, {}).get("type")
            if col_type in ("numeric_scale", "numeric_score"):
                valid_columns.append(col)
            else:
                invalid_columns.append(f"{col} (not numeric)")
        else:
            invalid_columns.append(f"{col} (not found)")
    
    if not valid_columns:
        return {
            "success": False,
            "error": "No valid numeric columns provided",
            "invalid_columns": invalid_columns,
            "available_numeric": get_numeric_columns(schema)
        }
    
    try:
        anomalies_by_column = []
        total_anomalies = 0
        
        for col in valid_columns:
            series = df[col].dropna()
            
            if len(series) < 10:
                continue
            
            mean = series.mean()
            std = series.std()
            
            if std == 0:
                continue
            
            # Calculate z-scores
            z_scores = np.abs((series - mean) / std)
            
            # Find outliers
            outliers = series[z_scores > z_threshold]
            
            if len(outliers) > 0:
                # Get outlier details
                outlier_indices = outliers.index.tolist()
                outlier_values = outliers.tolist()
                
                # Calculate percentiles
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                
                # IQR method outliers (more robust)
                iqr_outliers = series[
                    (series < (q1 - 1.5 * iqr)) | (series > (q3 + 1.5 * iqr))
                ]
                
                anomaly_record = {
                    "column": col,
                    "mean": round(float(mean), 2),
                    "std": round(float(std), 2),
                    "min": round(float(series.min()), 2),
                    "max": round(float(series.max()), 2),
                    "outlier_count": len(outliers),
                    "outlier_percent": round(len(outliers) / len(series) * 100, 1),
                    "z_threshold": z_threshold,
                    "extreme_values": [
                        {
                            "value": round(float(v), 2),
                            "z_score": round(float(z_scores[idx]), 2),
                            "direction": "high" if v > mean else "low"
                        }
                        for idx, v in outliers.nlargest(5).items()
                    ],
                    "low_values": [
                        round(float(v), 2)
                        for v in outliers.nsmallest(3).values
                    ] if len(outliers) > 5 else [],
                    "iqr_outliers": len(iqr_outliers)
                }
                
                anomalies_by_column.append(anomaly_record)
                total_anomalies += len(outliers)
        
        # Data quality checks
        quality_issues = []
        
        for col in valid_columns:
            series = df[col]
            null_pct = series.isnull().mean() * 100
            
            if null_pct > 20:
                quality_issues.append({
                    "column": col,
                    "issue": "high_missing",
                    "details": f"{null_pct:.1f}% missing values"
                })
            
            # Check for suspicious values (all same value)
            if series.nunique() == 1:
                quality_issues.append({
                    "column": col,
                    "issue": "constant_values",
                    "details": f"All values are {series.iloc[0]}"
                })
            
            # Check for potential data entry errors (negative scores where unexpected)
            if series.min() < 0 and schema.get(col, {}).get("type") == "numeric_scale":
                negative_count = (series < 0).sum()
                if negative_count > 0:
                    quality_issues.append({
                        "column": col,
                        "issue": "negative_values",
                        "details": f"{negative_count} negative values found"
                    })
        
        result = {
            "success": True,
            "columns_analyzed": valid_columns,
            "invalid_columns": invalid_columns if invalid_columns else None,
            "z_threshold": z_threshold,
            "total_anomalies_found": total_anomalies,
            "anomalies_by_column": anomalies_by_column,
            "columns_with_anomalies": len(anomalies_by_column),
            "data_quality_issues": quality_issues if quality_issues else None,
            "summary": _generate_anomaly_summary(anomalies_by_column, quality_issues)
        }
        
        logger.info(f"Anomaly detection complete: {total_anomalies} anomalies in {len(anomalies_by_column)} columns")
        return result
        
    except Exception as e:
        logger.error(f"Error detecting anomalies: {e}")
        return {
            "success": False,
            "error": f"Anomaly detection failed: {str(e)}"
        }


def _generate_anomaly_summary(
    anomalies: list[dict[str, Any]],
    quality_issues: list[dict[str, Any]]
) -> str:
    """Generate a human-readable summary of anomalies."""
    if not anomalies and not quality_issues:
        return "No significant anomalies or data quality issues detected."
    
    parts = []
    
    if anomalies:
        total_outliers = sum(a["outlier_count"] for a in anomalies)
        parts.append(f"Found {total_outliers} statistical outliers across {len(anomalies)} columns.")
        
        # Mention most extreme
        if anomalies:
            most_extreme = max(anomalies, key=lambda x: x["outlier_percent"])
            parts.append(
                f"Highest outlier rate in '{most_extreme['column']}' "
                f"({most_extreme['outlier_percent']}% of values)."
            )
    
    if quality_issues:
        parts.append(f"Found {len(quality_issues)} data quality issues.")
    
    return " ".join(parts)
