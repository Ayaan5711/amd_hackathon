"""Trend comparison tool - compare metrics across time/dimensions."""

import logging
from typing import Any

import pandas as pd

from app.utils.csv_loader import get_numeric_columns

logger = logging.getLogger(__name__)


def compare_trends(
    df: pd.DataFrame,
    schema: dict[str, Any],
    dimension_column: str,
    metric_columns: list[str]
) -> dict[str, Any]:
    """
    Compare metrics across time periods or dimensions.
    
    Args:
        df: Survey DataFrame
        schema: Column schema
        dimension_column: Column defining groups (e.g., 'Quarter', 'Year')
        metric_columns: List of numeric columns to compare
        
    Returns:
        Dictionary with comparison results
    """
    logger.info(f"Comparing trends: {metric_columns} by {dimension_column}")
    
    # Validate dimension column
    if dimension_column not in df.columns:
        return {
            "success": False,
            "error": f"Dimension column '{dimension_column}' not found",
            "available_columns": list(df.columns)
        }
    
    # Filter to valid numeric columns
    valid_metrics = []
    invalid_metrics = []
    
    for col in metric_columns:
        if col in df.columns:
            col_type = schema.get(col, {}).get("type")
            if col_type in ("numeric_scale", "numeric_score"):
                valid_metrics.append(col)
            else:
                invalid_metrics.append(f"{col} (not numeric)")
        else:
            invalid_metrics.append(f"{col} (not found)")
    
    if not valid_metrics:
        return {
            "success": False,
            "error": "No valid numeric columns provided",
            "invalid_columns": invalid_metrics,
            "available_numeric": get_numeric_columns(schema)
        }
    
    try:
        # Get unique dimension values, sorted
        dim_values = df[dimension_column].dropna().unique()
        
        # Try to sort numerically if possible
        try:
            dim_values = sorted(dim_values, key=lambda x: pd.to_numeric(x, errors="ignore"))
        except Exception:
            dim_values = sorted(dim_values)
        
        if len(dim_values) < 2:
            return {
                "success": False,
                "error": f"Need at least 2 groups in '{dimension_column}', found {len(dim_values)}"
            }
        
        # Compute stats for each dimension value
        comparison_data = []
        
        for dim_val in dim_values:
            subset = df[df[dimension_column] == dim_val]
            row_data = {
                "dimension_value": str(dim_val),
                "count": len(subset)
            }
            
            for metric in valid_metrics:
                mean_val = subset[metric].mean()
                row_data[f"{metric}_mean"] = round(float(mean_val), 2) if pd.notna(mean_val) else None
            
            comparison_data.append(row_data)
        
        # Calculate changes between consecutive periods
        changes = []
        for i in range(1, len(comparison_data)):
            prev = comparison_data[i - 1]
            curr = comparison_data[i]
            
            change_record = {
                "from": prev["dimension_value"],
                "to": curr["dimension_value"]
            }
            
            for metric in valid_metrics:
                prev_val = prev.get(f"{metric}_mean")
                curr_val = curr.get(f"{metric}_mean")
                
                if prev_val is not None and curr_val is not None:
                    abs_change = round(curr_val - prev_val, 2)
                    pct_change = round((curr_val - prev_val) / prev_val * 100, 1) if prev_val != 0 else None
                    change_record[f"{metric}_change"] = abs_change
                    change_record[f"{metric}_pct_change"] = pct_change
            
            changes.append(change_record)
        
        # Find significant changes
        significant_changes = []
        for change in changes:
            for metric in valid_metrics:
                change_key = f"{metric}_change"
                if change_key in change and abs(change[change_key]) >= 0.5:
                    significant_changes.append({
                        "metric": metric,
                        "period": f"{change['from']} → {change['to']}",
                        "change": change[change_key],
                        "pct_change": change.get(f"{metric}_pct_change")
                    })
        
        # Sort by absolute change
        significant_changes.sort(key=lambda x: abs(x["change"]), reverse=True)
        
        result = {
            "success": True,
            "dimension_column": dimension_column,
            "metric_columns": valid_metrics,
            "invalid_columns": invalid_metrics if invalid_metrics else None,
            "periods": [d["dimension_value"] for d in comparison_data],
            "period_count": len(comparison_data),
            "data": comparison_data,
            "changes": changes,
            "significant_changes": significant_changes[:5]  # Top 5
        }
        
        logger.info(f"Trend comparison complete: {len(comparison_data)} periods, {len(valid_metrics)} metrics")
        return result
        
    except Exception as e:
        logger.error(f"Error comparing trends: {e}")
        return {
            "success": False,
            "error": f"Failed to compare trends: {str(e)}"
        }
