"""Action recommendation tool - generate actionable insights."""

import json
import logging
from typing import Any

import pandas as pd

from app.config import MAX_TOKENS_SYNTHESIS, VLLM_MODEL_SYNTHESIS
from app.utils.csv_loader import get_categorical_columns, get_numeric_columns
from app.utils.llm_client import call_llm

logger = logging.getLogger(__name__)


def recommend_actions(
    df: pd.DataFrame,
    schema: dict[str, Any],
    focus_area: str | None = None
) -> dict[str, Any]:
    """
    Generate actionable recommendations based on survey data.
    
    Args:
        df: Survey DataFrame
        schema: Column schema
        focus_area: Optional specific focus area
        
    Returns:
        Dictionary with recommendations
    """
    logger.info(f"Generating recommendations, focus: {focus_area or 'general'}")
    
    try:
        # Gather key statistics for context
        numeric_cols = get_numeric_columns(schema)
        categorical_cols = get_categorical_columns(schema)
        
        if not numeric_cols:
            return {
                "success": False,
                "error": "No numeric columns found for analysis"
            }
        
        # Compute key metrics
        key_metrics = []
        for col in numeric_cols[:5]:  # Limit to first 5
            series = df[col].dropna()
            if len(series) > 0:
                key_metrics.append({
                    "name": col,
                    "mean": round(float(series.mean()), 2),
                    "median": round(float(series.median()), 2),
                    "std": round(float(series.std()), 2),
                    "min": round(float(series.min()), 2),
                    "max": round(float(series.max()), 2)
                })
        
        # Find lowest performing areas
        lowest_metrics = sorted(key_metrics, key=lambda x: x["mean"])[:3]
        
        # Segment analysis for recommendations
        segment_insights = []
        if categorical_cols and numeric_cols:
            segment_col = categorical_cols[0]
            metric_col = numeric_cols[0]
            
            grouped = df.groupby(segment_col, observed=True)[metric_col].agg(["mean", "count"])
            grouped = grouped[grouped["count"] >= 10]
            
            if len(grouped) > 1:
                best = grouped["mean"].idxmax()
                worst = grouped["mean"].idxmin()
                gap = grouped["mean"].max() - grouped["mean"].min()
                
                segment_insights.append({
                    "segment_column": segment_col,
                    "metric_column": metric_col,
                    "best_segment": str(best),
                    "worst_segment": str(worst),
                    "gap": round(float(gap), 2)
                })
        
        # Build context for LLM
        context = {
            "total_responses": len(df),
            "key_metrics": key_metrics,
            "lowest_performing": lowest_metrics,
            "segment_insights": segment_insights,
            "focus_area": focus_area
        }
        
        # Generate recommendations via LLM
        prompt = f"""Based on the following survey data analysis, generate 3-5 specific, actionable recommendations.

SURVEY CONTEXT:
- Total responses: {context['total_responses']}
- Key metrics: {json.dumps(context['key_metrics'], indent=2)}
- Lowest performing areas: {json.dumps(context['lowest_performing'], indent=2)}
{f"- Segment analysis: {json.dumps(context['segment_insights'], indent=2)}" if segment_insights else ""}
{f"- Specific focus area: {focus_area}" if focus_area else ""}

TASK:
Generate recommendations that are:
1. Specific and actionable (not generic advice)
2. Prioritized by impact and feasibility
3. Based on the actual data patterns shown above
4. Include expected outcomes if implemented

For each recommendation provide:
- Title (concise action statement)
- Rationale (why this matters based on the data)
- Action steps (specific steps to take)
- Expected impact (high/medium/low)
- Timeline (quick win / short term / long term)

Respond in this exact JSON format:
{{
    "recommendations": [
        {{
            "priority": 1,
            "title": "Specific action to take",
            "rationale": "Why this matters based on data",
            "action_steps": ["Step 1", "Step 2", "Step 3"],
            "expected_impact": "high",
            "timeline": "quick win"
        }}
    ],
    "summary": "Overall strategic summary in 2-3 sentences"
}}"""

        response = call_llm(
            messages=[
                {"role": "system", "content": "You are a strategic survey analyst. Provide specific, actionable recommendations based on data."},
                {"role": "user", "content": prompt}
            ],
            model=VLLM_MODEL_SYNTHESIS,
            max_tokens=MAX_TOKENS_SYNTHESIS,
            json_mode=True,
            response_schema="recommend",
            agent="survey_recommend",
        )
        
        recommendations = json.loads(response)
        
        result = {
            "success": True,
            "focus_area": focus_area,
            "total_responses": len(df),
            "recommendations": recommendations.get("recommendations", []),
            "summary": recommendations.get("summary", ""),
            "data_context": {
                "key_metrics": key_metrics,
                "lowest_performing": lowest_metrics,
                "segment_insights": segment_insights
            }
        }
        
        logger.info(f"Generated {len(result['recommendations'])} recommendations")
        return result
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse recommendations JSON: {e}")
        return {
            "success": False,
            "error": "Failed to parse recommendations"
        }
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        return {
            "success": False,
            "error": f"Recommendation generation failed: {str(e)}"
        }
