"""Open text theme extraction tool with LLM-powered analysis."""

import json
import logging
from typing import Any

import pandas as pd

from app.config import MAX_OPEN_TEXT_SAMPLE, MAX_THEME_COUNT, MAX_TOKENS_THEMES, VLLM_MODEL_THEMES
from app.utils.csv_loader import get_column_by_type
from app.utils.llm_client import call_llm

logger = logging.getLogger(__name__)


def extract_open_text_themes(
    df: pd.DataFrame,
    schema: dict[str, Any],
    text_column: str,
    max_themes: int = MAX_THEME_COUNT
) -> dict[str, Any]:
    """
    Extract themes and sentiment from open-text responses using LLM.
    
    Args:
        df: Survey DataFrame
        schema: Column schema
        text_column: Column containing open-text responses
        max_themes: Maximum number of themes to extract
        
    Returns:
        Dictionary with themes, sentiment analysis, and examples
    """
    logger.info(f"Extracting themes from column: {text_column}")
    
    # Validate column
    if text_column not in df.columns:
        return {
            "success": False,
            "error": f"Text column '{text_column}' not found",
            "available_text_columns": get_column_by_type(schema, "open_text")
        }
    
    col_type = schema.get(text_column, {}).get("type")
    if col_type != "open_text":
        logger.warning(f"Column '{text_column}' is type '{col_type}', not 'open_text'")
    
    # Get non-empty responses
    responses = df[text_column].dropna().astype(str)
    responses = responses[responses.str.len() > 3]  # Filter very short responses
    
    total_responses = len(responses)
    
    if total_responses == 0:
        return {
            "success": False,
            "error": f"No valid text responses found in '{text_column}'"
        }
    
    # Sample if too many responses
    if len(responses) > MAX_OPEN_TEXT_SAMPLE:
        sample = responses.sample(n=MAX_OPEN_TEXT_SAMPLE, random_state=42)
        is_sampled = True
    else:
        sample = responses
        is_sampled = False
    
    # Prepare text for LLM
    text_sample = "\n---\n".join(sample.tolist()[:MAX_OPEN_TEXT_SAMPLE])
    
    # Build prompt
    prompt = f"""Analyze the following survey responses and extract key themes.

TASK:
1. Identify the top {max_themes} themes or topics mentioned
2. For each theme, provide:
   - Theme name (concise, 2-4 words)
   - Description (what the theme is about)
   - Frequency (approximate percentage of responses mentioning this)
   - Sentiment (positive, negative, or mixed)
   - 2-3 example quotes that represent this theme

3. Provide an overall sentiment distribution (positive/neutral/negative percentages)

RESPONSES TO ANALYZE:
{text_sample[:8000]}  # Truncate if too long

Respond in this exact JSON format:
{{
    "themes": [
        {{
            "theme": "Theme Name",
            "description": "Description of what this theme covers",
            "frequency_percent": 35,
            "sentiment": "negative",
            "examples": ["Example quote 1", "Example quote 2"]
        }}
    ],
    "overall_sentiment": {{
        "positive_percent": 30,
        "neutral_percent": 40,
        "negative_percent": 30
    }},
    "summary": "Brief 2-3 sentence summary of key findings"
}}"""

    try:
        # Call LLM for theme extraction
        response = call_llm(
            messages=[
                {"role": "system", "content": "You are a survey analysis expert. Extract themes from text responses accurately."},
                {"role": "user", "content": prompt}
            ],
            model=VLLM_MODEL_THEMES,
            max_tokens=MAX_TOKENS_THEMES,
            json_mode=True,
            response_schema="themes",
            agent="survey_themes",
        )
        
        # Parse JSON response
        analysis = json.loads(response)
        
        # Validate and limit themes
        themes = analysis.get("themes", [])[:max_themes]
        
        # Calculate basic stats
        avg_length = responses.str.len().mean()
        response_count = len(responses)
        
        result = {
            "success": True,
            "text_column": text_column,
            "total_responses": total_responses,
            "analyzed_responses": len(sample),
            "is_sampled": is_sampled,
            "avg_response_length": round(float(avg_length), 1),
            "themes": themes,
            "overall_sentiment": analysis.get("overall_sentiment", {}),
            "summary": analysis.get("summary", "")
        }
        
        logger.info(f"Theme extraction complete: {len(themes)} themes from {len(sample)} responses")
        return result
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse LLM response as JSON: {e}")
        return {
            "success": False,
            "error": "Failed to parse theme analysis",
            "raw_response": response[:500] if 'response' in locals() else None
        }
    except Exception as e:
        logger.error(f"Error extracting themes: {e}")
        return {
            "success": False,
            "error": f"Theme extraction failed: {str(e)}"
        }
