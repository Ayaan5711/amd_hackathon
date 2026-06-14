"""Chat tools for the Survey Analytics pack - "talk to results" over a completed
InvestigationState.

The existing survey analysis tools (`app/tools/*`) operate on `(df, schema, **kwargs)`,
but `app/agent/chat_nodes.py`'s `_tool_node` calls every chat tool as
`tool_func(investigation, **arguments)`. This module wraps those 5 tools into that
`(investigation, **kwargs) -> dict` contract (reconstructing the DataFrame/schema from
`investigation["entries"]` via `reconstruct_df_and_schema`), mirrors
`app/packs/governance/tool_registry.py`'s shape, and reuses governance's
pack-agnostic `get_entry_detail` / `get_risk_distribution` directly.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

import pandas as pd

from app.agent.state import InvestigationState
from app.packs.governance.tool_registry import get_entry_detail, get_risk_distribution
from app.packs.survey.categorical import (
    build_segment_response_crosstab,
    count_numeric_threshold,
    find_top_segment_for_numeric_threshold,
    find_top_segment_for_value,
    get_value_distribution,
    split_demographic_and_response_columns,
)
from app.packs.survey.common import (
    pick_dimension_column,
    pick_excluded_columns,
    pick_metric_columns,
    pick_segment_column,
    reconstruct_df_and_schema,
)
from app.tools import (
    compare_trends,
    extract_open_text_themes,
    flag_anomalies,
    get_segment_stats,
    recommend_actions,
)
from app.tools.registry import TOOL_REGISTRY
from app.utils.csv_loader import get_categorical_columns, get_column_by_type, get_numeric_columns
from app.utils.llm_client import MockFabricator


def _segment_stats_tool(investigation: InvestigationState, **kwargs: Any) -> dict[str, Any]:
    df, schema = reconstruct_df_and_schema(investigation.get("entries", []))
    return get_segment_stats(df, schema, **kwargs)


def _compare_trends_tool(investigation: InvestigationState, **kwargs: Any) -> dict[str, Any]:
    df, schema = reconstruct_df_and_schema(investigation.get("entries", []))
    return compare_trends(df, schema, **kwargs)


def _extract_open_text_themes_tool(investigation: InvestigationState, **kwargs: Any) -> dict[str, Any]:
    df, schema = reconstruct_df_and_schema(investigation.get("entries", []))
    return extract_open_text_themes(df, schema, **kwargs)


def _flag_anomalies_tool(investigation: InvestigationState, **kwargs: Any) -> dict[str, Any]:
    df, schema = reconstruct_df_and_schema(investigation.get("entries", []))
    return flag_anomalies(df, schema, **kwargs)


def _recommend_actions_tool(investigation: InvestigationState, **kwargs: Any) -> dict[str, Any]:
    df, schema = reconstruct_df_and_schema(investigation.get("entries", []))
    return recommend_actions(df, schema, **kwargs)


def _get_value_distribution_tool(investigation: InvestigationState, **kwargs: Any) -> dict[str, Any]:
    df, schema = reconstruct_df_and_schema(investigation.get("entries", []))
    return get_value_distribution(df, schema, **kwargs)


def _get_response_by_segment_tool(investigation: InvestigationState, **kwargs: Any) -> dict[str, Any]:
    df, schema = reconstruct_df_and_schema(investigation.get("entries", []))
    return build_segment_response_crosstab(df, schema, **kwargs)


def _find_top_segment_for_value_tool(investigation: InvestigationState, **kwargs: Any) -> dict[str, Any]:
    df, schema = reconstruct_df_and_schema(investigation.get("entries", []))
    return find_top_segment_for_value(df, schema, **kwargs)


def _count_numeric_threshold_tool(investigation: InvestigationState, **kwargs: Any) -> dict[str, Any]:
    df, schema = reconstruct_df_and_schema(investigation.get("entries", []))
    return count_numeric_threshold(df, schema, **kwargs)


def _find_top_segment_for_numeric_threshold_tool(investigation: InvestigationState, **kwargs: Any) -> dict[str, Any]:
    df, schema = reconstruct_df_and_schema(investigation.get("entries", []))
    return find_top_segment_for_numeric_threshold(df, schema, **kwargs)


SURVEY_TOOL_REGISTRY: list[dict[str, Any]] = [
    *TOOL_REGISTRY,
    {
        "name": "get_entry_detail",
        "description": (
            "Look up one survey response by its row number (e.g. 'response 19', 'row 19', "
            "or '#19'), including all of its answers, whether it was flagged as an outlier "
            "or contains PII, and its risk score."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "log_id": {"type": "string", "description": "The response's row number as a string, e.g. '19'."}
            },
            "required": ["log_id"],
        },
    },
    {
        "name": "get_risk_distribution",
        "description": (
            "Get an overview of how many survey responses were flagged for follow-up vs. "
            "normal, broken down by severity (low/medium/high/critical), plus the overall "
            "risk score for this dataset. Use this for 'how many responses were flagged' or "
            "'give me an overview' style questions."
        ),
        "parameters": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "get_value_distribution",
        "description": (
            "Get the % breakdown of every value in one categorical column (e.g. Gender, "
            "Age Band, or a Likert-style response column), plus which value is most common "
            "('dominant'). Use for 'what percentage of respondents are X' or 'what's the "
            "breakdown of <column>' style questions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "column": {"type": "string", "description": "The categorical column to summarize."}
            },
            "required": ["column"],
        },
    },
    {
        "name": "get_response_by_segment",
        "description": (
            "Cross-tabulate a Likert-style response column against a demographic segment "
            "column (e.g. 'Outlook by Gender'), showing the % choosing each response option "
            "within every segment. Use for 'compare <response> by <demographic>' style "
            "questions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "segment_column": {"type": "string", "description": "The demographic column to group by, e.g. 'Gender' or 'City'."},
                "response_column": {"type": "string", "description": "The Likert-style response column to break down."},
            },
            "required": ["segment_column", "response_column"],
        },
    },
    {
        "name": "find_top_segment_for_value",
        "description": (
            "Find which value of a demographic column has the highest % of respondents "
            "choosing a given response option (e.g. 'which city has the highest \"More "
            "than current\" for Food Prices?')."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "segment_column": {"type": "string", "description": "The demographic column to rank, e.g. 'City' or 'State'."},
                "response_column": {"type": "string", "description": "The response column to look at."},
                "value": {"type": "string", "description": "The response option to rank segments by, e.g. 'More than current'."},
            },
            "required": ["segment_column", "response_column", "value"],
        },
    },
    {
        "name": "count_numeric_threshold",
        "description": (
            "Count how many respondents have a numeric column at/above/below a given "
            "threshold (e.g. 'how many respondents selected >=16% for Inflation Rate?')."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "column": {"type": "string", "description": "The numeric column to threshold."},
                "op": {
                    "type": "string",
                    "enum": ["ge", "gt", "le", "lt", "eq"],
                    "description": "Comparison operator (default 'ge').",
                    "default": "ge",
                },
                "threshold": {"type": "number", "description": "The threshold value."},
            },
            "required": ["column", "threshold"],
        },
    },
    {
        "name": "find_top_segment_for_numeric_threshold",
        "description": (
            "Find which value of a demographic/segment column has the highest % of "
            "respondents at/above/below a numeric threshold (e.g. 'which segment is most "
            "likely to select >=16% for Inflation Rate after 3 Months?')."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "segment_column": {"type": "string", "description": "The demographic/segment column to rank."},
                "value_column": {"type": "string", "description": "The numeric column to threshold."},
                "op": {
                    "type": "string",
                    "enum": ["ge", "gt", "le", "lt", "eq"],
                    "description": "Comparison operator (default 'ge').",
                    "default": "ge",
                },
                "threshold": {"type": "number", "description": "The threshold value."},
            },
            "required": ["segment_column", "value_column", "threshold"],
        },
    },
]


SURVEY_TOOL_FUNCTIONS: dict[str, Callable[..., dict[str, Any]]] = {
    "get_segment_stats": _segment_stats_tool,
    "compare_trends": _compare_trends_tool,
    "extract_open_text_themes": _extract_open_text_themes_tool,
    "flag_anomalies": _flag_anomalies_tool,
    "recommend_actions": _recommend_actions_tool,
    "get_entry_detail": get_entry_detail,
    "get_risk_distribution": get_risk_distribution,
    "get_value_distribution": _get_value_distribution_tool,
    "get_response_by_segment": _get_response_by_segment_tool,
    "find_top_segment_for_value": _find_top_segment_for_value_tool,
    "count_numeric_threshold": _count_numeric_threshold_tool,
    "find_top_segment_for_numeric_threshold": _find_top_segment_for_numeric_threshold_tool,
}


_ROW_PATTERN = re.compile(r"(?:response|row|entry|record)\s*#?\s*(\d+)|#(\d+)", re.IGNORECASE)
_THRESHOLD_PATTERN = re.compile(r"(>=|<=|>|<|=)\s*(\d+(?:\.\d+)?)")
_THRESHOLD_OP_MAP = {">=": "ge", "<=": "le", ">": "gt", "<": "lt", "=": "eq"}


def _find_column_in_message(message: str, columns: list[str]) -> str | None:
    """First column whose name (e.g. 'Outlook_Food_Prices' -> 'outlook food prices')
    appears in the (lowercased) message."""
    for col in columns:
        if col.lower().replace("_", " ") in message:
            return col
    return None


def _find_value_in_message(message: str, df: pd.DataFrame, column: str) -> str | None:
    """First value of `column` that appears in the (lowercased) message."""
    for value in df[column].dropna().unique():
        if str(value).lower() in message:
            return str(value)
    return None


def _find_column_by_value(message: str, df: pd.DataFrame, columns: list[str]) -> str | None:
    """First column in `columns` whose VALUES (not name) appear in the message,
    e.g. 'Male' -> 'Gender'."""
    for col in columns:
        if _find_value_in_message(message, df, col) is not None:
            return col
    return None


def mock_survey_chat_intent(
    user_message: str, tool_registry: list[dict[str, Any]], investigation: InvestigationState
) -> MockFabricator:
    """Content-aware mock intent classifier for the Survey Analytics chat: keyword
    heuristics over the user message + the investigation's own columns pick a tool +
    arguments from `SURVEY_TOOL_REGISTRY`, so LLM_MODE=mock can exercise the
    tool -> synthesis path for segment, trend, theme, anomaly, recommendation, risk
    overview, and specific-response questions."""

    available = {tool["name"] for tool in tool_registry}
    df, schema = reconstruct_df_and_schema(investigation.get("entries", []))
    numeric_cols = get_numeric_columns(schema)
    categorical_cols = get_categorical_columns(schema)
    open_text_cols = get_column_by_type(schema, "open_text")

    dimension_col = pick_dimension_column(categorical_cols)
    metric_cols = pick_metric_columns(numeric_cols)
    segment_col = pick_segment_column(categorical_cols, exclude=dimension_col)
    primary_metric = metric_cols[0] if metric_cols else (numeric_cols[0] if numeric_cols else None)

    exclude = pick_excluded_columns(categorical_cols)
    demographic_cols, response_cols = split_demographic_and_response_columns(df, schema, categorical_cols, exclude=exclude)

    def fabricator(_messages: list[dict[str, str]]) -> dict[str, Any]:
        message = user_message.lower()
        row_match = _ROW_PATTERN.search(message)
        threshold_match = _THRESHOLD_PATTERN.search(message)
        matched_numeric_col = _find_column_in_message(message, numeric_cols)
        matched_demographic_col = _find_column_in_message(message, demographic_cols) or _find_column_by_value(
            message, df, demographic_cols
        )
        matched_response_col = _find_column_in_message(message, response_cols)
        matched_response_value = _find_value_in_message(message, df, matched_response_col) if matched_response_col else None

        tool_calls: list[dict[str, Any]] = []
        if row_match and "get_entry_detail" in available:
            log_id = row_match.group(1) or row_match.group(2)
            tool_calls.append({"tool_name": "get_entry_detail", "arguments": {"log_id": log_id}})
        elif (
            "find_top_segment_for_numeric_threshold" in available
            and threshold_match
            and matched_numeric_col
            and demographic_cols
            and any(w in message for w in ("segment", "group", "which", "demographic", "most likely"))
        ):
            tool_calls.append({
                "tool_name": "find_top_segment_for_numeric_threshold",
                "arguments": {
                    "segment_column": matched_demographic_col or demographic_cols[0],
                    "value_column": matched_numeric_col,
                    "op": _THRESHOLD_OP_MAP[threshold_match.group(1)],
                    "threshold": float(threshold_match.group(2)),
                },
            })
        elif (
            "count_numeric_threshold" in available
            and threshold_match
            and matched_numeric_col
            and any(w in message for w in ("how many", "count", "number of"))
        ):
            tool_calls.append({
                "tool_name": "count_numeric_threshold",
                "arguments": {
                    "column": matched_numeric_col,
                    "op": _THRESHOLD_OP_MAP[threshold_match.group(1)],
                    "threshold": float(threshold_match.group(2)),
                },
            })
        elif (
            "find_top_segment_for_value" in available
            and demographic_cols
            and matched_response_col
            and matched_response_value
            and any(w in message for w in ("which city", "which state", "which segment", "which demographic", "highest", "most"))
        ):
            tool_calls.append({
                "tool_name": "find_top_segment_for_value",
                "arguments": {
                    "segment_column": matched_demographic_col or demographic_cols[0],
                    "response_column": matched_response_col,
                    "value": matched_response_value,
                },
            })
        elif (
            "get_response_by_segment" in available
            and matched_demographic_col
            and response_cols
            and any(w in message for w in ("compare", "by", "across", "breakdown"))
        ):
            tool_calls.append({
                "tool_name": "get_response_by_segment",
                "arguments": {
                    "segment_column": matched_demographic_col,
                    "response_column": matched_response_col or response_cols[0],
                },
            })
        elif (
            "get_value_distribution" in available
            and (matched_response_col or matched_demographic_col)
            and any(w in message for w in ("breakdown", "distribution", "% of", "percentage", "most common", "dominant"))
        ):
            tool_calls.append({
                "tool_name": "get_value_distribution",
                "arguments": {"column": matched_response_col or matched_demographic_col},
            })
        elif "recommend_actions" in available and any(
            w in message for w in ("recommend", "advice", "suggest", "should we", "should i", "improve", "priorit")
        ):
            tool_calls.append({"tool_name": "recommend_actions", "arguments": {}})
        elif "extract_open_text_themes" in available and open_text_cols and any(
            w in message for w in ("theme", "comment", "feedback", "saying", "sentiment", "qualitative")
        ):
            tool_calls.append(
                {"tool_name": "extract_open_text_themes", "arguments": {"text_column": open_text_cols[0]}}
            )
        elif "flag_anomalies" in available and numeric_cols and any(
            w in message for w in ("anomal", "outlier", "unusual", "weird", "strange")
        ):
            tool_calls.append({"tool_name": "flag_anomalies", "arguments": {"columns": numeric_cols}})
        elif "compare_trends" in available and dimension_col and metric_cols and any(
            w in message for w in ("trend", "over time", "quarter", "change", "period", "month", "year")
        ):
            tool_calls.append(
                {"tool_name": "compare_trends", "arguments": {"dimension_column": dimension_col, "metric_columns": metric_cols}}
            )
        elif "get_segment_stats" in available and segment_col and primary_metric and any(
            w in message for w in ("segment", "compare", "department", "team", "group", "by", "vs", "versus", "between")
        ):
            tool_calls.append(
                {"tool_name": "get_segment_stats", "arguments": {"segment_column": segment_col, "metric_column": primary_metric}}
            )
        elif "get_risk_distribution" in available and any(
            w in message for w in ("risk", "overview", "summary", "overall", "how many", "flagged")
        ):
            tool_calls.append({"tool_name": "get_risk_distribution", "arguments": {}})

        return {
            "intent": "tool_use" if tool_calls else "general",
            "reasoning": "Mock mode (LLM_MODE=mock): keyword heuristic over the survey chat message.",
            "tool_calls": tool_calls,
            "clarification_needed": False,
            "clarification_options": [],
        }

    return fabricator
