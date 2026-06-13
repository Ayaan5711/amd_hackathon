"""Tool registry - single source of truth for all tools."""

from typing import Any

# MCP-compatible tool registry
# Format mirrors OpenAI function-calling / MCP tool spec
TOOL_REGISTRY: list[dict[str, Any]] = [
    {
        "name": "get_segment_stats",
        "description": (
            "Compute average scores and statistics for a numeric column "
            "grouped by a categorical segment column. Use this when the user asks "
            "about differences between groups, departments, regions, or any segment."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "segment_column": {
                    "type": "string",
                    "description": "The categorical column to group by (e.g., 'Department', 'Region')"
                },
                "metric_column": {
                    "type": "string",
                    "description": "The numeric column to aggregate (e.g., 'Satisfaction_Score', 'NPS')"
                }
            },
            "required": ["segment_column", "metric_column"]
        }
    },
    {
        "name": "compare_trends",
        "description": (
            "Compare metrics across time periods or dimensions. "
            "Use this when the user asks about changes over time, before/after comparisons, "
            "or comparing two different groups."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "dimension_column": {
                    "type": "string",
                    "description": "Column defining the groups to compare (e.g., 'Quarter', 'Year', 'Period')"
                },
                "metric_columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of numeric columns to compare"
                }
            },
            "required": ["dimension_column", "metric_columns"]
        }
    },
    {
        "name": "extract_open_text_themes",
        "description": (
            "Extract themes and sentiment from open-text responses. "
            "Use this when the user asks about comments, feedback, suggestions, "
            "or wants to understand what people are saying in free-text fields."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "text_column": {
                    "type": "string",
                    "description": "The column containing open-text responses"
                },
                "max_themes": {
                    "type": "integer",
                    "description": "Maximum number of themes to extract (default: 8)",
                    "default": 8
                }
            },
            "required": ["text_column"]
        }
    },
    {
        "name": "flag_anomalies",
        "description": (
            "Identify statistical outliers and anomalies in the data. "
            "Use this when the user asks about unusual patterns, outliers, "
            "unexpected values, or wants to find problems in the data."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of numeric columns to check for anomalies"
                },
                "z_threshold": {
                    "type": "number",
                    "description": "Z-score threshold for flagging outliers (default: 2.0)",
                    "default": 2.0
                }
            },
            "required": ["columns"]
        }
    },
    {
        "name": "recommend_actions",
        "description": (
            "Generate actionable recommendations based on survey insights. "
            "Use this when the user asks what they should do, for advice, "
            "or wants recommendations to improve scores."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "focus_area": {
                    "type": "string",
                    "description": "Optional focus area (e.g., 'lowest scoring department', 'NPS improvement')",
                    "default": None
                }
            },
            "required": []
        }
    }
]


def get_tool_descriptions() -> str:
    """Get formatted tool descriptions for LLM prompts."""
    descriptions = []
    for tool in TOOL_REGISTRY:
        params = tool.get("parameters", {}).get("properties", {})
        required = tool.get("parameters", {}).get("required", [])
        
        param_str = ", ".join([
            f"{p}: {params[p].get('type', 'any')}{' (required)' if p in required else ''}"
            for p in params
        ])
        
        descriptions.append(
            f"- {tool['name']}({param_str}): {tool['description']}"
        )
    
    return "\n".join(descriptions)


def get_tool_schema(tool_name: str) -> dict[str, Any] | None:
    """Get schema for a specific tool."""
    for tool in TOOL_REGISTRY:
        if tool["name"] == tool_name:
            return tool
    return None
