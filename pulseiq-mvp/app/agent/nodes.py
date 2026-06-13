"""LangGraph node implementations."""

import json
import logging
from typing import Any

import pandas as pd

from app.agent.prompts import build_intent_prompt, build_synthesis_prompt
from app.agent.state import AgentState, ToolCall, ToolResult
from app.config import MAX_TOKENS_INTENT, MAX_TOKENS_SYNTHESIS, VLLM_MODEL_INTENT, VLLM_MODEL_SYNTHESIS
from app.session.store import get_session_store
from app.tools import (
    compare_trends,
    extract_open_text_themes,
    flag_anomalies,
    get_segment_stats,
    recommend_actions,
)
from app.utils.llm_client import call_llm

logger = logging.getLogger(__name__)

# Tool function mapping
TOOL_FUNCTIONS = {
    "get_segment_stats": get_segment_stats,
    "compare_trends": compare_trends,
    "extract_open_text_themes": extract_open_text_themes,
    "flag_anomalies": flag_anomalies,
    "recommend_actions": recommend_actions,
}


def intent_node(state: AgentState) -> dict[str, Any]:
    """
    Classify user intent and determine which tools to call.
    
    This node:
    1. Analyzes the user message
    2. Determines the appropriate intent
    3. Selects tools and extracts parameters
    4. Identifies if clarification is needed
    """
    logger.info(f"Intent node processing: {state['user_message'][:50]}...")
    
    try:
        # Build prompt
        prompt = build_intent_prompt(
            user_message=state["user_message"],
            schema=state["schema"],
            history=state["history"]
        )
        
        # Call LLM for intent classification
        response = call_llm(
            messages=[
                {"role": "system", "content": "You are an intent classifier. Respond only in JSON format."},
                {"role": "user", "content": prompt}
            ],
            model=VLLM_MODEL_INTENT,
            max_tokens=MAX_TOKENS_INTENT,
            json_mode=True,
            response_schema="intent",
            agent="chat_intent",
        )
        
        # Parse response
        classification = json.loads(response)
        
        intent = classification.get("intent", "general")
        tool_calls_data = classification.get("tool_calls", [])
        clarification_needed = classification.get("clarification_needed", False)
        
        # Build ToolCall objects
        tool_calls: list[ToolCall] = []
        for tc in tool_calls_data:
            tool_calls.append({
                "tool_name": tc.get("tool_name", ""),
                "arguments": tc.get("arguments", {})
            })
        
        logger.info(f"Intent classified: {intent}, tools: {[tc['tool_name'] for tc in tool_calls]}")
        
        return {
            "intent": intent,
            "tool_calls": tool_calls,
            "clarification_needed": clarification_needed,
            "clarification_options": classification.get("clarification_options", [])
        }
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse intent classification JSON: {e}")
        return {
            "intent": "general",
            "tool_calls": [],
            "clarification_needed": False,
            "clarification_options": []
        }
    except Exception as e:
        logger.error(f"Error in intent node: {e}")
        return {
            "intent": "general",
            "tool_calls": [],
            "clarification_needed": False,
            "clarification_options": []
        }


def tool_node(state: AgentState) -> dict[str, Any]:
    """
    Execute the selected tools.
    
    This node:
    1. Retrieves the session DataFrame
    2. Executes each tool call
    3. Collects results
    """
    logger.info(f"Tool node executing {len(state['tool_calls'])} tool calls")
    
    # Get session data
    session_store = get_session_store()
    session = session_store.get(state["session_id"])
    
    if not session:
        logger.error(f"Session not found: {state['session_id']}")
        return {
            "tool_results": [{
                "tool_name": "error",
                "success": False,
                "result": None,
                "error": "Session not found or expired"
            }]
        }
    
    df = session.df
    schema = session.schema
    
    results: list[ToolResult] = []
    
    for tool_call in state["tool_calls"]:
        tool_name = tool_call["tool_name"]
        arguments = tool_call["arguments"]
        
        logger.info(f"Executing tool: {tool_name} with args: {arguments}")
        
        # Get tool function
        tool_func = TOOL_FUNCTIONS.get(tool_name)
        
        if not tool_func:
            results.append({
                "tool_name": tool_name,
                "success": False,
                "result": None,
                "error": f"Unknown tool: {tool_name}"
            })
            continue
        
        try:
            # Execute tool
            result = tool_func(df=df, schema=schema, **arguments)
            
            results.append({
                "tool_name": tool_name,
                "success": result.get("success", False),
                "result": result,
                "error": result.get("error")
            })
            
        except Exception as e:
            logger.error(f"Tool {tool_name} failed: {e}")
            results.append({
                "tool_name": tool_name,
                "success": False,
                "result": None,
                "error": str(e)
            })
    
    logger.info(f"Tool execution complete: {len([r for r in results if r['success']])} succeeded")
    
    return {"tool_results": results}


def synthesis_node(state: AgentState) -> dict[str, Any]:
    """
    Synthesize tool results into a natural language response.
    
    This node:
    1. Takes tool results
    2. Generates a narrative response
    3. Suggests follow-up questions
    """
    logger.info("Synthesis node generating response")
    
    # Handle clarification case
    if state.get("clarification_needed"):
        return {
            "response_narrative": "I need a bit more information to help you with that.",
            "follow_up_suggestions": state.get("clarification_options", []),
            "evidence": {}
        }
    
    # Handle no tools case (general conversation)
    if not state["tool_calls"]:
        return {
            "response_narrative": (
                "I can help you analyze your survey data. Try asking about:\n"
                "- Segment comparisons (e.g., 'How do departments compare on satisfaction?')\n"
                "- Trends over time (e.g., 'Show me NPS by quarter')\n"
                "- Open text themes (e.g., 'What are people saying about management?')\n"
                "- Anomalies (e.g., 'Are there any unusual patterns?')\n"
                "- Recommendations (e.g., 'What should we focus on?')"
            ),
            "follow_up_suggestions": [
                "Which department has the highest satisfaction?",
                "What are the main themes in the comments?",
                "Are there any outliers in the data?"
            ],
            "evidence": {}
        }
    
    # Check if all tools failed
    all_failed = all(not r["success"] for r in state["tool_results"])
    if all_failed:
        errors = [r.get("error", "Unknown error") for r in state["tool_results"]]
        return {
            "response_narrative": (
                "I wasn't able to complete that analysis. "
                f"Issues encountered: {'; '.join(errors)}. "
                "Could you try rephrasing your question or check if the data contains the expected columns?"
            ),
            "follow_up_suggestions": [
                "What columns are available in the data?",
                "Can you show me a summary of the data?"
            ],
            "evidence": {"errors": errors}
        }
    
    try:
        # Build synthesis prompt
        prompt = build_synthesis_prompt(
            user_message=state["user_message"],
            tool_results=state["tool_results"],
            history=state["history"]
        )
        
        # Call LLM for synthesis
        response = call_llm(
            messages=[
                {"role": "system", "content": "You are a survey analysis assistant. Be concise and data-driven."},
                {"role": "user", "content": prompt}
            ],
            model=VLLM_MODEL_SYNTHESIS,
            max_tokens=MAX_TOKENS_SYNTHESIS,
            json_mode=True,
            response_schema="synthesis",
            agent="chat_synthesis",
        )
        
        # Parse response
        synthesis = json.loads(response)
        
        return {
            "response_narrative": synthesis.get("narrative", ""),
            "follow_up_suggestions": synthesis.get("follow_up_suggestions", []),
            "evidence": synthesis.get("evidence", {})
        }
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse synthesis JSON: {e}")
        # Fallback: create simple response from results
        return _fallback_synthesis(state)
    except Exception as e:
        logger.error(f"Error in synthesis node: {e}")
        return _fallback_synthesis(state)


def _fallback_synthesis(state: AgentState) -> dict[str, Any]:
    """Generate a simple response when LLM synthesis fails."""
    parts = []
    
    for result in state["tool_results"]:
        if result["success"]:
            tool_result = result.get("result", {})
            if "summary" in tool_result:
                parts.append(tool_result["summary"])
            elif "narrative" in str(tool_result):
                parts.append(str(tool_result)[:200])
    
    narrative = " ".join(parts) if parts else "Analysis complete. Here are the results."
    
    return {
        "response_narrative": narrative,
        "follow_up_suggestions": [
            "Can you tell me more about this?",
            "What other insights can you find?"
        ],
        "evidence": {}
    }
