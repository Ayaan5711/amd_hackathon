"""LangGraph StateGraph definition and execution."""

import logging
from typing import Any

from langgraph.graph import END, StateGraph

from app.agent.nodes import intent_node, synthesis_node, tool_node
from app.agent.state import AgentState

logger = logging.getLogger(__name__)


def should_call_tools(state: AgentState) -> str:
    """
    Conditional edge: determine if we should call tools or go straight to synthesis.
    
    Returns:
        'tools' if tools should be called, 'synthesize' otherwise
    """
    if state.get("clarification_needed"):
        return "synthesize"
    
    if state.get("tool_calls"):
        return "tools"
    
    return "synthesize"


def create_agent_graph() -> StateGraph:
    """
    Create the LangGraph agent graph.
    
    Graph structure:
        intent -> [conditional] -> tools -> synthesize -> END
                            |-> synthesize -> END
    """
    # Create the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("intent", intent_node)
    workflow.add_node("tools", tool_node)
    workflow.add_node("synthesize", synthesis_node)
    
    # Add edges
    workflow.set_entry_point("intent")
    
    # Conditional edge from intent
    workflow.add_conditional_edges(
        "intent",
        should_call_tools,
        {
            "tools": "tools",
            "synthesize": "synthesize"
        }
    )
    
    # Tool node always goes to synthesis
    workflow.add_edge("tools", "synthesize")
    
    # Synthesis ends the graph
    workflow.add_edge("synthesize", END)
    
    return workflow.compile()


# Global graph instance
_agent_graph = None


def get_agent_graph():
    """Get or create the agent graph singleton."""
    global _agent_graph
    if _agent_graph is None:
        _agent_graph = create_agent_graph()
        logger.info("Agent graph initialized")
    return _agent_graph


def invoke_agent(
    session_id: str,
    user_message: str,
    history: list[dict[str, str]],
    schema: dict[str, Any]
) -> dict[str, Any]:
    """
    Invoke the agent graph with user input.
    
    Args:
        session_id: Current session ID
        user_message: User's message
        history: Conversation history
        schema: Data schema
        
    Returns:
        Final agent state with response
    """
    graph = get_agent_graph()
    
    # Initial state
    initial_state: AgentState = {
        "session_id": session_id,
        "user_message": user_message,
        "history": history,
        "schema": schema,
        "intent": None,
        "tool_calls": [],
        "clarification_needed": False,
        "clarification_options": [],
        "tool_results": [],
        "response_narrative": "",
        "follow_up_suggestions": [],
        "evidence": {},
        "streaming": False
    }
    
    logger.info(f"Invoking agent for session {session_id}")
    
    # Run the graph
    try:
        result = graph.invoke(initial_state)
        logger.info("Agent execution complete")
        return result
    except Exception as e:
        logger.error(f"Agent execution failed: {e}")
        return {
            **initial_state,
            "response_narrative": f"I encountered an error processing your request: {str(e)}",
            "follow_up_suggestions": ["Please try again", "Try a different question"],
            "evidence": {"error": str(e)}
        }
