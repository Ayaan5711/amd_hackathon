"""Governance pack chat graph - "talk to results" over a completed InvestigationState.

Mirrors app/agent/graph.py + nodes.py's intent -> [conditional] -> tool -> synthesis
pattern, but the data source is a completed InvestigationState (dashboard, triage
results, specialist findings, risk scores) rather than a raw survey DataFrame. The
tool registry/implementations come from `pack.chat_tool_registry` /
`pack.chat_tool_functions` (app/packs/base.py), so this graph works for any AgentPack
that supplies a governance-shaped chat tool set - no pack-specific branching here.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import TYPE_CHECKING, Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph

from app.agent.prompts import build_governance_chat_intent_prompt, build_governance_chat_synthesis_prompt
from app.agent.state import ChatState, InvestigationState, ToolCall, ToolResult
from app.config import LLM_MODE, MAX_TOKENS_INTENT, MAX_TOKENS_SYNTHESIS, VLLM_MODEL_INTENT, VLLM_MODEL_SYNTHESIS
from app.packs.governance.llm_utils import parse_json_response
from app.utils.llm_client import call_llm_async, stream_llm_response

if TYPE_CHECKING:
    # Deferred for the same reason as investigation_graph.py's AgentPack import:
    # `from __future__ import annotations` makes the `pack: AgentPack` annotations
    # below lazy strings, never evaluated at runtime.
    from app.packs.base import AgentPack

logger = logging.getLogger(__name__)

_LOG_ID_PATTERN = re.compile(r"LOG-[A-Za-z0-9_-]+", re.IGNORECASE)
_CATEGORY_PATTERN = re.compile(r"\b(pii|security|injection|compliance|hallucination)\b", re.IGNORECASE)
_CATEGORY_ALIASES = {"injection": "security"}

_THINK_CLOSED_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_THINK_OPEN_PATTERN = re.compile(r"<think>(.*)", re.DOTALL)


def _investigation_summary(investigation: InvestigationState) -> dict[str, Any]:
    dashboard = investigation.get("dashboard", {})
    return {
        "total_entries": dashboard.get("total_entries", 0),
        "total_flagged": dashboard.get("total_flagged", 0),
        "findings_by_category": dashboard.get("findings_by_category", {}),
        "risk_distribution": dashboard.get("risk_distribution", {}),
        "overall_risk_score": dashboard.get("overall_risk_score", 0),
    }


def _mock_chat_intent(user_message: str, tool_registry: list[dict[str, Any]]):
    """Content-aware mock intent classifier: keyword/regex heuristics over the user
    message pick a tool + arguments from the governance tool registry, so
    LLM_MODE=mock can still exercise the tool -> synthesis path end-to-end."""

    available = {tool["name"] for tool in tool_registry}

    def fabricator(_messages: list[dict[str, str]]) -> dict[str, Any]:
        message = user_message.lower()
        log_id_match = _LOG_ID_PATTERN.search(user_message)
        category_match = _CATEGORY_PATTERN.search(message)
        category = None
        if category_match:
            category = _CATEGORY_ALIASES.get(category_match.group(1).lower(), category_match.group(1).lower())

        tool_calls: list[dict[str, Any]] = []
        if log_id_match and "explain_finding" in available and any(
            w in message for w in ("why", "explain", "reason")
        ):
            tool_calls.append({"tool_name": "explain_finding", "arguments": {"log_id": log_id_match.group(0)}})
        elif log_id_match and "get_entry_detail" in available:
            tool_calls.append({"tool_name": "get_entry_detail", "arguments": {"log_id": log_id_match.group(0)}})
        elif "compare_categories" in available and any(
            w in message for w in ("compare", "most", "which category")
        ):
            tool_calls.append({"tool_name": "compare_categories", "arguments": {}})
        elif category and "get_findings_by_category" in available:
            tool_calls.append({"tool_name": "get_findings_by_category", "arguments": {"category": category}})
        elif "get_risk_distribution" in available and any(
            w in message for w in ("risk", "distribution", "overall", "risky")
        ):
            tool_calls.append({"tool_name": "get_risk_distribution", "arguments": {}})
        elif "get_accuracy_metrics" in available and any(
            w in message for w in ("efficien", "accuracy", "token", "calls", "cost")
        ):
            tool_calls.append({"tool_name": "get_accuracy_metrics", "arguments": {}})

        return {
            "intent": "tool_use" if tool_calls else "general",
            "reasoning": "Mock mode (LLM_MODE=mock): keyword heuristic over the user message.",
            "tool_calls": tool_calls,
            "clarification_needed": False,
            "clarification_options": [],
        }

    return fabricator


async def _intent_node(state: ChatState, config: RunnableConfig) -> dict[str, Any]:
    pack: AgentPack = config["configurable"]["pack"]
    investigation: InvestigationState = config["configurable"]["investigation"]

    prompt = build_governance_chat_intent_prompt(
        user_message=state["user_message"],
        tool_registry=pack.chat_tool_registry,
        investigation_summary=_investigation_summary(investigation),
        history=state["history"],
        chat_persona=pack.chat_persona,
        chat_entry_noun=pack.chat_entry_noun,
    )
    if pack.chat_intent_fn is not None:
        mock_fabricator = pack.chat_intent_fn(state["user_message"], pack.chat_tool_registry, investigation)
    else:
        mock_fabricator = _mock_chat_intent(state["user_message"], pack.chat_tool_registry)

    raw = await call_llm_async(
        messages=[{"role": "user", "content": prompt}],
        model=VLLM_MODEL_INTENT,
        max_tokens=MAX_TOKENS_INTENT,
        json_mode=True,
        enable_thinking=False,
        response_schema="chat_intent",
        agent="chat_intent",
        mock_fabricator=mock_fabricator,
    )
    classification = parse_json_response(raw)

    tool_calls: list[ToolCall] = [
        {"tool_name": tc.get("tool_name", ""), "arguments": tc.get("arguments", {})}
        for tc in classification.get("tool_calls", [])
    ]

    return {
        "intent": classification.get("intent", "general"),
        "tool_calls": tool_calls,
        "clarification_needed": classification.get("clarification_needed", False),
        "clarification_options": classification.get("clarification_options", []),
    }


def _should_call_tools(state: ChatState) -> str:
    if state.get("clarification_needed"):
        return "synthesize"
    if state.get("tool_calls"):
        return "tools"
    return "synthesize"


def _tool_node(state: ChatState, config: RunnableConfig) -> dict[str, Any]:
    pack: AgentPack = config["configurable"]["pack"]
    investigation: InvestigationState = config["configurable"]["investigation"]

    results: list[ToolResult] = []
    for tool_call in state["tool_calls"]:
        tool_name = tool_call["tool_name"]
        arguments = tool_call["arguments"]
        tool_func = pack.chat_tool_functions.get(tool_name)

        if not tool_func:
            results.append(
                {"tool_name": tool_name, "success": False, "result": None, "error": f"Unknown tool: {tool_name}"}
            )
            continue

        try:
            result = tool_func(investigation, **arguments)
            results.append(
                {
                    "tool_name": tool_name,
                    "success": result.get("success", False),
                    "result": result,
                    "error": result.get("error"),
                }
            )
        except Exception as e:
            logger.error(f"Chat tool {tool_name} failed: {e}")
            results.append({"tool_name": tool_name, "success": False, "result": None, "error": str(e)})

    return {"tool_results": results}


def _summarize_tool_result(tool_name: str, data: dict[str, Any]) -> str:
    if tool_name == "get_findings_by_category":
        return f"{data['flagged_count']} entries were flagged for {data['category']}."
    if tool_name == "get_entry_detail":
        risk = data.get("risk", {})
        return f"{data['log_id']} has risk severity '{risk.get('severity', 'low')}' (score {risk.get('score', 0)})."
    if tool_name == "get_risk_distribution":
        return f"Risk distribution: {data['risk_distribution']}, overall score {data['overall_risk_score']}."
    if tool_name == "explain_finding":
        contributors = ", ".join(c["contributor"] for c in data["contributors"]) or "no contributing factors"
        return f"{data['log_id']} scored {data['score']} ({data['severity']}) due to: {contributors}."
    if tool_name == "compare_categories":
        return f"Category comparison: {data['comparison']}; most flagged: {data['most_flagged_category']}."
    if tool_name == "get_accuracy_metrics":
        eff = data.get("efficiency") or {}
        return (
            f"This run made {data['total_calls']} LLM calls "
            f"({eff.get('reduction_pct', 'N/A')}% reduction vs. naive), "
            f"averaging {data.get('avg_latency_ms', 0)}ms per call."
        )
    return str(data)


def _mock_chat_synthesis(state: ChatState):
    """Content-aware mock synthesis: turn the actual tool_results into a narrative."""

    def fabricator(_messages: list[dict[str, str]]) -> dict[str, Any]:
        parts: list[str] = []
        for result in state["tool_results"]:
            if not result["success"]:
                parts.append(f"The {result['tool_name']} lookup failed: {result.get('error')}")
                continue
            parts.append(_summarize_tool_result(result["tool_name"], result["result"]))

        narrative = " ".join(parts) if parts else "Here's what I found in the investigation results."
        return {
            "narrative": narrative,
            "follow_up_suggestions": [
                "What's the overall risk distribution?",
                "Which category has the most findings?",
            ],
            "evidence": {"tool_results": [r["tool_name"] for r in state["tool_results"]]},
        }

    return fabricator


async def _synthesis_node(state: ChatState, config: RunnableConfig) -> dict[str, Any]:
    if state.get("clarification_needed"):
        return {
            "response_narrative": "I need a bit more information to help you with that.",
            "follow_up_suggestions": state.get("clarification_options", []),
            "evidence": {},
        }

    if not state["tool_calls"]:
        pack: AgentPack = config["configurable"]["pack"]
        return {
            "response_narrative": pack.chat_fallback_narrative,
            "follow_up_suggestions": list(pack.chat_fallback_suggestions),
            "evidence": {},
        }

    all_failed = all(not r["success"] for r in state["tool_results"])
    if all_failed:
        errors = [r.get("error", "Unknown error") for r in state["tool_results"]]
        return {
            "response_narrative": (
                f"I wasn't able to look that up. Issues encountered: {'; '.join(str(e) for e in errors)}."
            ),
            "follow_up_suggestions": [
                "What's the overall risk distribution?",
                "Which category has the most findings?",
            ],
            "evidence": {"errors": errors},
        }

    prompt = build_governance_chat_synthesis_prompt(
        user_message=state["user_message"],
        tool_results=state["tool_results"],
        history=state["history"],
    )
    raw = await call_llm_async(
        messages=[{"role": "user", "content": prompt}],
        model=VLLM_MODEL_SYNTHESIS,
        max_tokens=MAX_TOKENS_SYNTHESIS,
        json_mode=True,
        enable_thinking=False,
        response_schema="chat_synthesis",
        agent="chat_synthesis",
        mock_fabricator=_mock_chat_synthesis(state),
    )
    synthesis = parse_json_response(raw)
    if not synthesis:
        return _fallback_synthesis(state)

    return {
        "response_narrative": synthesis.get("narrative", ""),
        "follow_up_suggestions": synthesis.get("follow_up_suggestions", []),
        "evidence": {**synthesis.get("evidence", {}), "chart_data": _chart_data(state)},
    }


def _chart_data(state: ChatState) -> list[dict[str, Any]]:
    """Successful tool results, ready for the frontend to render as charts
    (frontend/charts.js's renderEvidenceChart) - independent of LLM JSON quality."""
    return [{"tool_name": r["tool_name"], "result": r["result"]} for r in state["tool_results"] if r["success"]]


def _fallback_synthesis(state: ChatState) -> dict[str, Any]:
    parts = [_summarize_tool_result(r["tool_name"], r["result"]) for r in state["tool_results"] if r["success"]]
    return {
        "response_narrative": " ".join(parts) if parts else "Here are the results.",
        "follow_up_suggestions": ["Can you tell me more about this?"],
        "evidence": {"chart_data": _chart_data(state)},
    }


def create_chat_graph() -> Any:
    """Compile the governance chat StateGraph. The investigation results and tool
    registry are supplied per-invocation via `config["configurable"]` (see
    invoke_governance_chat)."""
    workflow = StateGraph(ChatState)

    workflow.add_node("intent", _intent_node)
    workflow.add_node("tools", _tool_node)
    workflow.add_node("synthesize", _synthesis_node)

    workflow.set_entry_point("intent")
    workflow.add_conditional_edges("intent", _should_call_tools, {"tools": "tools", "synthesize": "synthesize"})
    workflow.add_edge("tools", "synthesize")
    workflow.add_edge("synthesize", END)

    return workflow.compile()


_chat_graph: Any = None


def get_chat_graph() -> Any:
    """Get or create the governance chat graph singleton."""
    global _chat_graph
    if _chat_graph is None:
        _chat_graph = create_chat_graph()
        logger.info("Governance chat graph initialized")
    return _chat_graph


async def invoke_governance_chat(
    pack: AgentPack,
    investigation: InvestigationState,
    session_id: str,
    user_message: str,
    history: list[dict[str, str]],
) -> ChatState:
    """Run one "talk to results" chat turn over a completed InvestigationState."""
    graph = get_chat_graph()

    initial_state: ChatState = {
        "session_id": session_id,
        "user_message": user_message,
        "history": history,
        "intent": None,
        "tool_calls": [],
        "clarification_needed": False,
        "clarification_options": [],
        "tool_results": [],
        "response_narrative": "",
        "follow_up_suggestions": [],
        "evidence": {},
    }

    config = {"configurable": {"pack": pack, "investigation": investigation}}
    result = await graph.ainvoke(initial_state, config=config)
    return result


async def stream_governance_chat_turn(
    pack: AgentPack,
    investigation: InvestigationState,
    session_id: str,
    user_message: str,
    history: list[dict[str, str]],
):
    """Async-generator sibling of `invoke_governance_chat` that streams Qwen3's
    `<think>...</think>` reasoning trace live during the synthesis step.

    Reuses the same intent/tool node logic, then - instead of one
    `call_llm_async` for synthesis - drives `stream_llm_response` with
    `enable_thinking=True` and yields `{"type": "thinking", "data": {"delta": ...}}`
    events as the reasoning trace arrives, followed by a single
    `{"type": "complete", "data": {...}}` event shaped like
    `invoke_governance_chat`'s return value (`narrative`, `follow_up_suggestions`,
    `evidence`, `tool_calls`). Paths that don't need an LLM call (clarification,
    no tool calls, all tools failed) skip straight to the `complete` event.
    """
    state: ChatState = {
        "session_id": session_id,
        "user_message": user_message,
        "history": history,
        "intent": None,
        "tool_calls": [],
        "clarification_needed": False,
        "clarification_options": [],
        "tool_results": [],
        "response_narrative": "",
        "follow_up_suggestions": [],
        "evidence": {},
    }
    config = {"configurable": {"pack": pack, "investigation": investigation}}

    intent_delta = await _intent_node(state, config)
    state.update(intent_delta)

    tool_calls_list = [{"tool_name": tc["tool_name"], "arguments": tc["arguments"]} for tc in state["tool_calls"]]

    if state.get("clarification_needed"):
        yield {
            "type": "complete",
            "data": {
                "narrative": "I need a bit more information to help you with that.",
                "follow_up_suggestions": state.get("clarification_options", []),
                "evidence": {},
                "tool_calls": tool_calls_list,
            },
        }
        return

    if not state["tool_calls"]:
        yield {
            "type": "complete",
            "data": {
                "narrative": pack.chat_fallback_narrative,
                "follow_up_suggestions": list(pack.chat_fallback_suggestions),
                "evidence": {},
                "tool_calls": tool_calls_list,
            },
        }
        return

    tool_delta = _tool_node(state, config)
    state.update(tool_delta)

    all_failed = all(not r["success"] for r in state["tool_results"])
    if all_failed:
        errors = [r.get("error", "Unknown error") for r in state["tool_results"]]
        yield {
            "type": "complete",
            "data": {
                "narrative": (
                    f"I wasn't able to look that up. Issues encountered: {'; '.join(str(e) for e in errors)}."
                ),
                "follow_up_suggestions": [
                    "What's the overall risk distribution?",
                    "Which category has the most findings?",
                ],
                "evidence": {"errors": errors},
                "tool_calls": tool_calls_list,
            },
        }
        return

    prompt = build_governance_chat_synthesis_prompt(
        user_message=state["user_message"],
        tool_results=state["tool_results"],
        history=state["history"],
    )
    chart_data = _chart_data(state)

    if LLM_MODE == "mock":
        thinking_text = (
            f"Mock mode (LLM_MODE=mock): reviewing {len(state['tool_results'])} tool result(s) "
            f'to compose a response to "{state["user_message"]}"...'
        )
        for word in thinking_text.split(" "):
            yield {"type": "thinking", "data": {"delta": word + " "}}
            await asyncio.sleep(0.02)
        synthesis = _mock_chat_synthesis(state)([{"role": "user", "content": prompt}])
        yield {
            "type": "complete",
            "data": {
                "narrative": synthesis.get("narrative", ""),
                "follow_up_suggestions": synthesis.get("follow_up_suggestions", []),
                "evidence": {**synthesis.get("evidence", {}), "chart_data": chart_data},
                "tool_calls": tool_calls_list,
            },
        }
        return

    full_content = ""
    think_emitted = 0
    think_closed = False
    async for chunk in stream_llm_response(
        messages=[{"role": "user", "content": prompt}],
        model=VLLM_MODEL_SYNTHESIS,
        max_tokens=MAX_TOKENS_SYNTHESIS,
        enable_thinking=True,
    ):
        full_content += chunk
        if think_closed:
            continue

        closed_match = _THINK_CLOSED_PATTERN.search(full_content)
        if closed_match:
            remaining = closed_match.group(1)[think_emitted:]
            if remaining:
                yield {"type": "thinking", "data": {"delta": remaining}}
            think_closed = True
            continue

        open_match = _THINK_OPEN_PATTERN.search(full_content)
        if open_match:
            new_text = open_match.group(1)[think_emitted:]
            if new_text:
                yield {"type": "thinking", "data": {"delta": new_text}}
                think_emitted += len(new_text)

    synthesis = parse_json_response(full_content)
    if synthesis:
        yield {
            "type": "complete",
            "data": {
                "narrative": synthesis.get("narrative", ""),
                "follow_up_suggestions": synthesis.get("follow_up_suggestions", []),
                "evidence": {**synthesis.get("evidence", {}), "chart_data": chart_data},
                "tool_calls": tool_calls_list,
            },
        }
        return

    parts = [_summarize_tool_result(r["tool_name"], r["result"]) for r in state["tool_results"] if r["success"]]
    yield {
        "type": "complete",
        "data": {
            "narrative": " ".join(parts) if parts else "Here are the results.",
            "follow_up_suggestions": ["Can you tell me more about this?"],
            "evidence": {"chart_data": chart_data},
            "tool_calls": tool_calls_list,
        },
    }
