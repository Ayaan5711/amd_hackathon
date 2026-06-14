"""Governance "talk to results" chat graph tests.

Runs in LLM_MODE=mock (the default). Drives the chat graph
(intent -> [conditional] -> tools -> synthesize) over the completed
InvestigationState produced by the full investigation-graph run on the
seeded synthetic dataset (see tests/test_investigation_graph.py).
"""

from __future__ import annotations

import asyncio

import pandas as pd
import pytest

from app.agent import invoke_governance_chat, run_investigation, stream_governance_chat_turn
from app.config import SYNTHETIC_LOGS_DIR
from app.packs.governance import GOVERNANCE_PACK

LOGS_PATH = SYNTHETIC_LOGS_DIR / "logs.csv"


@pytest.fixture(scope="session")
def logs_df() -> pd.DataFrame:
    return pd.read_csv(LOGS_PATH)


@pytest.fixture(scope="session")
def full_run(logs_df):
    return asyncio.run(run_investigation(GOVERNANCE_PACK, logs_df, session_id="chat-session", run_id="chat-run"))


def _chat(full_run, message, history=None):
    return asyncio.run(
        invoke_governance_chat(
            GOVERNANCE_PACK,
            full_run,
            session_id="chat-session",
            user_message=message,
            history=history or [],
        )
    )


def _chat_stream(full_run, message, history=None):
    async def _collect():
        events = []
        async for event in stream_governance_chat_turn(
            GOVERNANCE_PACK,
            full_run,
            session_id="chat-session",
            user_message=message,
            history=history or [],
        ):
            events.append(event)
        return events

    return asyncio.run(_collect())


def _first_log_id(full_run, category: str) -> str:
    """Pick a log_id with a flagged finding for `category` from the completed run."""
    for finding in full_run["specialist_findings"]:
        if finding["agent"] == category and finding["flagged"]:
            return finding["log_id"]
    raise AssertionError(f"No flagged finding for category '{category}' in this run")


class TestFindingsByCategory:
    def test_pii_findings_question(self, full_run):
        result = _chat(full_run, "What PII issues were found?")
        assert result["tool_calls"] == [{"tool_name": "get_findings_by_category", "arguments": {"category": "pii"}}]
        assert result["tool_results"][0]["success"] is True
        assert result["response_narrative"]
        assert result["follow_up_suggestions"]

    def test_security_findings_question(self, full_run):
        result = _chat(full_run, "Were there any security issues?")
        tool_result = result["tool_results"][0]["result"]
        assert tool_result["category"] == "security"
        assert tool_result["success"] is True


class TestEntryDetailAndExplain:
    def test_get_entry_detail(self, full_run):
        log_id = full_run["entries"][0]["log_id"]
        result = _chat(full_run, f"Tell me about {log_id}")
        assert result["tool_calls"] == [{"tool_name": "get_entry_detail", "arguments": {"log_id": log_id}}]
        tool_result = result["tool_results"][0]["result"]
        assert tool_result["success"] is True
        assert tool_result["log_id"] == log_id
        assert result["response_narrative"]

    def test_explain_finding(self, full_run):
        log_id = _first_log_id(full_run, "security")
        result = _chat(full_run, f"Why was {log_id} flagged?")
        assert result["tool_calls"] == [{"tool_name": "explain_finding", "arguments": {"log_id": log_id}}]
        tool_result = result["tool_results"][0]["result"]
        assert tool_result["success"] is True
        assert tool_result["log_id"] == log_id
        assert tool_result["contributors"]
        assert result["response_narrative"]

    def test_entry_detail_unknown_log_id(self, full_run):
        result = _chat(full_run, "Tell me about LOG-DOES-NOT-EXIST")
        tool_result = result["tool_results"][0]["result"]
        assert tool_result["success"] is False
        assert "LOG-DOES-NOT-EXIST" in tool_result["error"]
        # All tools failed -> the narrative still says something useful, not a crash.
        assert result["response_narrative"]


class TestRiskAndComparison:
    def test_risk_distribution_question(self, full_run):
        result = _chat(full_run, "What's the overall risk distribution?")
        assert result["tool_calls"] == [{"tool_name": "get_risk_distribution", "arguments": {}}]
        tool_result = result["tool_results"][0]["result"]
        assert tool_result["risk_distribution"] == full_run["risk_scores"]["risk_distribution"]
        assert tool_result["overall_risk_score"] == full_run["risk_scores"]["overall_risk_score"]

        chart_data = result["evidence"]["chart_data"]
        assert chart_data
        assert chart_data[0]["tool_name"] == "get_risk_distribution"
        assert chart_data[0]["result"]["risk_distribution"] == full_run["risk_scores"]["risk_distribution"]

    def test_compare_categories_question(self, full_run):
        result = _chat(full_run, "Which category has the most findings, compare them all?")
        assert result["tool_calls"] == [{"tool_name": "compare_categories", "arguments": {}}]
        tool_result = result["tool_results"][0]["result"]
        assert set(tool_result["comparison"]) == {"pii", "security", "compliance", "hallucination"}


class TestEfficiencyMetrics:
    def test_accuracy_metrics_question(self, full_run):
        result = _chat(full_run, "How efficient was this run in terms of LLM calls?")
        assert result["tool_calls"] == [{"tool_name": "get_accuracy_metrics", "arguments": {}}]
        tool_result = result["tool_results"][0]["result"]
        assert tool_result["success"] is True
        assert tool_result["calls_by_agent"] == full_run["metrics"]["calls_by_agent"]
        assert tool_result["efficiency"]["reduction_pct"] > 0
        assert tool_result["total_latency_ms"] == full_run["metrics"]["total_latency_ms"]
        assert tool_result["avg_latency_ms"] == round(
            tool_result["total_latency_ms"] / tool_result["total_calls"], 1
        )
        assert str(tool_result["avg_latency_ms"]) in result["response_narrative"]


class TestGeneralConversation:
    def test_general_question_no_tools(self, full_run):
        result = _chat(full_run, "Hello there!")
        assert result["tool_calls"] == []
        assert result["intent"] == "general"
        assert "investigation" in result["response_narrative"].lower()
        assert result["follow_up_suggestions"]


class TestStreamingChatTurn:
    """stream_governance_chat_turn (option #2: live 'agent thinking' trace) -
    mock-mode coverage of the event shapes consumed by the chat_stream SSE
    endpoints (app/api/governance_routes.py, app/api/survey_routes.py)."""

    def test_tool_question_streams_thinking_then_complete(self, full_run):
        events = _chat_stream(full_run, "What's the overall risk distribution?")

        assert events[-1]["type"] == "complete"
        thinking_events = [e for e in events[:-1] if e["type"] == "thinking"]
        assert thinking_events
        assert all("delta" in e["data"] for e in thinking_events)

        complete_data = events[-1]["data"]
        assert complete_data["tool_calls"] == [{"tool_name": "get_risk_distribution", "arguments": {}}]
        assert complete_data["narrative"]
        chart_data = complete_data["evidence"]["chart_data"]
        assert chart_data and chart_data[0]["tool_name"] == "get_risk_distribution"

    def test_general_question_skips_thinking(self, full_run):
        events = _chat_stream(full_run, "Hello there!")

        assert len(events) == 1
        assert events[0]["type"] == "complete"
        assert events[0]["data"]["tool_calls"] == []
        assert "investigation" in events[0]["data"]["narrative"].lower()
        assert events[0]["data"]["follow_up_suggestions"]
