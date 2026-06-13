"""LangGraph agent state definitions."""

from typing import Any, TypedDict


class ToolCall(TypedDict):
    """Represents a tool to be called."""
    tool_name: str
    arguments: dict[str, Any]


class ToolResult(TypedDict):
    """Represents the result of a tool execution."""
    tool_name: str
    success: bool
    result: Any
    error: str | None


class AgentState(TypedDict):
    """
    Complete state for the LangGraph agent.

    This state flows through the graph nodes and accumulates
    information at each step.
    """
    # Input fields (set at start)
    session_id: str
    user_message: str
    history: list[dict[str, str]]
    schema: dict[str, Any]

    # IntentNode outputs
    intent: str | None
    tool_calls: list[ToolCall]
    clarification_needed: bool
    clarification_options: list[str]

    # ToolNode outputs
    tool_results: list[ToolResult]

    # SynthesisNode outputs
    response_narrative: str
    follow_up_suggestions: list[str]
    evidence: dict[str, Any]

    # Streaming control
    streaming: bool


# =============================================================================
# Investigation graph state (Governance/Audit pack and future packs)
# =============================================================================

class LogEntry(TypedDict):
    """A single AI-assistant interaction log entry under investigation."""
    log_id: str
    timestamp: str
    user_prompt: str
    ai_response: str
    retrieved_context: str | None
    model_name: str | None


class TriageResult(TypedDict):
    """Cheap, no-LLM signal computed for every entry during triage_node."""
    log_id: str
    pii_findings: list[dict[str, Any]]  # Presidio entities: type, score, snippet
    has_pii: bool
    injection_suspect: bool   # heuristic prefilter for the Security agent
    compliance_suspect: bool  # heuristic prefilter for the Compliance agent
    has_context: bool         # retrieved_context present -> Hallucination candidate


class SpecialistFinding(TypedDict):
    """Result of a specialist agent's deep analysis of one flagged entry."""
    log_id: str
    agent: str  # "pii" | "security" | "compliance" | "hallucination"
    flagged: bool
    severity: str  # "low" | "medium" | "high" | "critical"
    summary: str
    evidence: dict[str, Any]


class ChatState(TypedDict):
    """
    State for the governance "talk to results" chat graph: intent -> [conditional]
    -> tool -> synthesis, operating over a completed InvestigationState (passed via
    config["configurable"]["investigation"]) instead of a raw DataFrame.
    """
    # Input fields (set at start)
    session_id: str
    user_message: str
    history: list[dict[str, str]]

    # IntentNode outputs
    intent: str | None
    tool_calls: list[ToolCall]
    clarification_needed: bool
    clarification_options: list[str]

    # ToolNode outputs
    tool_results: list[ToolResult]

    # SynthesisNode outputs
    response_narrative: str
    follow_up_suggestions: list[str]
    evidence: dict[str, Any]


class InvestigationState(TypedDict):
    """
    State for the investigation graph: triage -> orchestrator -> specialist
    dispatch -> risk scoring -> dashboard -> report.
    """
    # Input fields (set at start)
    session_id: str
    run_id: str
    entries: list[LogEntry]

    # triage_node outputs
    triage_results: list[TriageResult]

    # orchestrator_node outputs
    investigation_plan: list[dict[str, str]]  # [{"log_id": ..., "agent": ...}, ...]
    orchestrator_rationale: str
    total_flagged: int

    # specialist_dispatch_node outputs
    specialist_findings: list[SpecialistFinding]

    # risk_scoring_node outputs
    risk_scores: dict[str, dict[str, Any]]  # log_id -> {score, severity, contributors}

    # dashboard_node outputs
    dashboard: dict[str, Any]

    # report_node outputs
    report_sections: dict[str, str]

    # metrics (populated throughout)
    metrics: dict[str, Any]
