"""AgentPack contract - the seam that makes the investigation graph domain-agnostic.

The investigation graph (app/agent/graph.py) is written entirely against this
dataclass. A new domain (e.g. the Survey Analytics pack) is added by building
another AgentPack instance with the same shape - no changes to the graph itself.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pandas as pd

from app.utils.csv_loader import df_to_log_entries
from app.utils.llm_client import MockFabricator

if TYPE_CHECKING:
    # Deferred to type-checking only: `app.agent.state` is a submodule of
    # `app.agent`, and importing it at module load time would run
    # `app/agent/__init__.py`, which imports back into `app.packs.governance`
    # (which imports `AgentPack` from this module) - a circular import.
    from app.agent.state import InvestigationState, LogEntry, SpecialistFinding, TriageResult

# Converts the uploaded DataFrame into the LogEntry-shaped dicts the graph and
# specialists operate on. Defaults to the governance log-batch mapping
# (log_id/user_prompt/ai_response columns); other packs (e.g. Survey) provide
# their own mapping for non-log data.
EntriesFn = Callable[[pd.DataFrame], "list[LogEntry]"]

# Cheap, no-LLM pass over every uploaded entry. Produces the per-entry signals
# the orchestrator uses to decide what (if anything) needs specialist review.
TriageFn = Callable[[pd.DataFrame, dict[str, Any]], "list[TriageResult]"]

# A specialist agent: given one flagged log entry plus shared context
# (e.g. retrieved policy chunks, metrics collector), returns its finding.
SpecialistFn = Callable[["LogEntry", dict[str, Any]], "Awaitable[SpecialistFinding]"]

# Builds the investigation_plan ([{"log_id": ..., "agent": ...}, ...]) from the
# triage results - either directly (heuristic) or as input to an orchestrator prompt.
DispatchPlanFn = Callable[["list[TriageResult]", dict[str, Any]], list[dict[str, str]]]

# Pure aggregation: triage + specialist findings -> per-entry risk scores plus
# a dataset-level summary. No LLM calls.
RiskScoringFn = Callable[["list[TriageResult]", "list[SpecialistFinding]"], dict[str, Any]]

# Builds the dashboard JSON consumed by the frontend from the full set of
# investigation artifacts (entries, triage, findings, risk scores, metrics).
DashboardFn = Callable[
    ["list[LogEntry]", "list[TriageResult]", "list[SpecialistFinding]", dict[str, Any], dict[str, Any]],
    dict[str, Any],
]

# Builds the LLM prompt + mock fabricator for one report section from the
# accumulated investigation context (dashboard + findings + risk scores).
# The mock fabricator lets each section produce realistic, run-specific
# markdown in LLM_MODE=mock (same pattern as the specialist agents).
ReportSectionPromptFn = Callable[[dict[str, Any]], tuple[str, MockFabricator]]

# Chat tool implementation over the completed investigation results.
ChatToolFn = Callable[..., dict[str, Any]]

# Builds a content-aware mock fabricator for the chat intent classifier, given the
# user's message, the pack's chat tool registry, and the completed investigation.
# Lets each pack route chat questions to its own tools in LLM_MODE=mock. If unset,
# the chat graph falls back to the governance-flavored `_mock_chat_intent`.
ChatIntentFn = Callable[[str, list[dict[str, Any]], "InvestigationState"], MockFabricator]


@dataclass
class AgentPack:
    """Domain-specific wiring for the investigation graph.

    Each pack (governance, survey, ...) provides one instance of this dataclass.
    `app/agent/graph.py` takes a `pack: AgentPack` and never branches on `pack.name`
    for control flow - all domain logic lives behind these callables.
    """

    name: str
    required_columns: list[str]

    triage_fn: TriageFn
    specialists: dict[str, SpecialistFn]
    dispatch_plan_fn: DispatchPlanFn
    risk_scoring_fn: RiskScoringFn
    dashboard_fn: DashboardFn

    # DataFrame -> LogEntry-shaped dicts. Defaults to the governance log-batch mapping.
    entries_fn: EntriesFn = df_to_log_entries

    # Ordered: section_name -> prompt builder. Iteration order is report order.
    report_sections: dict[str, ReportSectionPromptFn] = field(default_factory=dict)

    # "Talk to results" chat layer - MCP-shaped tool registry + implementations.
    chat_tool_registry: list[dict[str, Any]] = field(default_factory=list)
    chat_tool_functions: dict[str, ChatToolFn] = field(default_factory=dict)

    # Optional pack-specific mock intent classifier (see ChatIntentFn above).
    chat_intent_fn: ChatIntentFn | None = None

    # Persona + framing for the chat intent classifier prompt (see
    # GOVERNANCE_CHAT_INTENT_PROMPT). A mismatched persona/noun can prime the LLM
    # to treat in-domain questions as out-of-scope "general" chat with no tool
    # calls, since the prompt's opening sentence sets the model's sense of what
    # questions are "in scope" for the available tools.
    chat_persona: str = "an AI-governance audit assistant"
    chat_entry_noun: str = "AI interaction log entries"

    # Shown by the chat synthesis node when the intent classifier decides no tool
    # call is needed (e.g. a greeting or "what can you do").
    chat_fallback_narrative: str = (
        "I can help you explore this investigation's findings. Try asking about:\n"
        "- Findings by category (e.g. 'What PII issues were found?')\n"
        "- A specific entry (e.g. 'Tell me about LOG-0042')\n"
        "- Why an entry was flagged (e.g. 'Why is LOG-0042 high risk?')\n"
        "- The overall risk distribution\n"
        "- Category comparisons\n"
        "- Token-efficiency metrics for this run"
    )
    chat_fallback_suggestions: list[str] = field(
        default_factory=lambda: [
            "What's the overall risk distribution?",
            "Which category has the most findings?",
            "How efficient was this run?",
        ]
    )
