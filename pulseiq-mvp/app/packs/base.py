"""AgentPack contract - the seam that makes the investigation graph domain-agnostic.

The investigation graph (app/agent/graph.py) is written entirely against this
dataclass. A new domain (e.g. the Survey Analytics pack) is added by building
another AgentPack instance with the same shape - no changes to the graph itself.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from app.agent.state import LogEntry, SpecialistFinding, TriageResult
from app.utils.llm_client import MockFabricator

# Cheap, no-LLM pass over every uploaded entry. Produces the per-entry signals
# the orchestrator uses to decide what (if anything) needs specialist review.
TriageFn = Callable[[pd.DataFrame, dict[str, Any]], list[TriageResult]]

# A specialist agent: given one flagged log entry plus shared context
# (e.g. retrieved policy chunks, metrics collector), returns its finding.
SpecialistFn = Callable[[LogEntry, dict[str, Any]], Awaitable[SpecialistFinding]]

# Builds the investigation_plan ([{"log_id": ..., "agent": ...}, ...]) from the
# triage results - either directly (heuristic) or as input to an orchestrator prompt.
DispatchPlanFn = Callable[[list[TriageResult], dict[str, Any]], list[dict[str, str]]]

# Pure aggregation: triage + specialist findings -> per-entry risk scores plus
# a dataset-level summary. No LLM calls.
RiskScoringFn = Callable[[list[TriageResult], list[SpecialistFinding]], dict[str, Any]]

# Builds the dashboard JSON consumed by the frontend from the full set of
# investigation artifacts (entries, triage, findings, risk scores, metrics).
DashboardFn = Callable[
    [list[LogEntry], list[TriageResult], list[SpecialistFinding], dict[str, Any], dict[str, Any]],
    dict[str, Any],
]

# Builds the LLM prompt + mock fabricator for one report section from the
# accumulated investigation context (dashboard + findings + risk scores).
# The mock fabricator lets each section produce realistic, run-specific
# markdown in LLM_MODE=mock (same pattern as the specialist agents).
ReportSectionPromptFn = Callable[[dict[str, Any]], tuple[str, MockFabricator]]

# Chat tool implementation over the completed investigation results.
ChatToolFn = Callable[..., dict[str, Any]]


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

    # Ordered: section_name -> prompt builder. Iteration order is report order.
    report_sections: dict[str, ReportSectionPromptFn] = field(default_factory=dict)

    # "Talk to results" chat layer - MCP-shaped tool registry + implementations.
    chat_tool_registry: list[dict[str, Any]] = field(default_factory=list)
    chat_tool_functions: dict[str, ChatToolFn] = field(default_factory=dict)
