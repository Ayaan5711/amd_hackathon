"""Investigation graph for the AgentPack-driven audit pipeline.

    triage -> orchestrator -> [conditional] -> specialist_dispatch -> risk_scoring -> dashboard -> report -> END
                                            \\___________________________/

When triage finds nothing to investigate, the conditional edge skips
specialist_dispatch entirely (clean-dataset fast path). Written entirely
against the AgentPack contract (app/packs/base.py) - no pack-specific
branching here, so a new pack (e.g. Survey Analytics) only needs to provide
another AgentPack instance with the same shape.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

import pandas as pd
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph

from app.agent.prompts import build_orchestrator_prompt
from app.agent.state import InvestigationState
from app.config import MAX_TOKENS_ORCHESTRATOR, MAX_TOKENS_REPORT, VLLM_MODEL_ORCHESTRATOR, VLLM_MODEL_REPORT
from app.packs.governance.llm_utils import parse_json_response
from app.utils.csv_loader import df_to_log_entries
from app.utils.llm_client import call_llm_async
from app.utils.metrics import MetricsCollector

if TYPE_CHECKING:
    # Deferred to break the app.packs.base <-> app.agent import cycle: app.packs.base
    # imports app.agent.state, which (via app.agent's package __init__) would otherwise
    # re-enter this module before AgentPack is defined. Safe here because
    # `from __future__ import annotations` makes the `pack: AgentPack` annotations
    # below lazy strings, never evaluated at runtime.
    from app.packs.base import AgentPack

logger = logging.getLogger(__name__)


def _triage_summary(triage_results: list[dict[str, Any]], dispatch_plan_size: int) -> dict[str, int]:
    return {
        "total_entries": len(triage_results),
        "has_pii": sum(1 for t in triage_results if t["has_pii"]),
        "injection_suspect": sum(1 for t in triage_results if t["injection_suspect"]),
        "compliance_suspect": sum(1 for t in triage_results if t["compliance_suspect"]),
        "has_context": sum(1 for t in triage_results if t["has_context"]),
        "dispatch_plan_size": dispatch_plan_size,
    }


def _mock_orchestrator_plan(summary: dict[str, int]):
    """Content-aware mock rationale built from this run's actual triage stats."""

    def fabricator(_messages: list[dict[str, str]]) -> dict[str, Any]:
        rationale = (
            f"Triage flagged {summary['injection_suspect']} of {summary['total_entries']} entries for "
            f"security review, {summary['compliance_suspect']} for compliance review, and "
            f"{summary['has_context']} for hallucination/groundedness review "
            f"({summary['has_pii']} entries also contain PII per Presidio, scored without a specialist "
            f"call). Dispatching {summary['dispatch_plan_size']} specialist reviews to the flagged subset."
        )
        return {"rationale": rationale, "priority_categories": ["security", "compliance", "hallucination"]}

    return fabricator


def _triage_node(state: InvestigationState, config: RunnableConfig) -> dict[str, Any]:
    pack: AgentPack = config["configurable"]["pack"]
    df: pd.DataFrame = config["configurable"]["df"]
    triage_results = pack.triage_fn(df, {})
    return {"triage_results": triage_results}


async def _orchestrator_node(state: InvestigationState, config: RunnableConfig) -> dict[str, Any]:
    pack: AgentPack = config["configurable"]["pack"]
    metrics: MetricsCollector = config["configurable"]["metrics"]
    triage_results = state["triage_results"]

    plan = pack.dispatch_plan_fn(triage_results, {})
    summary = _triage_summary(triage_results, len(plan))
    prompt = build_orchestrator_prompt(summary)

    raw = await call_llm_async(
        messages=[{"role": "user", "content": prompt}],
        model=VLLM_MODEL_ORCHESTRATOR,
        max_tokens=MAX_TOKENS_ORCHESTRATOR,
        json_mode=True,
        enable_thinking=True,
        response_schema="orchestrator_plan",
        agent="orchestrator",
        metrics=metrics,
        mock_fabricator=_mock_orchestrator_plan(summary),
    )
    verdict = parse_json_response(raw)

    return {
        "investigation_plan": plan,
        "orchestrator_rationale": verdict.get("rationale", ""),
        "total_flagged": len(plan),
    }


def _route_after_orchestrator(state: InvestigationState) -> str:
    return "specialist_dispatch" if state["total_flagged"] > 0 else "risk_scoring"


async def _specialist_dispatch_node(state: InvestigationState, config: RunnableConfig) -> dict[str, Any]:
    pack: AgentPack = config["configurable"]["pack"]
    metrics: MetricsCollector = config["configurable"]["metrics"]
    entries_by_id = {e["log_id"]: e for e in state["entries"]}
    context = {"metrics": metrics}

    tasks = [
        pack.specialists[item["agent"]](entries_by_id[item["log_id"]], context)
        for item in state["investigation_plan"]
        if item["agent"] in pack.specialists
    ]
    findings = list(await asyncio.gather(*tasks)) if tasks else []
    return {"specialist_findings": findings}


def _risk_scoring_node(state: InvestigationState, config: RunnableConfig) -> dict[str, Any]:
    pack: AgentPack = config["configurable"]["pack"]
    risk_scores = pack.risk_scoring_fn(state["triage_results"], state["specialist_findings"])
    return {"risk_scores": risk_scores}


def _dashboard_node(state: InvestigationState, config: RunnableConfig) -> dict[str, Any]:
    pack: AgentPack = config["configurable"]["pack"]
    metrics: MetricsCollector = config["configurable"]["metrics"]
    metrics_summary = metrics.summary(len(state["entries"]))
    dashboard = pack.dashboard_fn(
        state["entries"],
        state["triage_results"],
        state["specialist_findings"],
        state["risk_scores"],
        metrics_summary,
    )
    return {"dashboard": dashboard, "metrics": metrics_summary}


def _build_report_context(state: InvestigationState) -> dict[str, Any]:
    return {
        "total_entries": len(state["entries"]),
        "total_flagged": state["total_flagged"],
        "entries": state["entries"],
        "entries_by_id": {e["log_id"]: e for e in state["entries"]},
        "triage_results": state["triage_results"],
        "specialist_findings": state["specialist_findings"],
        "risk_scores": state["risk_scores"],
        "dashboard": state["dashboard"],
        "metrics": state["metrics"],
        "orchestrator_rationale": state["orchestrator_rationale"],
    }


async def _report_node(state: InvestigationState, config: RunnableConfig) -> dict[str, Any]:
    pack: AgentPack = config["configurable"]["pack"]
    metrics: MetricsCollector = config["configurable"]["metrics"]

    if not pack.report_sections:
        return {"report_sections": {}}

    report_context = _build_report_context(state)
    sections: dict[str, str] = {}
    for name, build_prompt in pack.report_sections.items():
        prompt, mock_fabricator = build_prompt(report_context)
        raw = await call_llm_async(
            messages=[{"role": "user", "content": prompt}],
            model=VLLM_MODEL_REPORT,
            max_tokens=MAX_TOKENS_REPORT,
            json_mode=True,
            enable_thinking=False,
            response_schema=f"report_{name}",
            agent="report",
            metrics=metrics,
            mock_fabricator=mock_fabricator,
        )
        verdict = parse_json_response(raw)
        sections[name] = verdict.get("content", "")

    # Recompute metrics now that the report-authoring calls have also run, so
    # state["metrics"] and dashboard["metrics"] both reflect the full run.
    metrics_summary = metrics.summary(len(state["entries"]))
    dashboard = {**state["dashboard"], "metrics": metrics_summary}
    return {"report_sections": sections, "metrics": metrics_summary, "dashboard": dashboard}


def create_investigation_graph() -> Any:
    """Compile the investigation StateGraph. Pack-specific behavior is supplied
    per-invocation via `config["configurable"]["pack"]` (see run_investigation)."""
    workflow = StateGraph(InvestigationState)

    workflow.add_node("triage", _triage_node)
    workflow.add_node("orchestrator", _orchestrator_node)
    workflow.add_node("specialist_dispatch", _specialist_dispatch_node)
    workflow.add_node("risk_scoring", _risk_scoring_node)
    workflow.add_node("dashboard", _dashboard_node)
    workflow.add_node("report", _report_node)

    workflow.set_entry_point("triage")
    workflow.add_edge("triage", "orchestrator")
    workflow.add_conditional_edges(
        "orchestrator",
        _route_after_orchestrator,
        {"specialist_dispatch": "specialist_dispatch", "risk_scoring": "risk_scoring"},
    )
    workflow.add_edge("specialist_dispatch", "risk_scoring")
    workflow.add_edge("risk_scoring", "dashboard")
    workflow.add_edge("dashboard", "report")
    workflow.add_edge("report", END)

    return workflow.compile()


_investigation_graph: Any = None


def get_investigation_graph() -> Any:
    """Get or create the investigation graph singleton."""
    global _investigation_graph
    if _investigation_graph is None:
        _investigation_graph = create_investigation_graph()
        logger.info("Investigation graph initialized")
    return _investigation_graph


async def run_investigation(pack: AgentPack, df: pd.DataFrame, session_id: str, run_id: str) -> InvestigationState:
    """Run the investigation graph for one log batch under the given AgentPack."""
    graph = get_investigation_graph()
    entries = df_to_log_entries(df)
    metrics = MetricsCollector()

    initial_state: InvestigationState = {
        "session_id": session_id,
        "run_id": run_id,
        "entries": entries,  # type: ignore[typeddict-item]
        "triage_results": [],
        "investigation_plan": [],
        "orchestrator_rationale": "",
        "total_flagged": 0,
        "specialist_findings": [],
        "risk_scores": {},
        "dashboard": {},
        "report_sections": {},
        "metrics": {},
    }

    logger.info(f"Starting investigation run {run_id} ({len(entries)} entries, pack={pack.name})")
    config = {"configurable": {"pack": pack, "df": df, "metrics": metrics}}
    result = await graph.ainvoke(initial_state, config=config)
    logger.info(f"Investigation run {run_id} complete: {result['total_flagged']} entries flagged")
    return result
