"""Pack-agnostic helpers shared by the Governance and Survey investigation routers.

Both `app/api/governance_routes.py` and `app/api/survey_routes.py` mount the same
upload -> investigate -> [stream progress] -> dashboard/report/chat/metrics flow over
the investigation graph (`app/agent/investigation_graph.py`), differing only in which
`AgentPack` they pass through. This module factors out the pack-agnostic pieces: the
per-node SSE progress summary, the background investigation task (entries built via
`pack.entries_fn`, not hardcoded to the governance log-batch mapping), run lookup, and
the SSE stream generator.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING, Any

import pandas as pd
from fastapi import HTTPException
from fastapi.responses import StreamingResponse

from app.agent.investigation_graph import get_investigation_graph
from app.session.run_store import InvestigationRun, get_run_store
from app.utils.metrics import MetricsCollector

if TYPE_CHECKING:
    from app.packs.base import AgentPack

logger = logging.getLogger(__name__)


def progress_event(node_name: str, delta: dict[str, Any]) -> dict[str, Any]:
    """Turn one investigation-graph node's state delta into an SSE-friendly summary."""
    if node_name == "triage":
        triage_results = delta.get("triage_results", [])
        return {
            "step": "triage",
            "message": f"Triaged {len(triage_results)} entries",
            "has_pii": sum(1 for t in triage_results if t["has_pii"]),
        }
    if node_name == "orchestrator":
        return {
            "step": "orchestrator",
            "message": delta.get("orchestrator_rationale", ""),
            "total_flagged": delta.get("total_flagged", 0),
        }
    if node_name == "specialist_dispatch":
        findings = delta.get("specialist_findings", [])
        return {"step": "specialist_dispatch", "message": f"Completed {len(findings)} specialist reviews"}
    if node_name == "risk_scoring":
        return {"step": "risk_scoring", "message": "Risk scoring complete"}
    if node_name == "dashboard":
        dashboard = delta.get("dashboard", {})
        return {
            "step": "dashboard",
            "message": "Dashboard ready",
            "overall_risk_score": dashboard.get("overall_risk_score"),
        }
    if node_name == "report":
        sections = delta.get("report_sections", {})
        return {"step": "report", "message": f"Report generated ({len(sections)} sections)"}
    return {"step": node_name, "message": "done"}


async def run_investigation_task(run_id: str, df: pd.DataFrame, pack: "AgentPack") -> None:
    """Background task: stream the investigation graph for `pack`, recording progress
    + the final InvestigationState into the RunStore as the run proceeds."""
    run_store = get_run_store()
    graph = get_investigation_graph()
    metrics = MetricsCollector()
    entries = pack.entries_fn(df)

    state: dict[str, Any] = {
        "session_id": run_id,
        "run_id": run_id,
        "entries": entries,
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
    config = {"configurable": {"pack": pack, "df": df, "metrics": metrics}}

    try:
        async for chunk in graph.astream(state, config=config, stream_mode="updates"):
            for node_name, delta in chunk.items():
                state.update(delta)
                run_store.append_progress(run_id, progress_event(node_name, delta))
        run_store.set_result(run_id, state)  # type: ignore[arg-type]
        logger.info(f"Investigation run {run_id} complete: {state['total_flagged']} entries flagged")
    except Exception as e:
        logger.error(f"Investigation run {run_id} failed: {e}")
        run_store.set_error(run_id, str(e))


def get_run_or_404(run_id: str) -> InvestigationRun:
    run = get_run_store().get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Investigation run not found")
    return run


async def stream_investigation(run_id: str) -> StreamingResponse:
    """SSE stream of investigation progress: one `progress` event per completed
    graph node, followed by a final `complete` or `error` event."""
    get_run_or_404(run_id)

    async def event_generator():
        run_store = get_run_store()
        sent = 0
        while True:
            run = run_store.get(run_id)
            if run is None:
                yield f"event: error\ndata: {json.dumps({'error': 'Run not found'})}\n\n"
                return

            while sent < len(run.progress):
                yield f"event: progress\ndata: {json.dumps(run.progress[sent])}\n\n"
                sent += 1

            if run.status == "complete":
                yield f"event: complete\ndata: {json.dumps({'run_id': run_id})}\n\n"
                return
            if run.status == "error":
                yield f"event: error\ndata: {json.dumps({'error': run.error})}\n\n"
                return

            await asyncio.sleep(0.3)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive", "X-Accel-Buffering": "no"},
    )
