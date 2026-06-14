"""FastAPI routes for the Governance ("Aegis") pack.

Additive to app/api/routes.py (the original Survey-pack routes, mounted at
/api): this router is mounted at /api/governance and covers the
upload -> investigate -> [stream progress] -> dashboard/report/chat/metrics
flow for the audit-log investigation pipeline.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app.agent import invoke_governance_chat
from app.api.governance_schemas import (
    GovernanceChatRequest,
    GovernanceChatResponse,
    InvestigateResponse,
    InvestigationStatusResponse,
    LogUploadResponse,
)
from app.api.investigation_common import (
    get_run_or_404,
    run_investigation_task,
    stream_chat_response,
    stream_investigation,
)
from app.config import SYNTHETIC_LOGS_DIR
from app.packs.governance import GOVERNANCE_PACK
from app.session.run_store import get_run_store
from app.session.store import get_session_store
from app.utils.csv_loader import load_log_batch

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/governance", tags=["governance"])


def _upload_response(session_id: str, filename: str, df: pd.DataFrame, schema: dict[str, Any]) -> LogUploadResponse:
    return LogUploadResponse(
        session_id=session_id,
        filename=filename,
        row_count=len(df),
        columns=schema["columns"],
        has_retrieved_context=schema["has_retrieved_context"],
        message="Log batch uploaded successfully. Start an investigation to analyze it.",
    )


@router.post("/upload", response_model=LogUploadResponse)
async def upload_log_batch(file: UploadFile = File(...)) -> LogUploadResponse:
    """Upload an AI-interaction log batch (CSV or JSON) for investigation."""
    logger.info(f"Governance upload request: {file.filename}")

    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    try:
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="File is empty")

        df, schema = load_log_batch(contents, file.filename)

        session_store = get_session_store()
        session_id = session_store.create(df=df, schema=schema, filename=file.filename)

        logger.info(f"Governance upload successful: {session_id} ({len(df)} entries)")
        return _upload_response(session_id, file.filename, df, schema)

    except ValueError as e:
        logger.warning(f"Governance upload validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Governance upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")


@router.post("/demo", response_model=LogUploadResponse)
async def load_demo_dataset() -> LogUploadResponse:
    """Load the seeded synthetic log dataset (no upload required)."""
    demo_path = SYNTHETIC_LOGS_DIR / "logs.csv"
    if not demo_path.exists():
        raise HTTPException(status_code=404, detail="Demo dataset not found on server")

    contents = demo_path.read_bytes()
    df, schema = load_log_batch(contents, "logs.csv")

    session_store = get_session_store()
    session_id = session_store.create(df=df, schema=schema, filename="logs.csv (demo)")

    logger.info(f"Demo dataset loaded: {session_id} ({len(df)} entries)")
    return _upload_response(session_id, "logs.csv (demo)", df, schema)


@router.post("/investigate/{session_id}", response_model=InvestigateResponse)
async def investigate(session_id: str) -> InvestigateResponse:
    """Kick off an investigation run for an uploaded log batch."""
    session_store = get_session_store()
    session = session_store.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired. Please upload your log batch again.")

    run_store = get_run_store()
    run_id = run_store.create(session_id=session_id, pack_name=GOVERNANCE_PACK.name)

    asyncio.create_task(run_investigation_task(run_id, session.df, GOVERNANCE_PACK))

    return InvestigateResponse(run_id=run_id, session_id=session_id, status="running")


@router.get("/status/{run_id}", response_model=InvestigationStatusResponse)
async def get_status(run_id: str) -> InvestigationStatusResponse:
    """Poll the status and progress log of an investigation run."""
    run = get_run_or_404(run_id)
    return InvestigationStatusResponse(run_id=run_id, status=run.status, progress=run.progress, error=run.error)


@router.get("/stream/{run_id}")
async def get_stream(run_id: str) -> StreamingResponse:
    """SSE stream of investigation progress: one `progress` event per completed
    graph node, followed by a final `complete` or `error` event."""
    return await stream_investigation(run_id)


@router.get("/dashboard/{run_id}")
async def get_dashboard(run_id: str) -> dict[str, Any]:
    """Get the dashboard summary for a completed investigation run."""
    run = get_run_or_404(run_id)
    if run.status == "running":
        raise HTTPException(status_code=409, detail="Investigation still running")
    if run.status == "error":
        raise HTTPException(status_code=500, detail=run.error or "Investigation failed")
    assert run.result is not None
    return run.result["dashboard"]


@router.get("/report/{run_id}")
async def get_report(run_id: str) -> dict[str, Any]:
    """Get the generated report sections for a completed investigation run."""
    run = get_run_or_404(run_id)
    if run.status == "running":
        raise HTTPException(status_code=409, detail="Investigation still running")
    if run.status == "error":
        raise HTTPException(status_code=500, detail=run.error or "Investigation failed")
    assert run.result is not None
    return run.result["report_sections"]


@router.get("/metrics/{run_id}")
async def get_metrics(run_id: str) -> dict[str, Any]:
    """Get the MetricsCollector snapshot (tokens, latency, efficiency, GPU) for a run."""
    run = get_run_or_404(run_id)
    if run.status == "running":
        raise HTTPException(status_code=409, detail="Investigation still running")
    if run.status == "error":
        raise HTTPException(status_code=500, detail=run.error or "Investigation failed")
    assert run.result is not None
    return run.result["metrics"]


@router.post("/chat/{run_id}", response_model=GovernanceChatResponse)
async def governance_chat(run_id: str, request: GovernanceChatRequest) -> GovernanceChatResponse:
    """Talk to the results of a completed investigation run."""
    run = get_run_or_404(run_id)
    if run.status == "running":
        raise HTTPException(status_code=409, detail="Investigation still running")
    if run.status == "error":
        raise HTTPException(status_code=500, detail=run.error or "Investigation failed")
    assert run.result is not None

    run_store = get_run_store()
    try:
        result = await invoke_governance_chat(
            GOVERNANCE_PACK,
            run.result,
            session_id=run_id,
            user_message=request.message,
            history=run.chat_history,
        )
    except Exception as e:
        logger.error(f"Governance chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process message: {str(e)}")

    run_store.append_chat_history(run_id, "user", request.message)
    run_store.append_chat_history(run_id, "assistant", result.get("response_narrative", ""))

    return GovernanceChatResponse(
        run_id=run_id,
        response=result.get("response_narrative", ""),
        follow_up_suggestions=result.get("follow_up_suggestions", []),
        evidence=result.get("evidence", {}),
        tool_calls=[
            {"tool_name": tc["tool_name"], "arguments": tc["arguments"]} for tc in result.get("tool_calls", [])
        ],
    )


@router.post("/chat_stream/{run_id}")
async def governance_chat_stream(run_id: str, request: GovernanceChatRequest) -> StreamingResponse:
    """SSE variant of /chat/{run_id}: streams Qwen3's live <think> reasoning trace
    as `thinking` events, followed by a single `complete` event with the same
    shape (minus `run_id`) as GovernanceChatResponse."""
    return await stream_chat_response(run_id, GOVERNANCE_PACK, request.message)
