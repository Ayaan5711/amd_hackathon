"""FastAPI routes for the Survey Analytics pack.

Mirrors `app/api/governance_routes.py`'s upload -> investigate ->
[stream progress] -> dashboard/report/chat/metrics flow, but mounted at
/api/survey and wired to `SURVEY_PACK` (survey CSV upload, no required
columns, survey-flavored dashboard/report/chat).
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from app.agent import invoke_governance_chat
from app.api.governance_schemas import (
    GovernanceChatRequest,
    GovernanceChatResponse,
    InvestigateResponse,
    InvestigationStatusResponse,
)
from app.api.investigation_common import (
    get_run_or_404,
    run_investigation_task,
    stream_investigation,
)
from app.api.survey_schemas import SurveyUploadResponse
from app.config import SURVEY_DEMO_PATH
from app.packs.survey import SURVEY_PACK
from app.session.run_store import get_run_store
from app.session.store import get_session_store
from app.utils.csv_loader import load_csv

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/survey", tags=["survey"])


def _upload_response(session_id: str, filename: str, df: Any, schema: dict[str, Any]) -> SurveyUploadResponse:
    return SurveyUploadResponse(
        session_id=session_id,
        filename=filename,
        row_count=len(df),
        columns=list(schema.keys()),
        message="Survey data uploaded successfully. Start an investigation to analyze it.",
    )


@router.post("/upload", response_model=SurveyUploadResponse)
async def upload_survey(file: UploadFile = File(...)) -> SurveyUploadResponse:
    """Upload a survey CSV for analysis."""
    logger.info(f"Survey upload request: {file.filename}")

    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    try:
        contents = await file.read()
        if len(contents) == 0:
            raise HTTPException(status_code=400, detail="File is empty")

        df, schema = load_csv(contents, file.filename)

        session_store = get_session_store()
        session_id = session_store.create(df=df, schema=schema, filename=file.filename)

        logger.info(f"Survey upload successful: {session_id} ({len(df)} rows)")
        return _upload_response(session_id, file.filename, df, schema)

    except ValueError as e:
        logger.warning(f"Survey upload validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Survey upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process file: {str(e)}")


@router.post("/demo", response_model=SurveyUploadResponse)
async def load_demo_dataset() -> SurveyUploadResponse:
    """Load the seeded synthetic survey dataset (no upload required)."""
    if not SURVEY_DEMO_PATH.exists():
        raise HTTPException(status_code=404, detail="Demo dataset not found on server")

    contents = SURVEY_DEMO_PATH.read_bytes()
    df, schema = load_csv(contents, SURVEY_DEMO_PATH.name)

    session_store = get_session_store()
    session_id = session_store.create(df=df, schema=schema, filename=f"{SURVEY_DEMO_PATH.name} (demo)")

    logger.info(f"Survey demo dataset loaded: {session_id} ({len(df)} rows)")
    return _upload_response(session_id, f"{SURVEY_DEMO_PATH.name} (demo)", df, schema)


@router.post("/investigate/{session_id}", response_model=InvestigateResponse)
async def investigate(session_id: str) -> InvestigateResponse:
    """Kick off an investigation run for an uploaded survey."""
    session_store = get_session_store()
    session = session_store.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found or expired. Please upload your survey again.")

    run_store = get_run_store()
    run_id = run_store.create(session_id=session_id, pack_name=SURVEY_PACK.name)

    asyncio.create_task(run_investigation_task(run_id, session.df, SURVEY_PACK))

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
async def survey_chat(run_id: str, request: GovernanceChatRequest) -> GovernanceChatResponse:
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
            SURVEY_PACK,
            run.result,
            session_id=run_id,
            user_message=request.message,
            history=run.chat_history,
        )
    except Exception as e:
        logger.error(f"Survey chat error: {e}")
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
