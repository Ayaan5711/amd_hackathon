"""Pydantic request/response models for the Governance ("Aegis") pack API.

Dashboard/report/metrics/status responses are returned as plain dicts (their
shape is defined by the AgentPack's dashboard_fn/report_sections/MetricsCollector
and can vary by pack), so only the request/response bodies with a fixed shape
get pydantic models here.
"""

from typing import Any

from pydantic import BaseModel, Field


class LogUploadResponse(BaseModel):
    """Response from uploading (or loading the demo) AI-interaction log batch."""

    session_id: str
    filename: str
    row_count: int
    columns: list[str]
    has_retrieved_context: bool
    message: str


class InvestigateResponse(BaseModel):
    """Response from kicking off an investigation run."""

    run_id: str
    session_id: str
    status: str


class InvestigationStatusResponse(BaseModel):
    """Polling-friendly status + progress log for a run."""

    run_id: str
    status: str
    progress: list[dict[str, Any]]
    error: str | None = None


class GovernanceChatRequest(BaseModel):
    """Request body for a governance "talk to results" chat turn."""

    message: str = Field(..., min_length=1, max_length=1000, description="User message")


class GovernanceChatResponse(BaseModel):
    """Response from a governance chat turn."""

    run_id: str
    response: str
    follow_up_suggestions: list[str]
    evidence: dict[str, Any]
    tool_calls: list[dict[str, Any]]
