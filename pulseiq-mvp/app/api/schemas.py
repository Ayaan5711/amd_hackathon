"""Pydantic request/response models."""

from typing import Any

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    """Response from CSV upload."""
    session_id: str
    filename: str
    row_count: int
    column_count: int
    data_schema: dict[str, Any]
    message: str


class ChatRequest(BaseModel):
    """Request body for chat endpoint."""
    session_id: str = Field(..., description="Session ID from upload")
    message: str = Field(..., min_length=1, max_length=1000, description="User message")


class ChatResponse(BaseModel):
    """Response from chat (non-streaming)."""
    session_id: str
    response: str
    follow_up_suggestions: list[str]
    evidence: dict[str, Any]
    tool_calls: list[dict[str, Any]]


class SessionInfo(BaseModel):
    """Session information."""
    session_id: str
    filename: str
    uploaded_at: float
    last_accessed: float
    row_count: int
    column_count: int
    history_length: int


class SessionsListResponse(BaseModel):
    """Response for listing sessions."""
    sessions: list[SessionInfo]
    total: int


class ErrorResponse(BaseModel):
    """Error response."""
    error: str
    detail: str | None = None


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    sessions_active: int
