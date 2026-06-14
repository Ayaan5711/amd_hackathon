"""Pydantic request/response models for the Survey Analytics pack API.

Dashboard/report/metrics/status responses are returned as plain dicts (their shape is
defined by SURVEY_PACK's dashboard_fn/report_sections/MetricsCollector), so only the
upload response gets its own model here. Investigate/status/chat shapes are
pack-agnostic and reused from `governance_schemas.py`.
"""

from pydantic import BaseModel


class SurveyUploadResponse(BaseModel):
    """Response from uploading (or loading the demo) survey CSV."""

    session_id: str
    filename: str
    row_count: int
    columns: list[str]
    message: str
