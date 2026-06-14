"""Survey Analytics API route tests (upload/demo -> investigate -> dashboard/report/
chat/metrics).

Runs in LLM_MODE=mock (the default) against the FastAPI app's full ASGI stack via
TestClient, exercising app/api/survey_routes.py end to end.
"""

from __future__ import annotations

import time
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app

POLL_TIMEOUT_S = 120
POLL_INTERVAL_S = 0.5

SAMPLE_SURVEY_CSV = Path(__file__).parent / "fixtures" / "sample_survey.csv"


def _wait_for_completion(client: TestClient, run_id: str) -> dict:
    deadline = time.monotonic() + POLL_TIMEOUT_S
    while time.monotonic() < deadline:
        resp = client.get(f"/api/survey/status/{run_id}")
        assert resp.status_code == 200
        body = resp.json()
        if body["status"] != "running":
            return body
        time.sleep(POLL_INTERVAL_S)
    raise AssertionError(f"Investigation run {run_id} did not complete within {POLL_TIMEOUT_S}s")


def _run_full_flow(client: TestClient, upload_resp) -> tuple[str, dict, dict]:
    assert upload_resp.status_code == 200
    upload = upload_resp.json()
    session_id = upload["session_id"]
    assert upload["row_count"] > 0
    assert "Department" in upload["columns"]

    # Kick off an investigation
    investigate_resp = client.post(f"/api/survey/investigate/{session_id}")
    assert investigate_resp.status_code == 200
    investigate = investigate_resp.json()
    run_id = investigate["run_id"]
    assert investigate["session_id"] == session_id
    assert investigate["status"] == "running"

    # Poll until complete
    status = _wait_for_completion(client, run_id)
    assert status["status"] == "complete"
    assert status["error"] is None
    steps = [event["step"] for event in status["progress"]]
    assert "triage" in steps
    assert "orchestrator" in steps
    assert "risk_scoring" in steps
    assert "dashboard" in steps
    assert "report" in steps

    # Dashboard: risk distribution + survey-specific chart blocks
    dashboard_resp = client.get(f"/api/survey/dashboard/{run_id}")
    assert dashboard_resp.status_code == 200
    dashboard = dashboard_resp.json()
    assert dashboard["total_entries"] == upload["row_count"]
    assert set(dashboard["findings_by_category"]) == {"pii", "security", "compliance", "hallucination"}
    assert dashboard["category_labels"]["compliance"] == "Outlier responses flagged for review"
    assert dashboard["metric_summary"]

    # Report: all six survey sections, each non-empty
    report_resp = client.get(f"/api/survey/report/{run_id}")
    assert report_resp.status_code == 200
    report = report_resp.json()
    assert set(report) == {
        "executive_summary",
        "segment_analysis",
        "trends_analysis",
        "themes_and_sentiment",
        "anomalies_and_quality",
        "recommendations",
    }
    for section_text in report.values():
        assert section_text

    # Metrics
    metrics_resp = client.get(f"/api/survey/metrics/{run_id}")
    assert metrics_resp.status_code == 200
    metrics = metrics_resp.json()
    assert metrics["total_calls"] > 0

    # Chat - risk overview question (dataset-shape independent)
    chat_resp = client.post(
        f"/api/survey/chat/{run_id}", json={"message": "Give me an overview of flagged responses"}
    )
    assert chat_resp.status_code == 200
    chat = chat_resp.json()
    assert chat["run_id"] == run_id
    assert chat["tool_calls"] == [{"tool_name": "get_risk_distribution", "arguments": {}}]
    assert chat["response"]

    # Follow-up turn (general intent, builds on chat history)
    chat_resp_2 = client.post(f"/api/survey/chat/{run_id}", json={"message": "Thanks!"})
    assert chat_resp_2.status_code == 200
    assert chat_resp_2.json()["response"]

    return run_id, dashboard, report


def test_full_survey_flow_with_demo_dataset():
    with TestClient(app) as client:
        upload_resp = client.post("/api/survey/demo")
        _, dashboard, _ = _run_full_flow(client, upload_resp)

        # The demo dataset has Gender/Age_Band/City demographic columns and
        # Outlook_General/Outlook_Food_Prices Likert "response" columns.
        assert dashboard["demographic_summary"]["success"] is True
        assert dashboard["response_summary"]["success"] is True


def test_full_survey_flow_with_uploaded_csv():
    with TestClient(app) as client:
        with open(SAMPLE_SURVEY_CSV, "rb") as f:
            upload_resp = client.post(
                "/api/survey/upload",
                files={"file": ("sample_survey.csv", f, "text/csv")},
            )
        _, dashboard, _ = _run_full_flow(client, upload_resp)

        # This dataset has no demographic columns (Gender/Age/City/...).
        assert dashboard["demographic_summary"] is None
        # "Department" (4 options) is generically classified as a small-cardinality
        # "response" column even without a demographic column to cross it with.
        response_summary = dashboard["response_summary"]
        assert response_summary["success"] is True
        assert [q["column"] for q in response_summary["questions"]] == ["Department"]


def test_chat_demographic_question_with_demo_dataset():
    with TestClient(app) as client:
        upload_resp = client.post("/api/survey/demo")
        run_id, _, _ = _run_full_flow(client, upload_resp)

        chat_resp = client.post(
            f"/api/survey/chat/{run_id}", json={"message": "What percentage of respondents are Male?"}
        )
        assert chat_resp.status_code == 200
        chat = chat_resp.json()
        assert chat["tool_calls"] == [{"tool_name": "get_value_distribution", "arguments": {"column": "Gender"}}]
        assert chat["response"]


def test_unknown_run_id_returns_404():
    with TestClient(app) as client:
        for path in (
            "/api/survey/status/does-not-exist",
            "/api/survey/dashboard/does-not-exist",
            "/api/survey/report/does-not-exist",
            "/api/survey/metrics/does-not-exist",
            "/api/survey/stream/does-not-exist",
        ):
            resp = client.get(path)
            assert resp.status_code == 404, path

        chat_resp = client.post("/api/survey/chat/does-not-exist", json={"message": "hi"})
        assert chat_resp.status_code == 404


def test_unknown_session_id_returns_404():
    with TestClient(app) as client:
        resp = client.post("/api/survey/investigate/does-not-exist")
        assert resp.status_code == 404
