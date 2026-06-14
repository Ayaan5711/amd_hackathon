"""Governance API route tests (upload -> investigate -> dashboard/report/chat/metrics).

Runs in LLM_MODE=mock (the default) against the FastAPI app's full ASGI stack via
TestClient, exercising app/api/governance_routes.py end to end on the seeded
synthetic dataset.
"""

from __future__ import annotations

import time

from fastapi.testclient import TestClient

from app.main import app

POLL_TIMEOUT_S = 120
POLL_INTERVAL_S = 0.5


def _wait_for_completion(client: TestClient, run_id: str) -> dict:
    deadline = time.monotonic() + POLL_TIMEOUT_S
    while time.monotonic() < deadline:
        resp = client.get(f"/api/governance/status/{run_id}")
        assert resp.status_code == 200
        body = resp.json()
        if body["status"] != "running":
            return body
        time.sleep(POLL_INTERVAL_S)
    raise AssertionError(f"Investigation run {run_id} did not complete within {POLL_TIMEOUT_S}s")


def test_full_governance_flow():
    with TestClient(app) as client:
        # 1. Load the demo dataset
        upload_resp = client.post("/api/governance/demo")
        assert upload_resp.status_code == 200
        upload = upload_resp.json()
        session_id = upload["session_id"]
        assert upload["row_count"] > 0
        assert "log_id" in upload["columns"]

        # 2. Kick off an investigation
        investigate_resp = client.post(f"/api/governance/investigate/{session_id}")
        assert investigate_resp.status_code == 200
        investigate = investigate_resp.json()
        run_id = investigate["run_id"]
        assert investigate["session_id"] == session_id
        assert investigate["status"] == "running"

        # 3. Poll until complete
        status = _wait_for_completion(client, run_id)
        assert status["status"] == "complete"
        assert status["error"] is None
        steps = [event["step"] for event in status["progress"]]
        assert "triage" in steps
        assert "orchestrator" in steps
        assert "risk_scoring" in steps
        assert "dashboard" in steps
        assert "report" in steps

        # Live "agent thinking" trace (mock-mode rationale, streamed word-by-word)
        # for the orchestrator step.
        thinking_events = [
            event for event in status["progress"] if event.get("step") == "orchestrator" and event.get("type") == "thinking"
        ]
        assert thinking_events
        assert all("delta" in event for event in thinking_events)
        assert "".join(event["delta"] for event in thinking_events).strip()

        # 4. Dashboard
        dashboard_resp = client.get(f"/api/governance/dashboard/{run_id}")
        assert dashboard_resp.status_code == 200
        dashboard = dashboard_resp.json()
        assert dashboard["total_entries"] == upload["row_count"]
        assert set(dashboard["findings_by_category"]) == {"pii", "security", "compliance", "hallucination"}

        # 5. Report
        report_resp = client.get(f"/api/governance/report/{run_id}")
        assert report_resp.status_code == 200
        report = report_resp.json()
        assert set(report) == {
            "executive_summary",
            "detailed_findings",
            "remediation_plan",
            "incident_notifications",
            "monitoring_recommendations",
        }
        for section_text in report.values():
            assert section_text

        # 6. Metrics
        metrics_resp = client.get(f"/api/governance/metrics/{run_id}")
        assert metrics_resp.status_code == 200
        metrics = metrics_resp.json()
        assert metrics["total_calls"] > 0
        assert metrics["efficiency"]["reduction_pct"] > 0

        # 7. Chat - tool-using question
        chat_resp = client.post(
            f"/api/governance/chat/{run_id}", json={"message": "What's the overall risk distribution?"}
        )
        assert chat_resp.status_code == 200
        chat = chat_resp.json()
        assert chat["run_id"] == run_id
        assert chat["tool_calls"] == [{"tool_name": "get_risk_distribution", "arguments": {}}]
        assert chat["response"]

        # Follow-up turn (general intent, builds on chat history)
        chat_resp_2 = client.post(f"/api/governance/chat/{run_id}", json={"message": "Hello again"})
        assert chat_resp_2.status_code == 200
        assert chat_resp_2.json()["response"]


def test_unknown_run_id_returns_404():
    with TestClient(app) as client:
        for path in (
            "/api/governance/status/does-not-exist",
            "/api/governance/dashboard/does-not-exist",
            "/api/governance/report/does-not-exist",
            "/api/governance/metrics/does-not-exist",
            "/api/governance/stream/does-not-exist",
        ):
            resp = client.get(path)
            assert resp.status_code == 404, path

        chat_resp = client.post("/api/governance/chat/does-not-exist", json={"message": "hi"})
        assert chat_resp.status_code == 404


def test_unknown_session_id_returns_404():
    with TestClient(app) as client:
        resp = client.post("/api/governance/investigate/does-not-exist")
        assert resp.status_code == 404


def test_csv_upload():
    with TestClient(app) as client:
        csv_bytes = b"log_id,timestamp,user_prompt,ai_response\nLOG-1,2026-01-01T00:00:00,Hello,Hi there!\n"
        resp = client.post(
            "/api/governance/upload",
            files={"file": ("mini.csv", csv_bytes, "text/csv")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["row_count"] == 1
        assert body["has_retrieved_context"] is False
