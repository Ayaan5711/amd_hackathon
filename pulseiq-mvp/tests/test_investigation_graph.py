"""Investigation graph integration tests - end-to-end pipeline behavior.

Runs in LLM_MODE=mock (the default). Covers:
- Full pipeline on the seeded synthetic dataset (logs.csv)
- The clean-dataset fast path (conditional edge skips specialist_dispatch
  when triage flags nothing)
- Token-efficiency metrics
- Dashboard population
"""

from __future__ import annotations

import asyncio

import pandas as pd
import pytest

from app.agent import run_investigation
from app.config import SYNTHETIC_LOGS_DIR
from app.packs.governance import GOVERNANCE_PACK
from app.utils.metrics import GATED_SPECIALIST_AGENTS

LOGS_PATH = SYNTHETIC_LOGS_DIR / "logs.csv"


@pytest.fixture(scope="session")
def logs_df() -> pd.DataFrame:
    return pd.read_csv(LOGS_PATH)


@pytest.fixture(scope="session")
def full_run(logs_df):
    return asyncio.run(run_investigation(GOVERNANCE_PACK, logs_df, session_id="test-session", run_id="test-run"))


@pytest.fixture(scope="session")
def clean_df() -> pd.DataFrame:
    """A small batch with no injection/compliance keywords and no retrieved_context,
    so dispatch_plan_fn returns an empty plan (PII triage alone never dispatches)."""
    return pd.DataFrame([
        {
            "log_id": "LOG-CLEAN-1",
            "timestamp": "2026-01-01T00:00:00",
            "user_prompt": "How do I reset my password?",
            "ai_response": "Go to the IT portal and click reset.",
            "retrieved_context": None,
            "model_name": "mock",
        },
        {
            "log_id": "LOG-CLEAN-2",
            "timestamp": "2026-01-01T00:01:00",
            "user_prompt": "How do I book a meeting room?",
            "ai_response": "Use the room finder in the calendar app.",
            "retrieved_context": None,
            "model_name": "mock",
        },
    ])


@pytest.fixture(scope="session")
def clean_run(clean_df):
    return asyncio.run(run_investigation(GOVERNANCE_PACK, clean_df, session_id="clean-session", run_id="clean-run"))


class TestEndToEnd:
    def test_pipeline_populates_all_state_fields(self, logs_df, full_run):
        assert len(full_run["entries"]) == len(logs_df)
        assert len(full_run["triage_results"]) == len(logs_df)
        assert full_run["total_flagged"] == len(full_run["investigation_plan"])
        assert len(full_run["specialist_findings"]) == full_run["total_flagged"]
        assert full_run["orchestrator_rationale"]

    def test_some_entries_flagged_on_seeded_dataset(self, full_run):
        assert full_run["total_flagged"] > 0
        assert full_run["specialist_findings"]

    def test_risk_scores_cover_every_entry(self, logs_df, full_run):
        assert len(full_run["risk_scores"]["by_log_id"]) == len(logs_df)
        assert sum(full_run["risk_scores"]["risk_distribution"].values()) == len(logs_df)


class TestCleanDatasetFastPath:
    def test_no_specialist_dispatch(self, clean_run):
        assert clean_run["total_flagged"] == 0
        assert clean_run["investigation_plan"] == []
        assert clean_run["specialist_findings"] == []

    def test_only_orchestrator_llm_call(self, clean_run):
        calls_by_agent = clean_run["metrics"]["calls_by_agent"]
        assert calls_by_agent.get("orchestrator") == 1
        for agent in GATED_SPECIALIST_AGENTS:
            assert agent not in calls_by_agent

    def test_risk_scoring_and_dashboard_still_run(self, clean_df, clean_run):
        assert len(clean_run["risk_scores"]["by_log_id"]) == len(clean_df)
        assert clean_run["dashboard"]["total_entries"] == len(clean_df)


class TestEfficiencyMetrics:
    def test_naive_vs_actual_calls(self, logs_df, full_run):
        efficiency = full_run["metrics"]["efficiency"]
        assert efficiency["naive_llm_calls"] == len(logs_df) * len(GATED_SPECIALIST_AGENTS)
        assert efficiency["actual_llm_calls"] == full_run["total_flagged"]
        assert 0 < efficiency["reduction_pct"] < 100

    def test_calls_by_agent_only_covers_flagged_subset(self, full_run):
        calls_by_agent = full_run["metrics"]["calls_by_agent"]
        gated_total = sum(calls_by_agent.get(agent, 0) for agent in GATED_SPECIALIST_AGENTS)
        assert gated_total == full_run["total_flagged"]
        assert calls_by_agent.get("orchestrator") == 1


class TestDashboard:
    def test_dashboard_matches_pipeline_state(self, logs_df, full_run):
        dashboard = full_run["dashboard"]
        assert dashboard["total_entries"] == len(logs_df)
        assert set(dashboard["findings_by_category"]) == {"pii", "security", "compliance", "hallucination"}
        assert dashboard["risk_distribution"] == full_run["risk_scores"]["risk_distribution"]
        assert dashboard["overall_risk_score"] == full_run["risk_scores"]["overall_risk_score"]
        assert dashboard["metrics"] == full_run["metrics"]

    def test_top_findings_are_high_or_critical(self, full_run):
        for finding in full_run["dashboard"]["top_findings"]:
            assert finding["severity"] in ("high", "critical")


class TestReport:
    def test_all_sections_populated(self, full_run):
        sections = full_run["report_sections"]
        assert set(sections) == {
            "executive_summary",
            "detailed_findings",
            "remediation_plan",
            "incident_notifications",
            "monitoring_recommendations",
        }
        for name, content in sections.items():
            assert content.strip(), f"section {name} is empty"

    def test_report_calls_recorded_and_metrics_updated(self, full_run):
        calls_by_agent = full_run["metrics"]["calls_by_agent"]
        assert calls_by_agent.get("report") == 5
        assert full_run["dashboard"]["metrics"] == full_run["metrics"]

    def test_executive_summary_references_run_numbers(self, logs_df, full_run):
        summary = full_run["report_sections"]["executive_summary"]
        assert str(len(logs_df)) in summary
        assert str(full_run["total_flagged"]) in summary

    def test_clean_dataset_report_sections_populated(self, clean_run):
        sections = clean_run["report_sections"]
        assert set(sections) == {
            "executive_summary",
            "detailed_findings",
            "remediation_plan",
            "incident_notifications",
            "monitoring_recommendations",
        }
        assert "No high or critical severity findings" in sections["incident_notifications"]
