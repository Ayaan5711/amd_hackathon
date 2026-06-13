"""Survey Analytics pack tests - AgentPack shape + end-to-end pipeline behavior.

Runs in LLM_MODE=mock (the default). Demonstrates that the same investigation
graph (triage -> orchestrator -> [conditional] -> specialist_dispatch ->
risk_scoring -> dashboard -> report) runs unchanged over survey responses by
providing another AgentPack instance (app/packs/base.py's AgentPack contract,
the same shape as GOVERNANCE_PACK).
"""

from __future__ import annotations

import asyncio
import json

import pandas as pd
import pytest

from app.agent import run_investigation
from app.packs.base import AgentPack
from app.packs.survey import SURVEY_PACK
from app.packs.survey.entries import survey_entries_fn
from app.tools.registry import TOOL_REGISTRY


@pytest.fixture(scope="session")
def survey_df() -> pd.DataFrame:
    """20 survey responses: 10 Sales (higher satisfaction) + 10 Marketing
    (lower satisfaction). Row 18 leaks an email address in Comments (PII).
    Row 19 has an NPS of 0, a clear statistical outlier vs. the rest."""
    departments = ["Sales"] * 10 + ["Marketing"] * 10
    satisfaction = [5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 3, 3, 4, 3, 4, 3, 3, 4, 3, 3]
    nps = [8, 9, 7, 8, 9, 8, 7, 9, 8, 7, 8, 9, 7, 8, 9, 8, 7, 9, 8, 0]
    comments = [
        "Great team support this quarter",
        "Manager feedback was helpful",
        "Happy with the new tools",
        "Workload is manageable",
        "Enjoyed the recent team event",
        "Communication has improved",
        "No major concerns this quarter",
        "Appreciate the flexible schedule",
        "Onboarding was smooth",
        "Looking forward to next quarter",
        "Could use more resources",
        "Deadlines are tight but ok",
        "Process changes were confusing",
        "Team morale is decent",
        "Need clearer goals",
        "More cross-team syncs would help",
        "Tooling could be improved",
        "Reach out to Jane Doe at jane.doe@example.com for follow up",
        "Training materials are outdated",
        "This quarter felt really isolating and difficult",
    ]
    return pd.DataFrame({
        "Department": departments,
        "Satisfaction": satisfaction,
        "NPS": nps,
        "Comments": comments,
    })


@pytest.fixture(scope="session")
def triage_results(survey_df):
    return SURVEY_PACK.triage_fn(survey_df, {})


@pytest.fixture(scope="session")
def full_run(survey_df):
    return asyncio.run(run_investigation(SURVEY_PACK, survey_df, session_id="survey-session", run_id="survey-run"))


class TestAgentPackShape:
    def test_is_agent_pack(self):
        assert isinstance(SURVEY_PACK, AgentPack)
        assert SURVEY_PACK.name == "survey"

    def test_entries_fn_overridden(self):
        assert SURVEY_PACK.entries_fn is survey_entries_fn

    def test_specialists_named_for_metrics_gating(self):
        # "compliance" so MetricsCollector.GATED_SPECIALIST_AGENTS and the
        # governance risk_scoring/dashboard/dispatch wiring apply unchanged.
        assert set(SURVEY_PACK.specialists) == {"compliance"}

    def test_report_sections_present(self):
        assert "survey_insights_summary" in SURVEY_PACK.report_sections

    def test_chat_tools_reused_from_existing_registry(self):
        assert SURVEY_PACK.chat_tool_registry == TOOL_REGISTRY
        for tool in TOOL_REGISTRY:
            assert tool["name"] in SURVEY_PACK.chat_tool_functions


class TestEntriesFn:
    def test_one_entry_per_row(self, survey_df):
        entries = survey_entries_fn(survey_df)
        assert len(entries) == len(survey_df)

    def test_row_serialized_into_ai_response(self, survey_df):
        entries = survey_entries_fn(survey_df)
        row = json.loads(entries[0]["ai_response"])
        assert row["Department"] == "Sales"

    def test_outlier_row_has_retrieved_context(self, survey_df):
        entries = survey_entries_fn(survey_df)
        outliers = json.loads(entries[19]["retrieved_context"])
        assert "NPS" in outliers

    def test_normal_row_has_no_retrieved_context(self, survey_df):
        entries = survey_entries_fn(survey_df)
        assert entries[0]["retrieved_context"] is None


class TestTriage:
    def test_one_triage_result_per_row(self, survey_df, triage_results):
        assert len(triage_results) == len(survey_df)

    def test_pii_detected_in_comments(self, triage_results):
        assert triage_results[17]["has_pii"] is True
        entity_types = {f["entity_type"] for f in triage_results[17]["pii_findings"]}
        assert "EMAIL_ADDRESS" in entity_types

    def test_nps_outlier_flagged_compliance_suspect(self, triage_results):
        assert triage_results[19]["compliance_suspect"] is True

    def test_no_injection_or_hallucination_signals(self, triage_results):
        assert all(t["injection_suspect"] is False for t in triage_results)
        assert all(t["has_context"] is False for t in triage_results)


class TestDispatchPlan:
    def test_only_outlier_row_dispatched(self, triage_results):
        plan = SURVEY_PACK.dispatch_plan_fn(triage_results, {})
        assert plan == [{"log_id": "19", "agent": "compliance"}]


class TestEndToEnd:
    def test_pipeline_populates_all_state_fields(self, survey_df, full_run):
        assert len(full_run["entries"]) == len(survey_df)
        assert len(full_run["triage_results"]) == len(survey_df)
        assert full_run["total_flagged"] == 1
        assert len(full_run["specialist_findings"]) == 1
        assert full_run["specialist_findings"][0]["agent"] == "compliance"
        assert full_run["specialist_findings"][0]["log_id"] == "19"

    def test_risk_scores_cover_every_entry(self, survey_df, full_run):
        assert len(full_run["risk_scores"]["by_log_id"]) == len(survey_df)
        assert sum(full_run["risk_scores"]["risk_distribution"].values()) == len(survey_df)

    def test_dashboard_reflects_pii_and_outlier(self, full_run):
        findings = full_run["dashboard"]["findings_by_category"]
        assert findings["pii"]["flagged"] == 1
        assert findings["compliance"]["flagged"] == 1
        assert findings["security"]["total"] == 0
        assert findings["hallucination"]["total"] == 0

    def test_report_section_is_data_grounded(self, full_run):
        sections = full_run["report_sections"]
        assert set(sections) == {"survey_insights_summary"}
        content = sections["survey_insights_summary"]
        assert "# Survey Insights Summary" in content
        assert "Satisfaction" in content
        assert "Sales" in content and "Marketing" in content
        assert "NPS" in content  # outlier column called out
