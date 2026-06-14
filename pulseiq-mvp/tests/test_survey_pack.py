"""Survey Analytics pack tests - AgentPack shape + end-to-end pipeline behavior.

Runs in LLM_MODE=mock (the default). Demonstrates that the same investigation
graph (triage -> orchestrator -> [conditional] -> specialist_dispatch ->
risk_scoring -> dashboard -> report) runs unchanged over survey responses by
providing another AgentPack instance (app/packs/base.py's AgentPack contract,
the same shape as GOVERNANCE_PACK), and that the survey pack's six-section
report, charted dashboard, and "talk to your data" chat tools are all
data-grounded.
"""

from __future__ import annotations

import asyncio
import json

import pandas as pd
import pytest

from app.agent import invoke_governance_chat, run_investigation
from app.packs.base import AgentPack
from app.packs.survey import SURVEY_PACK
from app.packs.survey.categorical import (
    build_demographic_profile,
    build_response_summary,
    build_segment_response_crosstab,
    count_numeric_threshold,
    find_top_segment_for_numeric_threshold,
    find_top_segment_for_value,
    get_value_distribution,
    split_demographic_and_response_columns,
)
from app.packs.survey.common import pick_excluded_columns, reconstruct_df_and_schema
from app.packs.survey.entries import survey_entries_fn
from app.packs.survey.report import _crosstab_finding_sentence
from app.packs.survey.tool_registry import mock_survey_chat_intent
from app.tools.registry import TOOL_REGISTRY
from app.utils.csv_loader import get_categorical_columns


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


@pytest.fixture(scope="session")
def tcs_df() -> pd.DataFrame:
    """30 survey responses styled after the TCS "Inflation Expectations Survey of
    Households" workbook: Department/Satisfaction/NPS/Comments (the same numeric-
    segment shape as `survey_df`) plus Gender/Age_Band/City demographic columns and
    Outlook_General/Outlook_Food_Prices Likert "response" columns. All deterministic
    (no randomness).

    - Gender: 15 Male / 15 Female. Male skews "More than current" for
      Outlook_General (10/15 = 66.7%); Female skews "Similar to current" (9/15 = 60%).
    - Age_Band: 10 each of 18-24 / 25-34 / 35-44 (every bucket clears MIN_SEGMENT_SIZE=10).
    - City: 10 each of Guwahati / Mumbai / Delhi. Guwahati clearly leads
      Outlook_Food_Prices "More than current" (8/10 = 80% vs. 50%/30%).
    """
    departments = ["Sales"] * 15 + ["Marketing"] * 15
    satisfaction = [5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5] + [3, 3, 4, 3, 4, 3, 3, 4, 3, 3, 4, 3, 3, 4, 3]
    nps = [8, 9, 7, 8, 9, 8, 7, 9, 8, 7, 8, 9, 7, 8, 9] + [6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6]
    comments = [f"Survey response number {i} about this quarter's experience" for i in range(30)]

    genders = ["Male"] * 15 + ["Female"] * 15
    age_bands = ["18-24"] * 10 + ["25-34"] * 10 + ["35-44"] * 10
    cities = ["Guwahati"] * 10 + ["Mumbai"] * 10 + ["Delhi"] * 10

    # Male: 10/15 "More than current", 5/15 "Similar to current".
    # Female: 9/15 "Similar to current", 6/15 "Less than current".
    outlook_general = (
        ["More than current"] * 10 + ["Similar to current"] * 5
        + ["Similar to current"] * 9 + ["Less than current"] * 6
    )
    # Guwahati: 8/10 "More than current". Mumbai: mixed, 5/10 "More than current".
    # Delhi: mixed, 3/10 "More than current".
    outlook_food_prices = (
        ["More than current"] * 8 + ["Similar to current"] * 2
        + ["More than current"] * 5 + ["Similar to current"] * 3 + ["Less than current"] * 2
        + ["More than current"] * 3 + ["Similar to current"] * 4 + ["No change"] * 3
    )

    return pd.DataFrame({
        "Department": departments,
        "Satisfaction": satisfaction,
        "NPS": nps,
        "Comments": comments,
        "Gender": genders,
        "Age_Band": age_bands,
        "City": cities,
        "Outlook_General": outlook_general,
        "Outlook_Food_Prices": outlook_food_prices,
    })


@pytest.fixture(scope="session")
def tcs_full_run(tcs_df):
    return asyncio.run(run_investigation(SURVEY_PACK, tcs_df, session_id="tcs-session", run_id="tcs-run"))


@pytest.fixture(scope="session")
def tcs_ctx(tcs_df):
    """(df, schema, demographic_cols, response_cols) reconstructed the same way the
    report/dashboard/chat tools see the data."""
    df, schema = reconstruct_df_and_schema(survey_entries_fn(tcs_df))
    categorical_cols = get_categorical_columns(schema)
    exclude = pick_excluded_columns(categorical_cols)
    demographic_cols, response_cols = split_demographic_and_response_columns(df, schema, categorical_cols, exclude=exclude)
    return df, schema, demographic_cols, response_cols


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
        assert set(SURVEY_PACK.report_sections) == {
            "executive_summary",
            "segment_analysis",
            "trends_analysis",
            "themes_and_sentiment",
            "anomalies_and_quality",
            "recommendations",
        }

    def test_chat_tool_registry_covers_base_and_survey_tools(self):
        base_names = {tool["name"] for tool in TOOL_REGISTRY}
        survey_names = {tool["name"] for tool in SURVEY_PACK.chat_tool_registry}

        # All 5 segment/trend/theme/anomaly/recommendation tools plus the 2
        # governance lookup tools (entry detail + risk distribution overview)
        # plus the 5 demographic/Likert "response" analysis tools.
        assert base_names <= survey_names
        assert survey_names - base_names == {
            "get_entry_detail",
            "get_risk_distribution",
            "get_value_distribution",
            "get_response_by_segment",
            "find_top_segment_for_value",
            "count_numeric_threshold",
            "find_top_segment_for_numeric_threshold",
        }
        for name in survey_names:
            assert name in SURVEY_PACK.chat_tool_functions

    def test_chat_intent_fn_wired(self):
        assert SURVEY_PACK.chat_intent_fn is mock_survey_chat_intent


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

    def test_dashboard_has_chartable_metric_and_segment_blocks(self, full_run):
        dashboard = full_run["dashboard"]

        metric_summary = dashboard["metric_summary"]
        assert {m["column"] for m in metric_summary} == {"Satisfaction", "NPS"}
        for metric in metric_summary:
            assert metric["count"] == 20

        segment_breakdown = dashboard["segment_breakdown"]
        assert segment_breakdown is not None
        assert segment_breakdown["success"] is True
        assert segment_breakdown["segment_column"] == "Department"
        assert segment_breakdown["best_segment"] == "Sales"
        assert segment_breakdown["worst_segment"] == "Marketing"

        assert dashboard["category_labels"]["compliance"] == "Outlier responses flagged for review"

    def test_report_section_is_data_grounded(self, full_run):
        sections = full_run["report_sections"]
        assert set(sections) == {
            "executive_summary",
            "segment_analysis",
            "trends_analysis",
            "themes_and_sentiment",
            "anomalies_and_quality",
            "recommendations",
        }

        exec_summary = sections["executive_summary"]
        assert "# Executive Summary" in exec_summary
        assert "Satisfaction" in exec_summary and "Department" in exec_summary
        assert "Sales" in exec_summary and "Marketing" in exec_summary
        assert "NPS" in exec_summary  # the NPS outlier called out in "Outliers"
        assert "PII" in exec_summary

        segment = sections["segment_analysis"]
        assert "# Segment Analysis" in segment
        assert "Department" in segment
        assert "Sales" in segment and "Marketing" in segment
        assert "Satisfaction" in segment and "NPS" in segment
        assert "| Metric | Sales | Marketing |" in segment

        # This fixture has no Quarter/Period-like column, so trends hits the
        # documented "no time dimension" fallback.
        trends = sections["trends_analysis"]
        assert "# Trends Analysis" in trends
        assert "No time-based dimension" in trends

        themes = sections["themes_and_sentiment"]
        assert "# Themes & Sentiment" in themes
        assert "Comments" in themes
        assert "Top Keywords" in themes
        assert "Overall Sentiment" in themes

        anomalies = sections["anomalies_and_quality"]
        assert "# Anomalies & Data Quality" in anomalies
        assert "NPS" in anomalies
        assert "outlier" in anomalies.lower()

        recommendations = sections["recommendations"]
        assert "# Recommendations" in recommendations
        assert "Satisfaction" in recommendations and "NPS" in recommendations
        assert "Sales" in recommendations and "Marketing" in recommendations


class TestChat:
    """Exercises the survey pack's "talk to your data" chat tools end to end via
    `invoke_governance_chat`, covering segment/trend/theme/anomaly/recommendation/
    risk-overview/entry-detail style questions on the same investigation run."""

    def _chat(self, full_run, message: str):
        return asyncio.run(
            invoke_governance_chat(SURVEY_PACK, full_run, session_id="chat-session", user_message=message, history=[])
        )

    def test_segment_comparison_question(self, full_run):
        result = self._chat(full_run, "Which department has the highest satisfaction?")
        assert result["tool_calls"] == [
            {"tool_name": "get_segment_stats", "arguments": {"segment_column": "Department", "metric_column": "Satisfaction"}}
        ]
        assert result["tool_results"][0]["success"] is True
        assert result["response_narrative"]

    def test_trend_question_falls_back_to_segment_stats_without_quarter_column(self, full_run):
        # No Quarter/Period-like column exists in this fixture, so a "trend"
        # question gracefully degrades to a segment comparison instead.
        result = self._chat(full_run, "Show me trends by quarter")
        assert result["tool_calls"] == [
            {"tool_name": "get_segment_stats", "arguments": {"segment_column": "Department", "metric_column": "Satisfaction"}}
        ]
        assert result["tool_results"][0]["success"] is True
        assert result["response_narrative"]

    def test_themes_question(self, full_run):
        result = self._chat(full_run, "What are the main themes in the comments?")
        assert result["tool_calls"] == [{"tool_name": "extract_open_text_themes", "arguments": {"text_column": "Comments"}}]
        assert result["tool_results"][0]["success"] is True
        assert result["response_narrative"]

    def test_anomaly_question(self, full_run):
        result = self._chat(full_run, "Are there any outliers in the data?")
        assert result["tool_calls"] == [{"tool_name": "flag_anomalies", "arguments": {"columns": ["Satisfaction", "NPS"]}}]
        assert result["tool_results"][0]["success"] is True
        assert result["response_narrative"]

    def test_recommendation_question(self, full_run):
        result = self._chat(full_run, "What actions should we take?")
        assert result["tool_calls"] == [{"tool_name": "recommend_actions", "arguments": {}}]
        assert result["tool_results"][0]["success"] is True
        assert result["response_narrative"]

    def test_risk_overview_question(self, full_run):
        result = self._chat(full_run, "Give me an overview of flagged responses")
        assert result["tool_calls"] == [{"tool_name": "get_risk_distribution", "arguments": {}}]
        assert result["tool_results"][0]["success"] is True
        assert "Risk distribution" in result["response_narrative"]

    def test_entry_detail_question(self, full_run):
        result = self._chat(full_run, "Tell me about response 19")
        assert result["tool_calls"] == [{"tool_name": "get_entry_detail", "arguments": {"log_id": "19"}}]
        assert result["tool_results"][0]["success"] is True
        assert "19" in result["response_narrative"]


class TestCategoricalAnalysis:
    """Unit tests for app/packs/survey/categorical.py's demographic profile /
    Likert "response" / cross-tab primitives, using the tcs_ctx fixture
    (Gender/Age_Band/City demographics + Outlook_General/Outlook_Food_Prices
    Likert "response" columns)."""

    def test_split_demographic_and_response_columns(self, tcs_ctx):
        _, _, demographic_cols, response_cols = tcs_ctx
        assert demographic_cols == ["Gender", "Age_Band", "City"]
        assert response_cols == ["Outlook_General", "Outlook_Food_Prices"]

    def test_get_value_distribution(self, tcs_ctx):
        df, schema, _, _ = tcs_ctx
        result = get_value_distribution(df, schema, "Outlook_General")
        assert result["success"] is True
        assert result["total"] == 30
        assert result["dominant_value"] == "Similar to current"
        assert result["dominant_percent"] == 46.7

    def test_build_demographic_profile(self, tcs_ctx):
        df, schema, demographic_cols, _ = tcs_ctx
        result = build_demographic_profile(df, schema, demographic_cols)
        assert result["success"] is True
        assert result["total_responses"] == 30
        assert {p["column"] for p in result["profiles"]} == {"Gender", "Age_Band", "City"}

    def test_build_response_summary(self, tcs_ctx):
        df, schema, _, response_cols = tcs_ctx
        result = build_response_summary(df, schema, response_cols)
        assert result["success"] is True
        assert result["total_responses"] == 30
        questions = {q["column"]: q for q in result["questions"]}
        assert set(questions) == {"Outlook_General", "Outlook_Food_Prices"}
        assert questions["Outlook_General"]["dominant_value"] == "Similar to current"
        assert questions["Outlook_General"]["dominant_percent"] == 46.7
        assert questions["Outlook_Food_Prices"]["dominant_value"] == "More than current"
        assert questions["Outlook_Food_Prices"]["dominant_percent"] == 53.3

    def test_build_segment_response_crosstab(self, tcs_ctx):
        df, schema, _, _ = tcs_ctx
        result = build_segment_response_crosstab(df, schema, "Gender", "Outlook_General")
        assert result["success"] is True
        segments = {s["segment"]: s for s in result["segments"]}
        assert segments["Male"]["dominant_value"] == "More than current"
        assert segments["Male"]["dominant_percent"] == 66.7
        assert segments["Female"]["dominant_value"] == "Similar to current"
        assert segments["Female"]["dominant_percent"] == 60.0

    def test_find_top_segment_for_value(self, tcs_ctx):
        df, schema, _, _ = tcs_ctx
        result = find_top_segment_for_value(df, schema, "City", "Outlook_Food_Prices", "More than current")
        assert result["success"] is True
        assert result["top_segment"] == "Guwahati"
        assert result["top_percent"] == 80.0
        assert result["top_count"] == 8

    def test_count_numeric_threshold(self, tcs_ctx):
        df, schema, _, _ = tcs_ctx
        result = count_numeric_threshold(df, schema, "Satisfaction", op="ge", threshold=4)
        assert result["success"] is True
        assert result["count"] == 20
        assert result["total"] == 30
        assert result["percent"] == 66.7

    def test_find_top_segment_for_numeric_threshold(self, tcs_ctx):
        df, schema, _, _ = tcs_ctx
        result = find_top_segment_for_numeric_threshold(df, schema, "Department", "Satisfaction", op="ge", threshold=4)
        assert result["success"] is True
        assert result["top_segment"] == "Sales"
        assert result["top_percent"] == 100.0
        assert result["top_count"] == 15


class TestTcsReportAndDashboard:
    """End-to-end coverage of the demographic profile / Likert "response" / cross-tab
    analysis on a dataset that ALSO has Department/Satisfaction/NPS-style numeric
    segment data, proving both analyses coexist in the same report and dashboard."""

    def test_dashboard_has_demographic_and_response_blocks(self, tcs_full_run):
        dashboard = tcs_full_run["dashboard"]

        demographic_summary = dashboard["demographic_summary"]
        assert demographic_summary["success"] is True
        assert {p["column"] for p in demographic_summary["profiles"]} == {"Gender", "Age_Band", "City"}

        response_summary = dashboard["response_summary"]
        assert response_summary["success"] is True
        assert {q["column"] for q in response_summary["questions"]} == {"Outlook_General", "Outlook_Food_Prices"}

        crosstabs = dashboard["crosstabs"]
        assert crosstabs
        assert all(c["success"] for c in crosstabs)
        pairs = {(c["segment_column"], c["response_column"]) for c in crosstabs}
        assert ("Gender", "Outlook_General") in pairs
        assert ("Gender", "Outlook_Food_Prices") in pairs

    def test_executive_summary_has_demographic_sections(self, tcs_full_run):
        exec_summary = tcs_full_run["report_sections"]["executive_summary"]
        assert "## Demographic Profile" in exec_summary
        assert "## Key Findings" in exec_summary
        assert "## Recommended Actions" in exec_summary
        assert "Gender" in exec_summary
        assert "Male" in exec_summary and "Female" in exec_summary

    def test_segment_analysis_has_response_and_crosstab_sections(self, tcs_full_run):
        segment = tcs_full_run["report_sections"]["segment_analysis"]
        assert "## Response Distribution" in segment
        assert "## Full Demographic Analysis" in segment
        assert "Outlook_General" in segment
        assert "Outlook_Food_Prices" in segment
        # The pre-existing numeric segment table still coexists.
        assert "| Metric | Sales | Marketing |" in segment


class TestCrosstabFindingSentence:
    """Unit coverage for _crosstab_finding_sentence's tie vs. non-tie phrasing."""

    def _crosstab(self, male_pct: float, female_pct: float) -> dict:
        return {
            "response_column": "Outlook_General",
            "options": ["More than current", "Similar to current", "Less than current"],
            "segments": [
                {
                    "segment": "Male",
                    "distribution": [
                        {"value": "More than current", "percent": male_pct},
                        {"value": "Similar to current", "percent": 100.0 - male_pct},
                    ],
                },
                {
                    "segment": "Female",
                    "distribution": [
                        {"value": "More than current", "percent": female_pct},
                        {"value": "Similar to current", "percent": 100.0 - female_pct},
                    ],
                },
            ],
        }

    def _response_summary(self) -> dict:
        return {"questions": [{"column": "Outlook_General", "dominant_value": "More than current"}]}

    def test_tie_produces_neutral_phrasing(self):
        sentence = _crosstab_finding_sentence(self._crosstab(50.0, 50.0), self._response_summary())

        assert sentence is not None
        assert "similar" in sentence
        assert "materially higher" not in sentence
        assert "Male" in sentence and "Female" in sentence
        assert "50.0%" in sentence

    def test_non_tie_produces_materially_higher_phrasing(self):
        sentence = _crosstab_finding_sentence(self._crosstab(70.0, 30.0), self._response_summary())

        assert sentence is not None
        assert "materially higher" in sentence
        assert "Male" in sentence and "Female" in sentence
        assert "70.0%" in sentence and "30.0%" in sentence


class TestTcsChat:
    """Exercises the 5 new demographic/Likert "response" chat tools end to end via
    invoke_governance_chat, on the tcs_full_run investigation (which has both
    Department/Satisfaction/NPS numeric data AND Gender/Age_Band/City/Outlook_*
    demographic + response columns)."""

    def _chat(self, tcs_full_run, message: str):
        return asyncio.run(
            invoke_governance_chat(SURVEY_PACK, tcs_full_run, session_id="tcs-chat-session", user_message=message, history=[])
        )

    def test_demographic_distribution_question(self, tcs_full_run):
        result = self._chat(tcs_full_run, "What percentage of respondents are Male?")
        assert result["tool_calls"] == [{"tool_name": "get_value_distribution", "arguments": {"column": "Gender"}}]
        assert result["tool_results"][0]["success"] is True
        assert result["response_narrative"]

        chart_data = result["evidence"]["chart_data"]
        assert chart_data
        assert chart_data[0]["tool_name"] == "get_value_distribution"
        assert chart_data[0]["result"]["column"] == "Gender"

    def test_response_by_segment_question(self, tcs_full_run):
        result = self._chat(tcs_full_run, "Compare Outlook General by Gender")
        assert result["tool_calls"] == [
            {"tool_name": "get_response_by_segment", "arguments": {"segment_column": "Gender", "response_column": "Outlook_General"}}
        ]
        assert result["tool_results"][0]["success"] is True
        assert result["response_narrative"]

    def test_top_segment_for_value_question(self, tcs_full_run):
        result = self._chat(tcs_full_run, "Which city has the highest 'More than current' for Outlook Food Prices?")
        assert result["tool_calls"] == [
            {
                "tool_name": "find_top_segment_for_value",
                "arguments": {
                    "segment_column": "City",
                    "response_column": "Outlook_Food_Prices",
                    "value": "More than current",
                },
            }
        ]
        assert result["tool_results"][0]["success"] is True
        assert result["tool_results"][0]["result"]["top_segment"] == "Guwahati"
        assert result["response_narrative"]

    def test_numeric_threshold_question(self, tcs_full_run):
        result = self._chat(tcs_full_run, "How many respondents have Satisfaction >= 4?")
        assert result["tool_calls"] == [
            {"tool_name": "count_numeric_threshold", "arguments": {"column": "Satisfaction", "op": "ge", "threshold": 4.0}}
        ]
        assert result["tool_results"][0]["success"] is True
        assert result["tool_results"][0]["result"]["count"] == 20
        assert result["response_narrative"]
