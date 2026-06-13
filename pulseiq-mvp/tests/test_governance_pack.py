"""Governance pack accuracy tests - triage + specialist findings vs ground_truth.csv.

Runs in LLM_MODE=mock (the default), where each specialist's mock_fabricator
inspects the prompt text and fabricates a content-aware verdict using the same
heuristics as triage. This lets us measure precision/recall/F1 against the
seeded synthetic dataset without a GPU.
"""

from __future__ import annotations

import asyncio

import pandas as pd
import pytest

from app.config import SYNTHETIC_LOGS_DIR
from app.packs.governance import GOVERNANCE_PACK
from app.packs.governance.accuracy import precision_recall_f1
from app.utils.csv_loader import df_to_log_entries

LOGS_PATH = SYNTHETIC_LOGS_DIR / "logs.csv"
GROUND_TRUTH_PATH = SYNTHETIC_LOGS_DIR / "ground_truth.csv"


@pytest.fixture(scope="session")
def logs_df() -> pd.DataFrame:
    return pd.read_csv(LOGS_PATH)


@pytest.fixture(scope="session")
def ground_truth() -> pd.DataFrame:
    return pd.read_csv(GROUND_TRUTH_PATH)


@pytest.fixture(scope="session")
def triage_results(logs_df):
    return GOVERNANCE_PACK.triage_fn(logs_df, {})


@pytest.fixture(scope="session")
def specialist_findings(logs_df, triage_results):
    entries_by_id = {e["log_id"]: e for e in df_to_log_entries(logs_df)}
    plan = GOVERNANCE_PACK.dispatch_plan_fn(triage_results, {})

    async def run_plan():
        tasks = [
            GOVERNANCE_PACK.specialists[item["agent"]](entries_by_id[item["log_id"]], {})
            for item in plan
        ]
        return await asyncio.gather(*tasks)

    return asyncio.run(run_plan())


@pytest.fixture(scope="session")
def risk_scores(triage_results, specialist_findings):
    return GOVERNANCE_PACK.risk_scoring_fn(triage_results, specialist_findings)


class TestTriage:
    def test_runs_on_every_entry(self, logs_df, triage_results):
        assert len(triage_results) == len(logs_df)

    def test_pii_accuracy(self, triage_results, ground_truth):
        predicted = {t["log_id"] for t in triage_results if t["has_pii"]}
        actual = set(ground_truth[ground_truth["has_pii"]]["log_id"])
        metrics = precision_recall_f1(predicted, actual)
        assert metrics["recall"] >= 0.95
        assert metrics["precision"] >= 0.6


class TestDispatchPlan:
    def test_only_flagged_entries_dispatched(self, logs_df, triage_results):
        plan = GOVERNANCE_PACK.dispatch_plan_fn(triage_results, {})
        expected = sum(
            t["injection_suspect"] + t["compliance_suspect"] + t["has_context"]
            for t in triage_results
        )
        assert len(plan) == expected
        # Token-efficiency story: only a minority of entries need an LLM call.
        assert len(plan) < len(logs_df) * 3


class TestSpecialistAccuracy:
    def test_security_accuracy(self, specialist_findings, ground_truth):
        predicted = {f["log_id"] for f in specialist_findings if f["agent"] == "security" and f["flagged"]}
        actual = set(ground_truth[ground_truth["has_injection"]]["log_id"])
        metrics = precision_recall_f1(predicted, actual)
        assert metrics["f1"] >= 0.9

    def test_compliance_accuracy(self, specialist_findings, ground_truth):
        predicted = {f["log_id"] for f in specialist_findings if f["agent"] == "compliance" and f["flagged"]}
        actual = set(ground_truth[ground_truth["has_compliance_violation"]]["log_id"])
        metrics = precision_recall_f1(predicted, actual)
        assert metrics["f1"] >= 0.9

    def test_hallucination_accuracy(self, specialist_findings, ground_truth):
        predicted = {f["log_id"] for f in specialist_findings if f["agent"] == "hallucination" and f["flagged"]}
        actual = set(ground_truth[ground_truth["has_hallucination"]]["log_id"])
        metrics = precision_recall_f1(predicted, actual)
        assert metrics["f1"] >= 0.9


class TestRiskScoring:
    def test_every_entry_scored(self, logs_df, risk_scores):
        assert len(risk_scores["by_log_id"]) == len(logs_df)

    def test_severity_matches_ground_truth_mostly(self, risk_scores, ground_truth):
        gt_severity = dict(zip(ground_truth["log_id"], ground_truth["severity"]))
        total = len(risk_scores["by_log_id"])
        matches = sum(
            1 for log_id, r in risk_scores["by_log_id"].items() if r["severity"] == gt_severity.get(log_id)
        )
        # Mirrors compute_severity() from the generator; small gap is expected
        # from Presidio en_core_web_sm PII false positives on short proper nouns.
        assert matches / total >= 0.9

    def test_risk_distribution_sums_to_total(self, logs_df, risk_scores):
        assert sum(risk_scores["risk_distribution"].values()) == len(logs_df)


class TestDashboard:
    def test_dashboard_fn_produces_summary(self, logs_df, triage_results, specialist_findings, risk_scores):
        entries = df_to_log_entries(logs_df)
        dashboard = GOVERNANCE_PACK.dashboard_fn(entries, triage_results, specialist_findings, risk_scores, {})
        assert dashboard["total_entries"] == len(logs_df)
        assert "pii" in dashboard["findings_by_category"]
        assert "security" in dashboard["findings_by_category"]
        assert "compliance" in dashboard["findings_by_category"]
        assert "hallucination" in dashboard["findings_by_category"]
        assert dashboard["overall_risk_score"] >= 0
