"""Survey Analytics pack - proves the AgentPack contract is domain-agnostic.

Wires the same investigation graph (triage -> orchestrator -> [conditional] ->
specialist_dispatch -> risk_scoring -> dashboard -> report) over survey
responses instead of AI-interaction logs:

- entries_fn: each response row becomes one LogEntry (full row as JSON).
- triage_fn: reuses the governance Presidio PII scan over that JSON, plus a
  no-LLM z-score outlier check (repurposing `compliance_suspect`).
- specialists: a single "compliance" specialist (insight_agent) reviews
  flagged outlier rows - named "compliance" so risk_scoring, dashboard,
  dispatch_plan_fn, and MetricsCollector's specialist gating are reused
  verbatim from the governance pack.
- dashboard_fn: wraps the governance dashboard with survey-flavored chart
  data (metric_summary, segment_breakdown, category_labels).
- report_sections: six sections (executive summary, segment analysis, trends,
  themes & sentiment, anomalies & quality, recommendations), each re-deriving
  a DataFrame from the investigation entries and running the existing
  segment/trend/anomaly survey tools.
- chat tools/intent: survey-flavored tool registry + mock intent classifier so
  "talk to results" routes segment/trend/theme/anomaly/recommendation/risk/
  entry-detail questions to the right tool.
"""

from __future__ import annotations

from app.packs.base import AgentPack
from app.packs.governance.dispatch import dispatch_plan_fn
from app.packs.governance.risk_scoring import risk_scoring_fn
from app.packs.survey.dashboard import survey_dashboard_fn
from app.packs.survey.entries import survey_entries_fn
from app.packs.survey.insight_agent import compliance_specialist
from app.packs.survey.report import SURVEY_REPORT_SECTIONS
from app.packs.survey.tool_registry import SURVEY_TOOL_FUNCTIONS, SURVEY_TOOL_REGISTRY, mock_survey_chat_intent
from app.packs.survey.triage import triage_fn

SURVEY_PACK = AgentPack(
    name="survey",
    required_columns=[],
    triage_fn=triage_fn,
    specialists={"compliance": compliance_specialist},
    dispatch_plan_fn=dispatch_plan_fn,
    risk_scoring_fn=risk_scoring_fn,
    dashboard_fn=survey_dashboard_fn,
    entries_fn=survey_entries_fn,
    report_sections=SURVEY_REPORT_SECTIONS,
    chat_tool_registry=SURVEY_TOOL_REGISTRY,
    chat_tool_functions=SURVEY_TOOL_FUNCTIONS,
    chat_intent_fn=mock_survey_chat_intent,
)
