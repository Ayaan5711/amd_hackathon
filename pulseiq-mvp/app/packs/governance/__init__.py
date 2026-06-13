"""Governance/Audit pack ("Aegis") - the flagship AgentPack instance."""

from __future__ import annotations

from app.config import LOG_REQUIRED_COLUMNS
from app.packs.base import AgentPack
from app.packs.governance.compliance_agent import compliance_specialist
from app.packs.governance.dashboard import dashboard_fn
from app.packs.governance.dispatch import dispatch_plan_fn
from app.packs.governance.hallucination_agent import hallucination_specialist
from app.packs.governance.report import REPORT_SECTIONS
from app.packs.governance.risk_scoring import risk_scoring_fn
from app.packs.governance.security_agent import security_specialist
from app.packs.governance.tool_registry import GOVERNANCE_TOOL_FUNCTIONS, GOVERNANCE_TOOL_REGISTRY
from app.packs.governance.triage import triage_fn

GOVERNANCE_PACK = AgentPack(
    name="governance",
    required_columns=LOG_REQUIRED_COLUMNS,
    triage_fn=triage_fn,
    specialists={
        "security": security_specialist,
        "compliance": compliance_specialist,
        "hallucination": hallucination_specialist,
    },
    dispatch_plan_fn=dispatch_plan_fn,
    risk_scoring_fn=risk_scoring_fn,
    dashboard_fn=dashboard_fn,
    report_sections=REPORT_SECTIONS,
    chat_tool_registry=GOVERNANCE_TOOL_REGISTRY,
    chat_tool_functions=GOVERNANCE_TOOL_FUNCTIONS,
)
