# Aegis — AI Governance & Audit Platform
### (built on the "PulseIQ MVP" codebase) — TCS & AMD AI Hackathon 2026, Track 1 (Agents), team-2310

This document explains, in detail, what was built, why it was built this way, what
constraints shaped the design, and exactly how to run it — both on a local
Windows machine (no GPU) and on the AMD Developer Cloud (ROCm + vLLM).

---

## 1. Hackathon Context

| | |
|---|---|
| **Competition** | TCS & AMD AI Hackathon 2026 |
| **Track** | Track 1 — Agents |
| **Team** | team-2310 |
| **Target deployment env** | AMD Developer Cloud, ROCm + vLLM, Jupyter notebooks |
| **Allowed models** | Locally-served open models only (Qwen3, DeepSeek, Llama, etc.) via **vLLM's OpenAI-compatible server** — no OpenAI/Anthropic/external LLM API calls |
| **Eval focus (Track 1)** | Productivity, Latency, Token efficiency, Accuracy |
| **Overall scoring** | Problem Definition 10% · Technical Implementation 40% · Learnings & Future Work 20% · Innovation & Creativity 15% · Presentation & Demo 15% |
| **Deliverables** | 5-slide deck, demo video, GitHub repo |
| **Hackathon "avoid list"** | Plain chatbot-over-PDF, generic RAG Q&A, resume analyzers, etc. |

### The build constraint that shaped everything

The team has **no AMD GPU locally**. The required workflow is:

1. Build and test the entire application **on Windows, with no GPU**.
2. Push the code to GitHub.
3. Pull it into an AMD Developer Cloud Jupyter notebook (which has ROCm + a running vLLM server).
4. Flip **one environment variable** (`LLM_MODE=vllm`) and the exact same code now calls a real Qwen3 model over vLLM's OpenAI-compatible API instead of canned mock responses.

Every architectural decision below (the dual-mode LLM client, the mock fabricators, the
metrics collector's GPU-stats fallback) exists to make that single env-var flip work
without touching a line of application code.

---

## 2. What We Built — Product Overview

### 2.1 One-line pitch

**Aegis** is an agentic **AI Governance & Audit platform**: you upload a batch of an
enterprise AI assistant's interaction logs (prompts, responses, retrieved context), and
a swarm of specialist agents — triggered *only when needed* — finds PII leaks, prompt
injection attempts, policy/compliance violations, and hallucinations; scores the risk of
every entry; and produces a 5-document audit report plus a chat interface to interrogate
the results.

### 2.2 Two pluggable "packs" on one investigation engine

The platform is built around a single **investigation graph**
(`triage → orchestrator → [conditional] → specialist dispatch → risk scoring → dashboard
→ report → chat`) that is **domain-agnostic**. A "pack" (`AgentPack` dataclass,
[`app/packs/base.py`](app/packs/base.py)) plugs domain-specific behaviour into that
graph without changing the graph itself.

| Pack | Role | Domain | Specialists |
|---|---|---|---|
| **`GOVERNANCE_PACK`** (`app/packs/governance/`) | **Flagship / Aegis** | AI-assistant interaction logs | `security` (prompt injection), `compliance` (policy RAG), `hallucination` (groundedness) — plus a free `pii` finding from triage |
| **`SURVEY_PACK`** (`app/packs/survey/`) | **Stretch goal — pluggability proof** | Employee survey responses (the original "PulseIQ" dataset) | `compliance` (re-purposed as an "Insight Agent" that reviews statistical outliers) |

The Survey pack reuses **100% of the investigation graph's node logic**. The only new
code needed was a pack definition + 4 small files (`entries.py`, `triage.py`,
`insight_agent.py`, `report.py`) — proving the architecture is a genuine platform, not a
single-purpose app. This was verified end-to-end by 18 dedicated tests
(`tests/test_survey_pack.py`).

### 2.3 Why this design (mapping to the hackathon's eval axes)

| Eval axis | How Aegis addresses it |
|---|---|
| **Token efficiency** | Triage (PII scan + keyword heuristics, **zero LLM calls**) decides which of the 3 gated specialist agents (security/compliance/hallucination) actually need to run, *per entry*. The dashboard shows `naive_llm_calls` (run all 3 on every entry) vs. `actual_llm_calls` (triage-gated) and a `reduction_pct`. |
| **Accuracy** | A seeded synthetic dataset (`app/data/synthetic_logs/`) ships with hidden `ground_truth.csv` labels. `app/packs/governance/accuracy.py` computes precision/recall/F1 per finding category (PII, injection, compliance, hallucination) against the pipeline's actual output. |
| **Productivity / Latency** | SSE-streamed progress (`/api/governance/stream/{run_id}`), `asyncio.gather`-based parallel specialist dispatch, and a `MetricsCollector` recording per-call latency/tokens plus wall-clock time for the whole run. |
| **Innovation & Creativity** | A genuinely **conditional, dynamic routing graph** (clean datasets skip specialist dispatch entirely) + a **pluggable "agent pack" architecture** proven by a second, structurally-different domain (governance logs vs. survey responses) running through the *same* graph unmodified. Avoids the chatbot/RAG/PDF-QA "avoid list" — it's an audit/governance pipeline, not a Q&A bot (the chat layer is a secondary "talk to your results" feature, not the core product). |
| **Learnings & Future Work** | Documented limitations & roadmap in [§13](#13-known-limitations--future-work). |

---

## 3. Architecture

### 3.1 Repo layout (key paths)

```
pulseiq-mvp/
├── app/
│   ├── main.py                     # FastAPI app, CORS, static frontend mount, /health
│   ├── config.py                   # All tunables: LLM dual-mode, models, limits, weights
│   ├── api/
│   │   ├── routes.py                # Survey-pack routes ("PulseIQ" chat, mounted at /api)
│   │   ├── governance_routes.py     # Aegis routes (mounted at /api/governance)
│   │   └── schemas.py               # Pydantic request/response models
│   ├── agent/
│   │   ├── state.py                  # TypedDicts: LogEntry, TriageResult, SpecialistFinding,
│   │   │                             #   InvestigationState, AgentState, ChatState
│   │   ├── investigation_graph.py    # The pack-agnostic investigation LangGraph
│   │   ├── chat_nodes.py              # "Talk to results" chat graph (governance)
│   │   ├── graph.py / nodes.py /      # Original PulseIQ "talk to your CSV" chat graph
│   │   │   prompts.py                 #   (survey pack's chat layer)
│   ├── packs/
│   │   ├── base.py                    # AgentPack contract (the domain-agnostic seam)
│   │   ├── governance/                # Aegis: PII/Security/Compliance/Hallucination agents,
│   │   │                              #   risk scoring, dashboard, 5-section report, chat tools
│   │   └── survey/                    # Survey Analytics pack (stretch goal)
│   ├── tools/                         # 5 survey analytics tools (segment stats, trends, etc.)
│   ├── session/
│   │   ├── store.py                   # SessionStore — uploaded DataFrame + chat history
│   │   └── run_store.py               # RunStore — investigation run status/progress/result
│   ├── utils/
│   │   ├── llm_client.py              # Dual-mode (mock | vLLM) OpenAI-compatible client
│   │   ├── metrics.py                 # MetricsCollector (tokens/latency/efficiency/GPU)
│   │   └── csv_loader.py              # CSV/JSON loading, schema inference, log-batch loading
│   └── data/
│       ├── synthetic_logs/            # logs.csv (136 rows) + ground_truth.csv
│       └── policies/                  # 4 markdown policy docs (RAG corpus for Compliance agent)
├── scripts/
│   └── generate_synthetic_logs.py     # Builds the seeded synthetic log batch + ground truth
├── frontend/
│   ├── index.html / aegis.js / aegis.css   # Aegis governance dashboard UI
│   └── survey.html / app.js / style.css    # PulseIQ survey-chat UI
├── tests/                              # 85 tests (pytest)
├── requirements.txt
└── .env.example
```

### 3.2 The `AgentPack` contract — the domain-agnostic seam

`app/packs/base.py` defines a single dataclass that every pack must provide:

```python
@dataclass
class AgentPack:
    name: str
    required_columns: list[str]

    triage_fn: TriageFn                 # (df, ctx) -> list[TriageResult]   — no LLM
    specialists: dict[str, SpecialistFn]  # name -> async (entry, ctx) -> SpecialistFinding
    dispatch_plan_fn: DispatchPlanFn     # (triage_results, ctx) -> [{"log_id", "agent"}, ...]
    risk_scoring_fn: RiskScoringFn       # (triage_results, findings) -> risk dict
    dashboard_fn: DashboardFn            # (...) -> dashboard dict

    entries_fn: EntriesFn = df_to_log_entries   # DataFrame -> list[LogEntry]

    report_sections: dict[str, ReportSectionPromptFn] = field(default_factory=dict)
    chat_tool_registry: list[dict] = field(default_factory=list)
    chat_tool_functions: dict[str, ChatToolFn] = field(default_factory=dict)
```

The investigation graph (`app/agent/investigation_graph.py`) is written **entirely
against this contract** — it never imports anything governance- or survey-specific. To
add a third pack (say, "Code Review Logs"), you would write a new
`app/packs/<name>/__init__.py` instantiating `AgentPack(...)` with that domain's
functions and **change zero lines** of the graph.

### 3.3 `InvestigationState` (the shape that flows through the graph)

```python
class InvestigationState(TypedDict):
    session_id: str
    run_id: str
    entries: list[LogEntry]                 # from pack.entries_fn(df)

    triage_results: list[TriageResult]      # triage_node
    investigation_plan: list[dict]          # orchestrator_node: [{"log_id","agent"}, ...]
    orchestrator_rationale: str
    total_flagged: int

    specialist_findings: list[SpecialistFinding]   # specialist_dispatch_node
    risk_scores: dict[str, dict]            # risk_scoring_node
    dashboard: dict                          # dashboard_node
    report_sections: dict[str, str]          # report_node
    metrics: dict                            # MetricsCollector summary
```

### 3.4 Dual-mode LLM client (`app/utils/llm_client.py`)

A single async function, `call_llm_async(messages, model, max_tokens, temperature,
json_mode, enable_thinking, response_schema, agent, metrics, mock_fabricator)`, is
called by every agent. Its behaviour is controlled by the `LLM_MODE` env var:

- **`LLM_MODE=mock`** (default, used for *all* local Windows development):
  No network call at all. Returns a structurally-valid JSON response either from a
  per-call `mock_fabricator` closure (a **content-aware mock** — e.g. the Security
  agent's mock actually looks at the prompt text and decides `is_injection` based on
  which heuristic pattern matched) or from a static `_MOCK_RESPONSES` table.
- **`LLM_MODE=vllm`**: Uses an OpenAI-compatible client (`AsyncOpenAI(base_url=VLLM_BASE_URL,
  api_key=VLLM_API_KEY)`) pointed at vLLM's server. If `enable_thinking=True`, passes
  `extra_body={"chat_template_kwargs": {"enable_thinking": True}}` — the Qwen3
  thinking-mode toggle.

Every call (mock or real) is recorded into a `MetricsCollector`
(`app/utils/metrics.py`) with `{agent, model, tokens_in, tokens_out, latency_ms,
enable_thinking, response_schema}` — this is what powers the token-efficiency and
latency numbers on the dashboard, in **both** modes.

### 3.5 `MetricsCollector` — the token-efficiency story

```python
GATED_SPECIALIST_AGENTS = ("security", "compliance", "hallucination")

def efficiency(self, num_entries, agents_per_entry_naive=3):
    naive_calls  = num_entries * agents_per_entry_naive   # "run everything on everything"
    actual_calls = sum(1 for c in self.calls if c.agent in GATED_SPECIALIST_AGENTS)
    reduction_pct = round((1 - actual_calls / naive_calls) * 100, 1)
    return {"naive_llm_calls": naive_calls, "actual_llm_calls": actual_calls, "reduction_pct": reduction_pct}
```

`gpu_stats()` shells out to `rocm-smi --showuse --showmeminfo vram --json`; on Windows
(no ROCm) this fails cleanly and returns `{"gpu_available": false}` — on the AMD
Developer Cloud it returns real GPU utilization/VRAM numbers for the metrics panel.

---

## 4. Application Flow — Aegis Governance Pack (end to end)

```
 Upload CSV/JSON log batch
        │
        ▼
 ┌─────────────────────────────────────────────────────────────┐
 │ 1. TRIAGE  (triage_node — no LLM, runs on every entry)        │
 │    • Presidio PII scan of user_prompt + ai_response           │
 │    • Regex injection-pattern check on user_prompt              │
 │    • Regex compliance-category check on ai_response            │
 │    • has_context = bool(retrieved_context)                     │
 └─────────────────────────────────────────────────────────────┘
        │  TriageResult per entry: {pii_findings, has_pii,
        │   injection_suspect, compliance_suspect, has_context}
        ▼
 ┌─────────────────────────────────────────────────────────────┐
 │ 2. ORCHESTRATOR  (orchestrator_node — 1 LLM call,              │
 │    enable_thinking=True, response_schema="orchestrator_plan") │
 │    Input: triage summary stats + dispatch_plan_fn's proposed   │
 │    plan. Output: investigation_plan [{log_id, agent}, ...]     │
 │    + rationale + total_flagged.                                │
 └─────────────────────────────────────────────────────────────┘
        │
        ▼
   ┌─────────────────────────────┐
   │ total_flagged == 0 ?          │── yes ──► skip straight to risk_scoring
   └─────────────────────────────┘
        │ no
        ▼
 ┌─────────────────────────────────────────────────────────────┐
 │ 3. SPECIALIST DISPATCH  (specialist_dispatch_node)             │
 │    asyncio.gather over investigation_plan — runs ONLY the      │
 │    flagged (log_id, agent) pairs in parallel:                  │
 │      • security      → prompt-injection LLM classifier         │
 │      • compliance     → policy-RAG LLM judge                    │
 │      • hallucination  → groundedness LLM judge                  │
 └─────────────────────────────────────────────────────────────┘
        │  list[SpecialistFinding]
        ▼
 ┌─────────────────────────────────────────────────────────────┐
 │ 4. RISK SCORING  (risk_scoring_node — pure aggregation,        │
 │    no LLM). Combines triage PII signal + specialist findings   │
 │    into a 0-100 score + severity per log_id, and a dataset-     │
 │    wide risk_distribution + overall_risk_score.                │
 └─────────────────────────────────────────────────────────────┘
        ▼
 ┌─────────────────────────────────────────────────────────────┐
 │ 5. DASHBOARD  (dashboard_node — pure aggregation)               │
 │    findings_by_category, risk_distribution, top_findings,       │
 │    accuracy (vs ground truth, if present), metrics summary       │
 │    (token efficiency, latency, GPU).                             │
 └─────────────────────────────────────────────────────────────┘
        ▼
 ┌─────────────────────────────────────────────────────────────┐
 │ 6. REPORT  (report_node — 5 sequential LLM authoring calls,     │
 │    enable_thinking=False, each grounded in the dashboard data)  │
 │    executive_summary → detailed_findings → remediation_plan →   │
 │    incident_notifications → monitoring_recommendations           │
 └─────────────────────────────────────────────────────────────┘
        ▼
   Dashboard + Report served via REST; user can now open the
   "Chat" tab and ask questions about the results.
        ▼
 ┌─────────────────────────────────────────────────────────────┐
 │ 7. CHAT ("talk to results")  intent → [conditional] → tools     │
 │    → synthesis. Tools query the completed InvestigationState   │
 │    (get_findings_by_category, get_entry_detail,                 │
 │    get_risk_distribution, explain_finding, compare_categories,  │
 │    get_accuracy_metrics).                                        │
 └─────────────────────────────────────────────────────────────┘
```

### 4.1 Triage details (`app/packs/governance/triage.py`, `pii_agent.py`)

- **PII** — `pii_agent.scan_entry()` runs a Presidio `AnalyzerEngine` (spaCy model
  `en_core_web_sm`, configurable via `PRESIDIO_SPACY_MODEL`) over `user_prompt` and
  `ai_response`, looking for `PERSON, EMAIL_ADDRESS, PHONE_NUMBER, CREDIT_CARD, US_SSN,
  US_BANK_NUMBER, IBAN_CODE, LOCATION` above `PII_SCORE_THRESHOLD=0.4`. This is **free**
  (no LLM) and runs on every entry — it doubles as a standalone "pii" finding in
  `specialist_findings` (always `flagged=True` when PII is found, severity escalated for
  `PII_CRITICAL_ENTITIES = {US_SSN, CREDIT_CARD, US_BANK_NUMBER, IBAN_CODE}`).
- **Injection** — a regex bank (`INJECTION_PATTERNS`, 14 patterns covering instruction
  override, persona hijack, system-prompt extraction, "developer mode", delimiter
  injection, etc.) scans `user_prompt` → `injection_suspect`.
- **Compliance** — a regex bank scans `ai_response` for financial-advice language
  (guarantees, "can't lose", 401k/investment/loan/refinance specifics) or medical-advice
  language (specific drug names/dosages) → `compliance_suspect`, tagged with a category
  (`financial_advice` | `medical_advice`).
- **Hallucination candidate** — `has_context = bool(retrieved_context)`.

### 4.2 Dispatch plan (`app/packs/governance/dispatch.py`)

For each entry, `dispatch_plan_fn` proposes `{"log_id", "agent"}` pairs (an entry can
appear multiple times for different agents):

| Triage signal | Agent dispatched |
|---|---|
| `injection_suspect` | `security` |
| `compliance_suspect` | `compliance` |
| `has_context` | `hallucination` |

This proposed plan is handed to the orchestrator LLM as context for its
`investigation_plan` decision.

### 4.3 Specialist agents

| Agent | Trigger (triage) | Prompt asks the LLM to... | `response_schema` | `SpecialistFinding` |
|---|---|---|---|---|
| **Security** (`security_agent.py`) | `injection_suspect` (regex hit on prompt) | Decide whether the user prompt is a prompt-injection attempt, name the technique, give a confidence score | `"security_verdict"` → `{is_injection, technique, confidence}` | `agent="security"`, `flagged=is_injection`, `severity="medium"` if flagged |
| **Compliance** (`compliance_agent.py`) | `compliance_suspect` (financial/medical keyword hit) | Given top-k retrieved policy excerpts (ChromaDB RAG over `app/data/policies/*.md`, `all-MiniLM-L6-v2` embeddings), decide if the AI response violates policy, cite the policy & clause | `"compliance_verdict"` → `{violates, policy, clause, explanation}` | `agent="compliance"`, `flagged=violates` |
| **Hallucination** (`hallucination_agent.py`) | `has_context` (non-null `retrieved_context`) | Given the retrieved context and the AI response, decide if every factual claim is grounded; list unsupported claims | `"hallucination_verdict"` → `{grounded, unsupported_claims, severity}` | `agent="hallucination"`, `flagged=!grounded` |
| **PII** (`pii_agent.py`) | n/a — computed during triage, no specialist call | — | — | `agent="pii"`, `flagged=True` whenever `has_pii` |

All specialist calls use `enable_thinking=False` (these are classification/judgment
calls, not multi-step reasoning) and `json_mode=True`. Responses are parsed via
`app/packs/governance/llm_utils.py::parse_json_response()`, a best-effort JSON
extractor that tolerates minor LLM formatting noise.

### 4.4 Risk scoring (`app/packs/governance/risk_scoring.py`)

Pure aggregation, weighted-sum rubric from `app/config.py`:

```python
RISK_WEIGHTS = {
    "pii_critical":  40,   # SSN / credit card / bank account / IBAN
    "pii_other":     20,   # other PII (name, email, phone, location)
    "injection":     35,
    "compliance":    30,
    "hallucination": 20,
}
RISK_SEVERITY_THRESHOLDS = [("critical", 70), ("high", 45), ("medium", 20)]
RISK_SEVERITY_DEFAULT = "low"
```

For each `log_id`: start from the triage PII signal (critical entity → `+40`, other PII
→ `+20`), then add each flagged specialist's weight. The summed score is bucketed into
`low | medium | high | critical` via the thresholds above (first threshold the score
meets/exceeds, scanning from `critical` down). Output:

```python
{
  "by_log_id": {"<log_id>": {"score": int, "severity": str, "contributors": [str, ...]}},
  "risk_distribution": {"low": n, "medium": n, "high": n, "critical": n},
  "overall_risk_score": float,   # mean score across all entries
}
```

### 4.5 Dashboard (`app/packs/governance/dashboard.py`)

```python
{
  "total_entries": int,
  "total_flagged": int,                # entries with severity != "low"
  "findings_by_category": {
    "pii": {"flagged": int, "total": int},
    "security": {"flagged": int, "total": int},
    "compliance": {"flagged": int, "total": int},
    "hallucination": {"flagged": int, "total": int},
  },
  "risk_distribution": {...},
  "overall_risk_score": float,
  "top_findings": [...],               # highest-severity entries, sorted desc
  "accuracy": {...} | None,            # only when ground_truth.csv is available
  "metrics": {...},                    # MetricsCollector.summary(num_entries)
}
```

### 4.6 The 5-section report (`app/packs/governance/report.py`)

Each section is one LLM authoring call (`enable_thinking=False`), grounded in the
dashboard data via a shared `REPORT_SECTION_PROMPT` template and a content-aware mock
fabricator for `LLM_MODE=mock`:

1. **`executive_summary`** — for leadership: headline numbers, top 2-3 findings, the
   token-efficiency callout (e.g. "73% fewer specialist LLM calls than the naive
   baseline").
2. **`detailed_findings`** — every flagged entry, grouped by category (pii / security /
   compliance / hallucination), with `log_id`-linked evidence.
3. **`remediation_plan`** — a 30/60/90-day action plan ordered by severity.
4. **`incident_notifications`** — draft internal notifications (to a Data Protection
   Officer / Security team) for the top critical/high findings.
5. **`monitoring_recommendations`** — productionizable monitoring rules derived from
   *this run's* findings (e.g. "alert on CREDIT_CARD/US_SSN Presidio hits on every
   entry"; "quarterly refresh of the policy RAG corpus").

### 4.7 "Talk to results" chat layer (`app/agent/chat_nodes.py`)

A 3-node LangGraph (`intent → [conditional] → tools → synthesis`, mirroring the
original PulseIQ survey-chat pattern but operating over the **completed
`InvestigationState`** instead of a raw DataFrame). Tool registry
(`app/packs/governance/tool_registry.py`):

| Tool | Purpose |
|---|---|
| `get_findings_by_category` | All flagged entries for one category (pii/security/compliance/hallucination) |
| `get_entry_detail` | Full detail for one `log_id` (entry, triage, specialist findings, risk) |
| `get_risk_distribution` | Dataset-wide risk distribution + overall score |
| `explain_finding` | Why a specific `log_id` scored what it did (contributors + explanations) |
| `compare_categories` | Side-by-side flagged/total counts per category |
| `get_accuracy_metrics` | MetricsCollector totals + token-efficiency numbers |

---

## 5. Survey Analytics Pack — Pluggability Proof (`app/packs/survey/`)

This is the "Day 6 stretch goal": the **same investigation graph**, with zero changes to
`app/agent/investigation_graph.py`'s node logic, runs over employee-survey data (the
original PulseIQ dataset shape — `Department, Satisfaction, NPS, Comments, ...`).

| Seam | Survey pack's implementation |
|---|---|
| `entries_fn` | `survey_entries_fn` — each row → one `LogEntry`; the whole row is JSON-serialized into `ai_response` (so the Presidio scanner can catch PII leaking into open-text comments); any numeric column that's a statistical outlier for that row (`\|z\| > ANOMALY_Z_THRESHOLD`) is recorded as JSON in `retrieved_context` |
| `triage_fn` | Reuses `pii_agent.scan_entry()` unchanged for PII; repurposes `compliance_suspect` to mean "this row has \>= 1 statistically anomalous numeric answer" (i.e. `retrieved_context` is non-empty) |
| `specialists` | One agent, named **`"compliance"`** — an "Insight Agent" (`insight_agent.py`) that reviews flagged outlier rows and judges whether the response is a genuine actionable signal vs. noise |
| `dispatch_plan_fn`, `risk_scoring_fn`, `dashboard_fn` | **Reused directly from the governance pack**, unmodified |
| `report_sections` | One section, `survey_insights_summary` — reconstructs the survey DataFrame from `entries`, re-derives a schema, and runs the *existing* survey tools (`get_segment_stats`, `flag_anomalies`) to produce a grounded "Risk Distribution / Segment Highlight / Outliers / PII" markdown report |
| `chat_tool_registry` / `chat_tool_functions` | Reuses the original 5-tool survey registry (`app/tools/registry.py`) unchanged |

**Naming the specialist `"compliance"`** (instead of, say, `"insight"`) was the key
trick: `MetricsCollector.GATED_SPECIALIST_AGENTS = ("security", "compliance",
"hallucination")` and the governance `risk_scoring_fn` / `dashboard_fn` /
`dispatch_plan_fn` are all keyed on those three agent names — by reusing one of them,
the Survey pack gets correct risk scores, dashboard category counts, and
token-efficiency metrics **for free**.

Verified by 18 tests in `tests/test_survey_pack.py` (AgentPack shape, entries mapping,
triage, dispatch plan, and a full `run_investigation()` end-to-end check).

---

## 6. Synthetic Dataset & Accuracy Evaluation

`scripts/generate_synthetic_logs.py` (seeded, `SEED=42`) builds a fictional
**"InsightBot" enterprise-assistant log batch** of **136 entries**
(`app/data/synthetic_logs/logs.csv`, columns: `log_id, timestamp, user_prompt,
ai_response, retrieved_context, model_name`), covering:

- Clean IT-helpdesk, HR (grounded), and general interactions (no violations)
- **PII leaks** — names, emails, phone numbers, SSNs, credit card numbers (Luhn-valid),
  bank accounts, IBANs, embedded in prompts/responses via templated generators
- **Prompt-injection attempts** — 10 distinct techniques ("ignore previous
  instructions", persona hijacks, system-prompt extraction, "developer mode", etc.)
- **Hallucinations** — paired `retrieved_context` (real HR policy facts: PTO accrual,
  parental leave, 401k match, tuition reimbursement, etc.) where `ai_response`
  contradicts or embellishes the context
- **Compliance violations** — specific financial advice (guaranteed returns, "can't
  lose" investment tips) and medical advice (dosages/specific drugs)
- **Combo entries** — multiple violation categories in one entry

A parallel `ground_truth.csv` (`log_id, has_pii, pii_types, has_injection,
has_hallucination, has_compliance_violation, violation_category, severity`) is **never
passed to any agent** — it's used only by:

- `app/packs/governance/accuracy.py::precision_recall_f1(predicted, actual)` — standard
  TP/FP/FN → precision/recall/F1, computed per category (pii, security, compliance,
  hallucination) and surfaced in `dashboard["accuracy"]`.
- `tests/test_governance_pack.py` — asserts precision/recall/F1 stay above a floor for
  the seeded dataset.

The "Load demo dataset" button in the Aegis UI loads exactly this file via
`POST /api/governance/demo`.

---

## 7. API Reference

Mounted by `app/main.py`: `app.include_router(api_router, prefix="/api")` (survey) and
`app.include_router(governance_router, prefix="/api")` where `governance_routes.py`
itself has `prefix="/governance"` → final paths are `/api/governance/...`.

### 7.1 Survey pack routes (`app/api/routes.py`, prefix `/api`)

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/upload` | Upload a survey CSV → creates a session, returns `session_id` + inferred schema |
| `POST` | `/api/chat` | Send a chat message about the uploaded survey (non-streaming) |
| `POST` | `/api/chat/stream` | Same, but SSE-streamed word-by-word |
| `GET` | `/api/sessions` | List active sessions |
| `GET` | `/api/sessions/{session_id}` | Session details |
| `DELETE` | `/api/sessions/{session_id}` | Delete a session |
| `GET` | `/api/health` | Health check (sessions count) |

### 7.2 Governance ("Aegis") routes (`app/api/governance_routes.py`, prefix `/api/governance`)

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/governance/upload` | Upload a CSV/JSON log batch → session via `load_log_batch()` |
| `POST` | `/api/governance/demo` | Load the seeded synthetic dataset (`logs.csv`) as a demo session |
| `POST` | `/api/governance/investigate/{session_id}` | Kick off the investigation graph as a background task → returns `run_id` |
| `GET` | `/api/governance/status/{run_id}` | Poll run status (`running` / `complete` / `error`) + progress |
| `GET` | `/api/governance/stream/{run_id}` | SSE stream of per-node progress events (triage → orchestrator → specialist_dispatch → risk_scoring → dashboard → report → complete) |
| `GET` | `/api/governance/dashboard/{run_id}` | Dashboard JSON (§4.5 shape) |
| `GET` | `/api/governance/report/{run_id}` | The 5 report sections |
| `GET` | `/api/governance/metrics/{run_id}` | `MetricsCollector` summary (tokens, latency, efficiency, GPU) |
| `POST` | `/api/governance/chat/{run_id}` | "Talk to results" chat turn over the completed investigation |

### 7.3 Top-level

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Basic liveness check (`{"status": "healthy", "service": "pulseiq-mvp"}`) |
| `GET` | `/` | Serves `frontend/` (StaticFiles mount, `html=True`) |

State for governance runs lives in `RunStore` (`app/session/run_store.py`) — an
in-memory, thread-safe singleton keyed by `run_id`, tracking `status`, `progress`
events, the final `InvestigationState` result, and per-run `chat_history`. (Like
`SessionStore`, this is **in-memory and lost on restart** — acceptable for a hackathon
demo, called out in §13.)

---

## 8. Frontend (`frontend/`, static HTML/JS/CSS served by FastAPI's `StaticFiles`)

| File | Provides |
|---|---|
| `index.html` + `aegis.js` + `aegis.css` | **Aegis governance dashboard**: drag-drop upload (or "Load demo dataset"), live SSE progress through the 6 investigation steps, a results view with **Dashboard** (risk-distribution chart, findings-by-category, top findings, efficiency callout), **Report** (5 tabbed sections), and **Chat** (talk to the findings, with tool-call badges and follow-up suggestions) |
| `survey.html` + `app.js` + `style.css` | **PulseIQ survey chat**: upload a survey CSV, then chat about it (suggestion chips, streaming responses, evidence/tool-call detail panels) |

---

## 9. Constraints & Design Decisions

### 9.1 Build environment

- **No AMD GPU locally** → the entire app must run, and be testable, on plain Windows
  CPU. This is the reason `LLM_MODE=mock` exists at all and is the default.
- **`LLM_MODE` env var** (`app/config.py`) is the single switch:
  - `mock` (default) — every `call_llm_async`/`call_llm` returns a content-aware,
    schema-shaped canned JSON response with **no network/GPU dependency**. The entire
    investigation graph, report generation, and chat layer run end-to-end this way —
    this is how all 85 tests run in CI / on a laptop.
  - `vllm` — routes through an OpenAI-compatible client at `VLLM_BASE_URL` (default
    `http://localhost:8000/v1`), with per-agent model overrides (all default to
    `Qwen/Qwen3-8B`): `VLLM_MODEL_INTENT`, `VLLM_MODEL_SYNTHESIS`, `VLLM_MODEL_THEMES`,
    `VLLM_MODEL_ORCHESTRATOR`, `VLLM_MODEL_SPECIALIST`, `VLLM_MODEL_REPORT`.
  - Qwen3's **thinking mode** is opt-in per call (`enable_thinking=True`/`False`) via
    `extra_body={"chat_template_kwargs": {"enable_thinking": True}}` — used for the
    orchestrator (which needs to reason about a dispatch plan), and turned off for
    classification/authoring calls to save tokens.
- **No external LLM APIs** — `requirements.txt` depends only on the `openai` package as
  a *protocol client* (vLLM exposes an OpenAI-compatible HTTP API); no API key is
  required (`VLLM_API_KEY=not-needed`).

### 9.2 Data & sizing limits (`app/config.py`)

| Constant | Default | Purpose |
|---|---|---|
| `MAX_CSV_ROWS` | 50,000 | Upload row cap |
| `MAX_CSV_MB` | 50 | Upload size cap |
| `SESSION_MAX_AGE_HOURS` | 4 | Session TTL (background eviction thread) |
| `SESSION_MAX_HISTORY` | 10 | Chat turns retained per session |
| `ANOMALY_Z_THRESHOLD` | 2.0 | Outlier z-score threshold (survey pack) |
| `MIN_SEGMENT_SIZE` | 10 | Minimum rows per segment for segment comparisons |
| `PII_SCORE_THRESHOLD` | 0.4 | Presidio confidence floor |
| `PRESIDIO_SPACY_MODEL` | `en_core_web_sm` | Smallest viable spaCy model — keeps install footprint hackathon-friendly |
| `LLM_REQUEST_TIMEOUT` | 60s | vLLM request timeout |

### 9.3 Risk-weighting constants are centralized

`RISK_WEIGHTS`, `RISK_SEVERITY_THRESHOLDS`, `PII_CRITICAL_ENTITIES`, `PII_ENTITIES`,
`ANOMALY_Z_THRESHOLD`, `MIN_SEGMENT_SIZE` all live in `app/config.py` and are shared
verbatim between the governance and survey packs — one rubric, two domains.

### 9.4 Hackathon "avoid list" compliance

The product is an **audit/investigation pipeline with a chat add-on**, not a
chatbot/RAG/PDF-Q&A tool. The compliance specialist *does* use RAG internally (ChromaDB
over policy docs) but that's one judgment input to a larger pipeline (triage → dispatch
→ risk scoring → report), not the product's interface.

---

## 10. How to Run

### 10.1 Local development — mock mode (Windows, no GPU)

```powershell
cd pulseiq-mvp
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python -m spacy download en_core_web_sm   # required by the PII agent

copy .env.example .env       # LLM_MODE=mock by default — no further edits needed

uvicorn app.main:app --reload --host 127.0.0.1 --port 8001
```

Then open `http://127.0.0.1:8001/`:

- **Aegis dashboard** (`index.html`) — click "Load demo dataset" (uses the seeded
  `app/data/synthetic_logs/logs.csv`) or upload your own CSV/JSON log batch, then watch
  the live SSE progress through triage → orchestrator → specialist dispatch → risk
  scoring → dashboard → report, and explore the Dashboard / Report / Chat tabs.
- **PulseIQ survey chat** (`survey.html`) — upload a survey CSV (e.g.
  `tests/fixtures/sample_survey.csv`) and chat about it.

Run the test suite:

```powershell
python -m pytest tests/ -q
# 85 passed
```

### 10.2 AMD Developer Cloud — vLLM mode

On the AMD notebook (ROCm + vLLM already available):

```bash
# 1. Start a vLLM OpenAI-compatible server with Qwen3-8B
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-8B \
    --port 8000

# 2. Pull the repo (already pushed from Windows)
git clone <repo-url> && cd pulseiq-mvp
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 3. Point the app at the vLLM server
cp .env.example .env
#   edit .env:
#     LLM_MODE=vllm
#     VLLM_BASE_URL=http://localhost:8000/v1
#     VLLM_MODEL_ORCHESTRATOR=Qwen/Qwen3-8B   (and the other VLLM_MODEL_* vars,
#                                              all already default to Qwen/Qwen3-8B)

# 4. Run the app — identical command, identical code
uvicorn app.main:app --host 0.0.0.0 --port 8001
```

No code changes are required between the two environments — only the `.env` file
differs. The `MetricsCollector`'s `gpu_stats()` will now return real `rocm-smi` output
(GPU utilization + VRAM) instead of `{"gpu_available": false}`, populating the metrics
panel with real numbers for the demo.

**Smoke test after switching to vLLM mode**: re-run `python -m pytest tests/ -q` with
`LLM_MODE=vllm` set — this exercises real Qwen3 calls through every agent (orchestrator
with `enable_thinking=True`, specialists with `json_mode=True`, report authoring,
chat intent/synthesis) and confirms the JSON responses parse correctly against each
`response_schema`.

---

## 11. Testing

**85/85 tests passing** (`python -m pytest tests/ -q`, `LLM_MODE=mock`):

| File | Covers |
|---|---|
| `test_session.py` | `SessionStore` CRUD, TTL eviction |
| `test_csv_loader.py` | CSV/JSON loading, schema inference, `load_log_batch`, `df_to_log_entries` |
| `test_tools.py` | The 5 survey analytics tools (segment stats, trends, themes, anomalies, recommendations) |
| `test_chat_graph.py` | Original PulseIQ "talk to your CSV" chat graph (intent → tool → synthesis) |
| `test_governance_pack.py` | Per-agent precision/recall/F1 vs. `ground_truth.csv` |
| `test_investigation_graph.py` | Full governance pipeline end-to-end (incl. the clean-dataset fast path, efficiency metrics, dashboard, report) |
| `test_governance_routes.py` | `/api/governance/*` via `TestClient`: upload → investigate → poll → dashboard/report/chat |
| `test_survey_pack.py` | Survey `AgentPack` shape, entries mapping, triage, dispatch plan, full end-to-end run (18 tests) |

---

## 12. Known Limitations / Future Work

- **In-memory state** — `SessionStore` and `RunStore` are process-local singletons;
  restarting the server loses all sessions/runs. Fine for a demo, would need Redis/DB
  for production.
- **Per-entry routing, not per-(entry, agent) graph branching** — the conditional edge
  is dataset-level (flagged vs. clean), and fan-out to specialists happens via
  `asyncio.gather` inside one node rather than LangGraph `Send`-based dynamic graph
  branching. A `Send`-based implementation would make each specialist call a first-class
  graph node (better observability/streaming per specialist) — noted as a documented
  next step.
- **Policy RAG corpus is small** (4 seed policy docs) — sufficient to demonstrate the
  retrieval → judge pattern; a production deployment would index a real policy library.
- **Synthetic dataset is templated/seeded** — good for reproducible accuracy metrics,
  but a real deployment's accuracy numbers would come from a held-out sample of real
  (redacted) interaction logs.
- **Survey pack's "compliance"-named specialist** is a deliberate metrics-reuse hack
  (see §5) — clean, but if a third pack needed *both* a real compliance agent *and* a
  differently-named specialist, `GATED_SPECIALIST_AGENTS` would need to become
  per-pack configuration rather than a single global tuple.
