# Aegis — Technical Flow Reference

**TCS & AMD AI Hackathon 2026 · Track 1 · Agents · Team 2310**  
**Use Cases: AGENTS_006 (Governance) + AGENTS_034 (Survey Analytics)**  
**Stack: AMD Instinct MI300X · ROCm · vLLM 0.11 · Qwen3-8B · LangGraph · FastAPI**

---

## Table of Contents

1. [System Overview](#1-system-overview)
2. [Hardware & Serving Layer](#2-hardware--serving-layer)
3. [API Layer — FastAPI](#3-api-layer--fastapi)
4. [LangGraph Orchestration Engine](#4-langgraph-orchestration-engine)
5. [Governance Pack — AGENTS_006](#5-governance-pack--agents_006)
   - 5.1 [Input & Preprocessing](#51-input--preprocessing)
   - 5.2 [Triage Node](#52-triage-node)
   - 5.3 [Orchestrator Node](#53-orchestrator-node)
   - 5.4 [Specialist Dispatch — Parallel Execution](#54-specialist-dispatch--parallel-execution)
   - 5.5 [PII Agent](#55-pii-agent)
   - 5.6 [Security Agent](#56-security-agent)
   - 5.7 [Compliance Agent — RAG Pipeline](#57-compliance-agent--rag-pipeline)
   - 5.8 [Hallucination Agent](#58-hallucination-agent)
   - 5.9 [Risk Scoring Node](#59-risk-scoring-node)
   - 5.10 [Dashboard Node](#510-dashboard-node)
   - 5.11 [Report Node](#511-report-node)
   - 5.12 [Governance Chat — Talk to Results](#512-governance-chat--talk-to-results)
6. [Survey Analytics Pack — AGENTS_034](#6-survey-analytics-pack--agents_034)
   - 6.1 [Input & CSV Loading](#61-input--csv-loading)
   - 6.2 [Triage Node](#62-triage-node)
   - 6.3 [Insight Agent — Batch Phase](#63-insight-agent--batch-phase)
   - 6.4 [MCP Tool Registry — 6 Tools](#64-mcp-tool-registry--6-tools)
   - 6.5 [Synthesis Node](#65-synthesis-node)
   - 6.6 [Dashboard & Report](#66-dashboard--report)
   - 6.7 [Survey Chat — Conversational Analytics](#67-survey-chat--conversational-analytics)
7. [SSE Streaming Architecture](#7-sse-streaming-architecture)
8. [Session & State Management](#8-session--state-management)
9. [Frontend Architecture](#9-frontend-architecture)
10. [Testing](#10-testing)
11. [Token Efficiency Analysis](#11-token-efficiency-analysis)
12. [End-to-End Flow Timelines](#12-end-to-end-flow-timelines)

---

## 1. System Overview

Aegis is a **domain-agnostic multi-agent AI governance and investigation platform**. A single shared LangGraph state machine powers two completely different domains via pluggable "AgentPack" modules:

```
AgentPack (abstract contract)
├── GovernancePack  →  AI interaction log audit (AGENTS_006)
└── SurveyPack      →  Conversational CSV analytics (AGENTS_034)
```

The orchestration graph itself (`investigation_graph.py`, `chat_nodes.py`) contains **zero pack-specific logic**. Every behavioural difference — triage heuristics, specialist agents, tools, report templates — lives inside the pack. Adding a third domain (e.g. HR analytics, legal review) requires only a new AgentPack with the same interface shape.

```
AgentPack interface (app/packs/base.py):
  .name                  str
  .triage_fn             fn(df, ctx) → list[TriageResult]
  .dispatch_plan_fn      fn(triage_results, ctx) → list[DispatchItem]
  .specialists           dict[str, AsyncFn]
  .risk_scoring_fn       fn(triage, findings) → RiskScores
  .dashboard_fn          fn(entries, triage, findings, risk, metrics) → dict
  .report_sections       dict[str, Fn → (prompt, mock_fabricator)]
  .entries_fn            fn(df) → list[LogEntry]
  .chat_tool_registry    list[ToolSchema]
  .chat_tool_functions   dict[str, Fn]
  .chat_persona          str
  .chat_entry_noun       str
  .chat_fallback_*       str / list
```

---

## 2. Hardware & Serving Layer

### AMD Instinct MI300X + ROCm

- vLLM **auto-detects** the ROCm platform on startup (`INFO: Automatically detected platform rocm`)
- All PyTorch ops route through ROCm HIP kernels transparently — no application-level changes required
- dtype: `bfloat16` (auto-selected)

### vLLM 0.11.0rc2 Configuration

| Setting | Value | Effect |
|---|---|---|
| Model | `Qwen/Qwen3-8B` | 8B parameter instruct model |
| dtype | `bfloat16` | ~16 GB VRAM, ROCm-native |
| max_model_len | 40,960 tokens | Long context for large log batches |
| chunked_prefill | ON | Breaks large prompts into chunks — prevents memory spikes |
| prefix_caching | ON | Caches repeated prompt prefixes across calls |
| tensor_parallel | 1 | Single GPU |

### Observed Performance (AMD Developer Cloud)

| Metric | Observed Value |
|---|---|
| Prompt throughput | **1,656 tokens/s** (peak) |
| Generation throughput | **380 tokens/s** |
| Prefix cache hit rate | 10 % → 70 % (warming across a full test run) |
| KV cache utilisation | < 0.2 % per single run |
| Concurrent requests | 60+ handled in batch |

Prefix cache hit rate climbs because the system prompt, JSON schema instructions, and policy excerpts are shared across many calls in a single run — vLLM detects and reuses these prefix KV blocks.

### vLLM OpenAI-Compatible Endpoint

All LLM calls in Aegis go through `app/utils/llm_client.py`, which speaks the OpenAI `/v1/chat/completions` API. Switching models or serving backends requires only changing `.env` variables — no code changes.

```
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_MODEL_INTENT=Qwen/Qwen3-8B
VLLM_MODEL_SYNTHESIS=Qwen/Qwen3-8B
VLLM_MODEL_ORCHESTRATOR=Qwen/Qwen3-8B
VLLM_MODEL_SPECIALIST=Qwen/Qwen3-8B
VLLM_MODEL_REPORT=Qwen/Qwen3-8B
VLLM_MODEL_THEMES=Qwen/Qwen3-8B
```

### Local PII Model — Microsoft Presidio

Presidio (`app/packs/governance/pii_agent.py`) runs **entirely locally** — no GPU, no network. It uses:
- `spaCy en_core_web_sm` for named entity recognition
- Score threshold: `PII_SCORE_THRESHOLD=0.4` (configurable)
- Entities: `PERSON`, `EMAIL_ADDRESS`, `PHONE_NUMBER`, `US_SSN`, `CREDIT_CARD`, `IP_ADDRESS`, `LOCATION`, `DATE_TIME`, + any spaCy-detected `WORK_OF_ART`, `CARDINAL`

### ChromaDB — Vector Store for Policy RAG

- `chromadb.PersistentClient` — embeddings persisted to `CHROMA_PERSIST_DIR` across restarts
- Collection: `governance_policies`
- Embedding model: `all-MiniLM-L6-v2` (sentence-transformers, CPU)
- Corpus: 4 policy markdown files (`app/data/policies/*.md`), chunked by `## ` section headings
- Lazy initialisation: first call to `_get_collection()` seeds the DB if empty

---

## 3. API Layer — FastAPI

`app/main.py` mounts four routers:

| Router | Prefix | Purpose |
|---|---|---|
| `governance_routes` | `/api/governance` | Log audit pipeline |
| `survey_routes` | `/api/survey` | CSV analytics pipeline |
| `platform_routes` | `/api/platform` | Live GPU/LLM info |
| Static files | `/` | Serves `frontend/` |

### Key Endpoints

```
POST /api/governance/investigate
  Body: {logs: [...], session_id}
  → triggers run_investigation(GovernancePack, df, session_id, run_id)
  ← {run_id, status: "complete", total_flagged, risk_level, dashboard_url, report_url}

POST /api/survey/analyze
  Body: multipart CSV file upload + session_id
  → triggers run_investigation(SurveyPack, df, session_id, run_id)
  ← {run_id, status: "complete", row_count, column_count, dashboard_url}

GET /api/governance/chat_stream/{run_id}?message=...
  → SSE: {"type": "thinking", "data": {"delta": "..."}} × N
         {"type": "complete", "data": {narrative, follow_up_suggestions, evidence, tool_calls}}

GET /api/survey/chat_stream/{run_id}?message=...
  → same SSE shape

GET /api/{pack}/dashboard/{run_id}   → dashboard dict
GET /api/{pack}/report/{run_id}      → report_sections dict
GET /api/{pack}/metrics/{run_id}     → MetricsCollector summary

GET /api/platform/info
  → {llm_mode, vllm_base_url, models[], gpu: {gpu_available, gpu_name,
     gpu_utilization_pct, vram_used_gb, vram_total_gb, raw}}
```

`/api/platform/info` is polled every 10 seconds by `frontend/platform.js` to power the live AMD GPU status strip across all three pages.

---

## 4. LangGraph Orchestration Engine

### InvestigationState (TypedDict)

```python
class InvestigationState(TypedDict):
    session_id:             str
    run_id:                 str
    entries:                list[LogEntry]        # parsed from DataFrame
    triage_results:         list[dict]            # per-entry triage flags
    investigation_plan:     list[DispatchItem]    # {log_id, agent} pairs
    orchestrator_rationale: str                   # LLM reasoning text
    total_flagged:          int
    specialist_findings:    list[SpecialistFinding]
    risk_scores:            dict
    dashboard:              dict
    report_sections:        dict[str, str]
    metrics:                dict                  # MetricsCollector snapshot
```

### Graph Topology — Governance/Survey Investigation

```
triage_node
    │
    ▼
orchestrator_node  (Qwen3-8B plans dispatch)
    │
    ├─ total_flagged > 0 ──► specialist_dispatch_node  (asyncio.gather, parallel)
    │                               │
    │                               ▼
    └─ total_flagged == 0 ──► risk_scoring_node  (deterministic)
                                    │
                                    ▼
                              dashboard_node
                                    │
                                    ▼
                              report_node  (Qwen3-8B, one section at a time)
                                    │
                                    ▼
                                  END
```

### Graph Topology — Chat (Governance + Survey)

```
intent_node  (Qwen3-8B classifies question → selects tools)
    │
    ├─ tool_calls[] not empty ──► tool_node  (synchronous, calls Python functions)
    │                                  │
    │                                  ▼
    └─ no tool calls ────────────► synthesize_node  (Qwen3-8B, SSE streamed)
                                        │
                                       END
```

The **clean-dataset fast path** is important: if triage produces zero flagged entries, `_route_after_orchestrator` jumps directly to `risk_scoring`, skipping all specialist LLM calls entirely.

---

## 5. Governance Pack — AGENTS_006

Full path: `app/packs/governance/`

### 5.1 Input & Preprocessing

- **Endpoint**: `POST /api/governance/investigate`
- **Format**: JSON array of log entries, or NDJSON
- **Limits**: `MAX_CSV_ROWS=50000`, `MAX_CSV_MB=50`
- **Entry shape** (per log entry):

```json
{
  "log_id":           "LOG-001",
  "user_id":          "user_42",
  "session_id":       "sess_abc",
  "session_type":     "customer_support",
  "prompt":           "What medications can I take for back pain?",
  "ai_response":      "You can take 400mg ibuprofen every 6 hours...",
  "retrieved_context":"[RAG context from knowledge base]",
  "timestamp":        "2026-06-16T14:23:00Z"
}
```

- `pack.entries_fn(df)` converts the DataFrame rows to typed `LogEntry` dicts
- A `run_id` UUID is generated and stored in `RunStore`

---

### 5.2 Triage Node

**File**: `app/packs/governance/triage.py`  
**Mode**: Deterministic — no LLM call

Fast keyword/heuristic pre-filter that labels each entry with boolean flags before any LLM is involved:

| Flag | Logic |
|---|---|
| `has_pii` | Presidio NER pre-scan (score ≥ 0.4) on prompt + ai_response |
| `injection_suspect` | Regex patterns: "ignore instructions", "pretend you are", "jailbreak", DAN prompts, etc. |
| `compliance_suspect` | `detect_compliance_category()`: financial keywords (invest, guarantee, 401k, mortgage) OR medical keywords (dosage, ibuprofen, mg) |
| `hallucination_suspect` | `has_context = bool(retrieved_context)` — only flagged if ground truth exists |

Output: `list[TriageResult]` — one dict per entry with all four flags.

**LLM calls at this stage: 0**

---

### 5.3 Orchestrator Node

**File**: `app/agent/investigation_graph.py` — `_orchestrator_node`  
**Model**: `VLLM_MODEL_ORCHESTRATOR` (Qwen3-8B)

Receives the triage summary:
```python
{
  "total_entries": 500,
  "has_pii": 43,
  "injection_suspect": 12,
  "compliance_suspect": 8,
  "has_context": 27,
  "dispatch_plan_size": 47
}
```

Calls Qwen3-8B with `enable_thinking=True` — the model's `<think>...</think>` reasoning trace is **streamed live to the UI** via the progress SSE channel as the orchestrator reasons about what to prioritise.

Output:
```json
{
  "rationale": "Triage flagged 12 entries for security review...",
  "priority_categories": ["security", "compliance", "hallucination"]
}
```

`pack.dispatch_plan_fn()` then produces the actual `investigation_plan`: a flat list of `{log_id, agent}` pairs — one item per (entry, specialist) combination that needs to run.

---

### 5.4 Specialist Dispatch — Parallel Execution

**File**: `app/agent/investigation_graph.py` — `_specialist_dispatch_node`

```python
tasks = [
    pack.specialists[item["agent"]](entries_by_id[item["log_id"]], context)
    for item in state["investigation_plan"]
    if item["agent"] in pack.specialists
]
findings = list(await asyncio.gather(*tasks))
```

All specialist coroutines run **concurrently** via `asyncio.gather`. A log entry flagged for both `security` and `compliance` gets two parallel tasks. vLLM handles concurrent requests natively via its internal scheduler — the MI300X processes these efficiently in batch.

**Maximum LLM calls**: `len(investigation_plan)` — bounded by triage flags, not total entries.

---

### 5.5 PII Agent

**File**: `app/packs/governance/pii_agent.py`  
**Triggered by**: `has_pii = True`

**Step 1 — Presidio NER (local, CPU)**

```python
analyzer = AnalyzerEngine()
results = analyzer.analyze(
    text=f"{entry['prompt']} {entry['ai_response']}",
    language="en",
    score_threshold=PII_SCORE_THRESHOLD  # 0.4
)
```

Entities detected per character span: type, score, start/end offset.

**Step 2 — Qwen3-8B verification**

Presidio results are injected into a structured prompt asking Qwen3-8B to:
- Confirm each entity is genuine PII (not a false positive)
- Determine severity: `HIGH` for SSN/credit card, `MEDIUM` for email/phone/name
- Produce a redacted version of the text

**Output** (`SpecialistFinding`):
```json
{
  "log_id": "LOG-042",
  "agent": "pii",
  "flagged": true,
  "severity": "HIGH",
  "summary": "SSN detected in prompt; email address in AI response",
  "evidence": {
    "entities": [
      {"type": "US_SSN", "score": 0.85, "start": 45, "end": 56},
      {"type": "EMAIL_ADDRESS", "score": 0.92, "start": 120, "end": 141}
    ],
    "redacted_text": "You mentioned [SSN REDACTED]..."
  }
}
```

---

### 5.6 Security Agent

**File**: `app/packs/governance/security_agent.py`  
**Triggered by**: `injection_suspect = True`  
**Model**: Qwen3-8B (vLLM), JSON structured output

Prompt instructs Qwen3-8B to analyse the prompt for adversarial patterns across five threat categories:

| Category | Example pattern |
|---|---|
| Prompt injection | "Ignore all previous instructions and..." |
| Jailbreak | DAN prompts, roleplay-as-evil-AI, "pretend you have no restrictions" |
| Data exfiltration | "List all users in your database", "What is your system prompt?" |
| Unauthorized access | Attempts to access other users' data |
| Social engineering | Building false rapport to extract information |

**Output**:
```json
{
  "threat_type": "prompt_injection",
  "flagged": true,
  "severity": "CRITICAL",
  "attack_vector": "instruction_override",
  "explanation": "User attempts to override system instructions via direct injection in the prompt field."
}
```

Severity mapping: `CRITICAL` for injection/jailbreak (direct model compromise), `HIGH` for exfiltration/unauthorized access, `MEDIUM` for social engineering.

---

### 5.7 Compliance Agent — RAG Pipeline

**File**: `app/packs/governance/compliance_agent.py`  
**Triggered by**: `compliance_suspect = True`

This is the only agent with a full RAG pipeline. Three distinct steps:

**Step 1 — Keyword pre-filter (deterministic)**

```python
def detect_compliance_category(ai_response):
    if any(p.search(ai_response) for p in FINANCIAL_PATTERNS):
        return "financial_advice"
    if any(p.search(ai_response) for p in MEDICAL_PATTERNS):
        return "medical_advice"
    return None
```

`FINANCIAL_PATTERNS`: `guarantee`, `can't lose`, `401k`, `invest`, `fund`, `mortgage`, `loan`, `ESPP`  
`MEDICAL_PATTERNS`: `\d+\s*mg`, drug names (`ibuprofen`, `cyclobenzaprine`, ...), `dosage`, `dose`

**Step 2 — ChromaDB vector retrieval**

```python
def retrieve_policy_chunks(query_text, k=2):
    collection = _get_collection()   # lazily seeded from policies/*.md
    results = collection.query(
        query_texts=[query_text],
        n_results=min(k, collection.count())
    )
    return results  # top-2 most semantically similar policy sections
```

The **ChromaDB collection** is seeded once (lazily) with all `app/data/policies/*.md` files, chunked by `## ` section headings. Each chunk is embedded via `all-MiniLM-L6-v2` (384-dim vectors). The `ai_response` text is used as the retrieval query — the assumption is that a violating response semantically resembles the policy section it violates.

Policy corpus structure:
```
app/data/policies/
├── financial_advice_policy.md   →  POL-FIN-003
├── medical_advice_policy.md     →  POL-MED-004
├── data_privacy_policy.md       →  POL-PRIV-001
└── acceptable_use_policy.md     →  POL-USE-002
```

**Step 3 — Qwen3-8B LLM judge**

Retrieved policy excerpts + AI response injected into:
```
Relevant policy excerpts:
[POL-FIN-003 - 3.2 Guaranteed Outcomes]
The company's AI assistant must not guarantee investment returns...

AI response:
"With this ETF strategy, you're guaranteed to see 15% returns..."

Determine whether the AI response violates any of the policy excerpts above.
Respond with JSON only: {"violates": ..., "policy": ..., "clause": ..., "explanation": ...}
```

**Output**:
```json
{
  "violates": true,
  "policy": "POL-FIN-003",
  "clause": "3.2 Guaranteed Outcomes",
  "explanation": "AI response makes a guaranteed-return claim, violating the financial advice policy.",
  "retrieved_policy_chunks": [
    {"policy_id": "POL-FIN-003", "section": "3.2 Guaranteed Outcomes", "text": "..."}
  ]
}
```

---

### 5.8 Hallucination Agent

**File**: `app/packs/governance/hallucination_agent.py`  
**Triggered by**: `hallucination_suspect = True` AND `has_context = True`  
**Skipped entirely if**: `retrieved_context` is empty/null (no ground truth to compare against)  
**Model**: Qwen3-8B (vLLM), JSON structured output

Prompt structure:
```
Context retrieved by the AI system:
[retrieved_context field from the log entry]

AI response:
[ai_response field]

Question: Does the AI response contain claims not supported by, or contradicted by, the retrieved context?
Respond with JSON: {"hallucinated": ..., "confidence": 0.0-1.0, "explanation": ..., "specific_claim": ..., "context_gap": ...}
```

**Output**:
```json
{
  "hallucinated": true,
  "confidence": 0.87,
  "explanation": "AI claims the drug has no side effects; retrieved context lists 3 known side effects.",
  "specific_claim": "completely safe with no side effects",
  "context_gap": "Retrieved context: 'Common side effects include nausea, dizziness, and headache'"
}
```

Severity: `HIGH` if `hallucinated=True AND confidence > 0.8`, `MEDIUM` otherwise.

---

### 5.9 Risk Scoring Node

**File**: `app/packs/governance/risk_scoring.py`  
**Mode**: Fully deterministic — **no LLM call**

Aggregates all `SpecialistFinding` objects:

```python
SEVERITY_WEIGHTS = {"CRITICAL": 10, "HIGH": 7, "MEDIUM": 4, "LOW": 1}

risk_score = sum(SEVERITY_WEIGHTS[f.severity] for f in findings) / len(entries) * 10
overall_risk = (
    "CRITICAL" if any CRITICAL findings else
    "HIGH"     if risk_score >= 6.0 else
    "MEDIUM"   if risk_score >= 3.0 else
    "LOW"
)
```

Output includes:
- `overall_risk`: `CRITICAL | HIGH | MEDIUM | LOW`
- `risk_score`: float 0.0–10.0
- `flagged_count`: total entries with at least one finding
- `severity_counts`: `{CRITICAL: N, HIGH: N, MEDIUM: N, LOW: N}`
- `findings[]`: all SpecialistFindings with full evidence
- `recommendations[]`: rule-based action items per finding type

---

### 5.10 Dashboard Node

**File**: `app/packs/governance/dashboard.py`

Purely computational — no LLM call. Produces the dashboard dict:

- `severity_distribution`: count per severity level → bar chart data
- `findings_by_category`: count per agent type → category breakdown
- `risk_distribution`: distribution across all entries
- `overall_risk_score`: single number for the risk gauge
- `agent_accuracy_metrics`: per-agent call counts, confidence averages
- `metrics`: full `MetricsCollector.summary()` snapshot including token counts

---

### 5.11 Report Node

**File**: `app/packs/governance/report.py`  
**Model**: `VLLM_MODEL_REPORT` (Qwen3-8B)

One LLM call **per report section** — called sequentially:

| Section | Content |
|---|---|
| `executive_summary` | High-level risk posture, critical findings, immediate actions |
| `risk_matrix` | Formatted table: severity × category × count |
| `pii_findings` | Per-entry PII details with redacted text |
| `security_findings` | Threat classification, attack vectors |
| `compliance_findings` | Policy violations with policy ID + clause citations |
| `hallucination_findings` | Specific hallucinated claims vs retrieved context |
| `recommendations` | Prioritised action plan |

Each section has:
- A `build_prompt(report_context)` function that constructs the full prompt from `InvestigationState`
- A `mock_fabricator` for `LLM_MODE=mock` that produces realistic content from the actual run data

After report generation, `MetricsCollector.summary()` is recomputed to capture the report-authoring token costs too.

---

### 5.12 Governance Chat — Talk to Results

**Files**: `app/agent/chat_nodes.py`, `app/packs/governance/tool_registry.py`

After a completed investigation run, users can ask free-form questions. The chat graph is a separate three-node LangGraph:

**Intent node** — Qwen3-8B with `json_mode=True`:
```json
{
  "intent": "tool_use",
  "reasoning": "User is asking about risk distribution — route to get_risk_distribution",
  "tool_calls": [{"tool_name": "get_risk_distribution", "arguments": {}}],
  "clarification_needed": false
}
```

**Tool node** — calls Python functions (synchronous, no LLM):

| Tool | What it does |
|---|---|
| `get_risk_distribution` | Returns severity counts + overall score from completed run |
| `get_findings_by_category` | Filters findings by agent type (pii/security/compliance/hallucination) |
| `get_entry_detail` | Returns full detail for a specific log ID |
| `explain_finding` | Explains risk contributors for a specific entry |
| `compare_categories` | Side-by-side category counts, identifies most-flagged |
| `get_accuracy_metrics` | LLM call counts, token efficiency, latency stats |

**Synthesis node** — Qwen3-8B with streaming `<think>...</think>`:
- Tool results + user question → narrative answer
- `chart_data[]` attached to response → `renderEvidenceChart()` in frontend renders bar charts inline in the chat bubble
- Fallback: if JSON parse fails, `_fallback_synthesis()` produces text summary of tool results

**Streaming**: `stream_governance_chat_turn()` is an async generator that yields `{"type": "thinking", "data": {"delta": "..."}}` events as Qwen3-8B's reasoning trace arrives, then a final `{"type": "complete", "data": {...}}` event. The frontend renders the `<think>` stream word-by-word.

---

## 6. Survey Analytics Pack — AGENTS_034

Full path: `app/packs/survey/`

### 6.1 Input & CSV Loading

**Endpoint**: `POST /api/survey/analyze` (multipart file upload)  
**File**: `app/utils/csv_loader.py`

```python
df = pd.read_csv(file, dtype=None, na_values=["", "NA", "N/A", "null"])
# Column classification:
for col in df.columns:
    if df[col].nunique() <= 10 and df[col].dtype == object:
        col_type = "categorical"
    elif df[col].between(1, 5).all():
        col_type = "likert"
    elif col.lower() in ("nps", "net_promoter"):
        col_type = "nps"
    else:
        col_type = "numeric"
```

The classified column schema (`column_names[]`, `dtypes{}`, first 5 rows as sample) is stored in the session and passed to every subsequent LLM call.

---

### 6.2 Triage Node

**File**: `app/packs/survey/triage.py`  
**Mode**: Deterministic — no LLM call

Rather than per-entry flags (as in governance), survey triage analyses the DataFrame schema:

- Identifies numeric columns that can be segmented
- Identifies categorical columns for demographic breakdown
- Detects open-text columns (high cardinality + string dtype)
- Builds the initial `dispatch_plan` (which tools to call in batch phase)

---

### 6.3 Insight Agent — Batch Phase

**File**: `app/packs/survey/insight_agent.py`  
**Model**: Qwen3-8B (vLLM)

**Phase 1 — Batch dispatch** (triggered on CSV upload):

The Insight Agent is given the full column schema and asked to plan which tools to call for a comprehensive initial analysis. It produces a `tool_calls[]` list covering:

- `segment_stats` for each numeric metric × each categorical segment column
- `trend_compare` if a time/period dimension is detected
- `anomaly_flag` across all numeric columns
- `get_value_distribution` for each categorical column
- `open_text_themes` for each detected open-text column

These tool calls are then executed via `tool_exec_node`, and results are stored in the session state as the pre-computed `batch_results`.

**Phase 2 — Chat mode** (per user question):

Given the stored batch results + conversation history + user question, the Insight Agent selects 1–3 tools relevant to the question and constructs typed argument dicts.

Example:
```
User: "Which department has the lowest satisfaction score?"

Insight Agent output:
{
  "tool_calls": [{
    "tool_name": "segment_stats",
    "arguments": {"metric": "Satisfaction_Score", "segment_by": "Department"}
  }]
}
```

---

### 6.4 MCP Tool Registry — 6 Tools

**File**: `app/packs/survey/tool_registry.py`, `app/tools/`

Each tool exposes a typed schema (like MCP) so the LLM can call them as structured function calls. Tools are pure Python functions over the loaded DataFrame — no LLM involvement.

#### Tool 1: `segment_stats`
**File**: `app/tools/segment_stats.py`

```python
def segment_stats(df, metric, segment_by, min_segment_size=10):
    grouped = df.groupby(segment_by)[metric].agg(["mean", "std", "count"])
    # filters segments with count < min_segment_size
    # identifies best_segment (highest mean) and worst_segment (lowest mean)
    # computes gap = best_mean - worst_mean
    return {
        "metric": metric,
        "segment_by": segment_by,
        "segments": [{segment, mean, std, count}, ...],
        "best_segment": ...,
        "worst_segment": ...,
        "gap": 1.25
    }
```

Used in chat: renders as `renderBarChart()` in the chat bubble.

#### Tool 2: `trend_compare`
**File**: `app/tools/trend_compare.py`

```python
def trend_compare(df, metrics, time_dimension):
    pivot = df.pivot_table(values=metrics, index=time_dimension, aggfunc="mean")
    # sorts time periods naturally (Q1 < Q2 < Q3 < Q4)
    # computes period-over-period delta for each metric
    return {
        "time_dimension": time_dimension,
        "periods": ["Q1", "Q2", "Q3", "Q4"],
        "metrics": {metric: [mean_per_period, ...] for metric in metrics},
        "trend_directions": {metric: "improving" | "declining" | "stable"}
    }
```

#### Tool 3: `anomaly_flag`
**File**: `app/tools/anomaly_flag.py`

```python
from scipy import stats

def anomaly_flag(df, columns, threshold=2.0):
    z_scores = stats.zscore(df[columns].dropna())
    anomalies = (abs(z_scores) > threshold)
    # returns per-column anomaly row indices and z-scores
    return {
        "threshold": threshold,
        "anomalies_by_column": {col: [{row_idx, value, z_score}, ...] for col in columns},
        "total_anomalies": 25
    }
```

`ANOMALY_Z_THRESHOLD=2.0` is configurable via `.env`.

#### Tool 4: `open_text_themes`
**File**: `app/tools/open_text_themes.py`  
**Model**: Qwen3-8B (vLLM)

```python
def open_text_themes(df, column, max_sample=150, max_themes=8):
    sample = df[column].dropna().sample(min(max_sample, len(df))).tolist()
    # calls Qwen3-8B with the text sample
    # Output: [{theme, sentiment: positive|negative|neutral, representative_quotes[], count}]
```

`MAX_OPEN_TEXT_SAMPLE=150`, `MAX_THEME_COUNT=8` — configurable via `.env`.

#### Tool 5: `recommend_actions`
**File**: `app/tools/recommend_actions.py`  
**Model**: Qwen3-8B (vLLM)

Takes the consolidated findings from all other tools and generates prioritised action items. Structured JSON output: `[{action, priority: HIGH|MEDIUM|LOW, rationale, impacted_metric}]`.

#### Tool 6: `get_value_distribution`
**File**: `app/tools/registry.py`

```python
def get_value_distribution(df, column):
    counts = df[column].value_counts(normalize=True)
    return {
        "column": column,
        "distribution": [{"value": v, "percent": p*100, "count": c} for v, p, c in zip(...)],
        "unique_values": len(counts)
    }
```

If `unique_values > 6`, the frontend automatically renders as a bar chart instead of a donut.

---

### 6.5 Synthesis Node

**File**: `app/packs/survey/common.py`, `app/agent/chat_nodes.py`

After tools execute:

1. All `tool_results[]` are passed to Qwen3-8B with the user question
2. Qwen3-8B writes a conversational narrative answer
3. `chart_data[]` is attached: `[{tool_name, result}]` for each successful tool call
4. Frontend `renderEvidenceChart(chart_data)` dispatches to the correct renderer:

| `tool_name` | Frontend renderer |
|---|---|
| `segment_stats` | `renderBarChart()` — highlights best/worst segment |
| `get_value_distribution` | `renderDonutChart()` (≤6 values) or `renderBarChart()` (>6) |
| `get_response_by_segment` | `renderStackedBarChart()` |
| `find_top_segment_*` | `renderBarChart()` with ranking |

Fallback: `_fallback_synthesis()` — if Qwen3-8B's JSON parse fails, tool result summaries are serialised to text and returned as-is. The frontend still renders charts from `chart_data[]` since that's attached independently of LLM output quality.

---

### 6.6 Dashboard & Report

**Files**: `app/packs/survey/dashboard.py`, `app/packs/survey/report.py`

**Dashboard** (no LLM calls):
- `demographic_summary()` — categorical columns → donut chart data
- `response_summary()` — Likert/NPS columns → stacked bar data
- `crosstab()` — `pandas.crosstab()` for any two categorical columns
- `segment_comparisons()` — pre-computed `segment_stats` for all metric × segment pairs

**Report sections** (Qwen3-8B, one call per section):
- `Demographic Profile` — describes respondent composition
- `Key Findings` — top 3–5 insights from batch analysis
- `Response Distribution` — Likert and NPS narrative
- `Crosstab Analysis` — `_crosstab_finding_sentence()` generates tie-aware natural language: if best_pct ≈ worst_pct (within 0.1pt), emits neutral phrasing instead of "materially higher"
- `Trend Analysis` — period-over-period changes
- `Recommended Actions` — from `recommend_actions` tool

**Categorical analysis** (`app/packs/survey/categorical.py` — 7 functions):
```
get_demographic_summary()
get_response_distribution()
get_response_by_segment()
get_crosstab()
find_top_segment_for_value()
find_top_segment_for_numeric_threshold()
get_segment_comparison()
```

---

### 6.7 Survey Chat — Conversational Analytics

Same three-node LangGraph as governance chat (intent → tools → synthesize), but with the survey tool registry and survey session state. Conversation history is maintained across turns (`SESSION_MAX_HISTORY=10`).

The live AMD platform strip (`GET /api/platform/info` every 10s) shows GPU utilization and VRAM usage from `rocm-smi` — parsed by `_parse_gpu_summary()` in `app/utils/metrics.py`.

---

## 7. SSE Streaming Architecture

All chat endpoints use **Server-Sent Events** (SSE) for token-by-token streaming.

**Server side** (`stream_governance_chat_turn`, `stream_survey_chat_turn`):

```python
async for chunk in stream_llm_response(messages, model, max_tokens, enable_thinking=True):
    full_content += chunk
    # detect <think>...</think> boundary
    if inside_think:
        yield {"type": "thinking", "data": {"delta": chunk}}
    # after </think>: accumulate content, yield on complete
yield {"type": "complete", "data": {narrative, follow_up_suggestions, evidence, tool_calls}}
```

**vLLM streaming**: the `/v1/chat/completions` endpoint with `stream=True` returns chunks in the OpenAI `text/event-stream` format. `stream_llm_response()` in `llm_client.py` iterates the chunks and yields delta text.

**Client side** (`frontend/aegis.js`, `frontend/survey.js`):

```javascript
const es = new EventSource(`/api/${pack}/chat_stream/${runId}?message=${msg}`);
es.onmessage = (event) => {
    const data = JSON.parse(event.data);
    if (data.type === 'thinking') {
        appendToThinkingBubble(data.data.delta);
    } else if (data.type === 'complete') {
        renderCompleteBubble(data.data);
        renderEvidenceChart(data.data.evidence?.chart_data || []);
    }
};
```

---

## 8. Session & State Management

**File**: `app/session/store.py`, `app/session/run_store.py`

```python
class RunStore:
    _runs: dict[str, InvestigationState]     # run_id → completed state
    _sessions: dict[str, list[str]]          # session_id → [run_id, ...]
    _timestamps: dict[str, float]
    SESSION_MAX_AGE_HOURS = 4
    SESSION_MAX_HISTORY = 10
```

- All state is in-memory (no database dependency)
- Each analysis run gets a UUID `run_id`; results persist for 4 hours
- Chat history (last 10 turns) is maintained per session and injected into every chat prompt for context continuity

---

## 9. Frontend Architecture

Three pages, vanilla JavaScript (no framework, no bundler):

| Page | File | Purpose |
|---|---|---|
| Home | `frontend/index.html` | Pack selector, entry point |
| Governance | `frontend/governance.html` | Log upload + audit results |
| Survey | `frontend/survey.html` | CSV upload + analytics chat |

**Shared scripts**:
- `platform.js` — live AMD GPU strip, polls `/api/platform/info` every 10s
- `charts.js` — `renderBarChart()`, `renderDonutChart()`, `renderStackedBarChart()`, `renderEvidenceChart()` dispatcher

**Chart rendering** (all CSS + Canvas, no chart library):
```javascript
function renderBarChart(title, rows) {
    // rows: [{label, pct, count, colorClass?}]
    // renders .bar-row > .bar-track > .bar-fill divs with inline width style
}

function renderDonutChart(title, distribution) {
    // distribution: [{value, percent}]
    // renders SVG conic-gradient donut
    // delegates to renderBarChart() if distribution.length > 6
}
```

**Print Report**: `window.print()` triggered by a "Print Report" button. `@media print` CSS hides navigation, platform strip, tabs, and chat panel — shows only the report content full-width.

---

## 10. Testing

**File**: `tests/` — 122 tests, all passing

| Test file | Coverage |
|---|---|
| `test_investigation_graph.py` | LangGraph node functions, routing logic |
| `test_governance_pack.py` | Triage flags, each specialist agent, risk scoring |
| `test_governance_routes.py` | FastAPI endpoints, response schemas |
| `test_survey_pack.py` | Triage, insight agent, synthesis, crosstab sentence generation |
| `test_survey_routes.py` | Survey API endpoints |
| `test_platform_routes.py` | `/api/platform/info` shape and mock mode values |
| `test_tools.py` | All 6 MCP tools with real pandas DataFrames |
| `test_csv_loader.py` | CSV parsing, column classification, limits |
| `test_session.py` | RunStore CRUD, TTL, history trimming |
| `test_chat_graph.py` | Intent node routing, tool node execution, synthesis paths |
| `test_metrics.py` | MetricsCollector accumulation and summary |

All tests run in `LLM_MODE=mock` — no GPU or vLLM required. Mock fabricators are **content-aware**: they inspect the actual prompt/state to produce realistic responses, so tests exercise the full data flow even without a live model.

---

## 11. Token Efficiency Analysis

The core efficiency claim is that **triage routing eliminates ~70 % of LLM calls** compared to calling all specialist agents on every entry.

### Naive approach (baseline):

```
500 entries × 4 agents = 2,000 LLM calls
```

### Aegis with triage routing:

```
1 orchestrator call (summarised triage stats, not 500 × 1)
+ N specialist calls where N = len(investigation_plan)
  (only flagged entries × only relevant agents)

Typical for 500-entry batch:
  pii_suspect:           ~43 entries  → 43 PII calls
  injection_suspect:     ~12 entries  → 12 Security calls
  compliance_suspect:    ~8 entries   → 8 Compliance calls
  hallucination_suspect: ~27 entries  (only if has_context=True)

Total: 1 + 43 + 12 + 8 + 27 = 91 calls  vs  2,000 baseline
Reduction: 95.5 %
```

Real reduction depends on data distribution. For survey pack:

```
Batch phase: ~13 tool invocations (most are pandas calls, 2-3 are LLM)
Chat turn:   1 intent call + 0-3 tool calls + 1 synthesis call = 2-5 LLM calls
```

---

## 12. End-to-End Flow Timelines

### Governance — 10-entry demo run (AMD MI300X, vLLM mode)

```
00:00  POST /api/governance/investigate
00:00  triage_node          → 0 LLM calls, keyword scan of 10 entries
00:01  orchestrator_node    → 1 LLM call (Qwen3-8B, <think> streamed)
00:03  specialist_dispatch  → asyncio.gather([pii×2, security×2, compliance×1, hallucination×3])
                              = 8 parallel LLM calls to vLLM
00:28  risk_scoring_node    → 0 LLM calls, deterministic
00:28  dashboard_node       → 0 LLM calls, pandas aggregation
00:29  report_node          → 4 sequential LLM calls (one per section)
00:38  Complete
       Total: ~38s, 13 LLM calls
```

### Survey — full batch + 3 chat turns (AMD MI300X, vLLM mode)

```
00:00  POST /api/survey/analyze
00:00  triage_node          → schema analysis, no LLM
00:01  insight_agent        → 1 LLM call to plan batch tool calls
00:02  tool_exec_node       → 60+ pandas/scipy tool calls (< 1s total)
00:04  synthesis            → 1 LLM call (batch narrative)
00:10  dashboard_node       → 0 LLM calls
00:12  report_node          → 6 sequential LLM calls
00:45  Complete

Chat turn 1 (segment stats):
00:00  intent_node          → 1 LLM call
00:03  tool_node            → segment_stats() in < 10ms
00:06  synthesis + stream   → 1 LLM call, <think> streamed live
00:08  Complete, bar chart rendered in chat bubble

Chat turn 2 (anomaly):  ~6s
Chat turn 3 (recommend): ~8s
```

---

*Generated from source code — `app/`, `frontend/`, `tests/` — TCS & AMD AI Hackathon 2026*
