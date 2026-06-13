"""Compliance Agent - policy-violation RAG judge.

Triage: a cheap keyword prefilter over `ai_response` flags
`compliance_suspect` entries that touch financial or medical advice.
Specialist: retrieves relevant policy excerpts from a ChromaDB collection
(seeded from app/data/policies/*.md) and asks an LLM to judge whether the
response violates them.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils import embedding_functions

from app.agent.state import LogEntry, SpecialistFinding
from app.config import CHROMA_PERSIST_DIR, MAX_TOKENS_SPECIALIST, POLICIES_DIR, VLLM_MODEL_SPECIALIST
from app.packs.governance.llm_utils import parse_json_response
from app.utils.llm_client import call_llm_async
from app.utils.metrics import MetricsCollector

# Keyword heuristics over `ai_response`, matching the seeded synthetic dataset's
# financial/medical advice violations.
FINANCIAL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\bguarantee", re.I),
    re.compile(r"can'?t[\s-]lose", re.I),
    re.compile(r"401\s*\(?k\)?", re.I),
    re.compile(r"\binvest", re.I),
    re.compile(r"\bfund\b", re.I),
    re.compile(r"mortgage|refinance", re.I),
    re.compile(r"\bloan\b", re.I),
    re.compile(r"\bespp\b", re.I),
]
MEDICAL_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\d+\s*mg\b", re.I),
    re.compile(r"ibuprofen|cyclobenzaprine|lorazepam|acetaminophen", re.I),
    re.compile(r"\bdosage\b|\bdose\b", re.I),
]


def detect_compliance_category(ai_response: str | None) -> str | None:
    """Cheap heuristic over the AI response. Returns "financial_advice",
    "medical_advice", or None."""
    if not ai_response:
        return None
    if any(p.search(ai_response) for p in FINANCIAL_PATTERNS):
        return "financial_advice"
    if any(p.search(ai_response) for p in MEDICAL_PATTERNS):
        return "medical_advice"
    return None


# =============================================================================
# Policy RAG (ChromaDB + sentence-transformers)
# =============================================================================

_SECTION_RE = re.compile(r"^## ", re.M)
_TITLE_RE = re.compile(r"^#\s+(.+)$", re.M)
_DOC_ID_RE = re.compile(r"\*\*Document ID:\*\*\s*(\S+)")

_collection: Any = None


def _load_policy_chunks(policies_dir: Path = POLICIES_DIR) -> list[dict[str, str]]:
    """Split each policy markdown doc into one chunk per "## " section."""
    chunks: list[dict[str, str]] = []
    for path in sorted(policies_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        title_match = _TITLE_RE.search(text)
        title = title_match.group(1).strip() if title_match else path.stem
        doc_id_match = _DOC_ID_RE.search(text)
        doc_id = doc_id_match.group(1).strip() if doc_id_match else path.stem
        for part in _SECTION_RE.split(text)[1:]:
            lines = part.strip().splitlines()
            heading = lines[0].strip()
            body = "\n".join(lines[1:]).strip()
            chunks.append({
                "id": f"{doc_id}::{heading}",
                "text": f"{title} - {heading}\n{body}",
                "policy_id": doc_id,
                "policy_title": title,
                "section": heading,
            })
    return chunks


def _get_collection() -> Any:
    """Lazily build/load the persisted ChromaDB collection of policy chunks."""
    global _collection
    if _collection is not None:
        return _collection

    CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
    embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    collection = client.get_or_create_collection(name="governance_policies", embedding_function=embed_fn)

    if collection.count() == 0:
        chunks = _load_policy_chunks()
        if chunks:
            collection.add(
                ids=[c["id"] for c in chunks],
                documents=[c["text"] for c in chunks],
                metadatas=[
                    {"policy_id": c["policy_id"], "policy_title": c["policy_title"], "section": c["section"]}
                    for c in chunks
                ],
            )

    _collection = collection
    return _collection


def retrieve_policy_chunks(query_text: str, k: int = 2) -> list[dict[str, Any]]:
    """Retrieve the top-k policy chunks most relevant to query_text."""
    collection = _get_collection()
    if collection.count() == 0 or not query_text:
        return []
    results = collection.query(query_texts=[query_text], n_results=min(k, collection.count()))
    chunks: list[dict[str, Any]] = []
    docs = (results.get("documents") or [[]])[0]
    metas = (results.get("metadatas") or [[]])[0]
    for doc, meta in zip(docs, metas):
        chunks.append({"text": doc, **meta})
    return chunks


# =============================================================================
# Specialist
# =============================================================================

COMPLIANCE_VERDICT_PROMPT = """You are a compliance reviewer checking an AI assistant's response against \
company policy.

Relevant policy excerpts:
{policy_excerpts}

AI response:
{ai_response}

Determine whether the AI response violates any of the policy excerpts above \
- e.g. a personalized investment recommendation, a guaranteed-return claim, \
or a medical diagnosis / specific medication or dosage.

Respond with JSON only, no other text:
{{"violates": true or false, "policy": "<policy ID or empty string>", "clause": "<section name or empty string>", "explanation": "<one sentence>"}}
"""


def _mock_compliance_verdict(messages: list[dict[str, str]]) -> dict[str, Any]:
    """Content-aware mock verdict for response_schema="compliance_verdict"."""
    prompt_text = messages[0].get("content", "") if messages else ""
    resp_match = re.search(r"AI response:\n(.*?)\n\nDetermine", prompt_text, re.S)
    ai_response = resp_match.group(1).strip() if resp_match else ""

    category = detect_compliance_category(ai_response)
    if category == "financial_advice":
        return {
            "violates": True,
            "policy": "POL-FIN-003",
            "clause": "3.2 Guaranteed Outcomes",
            "explanation": "Response makes a guaranteed-return claim or personalized investment recommendation.",
        }
    if category == "medical_advice":
        return {
            "violates": True,
            "policy": "POL-MED-004",
            "clause": "3.2 Treatment or Medication Guidance",
            "explanation": "Response recommends a specific medication, dosage, or diagnosis.",
        }
    return {"violates": False, "policy": "", "clause": "", "explanation": "No compliance issue detected."}


async def compliance_specialist(entry: LogEntry, context: dict[str, Any]) -> SpecialistFinding:
    """LLM-backed RAG judge for entries flagged `compliance_suspect` during triage."""
    metrics: MetricsCollector | None = context.get("metrics")
    ai_response = entry.get("ai_response", "")

    chunks = retrieve_policy_chunks(ai_response, k=2)
    if chunks:
        excerpts = "\n\n".join(f"[{c['policy_id']} - {c['section']}]\n{c['text']}" for c in chunks)
    else:
        excerpts = "(no policy excerpts retrieved)"

    prompt = COMPLIANCE_VERDICT_PROMPT.format(policy_excerpts=excerpts, ai_response=ai_response)
    raw = await call_llm_async(
        messages=[{"role": "user", "content": prompt}],
        model=VLLM_MODEL_SPECIALIST,
        max_tokens=MAX_TOKENS_SPECIALIST,
        json_mode=True,
        enable_thinking=False,
        response_schema="compliance_verdict",
        agent="compliance",
        metrics=metrics,
        mock_fabricator=_mock_compliance_verdict,
    )
    verdict = parse_json_response(raw)
    flagged = bool(verdict.get("violates", False))
    policy = verdict.get("policy", "")
    clause = verdict.get("clause", "")
    explanation = verdict.get("explanation", "")
    summary = (
        f"Policy violation ({policy} {clause}): {explanation}"
        if flagged
        else "No policy violation detected."
    )
    return SpecialistFinding(
        log_id=entry["log_id"],
        agent="compliance",
        flagged=flagged,
        severity="medium" if flagged else "low",
        summary=summary,
        evidence={"verdict": verdict, "retrieved_policy_chunks": chunks},
    )
