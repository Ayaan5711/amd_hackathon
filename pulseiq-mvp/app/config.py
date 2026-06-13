"""Application configuration - all constants in one place."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

BASE_DIR: Path = Path(__file__).parent.parent


# =============================================================================
# LLM Configuration (AMD vLLM, dual-mode)
# =============================================================================
# "mock"  -> no network calls, returns schema-aware canned responses (local Windows dev)
# "vllm"  -> OpenAI-compatible calls to a vLLM server (AMD Developer Cloud)
LLM_MODE: str = os.getenv("LLM_MODE", "mock")

VLLM_BASE_URL: str = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_API_KEY: str = os.getenv("VLLM_API_KEY", "not-needed")

VLLM_MODEL_INTENT: str = os.getenv("VLLM_MODEL_INTENT", "Qwen/Qwen3-8B")
VLLM_MODEL_SYNTHESIS: str = os.getenv("VLLM_MODEL_SYNTHESIS", "Qwen/Qwen3-8B")
VLLM_MODEL_THEMES: str = os.getenv("VLLM_MODEL_THEMES", "Qwen/Qwen3-8B")
VLLM_MODEL_ORCHESTRATOR: str = os.getenv("VLLM_MODEL_ORCHESTRATOR", "Qwen/Qwen3-8B")
VLLM_MODEL_SPECIALIST: str = os.getenv("VLLM_MODEL_SPECIALIST", "Qwen/Qwen3-8B")
VLLM_MODEL_REPORT: str = os.getenv("VLLM_MODEL_REPORT", "Qwen/Qwen3-8B")

MAX_TOKENS_INTENT: int = int(os.getenv("MAX_TOKENS_INTENT", "400"))
MAX_TOKENS_SYNTHESIS: int = int(os.getenv("MAX_TOKENS_SYNTHESIS", "800"))
MAX_TOKENS_THEMES: int = int(os.getenv("MAX_TOKENS_THEMES", "600"))
MAX_TOKENS_ORCHESTRATOR: int = int(os.getenv("MAX_TOKENS_ORCHESTRATOR", "1200"))
MAX_TOKENS_SPECIALIST: int = int(os.getenv("MAX_TOKENS_SPECIALIST", "500"))
MAX_TOKENS_REPORT: int = int(os.getenv("MAX_TOKENS_REPORT", "1000"))

LLM_TEMPERATURE: float = 0.1  # Low temperature for consistent classification/extraction
LLM_REQUEST_TIMEOUT: float = float(os.getenv("LLM_REQUEST_TIMEOUT", "60"))


# =============================================================================
# CSV / Dataset Processing Limits
# =============================================================================
MAX_CSV_ROWS: int = int(os.getenv("MAX_CSV_ROWS", "50000"))
MAX_CSV_MB: int = int(os.getenv("MAX_CSV_MB", "50"))


# =============================================================================
# Session Configuration
# =============================================================================
SESSION_MAX_AGE_HOURS: int = int(os.getenv("SESSION_MAX_AGE_HOURS", "4"))
SESSION_MAX_HISTORY: int = int(os.getenv("SESSION_MAX_HISTORY", "10"))


# =============================================================================
# Survey Pack Tool Configuration
# =============================================================================
MAX_OPEN_TEXT_SAMPLE: int = int(os.getenv("MAX_OPEN_TEXT_SAMPLE", "150"))
MAX_THEME_COUNT: int = int(os.getenv("MAX_THEME_COUNT", "8"))
ANOMALY_Z_THRESHOLD: float = float(os.getenv("ANOMALY_Z_THRESHOLD", "2.0"))
MIN_SEGMENT_SIZE: int = int(os.getenv("MIN_SEGMENT_SIZE", "10"))


# =============================================================================
# Server Configuration
# =============================================================================
HOST: str = os.getenv("HOST", "0.0.0.0")
PORT: int = int(os.getenv("PORT", "8000"))
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "info")


# =============================================================================
# Column Type Definitions (Survey pack schema inference)
# =============================================================================
INFERRED_TYPES: dict[str, str] = {
    "numeric_scale": "Likert/rating scale (1-5 or 1-10)",
    "numeric_score": "Continuous score",
    "categorical": "Category/group column",
    "open_text": "Free-text response",
    "boolean": "Yes/No or True/False",
    "datetime": "Date or timestamp",
}


# =============================================================================
# Intent Types (Survey pack chat graph)
# =============================================================================
VALID_INTENTS: list[str] = [
    "segment_stats",
    "trend_compare",
    "open_text",
    "anomaly",
    "recommend",
    "clarify",
    "general",
]


# =============================================================================
# Governance Pack - AI Interaction Log Schema
# =============================================================================
LOG_REQUIRED_COLUMNS: list[str] = [
    "log_id",
    "timestamp",
    "user_prompt",
    "ai_response",
]
LOG_OPTIONAL_COLUMNS: list[str] = [
    "retrieved_context",
    "model_name",
]


# =============================================================================
# Governance Pack - Presidio / PII Configuration
# =============================================================================
PII_ENTITIES: list[str] = [
    "PERSON",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "US_SSN",
    "US_BANK_NUMBER",
    "IBAN_CODE",
    "LOCATION",
]
PII_SCORE_THRESHOLD: float = float(os.getenv("PII_SCORE_THRESHOLD", "0.4"))
PRESIDIO_SPACY_MODEL: str = os.getenv("PRESIDIO_SPACY_MODEL", "en_core_web_sm")

# Entity types that immediately push a finding to "critical" severity
PII_CRITICAL_ENTITIES: set[str] = {"US_SSN", "CREDIT_CARD", "US_BANK_NUMBER", "IBAN_CODE"}


# =============================================================================
# Governance Pack - Risk Scoring Weights & Severity Thresholds
# =============================================================================
RISK_WEIGHTS: dict[str, int] = {
    "pii_critical": 40,
    "pii_other": 20,
    "injection": 35,
    "compliance": 30,
    "hallucination": 20,
}

# A risk_score >= threshold maps to the given severity bucket (checked highest-first)
RISK_SEVERITY_THRESHOLDS: list[tuple[str, int]] = [
    ("critical", 70),
    ("high", 45),
    ("medium", 20),
]
RISK_SEVERITY_DEFAULT: str = "low"


# =============================================================================
# Paths
# =============================================================================
DATA_DIR: Path = BASE_DIR / "app" / "data"
SYNTHETIC_LOGS_DIR: Path = DATA_DIR / "synthetic_logs"
POLICIES_DIR: Path = DATA_DIR / "policies"
CHROMA_PERSIST_DIR: Path = DATA_DIR / "chroma"
