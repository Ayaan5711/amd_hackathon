"""Platform info endpoint - surfaces the AMD ROCm/vLLM stack and live GPU stats
that power this app, for the frontend's persistent platform strip."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from app.config import (
    LLM_MODE,
    VLLM_BASE_URL,
    VLLM_MODEL_INTENT,
    VLLM_MODEL_ORCHESTRATOR,
    VLLM_MODEL_REPORT,
    VLLM_MODEL_SPECIALIST,
    VLLM_MODEL_SYNTHESIS,
    VLLM_MODEL_THEMES,
)
from app.utils.metrics import gpu_stats

router = APIRouter(prefix="/platform", tags=["platform"])


@router.get("/info")
async def platform_info() -> dict[str, Any]:
    """LLM mode/models + live GPU stats, polled by frontend/platform.js."""
    models = sorted(
        {
            VLLM_MODEL_INTENT,
            VLLM_MODEL_SYNTHESIS,
            VLLM_MODEL_ORCHESTRATOR,
            VLLM_MODEL_SPECIALIST,
            VLLM_MODEL_REPORT,
            VLLM_MODEL_THEMES,
        }
    )
    return {
        "llm_mode": LLM_MODE,
        "vllm_base_url": VLLM_BASE_URL if LLM_MODE == "vllm" else None,
        "models": models,
        "gpu": gpu_stats(),
    }
