"""Dual-mode LLM client.

LLM_MODE=mock  -> no network calls; returns schema-aware canned responses so the
                  full pipeline can run end-to-end on a GPU-less Windows dev box.
LLM_MODE=vllm  -> OpenAI-compatible calls against an AMD vLLM server (Qwen3, etc.),
                  with optional Qwen3 `enable_thinking` support.
"""

import json
import logging
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from openai import AsyncOpenAI, OpenAI

from app.config import (
    LLM_MODE,
    LLM_REQUEST_TIMEOUT,
    LLM_TEMPERATURE,
    VLLM_API_KEY,
    VLLM_BASE_URL,
    VLLM_MODEL_SYNTHESIS,
)

if TYPE_CHECKING:
    from app.utils.metrics import MetricsCollector

logger = logging.getLogger(__name__)


class LLMClient:
    """Singleton client wrapping vLLM's OpenAI-compatible endpoint."""

    _instance: "LLMClient | None" = None

    def __new__(cls) -> "LLMClient":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        self._client: OpenAI | None = None
        self._async_client: AsyncOpenAI | None = None
        if LLM_MODE == "vllm":
            self._client = OpenAI(
                base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY, timeout=LLM_REQUEST_TIMEOUT
            )
            self._async_client = AsyncOpenAI(
                base_url=VLLM_BASE_URL, api_key=VLLM_API_KEY, timeout=LLM_REQUEST_TIMEOUT
            )
            logger.info(f"LLM client initialized in vLLM mode -> {VLLM_BASE_URL}")
        else:
            logger.info("LLM client initialized in MOCK mode (LLM_MODE=mock, no network calls)")

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            raise RuntimeError("Sync LLM client unavailable in LLM_MODE=mock")
        return self._client

    @property
    def async_client(self) -> AsyncOpenAI:
        if self._async_client is None:
            raise RuntimeError("Async LLM client unavailable in LLM_MODE=mock")
        return self._async_client


def get_llm_client() -> LLMClient:
    """Get the singleton LLM client instance."""
    return LLMClient()


def _estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars/token) used in mock mode and as a usage fallback."""
    return max(1, len(text) // 4)


def _messages_text(messages: list[dict[str, str]]) -> str:
    return "\n".join(m.get("content", "") for m in messages)


def _build_kwargs(
    messages: list[dict[str, str]],
    model: str,
    max_tokens: int,
    temperature: float,
    json_mode: bool,
    enable_thinking: bool,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    if enable_thinking:
        kwargs["extra_body"] = {"chat_template_kwargs": {"enable_thinking": True}}
    return kwargs


def _record(
    metrics: "MetricsCollector | None",
    agent: str,
    model: str,
    tokens_in: int,
    tokens_out: int,
    latency_ms: float,
    enable_thinking: bool,
    response_schema: str | None,
) -> None:
    if metrics is not None:
        metrics.record_call(
            agent=agent,
            model=model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            enable_thinking=enable_thinking,
            response_schema=response_schema,
        )


# =============================================================================
# Mock responses — schema-aware so JSON-parsing callers work without a real LLM
# =============================================================================

_MOCK_RESPONSES: dict[str, dict[str, Any]] = {
    "intent": {
        "intent": "general",
        "reasoning": "Mock mode (LLM_MODE=mock): no model invoked.",
        "tool_calls": [],
        "clarification_needed": False,
        "clarification_question": None,
        "clarification_options": [],
    },
    "synthesis": {
        "narrative": "Mock mode (LLM_MODE=mock): placeholder synthesis response.",
        "key_insights": [],
        "follow_up_suggestions": [],
        "evidence": {},
    },
    "themes": {
        "themes": [],
        "overall_sentiment": {"positive_percent": 0, "neutral_percent": 100, "negative_percent": 0},
        "summary": "Mock mode (LLM_MODE=mock): theme extraction skipped.",
    },
    "recommend": {
        "recommendations": [],
        "summary": "Mock mode (LLM_MODE=mock): recommendation generation skipped.",
    },
}


# A pack-specific callable that inspects the outgoing prompt messages and fabricates
# a structurally-valid, content-aware verdict — lets mock mode produce realistic
# flagged/clean splits (and therefore meaningful precision/recall) without a GPU.
MockFabricator = Callable[[list[dict[str, str]]], dict[str, Any]]


def _mock_response(
    response_schema: str | None,
    messages: list[dict[str, str]],
    json_mode: bool,
    mock_fabricator: MockFabricator | None = None,
) -> str:
    """Fabricate a structurally-valid response for the given schema tag."""
    if mock_fabricator is not None:
        return json.dumps(mock_fabricator(messages))
    if response_schema and response_schema in _MOCK_RESPONSES:
        return json.dumps(_MOCK_RESPONSES[response_schema])
    if json_mode:
        return "{}"
    return "Mock mode (LLM_MODE=mock): no model invoked."


# =============================================================================
# Public API
# =============================================================================

def call_llm(
    messages: list[dict[str, str]],
    model: str = VLLM_MODEL_SYNTHESIS,
    max_tokens: int = 800,
    temperature: float = LLM_TEMPERATURE,
    json_mode: bool = False,
    enable_thinking: bool = False,
    response_schema: str | None = None,
    agent: str = "unknown",
    metrics: "MetricsCollector | None" = None,
    mock_fabricator: MockFabricator | None = None,
) -> str:
    """Synchronous LLM call with dual-mode (mock/vLLM) support and metrics recording."""
    start = time.perf_counter()

    if LLM_MODE == "mock":
        content = _mock_response(response_schema, messages, json_mode, mock_fabricator)
        tokens_in = _estimate_tokens(_messages_text(messages))
        tokens_out = _estimate_tokens(content)
    else:
        client = get_llm_client().client
        kwargs = _build_kwargs(messages, model, max_tokens, temperature, json_mode, enable_thinking)
        try:
            response = client.chat.completions.create(**kwargs)
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise RuntimeError(f"LLM call failed: {e}")

        if not response.choices:
            raise RuntimeError("No response choices from LLM")
        content = response.choices[0].message.content
        if content is None:
            raise RuntimeError("Empty response from LLM")
        content = content.strip()

        usage = response.usage
        tokens_in = usage.prompt_tokens if usage else _estimate_tokens(_messages_text(messages))
        tokens_out = usage.completion_tokens if usage else _estimate_tokens(content)

    latency_ms = (time.perf_counter() - start) * 1000
    _record(metrics, agent, model, tokens_in, tokens_out, latency_ms, enable_thinking, response_schema)
    return content


async def call_llm_async(
    messages: list[dict[str, str]],
    model: str = VLLM_MODEL_SYNTHESIS,
    max_tokens: int = 800,
    temperature: float = LLM_TEMPERATURE,
    json_mode: bool = False,
    enable_thinking: bool = False,
    response_schema: str | None = None,
    agent: str = "unknown",
    metrics: "MetricsCollector | None" = None,
    mock_fabricator: MockFabricator | None = None,
) -> str:
    """Asynchronous LLM call with dual-mode (mock/vLLM) support and metrics recording."""
    start = time.perf_counter()

    if LLM_MODE == "mock":
        content = _mock_response(response_schema, messages, json_mode, mock_fabricator)
        tokens_in = _estimate_tokens(_messages_text(messages))
        tokens_out = _estimate_tokens(content)
    else:
        client = get_llm_client().async_client
        kwargs = _build_kwargs(messages, model, max_tokens, temperature, json_mode, enable_thinking)
        try:
            response = await client.chat.completions.create(**kwargs)
        except Exception as e:
            logger.error(f"Async LLM call failed: {e}")
            raise RuntimeError(f"LLM call failed: {e}")

        if not response.choices:
            raise RuntimeError("No response choices from LLM")
        content = response.choices[0].message.content
        if content is None:
            raise RuntimeError("Empty response from LLM")
        content = content.strip()

        usage = response.usage
        tokens_in = usage.prompt_tokens if usage else _estimate_tokens(_messages_text(messages))
        tokens_out = usage.completion_tokens if usage else _estimate_tokens(content)

    latency_ms = (time.perf_counter() - start) * 1000
    _record(metrics, agent, model, tokens_in, tokens_out, latency_ms, enable_thinking, response_schema)
    return content


async def stream_llm_response(
    messages: list[dict[str, str]],
    model: str = VLLM_MODEL_SYNTHESIS,
    max_tokens: int = 800,
    temperature: float = LLM_TEMPERATURE,
    enable_thinking: bool = False,
):
    """Stream LLM response chunks for SSE. In mock mode, streams the canned response word-by-word."""
    if LLM_MODE == "mock":
        content = _mock_response(None, messages, json_mode=False)
        for word in content.split(" "):
            yield word + " "
        return

    client = get_llm_client().async_client
    kwargs = _build_kwargs(messages, model, max_tokens, temperature, json_mode=False, enable_thinking=enable_thinking)
    kwargs["stream"] = True

    try:
        stream = await client.chat.completions.create(**kwargs)
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        logger.error(f"LLM streaming failed: {e}")
        raise RuntimeError(f"LLM streaming failed: {e}")
