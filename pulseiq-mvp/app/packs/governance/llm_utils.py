"""Small shared helpers for the LLM-backed governance specialists."""

from __future__ import annotations

import json
import re
from typing import Any

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.S)


def parse_json_response(raw: str) -> dict[str, Any]:
    """Best-effort JSON parsing for LLM verdicts.

    Handles markdown code fences and stray reasoning text around the JSON
    object (e.g. from Qwen3 `enable_thinking` traces in vLLM mode).
    """
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = _JSON_OBJECT_RE.search(text)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return {}
