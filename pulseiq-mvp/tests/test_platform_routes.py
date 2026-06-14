"""Platform info route tests (app/api/platform_routes.py).

Runs in LLM_MODE=mock (the default, no rocm-smi on this machine), so the
endpoint should report mock mode with no live GPU stats.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


def test_platform_info_shape():
    with TestClient(app) as client:
        resp = client.get("/api/platform/info")
        assert resp.status_code == 200

        body = resp.json()
        assert set(body.keys()) == {"llm_mode", "vllm_base_url", "models", "gpu"}
        assert body["llm_mode"] == "mock"
        assert body["vllm_base_url"] is None
        assert body["models"]
        assert body["gpu"] == {"gpu_available": False}
