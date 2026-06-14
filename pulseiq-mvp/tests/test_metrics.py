"""Unit tests for app/utils/metrics.py GPU summary parsing.

rocm-smi --json field names vary across ROCm versions and often report only a
raw PCI device ID (or "N/A") for the card name, so `_parse_gpu_summary` must
resolve those to a friendly name when possible and degrade gracefully to None
otherwise.
"""

from __future__ import annotations

from app.utils.metrics import _parse_gpu_summary


def test_mi300x_device_id_resolves_to_friendly_name():
    raw = {
        "card0": {
            "Card series": "N/A",
            "Card model": "0x74a1",
            "Card vendor": "Advanced Micro Devices, Inc. [AMD/ATI]",
            "GFX Version": "gfx942",
            "GPU use (%)": "37",
            "VRAM Total Memory (B)": str(64 * 1024**3),
            "VRAM Total Used Memory (B)": str(12 * 1024**3),
        }
    }

    summary = _parse_gpu_summary(raw)

    assert summary["gpu_name"] == "AMD Instinct MI300X"
    assert summary["gpu_utilization_pct"] == 37.0
    assert summary["vram_total_gb"] == 64.0
    assert summary["vram_used_gb"] == 12.0


def test_unknown_card_falls_back_gracefully():
    raw = {"card0": {"Card series": "N/A", "Card model": "0xdead", "GFX Version": "gfx1100"}}

    summary = _parse_gpu_summary(raw)

    assert summary["gpu_name"] == "0xdead"
    assert summary["gpu_utilization_pct"] is None
    assert summary["vram_used_gb"] is None
    assert summary["vram_total_gb"] is None


def test_no_card_key_returns_all_none():
    assert _parse_gpu_summary({}) == {
        "gpu_name": None,
        "gpu_utilization_pct": None,
        "vram_used_gb": None,
        "vram_total_gb": None,
    }
