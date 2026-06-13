"""Pytest configuration and fixtures."""

import pytest


@pytest.fixture(scope="session")
def mock_env_vars(monkeypatch):
    """Set up mock environment variables for testing."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    monkeypatch.setenv("MAX_CSV_ROWS", "1000")
    monkeypatch.setenv("MAX_CSV_MB", "10")
