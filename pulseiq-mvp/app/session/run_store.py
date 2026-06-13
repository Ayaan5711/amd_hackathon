"""In-memory store for investigation runs (any AgentPack), keyed by run_id.

Mirrors app/session/store.py's SessionStore singleton pattern, but holds
InvestigationState results plus incremental progress events (for the SSE
stream endpoint) and a per-run chat history (for the "talk to results" graph).
"""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from app.agent.state import InvestigationState

logger = __import__("logging").getLogger(__name__)


@dataclass
class InvestigationRun:
    """Tracks one investigation run from kickoff to completion."""

    run_id: str
    session_id: str
    pack_name: str
    status: str = "running"  # "running" | "complete" | "error"
    progress: list[dict[str, Any]] = field(default_factory=list)
    result: Optional["InvestigationState"] = None
    chat_history: list[dict[str, str]] = field(default_factory=list)
    error: str | None = None
    created_at: float = field(default_factory=time.time)


class RunStore:
    """Thread-safe in-memory store for investigation runs. Lost on restart."""

    _instance: Optional["RunStore"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "RunStore":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        self._runs: dict[str, InvestigationRun] = {}
        self._store_lock = threading.RLock()
        logger.info("RunStore initialized")

    def create(self, session_id: str, pack_name: str) -> str:
        run_id = str(uuid.uuid4())
        with self._store_lock:
            self._runs[run_id] = InvestigationRun(run_id=run_id, session_id=session_id, pack_name=pack_name)
        logger.info(f"Created investigation run {run_id} for session {session_id} (pack={pack_name})")
        return run_id

    def get(self, run_id: str) -> InvestigationRun | None:
        with self._store_lock:
            return self._runs.get(run_id)

    def append_progress(self, run_id: str, event: dict[str, Any]) -> None:
        with self._store_lock:
            run = self._runs.get(run_id)
            if run:
                run.progress.append(event)

    def set_result(self, run_id: str, result: "InvestigationState") -> None:
        with self._store_lock:
            run = self._runs.get(run_id)
            if run:
                run.result = result
                run.status = "complete"

    def set_error(self, run_id: str, error: str) -> None:
        with self._store_lock:
            run = self._runs.get(run_id)
            if run:
                run.error = error
                run.status = "error"

    def append_chat_history(self, run_id: str, role: str, content: str) -> None:
        with self._store_lock:
            run = self._runs.get(run_id)
            if run:
                run.chat_history.append({"role": role, "content": content})


def get_run_store() -> RunStore:
    """Get the singleton RunStore instance."""
    return RunStore()
