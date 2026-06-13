"""In-memory session store with TTL eviction."""

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

import pandas as pd

from app.config import SESSION_MAX_AGE_HOURS, SESSION_MAX_HISTORY

logger = logging.getLogger(__name__)


@dataclass
class Session:
    """Represents a user session with uploaded survey data."""
    
    session_id: str
    df: pd.DataFrame
    schema: dict[str, Any]
    filename: str
    uploaded_at: float
    history: list[dict[str, str]] = field(default_factory=list)
    last_accessed: float = field(default_factory=time.time)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert session to dictionary (excluding DataFrame)."""
        return {
            "session_id": self.session_id,
            "filename": self.filename,
            "uploaded_at": self.uploaded_at,
            "last_accessed": self.last_accessed,
            "row_count": len(self.df),
            "column_count": len(self.df.columns),
            "schema": self.schema,
            "history_length": len(self.history),
        }


class SessionStore:
    """
    Thread-safe in-memory session store with TTL eviction.
    
    All sessions are lost on server restart (acceptable for MVP).
    """
    
    _instance: Optional["SessionStore"] = None
    _lock: threading.Lock = threading.Lock()
    
    def __new__(cls) -> "SessionStore":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialize()
        return cls._instance
    
    def _initialize(self) -> None:
        """Initialize the session store."""
        self._store: dict[str, Session] = {}
        self._store_lock = threading.RLock()
        self._eviction_thread: Optional[threading.Thread] = None
        self._shutdown = False
        self._start_eviction_thread()
        logger.info("SessionStore initialized")
    
    def _start_eviction_thread(self) -> None:
        """Start background thread for periodic eviction."""
        def eviction_loop() -> None:
            while not self._shutdown:
                time.sleep(300)  # Run every 5 minutes
                if not self._shutdown:
                    try:
                        self.evict_old()
                    except Exception as e:
                        logger.error(f"Error during session eviction: {e}")
        
        self._eviction_thread = threading.Thread(
            target=eviction_loop,
            daemon=True,
            name="SessionEviction"
        )
        self._eviction_thread.start()
        logger.info("Session eviction thread started")
    
    def shutdown(self) -> None:
        """Shutdown the session store and cleanup."""
        self._shutdown = True
        if self._eviction_thread and self._eviction_thread.is_alive():
            self._eviction_thread.join(timeout=1)
        with self._store_lock:
            self._store.clear()
        logger.info("SessionStore shutdown complete")
    
    def create(
        self,
        df: pd.DataFrame,
        schema: dict[str, Any],
        filename: str
    ) -> str:
        """
        Create a new session with uploaded data.
        
        Args:
            df: The uploaded survey DataFrame
            schema: Detected schema dictionary
            filename: Original filename
            
        Returns:
            New session ID
        """
        sid = str(uuid.uuid4())
        now = time.time()
        
        with self._store_lock:
            self._store[sid] = Session(
                session_id=sid,
                df=df,
                schema=schema,
                filename=filename,
                uploaded_at=now,
                last_accessed=now,
                history=[]
            )
        
        logger.info(f"Created session {sid} for file '{filename}'")
        return sid
    
    def get(self, sid: str) -> Optional[Session]:
        """
        Get a session by ID and update last accessed time.
        
        Args:
            sid: Session ID
            
        Returns:
            Session if found, None otherwise
        """
        with self._store_lock:
            session = self._store.get(sid)
            if session:
                session.last_accessed = time.time()
            return session
    
    def exists(self, sid: str) -> bool:
        """Check if a session exists."""
        with self._store_lock:
            return sid in self._store
    
    def append_history(self, sid: str, role: str, content: str) -> bool:
        """
        Append a message to session history.
        
        Args:
            sid: Session ID
            role: 'user' or 'assistant'
            content: Message content
            
        Returns:
            True if successful, False if session not found
        """
        with self._store_lock:
            session = self._store.get(sid)
            if not session:
                return False
            
            session.history.append({"role": role, "content": content})
            session.last_accessed = time.time()
            
            # Trim history to max length
            if len(session.history) > SESSION_MAX_HISTORY * 2:
                session.history = session.history[-SESSION_MAX_HISTORY * 2:]
            
            return True
    
    def get_history(self, sid: str, max_turns: int = SESSION_MAX_HISTORY) -> list[dict[str, str]]:
        """
        Get recent conversation history.
        
        Args:
            sid: Session ID
            max_turns: Maximum number of turns to return
            
        Returns:
            List of message dicts, empty if session not found
        """
        with self._store_lock:
            session = self._store.get(sid)
            if not session:
                return []
            
            # Return last N turns (each turn = user + assistant)
            max_messages = max_turns * 2
            return session.history[-max_messages:]
    
    def delete(self, sid: str) -> bool:
        """
        Delete a session.
        
        Args:
            sid: Session ID
            
        Returns:
            True if deleted, False if not found
        """
        with self._store_lock:
            if sid in self._store:
                del self._store[sid]
                logger.info(f"Deleted session {sid}")
                return True
            return False
    
    def evict_old(self, max_age_hours: int = SESSION_MAX_AGE_HOURS) -> int:
        """
        Remove sessions older than max_age_hours.
        
        Args:
            max_age_hours: Maximum age in hours
            
        Returns:
            Number of sessions evicted
        """
        cutoff = time.time() - (max_age_hours * 3600)
        
        with self._store_lock:
            old_keys = [
                k for k, v in self._store.items()
                if v.last_accessed < cutoff
            ]
            for k in old_keys:
                del self._store[k]
        
        if old_keys:
            logger.info(f"Evicted {len(old_keys)} old sessions")
        return len(old_keys)
    
    def list_sessions(self) -> list[dict[str, Any]]:
        """List all active sessions (for admin/debugging)."""
        with self._store_lock:
            return [session.to_dict() for session in self._store.values()]
    
    def get_stats(self) -> dict[str, Any]:
        """Get store statistics."""
        with self._store_lock:
            total_sessions = len(self._store)
            total_rows = sum(len(s.df) for s in self._store.values())
            oldest_session = min(
                (s.uploaded_at for s in self._store.values()),
                default=0
            )
            
        return {
            "total_sessions": total_sessions,
            "total_rows": total_rows,
            "oldest_session_age_hours": (
                (time.time() - oldest_session) / 3600 if oldest_session else 0
            ),
        }


def get_session_store() -> SessionStore:
    """Get the singleton SessionStore instance."""
    return SessionStore()
