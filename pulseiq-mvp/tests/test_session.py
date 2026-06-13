"""Tests for session store."""

import time

import pandas as pd
import pytest

from app.session.store import SessionStore


class TestSessionStore:
    """Tests for SessionStore."""
    
    @pytest.fixture
    def store(self):
        """Create a fresh session store."""
        store = SessionStore()
        store._store.clear()
        yield store
        store._store.clear()
    
    @pytest.fixture
    def sample_df(self):
        """Create sample DataFrame."""
        return pd.DataFrame({
            "A": [1, 2, 3],
            "B": ["x", "y", "z"]
        })
    
    @pytest.fixture
    def sample_schema(self):
        """Create sample schema."""
        return {
            "A": {"type": "numeric"},
            "B": {"type": "categorical"}
        }
    
    def test_create_session(self, store, sample_df, sample_schema):
        """Create a new session."""
        sid = store.create(sample_df, sample_schema, "test.csv")
        
        assert sid is not None
        assert len(sid) > 0
        assert store.exists(sid)
    
    def test_get_session(self, store, sample_df, sample_schema):
        """Retrieve a session."""
        sid = store.create(sample_df, sample_schema, "test.csv")
        session = store.get(sid)
        
        assert session is not None
        assert session.session_id == sid
        assert session.filename == "test.csv"
        assert len(session.df) == 3
    
    def test_get_nonexistent_session(self, store):
        """Handle missing sessions."""
        session = store.get("nonexistent-id")
        assert session is None
    
    def test_append_history(self, store, sample_df, sample_schema):
        """Add messages to history."""
        sid = store.create(sample_df, sample_schema, "test.csv")
        
        store.append_history(sid, "user", "Hello")
        store.append_history(sid, "assistant", "Hi there")
        
        history = store.get_history(sid)
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"
    
    def test_history_limit(self, store, sample_df, sample_schema):
        """History should be trimmed after max length."""
        sid = store.create(sample_df, sample_schema, "test.csv")
        
        # Add many messages
        for i in range(30):
            store.append_history(sid, "user", f"Message {i}")
        
        history = store.get_history(sid, max_turns=10)
        # Should be limited to max_turns * 2 messages
        assert len(history) <= 20
    
    def test_delete_session(self, store, sample_df, sample_schema):
        """Delete a session."""
        sid = store.create(sample_df, sample_schema, "test.csv")
        
        assert store.delete(sid) is True
        assert not store.exists(sid)
        assert store.get(sid) is None
    
    def test_delete_nonexistent(self, store):
        """Delete missing session returns False."""
        assert store.delete("nonexistent") is False
    
    def test_evict_old_sessions(self, store, sample_df, sample_schema):
        """Evict expired sessions."""
        sid = store.create(sample_df, sample_schema, "test.csv")
        
        # Manually set old access time
        store._store[sid].last_accessed = time.time() - (5 * 3600)  # 5 hours ago
        
        evicted = store.evict_old(max_age_hours=4)
        
        assert evicted == 1
        assert not store.exists(sid)
    
    def test_list_sessions(self, store, sample_df, sample_schema):
        """List all sessions."""
        sid1 = store.create(sample_df, sample_schema, "test1.csv")
        sid2 = store.create(sample_df, sample_schema, "test2.csv")
        
        sessions = store.list_sessions()
        
        assert len(sessions) == 2
        session_ids = [s["session_id"] for s in sessions]
        assert sid1 in session_ids
        assert sid2 in session_ids
    
    def test_get_stats(self, store, sample_df, sample_schema):
        """Get store statistics."""
        store.create(sample_df, sample_schema, "test.csv")
        
        stats = store.get_stats()
        
        assert stats["total_sessions"] == 1
        assert stats["total_rows"] == 3


class TestSessionStoreThreading:
    """Thread safety tests."""
    
    def test_concurrent_access(self):
        """Handle concurrent access safely."""
        import threading
        
        store = SessionStore()
        store._store.clear()
        
        df = pd.DataFrame({"A": [1, 2, 3]})
        schema = {"A": {"type": "numeric"}}
        
        sids = []
        errors = []
        
        def create_sessions():
            try:
                for i in range(10):
                    sid = store.create(df, schema, f"test{i}.csv")
                    sids.append(sid)
            except Exception as e:
                errors.append(str(e))
        
        # Run concurrently
        threads = [threading.Thread(target=create_sessions) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        assert len(errors) == 0
        assert len(sids) == 50
        
        store._store.clear()
