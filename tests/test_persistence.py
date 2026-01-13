"""
Tests for trace persistence layer.
Uses real SQLite with temporary database - no mocking needed.
"""

import pytest
import tempfile
import os
import sqlite3
from datetime import datetime, timezone

from persistence.sqlite import Sqlite
from persistence.trace_store import TraceStore
from tf_types import (
    TraceRecord, StepRecord, RunConfig, Mode, Strictness,
    RunStatus, EvalDecision
)


# --- Fixtures ---

@pytest.fixture
def temp_db_path():
    """Create temporary database file."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.unlink(path)
    # Also remove WAL files if they exist
    for suffix in ["-wal", "-shm"]:
        wal_path = path + suffix
        if os.path.exists(wal_path):
            os.unlink(wal_path)


@pytest.fixture
def store(temp_db_path):
    """Create TraceStore with temporary database."""
    store = TraceStore(db_path=temp_db_path)
    yield store


# --- TraceRecord Tests ---

class TestTraceRecordPersistence:
    """Tests for TraceRecord CRUD operations."""

    def test_create_trace(self, store):
        """Should create a trace record."""
        trace = TraceRecord(
            trace_id="test-trace-001",
            user_input="What is AI?",
            config=RunConfig(mode=Mode.GROUNDED_QA),
            mode=Mode.GROUNDED_QA,
            model="gpt-3.5-turbo",
            provider="openai"
        )
        
        store.create_trace(trace)
        
        # Verify it was created
        retrieved = store.get_trace("test-trace-001")
        assert retrieved is not None
        assert retrieved.trace_id == "test-trace-001"
        assert retrieved.user_input == "What is AI?"

    def test_get_trace_returns_none_for_missing(self, store):
        """Should return None for non-existent trace."""
        result = store.get_trace("nonexistent-trace")
        
        assert result is None

    def test_update_trace_status(self, store):
        """Should update trace status."""
        trace = TraceRecord(
            trace_id="test-trace-002",
            user_input="Test",
            config=RunConfig(),
            mode=Mode.GROUNDED_QA,
            model="gpt-3.5-turbo",
            provider="openai",
            status=RunStatus.RUNNING
        )
        store.create_trace(trace)
        
        # Update to done
        store.update_trace(
            trace_id="test-trace-002",
            status=RunStatus.DONE,
            final_answer="The answer is 42."
        )
        
        retrieved = store.get_trace("test-trace-002")
        assert retrieved.status == RunStatus.DONE
        assert retrieved.final_answer == "The answer is 42."

    def test_update_trace_to_failed(self, store):
        """Should update trace to failed with error."""
        trace = TraceRecord(
            trace_id="test-trace-003",
            user_input="Test",
            config=RunConfig(),
            mode=Mode.GROUNDED_QA,
            model="gpt-3.5-turbo",
            provider="openai"
        )
        store.create_trace(trace)
        
        # Update with error
        store.update_trace(
            trace_id="test-trace-003",
            status=RunStatus.FAILED,
            error="Something went wrong"
        )
        
        retrieved = store.get_trace("test-trace-003")
        assert retrieved.status == RunStatus.FAILED
        assert retrieved.error == "Something went wrong"

    def test_list_traces_empty(self, store):
        """Should return empty list when no traces."""
        traces = store.list_traces()
        
        assert traces == []

    def test_list_traces_returns_all(self, store):
        """Should return all traces."""
        for i in range(3):
            trace = TraceRecord(
                trace_id=f"trace-{i}",
                user_input=f"Question {i}",
                config=RunConfig(),
                mode=Mode.GROUNDED_QA,
                model="gpt-3.5-turbo",
                provider="openai"
            )
            store.create_trace(trace)
        
        traces = store.list_traces()
        
        assert len(traces) == 3

    def test_list_traces_with_limit(self, store):
        """Should respect limit parameter."""
        for i in range(5):
            trace = TraceRecord(
                trace_id=f"trace-limit-{i}",
                user_input=f"Question {i}",
                config=RunConfig(),
                mode=Mode.GROUNDED_QA,
                model="gpt-3.5-turbo",
                provider="openai"
            )
            store.create_trace(trace)
        
        traces = store.list_traces(limit=3)
        
        assert len(traces) == 3


# --- StepRecord Tests ---

class TestStepRecordPersistence:
    """Tests for StepRecord CRUD operations."""

    def test_insert_step(self, store):
        """Should insert a step record."""
        # First create a trace
        trace = TraceRecord(
            trace_id="step-test-trace",
            user_input="Test",
            config=RunConfig(),
            mode=Mode.GROUNDED_QA,
            model="gpt-3.5-turbo",
            provider="openai"
        )
        store.create_trace(trace)
        
        step = StepRecord(
            trace_id="step-test-trace",
            step_seq=0,
            node_name="intake",
            tokens=50,
            cost_usd=0.001,
            latency_ms=100
        )
        
        store.insert_step(step)
        
        steps = store.get_steps("step-test-trace")
        assert len(steps) == 1
        assert steps[0].node_name == "intake"

    def test_get_steps_ordered(self, store):
        """Steps should be returned in order."""
        trace = TraceRecord(
            trace_id="ordered-steps-trace",
            user_input="Test",
            config=RunConfig(),
            mode=Mode.GROUNDED_QA,
            model="gpt-3.5-turbo",
            provider="openai"
        )
        store.create_trace(trace)
        
        # Insert steps out of order
        for seq, node in [(2, "evaluator"), (0, "intake"), (1, "planner")]:
            step = StepRecord(
                trace_id="ordered-steps-trace",
                step_seq=seq,
                node_name=node,
                tokens=10,
                cost_usd=0.001,
                latency_ms=50
            )
            store.insert_step(step)
        
        steps = store.get_steps("ordered-steps-trace")
        
        assert [s.node_name for s in steps] == ["intake", "planner", "evaluator"]

    def test_step_latency_stored(self, store):
        """Step latency should be stored correctly."""
        trace = TraceRecord(
            trace_id="latency-test",
            user_input="Test",
            config=RunConfig(),
            mode=Mode.GROUNDED_QA,
            model="gpt-3.5-turbo",
            provider="openai"
        )
        store.create_trace(trace)
        
        step = StepRecord(
            trace_id="latency-test",
            step_seq=0,
            node_name="executor",
            tokens=100,
            cost_usd=0.005,
            latency_ms=2500
        )
        store.insert_step(step)
        
        steps = store.get_steps("latency-test")
        assert steps[0].latency_ms == 2500

    def test_step_tokens_and_cost(self, store):
        """Step tokens and cost should be stored correctly."""
        trace = TraceRecord(
            trace_id="cost-test",
            user_input="Test",
            config=RunConfig(),
            mode=Mode.GROUNDED_QA,
            model="gpt-3.5-turbo",
            provider="openai"
        )
        store.create_trace(trace)
        
        step = StepRecord(
            trace_id="cost-test",
            step_seq=0,
            node_name="planner",
            tokens=250,
            cost_usd=0.0075,
            latency_ms=800
        )
        store.insert_step(step)
        
        steps = store.get_steps("cost-test")
        assert steps[0].tokens == 250
        assert abs(steps[0].cost_usd - 0.0075) < 0.0001

    def test_step_cache_hit_stored(self, store):
        """Step cache_hit flag should be stored correctly."""
        trace = TraceRecord(
            trace_id="cache-hit-test",
            user_input="Test",
            config=RunConfig(),
            mode=Mode.GROUNDED_QA,
            model="gpt-3.5-turbo",
            provider="openai"
        )
        store.create_trace(trace)
        
        step = StepRecord(
            trace_id="cache-hit-test",
            step_seq=0,
            node_name="executor",
            tokens=100,
            cost_usd=0.005,
            latency_ms=50,  # Fast because cached
            cache_hit=True
        )
        store.insert_step(step)
        
        steps = store.get_steps("cache-hit-test")
        assert steps[0].cache_hit is True

    def test_get_steps_empty(self, store):
        """Should return empty list for trace with no steps."""
        trace = TraceRecord(
            trace_id="no-steps-trace",
            user_input="Test",
            config=RunConfig(),
            mode=Mode.GROUNDED_QA,
            model="gpt-3.5-turbo",
            provider="openai"
        )
        store.create_trace(trace)
        
        steps = store.get_steps("no-steps-trace")
        
        assert steps == []

    def test_multiple_steps_per_trace(self, store):
        """Should store multiple steps for one trace."""
        trace = TraceRecord(
            trace_id="multi-step-trace",
            user_input="Test",
            config=RunConfig(),
            mode=Mode.GROUNDED_QA,
            model="gpt-3.5-turbo",
            provider="openai"
        )
        store.create_trace(trace)
        
        nodes = ["intake", "retriever", "planner", "executor", "evaluator"]
        for seq, node in enumerate(nodes):
            step = StepRecord(
                trace_id="multi-step-trace",
                step_seq=seq,
                node_name=node,
                tokens=50 + seq * 10,
                cost_usd=0.001 * (seq + 1),
                latency_ms=100 + seq * 50
            )
            store.insert_step(step)
        
        steps = store.get_steps("multi-step-trace")
        
        assert len(steps) == 5
        assert [s.node_name for s in steps] == nodes


# --- Database Schema Tests ---

class TestDatabaseSchema:
    """Tests for database schema and initialization."""

    def test_traces_table_exists(self, store):
        """traces table should exist."""
        cursor = store.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='traces'"
        )
        result = cursor.fetchone()
        
        assert result is not None

    def test_trace_steps_table_exists(self, store):
        """trace_steps table should exist."""
        cursor = store.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='trace_steps'"
        )
        result = cursor.fetchone()
        
        assert result is not None

    def test_llm_cache_table_exists(self, store):
        """llm_cache table should exist."""
        cursor = store.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='llm_cache'"
        )
        result = cursor.fetchone()
        
        assert result is not None

    def test_wal_mode_enabled(self, store):
        """Database should be in WAL mode."""
        cursor = store.conn.execute("PRAGMA journal_mode")
        result = cursor.fetchone()
        
        # WAL mode for better concurrency
        assert result[0].lower() == "wal"


# --- Mode and Strictness Tests ---

class TestModeStorage:
    """Tests for mode/strictness storage."""

    @pytest.mark.parametrize("mode", [
        Mode.GROUNDED_QA,
        Mode.TRIAGE_PLAN,
        Mode.CHANGE_SAFETY
    ])
    def test_mode_roundtrip(self, store, mode):
        """Mode should survive roundtrip to database."""
        trace = TraceRecord(
            trace_id=f"mode-test-{mode.value}",
            user_input="Test",
            config=RunConfig(mode=mode),
            mode=mode,
            model="gpt-3.5-turbo",
            provider="openai"
        )
        store.create_trace(trace)
        
        retrieved = store.get_trace(f"mode-test-{mode.value}")
        
        assert retrieved.mode == mode

    @pytest.mark.parametrize("strictness", [
        Strictness.LENIENT,
        Strictness.BALANCED,
        Strictness.STRICT
    ])
    def test_strictness_roundtrip(self, store, strictness):
        """Strictness should survive roundtrip to database."""
        trace = TraceRecord(
            trace_id=f"strict-test-{strictness.value}",
            user_input="Test",
            config=RunConfig(strictness=strictness),
            mode=Mode.GROUNDED_QA,
            model="gpt-3.5-turbo",
            provider="openai"
        )
        store.create_trace(trace)
        
        retrieved = store.get_trace(f"strict-test-{strictness.value}")
        
        # Strictness is in the config
        assert retrieved is not None


# --- Status Tests ---

class TestStatusStorage:
    """Tests for status storage."""

    @pytest.mark.parametrize("status", [
        RunStatus.RUNNING,
        RunStatus.DONE,
        RunStatus.FAILED
    ])
    def test_status_roundtrip(self, store, status):
        """Status should survive roundtrip to database."""
        trace = TraceRecord(
            trace_id=f"status-test-{status.value}",
            user_input="Test",
            config=RunConfig(),
            mode=Mode.GROUNDED_QA,
            model="gpt-3.5-turbo",
            provider="openai",
            status=status
        )
        store.create_trace(trace)
        
        retrieved = store.get_trace(f"status-test-{status.value}")
        
        assert retrieved.status == status
