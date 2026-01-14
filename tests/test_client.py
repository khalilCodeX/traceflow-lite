# tests/test_client.py

import os
import pytest
from client import TraceFlowClient

pytestmark = pytest.mark.requires_api  # All tests in this file need API keys
from tf_types import RunConfig, Mode, RetrievedChunk
from utils.retriever_utils import chroma_retriever
from utils.vector_types import chroma_params


@pytest.fixture
def client():
    """Create a test client."""
    return TraceFlowClient()


def test_basic_run_without_retriever(client):
    """Test a basic run with executor only."""
    config = RunConfig(mode=Mode.GROUNDED_QA, model="gpt-3.5-turbo", max_tokens=100)
    result = client.run("What is 2+2?", config)

    assert result.status == "done"
    assert result.answer
    assert len(result.answer) > 0


def test_run_with_retriever(client):
    """Test run with a retriever callback."""

    # Mock retriever function
    def mock_retriever(query: str) -> list[RetrievedChunk]:
        return [
            RetrievedChunk(
                chunk_id="1", content="2+2 equals 4", source="math_kb", relevance_score=0.95
            )
        ]

    config = RunConfig(mode=Mode.GROUNDED_QA, model="gpt-3.5-turbo", retriever_fn=mock_retriever)
    result = client.run("What is 2+2?", config)

    assert result.status == "done"
    assert "4" in result.answer.lower()


def test_cost_constraint(client):
    """Test that cost constraints trigger fallback."""
    config = RunConfig(
        model="gpt-3.5-turbo",
        max_cost_usd=0.00001,  # Impossibly low
    )
    result = client.run("Expensive question", config)

    # Should either fallback or fail due to cost
    assert result.status in ["done", "failed"]


def test_latency_constraint(client):
    """Test that latency constraints trigger fallback."""
    config = RunConfig(
        model="gpt-3.5-turbo",
        max_latency_ms=1,  # Impossibly low
        max_revisions=0,
    )
    result = client.run("Quick question", config)

    # Should fallback due to latency
    assert result.status == "done"
    # Eval decided to fallback
    assert result.eval_decision.decision == "fallback"


def test_trace_persistence(client):
    """Test that traces are saved and retrievable."""
    config = RunConfig(mode=Mode.GROUNDED_QA)
    result = client.run("Test question", config)

    # Get the trace back
    trace = client.get_trace(result.trace_id)

    assert trace.trace_id == result.trace_id
    assert trace.user_input == "Test question"
    assert trace.final_answer == result.answer


def test_list_traces(client):
    """Test listing traces."""
    # Run a few times
    config = RunConfig(mode=Mode.GROUNDED_QA)
    result1 = client.run("Question 1", config)
    result2 = client.run("Question 2", config)

    traces = client.list_traces(limit=10)

    assert len(traces) >= 2
    trace_ids = [t.trace_id for t in traces]
    assert result1.trace_id in trace_ids
    assert result2.trace_id in trace_ids


def test_replay(client):
    """Test replaying a trace with different config."""
    config = RunConfig(model="gpt-3.5-turbo", max_tokens=100)
    result1 = client.run("Original question", config)

    # Replay with different config
    new_config = RunConfig(model="gpt-3.5-turbo", max_tokens=50)
    result2 = client.replay(result1.trace_id, new_config)

    assert result2.trace_id != result1.trace_id
    assert result2.answer  # Got a new answer


def test_different_strictness_levels(client):
    """Test evaluator with different strictness levels."""
    for strictness in ["lenient", "balanced", "strict"]:
        config = RunConfig(strictness=strictness)
        result = client.run("Test question", config)
        assert result.status == "done"


def test_chroma_retriever_integration():
    """Test with actual Chroma setup."""
    import shutil

    # Sample documents
    documents = [
        "Artificial Intelligence (AI) is the simulation of human intelligence by machines.",
        "Machine Learning (ML) is a subset of AI that learns from data.",
        "Deep Learning uses neural networks with many layers.",
        "Natural Language Processing (NLP) helps computers understand human language.",
    ]

    # Setup
    test_dir = "./test_chroma_db"
    params = chroma_params(documents=documents, collection="test_kb", directory=test_dir)

    try:
        retriever = chroma_retriever(local=True, params=params)
        retriever.create_vector_store(documents)

        # Test retrieval
        chunks = retriever.retrieve_similar_docs("What is AI?", n_results=2)
        assert len(chunks) > 0
        assert any("AI" in chunk.content or "Artificial" in chunk.content for chunk in chunks)

        # Test with client
        client = TraceFlowClient()
        config = RunConfig(retriever_fn=retriever.retrieve_similar_docs)
        result = client.run("What is AI?", config)

        assert result.status == "done"
        assert result.answer

    finally:
        # Cleanup test database
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
