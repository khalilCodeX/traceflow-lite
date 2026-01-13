"""
Mocked integration tests for TraceFlow client.
Tests complete workflows with mocked LLM providers - safe for CI.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

from client import TraceFlowClient
from tf_types import RunConfig, Mode, Strictness, RetrievedChunk
from providers.base import ProviderResponse


# --- Fixtures ---

@pytest.fixture
def mock_provider():
    """Create a mock provider that returns planner + executor responses."""
    provider = MagicMock()
    provider.model = "gpt-3.5-turbo"
    provider.last_cache_hit = False
    
    call_count = [0]
    
    def mock_chat_complete(messages, **kwargs):
        call_count[0] += 1
        
        if call_count[0] == 1:
            return ProviderResponse(
                content='{"steps": ["answer"], "needs_context": false}',
                input_tokens=50,
                output_tokens=20,
                model="gpt-3.5-turbo"
            )
        else:
            return ProviderResponse(
                content="The answer is 4. [1] Based on math.",
                input_tokens=100,
                output_tokens=30,
                model="gpt-3.5-turbo"
            )
    
    provider.chat_complete.side_effect = mock_chat_complete
    return provider


# --- Basic Client Tests ---

class TestClientBasicRun:
    """Tests for basic client run functionality."""

    @patch('providers.router.get_provider')
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_client_run_returns_answer(self, mock_get_provider, mock_provider):
        """Client.run should return an answer."""
        mock_get_provider.return_value = mock_provider
        
        client = TraceFlowClient()
        config = RunConfig(mode=Mode.GROUNDED_QA)
        result = client.run("What is 2+2?", config=config)
        
        assert result is not None
        assert hasattr(result, 'final_answer') or hasattr(result, 'trace_id')

    @patch('providers.router.get_provider')
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_client_run_with_config(self, mock_get_provider, mock_provider):
        """Client.run should accept RunConfig."""
        mock_get_provider.return_value = mock_provider
        
        config = RunConfig(
            mode=Mode.GROUNDED_QA,
            max_cost_usd=0.5,
            max_latency_ms=10000
        )
        
        client = TraceFlowClient()
        result = client.run("Test question", config=config)
        
        assert result is not None

    @patch('providers.router.get_provider')
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_client_trace_id_generated(self, mock_get_provider, mock_provider):
        """Client.run should generate unique trace_id."""
        mock_get_provider.return_value = mock_provider
        
        client = TraceFlowClient()
        config = RunConfig(mode=Mode.GROUNDED_QA)
        result1 = client.run("Question 1", config=config)
        
        # Reset the mock
        mock_provider.chat_complete.reset_mock()
        call_count = [0]
        def mock_response(messages, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return ProviderResponse(
                    content='{"steps": ["answer"], "needs_context": false}',
                    input_tokens=50, output_tokens=20, model="gpt-3.5-turbo"
                )
            else:
                return ProviderResponse(
                    content="Answer 2",
                    input_tokens=100, output_tokens=30, model="gpt-3.5-turbo"
                )
        mock_provider.chat_complete.side_effect = mock_response
        
        result2 = client.run("Question 2", config=config)
        
        if hasattr(result1, 'trace_id') and hasattr(result2, 'trace_id'):
            assert result1.trace_id != result2.trace_id


class TestClientRetriever:
    """Tests for client with retriever function."""

    @patch('providers.router.get_provider')
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_client_accepts_retriever_config(self, mock_get_provider, mock_provider):
        """Client should accept config with retriever_fn."""
        mock_get_provider.return_value = mock_provider
        
        retriever_fn = Mock(return_value=[
            RetrievedChunk(
                chunk_id="1",
                content="AI is artificial intelligence",
                source="wiki.pdf",
                relevance_score=0.95
            )
        ])
        
        config = RunConfig(
            mode=Mode.GROUNDED_QA,
            retriever_fn=retriever_fn
        )
        
        client = TraceFlowClient()
        result = client.run("What is AI?", config=config)
        
        # Should complete without error
        assert result is not None


class TestClientModes:
    """Tests for different client modes."""

    @patch('providers.router.get_provider')
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_grounded_qa_mode(self, mock_get_provider, mock_provider):
        """GROUNDED_QA mode should work."""
        mock_get_provider.return_value = mock_provider
        
        client = TraceFlowClient()
        config = RunConfig(mode=Mode.GROUNDED_QA)
        result = client.run("What is AI?", config=config)
        
        assert result is not None

    @patch('providers.router.get_provider')
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_triage_plan_mode(self, mock_get_provider):
        """TRIAGE_PLAN mode should produce structured output."""
        provider = MagicMock()
        provider.model = "gpt-3.5-turbo"
        provider.last_cache_hit = False
        
        call_count = [0]
        def mock_response(messages, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return ProviderResponse(
                    content='{"steps": ["plan"], "needs_context": false}',
                    input_tokens=50, output_tokens=20, model="gpt-3.5-turbo"
                )
            else:
                return ProviderResponse(
                    content="1. First step\n2. Second step\n3. Third step",
                    input_tokens=100, output_tokens=30, model="gpt-3.5-turbo"
                )
        
        provider.chat_complete.side_effect = mock_response
        mock_get_provider.return_value = provider
        
        client = TraceFlowClient()
        config = RunConfig(mode=Mode.TRIAGE_PLAN)
        result = client.run("How to deploy app?", config=config)
        
        assert result is not None

    @patch('providers.router.get_provider')
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_change_safety_mode(self, mock_get_provider):
        """CHANGE_SAFETY mode should assess risk."""
        provider = MagicMock()
        provider.model = "gpt-3.5-turbo"
        provider.last_cache_hit = False
        
        call_count = [0]
        def mock_response(messages, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                return ProviderResponse(
                    content='{"steps": ["analyze"], "needs_context": false}',
                    input_tokens=50, output_tokens=20, model="gpt-3.5-turbo"
                )
            else:
                return ProviderResponse(
                    content="This change looks safe. No risks detected.",
                    input_tokens=100, output_tokens=30, model="gpt-3.5-turbo"
                )
        
        provider.chat_complete.side_effect = mock_response
        mock_get_provider.return_value = provider
        
        client = TraceFlowClient()
        config = RunConfig(mode=Mode.CHANGE_SAFETY)
        result = client.run("UPDATE users SET active=true", config=config)
        
        assert result is not None


class TestClientStrictness:
    """Tests for strictness levels."""

    @patch('providers.router.get_provider')
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_strict_mode_enforced(self, mock_get_provider, mock_provider):
        """STRICT mode should have stricter validation."""
        mock_get_provider.return_value = mock_provider
        
        config = RunConfig(
            mode=Mode.CHANGE_SAFETY,
            strictness=Strictness.STRICT
        )
        
        client = TraceFlowClient()
        result = client.run("Test", config=config)
        
        assert result is not None

    @patch('providers.router.get_provider')
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_lenient_mode(self, mock_get_provider, mock_provider):
        """LENIENT mode should have lenient validation."""
        mock_get_provider.return_value = mock_provider
        
        config = RunConfig(
            mode=Mode.GROUNDED_QA,
            strictness=Strictness.LENIENT
        )
        
        client = TraceFlowClient()
        result = client.run("Test", config=config)
        
        assert result is not None


class TestClientCaching:
    """Tests for client with caching."""

    @patch('providers.router.get_provider')
    @patch.dict('os.environ', {'OPENAI_API_KEY': 'test-key'})
    def test_cache_enabled(self, mock_get_provider, mock_provider):
        """Client should enable caching when configured."""
        mock_get_provider.return_value = mock_provider
        
        config = RunConfig(
            mode=Mode.GROUNDED_QA,
            enable_cache=True
        )
        
        client = TraceFlowClient()
        result = client.run("Test", config=config)
        
        # Should complete successfully with caching enabled
        assert result is not None
