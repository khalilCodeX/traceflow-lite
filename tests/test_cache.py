"""
Tests for LLM caching functionality.
These tests use mocking and don't require API keys.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import hashlib
import json

from providers.cache import LLMCache
from providers.cache_provider import CachedProvider
from providers.base import ProviderResponse


class TestLLMCache:
    """Tests for the LLMCache class."""

    @pytest.fixture
    def mock_conn(self):
        """Create a mock database connection."""
        conn = MagicMock()
        conn.execute.return_value.fetchone.return_value = None
        return conn

    @pytest.fixture
    def cache(self, mock_conn):
        """Create an LLMCache with mocked database."""
        with patch('providers.cache.Sqlite') as mock_sqlite:
            mock_sqlite.get_connection.return_value = mock_conn
            mock_sqlite.init_db.return_value = None
            cache = LLMCache()
            cache.conn = mock_conn
            return cache

    def test_compute_key_deterministic(self, cache):
        """Same inputs should produce same cache key."""
        messages = [{"role": "user", "content": "Hello"}]
        
        key1 = cache.compute_key(model="gpt-4", messages=messages, temperature=0.2)
        key2 = cache.compute_key(model="gpt-4", messages=messages, temperature=0.2)
        
        assert key1 == key2

    def test_compute_key_different_for_different_inputs(self, cache):
        """Different inputs should produce different cache keys."""
        messages = [{"role": "user", "content": "Hello"}]
        
        key1 = cache.compute_key(model="gpt-4", messages=messages, temperature=0.2)
        key2 = cache.compute_key(model="gpt-4", messages=messages, temperature=0.5)
        key3 = cache.compute_key(model="gpt-3.5-turbo", messages=messages, temperature=0.2)
        
        assert key1 != key2
        assert key1 != key3

    def test_compute_key_different_for_different_messages(self, cache):
        """Different message content should produce different cache keys."""
        key1 = cache.compute_key(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0.2
        )
        key2 = cache.compute_key(
            model="gpt-4",
            messages=[{"role": "user", "content": "Goodbye"}],
            temperature=0.2
        )
        
        assert key1 != key2

    def test_compute_key_is_sha256_hash(self, cache):
        """Cache key should be a valid SHA256 hash."""
        messages = [{"role": "user", "content": "Test"}]
        key = cache.compute_key(model="gpt-4", messages=messages)
        
        # SHA256 produces 64 hex characters
        assert len(key) == 64
        assert all(c in '0123456789abcdef' for c in key)

    def test_get_returns_none_on_cache_miss(self, cache, mock_conn):
        """get() should return None when key not found."""
        mock_conn.execute.return_value.fetchone.return_value = None
        
        result = cache.get("nonexistent_key")
        
        assert result is None

    def test_get_returns_response_on_cache_hit(self, cache, mock_conn):
        """get() should return ProviderResponse when key found."""
        mock_conn.execute.return_value.fetchone.return_value = (
            "Hello, world!",  # response_content
            10,               # input_tokens
            5,                # output_tokens
            "gpt-4"           # model
        )
        
        result = cache.get("existing_key")
        
        assert result is not None
        assert isinstance(result, ProviderResponse)
        assert result.content == "Hello, world!"
        assert result.input_tokens == 10
        assert result.output_tokens == 5
        assert result.model == "gpt-4"

    def test_get_increments_hit_count(self, cache, mock_conn):
        """get() should increment hit_count on cache hit."""
        mock_conn.execute.return_value.fetchone.return_value = (
            "Hello", 10, 5, "gpt-4"
        )
        
        cache.get("existing_key")
        
        # Check that UPDATE was called to increment hit_count
        update_calls = [
            call for call in mock_conn.execute.call_args_list
            if "UPDATE llm_cache SET hit_count" in str(call)
        ]
        assert len(update_calls) == 1

    def test_set_inserts_into_database(self, cache, mock_conn):
        """set() should insert response into database."""
        response = ProviderResponse(
            content="Test response",
            input_tokens=15,
            output_tokens=8,
            model="gpt-4"
        )
        
        cache.set("test_key", "gpt-4", response)
        
        # Check INSERT was called
        insert_calls = [
            call for call in mock_conn.execute.call_args_list
            if "INSERT OR REPLACE INTO llm_cache" in str(call)
        ]
        assert len(insert_calls) == 1
        mock_conn.commit.assert_called()


class TestCachedProvider:
    """Tests for the CachedProvider wrapper."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mock base provider."""
        provider = Mock()
        provider.chat_complete.return_value = ProviderResponse(
            content="Fresh response",
            input_tokens=20,
            output_tokens=10,
            model="gpt-4"
        )
        return provider

    @pytest.fixture
    def mock_cache(self):
        """Create a mock LLMCache."""
        cache = Mock(spec=LLMCache)
        cache.compute_key.return_value = "test_cache_key"
        cache.get.return_value = None  # Default to cache miss
        return cache

    def test_cache_miss_calls_provider(self, mock_provider, mock_cache):
        """On cache miss, should call underlying provider."""
        with patch('providers.cache_provider.LLMCache', return_value=mock_cache):
            cached = CachedProvider(mock_provider, enable_cache=True)
            cached.cache = mock_cache
            
            messages = [{"role": "user", "content": "Hello"}]
            result = cached.chat_complete(messages, model="gpt-4")
            
            mock_provider.chat_complete.assert_called_once()
            assert result.content == "Fresh response"
            assert cached.last_cache_hit is False

    def test_cache_hit_skips_provider(self, mock_provider, mock_cache):
        """On cache hit, should return cached response without calling provider."""
        cached_response = ProviderResponse(
            content="Cached response",
            input_tokens=10,
            output_tokens=5,
            model="gpt-4"
        )
        mock_cache.get.return_value = cached_response
        
        with patch('providers.cache_provider.LLMCache', return_value=mock_cache):
            cached = CachedProvider(mock_provider, enable_cache=True)
            cached.cache = mock_cache
            
            messages = [{"role": "user", "content": "Hello"}]
            result = cached.chat_complete(messages, model="gpt-4")
            
            mock_provider.chat_complete.assert_not_called()
            assert result.content == "Cached response"
            assert cached.last_cache_hit is True

    def test_cache_miss_stores_response(self, mock_provider, mock_cache):
        """On cache miss, should store response in cache."""
        with patch('providers.cache_provider.LLMCache', return_value=mock_cache):
            cached = CachedProvider(mock_provider, enable_cache=True)
            cached.cache = mock_cache
            
            messages = [{"role": "user", "content": "Hello"}]
            cached.chat_complete(messages, model="gpt-4")
            
            mock_cache.set.assert_called_once()

    def test_cache_disabled_always_calls_provider(self, mock_provider):
        """When cache disabled, should always call provider."""
        cached = CachedProvider(mock_provider, enable_cache=False)
        
        messages = [{"role": "user", "content": "Hello"}]
        result = cached.chat_complete(messages, model="gpt-4")
        
        mock_provider.chat_complete.assert_called_once()
        assert result.content == "Fresh response"
        assert cached.last_cache_hit is False

    def test_last_cache_hit_resets_each_call(self, mock_provider, mock_cache):
        """last_cache_hit should reset at the start of each call."""
        cached_response = ProviderResponse(
            content="Cached", input_tokens=10, output_tokens=5, model="gpt-4"
        )
        
        with patch('providers.cache_provider.LLMCache', return_value=mock_cache):
            cached = CachedProvider(mock_provider, enable_cache=True)
            cached.cache = mock_cache
            
            # First call - cache hit
            mock_cache.get.return_value = cached_response
            cached.chat_complete([{"role": "user", "content": "Hello"}], model="gpt-4")
            assert cached.last_cache_hit is True
            
            # Second call - cache miss
            mock_cache.get.return_value = None
            cached.chat_complete([{"role": "user", "content": "World"}], model="gpt-4")
            assert cached.last_cache_hit is False


class TestCacheKeyConsistency:
    """Tests to ensure cache key generation is consistent and correct."""

    def test_key_order_independence(self):
        """Cache key should be the same regardless of kwarg order."""
        with patch('providers.cache.Sqlite') as mock_sqlite:
            mock_sqlite.get_connection.return_value = MagicMock()
            cache = LLMCache()
        
        messages = [{"role": "user", "content": "Test"}]
        
        # Call with kwargs in different orders
        key1 = cache.compute_key(model="gpt-4", messages=messages, temperature=0.2, max_tokens=100)
        key2 = cache.compute_key(messages=messages, model="gpt-4", max_tokens=100, temperature=0.2)
        
        # Keys should be identical because we use sort_keys=True in JSON serialization
        assert key1 == key2

    def test_message_order_matters(self):
        """Different message order should produce different keys."""
        with patch('providers.cache.Sqlite') as mock_sqlite:
            mock_sqlite.get_connection.return_value = MagicMock()
            cache = LLMCache()
        
        messages1 = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"}
        ]
        messages2 = [
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "Hello"}
        ]
        
        key1 = cache.compute_key(model="gpt-4", messages=messages1)
        key2 = cache.compute_key(model="gpt-4", messages=messages2)
        
        assert key1 != key2
