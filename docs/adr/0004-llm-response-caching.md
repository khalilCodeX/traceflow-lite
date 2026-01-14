# ADR-0004: LLM Response Caching Strategy

## Status

Accepted

## Context

LLM API calls are expensive (cost) and slow (latency). During development and testing, the same prompts are often sent repeatedly. TraceFlow Lite needs a caching mechanism to:

- Reduce API costs during development iterations
- Speed up test execution
- Enable deterministic replay for debugging
- Support offline development with cached responses

Requirements:
- Cache key must uniquely identify equivalent requests
- Cache must be persistent across sessions
- Cache hits must be trackable for observability
- Caching must be toggleable (disable for production)

Options considered:
1. **In-memory LRU cache** - Fast but not persistent, lost on restart
2. **Redis** - Persistent but requires external service
3. **File-based (JSON)** - Simple but poor query performance
4. **SQLite table** - Persistent, queryable, no new dependencies
5. **LangChain caching** - Built-in but couples to LangChain

## Decision

Implement SQLite-based LLM response caching with content-addressable keys.

Cache key computation:
```python
def compute_key(
    model: str,
    messages: list[dict],
    temperature: float,
    max_tokens: int
) -> str:
    """SHA256 hash of normalized request parameters."""
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()
```

Schema:
```sql
CREATE TABLE llm_cache (
    cache_key TEXT PRIMARY KEY,
    model TEXT,
    response_content TEXT,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    created_at TEXT
)
```

Integration via decorator pattern:
```python
class CachedProvider:
    """Wrapper that checks cache before calling underlying provider."""
    
    def complete(self, messages, model, **kwargs) -> LLMResponse:
        key = self.cache.compute_key(model, messages, **kwargs)
        if cached := self.cache.get(key):
            return LLMResponse(..., cache_hit=True)
        response = self.provider.complete(messages, model, **kwargs)
        self.cache.set(key, response)
        return response
```

## Consequences

### Positive

- **Cost savings** - Identical requests return cached responses instantly
- **Faster iteration** - Development loop accelerated significantly
- **Deterministic tests** - Same input always produces same output
- **Observability** - `cache_hit` field tracks cache effectiveness
- **No new dependencies** - Reuses existing SQLite infrastructure
- **Toggle support** - Enable/disable via UI or configuration

### Negative

- **Stale responses** - Cache doesn't invalidate when models update
- **Storage growth** - Cache table grows unbounded (manual cleanup needed)
- **Temperature sensitivity** - Different temperatures = different cache keys
- **Not suitable for production** - Should be disabled for real user requests

### Neutral

- Cache key includes all parameters affecting output
- JSON normalization with `sort_keys=True` ensures consistent hashing
- Cache miss incurs small overhead for key computation

## References

- [Content-addressable storage](https://en.wikipedia.org/wiki/Content-addressable_storage)
- [SHA-256 Hash Function](https://docs.python.org/3/library/hashlib.html)
