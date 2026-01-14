# ADR-0003: Provider Abstraction Pattern for LLM Integration

## Status

Accepted

## Context

TraceFlow Lite needs to support multiple LLM providers (OpenAI, Anthropic) with the ability to:

- Switch providers without changing application code
- Route different tasks to different models (e.g., fast model for planning, powerful model for execution)
- Add new providers with minimal effort
- Apply cross-cutting concerns (caching, retry, cost tracking) uniformly

Options considered:
1. **Direct SDK calls** - Simple but tightly coupled, no abstraction
2. **LangChain ChatModel** - Standard interface but heavy dependency
3. **Custom Protocol/ABC** - Lightweight abstraction, full control
4. **litellm** - Unified interface but additional dependency
5. **Provider factory pattern** - Abstraction with runtime selection

## Decision

Implement a custom provider abstraction using Python's Protocol pattern.

Architecture:
```python
class LLMProvider(Protocol):
    """Protocol defining the LLM provider interface."""
    
    def complete(
        self,
        messages: list[dict[str, str]],
        model: str,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> LLMResponse: ...
```

Components:
- **`base.py`** - Protocol definition and response types
- **`openai_provider.py`** - OpenAI implementation
- **`anthropic_provider.py`** - Anthropic implementation
- **`router.py`** - Model-to-provider routing logic
- **`cache_provider.py`** - Decorator for caching responses
- **`retry.py`** - Retry logic with exponential backoff
- **`cost.py`** - Token counting and cost calculation

## Consequences

### Positive

- **Loose coupling** - Application code depends on Protocol, not implementations
- **Easy testing** - Mock providers for unit tests without API calls
- **Uniform interface** - Same `complete()` signature across all providers
- **Decorator pattern** - Caching, retry, logging applied transparently
- **No heavy dependencies** - Only `openai` and `anthropic` SDKs needed
- **Type safety** - Protocol provides IDE autocomplete and type checking

### Negative

- **Maintenance burden** - Must update implementations for SDK changes
- **Feature gaps** - May not expose all provider-specific features
- **Translation overhead** - Must convert between provider message formats

### Neutral

- Each provider handles its own message format translation
- Response normalization strips provider-specific metadata
- Cost tracking requires manual price table maintenance

## References

- [Python Protocol (PEP 544)](https://peps.python.org/pep-0544/)
- [OpenAI Python SDK](https://github.com/openai/openai-python)
- [Anthropic Python SDK](https://github.com/anthropics/anthropic-sdk-python)
