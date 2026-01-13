from .base import BaseProvider, ProviderResponse
from .cache import LLMCache


class CachedProvider(BaseProvider):
    """Wrapper that adds caching to any provider."""

    def __init__(self, provider: BaseProvider, enable_cache: bool = True):
        self.provider = provider
        self.cache = LLMCache() if enable_cache else None
        self.last_cache_hit = False

    def chat_complete(self, messages: list[dict], model: str, **kwargs) -> ProviderResponse:
        self.last_cache_hit = False

        if self.cache:
            cache_key = self.cache.compute_key(model, messages, **kwargs)
            cached = self.cache.get(cache_key)
            if cached:
                self.last_cache_hit = True
                return cached

        # Cache miss - call real provider
        response = self.provider.chat_complete(messages, model, **kwargs)

        if self.cache:
            self.cache.set(cache_key, model, response)

        return response
