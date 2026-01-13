from .anthropic_provider import AnthropicProvider
from tf_types import RunConfig
from .base import BaseProvider
from .openai_provider import OpenAIProvider

_PROVIDERS: dict[str, type[BaseProvider]] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
}

def get_provider(config: RunConfig) -> BaseProvider:
    """Factory to get the appropriate provider based on config."""
    provider_cls = _PROVIDERS.get(config.provider)
    if not provider_cls:
        raise ValueError(f"Unsupported provider: {config.provider}")
    return provider_cls()