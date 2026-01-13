import os
from anthropic import Anthropic
from .base import BaseProvider, ProviderResponse
from .retry import llm_retry
from dotenv import load_dotenv

load_dotenv()

class AnthropicProvider(BaseProvider):
    """Anthropic Claude provider."""

    def __init__(self):
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        self.client = Anthropic(api_key=api_key)

    @llm_retry
    def chat_complete(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.2,
        max_tokens: int = 1024,
        **kwargs
    ) -> ProviderResponse:
        # Anthropic expects system prompt separately
        system_prompt = None
        chat_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                chat_messages.append(msg)

        response = self.client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system_prompt or "You are a helpful assistant.",
            messages=chat_messages,
            temperature=temperature,
        )

        return ProviderResponse(
            content=response.content[0].text,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            model=response.model,
        )