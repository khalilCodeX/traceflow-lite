import os
from openai import OpenAI
from providers.base import BaseProvider, ProviderResponse
from .retry import llm_retry
from dotenv import load_dotenv

load_dotenv()


class OpenAIProvider(BaseProvider):
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    @llm_retry
    def chat_complete(self, messages: list[dict], model: str, **kwargs) -> ProviderResponse:
        response = self.client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=kwargs.get("temperature", 0.2),
            max_tokens=kwargs.get("max_tokens", 100),
        )

        choice = response.choices[0]
        content = choice.message.content
        usage = response.usage

        return ProviderResponse(
            content=content,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            model=model,
        )
