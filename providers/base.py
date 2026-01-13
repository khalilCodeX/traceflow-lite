from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ProviderResponse:
    content: str
    input_tokens: int
    output_tokens: int
    model: str


class BaseProvider(ABC):
    @abstractmethod
    def chat_complete(self, messages: list[dict], model: str, **kwargs) -> ProviderResponse:
        """Send a chat completion request and return a normalized response"""
        pass
