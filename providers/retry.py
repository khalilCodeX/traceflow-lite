from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential_jitter,
    retry_if_exception_type,
)
from openai import RateLimitError, APIConnectionError, APIStatusError

try:
    from anthropic import RateLimitError as AnthropicRateLimitError
    from anthropic import APIStatusError as AnthropicAPIStatusError
    ANTHROPIC_ERRORS = (AnthropicRateLimitError, AnthropicAPIStatusError)
except ImportError:
    ANTHROPIC_ERRORS = ()

llm_retry = retry(
    wait=wait_exponential_jitter(initial=1, max=20, jitter=5),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type((RateLimitError, APIStatusError) + ANTHROPIC_ERRORS)
)