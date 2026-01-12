from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential_jitter,
    retry_if_exception_type,
)
from openai import RateLimitError, APIConnectionError, APIStatusError

def is_retryable_error(exception: BaseException) -> bool:
    """Check if the exception is retryable."""
    if isinstance(exception, (RateLimitError, APIConnectionError)):
        return True
    if isinstance(exception, APIStatusError) and exception.status_code >= 500:
        return True
    return False

llm_retry = retry(
    wait=wait_exponential_jitter(initial=1, max=20, jitter=5),
    stop=stop_after_attempt(3),
    retry=retry_if_exception_type(is_retryable_error)
)