import logging
import tiktoken

# Pricing per 1M tokens (input, output) as of Jan 2026
PRICING: dict[str, dict[str, float]] = {
    # we can make this pricing more robust in the future by loading from API
    # OpenAI models
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    # Anthropic Claude models (per 1M tokens)
    "claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "claude-3-5-haiku-20241022": {"input": 0.80, "output": 4.00},
    "claude-3-opus-20240229": {"input": 15.00, "output": 75.00},
    "claude-3-sonnet-20240229": {"input": 3.00, "output": 15.00},
    "claude-3-haiku-20240307": {"input": 0.25, "output": 1.25},
}


def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """Calculate cost of given text for specified model."""

    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        logging.warning(
            f"Model {model} not found. Falling back to cl100k_base (GPT-4o/o1/GPT-5 standard)."
        )
        encoding = tiktoken.get_encoding("cl100k_base")

    return len(encoding.encode(text))


def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate cost in USD from token counts."""

    pricing = PRICING.get(model)
    if pricing is None:
        logging.warning(f"Pricing for model {model} not found. Defaulting cost to $0.0")
        return 0.0

    input_cost = (input_tokens / 1_000_000) * PRICING[model]["input"]
    output_cost = (output_tokens / 1_000_000) * PRICING[model]["output"]

    total_cost = input_cost + output_cost
    return round(total_cost, 6)  # Round to microdollar precision
