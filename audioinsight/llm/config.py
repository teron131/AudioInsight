from typing import Optional

from pydantic import BaseModel, Field

from ..config import get_config


# Simple LLM config for UniversalLLM client
class LLMConfig(BaseModel):
    """Simple LLM configuration for UniversalLLM client."""

    model_id: str = Field(default="openai/gpt-4.1-nano", description="LLM model identifier")
    api_key: Optional[str] = Field(default=None, description="API key for LLM service")
    temperature: float = Field(default=0.0, description="Sampling temperature (fixed at 0.0 for consistency)")
    max_output_tokens: Optional[int] = Field(default=4000, gt=0, le=100000, description="Maximum output tokens")
    timeout: float = Field(default=30.0, gt=0, le=300, description="Request timeout in seconds")

    def model_post_init(self, __context) -> None:
        """Ensure temperature is always 0.0 for consistent results."""
        object.__setattr__(self, "temperature", 0.0)


class LLMTrigger(BaseModel):
    """Configuration for when to trigger LLM inference with validation."""

    # Text length limits are for memory management, not token limits
    max_text_length: int = Field(default=2000000, gt=10000, le=8000000, description="Maximum text length for memory management")
    min_text_length: int = Field(default=100, gt=0, le=1000, description="Minimum text length before considering triggers")
    summary_interval_seconds: float = Field(default=1.0, gt=0, le=300, description="Minimum time between summaries")
    new_text_trigger_chars: int = Field(default=300, gt=10, le=5000, description="Characters of new text to trigger summary")


class ParserConfig(BaseModel):
    """Configuration for text parser - focused on OUTPUT token limits for 1-to-1 conversion."""

    model_id: str = Field(default="openai/gpt-4.1-nano", description="Model ID for text parsing")
    max_output_tokens: int = Field(default=33000, gt=1000, le=100000, description="Maximum output tokens")
    trigger_interval_seconds: float = Field(default=1.0, gt=0.1, le=60.0, description="Minimum time between parser calls")

    def estimate_output_tokens(self, input_text: str) -> int:
        """Estimate output tokens for parser (1-to-1 conversion)."""
        if not input_text:
            return 0
        # ~4 chars per token average
        estimated_tokens = int(len(input_text) * 0.25)
        return max(1, estimated_tokens)

    def needs_chunking(self, input_text: str) -> bool:
        """Check if input text needs chunking based on output token limits."""
        estimated_output = self.estimate_output_tokens(input_text)
        return estimated_output > self.max_output_tokens

    def get_chunk_size_chars(self) -> int:
        """Get approximate input chunk size in characters."""
        # Convert output tokens back to input characters with 15% buffer
        return int(self.max_output_tokens * 4 * 0.85)


class SummarizerConfig(BaseModel):
    """Configuration for summarizer - massive input capacity, small output needs."""

    model_id: str = Field(default="openai/gpt-4.1-mini", description="Model ID for summarization")
    max_output_tokens: int = Field(default=4000, gt=100, le=8000, description="Maximum output tokens for summaries")
    max_input_length: int = Field(default=2000000, gt=10000, le=8000000, description="Maximum input length in characters")

    def can_handle_input(self, text: str) -> bool:
        """Check if input text can be handled without truncation."""
        return len(text) <= self.max_input_length

    def truncate_if_needed(self, text: str) -> str:
        """Truncate text if it exceeds input limits."""
        if len(text) <= self.max_input_length:
            return text
        # Truncate from the beginning, keeping the most recent content
        truncated = text[-self.max_input_length :]
        return f"[...truncated...]\n{truncated}"


# =============================================================================
# Configuration Factory Functions
# =============================================================================


def get_parser_config() -> ParserConfig:
    """Get parser configuration from main config."""
    config = get_config()
    return ParserConfig(
        model_id=config.llm.fast_llm,
        max_output_tokens=config.llm.parser_output_tokens,
        trigger_interval_seconds=config.llm.parser_trigger_interval,
    )


def get_summarizer_config() -> SummarizerConfig:
    """Get summarizer configuration from main config."""
    config = get_config()
    return SummarizerConfig(
        model_id=config.llm.base_llm,
        max_output_tokens=config.llm.summarizer_output_tokens,
        max_input_length=config.llm.summarizer_max_input_length,
    )


def get_llm_trigger() -> LLMTrigger:
    """Get LLM trigger configuration from main config."""
    config = get_config()
    return LLMTrigger(
        max_text_length=config.llm.summarizer_max_input_length,
        summary_interval_seconds=config.llm.llm_summary_interval,
        new_text_trigger_chars=config.llm.llm_new_text_trigger,
    )


# =============================================================================
# Domain-Specific Configuration Management
# =============================================================================


def get_runtime_settings() -> dict:
    """Get all runtime configurable LLM settings for settings page."""
    config = get_config()

    return {
        "fast_llm": config.llm.fast_llm,
        "base_llm": config.llm.base_llm,
        "llm_summary_interval": config.llm.llm_summary_interval,
        "llm_new_text_trigger": config.llm.llm_new_text_trigger,
        "parser_trigger_interval": config.llm.parser_trigger_interval,
        "parser_output_tokens": config.llm.parser_output_tokens,
        "llm_inference": config.features.llm_inference,
        "summarizer_output_tokens": config.llm.summarizer_output_tokens,
        "summarizer_max_input_length": config.llm.summarizer_max_input_length,
    }


def get_startup_settings() -> dict:
    """Get startup-only LLM settings (not configurable at runtime)."""
    config = get_config()

    return {
        "api_key": config.llm.api_key,
        "temperature": config.llm.temperature,
        "timeout": config.llm.timeout,
    }


def update_runtime_config(updates: dict) -> dict:
    """Update runtime LLM configuration and return the updated values."""
    config = get_config()
    updated = {}

    # LLM-specific fields that can be updated at runtime
    llm_fields = {"fast_llm", "base_llm", "llm_summary_interval", "llm_new_text_trigger", "parser_trigger_interval", "parser_output_tokens", "summarizer_output_tokens", "summarizer_max_input_length"}

    for key, value in updates.items():
        if key in llm_fields and hasattr(config.llm, key):
            setattr(config.llm, key, value)
            updated[key] = value
        elif key == "llm_inference" and hasattr(config.features, key):
            setattr(config.features, key, value)
            updated[key] = value

    return updated
