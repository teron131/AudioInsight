from typing import Optional

from pydantic import BaseModel, Field

from ..config import get_config
from ..logging_config import get_logger

logger = get_logger(__name__)


# Simple LLM config for UniversalLLM client
class LLMConfig(BaseModel):
    """Configuration for LLM operations.

    Uses environment variables for secure API key management.
    """

    model_id: str = Field(default="openai/gpt-4.1-mini", description="Model identifier")
    api_key: Optional[str] = Field(default=None, description="API key (optional, uses env vars)")
    max_output_tokens: int = Field(default=6000, description="Maximum output tokens")
    temperature: float = Field(default=0.3, description="Temperature for generation")
    timeout: float = Field(default=15.0, description="Request timeout in seconds")

    class Config:
        """Pydantic model configuration."""

        arbitrary_types_allowed = True

    def model_post_init(self, __pydantic_context__) -> None:
        """Ensure temperature is always 0.0 for consistent results."""
        object.__setattr__(self, "temperature", 0.0)


class LLMTrigger(BaseModel):
    """Configuration for LLM trigger conditions."""

    # ENHANCED: More aggressive triggering for better coverage
    analysis_interval_seconds: float = Field(default=5.0, description="Trigger analysis after this many seconds (reduced from 15.0)")
    new_text_trigger_chars: int = Field(default=50, description="Trigger analysis after this many new characters (reduced from 200)")
    min_text_length: int = Field(default=30, description="Minimum text length before triggering analysis (reduced from 50)")
    max_text_length: int = Field(default=8000, description="Maximum text length for analysis (reduced from 10000)")

    # Additional triggering conditions for more comprehensive coverage
    force_analysis_on_silence: bool = Field(default=True, description="Force analysis when processing stops")
    min_words_for_analysis: int = Field(default=15, description="Minimum words needed for analysis (reduced from 20)")


class ParserConfig(LLMConfig):
    """Configuration for transcript parsing operations."""

    model_id: str = Field(default="openai/gpt-4.1-nano", description="Fast model for parsing")
    max_output_tokens: int = Field(default=33000, description="Large token limit for comprehensive parsing")
    trigger_interval_seconds: float = Field(default=0.8, description="Minimum time between parsing triggers")

    def needs_chunking(self) -> bool:
        """Check if input needs to be chunked based on model limits."""
        # For nano models, chunk if input exceeds ~16K chars (safe estimate)
        return "nano" in self.model_id.lower()

    def get_chunk_size_chars(self) -> int:
        """Get appropriate chunk size for the model."""
        if "nano" in self.model_id.lower():
            return 15000  # Conservative chunk size for nano models
        return 50000  # Larger chunk size for bigger models


class AnalyzerConfig(LLMConfig):
    """Configuration for conversation analysis."""

    model_id: str = Field(default="openai/gpt-4.1-mini", description="Model for analysis")
    max_output_tokens: int = Field(default=6000, description="Output tokens for analyses")


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


def get_analyzer_config() -> AnalyzerConfig:
    """Get analyzer configuration from main config."""
    config = get_config()
    return AnalyzerConfig(
        model_id=config.llm.base_llm,
        max_output_tokens=config.llm.analyzer_output_tokens,
    )


def get_llm_trigger() -> LLMTrigger:
    """Get LLM trigger configuration from main config."""
    config = get_config()
    return LLMTrigger(
        max_text_length=config.llm.analyzer_max_input_length,
        analysis_interval_seconds=config.llm.llm_analysis_interval,
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
        "llm_analysis_interval": config.llm.llm_analysis_interval,
        "llm_new_text_trigger": config.llm.llm_new_text_trigger,
        "parser_trigger_interval": config.llm.parser_trigger_interval,
        "parser_output_tokens": config.llm.parser_output_tokens,
        "llm_inference": config.features.llm_inference,
        "analyzer_output_tokens": config.llm.analyzer_output_tokens,
        "analyzer_max_input_length": config.llm.analyzer_max_input_length,
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
    llm_fields = {"fast_llm", "base_llm", "llm_analysis_interval", "llm_new_text_trigger", "parser_trigger_interval", "parser_output_tokens", "analyzer_output_tokens", "analyzer_max_input_length"}

    for key, value in updates.items():
        if key in llm_fields and hasattr(config.llm, key):
            setattr(config.llm, key, value)
            updated[key] = value
        elif key == "llm_inference" and hasattr(config.features, key):
            setattr(config.features, key, value)
            updated[key] = value

    return updated
