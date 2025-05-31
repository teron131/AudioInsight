from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class LLMConfig(BaseSettings):
    """Base configuration for LLM operations with environment variable support."""

    model_id: str = Field(default="openai/gpt-4.1-mini", description="LLM model identifier")
    api_key: Optional[str] = Field(default=None, env="OPENROUTER_API_KEY", description="API key for LLM service")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0, description="Sampling temperature for text generation")
    # Modern LLMs have massive input capacity (1M+ tokens), no need to limit
    # Output limits depend on use case - summarizer needs less, parser needs more
    max_output_tokens: Optional[int] = Field(default=4000, gt=0, le=100000, description="Maximum output tokens to generate")  # Conservative default for summarization tasks
    timeout: float = Field(default=30.0, gt=0, le=300, description="Request timeout in seconds")

    class Config:
        env_file = ".env"
        env_prefix = "LLM_"
        extra = "ignore"  # Ignore extra environment variables


class LLMTrigger(BaseModel):
    """Configuration for when to trigger LLM inference with validation."""

    # Text length limits are for memory management, not token limits
    # Modern LLMs can handle 1M+ input tokens easily
    max_text_length: int = Field(default=500000, gt=10000, le=2000000, description="Maximum text length for memory management (not token limit)")  # ~125k tokens at 4 chars/token - well within 1M limit  # ~500k tokens - still safe for 1M input models
    min_text_length: int = Field(default=100, gt=0, le=1000, description="Minimum text length before considering triggers")
    summary_interval_seconds: float = Field(default=1.0, gt=0, le=300, description="Minimum time between summaries (OR condition)")
    new_text_trigger_chars: int = Field(default=100, gt=10, le=5000, description="Characters of new text to trigger summary (OR condition)")


class ParserConfig(BaseModel):
    """Configuration for text parser - focused on OUTPUT token limits for 1-to-1 conversion."""

    model_id: str = Field(default="openai/gpt-4.1-nano", description="Model ID for text parsing (should be fast/cheap)")
    # Parser does 1-to-1 conversion, so output tokens matter for chunking
    # Input is never a concern with modern LLMs (1M+ capacity)
    max_output_tokens: int = Field(default=33000, gt=1000, le=100000, description="Maximum output tokens the model can generate (determines chunking)")  # Model's actual output limit
    trigger_interval_seconds: float = Field(default=1.0, gt=0.1, le=60.0, description="Minimum time between parser calls")

    # Simple token estimation - only used for output size estimation
    def estimate_output_tokens(self, input_text: str) -> int:
        """Estimate output tokens for parser (1-to-1 conversion).

        Args:
            input_text: Input text to be parsed

        Returns:
            int: Estimated output token count
        """
        if not input_text:
            return 0

        # For parsing, output ≈ input size (1-to-1 conversion)
        # ~4 chars per token average
        estimated_tokens = int(len(input_text) * 0.25)
        return max(1, estimated_tokens)

    def needs_chunking(self, input_text: str) -> bool:
        """Check if input text needs chunking based on output token limits.

        Args:
            input_text: Text to check

        Returns:
            bool: True if text needs to be chunked
        """
        estimated_output = self.estimate_output_tokens(input_text)
        return estimated_output > self.max_output_tokens

    def get_chunk_size_chars(self) -> int:
        """Get approximate input chunk size in characters.

        Returns:
            int: Characters per chunk based on output token limits
        """
        # Convert output tokens back to input characters
        # Leave 15% buffer for parsing expansion
        return int(self.max_output_tokens * 4 * 0.85)


class ParsedTranscript(BaseModel):
    """Structured response from transcript parsing."""

    original_text: str = Field(description="Original transcribed text")
    parsed_text: str = Field(description="Parsed and corrected text")
    segments: List[Dict[str, Any]] = Field(default_factory=list, description="Text segments with metadata")
    timestamps: Dict[str, float] = Field(default_factory=dict, description="Important timestamps")
    speakers: List[Dict[str, Any]] = Field(default_factory=list, description="Speaker information")
    parsing_time: float = Field(default=0.0, description="Time taken to parse the transcript")


class LLMResponse(BaseModel):
    """Structured response from the LLM inference."""

    summary: str = Field(description="Concise summary of the transcription")
    key_points: list[str] = Field(default_factory=list, description="Main points discussed")


class BaseStats:
    """Base class for statistics tracking with common patterns."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all statistics. Should be overridden by subclasses."""
        pass

    def to_dict(self) -> Dict[str, Any]:
        """Get statistics as a dictionary. Should be overridden by subclasses."""
        return {}

    def update_average_time(self, current_avg: float, count: int, new_time: float) -> float:
        """Helper method to update running average time."""
        if count == 0:
            return new_time
        return (current_avg * (count - 1) + new_time) / count


class ParserStats(BaseStats):
    """Statistics for core text parser operations."""

    def reset(self):
        self.texts_processed = 0
        self.total_chars_processed = 0
        self.average_processing_time = 0.0
        self.chunks_processed = 0

    def record_processing(self, processing_time: float, chars_processed: int, chunks_used: int = 1):
        """Record a processing operation."""
        self.texts_processed += 1
        self.total_chars_processed += chars_processed
        self.chunks_processed += chunks_used
        self.average_processing_time = self.update_average_time(self.average_processing_time, self.texts_processed, processing_time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "texts_processed": self.texts_processed,
            "total_chars_processed": self.total_chars_processed,
            "average_processing_time": self.average_processing_time,
            "chunks_processed": self.chunks_processed,
        }


class DisplayParserStats(BaseStats):
    """Statistics for display parser operations with caching."""

    def reset(self):
        self.texts_parsed = 0
        self.cache_hits = 0
        self.total_parse_time = 0.0
        self.average_parse_time = 0.0

    def record_cache_hit(self):
        """Record a cache hit."""
        self.cache_hits += 1

    def record_parsing(self, parse_time: float):
        """Record a parsing operation (cache miss)."""
        self.texts_parsed += 1
        self.total_parse_time += parse_time
        self.average_parse_time = self.update_average_time(self.average_parse_time, self.texts_parsed, parse_time)

    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total_requests = self.texts_parsed + self.cache_hits
        return self.cache_hits / total_requests if total_requests > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "texts_parsed": self.texts_parsed,
            "cache_hits": self.cache_hits,
            "total_parse_time": self.total_parse_time,
            "average_parse_time": self.average_parse_time,
            "cache_hit_rate": self.cache_hit_rate,
            "total_requests": self.texts_parsed + self.cache_hits,
        }


class LLMStats:
    """Statistics tracking for LLM operations."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all statistics."""
        self.inference_generated = 0
        self.total_text_processed = 0
        self.average_inference_time = 0.0
        self.inference_by_time_interval = 0
        self.inference_by_new_text = 0
        self.inference_by_forced = 0

    def record_inference(self, trigger_reason: str, processing_time: float, text_length: int):
        """Record an inference operation.

        Args:
            trigger_reason: Reason for triggering the inference
            processing_time: Time taken to process
            text_length: Length of text processed
        """
        self.inference_generated += 1
        self.total_text_processed += text_length

        # Update average time
        prev_avg = self.average_inference_time
        count = self.inference_generated
        self.average_inference_time = (prev_avg * (count - 1) + processing_time) / count

        # Track trigger reason statistics
        if trigger_reason == "time_interval":
            self.inference_by_time_interval += 1
        elif trigger_reason == "new_text":
            self.inference_by_new_text += 1
        elif trigger_reason == "forced":
            self.inference_by_forced += 1

    def to_dict(self) -> Dict[str, Any]:
        """Get statistics as a dictionary."""
        return {
            "inference_generated": self.inference_generated,
            "total_text_processed": self.total_text_processed,
            "average_inference_time": self.average_inference_time,
            "inference_by_time_interval": self.inference_by_time_interval,
            "inference_by_new_text": self.inference_by_new_text,
            "inference_by_forced": self.inference_by_forced,
        }


class SummarizerConfig(BaseModel):
    """Configuration for summarizer - massive input capacity, small output needs."""

    model_id: str = Field(default="openai/gpt-4.1-mini", description="Model ID for summarization (can be more powerful)")
    # Summarizers can handle massive input (1M+ tokens) and produce small output
    max_output_tokens: int = Field(default=4000, gt=100, le=8000, description="Maximum output tokens for summaries (much smaller than parser)")  # Much smaller than parser - summaries are concise  # Reasonable limit for summaries
    # Input is practically unlimited with modern LLMs
    max_input_length: int = Field(default=2000000, gt=10000, le=8000000, description="Maximum input length in characters (memory management only)")  # ~500k tokens - well within 1M limit  # ~2M tokens - extremely generous

    def can_handle_input(self, text: str) -> bool:
        """Check if input text can be handled without truncation.

        Args:
            text: Input text to check

        Returns:
            bool: True if text can be processed without truncation
        """
        return len(text) <= self.max_input_length

    def truncate_if_needed(self, text: str) -> str:
        """Truncate text if it exceeds input limits.

        Args:
            text: Input text

        Returns:
            str: Potentially truncated text
        """
        if len(text) <= self.max_input_length:
            return text

        # Truncate from the beginning, keeping the most recent content
        truncated = text[-self.max_input_length :]
        return f"[...truncated...]\n{truncated}"
