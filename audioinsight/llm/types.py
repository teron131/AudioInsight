from dataclasses import dataclass
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


@dataclass
class LLMConfig:
    """Base configuration for LLM operations."""

    model_id: str = "openai/gpt-4.1-mini"
    api_key: Optional[str] = None
    temperature: float = 0.0
    max_tokens: Optional[int] = None
    timeout: float = 30.0


@dataclass
class LLMTrigger:
    """Configuration for when to trigger LLM inference."""

    idle_time_seconds: float = 5.0
    max_text_length: int = 100000
    conversation_trigger_count: int = 2  # Trigger after this many conversations (speaker turns)
    min_text_length: int = 100  # OPTIMIZATION: Minimum text length before considering triggers


@dataclass
class ParserConfig:
    """Configuration for the text parser."""

    model_id: str = "google/gemini-flash-1.5-8b"  # Default to a faster model for text parsing
    chunk_size: int = 2000  # Process text in chunks of this size
    overlap_size: int = 200  # Overlap between chunks to maintain context


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
        self.inference_by_idle = 0
        self.inference_by_conversation_count = 0
        self.inference_by_text_length = 0
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
        if trigger_reason == "idle":
            self.inference_by_idle += 1
        elif trigger_reason == "conversation_count":
            self.inference_by_conversation_count += 1
        elif trigger_reason == "text_length":
            self.inference_by_text_length += 1
        elif trigger_reason == "forced":
            self.inference_by_forced += 1

    def to_dict(self) -> Dict[str, Any]:
        """Get statistics as a dictionary."""
        return {
            "inference_generated": self.inference_generated,
            "total_text_processed": self.total_text_processed,
            "average_inference_time": self.average_inference_time,
            "inference_by_idle": self.inference_by_idle,
            "inference_by_conversation_count": self.inference_by_conversation_count,
            "inference_by_text_length": self.inference_by_text_length,
            "inference_by_forced": self.inference_by_forced,
        }
