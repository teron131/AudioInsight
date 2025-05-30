"""
AudioInsight LLM Module

This module provides a universal LLM infrastructure for all AI operations in AudioInsight.
It includes:
- Universal LLM client for consistent inference
- Text parsing and correction
- Transcription summarization
- Shared utilities and types
"""

from .base import UniversalLLM
from .parser import Parser, parse_transcript
from .summarizer import LLMSummarizer
from .types import (
    BaseStats,
    DisplayParserStats,
    LLMConfig,
    LLMResponse,
    LLMStats,
    LLMTrigger,
    ParserConfig,
    ParserStats,
)
from .utils import LRUCache, contains_chinese, get_api_credentials, s2hk, truncate_text

__all__ = [
    # Core classes
    "UniversalLLM",
    "LLMSummarizer",
    "Parser",
    # Configuration and types
    "LLMConfig",
    "LLMResponse",
    "LLMStats",
    "LLMTrigger",
    "ParserConfig",
    # Statistics classes
    "BaseStats",
    "ParserStats",
    "DisplayParserStats",
    # Utilities
    "contains_chinese",
    "get_api_credentials",
    "s2hk",
    "truncate_text",
    "LRUCache",
    # Convenience functions
    "parse_transcript",
]
