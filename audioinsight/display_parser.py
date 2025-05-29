#!/usr/bin/env python3
"""
Display Text Parser - Standalone parser for frontend display enhancement.

This module provides text parsing functionality that works INDEPENDENTLY from
the transcription pipeline, ensuring smooth data transfer while allowing
optional text refinement for display purposes.

IMPORTANT: This parser is completely separate from transcription processing.
It only processes text that is already transcribed for display enhancement.
The transcription pipeline is never affected or blocked by this parser.
"""

import asyncio
import time
from typing import Any, Dict, Optional

from .llm import Parser, ParserConfig
from .logging_config import get_logger

logger = get_logger(__name__)


class DisplayParser:
    """
    Display-layer text parser that works INDEPENDENTLY from transcription.

    This parser operates ONLY on text that's being displayed to users without
    interfering with the real-time transcription pipeline. The transcription
    process continues unaffected whether this parser is enabled or disabled.

    Key Design Principles:
    - Zero impact on transcription latency
    - Optional display enhancement only
    - Works on already-transcribed text
    - Graceful fallback to original text
    """

    def __init__(self, config: Optional[ParserConfig] = None):
        """Initialize the display text parser.

        Args:
            config: Optional configuration for the text parser
        """
        self.config = config or ParserConfig()
        self.parser = None  # Created on demand
        self.enabled = False
        self.parse_cache = {}  # Cache parsed results
        self.cache_max_size = 100

        # Performance tracking
        self.stats = {
            "texts_parsed": 0,
            "cache_hits": 0,
            "total_parse_time": 0.0,
            "average_parse_time": 0.0,
        }

    def enable(self, enabled: bool = True):
        """Enable or disable the display parser.

        Args:
            enabled: Whether to enable text parsing
        """
        self.enabled = enabled
        logger.info(f"Display text parser {'enabled' if enabled else 'disabled'}")

    def is_enabled(self) -> bool:
        """Check if the display parser is enabled.

        Returns:
            bool: True if parser is enabled
        """
        return self.enabled

    async def parse_for_display(self, text: str) -> str:
        """Parse text for display enhancement ONLY.

        IMPORTANT: This method is designed to be called from the frontend/display layer
        and will NEVER block the transcription pipeline. It processes already-transcribed
        text for better user experience, but transcription continues independently.

        Args:
            text: Raw transcribed text to be enhanced for display

        Returns:
            str: Enhanced text for display, or original if parsing is disabled/fails
        """
        if not self.enabled or not text or not text.strip():
            return text

        # Check cache first
        text_hash = hash(text)
        if text_hash in self.parse_cache:
            self.stats["cache_hits"] += 1
            return self.parse_cache[text_hash]

        # Skip parsing for very short text
        if len(text.strip()) < 20:
            return text

        try:
            # Create parser on demand
            if not self.parser:
                self.parser = Parser(config=self.config)
                logger.info("Display Parser created on demand")

            start_time = time.time()
            parsed_text = await self.parser.parse_text(text)
            parse_time = time.time() - start_time

            # Update statistics
            self.stats["texts_parsed"] += 1
            self.stats["total_parse_time"] += parse_time
            self.stats["average_parse_time"] = self.stats["total_parse_time"] / self.stats["texts_parsed"]

            # Cache result (with size limit)
            if len(self.parse_cache) >= self.cache_max_size:
                # Remove oldest entry
                oldest_key = next(iter(self.parse_cache))
                del self.parse_cache[oldest_key]

            self.parse_cache[text_hash] = parsed_text

            logger.debug(f"Display text parsed in {parse_time:.2f}s: {len(text)} -> {len(parsed_text)} chars")
            return parsed_text

        except Exception as e:
            logger.warning(f"Display text parsing failed: {e}, returning original")
            return text

    def get_stats(self) -> Dict[str, Any]:
        """Get parsing statistics.

        Returns:
            dict: Statistics about parsing performance
        """
        return self.stats.copy()

    def clear_cache(self):
        """Clear the parsing cache."""
        self.parse_cache.clear()
        logger.info("Display text parser cache cleared")


# Global instance for easy access
_display_parser = None


def get_display_parser() -> DisplayParser:
    """Get the global display parser instance.

    Returns:
        DisplayParser: The global display parser instance
    """
    global _display_parser
    if _display_parser is None:
        _display_parser = DisplayParser()
    return _display_parser


async def parse_text_for_display(text: str) -> str:
    """Convenience function to parse transcribed text for display enhancement ONLY.

    IMPORTANT: This function processes already-transcribed text for display purposes.
    It does NOT affect the transcription pipeline in any way.

    Args:
        text: Transcribed text to enhance for display

    Returns:
        str: Enhanced text for display
    """
    parser = get_display_parser()
    return await parser.parse_for_display(text)


def enable_display_parsing(enabled: bool = True):
    """Enable or disable display text parsing globally.

    IMPORTANT: This setting only affects display enhancement of already-transcribed text.
    The transcription pipeline is completely independent and unaffected.

    Args:
        enabled: Whether to enable display text enhancement
    """
    parser = get_display_parser()
    parser.enable(enabled)
