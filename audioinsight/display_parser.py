import time
from typing import Any, Dict, Optional

from .llm import LRUCache, Parser, ParserConfig
from .llm.parser import DisplayParserStats
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

    def __init__(self, config: Optional[ParserConfig] = None, cache_size: int = 100):
        """Initialize the display text parser.

        Args:
            config: Optional configuration for the text parser
            cache_size: Maximum number of items to cache
        """
        self.config = config or ParserConfig()
        self.parser = None  # Created on demand
        self.enabled = False

        # Use the new LRU cache utility
        self.parse_cache = LRUCache(max_size=cache_size)

        # Use the new standardized statistics
        self.stats = DisplayParserStats()

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
        cache_hit, cached_result = self.parse_cache.get(text_hash)
        if cache_hit:
            self.stats.record_cache_hit()
            return cached_result

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

            # Update statistics using the new class
            self.stats.record_parsing(parse_time)

            # Cache result using the new LRU cache
            self.parse_cache.put(text_hash, parsed_text)

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
        stats_dict = self.stats.to_dict()
        # Add cache-specific information
        stats_dict.update(
            {
                "cache_size": self.parse_cache.size(),
                "cache_max_size": self.parse_cache.max_size,
            }
        )
        return stats_dict

    def clear_cache(self):
        """Clear the parsing cache."""
        self.parse_cache.clear()
        logger.info("Display text parser cache cleared")

    @property
    def cache_max_size(self) -> int:
        """Get maximum cache size for backward compatibility."""
        return self.parse_cache.max_size


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
