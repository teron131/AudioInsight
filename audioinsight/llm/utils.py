import os
from typing import Optional

import opencc
from dotenv import load_dotenv

load_dotenv()

# Cache OpenCC converter instance to avoid recreation
_s2hk_converter = None


def s2hk(text: str) -> str:
    """Convert Simplified Chinese to Traditional Chinese with cached converter."""
    if not text:
        return text

    global _s2hk_converter
    if _s2hk_converter is None:
        _s2hk_converter = opencc.OpenCC("s2hk")

    return _s2hk_converter.convert(text)


def contains_chinese(text: str) -> bool:
    """Check if text contains Chinese characters.

    Args:
        text: The text to check

    Returns:
        bool: True if text contains Chinese characters
    """
    for char in text:
        if "\u4e00" <= char <= "\u9fff":  # CJK Unified Ideographs
            return True
    return False


def get_api_credentials() -> tuple[Optional[str], Optional[str]]:
    """Get API key and base URL for LLM services.

    Returns:
        tuple: (api_key, base_url) - base_url is None for OpenAI direct
    """
    api_key = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
    base_url = "https://openrouter.ai/api/v1" if os.getenv("OPENROUTER_API_KEY") else None
    return api_key, base_url


def truncate_text(text: str, max_length: int = 1000) -> str:
    """Truncate text to a maximum length.

    Args:
        text: Text to truncate
        max_length: Maximum length of the text

    Returns:
        str: Truncated text with ellipsis if needed
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


class LRUCache:
    """Simple LRU (Least Recently Used) cache implementation."""

    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.cache = {}
        self.access_order = []

    def get(self, key) -> tuple[bool, any]:
        """Get item from cache.

        Returns:
            tuple: (found, value) where found is bool and value is the cached item or None
        """
        if key in self.cache:
            # Move to end (most recently used)
            self.access_order.remove(key)
            self.access_order.append(key)
            return True, self.cache[key]
        return False, None

    def put(self, key, value):
        """Put item in cache."""
        if key in self.cache:
            # Update existing item
            self.cache[key] = value
            self.access_order.remove(key)
            self.access_order.append(key)
        else:
            # Add new item
            if len(self.cache) >= self.max_size:
                # Remove least recently used
                oldest = self.access_order.pop(0)
                del self.cache[oldest]

            self.cache[key] = value
            self.access_order.append(key)

    def clear(self):
        """Clear all items from cache."""
        self.cache.clear()
        self.access_order.clear()

    def size(self) -> int:
        """Get current cache size."""
        return len(self.cache)

    def max_size_reached(self) -> bool:
        """Check if cache has reached maximum size."""
        return len(self.cache) >= self.max_size
