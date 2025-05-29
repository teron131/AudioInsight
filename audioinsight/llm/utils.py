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


def truncate_text(text: str, max_length: int) -> str:
    """Truncate text to a maximum length, preserving as much content as possible.

    Args:
        text: Text to truncate
        max_length: Maximum allowed length

    Returns:
        str: Truncated text
    """
    if len(text) <= max_length:
        return text

    # Try to truncate at sentence boundaries
    truncated = text[:max_length]

    # Look for sentence endings near the truncation point
    for punct in ["。", ".", "！", "!", "？", "?", "\n"]:
        last_punct = truncated.rfind(punct)
        if last_punct > max_length * 0.8:  # Only if we don't lose too much content
            return truncated[: last_punct + 1]

    # If no good break point, just truncate
    return truncated
