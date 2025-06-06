import re
from datetime import timedelta
from time import time

import opencc

from ..logging_config import get_logger

# Initialize logging using centralized configuration
logger = get_logger(__name__)

SENTINEL = object()  # unique sentinel object for end of stream marker

# Cache OpenCC converter instance to avoid recreation
_s2hk_converter = None

# Pre-compile regex for sentence splitting
_sentence_split_regex = re.compile(r"[.!?]+")

# Cache timedelta formatting for common values
_cached_timedeltas = {}


def format_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS with caching for performance."""
    if seconds == 0:
        return "0:00:00"

    int_seconds = int(seconds)
    if int_seconds in _cached_timedeltas:
        return _cached_timedeltas[int_seconds]

    result = str(timedelta(seconds=int_seconds))

    # Cache up to 3600 entries (1 hour worth of seconds)
    if len(_cached_timedeltas) < 3600:
        _cached_timedeltas[int_seconds] = result

    return result


def s2hk(text: str) -> str:
    """Convert Simplified Chinese to Traditional Chinese with cached converter."""
    if not text:
        return text

    global _s2hk_converter
    if _s2hk_converter is None:
        _s2hk_converter = opencc.OpenCC("s2hk")

    return _s2hk_converter.convert(text)


class BaseProcessor:
    """Base class for all audio processors."""

    def __init__(self, args):
        self.args = args
        self.logger = get_logger(self.__class__.__name__)

    def cleanup(self):
        """Clean up processor resources. Override in subclasses."""
        pass
