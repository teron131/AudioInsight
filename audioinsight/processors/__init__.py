"""
AudioInsight Processors Package

This package contains refactored audio processing components:
- AudioProcessor: Main coordinator class
- FFmpegProcessor: Handles FFmpeg process management
- TranscriptionProcessor: Speech-to-text processing
- DiarizationProcessor: Speaker diarization processing
- Formatter: Results formatting
- BaseProcessor: Base class with common utilities
"""

from .audio_processor import AudioProcessor
from .base_processor import SENTINEL, BaseProcessor, format_time, s2hk
from .diarization_processor import DiarizationProcessor
from .ffmpeg_processor import FFmpegProcessor
from .formatter import Formatter
from .transcription_processor import TranscriptionProcessor

__all__ = [
    "AudioProcessor",
    "BaseProcessor",
    "FFmpegProcessor",
    "TranscriptionProcessor",
    "DiarizationProcessor",
    "Formatter",
    "SENTINEL",
    "format_time",
    "s2hk",
]
