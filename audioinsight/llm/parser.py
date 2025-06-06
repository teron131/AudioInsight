import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple

from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from ..logging_config import get_logger
from .base import EventBasedProcessor, UniversalLLM, WorkItem
from .config import LLMConfig, ParserConfig
from .performance_monitor import get_performance_monitor, log_performance_if_needed
from .utils import contains_chinese, s2hk

logger = get_logger(__name__)


# =============================================================================
# Type Definitions for Parsing
# =============================================================================


class ParsedTranscript(BaseModel):
    """Structured response from transcript parsing."""

    original_text: str = Field(description="Original transcribed text")
    parsed_text: str = Field(description="Parsed and corrected text")
    segments: List[Dict[str, Any]] = Field(default_factory=list, description="Text segments with metadata")
    timestamps: Dict[str, float] = Field(default_factory=dict, description="Important timestamps")
    speakers: List[Dict[str, Any]] = Field(default_factory=list, description="Speaker information")
    parsing_time: float = Field(default=0.0, description="Time taken to parse the transcript")


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


# =============================================================================
# Parser Implementation
# =============================================================================


class Parser(EventBasedProcessor):
    """
    LLM-based text parser that fixes typos, punctuation, and creates smooth sentences.
    Returns structured ParsedTranscript objects for sharing between UI and analyzer.
    Uses the universal LLM client for consistent inference and EventBasedProcessor for queue management.

    This is a STATEFUL processor that manages incremental parsing state, so it uses single-worker
    mode to prevent race conditions and duplicate processing.
    """

    def __init__(
        self,
        model_id: str = "openai/gpt-4.1-nano",
        api_key: Optional[str] = None,
        config: Optional[ParserConfig] = None,
    ):
        """Initialize the text parser.

        Args:
            model_id: The model ID to use (defaults to openai/gpt-4.1-nano for faster processing)
            api_key: Optional API key override (defaults to OPENROUTER_API_KEY env var)
            config: Configuration for the text parser
        """
        # Initialize base class with coordination enabled for stateful operations
        # Single worker to avoid race conditions on incremental parsing state
        super().__init__(queue_maxsize=50, cooldown_seconds=0.5, max_concurrent_workers=1, enable_work_coordination=True)  # Smaller queue since single worker  # Conservative start, will adapt to actual processing times  # Force single worker for stateful operations  # Enable coordination for deduplication

        self.config = config or ParserConfig(model_id=model_id)
        self.api_key = api_key  # Store for lazy initialization

        # Lazy initialization - only create when first needed
        self._llm_client = None
        self._prompt = None

        # Statistics using the new standardized class - lightweight initialization
        self.stats = ParserStats()

        # Stateful incremental parsing management - CRITICAL: Atomic state management
        self._state_lock = asyncio.Lock()  # Protect shared state
        self._last_processed_text = ""  # Track what was last processed to avoid duplicates
        self._processing_in_progress = False  # Prevent concurrent processing

        logger.info(f"Parser initialized in stateful mode with work coordination and single worker")

    def _is_stateful_processor(self) -> bool:
        """Mark this processor as stateful to enable single-worker coordination."""
        return True

    @property
    def llm_client(self) -> UniversalLLM:
        """Lazy initialization of LLM client to speed up startup."""
        if self._llm_client is None:
            llm_config = LLMConfig(model_id=self.config.model_id, api_key=self.api_key, timeout=12.0, temperature=0.0)
            self._llm_client = UniversalLLM(llm_config)
            logger.debug(f"Lazy-initialized LLM client for model: {self.config.model_id} with temperature=0.0")
        return self._llm_client

    @property
    def prompt(self) -> ChatPromptTemplate:
        """Lazy initialization of prompt template to speed up startup."""
        if self._prompt is None:
            self._prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """Refine a sequence of piecemeal subtitle derived from transcription.
- Make minimal contextual changes.
- Only fix typos if you are highly confident.
- Add punctuation appropriately.

IMPORTANT: Always respond in the same language and script as the input text.""",
                    ),
                    ("human", "{text}"),
                ]
            )
            logger.debug("Lazy-initialized prompt template for parser")
        return self._prompt

    async def _process_item(self, item: Tuple[str, Optional[List[Dict]], Optional[Dict[str, float]]]):
        """Process a single parsing request from the queue with atomic state management.

        Args:
            item: Tuple of (text, speaker_info, timestamps)
        """
        start_time = time.time()
        monitor = get_performance_monitor()

        try:
            text, speaker_info, timestamps = item

            # Atomic check for incremental processing
            async with self._state_lock:
                # Check if this text was already processed
                if text == self._last_processed_text:
                    logger.debug(f"Skipping already processed text: {text[:50]}...")
                    return

                # Check if we're already processing something
                if self._processing_in_progress:
                    logger.debug(f"Processing already in progress, queuing: {text[:50]}...")
                    return

                # Mark as processing and update state atomically
                self._processing_in_progress = True
                current_last_processed = self._last_processed_text

            try:
                # Get only the incremental text to process
                incremental_text = self._get_incremental_text(text, current_last_processed)

                if not incremental_text or not incremental_text.strip():
                    logger.debug("No new incremental text to process")
                    return

                logger.info(f"Processing incremental text: {len(incremental_text)} chars (from {len(text)} total)")

                # Process the incremental text
                result = await self.parse_transcript(incremental_text, speaker_info, timestamps)

                # Atomically update the processed text state
                async with self._state_lock:
                    self._last_processed_text = text
                    logger.debug(f"Updated last processed text to: {len(text)} chars")

                # Record successful processing
                processing_time = time.time() - start_time

                # Record statistics
                self.stats.record_processing(processing_time, len(incremental_text))

                logger.info(f"Parser processed {len(incremental_text)} chars in {processing_time:.2f}s")

            finally:
                # Always clear the processing flag
                async with self._state_lock:
                    self._processing_in_progress = False

        except Exception as e:
            logger.error(f"Parser processing failed: {e}")
            # Clear processing flag on error
            async with self._state_lock:
                self._processing_in_progress = False
            raise

    def _get_incremental_text(self, current_text: str, last_processed_text: str) -> str:
        """Get only the new incremental text that needs processing.

        Args:
            current_text: The full current accumulated text
            last_processed_text: The text that was last processed

        Returns:
            str: Only the new text that needs to be parsed
        """
        if not last_processed_text:
            # First time processing - return all text if reasonable size
            if len(current_text) < 500:
                logger.debug(f"First time processing: returning all {len(current_text)} chars")
                return current_text
            else:
                # For very long text, just return the last portion
                incremental = current_text[-300:]  # Last 300 chars
                logger.debug(f"First time processing long text: returning last 300 chars from {len(current_text)} total")
                return incremental

        # Find the new text since last processing
        if last_processed_text in current_text:
            # Find the last occurrence of the processed text
            last_index = current_text.rfind(last_processed_text)
            new_start = last_index + len(last_processed_text)
            incremental_text = current_text[new_start:].strip()

            if incremental_text:
                logger.debug(f"Incremental text: {len(incremental_text)} new chars from position {new_start}")
                return incremental_text
            else:
                logger.debug("No new text found since last processing")
                return ""
        else:
            # Text doesn't contain last processed - might be a reset or different content
            # In this case, process the last reasonable portion
            if len(current_text) < 500:
                logger.debug(f"Text changed, processing all {len(current_text)} chars")
                return current_text
            else:
                incremental = current_text[-400:]  # Last 400 chars for changed content
                logger.debug(f"Text changed, processing last 400 chars from {len(current_text)} total")
                return incremental

    async def queue_parsing_request(self, text: str, speaker_info: Optional[List[Dict]] = None, timestamps: Optional[Dict[str, float]] = None) -> bool:
        """Queue a parsing request with proper work item wrapping for deduplication.

        Args:
            text: Text to parse
            speaker_info: Optional speaker information
            timestamps: Optional timestamp information

        Returns:
            bool: True if successfully queued, False otherwise
        """
        if not text.strip():
            return False

        # Create work item with content-based ID for deduplication
        work_item = WorkItem((text, speaker_info, timestamps))
        return await self.queue_for_processing(work_item)

    async def parse_transcript(self, text: str, speaker_info: Optional[List[Dict]] = None, timestamps: Optional[Dict[str, float]] = None) -> ParsedTranscript:
        """Parse and correct a transcript text with incremental processing support.

        Args:
            text: Original transcript text to parse
            speaker_info: Optional speaker information for context
            timestamps: Optional timestamp information

        Returns:
            ParsedTranscript: Structured parsing result
        """
        start_time = time.time()

        try:
            # Quick validation
            if not text.strip():
                return ParsedTranscript(original_text=text, parsed_text="", parsing_time=time.time() - start_time)

            logger.debug(f"Parsing {len(text)} characters: '{text[:50]}...'")

            # Process text (with chunking if needed)
            if self.config.needs_chunking():
                parsed_text, chunks_used = await self._process_in_chunks(text)
                self.stats.record_processing(time.time() - start_time, len(text), chunks_used)
            else:
                parsed_text = await self._process_chunk(text)
                self.stats.record_processing(time.time() - start_time, len(text), 1)

            # Apply s2hk conversion if the text contains Chinese
            if contains_chinese(parsed_text):
                parsed_text = s2hk(parsed_text)

            # Create segments from the parsed content
            segments = self._create_segments(text, parsed_text, speaker_info, timestamps)

            parsing_time = time.time() - start_time

            result = ParsedTranscript(original_text=text, parsed_text=parsed_text, segments=segments, timestamps=timestamps or {}, speakers=speaker_info or [], parsing_time=parsing_time)

            logger.debug(f"Parsed transcript in {parsing_time:.2f}s: '{parsed_text[:50]}...'")
            return result

        except Exception as e:
            logger.error(f"Failed to parse transcript: {e}")
            # Return original text as fallback
            return ParsedTranscript(original_text=text, parsed_text=text, parsing_time=time.time() - start_time)  # Fallback to original

    async def parse_text(self, text: str) -> str:
        """Parse text and return only the corrected text string.

        Args:
            text: Text to parse

        Returns:
            str: Corrected text
        """
        result = await self.parse_transcript(text)
        return result.parsed_text

    def _create_segments(self, original_text: str, corrected_text: str, speaker_info: Optional[List[Dict]], timestamps: Optional[Dict[str, float]]) -> List[Dict]:
        """Create text segments with metadata.

        Args:
            original_text: Original transcript text
            corrected_text: Corrected text from LLM
            speaker_info: Speaker information
            timestamps: Timestamp information

        Returns:
            List of segments with metadata
        """
        segments = []

        # Simple segmentation by sentences for now
        sentences = corrected_text.split(". ")
        start_pos = 0

        for i, sentence in enumerate(sentences):
            if not sentence.strip():
                continue

            segment = {
                "text": sentence.strip(),
                "start_char": start_pos,
                "end_char": start_pos + len(sentence),
                "segment_id": i,
                "confidence": 1.0,  # Placeholder
            }

            # Add speaker info if available
            if speaker_info and len(speaker_info) > i:
                segment["speaker"] = speaker_info[i]

            # Add timing info if available
            if timestamps:
                segment["timestamps"] = timestamps

            segments.append(segment)
            start_pos += len(sentence) + 2  # +2 for '. '

        return segments

    async def _process_chunk(self, text: str) -> str:
        """Process a single chunk of text through the LLM.

        Args:
            text: Text chunk to process

        Returns:
            str: Processed text
        """
        try:
            result = await self.llm_client.invoke_text(self.prompt, {"text": text})
            return result
        except Exception as e:
            logger.warning(f"LLM processing failed: {e}")
            return text  # Fallback to original

    async def _process_in_chunks(self, text: str) -> tuple[str, int]:
        """Process long text by splitting into chunks.

        Args:
            text: Long text to process

        Returns:
            tuple: (processed_text, chunks_used)
        """
        chunks = self._split_by_output_tokens(text)
        processed_chunks = []

        for chunk in chunks:
            processed_chunk = await self._process_chunk(chunk)
            processed_chunks.append(processed_chunk)

        return " ".join(processed_chunks), len(chunks)

    def _split_by_output_tokens(self, text: str) -> list[str]:
        """Split text into chunks based on estimated output token limits.

        Args:
            text: Text to split

        Returns:
            list: Text chunks
        """
        # Use config method to get chunk size
        max_chars = self.config.get_chunk_size_chars()
        chunks = []

        start = 0
        while start < len(text):
            end = start + max_chars
            if end >= len(text):
                chunks.append(text[start:])
                break

            # Try to break at a sentence boundary
            chunk = text[start:end]
            last_period = chunk.rfind(". ")
            if last_period > max_chars * 0.7:  # At least 70% of chunk size
                end = start + last_period + 2
                chunks.append(text[start:end])
                start = end
            else:
                chunks.append(chunk)
            start = end

        return chunks

    def get_stats(self) -> dict:
        """Get processing statistics with coordination info.

        Returns:
            dict: Statistics including base stats and coordination details
        """
        base_stats = super().get_queue_status()
        parser_stats = self.stats.to_dict()

        # Add parser-specific state info
        parser_stats.update(
            {
                "stateful_processor": True,
                "last_processed_length": len(getattr(self, "_last_processed_text", "")),
                "processing_in_progress": getattr(self, "_processing_in_progress", False),
            }
        )

        return {**base_stats, **parser_stats}

    async def reset_state(self):
        """Reset the incremental parsing state for fresh sessions."""
        async with self._state_lock:
            self._last_processed_text = ""
            self._processing_in_progress = False
            logger.info("Parser state reset for fresh session")


# Convenience function for quick text parsing (returns structured data)
async def parse_transcript(
    text: str,
    model_id: str = "openai/gpt-4.1-nano",
    api_key: Optional[str] = None,
    speaker_info: Optional[List[Dict]] = None,
    timestamps: Optional[Dict[str, float]] = None,
) -> ParsedTranscript:
    """Convenience function to quickly parse and structure transcript text.

    Note: Always uses temperature=0.0 for consistent parsing results.

    Args:
        text: The text to parse and correct
        model_id: The model ID to use
        api_key: Optional API key override
        speaker_info: Optional speaker information
        timestamps: Optional timestamp information

    Returns:
        ParsedTranscript: Structured transcript data
    """
    parser = Parser(model_id=model_id, api_key=api_key)
    return await parser.parse_transcript(text, speaker_info, timestamps)
