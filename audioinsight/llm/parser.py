import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple

from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from ..logging_config import get_logger
from .llm_base import EventBasedProcessor, UniversalLLM, WorkItem
from .llm_config import LLMConfig, ParserConfig
from .llm_utils import contains_chinese, s2hk
from .performance_monitor import get_performance_monitor, log_performance_if_needed

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


# =============================================================================
# Parser Implementation
# =============================================================================


class Parser(EventBasedProcessor):
    """
    LLM-based text parser that fixes typos, punctuation, and creates smooth sentences.
    Returns structured ParsedTranscript objects for sharing between UI and analyzer.
    Uses the universal LLM client for consistent inference and EventBasedProcessor for queue management.

    This parser works on the entire committed transcript regardless of language.
    Duplication protection is removed since committed transcript already has underlying protection.
    s2hk conversion is NOT applied here - it will be applied only at the final step.
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
        # Remove duplication protection - committed transcript already handles this
        super().__init__(queue_maxsize=100, cooldown_seconds=0.5, max_concurrent_workers=4, enable_work_coordination=False)

        self.config = config or ParserConfig(model_id=model_id)
        self.api_key = api_key  # Store for lazy initialization

        # Lazy initialization - only create when first needed
        self._llm_client = None
        self._prompt = None

        # Statistics using the new standardized class - lightweight initialization
        self.stats = ParserStats()

        logger.info(f"Parser initialized - works on entire committed transcript, no duplication protection needed.")

        # Callback for storing parsed results
        self._result_callback = None

    def _is_stateful_processor(self) -> bool:
        """Parser is now stateless."""
        return False

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
- For Chinese text, use appropriate Chinese punctuation (、，。！？).
- Preserve the original meaning and context.

IMPORTANT: Always respond in the same language and script as the input text. For Chinese text, maintain Traditional or Simplified Chinese as provided in the input.""",
                    ),
                    ("human", "{text}"),
                ]
            )
            logger.debug("Lazy-initialized prompt template for parser")
        return self._prompt

    async def _process_item(self, item):
        """
        Process a single parsing request from the queue.
        This is now STATELESS and processes the full text provided.
        """
        start_time = time.time()
        try:
            # Handle WorkItem wrapper - extract the actual data
            if hasattr(item, "data"):
                text, speaker_info, timestamps = item.data
            else:
                # Fallback for direct tuple (backward compatibility)
                text, speaker_info, timestamps = item

            if not text or not text.strip():
                logger.debug("Skipping empty text for parsing.")
                return

            logger.info(f"Processing full text: {len(text)} chars")

            # Process the full text
            result = await self.parse_transcript(text, speaker_info, timestamps)

            # Use the callback to send the result back
            if self._result_callback:
                await self._result_callback(result)
                logger.debug(f"Sent parsed result via callback: {len(result.parsed_text)} chars")

            # Record statistics
            processing_time = time.time() - start_time
            self.stats.record_processing(processing_time, len(text))
            logger.info(f"Parser processed {len(text)} chars in {processing_time:.2f}s")

        except Exception as e:
            logger.error(f"Parser processing failed: {e}", exc_info=True)
            # Do not re-raise, as it would stop the worker.

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

            # CHINESE PARSER FIX: Enhanced logging for Chinese text
            is_chinese = contains_chinese(text)
            logger.debug(f"Parsing {len(text)} characters (Chinese: {is_chinese}): '{text[:50]}...'")

            # Process text (with chunking if needed)
            if self.config.needs_chunking():
                parsed_text, chunks_used = await self._process_in_chunks(text)
                self.stats.record_processing(time.time() - start_time, len(text), chunks_used)
            else:
                parsed_text = await self._process_chunk(text)
                self.stats.record_processing(time.time() - start_time, len(text), 1)

            # CHINESE PARSER FIX: Validate parsing result for Chinese text
            if is_chinese:
                if not parsed_text or not parsed_text.strip():
                    logger.warning(f"Chinese text parsing returned empty result, using original text")
                    parsed_text = text
                elif not contains_chinese(parsed_text):
                    logger.warning(f"Chinese text parsing lost Chinese characters, using original text")
                    parsed_text = text
                else:
                    logger.info(f"Chinese text parsed successfully: '{text[:30]}...' -> '{parsed_text[:30]}...'")

            # Create segments from the parsed content
            segments = self._create_segments(text, parsed_text, speaker_info, timestamps)

            parsing_time = time.time() - start_time

            result = ParsedTranscript(original_text=text, parsed_text=parsed_text, segments=segments, timestamps=timestamps or {}, speakers=speaker_info or [], parsing_time=parsing_time)

            logger.debug(f"Parsed transcript in {parsing_time:.2f}s: '{parsed_text[:50]}...'")
            return result

        except Exception as e:
            logger.error(f"Failed to parse transcript: {e}")
            # Return original text as fallback without s2hk conversion
            return ParsedTranscript(original_text=text, parsed_text=text, parsing_time=time.time() - start_time)

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

        # STATELESS REWORK: Remove stateful info from stats
        parser_stats.update(
            {
                "stateful_processor": False,
            }
        )

        return {**base_stats, **parser_stats}

    async def reset_state(self):
        """STATELESS REWORK: No state to reset."""
        logger.info("Parser is now stateless, no state to reset.")
        pass

    def set_result_callback(self, callback):
        """Set a callback function to be called when parsing results are available.

        Args:
            callback: Async function that takes a ParsedTranscript object
        """
        self._result_callback = callback
        logger.debug("Parser result callback set")


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
