import asyncio
import time
from typing import Dict, List, Optional, Tuple

from langchain.prompts import ChatPromptTemplate

from ..logging_config import get_logger
from .base import EventBasedProcessor, UniversalLLM
from .types import LLMConfig, ParsedTranscript, ParserConfig, ParserStats
from .utils import contains_chinese, s2hk

logger = get_logger(__name__)


class Parser(EventBasedProcessor):
    """
    LLM-based text parser that fixes typos, punctuation, and creates smooth sentences.
    Returns structured ParsedTranscript objects for sharing between UI and summarizer.
    Uses the universal LLM client for consistent inference and EventBasedProcessor for queue management.
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
        # Initialize base class with optimized queue for parser and concurrent workers
        super().__init__(queue_maxsize=50, cooldown_seconds=0.1, max_concurrent_workers=3)  # Increased queue size, reduced cooldown, added concurrency

        self.config = config or ParserConfig(model_id=model_id)

        # Create LLM config from text parser config
        llm_config = LLMConfig(
            model_id=self.config.model_id,
            api_key=api_key,
        )

        self.llm_client = UniversalLLM(llm_config)

        # Create prompt template for text parsing
        self.prompt = ChatPromptTemplate.from_messages(
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

        # Statistics using the new standardized class
        self.stats = ParserStats()

    async def _process_item(self, item: Tuple[str, Optional[List[Dict]], Optional[Dict[str, float]]]):
        """Process a single parsing request from the queue.

        Args:
            item: Tuple of (text, speaker_info, timestamps)
        """
        text, speaker_info, timestamps = item
        result = await self.parse_transcript(text, speaker_info, timestamps)
        self.update_processing_time()
        return result

    async def queue_parsing_request(self, text: str, speaker_info: Optional[List[Dict]] = None, timestamps: Optional[Dict[str, float]] = None) -> bool:
        """Queue a parsing request for event-based processing.

        Args:
            text: The transcript text to parse
            speaker_info: Optional speaker information
            timestamps: Optional timestamp information

        Returns:
            bool: True if successfully queued, False otherwise
        """
        if not self.should_process(text, min_size=50):  # Lower threshold for parser
            logger.debug(f"⏳ Parser: Accumulating text for batching: {len(text)} chars")
            return False

        return await self.queue_for_processing((text, speaker_info, timestamps))

    async def parse_transcript(self, text: str, speaker_info: Optional[List[Dict]] = None, timestamps: Optional[Dict[str, float]] = None) -> ParsedTranscript:
        """Parse and structure a transcript with full metadata.

        Args:
            text: The transcript text to parse and correct
            speaker_info: Optional speaker information
            timestamps: Optional timestamp information

        Returns:
            ParsedTranscript: Structured transcript data
        """
        if not text.strip():
            return ParsedTranscript(original_text=text, parsed_text=text, segments=[], timestamps=timestamps or {}, speakers=speaker_info or [], parsing_time=0.0)

        start_time = time.time()
        chunks_used = 0

        try:
            # Parse the text - check if it needs chunking based on output token limits
            if self.config.needs_chunking(text):
                # Process in chunks for longer text
                corrected_text, chunks_used = await self._process_in_chunks(text)
            else:
                corrected_text = await self._process_chunk(text)
                chunks_used = 1

            # Apply s2hk conversion to ensure Traditional Chinese output if input was Chinese
            if contains_chinese(corrected_text):
                corrected_text = s2hk(corrected_text)

            # Create segments from the parsed text
            segments = self._create_segments(text, corrected_text, speaker_info, timestamps)

            # Update statistics using the new class
            processing_time = time.time() - start_time
            self.stats.record_processing(processing_time, len(text), chunks_used)

            parsed_transcript = ParsedTranscript(
                original_text=text,
                parsed_text=corrected_text,
                segments=segments,
                timestamps=timestamps or {},
                speakers=speaker_info or [],
                parsing_time=processing_time,
            )

            logger.info(f"Parsed transcript in {processing_time:.2f}s: {len(text)} -> {len(corrected_text)} chars")
            return parsed_transcript

        except Exception as e:
            logger.error(f"Failed to parse transcript: {e}")
            # Return original text wrapped in ParsedTranscript structure
            return ParsedTranscript(original_text=text, parsed_text=text, segments=[{"text": text, "speaker": None, "timestamp": None}], timestamps=timestamps or {}, speakers=speaker_info or [], parsing_time=time.time() - start_time)

    async def parse_text(self, text: str) -> str:
        """Parse and correct a text string (legacy method for backward compatibility).

        Args:
            text: The text to parse and correct

        Returns:
            str: The corrected text
        """
        parsed_transcript = await self.parse_transcript(text)
        return parsed_transcript.parsed_text

    def _create_segments(self, original_text: str, corrected_text: str, speaker_info: Optional[List[Dict]], timestamps: Optional[Dict[str, float]]) -> List[Dict]:
        """Create segments from parsed text with metadata.

        Args:
            original_text: Original transcript text
            corrected_text: Corrected text
            speaker_info: Speaker information
            timestamps: Timestamp information

        Returns:
            List of segments with metadata
        """
        segments = []

        # For now, create a simple segment structure
        # In the future, this could be enhanced to split by sentences, speakers, etc.

        # Split by sentences or meaningful breaks
        import re

        sentence_endings = re.split(r"[.!?。！？]\s*", corrected_text)
        sentence_endings = [s.strip() for s in sentence_endings if s.strip()]

        current_position = 0
        for i, sentence in enumerate(sentence_endings):
            if not sentence:
                continue

            segment = {
                "text": sentence,
                "position": i,
                "character_start": current_position,
                "character_end": current_position + len(sentence),
                "speaker": None,
                "timestamp_start": None,
                "timestamp_end": None,
            }

            # Try to match with speaker info if available
            if speaker_info and i < len(speaker_info):
                segment["speaker"] = speaker_info[i].get("speaker")

            # Add timestamp info if available
            if timestamps:
                segment["timestamp_start"] = timestamps.get(f"segment_{i}_start")
                segment["timestamp_end"] = timestamps.get(f"segment_{i}_end")

            segments.append(segment)
            current_position += len(sentence) + 1  # +1 for the punctuation

        return segments

    async def _process_chunk(self, text: str) -> str:
        """Process a single chunk of text.

        Args:
            text: The text chunk to process

        Returns:
            str: The corrected text directly
        """
        result = await self.llm_client.invoke_text(self.prompt, {"text": text})
        return result

    async def _process_in_chunks(self, text: str) -> tuple[str, int]:
        """Process long text in chunks based on output token limits.

        Args:
            text: The full text to process

        Returns:
            tuple: (corrected_text, chunks_used)
        """
        # Check if chunking is needed based on output token limits
        if not self.config.needs_chunking(text):
            # Text is small enough to process in one go
            corrected_text = await self._process_chunk(text)
            return corrected_text, 1

        # Need to chunk based on output token limits
        logger.debug(f"Text requires chunking for output token limits...")

        chunks = self._split_by_output_tokens(text)
        corrected_chunks = []

        for i, chunk in enumerate(chunks):
            logger.debug(f"Processing chunk {i+1}/{len(chunks)} ({len(chunk)} chars)")
            corrected_chunk = await self._process_chunk(chunk)
            corrected_chunks.append(corrected_chunk)

        # Merge chunks back together
        corrected_text = " ".join(corrected_chunks)
        return corrected_text, len(chunks)

    def _split_by_output_tokens(self, text: str) -> list[str]:
        """Split text into chunks based on output token limits.

        Args:
            text: The text to split

        Returns:
            list[str]: List of text chunks
        """
        chunks = []
        chunk_size_chars = self.config.get_chunk_size_chars()
        overlap_chars = int(chunk_size_chars * 0.1)  # 10% overlap for context

        start = 0
        while start < len(text):
            end = start + chunk_size_chars

            # If this isn't the last chunk, try to break at a sentence boundary
            if end < len(text):
                # Look for sentence breaks in the overlap region
                overlap_start = max(start, end - overlap_chars)
                sentence_breaks = []

                # Find sentence-ending punctuation
                for i in range(overlap_start, min(end + overlap_chars, len(text))):
                    if text[i] in ".!?。！？；：":
                        sentence_breaks.append(i + 1)

                # Use the best sentence break, otherwise use the original boundary
                if sentence_breaks:
                    # Choose the sentence break closest to our target end
                    best_break = min(sentence_breaks, key=lambda x: abs(x - end))
                    end = best_break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end

        return chunks

    def get_stats(self) -> dict:
        """Get processing statistics.

        Returns:
            dict: Dictionary of processing statistics
        """
        base_stats = self.get_queue_status()
        parser_stats = self.stats.to_dict()
        return {**base_stats, **parser_stats}


# Convenience function for quick text parsing (returns structured data)
async def parse_transcript(
    text: str,
    model_id: str = "openai/gpt-4.1-nano",
    api_key: Optional[str] = None,
    speaker_info: Optional[List[Dict]] = None,
    timestamps: Optional[Dict[str, float]] = None,
) -> ParsedTranscript:
    """Convenience function to quickly parse and structure transcript text.

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
