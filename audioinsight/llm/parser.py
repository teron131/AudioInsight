import hashlib
import re
import time
from dataclasses import dataclass
from math import ceil
from typing import Any, Dict, List, Optional, Set

from langchain.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

from ..logging_config import get_logger
from .llm_base import EventBasedProcessor, UniversalLLM, WorkItem
from .llm_config import LLMConfig, ParserConfig
from .llm_utils import contains_chinese

logger = get_logger(__name__)


# =============================================================================
# Sentence-Level Caching System
# =============================================================================


@dataclass
class SentenceInfo:
    """Information about a sentence for caching and processing with similarity matching."""

    text: str
    index: int
    normalized_text: str = ""
    hash: str = ""
    processed: bool = False
    processed_text: Optional[str] = None
    timestamp: float = 0.0
    similarity_threshold: float = 0.85

    def __post_init__(self):
        if not self.normalized_text:
            self.normalized_text = self._normalize_text(self.text)
        if not self.hash:
            self.hash = hashlib.md5(self.normalized_text.encode()).hexdigest()
        if not self.timestamp:
            self.timestamp = time.time()

    def _normalize_text(self, text: str) -> str:
        """Normalize text for similarity comparison."""
        # Remove extra whitespace and convert to lowercase
        normalized = re.sub(r"\s+", " ", text.strip().lower())
        # Remove common punctuation for comparison
        normalized = re.sub(r"[.,!?;:，。！？；：、]", "", normalized)
        return normalized

    def calculate_similarity(self, other_text: str) -> float:
        """Calculate similarity with another text using Levenshtein distance."""
        other_normalized = self._normalize_text(other_text)

        # Quick exact match check
        if self.normalized_text == other_normalized:
            return 1.0

        # Calculate Levenshtein distance
        distance = self._levenshtein_distance(self.normalized_text, other_normalized)
        max_len = max(len(self.normalized_text), len(other_normalized))

        if max_len == 0:
            return 1.0

        similarity = 1.0 - (distance / max_len)
        return similarity

    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row

        return previous_row[-1]

    def is_similar_to(self, other_text: str, threshold: float = None) -> bool:
        """Check if this sentence is similar to another text."""
        threshold = threshold or self.similarity_threshold
        return self.calculate_similarity(other_text) >= threshold


class SentenceCache:
    """Cache for sentence processing with similarity-based matching to prevent near-duplication."""

    def __init__(self, max_size: int = 1000, similarity_threshold: float = 0.85):
        self.max_size = max_size
        self.similarity_threshold = similarity_threshold
        self.sentences: Dict[str, SentenceInfo] = {}
        self.processed_hashes: Set[str] = set()
        self.insertion_order: List[str] = []

        # Performance optimization: maintain a list of processed sentences for similarity search
        self.processed_sentences: List[SentenceInfo] = []

    def add_sentence(self, text: str, index: int) -> SentenceInfo:
        """Add a sentence to the cache, checking for similar existing sentences."""
        # First check for exact match using normalized text
        normalized_text = self._normalize_text(text)
        exact_hash = hashlib.md5(normalized_text.encode()).hexdigest()

        if exact_hash in self.sentences:
            return self.sentences[exact_hash]

        # Check for similar sentences in processed cache
        similar_sentence = self._find_similar_sentence(text)
        if similar_sentence:
            # Return the similar sentence info but update its text to current
            similar_sentence.text = text  # Update to current text variant
            return similar_sentence

        # Create new sentence info
        sentence_info = SentenceInfo(text=text, normalized_text=normalized_text, hash=exact_hash, index=index, similarity_threshold=self.similarity_threshold)

        self.sentences[exact_hash] = sentence_info
        self.insertion_order.append(exact_hash)

        # Maintain cache size
        if len(self.sentences) > self.max_size:
            oldest_hash = self.insertion_order.pop(0)
            if oldest_hash in self.sentences:
                # Remove from processed sentences list too
                old_sentence = self.sentences[oldest_hash]
                if old_sentence in self.processed_sentences:
                    self.processed_sentences.remove(old_sentence)

                del self.sentences[oldest_hash]
                self.processed_hashes.discard(oldest_hash)

        return sentence_info

    def _normalize_text(self, text: str) -> str:
        """Normalize text for similarity comparison."""
        # Remove extra whitespace and convert to lowercase
        normalized = re.sub(r"\s+", " ", text.strip().lower())
        # Remove common punctuation for comparison
        normalized = re.sub(r"[.,!?;:，。！？；：、]", "", normalized)
        return normalized

    def _find_similar_sentence(self, text: str) -> Optional[SentenceInfo]:
        """Find a similar processed sentence in the cache."""
        # Only search through processed sentences for efficiency
        for sentence_info in self.processed_sentences:
            if sentence_info.is_similar_to(text, self.similarity_threshold):
                return sentence_info
        return None

    def mark_processed(self, sentence_hash: str, processed_text: str):
        """Mark a sentence as processed with its result."""
        if sentence_hash in self.sentences:
            sentence_info = self.sentences[sentence_hash]
            sentence_info.processed = True
            sentence_info.processed_text = processed_text
            self.processed_hashes.add(sentence_hash)

            # Add to processed sentences list for similarity search
            if sentence_info not in self.processed_sentences:
                self.processed_sentences.append(sentence_info)

    def get_processed_text(self, sentence_hash: str) -> Optional[str]:
        """Get processed text for a sentence if available."""
        if sentence_hash in self.sentences and self.sentences[sentence_hash].processed:
            return self.sentences[sentence_hash].processed_text
        return None

    def is_processed(self, sentence_hash: str) -> bool:
        """Check if a sentence has been processed."""
        return sentence_hash in self.processed_hashes

    def find_similar_processed(self, text: str) -> Optional[str]:
        """Find processed text for a similar sentence."""
        similar_sentence = self._find_similar_sentence(text)
        if similar_sentence and similar_sentence.processed:
            return similar_sentence.processed_text
        return None

    def clear_old_entries(self, max_age_seconds: float = 3600):
        """Clear entries older than specified age."""
        current_time = time.time()
        to_remove = []

        for sentence_hash, sentence_info in self.sentences.items():
            if current_time - sentence_info.timestamp > max_age_seconds:
                to_remove.append(sentence_hash)

        for sentence_hash in to_remove:
            sentence_info = self.sentences[sentence_hash]

            # Remove from processed sentences list
            if sentence_info in self.processed_sentences:
                self.processed_sentences.remove(sentence_info)

            del self.sentences[sentence_hash]
            self.processed_hashes.discard(sentence_hash)
            if sentence_hash in self.insertion_order:
                self.insertion_order.remove(sentence_hash)

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get detailed cache statistics."""
        return {
            "total_sentences": len(self.sentences),
            "processed_sentences": len(self.processed_sentences),
            "similarity_threshold": self.similarity_threshold,
            "max_size": self.max_size,
            "cache_utilization": len(self.sentences) / self.max_size if self.max_size > 0 else 0,
        }


# =============================================================================
# Sentence Splitter with Multi-Language Support
# =============================================================================


class SentenceSplitter:
    """Advanced sentence splitter with support for English and Chinese punctuation."""

    def __init__(self):
        # Enhanced separators for better sentence boundary detection
        self.text_splitter = RecursiveCharacterTextSplitter(
            separators=["。", "！", "？", ".", "!", "?"],
            chunk_size=1000,  # Large chunk size for sentence-level splitting
            chunk_overlap=0,
            length_function=len,
            is_separator_regex=False,
        )

        # Regex patterns for sentence boundary detection
        self.sentence_patterns = [r"[.!?。！？]+\s*"]

        self.compiled_patterns = [re.compile(pattern) for pattern in self.sentence_patterns]

    def split_sentences(self, text: str) -> List[str]:
        """Split text into sentences with advanced boundary detection."""
        if not text.strip():
            return []

        # First, use LangChain's splitter for initial chunking
        initial_chunks = self.text_splitter.split_text(text)

        sentences = []
        for chunk in initial_chunks:
            # Further split each chunk using regex patterns
            chunk_sentences = self._split_chunk_by_patterns(chunk)
            sentences.extend(chunk_sentences)

        # Clean up and filter sentences
        sentences = [s.strip() for s in sentences if s.strip()]

        # Merge very short sentences with previous ones
        merged_sentences = self._merge_short_sentences(sentences)

        return merged_sentences

    def _split_chunk_by_patterns(self, chunk: str) -> List[str]:
        """Split a chunk using regex patterns for sentence boundaries."""
        sentences = [chunk]

        for pattern in self.compiled_patterns:
            new_sentences = []
            for sentence in sentences:
                if not sentence.strip():
                    continue
                parts = pattern.split(sentence)
                for part in parts:
                    if part.strip():
                        new_sentences.append(part.strip())
            sentences = new_sentences if new_sentences else sentences

        return sentences

    def _merge_short_sentences(self, sentences: List[str], min_length: int = 10) -> List[str]:
        """Merge very short sentences with previous ones to maintain context."""
        if not sentences:
            return []

        merged = []
        current_sentence = sentences[0]

        for sentence in sentences[1:]:
            if len(sentence) < min_length and merged:
                # Merge with previous sentence
                current_sentence += " " + sentence
            else:
                merged.append(current_sentence)
                current_sentence = sentence

        merged.append(current_sentence)
        return merged


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
    """Statistics for sentence-level parser operations."""

    def reset(self):
        self.texts_processed = 0
        self.sentences_processed = 0
        self.sentences_cached = 0
        self.total_chars_processed = 0
        self.average_processing_time = 0.0
        self.cache_hit_rate = 0.0
        self.last_25_percent_processed = 0

    def record_processing(self, processing_time: float, chars_processed: int, sentences_processed: int = 0, cache_hits: int = 0, cache_misses: int = 0):
        """Record a processing operation."""
        self.texts_processed += 1
        self.total_chars_processed += chars_processed
        self.sentences_processed += sentences_processed
        self.sentences_cached += cache_hits
        self.last_25_percent_processed += cache_misses

        # Update averages
        total_requests = cache_hits + cache_misses
        if total_requests > 0:
            self.cache_hit_rate = cache_hits / total_requests if total_requests > 0 else 0.0

        self.average_processing_time = self.update_average_time(self.average_processing_time, self.texts_processed, processing_time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "texts_processed": self.texts_processed,
            "sentences_processed": self.sentences_processed,
            "sentences_cached": self.sentences_cached,
            "total_chars_processed": self.total_chars_processed,
            "average_processing_time": self.average_processing_time,
            "cache_hit_rate": self.cache_hit_rate,
            "last_25_percent_processed": self.last_25_percent_processed,
        }


# =============================================================================
# Parser Implementation
# =============================================================================


class Parser(EventBasedProcessor):
    """
    Enhanced LLM-based text parser with sentence-level validation and caching.
    - Processes only the last 25% of sentences through LLM
    - Uses caching to prevent duplicate processing
    - Supports English and Chinese punctuation
    - Returns structured ParsedTranscript objects for sharing between UI and analyzer
    """

    def __init__(
        self,
        model_id: str = "openai/gpt-4.1-nano",
        api_key: Optional[str] = None,
        config: Optional[ParserConfig] = None,
        cache_size: int = 1000,
        parser_window: int = 100,
    ):
        """Initialize the sentence-level parser.

        Args:
            model_id: The model ID to use (defaults to openai/gpt-4.1-nano for faster processing)
            api_key: Optional API key override (defaults to OPENROUTER_API_KEY env var)
            config: Configuration for the text parser
            cache_size: Maximum number of sentences to cache
        """
        super().__init__(queue_maxsize=50, cooldown_seconds=0.3, max_concurrent_workers=2, enable_work_coordination=False)

        self.config = config or ParserConfig(model_id=model_id)
        self.api_key = api_key  # Store for lazy initialization

        # Initialize sentence processing components
        self.sentence_splitter = SentenceSplitter()
        self.sentence_cache = SentenceCache(max_size=cache_size, similarity_threshold=0.85)

        # Use parser_window from config if available, otherwise use parameter
        if config and hasattr(config, "parser_window"):
            self.parser_window = config.parser_window
        else:
            self.parser_window = parser_window

        # Lazy initialization - only create when first needed
        self._llm_client = None
        self._prompt = None

        # Enhanced statistics for sentence processing
        self.stats = ParserStats()

        logger.info(f"Sentence-level parser initialized - processes sentences within {parser_window}-char window with caching")

        # Callback for storing parsed results
        self._result_callback = None

    def _is_stateful_processor(self) -> bool:
        """Parser maintains cache state."""
        return True

    @property
    def llm_client(self) -> UniversalLLM:
        """Lazy initialization of LLM client to speed up startup."""
        if self._llm_client is None:
            llm_config = LLMConfig(model_id=self.config.model_id, api_key=self.api_key, timeout=15.0, temperature=0.0)
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
                        """Refine the given sentences from transcription:
- Make minimal contextual changes
- Fix typos only if highly confident
- Add appropriate punctuation
- For Chinese text, use Chinese punctuation (、，。！？)
- Preserve original meaning and context
- Maintain sentence boundaries

IMPORTANT: Always respond in the same language and script as the input text.""",
                    ),
                    ("human", "Sentences to refine:\n{sentences}"),
                ]
            )
            logger.debug("Lazy-initialized prompt template for sentence parser")
        return self._prompt

    async def _process_item(self, item):
        """
        Process a single parsing request from the queue using sentence-level validation.
        """
        start_time = time.time()
        try:
            # Handle WorkItem wrapper - extract the actual data
            if hasattr(item, "data"):
                text, speaker_info, timestamps = item.data
            else:
                # Direct tuple format
                text, speaker_info, timestamps = item

            if not text or not text.strip():
                logger.debug("Skipping empty text for sentence parsing.")
                return

            logger.info(f"Processing sentence-level parsing: {len(text)} chars")

            # Process using sentence-level validation
            result = await self.parse_transcript(text, speaker_info, timestamps)

            # Use the callback to send the result back
            if self._result_callback:
                await self._result_callback(result)
                logger.debug(f"Sent sentence-parsed result via callback: {len(result.parsed_text)} chars")

            # Record statistics
            processing_time = time.time() - start_time
            logger.info(f"Sentence parser processed {len(text)} chars in {processing_time:.2f}s")

        except Exception as e:
            logger.error(f"Sentence parser processing failed: {e}", exc_info=True)
            # Do not re-raise, as it would stop the worker.

    async def queue_parsing_request(self, text: str, speaker_info: Optional[List[Dict]] = None, timestamps: Optional[Dict[str, float]] = None) -> bool:
        """Queue a sentence-level parsing request.

        Args:
            text: Text to parse
            speaker_info: Optional speaker information
            timestamps: Optional timestamp information

        Returns:
            bool: True if successfully queued, False otherwise
        """
        if not text.strip():
            return False

        try:
            work_item = WorkItem(data=(text, speaker_info, timestamps), item_id=hashlib.md5(text.encode()).hexdigest())

            success = await self.queue_for_processing(work_item)
            if success:
                logger.debug(f"Queued sentence parsing request: {len(text)} chars")
            else:
                logger.debug("Sentence parsing queue full, will retry later")

            return success

        except Exception as e:
            logger.error(f"Failed to queue sentence parsing request: {e}")
            return False

    async def parse_transcript(self, text: str, speaker_info: Optional[List[Dict]] = None, timestamps: Optional[Dict[str, float]] = None) -> ParsedTranscript:
        """Parse transcript using sentence-level validation with caching.

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
            logger.debug(f"Sentence parsing {len(text)} characters (Chinese: {is_chinese}): '{text[:50]}...'")

            # Split text into sentences using enhanced splitter
            sentences = self.sentence_splitter.split_sentences(text)
            if not sentences:
                return ParsedTranscript(original_text=text, parsed_text=text, parsing_time=time.time() - start_time)

            logger.debug(f"Split into {len(sentences)} sentences")

            # Calculate sentences to process based on character window
            # Find the character position that's parser_window characters from the end
            text_length = len(text)
            window_start_char = max(0, text_length - self.parser_window)

            # Find which sentences are touched by this character window
            current_char_pos = 0
            sentences_to_process_start = 0

            for i, sentence in enumerate(sentences):
                sentence_start = current_char_pos
                sentence_end = current_char_pos + len(sentence)

                # If this sentence overlaps with our window, we need to process from here
                if sentence_end > window_start_char:
                    sentences_to_process_start = i
                    break

                current_char_pos = sentence_end + 1  # +1 for separator

            sentences_to_process_count = len(sentences) - sentences_to_process_start

            logger.debug(f"Processing last {sentences_to_process_count} sentences (from index {sentences_to_process_start}) covering {self.parser_window}-char window")

            # Process sentences with caching
            processed_sentences = []
            cache_hits = 0
            cache_misses = 0

            for i, sentence in enumerate(sentences):
                sentence_info = self.sentence_cache.add_sentence(sentence, i)

                if i >= sentences_to_process_start:
                    # Process this sentence (within character window)
                    if self.sentence_cache.is_processed(sentence_info.hash):
                        # Use exact cached result
                        processed_text = self.sentence_cache.get_processed_text(sentence_info.hash)
                        processed_sentences.append(processed_text or sentence)
                        cache_hits += 1
                    else:
                        # Check for similar processed sentence
                        similar_processed = self.sentence_cache.find_similar_processed(sentence)
                        if similar_processed:
                            # Use similar cached result
                            processed_sentences.append(similar_processed)
                            cache_hits += 1
                            logger.debug(f"🔄 Using similar cached result for: '{sentence[:30]}...'")
                        else:
                            # Process new sentence
                            processed_text = await self._process_sentence(sentence)
                            self.sentence_cache.mark_processed(sentence_info.hash, processed_text)
                            processed_sentences.append(processed_text)
                            cache_misses += 1
                else:
                    # Keep original sentence (outside character window)
                    processed_sentences.append(sentence)

            # Combine processed sentences
            separator = " " if not is_chinese else ""
            parsed_text = separator.join(processed_sentences)

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
            segments = self._create_segments(parsed_text, speaker_info, timestamps)

            parsing_time = time.time() - start_time

            # Record enhanced statistics
            self.stats.record_processing(processing_time=parsing_time, chars_processed=len(text), sentences_processed=len(sentences), cache_hits=cache_hits, cache_misses=cache_misses)

            result = ParsedTranscript(original_text=text, parsed_text=parsed_text, segments=segments, timestamps=timestamps or {}, speakers=speaker_info or [], parsing_time=parsing_time)

            logger.debug(f"Sentence parsing completed in {parsing_time:.2f}s: {cache_hits} cache hits, {cache_misses} cache misses")
            return result

        except Exception as e:
            logger.error(f"Failed to parse transcript with sentences: {e}")
            # Return original text as fallback
            return ParsedTranscript(original_text=text, parsed_text=text, parsing_time=time.time() - start_time)

    async def _process_sentence(self, sentence: str) -> str:
        """Process a single sentence through the LLM.

        Args:
            sentence: Sentence to process

        Returns:
            str: Processed sentence
        """
        try:
            result = await self.llm_client.invoke_text(self.prompt, {"sentences": sentence})
            return result.strip()
        except Exception as e:
            logger.warning(f"LLM sentence processing failed: {e}")
            return sentence  # Fallback to original

    async def parse_text(self, text: str) -> str:
        """Parse text and return only the corrected text string.

        Args:
            text: Text to parse

        Returns:
            str: Corrected text
        """
        result = await self.parse_transcript(text)
        return result.parsed_text

    def _create_segments(self, corrected_text: str, speaker_info: Optional[List[Dict]], timestamps: Optional[Dict[str, float]]) -> List[Dict]:
        """Create text segments with metadata using enhanced sentence splitting.

        Args:
            corrected_text: Corrected text from LLM
            speaker_info: Speaker information
            timestamps: Timestamp information

        Returns:
            List of segments with metadata
        """
        segments = []

        # Use sentence splitter for better segmentation
        sentences = self.sentence_splitter.split_sentences(corrected_text)
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
            start_pos += len(sentence) + 1  # +1 for separator

        return segments

    async def _process_chunk(self, text: str) -> str:
        """Process a single chunk of text through the LLM (legacy support).

        Args:
            text: Text chunk to process

        Returns:
            str: Processed text
        """
        try:
            result = await self.llm_client.invoke_text(self.prompt, {"sentences": text})
            return result
        except Exception as e:
            logger.warning(f"LLM processing failed: {e}")
            return text  # Fallback to original

    async def _process_in_chunks(self, text: str) -> tuple[str, int]:
        """Process long text by splitting into chunks (legacy support).

        Args:
            text: Long text to process

        Returns:
            tuple: (processed_text, chunks_used)
        """
        # For sentence-level processing, we use sentence splitting instead
        sentences = self.sentence_splitter.split_sentences(text)
        processed_chunks = []

        for sentence in sentences:
            processed_chunk = await self._process_sentence(sentence)
            processed_chunks.append(processed_chunk)

        return " ".join(processed_chunks), len(sentences)

    def _split_by_output_tokens(self, text: str) -> list[str]:
        """Split text into chunks based on sentences (updated for sentence processing).

        Args:
            text: Text to split

        Returns:
            list: Text chunks (sentences)
        """
        return self.sentence_splitter.split_sentences(text)

    def get_stats(self) -> dict:
        """Get processing statistics with coordination info.

        Returns:
            dict: Statistics including base stats and sentence processing details
        """
        base_stats = super().get_queue_status()
        parser_stats = self.stats.to_dict()

        # Add processor info to stats
        cache_stats = self.sentence_cache.get_cache_stats()
        parser_stats.update(
            {
                "stateful_processor": True,
                "cache_size": len(self.sentence_cache.sentences),
                "processed_sentences": len(self.sentence_cache.processed_hashes),
                "similarity_threshold": cache_stats["similarity_threshold"],
                "cache_utilization": cache_stats["cache_utilization"],
            }
        )

        return {**base_stats, **parser_stats}

    async def reset_state(self):
        """Reset parser state and clear sentence cache."""
        self.sentence_cache = SentenceCache(max_size=self.sentence_cache.max_size)
        self.stats.reset()
        logger.info("Sentence parser state reset - cache cleared")

    def cleanup_cache(self, max_age_seconds: float = 3600):
        """Clean up old cache entries."""
        self.sentence_cache.clear_old_entries(max_age_seconds)
        logger.debug(f"Cleaned up cache entries older than {max_age_seconds}s")

    def set_result_callback(self, callback):
        """Set a callback function to be called when parsing results are available.

        Args:
            callback: Async function that takes a ParsedTranscript object
        """
        self._result_callback = callback
        logger.debug("Sentence parser result callback set")


# Convenience function for quick text parsing (returns structured data)
async def parse_transcript(
    text: str,
    model_id: str = "openai/gpt-4.1-nano",
    api_key: Optional[str] = None,
    speaker_info: Optional[List[Dict]] = None,
    timestamps: Optional[Dict[str, float]] = None,
) -> ParsedTranscript:
    """Convenience function to quickly parse and structure transcript text using sentence-level processing.

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
