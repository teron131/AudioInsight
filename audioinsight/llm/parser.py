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
from .llm_base import UniversalLLM
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
            if sentence_hash in self.sentences:
                old_sentence = self.sentences[sentence_hash]
                if old_sentence in self.processed_sentences:
                    self.processed_sentences.remove(old_sentence)

                del self.sentences[sentence_hash]
                self.processed_hashes.discard(sentence_hash)
                if sentence_hash in self.insertion_order:
                    self.insertion_order.remove(sentence_hash)

        if to_remove:
            logger.info(f"Cleared {len(to_remove)} old cache entries")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "total_sentences": len(self.sentences),
            "processed_sentences": len(self.processed_hashes),
            "cache_hit_rate": len(self.processed_hashes) / max(len(self.sentences), 1),
            "similarity_threshold": self.similarity_threshold,
            "max_size": self.max_size,
        }


# =============================================================================
# Sentence Splitting
# =============================================================================


class SentenceSplitter:
    """Enhanced sentence splitter with support for English and Chinese punctuation."""

    def __init__(self):
        # Enhanced separators for better sentence boundary detection
        self.splitter = RecursiveCharacterTextSplitter(
            separators=[". ", "! ", "? ", "。", "！", "？"],
            chunk_size=1000,  # Large chunk size to avoid premature splitting
            chunk_overlap=0,
            length_function=len,
        )

    def split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using enhanced separators."""
        if not text.strip():
            return []

        # Use the text splitter to get initial chunks
        chunks = self.splitter.split_text(text)

        # Further split chunks by sentence patterns
        sentences = []
        for chunk in chunks:
            chunk_sentences = self._split_chunk_by_patterns(chunk)
            sentences.extend(chunk_sentences)

        # Merge very short sentences with the next one
        sentences = self._merge_short_sentences(sentences)

        # Filter out empty sentences
        sentences = [s.strip() for s in sentences if s.strip()]

        return sentences

    def _split_chunk_by_patterns(self, chunk: str) -> List[str]:
        """Split a chunk by sentence patterns."""
        # Pattern for sentence endings (English and Chinese)
        sentence_pattern = r"([.!?;:。！？；：])\s*"

        # Split by the pattern but keep the punctuation
        parts = re.split(sentence_pattern, chunk)

        sentences = []
        current_sentence = ""

        for i, part in enumerate(parts):
            if i % 2 == 0:  # Text part
                current_sentence += part
            else:  # Punctuation part
                current_sentence += part
                if current_sentence.strip():
                    sentences.append(current_sentence.strip())
                current_sentence = ""

        # Add any remaining text
        if current_sentence.strip():
            sentences.append(current_sentence.strip())

        return sentences

    def _merge_short_sentences(self, sentences: List[str], min_length: int = 10) -> List[str]:
        """Merge very short sentences with the next one."""
        if not sentences:
            return sentences

        merged = []
        i = 0

        while i < len(sentences):
            current = sentences[i]

            # If current sentence is too short and there's a next sentence, merge them
            if len(current) < min_length and i + 1 < len(sentences):
                next_sentence = sentences[i + 1]
                merged_sentence = f"{current} {next_sentence}".strip()
                merged.append(merged_sentence)
                i += 2  # Skip the next sentence as it's been merged
            else:
                merged.append(current)
                i += 1

        return merged


# =============================================================================
# Pydantic Models
# =============================================================================


class ParsedTranscript(BaseModel):
    """Structured response from transcript parsing."""

    original_text: str = Field(description="Original transcribed text")
    parsed_text: str = Field(description="Parsed and corrected text")
    segments: List[Dict[str, Any]] = Field(default_factory=list, description="Text segments with metadata")
    timestamps: Dict[str, float] = Field(default_factory=dict, description="Important timestamps")
    speakers: List[Dict[str, Any]] = Field(default_factory=list, description="Speaker information")
    parsing_time: float = Field(default=0.0, description="Time taken to parse the transcript")


# =============================================================================
# Statistics Tracking
# =============================================================================


class BaseStats:
    """Base class for statistics tracking."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all statistics to initial values."""
        pass

    def to_dict(self) -> Dict[str, Any]:
        """Convert statistics to dictionary format."""
        return {}

    def update_average_time(self, current_avg: float, count: int, new_time: float) -> float:
        """Update running average with new time measurement."""
        if count <= 1:
            return new_time
        return (current_avg * (count - 1) + new_time) / count


class ParserStats(BaseStats):
    """Statistics tracking for parser operations."""

    def reset(self):
        """Reset all parser statistics."""
        self.total_processed = 0
        self.total_chars_processed = 0
        self.total_sentences_processed = 0
        self.average_processing_time = 0.0
        self.cache_hits = 0
        self.cache_misses = 0
        self.similarity_matches = 0

    def record_processing(self, processing_time: float, chars_processed: int, sentences_processed: int = 0, cache_hits: int = 0, cache_misses: int = 0):
        """Record a processing operation."""
        self.total_processed += 1
        self.total_chars_processed += chars_processed
        self.total_sentences_processed += sentences_processed
        self.cache_hits += cache_hits
        self.cache_misses += cache_misses

        # Update average processing time
        self.average_processing_time = self.update_average_time(self.average_processing_time, self.total_processed, processing_time)

    def to_dict(self) -> Dict[str, Any]:
        """Get statistics as dictionary."""
        total_cache_operations = self.cache_hits + self.cache_misses
        cache_hit_rate = self.cache_hits / max(total_cache_operations, 1)

        return {
            "total_processed": self.total_processed,
            "total_chars_processed": self.total_chars_processed,
            "total_sentences_processed": self.total_sentences_processed,
            "average_processing_time": self.average_processing_time,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": cache_hit_rate,
            "similarity_matches": self.similarity_matches,
        }


# =============================================================================
# Simplified Parser Implementation
# =============================================================================


class Parser:
    """
    Simplified LLM-powered transcript parser with sentence-level processing and caching.

    Uses direct await calls instead of complex queue system for predictable time-ordered processing.
    """

    def __init__(
        self,
        model_id: str = "openai/gpt-4.1-nano",
        api_key: Optional[str] = None,
        config: Optional[ParserConfig] = None,
        cache_size: int = 1000,
        parser_window: int = 100,
    ):
        """Initialize the parser with configuration."""
        self.model_id = model_id
        self.api_key = api_key
        self.config = config or ParserConfig()
        self.parser_window = parser_window

        # Initialize LLM client
        llm_config = LLMConfig(model_id=model_id, api_key=api_key)
        self._llm_client = UniversalLLM(llm_config)

        # Initialize components
        self.sentence_splitter = SentenceSplitter()
        self.sentence_cache = SentenceCache(max_size=cache_size)
        self.stats = ParserStats()

        # Initialize prompt template for lazy loading
        self._prompt = None

        # Result callback
        self.result_callback = None

        logger.info("Parser initialized with direct await processing")

    @property
    def llm_client(self) -> UniversalLLM:
        """Lazy initialization of LLM client to speed up startup."""
        return self._llm_client

    @property
    def prompt(self) -> ChatPromptTemplate:
        """Lazy initialization of prompt template to speed up startup."""
        if self._prompt is None:
            self._prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """You are a transcript correction assistant. Your task is to:

1. Fix grammatical errors, typos, and speech recognition mistakes
2. Improve sentence structure and flow while preserving the original meaning
3. Maintain the speaker's tone, style, and intent
4. Keep technical terms, names, and specific details accurate

Guidelines:
- Make minimal changes - only fix clear errors
- Don't add new information or change the meaning
- Keep the same language and regional conventions as the original
- Maintain natural speech patterns and colloquialisms when appropriate
- If unsure about a correction, leave the text as-is""",
                    ),
                    ("human", "{text}"),
                ]
            )
            logger.debug("Lazy-initialized prompt template")
        return self._prompt

    async def parse_transcript_direct(self, text: str, speaker_info: Optional[List[Dict]] = None, timestamps: Optional[Dict[str, float]] = None) -> ParsedTranscript:
        """
        Parse transcript directly with await (simplified from queue-based processing).

        Args:
            text: Text to parse
            speaker_info: Optional speaker information
            timestamps: Optional timestamp information

        Returns:
            ParsedTranscript with corrected text and metadata
        """
        start_time = time.time()

        try:
            # Process the text using character window approach
            corrected_text = await self._process_with_character_window(text)

            # Create segments with metadata
            segments = self._create_segments(corrected_text, speaker_info, timestamps)

            # Calculate processing time
            processing_time = time.time() - start_time

            # Update statistics
            self.stats.record_processing(processing_time=processing_time, chars_processed=len(text), sentences_processed=len(self.sentence_splitter.split_sentences(text)))

            # Create result
            result = ParsedTranscript(
                original_text=text,
                parsed_text=corrected_text,
                segments=segments,
                timestamps=timestamps or {},
                speakers=speaker_info or [],
                parsing_time=processing_time,
            )

            # Call result callback if set
            if self.result_callback:
                try:
                    await self.result_callback(result)
                except Exception as e:
                    logger.error(f"Error in result callback: {e}")

            logger.debug(f"Parsed transcript in {processing_time:.2f}s: {len(text)} chars -> {len(corrected_text)} chars")
            return result

        except Exception as e:
            logger.error(f"Error parsing transcript: {e}")
            # Return original text on error
            return ParsedTranscript(
                original_text=text,
                parsed_text=text,
                segments=self._create_segments(text, speaker_info, timestamps),
                timestamps=timestamps or {},
                speakers=speaker_info or [],
                parsing_time=time.time() - start_time,
            )

    async def _process_with_character_window(self, text: str) -> str:
        """Process text using character window approach with sentence-level protection."""
        if not text.strip():
            return text

        # Split into sentences
        sentences = self.sentence_splitter.split_sentences(text)
        if not sentences:
            return text

        # Calculate character window - look at last N characters
        window_start = max(0, len(text) - self.parser_window)
        window_text = text[window_start:]

        # Find which sentences are touched by the window
        sentences_to_process = []
        current_pos = 0

        for i, sentence in enumerate(sentences):
            sentence_start = current_pos
            sentence_end = current_pos + len(sentence)

            # Check if this sentence overlaps with the window
            if sentence_end > window_start:
                sentences_to_process.append(i)

            current_pos = sentence_end + 1  # +1 for space between sentences

        if not sentences_to_process:
            return text

        # Process the identified sentences
        cache_hits = 0
        cache_misses = 0

        for sentence_idx in sentences_to_process:
            sentence = sentences[sentence_idx]
            sentence_info = self.sentence_cache.add_sentence(sentence, sentence_idx)

            if sentence_info.processed:
                # Use cached result
                sentences[sentence_idx] = sentence_info.processed_text
                cache_hits += 1
            else:
                # Process with LLM
                try:
                    processed_sentence = await self._process_sentence(sentence)
                    sentences[sentence_idx] = processed_sentence
                    self.sentence_cache.mark_processed(sentence_info.hash, processed_sentence)
                    cache_misses += 1
                except Exception as e:
                    logger.error(f"Error processing sentence: {e}")
                    # Keep original sentence on error
                    cache_misses += 1

        # Update cache statistics
        self.stats.cache_hits += cache_hits
        self.stats.cache_misses += cache_misses

        # Reconstruct the text
        return " ".join(sentences)

    async def _process_sentence(self, sentence: str) -> str:
        """Process a single sentence with the LLM."""
        try:
            result = await self.llm_client.invoke_text(self.prompt, {"text": sentence})
            return result.strip()
        except Exception as e:
            logger.error(f"Error processing sentence with LLM: {e}")
            return sentence  # Return original on error

    def _create_segments(self, corrected_text: str, speaker_info: Optional[List[Dict]], timestamps: Optional[Dict[str, float]]) -> List[Dict]:
        """Create text segments with metadata."""
        segments = []

        # Split corrected text into sentences for segmentation
        sentences = self.sentence_splitter.split_sentences(corrected_text)

        current_pos = 0
        for i, sentence in enumerate(sentences):
            segment = {
                "index": i,
                "text": sentence,
                "start_pos": current_pos,
                "end_pos": current_pos + len(sentence),
                "word_count": len(sentence.split()),
                "char_count": len(sentence),
            }

            # Add speaker info if available
            if speaker_info:
                # Simple heuristic: assign speaker based on position
                speaker_idx = min(i, len(speaker_info) - 1)
                if speaker_idx < len(speaker_info):
                    segment["speaker"] = speaker_info[speaker_idx]

            # Add timestamp info if available
            if timestamps:
                # Simple heuristic: interpolate timestamps
                if "start" in timestamps and "end" in timestamps:
                    total_duration = timestamps["end"] - timestamps["start"]
                    segment_start = timestamps["start"] + (i / len(sentences)) * total_duration
                    segment_end = timestamps["start"] + ((i + 1) / len(sentences)) * total_duration
                    segment["start_time"] = segment_start
                    segment["end_time"] = segment_end

            segments.append(segment)
            current_pos += len(sentence) + 1  # +1 for space

        return segments

    def get_stats(self) -> dict:
        """Get parser statistics."""
        stats = self.stats.to_dict()
        stats.update(self.sentence_cache.get_cache_stats())
        return stats

    def cleanup_cache(self, max_age_seconds: float = 3600):
        """Clean up old cache entries."""
        self.sentence_cache.clear_old_entries(max_age_seconds)

    def set_result_callback(self, callback):
        """Set callback function for processing results."""
        self.result_callback = callback

    async def reset_state(self):
        """Reset parser state."""
        self.stats.reset()
        self.sentence_cache = SentenceCache(max_size=self.sentence_cache.max_size)
        logger.info("Parser state reset")


# =============================================================================
# Factory Functions
# =============================================================================


async def parse_transcript(
    text: str,
    model_id: str = "openai/gpt-4.1-nano",
    api_key: Optional[str] = None,
    speaker_info: Optional[List[Dict]] = None,
    timestamps: Optional[Dict[str, float]] = None,
) -> ParsedTranscript:
    """
    Factory function to parse transcript with simplified direct processing.

    Args:
        text: Text to parse
        model_id: LLM model identifier
        api_key: API key for LLM service
        speaker_info: Optional speaker information
        timestamps: Optional timestamp information

    Returns:
        ParsedTranscript with corrected text and metadata
    """
    parser = Parser(model_id=model_id, api_key=api_key)
    return await parser.parse_transcript_direct(text, speaker_info, timestamps)
