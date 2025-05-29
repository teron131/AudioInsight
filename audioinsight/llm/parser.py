import asyncio
import time
from typing import Optional

from langchain.prompts import ChatPromptTemplate

from ..logging_config import get_logger
from .base import UniversalLLM
from .types import LLMConfig, ParserConfig
from .utils import contains_chinese, s2hk

logger = get_logger(__name__)


class Parser:
    """
    LLM-based text parser that fixes typos, punctuation, and creates smooth sentences.
    Uses the universal LLM client for consistent inference.
    """

    def __init__(
        self,
        model_id: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        config: Optional[ParserConfig] = None,
    ):
        """Initialize the text parser.

        Args:
            model_id: The model ID to use (defaults to gpt-4o-mini for faster processing)
            api_key: Optional API key override (defaults to OPENROUTER_API_KEY env var)
            config: Configuration for the text parser
        """
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
                    """You are an expert subtitle editor. Your task is to refine a sequence of piecemeal subtitle derived from transcription. These subtitle may contain typos and lack proper punctuation.

Follow the guidelines below to ensure high-quality subtitle:
1. Make minimal contextual changes.
2. Only make contextual changes if you are highly confident.
3. Add punctuation appropriately.

IMPORTANT: Always respond in the same language and script as the input text.""",
                ),
                (
                    "human",
                    """Original Subtitle: {text}

Refined Subtitle:""",
                ),
            ]
        )

        # Statistics
        self.stats = {
            "texts_processed": 0,
            "total_chars_processed": 0,
            "average_processing_time": 0.0,
            "chunks_processed": 0,
        }

    async def parse_text(self, text: str) -> str:
        """Parse and correct a text string.

        Args:
            text: The text to parse and correct

        Returns:
            str: The corrected text
        """
        if not text.strip():
            return text

        start_time = time.time()

        try:
            # If text is short enough, process in one go
            if len(text) <= self.config.chunk_size:
                corrected_text = await self._process_chunk(text)
            else:
                # Process in chunks for longer text
                corrected_text = await self._process_in_chunks(text)

            # Apply s2hk conversion to ensure Traditional Chinese output if input was Chinese
            if contains_chinese(corrected_text):
                corrected_text = s2hk(corrected_text)

            # Update statistics
            processing_time = time.time() - start_time
            self.stats["texts_processed"] += 1
            self.stats["total_chars_processed"] += len(text)

            # Update average processing time
            prev_avg = self.stats["average_processing_time"]
            count = self.stats["texts_processed"]
            self.stats["average_processing_time"] = (prev_avg * (count - 1) + processing_time) / count

            logger.info(f"Parsed text in {processing_time:.2f}s: {len(text)} -> {len(corrected_text)} chars")
            return corrected_text

        except Exception as e:
            logger.error(f"Failed to parse text: {e}")
            return text  # Return original text if parsing fails

    async def _process_chunk(self, text: str) -> str:
        """Process a single chunk of text.

        Args:
            text: The text chunk to process

        Returns:
            str: The corrected text directly
        """
        result = await self.llm_client.invoke_text(self.prompt, {"text": text})
        self.stats["chunks_processed"] += 1
        return result

    async def _process_in_chunks(self, text: str) -> str:
        """Process long text in overlapping chunks.

        Args:
            text: The full text to process

        Returns:
            str: The complete corrected text
        """
        chunks = self._split_into_chunks(text)
        corrected_chunks = []

        for i, chunk in enumerate(chunks):
            logger.debug(f"Processing chunk {i+1}/{len(chunks)}: {len(chunk)} chars")
            corrected_chunk = await self._process_chunk(chunk)
            corrected_chunks.append(corrected_chunk)

        # Merge chunks, handling overlaps
        return self._merge_chunks(corrected_chunks, text)

    def _split_into_chunks(self, text: str) -> list[str]:
        """Split text into overlapping chunks.

        Args:
            text: The text to split

        Returns:
            list[str]: List of text chunks
        """
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.config.chunk_size

            # If this isn't the last chunk, try to break at a sentence boundary
            if end < len(text):
                # Look for sentence breaks in the overlap region
                overlap_start = max(start, end - self.config.overlap_size)
                sentence_breaks = []

                # Look for various sentence endings
                for punct in ["。", ".", "！", "!", "？", "?", "\n\n"]:
                    pos = text.rfind(punct, overlap_start, end)
                    if pos != -1:
                        sentence_breaks.append(pos + len(punct))

                if sentence_breaks:
                    end = max(sentence_breaks)

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            start = end - self.config.overlap_size if end < len(text) else end

        return chunks

    def _merge_chunks(self, corrected_chunks: list[str], original_text: str) -> str:
        """Merge corrected chunks back together, handling overlaps.

        Args:
            corrected_chunks: List of corrected text chunks
            original_text: The original text for reference

        Returns:
            str: The merged corrected text
        """
        if not corrected_chunks:
            return original_text

        if len(corrected_chunks) == 1:
            return corrected_chunks[0]

        # Simple merging - for more sophisticated overlap handling,
        # we could implement fuzzy matching between chunk boundaries
        merged = corrected_chunks[0]

        for chunk in corrected_chunks[1:]:
            # Simple append with space separation
            # In a more sophisticated implementation, we would handle overlaps
            if not merged.endswith(" ") and not chunk.startswith(" "):
                merged += " "
            merged += chunk

        return merged

    def get_stats(self) -> dict:
        """Get processing statistics.

        Returns:
            dict: Dictionary of processing statistics
        """
        return self.stats.copy()


# Convenience function for quick text parsing
async def parse_transcript(
    text: str,
    model_id: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
) -> str:
    """Convenience function to quickly parse and correct transcript text.

    Args:
        text: The text to parse and correct
        model_id: The model ID to use
        api_key: Optional API key override

    Returns:
        str: The corrected text
    """
    parser = Parser(model_id=model_id, api_key=api_key)
    return await parser.parse_text(text)
