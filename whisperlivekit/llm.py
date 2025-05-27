import asyncio
import logging
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()

logger = logging.getLogger(__name__)


@dataclass
class SummaryTrigger:
    """Configuration for when to trigger summarization."""

    idle_time_seconds: float = 5.0
    max_text_length: int = 100000  # Just for sanity
    speakers_trigger_count: int = 2  # Trigger after this many speakers have spoken


class SummaryResponse(BaseModel):
    """Structured response from the LLM summarizer."""

    summary: str = Field(description="Concise summary of the transcription")
    key_points: list[str] = Field(default_factory=list, description="Main points discussed")


class LLM:
    """
    LLM-based transcription summarizer that monitors transcription activity
    and generates summaries after periods of inactivity.
    """

    def __init__(
        self,
        model_id: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        trigger_config: Optional[SummaryTrigger] = None,
    ):
        """Initialize the LLM summarizer.

        Args:
            model_id: The model ID to use (defaults to gpt-4o-mini)
            api_key: Optional API key override (defaults to OPENROUTER_API_KEY env var)
            trigger_config: Configuration for when to trigger summarization
        """
        self.model_id = model_id
        self.trigger_config = trigger_config or SummaryTrigger()

        # Initialize LLM
        api_key = api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = "https://openrouter.ai/api/v1" if os.getenv("OPENROUTER_API_KEY") else None

        self.llm = ChatOpenAI(
            model=model_id,
            api_key=api_key,
            base_url=base_url,
        )

        # Create structured LLM for summary responses
        self.structured_llm = self.llm.with_structured_output(SummaryResponse, method="function_calling")

        # Create prompt template
        self.prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are an expert at summarizing transcriptions from speech-to-text systems.
            
Your task is to analyze the transcription and provide:
1. A concise summary of what was discussed
2. Key points or topics mentioned

Focus on:
- Main topics and themes
- Important decisions or conclusions
- Action items if any
- Overall context and purpose of the conversation

Keep summaries clear and concise while capturing the essential information.

IMPORTANT: Always respond in the same language as the transcription. If the transcription is in French, respond in French. If it's in English, respond in English. Match the language of the input content.""",
                ),
                (
                    "human",
                    """Please summarize this transcription:

Transcription:
{transcription}

Additional context:
- Duration: {duration} seconds
- Has speaker diarization: {has_speakers}
- Number of lines: {num_lines}

Provide a structured summary with key points. Remember to respond in the same language as the transcription above.""",
                ),
            ]
        )

        # State tracking
        self.last_activity_time = time.time()
        self.accumulated_text = ""
        self.last_summarized_text = ""  # Keep track of what was already summarized
        self.last_summary = None
        self.summary_task = None
        self.is_running = False
        self.summary_callbacks = []
        self.consecutive_idle_checks = 0  # Track consecutive idle periods
        self.min_idle_checks = 5  # Require 5 consecutive idle checks (5 seconds)

        # Statistics
        self.stats = {
            "summaries_generated": 0,
            "total_text_summarized": 0,
            "average_summary_time": 0.0,
        }

    def add_summary_callback(self, callback):
        """Add a callback function to be called when a summary is generated.

        Args:
            callback: Function that takes (summary_response, transcription_text) as arguments
        """
        self.summary_callbacks.append(callback)

    def update_transcription(self, new_text: str, speaker_info: Optional[Dict] = None):
        """Update with new transcription text.

        Args:
            new_text: New transcription text to add
            speaker_info: Optional speaker/diarization information
        """
        if not new_text.strip():
            return

        self.last_activity_time = time.time()
        self.consecutive_idle_checks = 0  # Reset idle counter on new activity

        # Add to accumulated text with speaker info if available
        if speaker_info and "speaker" in speaker_info:
            formatted_text = f"[Speaker {speaker_info['speaker']}]: {new_text}"
        else:
            formatted_text = new_text

        if self.accumulated_text:
            self.accumulated_text += " " + formatted_text
        else:
            self.accumulated_text = formatted_text

        logger.debug(f"Updated transcription: {len(self.accumulated_text)} chars total, " f"idle_checks reset to 0")

    async def start_monitoring(self):
        """Start monitoring for summarization triggers."""
        if self.is_running:
            return

        self.is_running = True
        logger.info("Started LLM summarization monitoring")

        while self.is_running:
            try:
                await self._check_and_summarize()
                await asyncio.sleep(1.0)  # Check every second
            except Exception as e:
                logger.error(f"Error in summarization monitoring: {e}")
                await asyncio.sleep(5.0)  # Back off on error

    async def stop_monitoring(self):
        """Stop monitoring."""
        self.is_running = False
        logger.info("Stopped LLM summarization monitoring")

    async def _check_and_summarize(self):
        """Check if conditions are met for summarization."""
        if not self.accumulated_text.strip():
            self.consecutive_idle_checks = 0
            return

        time_since_activity = time.time() - self.last_activity_time
        text_length = len(self.accumulated_text)

        # Check if we're truly idle (no new activity for required time)
        if time_since_activity >= 1.0:  # At least 1 second since last activity
            self.consecutive_idle_checks += 1
        else:
            self.consecutive_idle_checks = 0
            return

        # Only summarize if we have sufficient idle time
        new_text_length = len(self.accumulated_text) - len(self.last_summarized_text)
        is_truly_idle = self.consecutive_idle_checks >= self.min_idle_checks

        logger.debug(f"Idle check: consecutive={self.consecutive_idle_checks}, " f"new_text={new_text_length} chars, truly_idle={is_truly_idle}")

        if is_truly_idle:
            logger.info(f"🔄 Triggering summary: {self.consecutive_idle_checks}s idle, " f"{new_text_length} new chars")
            await self._generate_summary()
            self.consecutive_idle_checks = 0  # Reset after summarizing
        elif self.consecutive_idle_checks > 0 and self.consecutive_idle_checks % 5 == 0:
            # Log every 5 seconds when we're in idle mode
            logger.info(f"⏳ Idle for {self.consecutive_idle_checks}s, new_text={new_text_length} chars")

    async def _generate_summary(self):
        """Generate summary using the LLM."""
        if not self.accumulated_text.strip():
            return

        # Get only the new text since last summary
        if self.last_summarized_text:
            # Find where the last summarized text ends in the current accumulated text
            last_summary_end = self.accumulated_text.find(self.last_summarized_text)
            if last_summary_end != -1:
                last_summary_end += len(self.last_summarized_text)
                text_to_summarize = self.accumulated_text[last_summary_end:].strip()
                if not text_to_summarize:
                    logger.debug("No new text to summarize")
                    return
            else:
                # If we can't find the overlap, summarize the recent portion
                text_to_summarize = self.accumulated_text[-self.trigger_config.max_text_length :].strip()
        else:
            text_to_summarize = self.accumulated_text

        # Truncate if too long
        if len(text_to_summarize) > self.trigger_config.max_text_length:
            text_to_summarize = text_to_summarize[-self.trigger_config.max_text_length :]
            logger.info(f"Truncated text to {self.trigger_config.max_text_length} characters")

        try:
            start_time = time.time()
            logger.info(f"Generating summary for {len(text_to_summarize)} chars of new content...")

            # Prepare context information
            lines = text_to_summarize.split("\n")
            has_speakers = "[Speaker" in text_to_summarize
            duration_estimate = len(text_to_summarize) / 10  # Rough estimate: 10 chars per second

            # Create chain and invoke
            chain = self.prompt | self.structured_llm
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: chain.invoke(
                    {
                        "transcription": text_to_summarize,
                        "duration": duration_estimate,
                        "has_speakers": has_speakers,
                        "num_lines": len(lines),
                    }
                ),
            )

            generation_time = time.time() - start_time

            # Update statistics
            self.stats["summaries_generated"] += 1
            self.stats["total_text_summarized"] += len(text_to_summarize)

            # Update average time
            prev_avg = self.stats["average_summary_time"]
            count = self.stats["summaries_generated"]
            self.stats["average_summary_time"] = (prev_avg * (count - 1) + generation_time) / count

            self.last_summary = response

            logger.info(f"Generated summary in {generation_time:.2f}s: {len(response.summary)} chars")
            logger.debug(f"Summary: {response.summary}")

            # Call registered callbacks
            for callback in self.summary_callbacks:
                try:
                    await callback(response, text_to_summarize)
                except Exception as e:
                    logger.error(f"Error in summary callback: {e}")

            # Update the last summarized text to include what we just summarized
            # Keep a rolling buffer to allow for new summaries of additional content
            self.last_summarized_text = self.accumulated_text

            # Keep the most recent text in buffer (last 5000 chars) to maintain context
            if len(self.accumulated_text) > 5000:
                self.accumulated_text = self.accumulated_text[-5000:]
                self.last_summarized_text = self.accumulated_text

        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")

    def get_last_summary(self) -> Optional[SummaryResponse]:
        """Get the most recent summary."""
        return self.last_summary

    def get_stats(self) -> Dict[str, Any]:
        """Get summarization statistics."""
        return self.stats.copy()

    async def force_summary(self) -> Optional[SummaryResponse]:
        """Force generate a summary of current accumulated text."""
        if not self.accumulated_text.strip():
            return None

        await self._generate_summary()
        return self.last_summary
