import asyncio
import time
from typing import Any, Dict, Optional

from langchain.prompts import ChatPromptTemplate

from ..logging_config import get_logger
from .base import EventBasedProcessor, UniversalLLM
from .types import LLMConfig, LLMResponse, LLMStats, LLMTrigger
from .utils import s2hk, truncate_text

logger = get_logger(__name__)


class LLMSummarizer(EventBasedProcessor):
    """
    LLM-based transcription processor that monitors transcription activity
    and generates inference after periods of inactivity or after a certain number of conversations.
    Uses the universal LLM client for consistent inference and EventBasedProcessor for queue management.
    """

    def __init__(
        self,
        model_id: str = "openai/gpt-4.1-mini",
        api_key: Optional[str] = None,
        trigger_config: Optional[LLMTrigger] = None,
    ):
        """Initialize the LLM inference processor.

        Args:
            model_id: The model ID to use (defaults to openai/gpt-4.1-mini)
            api_key: Optional API key override (defaults to OPENROUTER_API_KEY env var)
            trigger_config: Configuration for when to trigger LLM inference
        """
        # Initialize base class with optimized queue for summarizer and concurrent workers
        super().__init__(queue_maxsize=100, cooldown_seconds=0.5, max_concurrent_workers=2)  # Increased queue size, reduced cooldown, added concurrency

        self.model_id = model_id
        self.trigger_config = trigger_config or LLMTrigger()

        # Initialize universal LLM client
        llm_config = LLMConfig(
            model_id=model_id,
            api_key=api_key,
        )
        self.llm_client = UniversalLLM(llm_config)

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

IMPORTANT: Always respond in the same language and script as the transcription. 
- If the transcription is in Chinese (繁體中文), respond in Traditional Chinese using Hong Kong style conventions.
- If the transcription is in French, respond in French. 
- If it's in English, respond in English. 
- Match the exact language, script, and regional conventions of the input content.""",
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

Provide a structured summary with key points. Remember to respond in the same language, script, and regional conventions as the transcription above.""",
                ),
            ]
        )

        # State tracking (use inherited accumulated_data from base class)
        self.text_length_at_last_summary = 0  # Track text length at last summary for new text trigger
        self.last_inference = None
        self.inference_callbacks = []

        # Statistics
        self.stats = LLMStats()

    async def _process_item(self, item: Any):
        """Process a single inference request from the queue.

        Args:
            item: The trigger reason string
        """
        trigger_reason = item
        await self._generate_inference(trigger_reason=trigger_reason)
        self.update_processing_time()

    def add_inference_callback(self, callback):
        """Add a callback function to be called when inference is generated.

        Args:
            callback: Function that takes (inference_response, transcription_text) as arguments
        """
        self.inference_callbacks.append(callback)

    def add_summary_callback(self, callback):
        """Legacy method for backward compatibility. Use add_inference_callback instead."""
        self.add_inference_callback(callback)

    def update_transcription(self, new_text: str, speaker_info: Optional[Dict] = None):
        """Update with new transcription text.

        Args:
            new_text: New transcription text to add
            speaker_info: Optional speaker/diarization information
        """
        if not new_text.strip():
            return

        # Add to accumulated text with speaker info if available
        if speaker_info and "speaker" in speaker_info:
            formatted_text = f"[Speaker {speaker_info['speaker']}]: {new_text}"
        else:
            formatted_text = new_text

        if self.accumulated_data:
            self.accumulated_data += " " + formatted_text
        else:
            self.accumulated_data = formatted_text

        logger.debug(f"Updated transcription: {len(self.accumulated_data)} chars total")

        # Check if we should trigger inference
        self._check_inference_triggers()

    def _check_inference_triggers(self):
        """Check if conditions are met for inference and queue request if needed."""
        if not self.accumulated_data.strip():
            return

        # Use base class method for basic checks
        if not self.should_process(self.accumulated_data, self.trigger_config.min_text_length):
            return

        text_length = len(self.accumulated_data)
        current_time = time.time()

        # Calculate conditions for both time and new text triggers
        new_text_since_last_summary = text_length - self.text_length_at_last_summary
        time_since_last_summary = current_time - self.last_processing_time

        # Check both trigger conditions
        has_been_long_enough = time_since_last_summary > self.trigger_config.summary_interval_seconds
        has_enough_new_text = new_text_since_last_summary >= self.trigger_config.new_text_trigger_chars

        # Trigger inference based on OR logic - either condition can trigger
        if has_enough_new_text or has_been_long_enough:
            trigger_reason = "new_text" if has_enough_new_text else "time_interval"

            # Use base class method to queue for processing
            if self.queue_for_processing(trigger_reason):
                logger.debug(f"Queued inference request: {trigger_reason}")

    async def start_monitoring(self):
        """Start monitoring for inference triggers."""
        await self.start_worker()
        logger.info("Started LLM inference monitoring")

    async def stop_monitoring(self):
        """Stop monitoring."""
        await self.stop_worker()
        logger.info("Stopped LLM inference monitoring")

    async def _generate_inference(self, trigger_reason: str = "manual"):
        """Generate inference using the LLM.

        Args:
            trigger_reason: Reason for triggering the inference ('new_text', 'text_length', 'forced', or 'manual')
        """
        if not self.accumulated_data.strip():
            return

        try:
            # CHANGE: Always process the entire accumulated text for comprehensive summaries
            # This ensures summaries cover the full conversation context, not just incremental updates
            text_to_process = self.accumulated_data.strip()

            if trigger_reason == "forced":
                logger.info(f"Processing entire accumulated text for final comprehensive summary: {len(text_to_process)} chars")
            else:
                logger.info(f"Processing entire accumulated text for comprehensive summary: {len(text_to_process)} chars")

            # Truncate if too long
            if len(text_to_process) > self.trigger_config.max_text_length:
                text_to_process = truncate_text(text_to_process, self.trigger_config.max_text_length)
                logger.info(f"Truncated text to {self.trigger_config.max_text_length} characters")

            start_time = time.time()
            if trigger_reason == "forced":
                logger.info(f"Generating comprehensive final inference for {len(text_to_process)} chars...")
            else:
                logger.info(f"Generating comprehensive inference for {len(text_to_process)} chars of entire transcript...")

            # Prepare context information
            lines = text_to_process.split("\n")
            has_speakers = "[Speaker" in text_to_process
            duration_estimate = len(text_to_process) / 10  # Rough estimate: 10 chars per second

            # Generate structured response using universal LLM client
            response: LLMResponse = await self.llm_client.invoke_structured(
                self.prompt,
                {
                    "transcription": text_to_process,
                    "duration": duration_estimate,
                    "has_speakers": has_speakers,
                    "num_lines": len(lines),
                },
                LLMResponse,
            )

            generation_time = time.time() - start_time

            # Apply s2hk conversion to ensure Traditional Chinese output
            response.summary = s2hk(response.summary)
            response.key_points = [s2hk(point) for point in response.key_points]

            # Update statistics
            self.stats.record_inference(trigger_reason, generation_time, len(text_to_process))

            self.last_inference = response

            logger.info(f"Generated inference in {generation_time:.2f}s: {len(response.summary)} chars")
            logger.debug(f"Inference: {response.summary}")

            # Call registered callbacks
            for callback in self.inference_callbacks:
                try:
                    await callback(response, text_to_process)
                except Exception as e:
                    logger.error(f"Error in inference callback: {e}")

            # Reset text length tracking for new text trigger after processing
            if trigger_reason != "forced":
                self.text_length_at_last_summary = len(self.accumulated_data)
                logger.debug(f"Reset text length tracking to {self.text_length_at_last_summary} chars")

            # Update last_processed_data to the current accumulated text
            if trigger_reason != "forced":
                self.last_processed_data = self.accumulated_data

                # Keep the most recent text in buffer (last 5000 chars) to maintain context
                if len(self.accumulated_data) > 5000:
                    self.accumulated_data = self.accumulated_data[-5000:]
                    self.last_processed_data = self.accumulated_data
                    # Update text length tracking after truncation
                    self.text_length_at_last_summary = len(self.accumulated_data)

        except Exception as e:
            logger.error(f"Failed to generate inference: {e}")

    def get_last_inference(self) -> Optional[LLMResponse]:
        """Get the most recent inference."""
        return self.last_inference

    def get_last_summary(self) -> Optional[LLMResponse]:
        """Legacy method for backward compatibility. Use get_last_inference instead."""
        return self.get_last_inference()

    def get_stats(self) -> Dict[str, Any]:
        """Get inference statistics."""
        base_stats = self.get_queue_status()
        inference_stats = self.stats.to_dict()
        return {**base_stats, **inference_stats}

    async def force_inference(self) -> Optional[LLMResponse]:
        """Force generate a inference of current accumulated text."""
        if not self.accumulated_data.strip():
            return None

        # Force inference bypasses the queue system
        await self._generate_inference(trigger_reason="forced")
        return self.last_inference

    async def force_summary(self) -> Optional[LLMResponse]:
        """Legacy method for backward compatibility. Use force_inference instead."""
        return await self.force_inference()
