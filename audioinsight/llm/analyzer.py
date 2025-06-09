import time
from typing import Any, Dict, List, Optional

from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from ..logging_config import get_logger
from .llm_base import EventBasedProcessor, UniversalLLM
from .llm_config import LLMConfig, LLMTrigger
from .llm_utils import s2hk, truncate_text
from .performance_monitor import get_performance_monitor, log_performance_if_needed
from .retriever import prepare_rag_context

logger = get_logger(__name__)


# =============================================================================
# Type Definitions for LLM Operations
# =============================================================================


class AnalyzerResponse(BaseModel):
    """Structured response from the LLM inference."""

    key_points: List[str] = Field(default_factory=list, description="Main points discussed")
    response_suggestions: List[str] = Field(default_factory=list, description="Suggested responses to the speaker")
    action_plan: List[str] = Field(default_factory=list, description="Recommended actions or next steps")


class AnalyzerStats:
    """Statistics tracking for LLM operations."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all statistics."""
        self.inference_generated = 0
        self.total_text_processed = 0
        self.average_inference_time = 0.0
        self.inference_by_time_interval = 0
        self.inference_by_new_text = 0
        self.inference_by_forced = 0

    def record_inference(self, trigger_reason: str, processing_time: float, text_length: int):
        """Record an inference operation.

        Args:
            trigger_reason: Reason for triggering the inference
            processing_time: Time taken to process
            text_length: Length of text processed
        """
        self.inference_generated += 1
        self.total_text_processed += text_length

        # Update average time
        prev_avg = self.average_inference_time
        count = self.inference_generated
        self.average_inference_time = (prev_avg * (count - 1) + processing_time) / count

        # Track trigger reason statistics
        if trigger_reason == "time_interval":
            self.inference_by_time_interval += 1
        elif trigger_reason == "new_text":
            self.inference_by_new_text += 1
        elif trigger_reason == "forced":
            self.inference_by_forced += 1

    def to_dict(self) -> Dict[str, Any]:
        """Get statistics as a dictionary."""
        return {
            "inference_generated": self.inference_generated,
            "total_text_processed": self.total_text_processed,
            "average_inference_time": self.average_inference_time,
            "inference_by_time_interval": self.inference_by_time_interval,
            "inference_by_new_text": self.inference_by_new_text,
            "inference_by_forced": self.inference_by_forced,
        }


# =============================================================================
# LLM Analyzer Implementation
# =============================================================================


class Analyzer(EventBasedProcessor):
    """
    LLM-powered analysis processor for generating insights from transcription data.

    Features:
    - Event-based triggering (no polling)
    - Adaptive frequency matching LLM processing times
    - Non-blocking queue-based processing
    - Work coordination to prevent duplicate analyses
    """

    def __init__(
        self,
        model_id: str = "openai/gpt-4.1-mini",
        api_key: Optional[str] = None,
        trigger_config: Optional[LLMTrigger] = None,
    ):
        self.model_id = model_id
        self.api_key = api_key
        self.trigger_config = trigger_config or LLMTrigger()

        # Initialize LLM client
        llm_config = LLMConfig(model_id=model_id, api_key=api_key)
        self._llm_client = UniversalLLM(llm_config)

        # Initialize prompt template for lazy loading
        self._prompt = None

        # Initialize with 2 workers for analysis (stateless processing)
        # Increase queue size for better throughput
        super().__init__(queue_maxsize=50, cooldown_seconds=5.0, max_concurrent_workers=2, enable_work_coordination=True)

        # Analysis state
        self.last_inference = None
        self.stats = AnalyzerStats()
        self.inference_callbacks = []

        # Tracking for trigger conditions
        self.text_length_at_last_analysis = 0
        self.last_processed_data = ""

        # Add flag to prevent multiple simultaneous triggers
        self._trigger_in_progress = False

        logger.info(f"Analyzer initialized with {self.max_concurrent_workers} workers and work coordination")

    def _is_stateful_processor(self) -> bool:
        """Mark this processor as stateless to allow multiple workers with coordination."""
        return False  # Analyzer is mostly stateless, just uses coordination for deduplication

    @property
    def llm_client(self) -> UniversalLLM:
        """Lazy initialization of LLM client to speed up startup."""
        return self._llm_client

    @property
    def prompt(self) -> ChatPromptTemplate:
        """Lazy initialization of prompt template to speed up startup."""
        if self._prompt is None:
            # Start with base system message
            prompt_messages = [
                (
                    "system",
                    """You are an expert at analyzing speech transcriptions. 

Analyze the transcription and provide:
1. Key Points - Main topics and important information mentioned  
2. Response Suggestions - How to engage with the speaker
3. Action Plan - Recommended next steps

Always respond in the same language and script as the transcription.""",
                )
            ]

            # Add RAG context as system message if available
            rag_context = prepare_rag_context()
            if rag_context and rag_context.strip():
                prompt_messages.append(
                    (
                        "system",
                        rag_context,
                    )
                )
                logger.debug(f"Added RAG context to prompt: {len(rag_context)} chars")

            # Add human message
            prompt_messages.append(
                (
                    "human",
                    """Please analyze this transcription and provide response guidance:

Transcription:
{transcription}

Provide a structured analysis with:
1. Key Points - Main topics and important information mentioned
2. Response Suggestions - How to appropriately respond to the speaker(s)
3. Action Plan - Recommended next steps or actions to take

Remember to respond in the same language, script, and regional conventions as the transcription above.""",
                )
            )

            self._prompt = ChatPromptTemplate.from_messages(prompt_messages)
            logger.debug("Lazy-initialized prompt template with dynamic RAG context")
        return self._prompt

    async def _process_item(self, item: Any):
        """Process a single inference request from the queue.

        Args:
            item: The trigger reason string
        """
        start_time = time.time()
        monitor = get_performance_monitor()

        try:
            trigger_reason = item
            await self._generate_inference(trigger_reason=trigger_reason)
            self.update_processing_time()

            # Record successful processing
            processing_time = time.time() - start_time
            monitor.record_request("analyzer", processing_time)

            # Log performance periodically
            log_performance_if_needed()

        except Exception as e:
            # Record error
            processing_time = time.time() - start_time
            if "timeout" in str(e).lower():
                monitor.record_error("analyzer", "timeout")
            else:
                monitor.record_error("analyzer", "general")
            logger.error(f"Analyzer processing failed after {processing_time:.2f}s: {e}")
            raise

    def add_inference_callback(self, callback):
        """Add a callback function to be called when inference is generated.

        Args:
            callback: Function that takes (inference_response, transcription_text) as arguments
        """
        self.inference_callbacks.append(callback)

    def add_analysis_callback(self, callback):
        """Legacy method for backward compatibility. Use add_inference_callback instead."""
        self.add_inference_callback(callback)

    def update_transcription(self, new_text: str, speaker_info: Optional[Dict] = None):
        """Update with new transcription text - COMPLETELY NON-BLOCKING.

        Args:
            new_text: New transcription text to add
            speaker_info: Optional speaker/diarization information (ignored, not passed to LLM)
        """
        if not new_text.strip():
            return

        try:
            # Add to accumulated text without speaker info formatting
            if self.accumulated_data:
                self.accumulated_data += " " + new_text
            else:
                self.accumulated_data = new_text

            logger.debug(f"Updated transcription: {len(self.accumulated_data)} chars total")

            # CRITICAL: Defer inference checking to avoid blocking transcription
            # Schedule inference check for next event loop iteration
            try:
                import asyncio

                loop = asyncio.get_event_loop()
                loop.call_soon(self._check_inference_triggers)
            except Exception:
                # If we can't schedule, just skip inference checking to avoid blocking
                pass

        except Exception as e:
            # Never let transcription updates fail or block
            logger.debug(f"Non-critical transcription update error: {e}")

    def _check_inference_triggers(self):
        """Check if conditions are met for inference and queue request if needed - NON-BLOCKING."""
        if not self.accumulated_data.strip():
            return

        # Prevent multiple simultaneous triggers
        if self._trigger_in_progress:
            logger.debug("Skipping trigger check - another trigger already in progress")
            return

        # Check minimum text length requirement only (skip adaptive cooldown for analyzer)
        if len(self.accumulated_data) < self.trigger_config.min_text_length:
            return

        text_length = len(self.accumulated_data)
        current_time = time.time()

        # Calculate conditions for both time and new text triggers
        new_text_since_last_analysis = text_length - self.text_length_at_last_analysis
        time_since_last_analysis = current_time - self.last_processing_time

        # Check both trigger conditions
        has_been_long_enough = time_since_last_analysis > self.trigger_config.analysis_interval_seconds
        has_enough_new_text = new_text_since_last_analysis >= self.trigger_config.new_text_trigger_chars

        # Add debug logging for trigger analysis
        logger.debug(f"📊 Trigger check: {text_length} chars total, {new_text_since_last_analysis} new chars, {time_since_last_analysis:.1f}s elapsed")
        logger.debug(f"📊 Thresholds: {self.trigger_config.new_text_trigger_chars} chars, {self.trigger_config.analysis_interval_seconds}s")
        logger.debug(f"📊 Conditions: enough_text={has_enough_new_text}, enough_time={has_been_long_enough}")

        # Trigger inference based on OR logic - either condition can trigger
        if has_enough_new_text or has_been_long_enough:
            trigger_reason = "new_text" if has_enough_new_text else "time_interval"

            # Set flag to prevent multiple triggers
            self._trigger_in_progress = True

            # Use base class method to queue for processing - NON-BLOCKING
            try:
                # Create a task to handle the async queue operation
                import asyncio

                loop = asyncio.get_event_loop()
                loop.create_task(self._queue_inference_async(trigger_reason))
                logger.info(f"🎯 Triggering analysis: {trigger_reason} (new_chars={new_text_since_last_analysis}, time={time_since_last_analysis:.1f}s)")
            except Exception as e:
                # Don't let inference errors block transcription
                logger.debug(f"Non-critical inference queue error: {e}")
                # Reset flag on error
                self._trigger_in_progress = False
        else:
            logger.debug(f"⏳ Not triggering analysis yet: need {self.trigger_config.new_text_trigger_chars - new_text_since_last_analysis} more chars or {self.trigger_config.analysis_interval_seconds - time_since_last_analysis:.1f}s")

    async def _queue_inference_async(self, trigger_reason: str):
        """Async helper to queue inference requests."""
        try:
            success = await self.queue_for_processing(trigger_reason)
            if success:
                logger.debug(f"Successfully queued inference: {trigger_reason}")
            else:
                logger.debug(f"Inference queue full for: {trigger_reason}")
        except Exception as e:
            logger.debug(f"Error queuing inference: {e}")
        finally:
            # Always reset trigger flag when queue operation completes
            if hasattr(self, "_trigger_in_progress"):
                self._trigger_in_progress = False

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
            # Reset trigger flag even if no data
            if hasattr(self, "_trigger_in_progress"):
                self._trigger_in_progress = False
            return

        try:
            # CHANGE: Always process the entire accumulated text for comprehensive analyses
            # This ensures analyses cover the full conversation context, not just incremental updates
            text_to_process = self.accumulated_data.strip()

            if trigger_reason == "forced":
                logger.info(f"Processing entire accumulated text for final comprehensive analysis: {len(text_to_process)} chars")
            else:
                logger.info(f"Processing entire accumulated text for comprehensive analysis: {len(text_to_process)} chars")

            # Truncate if too long
            if len(text_to_process) > self.trigger_config.max_text_length:
                text_to_process = truncate_text(text_to_process, self.trigger_config.max_text_length)
                logger.info(f"Truncated text to {self.trigger_config.max_text_length} characters")

            start_time = time.time()
            if trigger_reason == "forced":
                logger.info(f"Generating comprehensive final inference for {len(text_to_process)} chars...")
            else:
                logger.info(f"Generating comprehensive inference for {len(text_to_process)} chars of entire transcript...")

            # Generate structured response using universal LLM client
            variables = {"transcription": text_to_process}
            response: AnalyzerResponse = await self.llm_client.invoke_structured(self.prompt, variables, AnalyzerResponse)

            generation_time = time.time() - start_time

            # Apply s2hk conversion to ensure Traditional Chinese output
            response.key_points = [s2hk(point) for point in response.key_points]
            response.response_suggestions = [s2hk(suggestion) for suggestion in response.response_suggestions]
            response.action_plan = [s2hk(action) for action in response.action_plan]

            # Update statistics
            self.stats.record_inference(trigger_reason, generation_time, len(text_to_process))

            self.last_inference = response

            logger.info(f"Generated inference in {generation_time:.2f}s: {len(response.key_points)} chars")
            logger.debug(f"Inference: {response.key_points}")

            # Call registered callbacks
            for callback in self.inference_callbacks:
                try:
                    await callback(response, text_to_process)
                except Exception as e:
                    logger.error(f"Error in inference callback: {e}")

            # Reset text length tracking for new text trigger after processing
            if trigger_reason != "forced":
                self.text_length_at_last_analysis = len(self.accumulated_data)
                logger.debug(f"Reset text length tracking to {self.text_length_at_last_analysis} chars")

                # Update last_processed_data to the current accumulated text
                self.last_processed_data = self.accumulated_data

                # Keep the most recent text in buffer (last 5000 chars) to maintain context
                if len(self.accumulated_data) > 5000:
                    self.accumulated_data = self.accumulated_data[-5000:]
                    self.last_processed_data = self.accumulated_data
                    # Update text length tracking after truncation
                    self.text_length_at_last_analysis = len(self.accumulated_data)

        except Exception as e:
            logger.error(f"Failed to generate inference: {e}")
        finally:
            # Always reset trigger flag when processing completes
            if hasattr(self, "_trigger_in_progress"):
                self._trigger_in_progress = False

    def get_last_inference(self) -> Optional[AnalyzerResponse]:
        """Get the most recent inference."""
        return self.last_inference

    def get_last_analysis(self) -> Optional[AnalyzerResponse]:
        """Legacy method for backward compatibility. Use get_last_inference instead."""
        return self.get_last_inference()

    def get_stats(self) -> Dict[str, Any]:
        """Get inference statistics."""
        base_stats = self.get_queue_status()
        inference_stats = self.stats.to_dict()
        return {**base_stats, **inference_stats}

    async def force_inference(self) -> Optional[AnalyzerResponse]:
        """Force generate a inference of current accumulated text."""
        if not self.accumulated_data.strip():
            return None

        # Force inference bypasses the queue system
        await self._generate_inference(trigger_reason="forced")
        return self.last_inference

    async def force_analysis(self) -> Optional[AnalyzerResponse]:
        """Legacy method for backward compatibility. Use force_inference instead."""
        return await self.force_inference()
