import time
from typing import Any, Dict, List, Optional

from langchain.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from ..logging_config import get_logger
from .llm_base import UniversalLLM
from .llm_config import LLMConfig, LLMTrigger
from .llm_utils import truncate_text
from .performance_monitor import get_performance_monitor, log_performance_if_needed
from .retriever import prepare_rag_context

logger = get_logger(__name__)


# =============================================================================
# Type Definitions for LLM Operations
# =============================================================================


class AnalyzerResponse(BaseModel):
    """Structured response from the LLM inference."""

    key_points: List[str] = Field(default_factory=list, description="Key topics, main ideas, and summary of the analyzed text.")
    response_suggestions: List[str] = Field(default_factory=list, description="Insightful questions for clarification, deeper understanding, or further discussion based on the text.")
    action_plan: List[str] = Field(default_factory=list, description="Suggested next steps, research topics, or actions to take based on the text's content.")


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
# Simplified Analyzer Implementation
# =============================================================================


class Analyzer:
    """
    Simplified LLM-powered analysis processor for generating insights from transcription data.

    Uses direct await calls instead of complex queue system for predictable time-ordered processing.
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

        # Analysis state
        self.last_inference = None
        self.stats = AnalyzerStats()
        self.inference_callbacks = []

        # Tracking for trigger conditions
        self.text_length_at_last_analysis = 0
        self.last_processed_data = ""

        # Processing state
        self.is_processing = False
        self.last_processing_time = 0.0
        self.last_completion_time = time.time()

        logger.info("Analyzer initialized with direct await processing")

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
                    """Based on the provided text:
1.  Identify and summarize the **Key Points**.
2.  Formulate **Insightful Questions** that could lead to deeper understanding, clarification, or further discussion.
3.  Suggest potential **Follow-up Actions** or areas for further exploration.

Always respond in the same language and script as the transcription. If a knowledge base is provided, use it to enrich your analysis and questions.""",
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
                    """Please analyze the following text:

Text:
{transcription}

Provide a structured analysis with:
1.  Key Points - A concise summary of the main ideas, topics, and arguments presented in the text.
2.  Insightful Questions - Thought-provoking questions that arise from the text. These could be for clarification, to explore related concepts, or to challenge assumptions.
3.  Follow-up Actions - Suggestions for what one might do next after reading this text, such as further research topics, related content to explore, or practical steps to take.

If a knowledge base or additional context was provided, please incorporate it into your analysis.
Remember to respond in the same language, script, and regional conventions as the transcription provided.""",
                )
            )

            self._prompt = ChatPromptTemplate.from_messages(prompt_messages)
            logger.debug("Lazy-initialized prompt template with dynamic RAG context")
        return self._prompt

    def add_inference_callback(self, callback):
        """Add a callback function to be called when inference is complete.

        Args:
            callback: Function that takes (AnalyzerResponse, trigger_reason) as arguments
        """
        self.inference_callbacks.append(callback)
        logger.debug(f"Added inference callback, total callbacks: {len(self.inference_callbacks)}")

    async def update_transcription_direct(self, new_text: str, speaker_info: Optional[Any] = None):
        """
        Update transcription and potentially trigger analysis using direct await.

        Args:
            new_text: New transcription text
            speaker_info: Optional speaker information
        """
        if not new_text or not new_text.strip():
            return

        # Check if we should trigger analysis
        should_analyze = self._check_inference_triggers(new_text)

        if should_analyze:
            try:
                # Generate analysis directly with await
                result = await self._generate_inference_direct("auto_trigger")

                # Call callbacks with result only if we have a valid result
                if result is not None:
                    for callback in self.inference_callbacks:
                        try:
                            await callback(result, "auto_trigger")
                        except Exception as e:
                            logger.error(f"Error in inference callback: {e}")

            except Exception as e:
                logger.error(f"Error in direct analysis: {e}")

        # Update tracking
        self.last_processed_data = new_text
        self.text_length_at_last_analysis = len(new_text)

    def _check_inference_triggers(self, new_text: str) -> bool:
        """Check if inference should be triggered based on current conditions."""
        current_time = time.time()

        # Time-based trigger
        time_since_last = current_time - self.last_completion_time
        if time_since_last >= self.trigger_config.analysis_interval_seconds:
            logger.debug(f"Time trigger: {time_since_last:.1f}s >= {self.trigger_config.analysis_interval_seconds}s")
            return True

        # Text length trigger
        if not self.last_processed_data:
            logger.debug("First text trigger")
            return True

        text_growth = len(new_text) - len(self.last_processed_data)
        if text_growth >= self.trigger_config.new_text_trigger_chars:
            logger.debug(f"Text growth trigger: {text_growth} chars >= {self.trigger_config.new_text_trigger_chars}")
            return True

        # Significant content change trigger
        if len(new_text) > 0 and len(self.last_processed_data) > 0:
            # Simple heuristic: if new text is significantly different
            similarity_ratio = len(set(new_text.split()) & set(self.last_processed_data.split())) / max(len(set(new_text.split())), 1)
            if similarity_ratio < 0.7:  # Less than 70% word overlap
                logger.debug(f"Content change trigger: similarity {similarity_ratio:.2f} < 0.7")
                return True

        return False

    async def _generate_inference_direct(self, trigger_reason: str = "manual") -> Optional[AnalyzerResponse]:
        """
        Generate inference directly with await (simplified from queue-based processing).

        Args:
            trigger_reason: Reason for triggering the inference

        Returns:
            AnalyzerResponse or None if processing fails
        """
        if self.is_processing:
            logger.debug("Analysis already in progress, skipping")
            return None

        self.is_processing = True
        start_time = time.time()
        monitor = get_performance_monitor()

        try:
            # Use the accumulated text for analysis
            text_to_analyze = self.last_processed_data

            if not text_to_analyze or not text_to_analyze.strip():
                logger.debug("No text to analyze")
                return None

            # Truncate text if too long for the model
            truncated_text = truncate_text(text_to_analyze, max_length=3000)
            if len(truncated_text) < len(text_to_analyze):
                logger.debug(f"Truncated text from {len(text_to_analyze)} to {len(truncated_text)} chars")

            # Generate analysis using LLM
            logger.debug(f"Generating analysis for {len(truncated_text)} characters (trigger: {trigger_reason})")

            result = await self.llm_client.invoke_structured(self.prompt, {"transcription": truncated_text}, AnalyzerResponse)

            # Update state
            self.last_inference = result
            processing_time = time.time() - start_time
            self.last_processing_time = processing_time
            self.last_completion_time = time.time()

            # Record statistics
            self.stats.record_inference(trigger_reason, processing_time, len(truncated_text))

            # Log performance
            log_performance_if_needed()

            logger.info(f"Generated analysis in {processing_time:.2f}s: {len(result.key_points)} key points, {len(result.response_suggestions)} questions, {len(result.action_plan)} actions")

            return result

        except Exception as e:
            logger.error(f"Failed to generate analysis: {e}")
            return None

        finally:
            self.is_processing = False

    async def force_inference_direct(self) -> Optional[AnalyzerResponse]:
        """
        Force inference generation directly with await.

        Returns:
            AnalyzerResponse or None if processing fails
        """
        logger.info("Forcing direct analysis generation")
        result = await self._generate_inference_direct("forced")

        if result is not None:
            # Call callbacks with result
            for callback in self.inference_callbacks:
                try:
                    await callback(result, "forced")
                except Exception as e:
                    logger.error(f"Error in forced inference callback: {e}")

        return result

    def get_last_inference(self) -> Optional[AnalyzerResponse]:
        """Get the last generated inference result."""
        return self.last_inference

    def get_stats(self) -> Dict[str, Any]:
        """Get analyzer statistics."""
        base_stats = self.stats.to_dict()
        base_stats.update(
            {
                "is_processing": self.is_processing,
                "last_processing_time": self.last_processing_time,
                "text_length_at_last_analysis": self.text_length_at_last_analysis,
                "trigger_config": {
                    "analysis_interval_seconds": self.trigger_config.analysis_interval_seconds,
                    "new_text_trigger_chars": self.trigger_config.new_text_trigger_chars,
                },
            }
        )
        return base_stats
