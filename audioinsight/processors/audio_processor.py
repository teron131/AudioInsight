import asyncio
import traceback
from time import time
from typing import Optional

from ..llm import Analyzer, LLMTrigger, ParsedTranscript, Parser, ParserConfig
from ..main import AudioInsight
from ..timed_objects import ASRToken
from .base_processor import BaseProcessor, format_time, logger, s2hk
from .diarization_processor import DiarizationProcessor
from .ffmpeg_processor import FFmpegProcessor
from .formatter import Formatter
from .transcription_processor import TranscriptionProcessor


class AudioProcessor(BaseProcessor):
    """
    Coordinates audio processing for transcription and diarization.
    Manages shared state and coordinates specialized processors.
    """

    # Class-level warm component cache to share across instances
    _warm_components = {"whisper_loaded": False, "llm_clients_warmed": False, "workers_prestarted": False}

    def __init__(self):
        """Initialize the audio processor with configuration, models, and state."""

        models = AudioInsight()

        # Audio processing settings - cache computed values
        self.args = models.args

        # FFmpeg state - these were missing and are needed for the watchdog!
        self.sample_rate = 16000
        self.channels = 1
        self.bytes_per_sample = 2
        self.samples_per_sec = int(self.sample_rate * self.args.min_chunk_size)
        self.bytes_per_sec = self.samples_per_sec * self.bytes_per_sample
        self.max_bytes_per_sec = 32000 * 5  # 5 seconds of audio at 32 kHz
        self.sample_rate_str = str(self.sample_rate)  # Cache string conversion

        # Timing settings - needed for watchdog monitoring
        self.last_ffmpeg_activity = time()
        self.ffmpeg_health_check_interval = 10  # Less frequent health checks
        self.ffmpeg_max_idle_time = 30  # Allow much longer idle time before restart

        # State management - SINGLE SOURCE OF TRUTH
        self.is_stopping = False
        self.lock = asyncio.Lock()
        self.beg_loop = time()
        self.sep = " "  # Default separator
        self.last_response_content = ""

        # SINGLE GLOBAL MEMORY STORE - all workers access this atomically
        self.global_transcript = {
            "committed_tokens": [],  # Committed ASR tokens
            "current_buffer": "",  # Current ASR buffer text
            "parsed_content": "",  # Processed/parsed text
            "end_buffer": 0,  # Buffer end time
            "end_attributed_speaker": 0,  # Diarization progress
        }

        # Legacy fields for compatibility - will be populated from global_transcript
        self.tokens = []
        self.buffer_transcription = ""
        self.buffer_diarization = ""
        self.full_transcription = ""
        self.end_buffer = 0
        self.end_attributed_speaker = 0

        self.analyses = []  # Initialize analyses list
        self._has_analyses = False  # Efficient flag to track analysis availability
        self._last_analysis_check = 0  # Timestamp of last analysis check

        # Initialize processor references as None - will be created on demand
        self.ffmpeg_processor = None
        self.transcription_processor = None
        self.diarization_processor = None
        self.formatter = Formatter(self.args)

        # Processing queues - will be created on demand
        self.transcription_queue = None
        self.diarization_queue = None

        # Task references
        self.transcription_task = None
        self.diarization_task = None
        self.ffmpeg_reader_task = None
        self.watchdog_task = None
        self.all_tasks_for_cleanup = []

        # Initialize LLM inference processor early if enabled
        self.llm = None
        self.llm_task = None
        self._final_analysis_generated = False  # Flag to prevent multiple final analyses

        # Initialize transcript parser for structured processing
        self.transcript_parser = None
        self.parsed_transcripts = []  # Store parsed transcript data
        self.last_parsed_transcript = None  # Most recent parsed transcript
        self._parser_enabled = True  # Enable transcript parsing by default
        self.last_parsed_text = ""  # Track what text has been parsed to avoid re-processing
        self.min_text_threshold = 100  # Variable: parse all if text < this many chars
        self.sentence_percentage = 0.40  # Variable: parse last 40% of sentences if text >= threshold

        # OPTIMIZATION: Pre-initialize LLM components for faster first connection
        self._initialize_llm_components()

        logger.info("🔧 AudioProcessor initialized - components will be created on demand")

    def _initialize_llm_components(self):
        """Pre-initialize LLM components to reduce first-connection latency."""
        # Initialize LLM if enabled in features
        if getattr(self.args, "llm_inference", False):
            logger.info("🔧 Pre-initializing LLM processor to reduce connection latency")
            try:
                trigger_config = LLMTrigger(
                    max_text_length=getattr(self.args, "analyzer_max_input_length", 8000),
                    analysis_interval_seconds=getattr(self.args, "llm_analysis_interval", 5.0),
                    new_text_trigger_chars=getattr(self.args, "llm_new_text_trigger", 50),
                )

                self.llm = Analyzer(
                    model_id=getattr(self.args, "base_llm", "openai/gpt-4.1-mini"),
                    trigger_config=trigger_config,
                )
                self.llm.add_inference_callback(self._handle_inference_callback)
                logger.info(f"LLM inference pre-initialized with model: {getattr(self.args, 'base_llm', 'openai/gpt-4.1-mini')}")
            except Exception as e:
                logger.warning(f"Failed to pre-initialize LLM inference processor: {e}")
                self.llm = None

        # Initialize transcript parser
        try:
            parser_config = ParserConfig(model_id=getattr(self.args, "fast_llm", "openai/gpt-4.1-nano"), max_output_tokens=getattr(self.args, "parser_output_tokens", 33000), trigger_interval_seconds=getattr(self.args, "parser_trigger_interval", 1.0))
            self.transcript_parser = Parser(config=parser_config)
            logger.info(f"Transcript parser pre-initialized with model: {getattr(self.args, 'fast_llm', 'openai/gpt-4.1-nano')}, max_output_tokens: {getattr(self.args, 'parser_output_tokens', 33000)}, trigger interval: {getattr(self.args, 'parser_trigger_interval', 1.0)}s")
        except Exception as e:
            logger.warning(f"Failed to pre-initialize transcript parser: {e}")
            self.transcript_parser = None

    @classmethod
    async def warm_up_system(cls):
        """Class method to warm up shared system components for faster instance creation."""
        if cls._warm_components["whisper_loaded"]:
            return  # Already warmed up

        logger.info("🔥 Starting system warm-up to reduce latency...")
        start_time = time()

        try:
            # Pre-load Whisper model if not already loaded
            if not cls._warm_components["whisper_loaded"]:
                models = AudioInsight()
                if hasattr(models, "asr") and models.asr:
                    logger.info("✅ Whisper model already loaded during system startup")
                cls._warm_components["whisper_loaded"] = True

            # Pre-warm LLM clients for faster first requests
            if not cls._warm_components["llm_clients_warmed"]:
                await cls._warm_llm_clients()
                cls._warm_components["llm_clients_warmed"] = True

            warm_time = time() - start_time
            logger.info(f"🔥 System warm-up completed in {warm_time:.2f}s")

        except Exception as e:
            logger.warning(f"System warm-up failed: {e}")

    @classmethod
    async def _warm_llm_clients(cls):
        """Pre-warm LLM clients with dummy requests for faster first real requests."""
        try:
            # Import here to avoid circular imports
            from ..llm.base import UniversalLLM
            from ..llm.config import LLMConfig

            # Warm up both fast and base LLM models
            models_to_warm = ["openai/gpt-4.1-nano", "openai/gpt-4.1-mini"]  # Fast model for parsing  # Base model for analysis

            for model_id in models_to_warm:
                try:
                    llm_config = LLMConfig(model_id=model_id, timeout=5.0)
                    llm = UniversalLLM(llm_config)

                    # Make a minimal warm-up request to establish connection
                    from langchain.prompts import ChatPromptTemplate

                    warm_prompt = ChatPromptTemplate.from_messages([("human", "Hi")])

                    # Use asyncio.wait_for with timeout to avoid hanging
                    await asyncio.wait_for(llm.invoke_text(warm_prompt, {}), timeout=3.0)
                    logger.info(f"✅ Pre-warmed LLM client: {model_id}")

                except asyncio.TimeoutError:
                    logger.debug(f"LLM warm-up timeout for {model_id} (non-critical)")
                except Exception as e:
                    logger.debug(f"LLM warm-up failed for {model_id}: {e} (non-critical)")

        except Exception as e:
            logger.debug(f"LLM client warm-up failed: {e} (non-critical)")

    async def update_transcription(self, new_tokens, buffer, end_buffer, full_transcription, sep):
        """Thread-safe update of transcription with SINGLE GLOBAL MEMORY."""
        async with self.lock:
            # Update the single source of truth
            self.global_transcript["committed_tokens"].extend(new_tokens)
            self.global_transcript["current_buffer"] = buffer
            self.global_transcript["end_buffer"] = end_buffer

            # Update legacy fields for compatibility
            self.tokens.extend(new_tokens)
            self.buffer_transcription = buffer
            self.end_buffer = end_buffer
            self.full_transcription = full_transcription
            self.sep = sep

    async def update_diarization(self, end_attributed_speaker, buffer_diarization=""):
        """Thread-safe update of diarization with new data."""
        async with self.lock:
            self.end_attributed_speaker = end_attributed_speaker
            if buffer_diarization:
                self.buffer_diarization = buffer_diarization

    async def add_dummy_token(self):
        """Placeholder token when no transcription is available."""
        async with self.lock:
            current_time = time() - self.beg_loop
            self.tokens.append(ASRToken(start=current_time, end=current_time + 1, text=".", speaker=0, is_dummy=True))

    async def _handle_inference_callback(self, inference_response, transcription_text):
        """Handle callback when a inference result is generated."""
        logger.info(f"📝 Inference result generated: {len(inference_response.key_points)} key points")
        logger.info(f"🔑 Key points: {', '.join(inference_response.key_points[:2])}...")

        # Store inference result in the response for client access
        async with self.lock:
            if not hasattr(self, "analyses"):
                self.analyses = []

            # FIXED: Allow final analyses during stopping state - only block AFTER final analysis is generated
            if getattr(self, "_final_analysis_generated", False):
                logger.info(f"⚠️ Ignoring inference result - final analysis already generated")
                return

            # Store inference result
            new_result = {
                "timestamp": time(),
                "key_points": inference_response.key_points,
                "response_suggestions": inference_response.response_suggestions,
                "action_plan": inference_response.action_plan,
                "text_length": len(transcription_text),
            }

            self.analyses.append(new_result)
            self._has_analyses = True  # Set efficient flag
            logger.info(f"✅ Added new inference result (total: {len(self.analyses)})")

    async def parse_and_store_transcript(self, text: str, speaker_info: Optional[list] = None, timestamps: Optional[dict] = None) -> Optional[ParsedTranscript]:
        """Parse transcript text and store it for sharing across the application.

        Args:
            text: Transcript text to parse
            speaker_info: Optional speaker information
            timestamps: Optional timestamp information

        Returns:
            ParsedTranscript: Parsed transcript data or None if parsing disabled/failed
        """
        if not self._parser_enabled or not self.transcript_parser or not text.strip():
            return None

        try:
            # Parse the transcript using the structured parser
            parsed_transcript = await self.transcript_parser.parse_transcript(text, speaker_info, timestamps)

            # Store the parsed transcript
            async with self.lock:
                self.parsed_transcripts.append(parsed_transcript)
                self.last_parsed_transcript = parsed_transcript

                # Keep only the last 50 parsed transcripts to prevent memory growth
                if len(self.parsed_transcripts) > 50:
                    self.parsed_transcripts = self.parsed_transcripts[-50:]

            # Note: LLM analyzer is now triggered by parser worker to ensure proper event order
            logger.debug(f"📝 Parsed and stored transcript: {len(text)} -> {len(parsed_transcript.parsed_text)} chars")
            return parsed_transcript

        except Exception as e:
            logger.warning(f"Failed to parse transcript: {e}")
            return None

    def enable_transcript_parsing(self, enabled: bool = True):
        """Enable or disable transcript parsing.

        Args:
            enabled: Whether to enable transcript parsing
        """
        self._parser_enabled = enabled
        logger.info(f"Transcript parsing {'enabled' if enabled else 'disabled'}")

    def get_parsed_transcripts(self) -> list:
        """Get all parsed transcripts.

        Returns:
            List of ParsedTranscript objects
        """
        return self.parsed_transcripts.copy()

    def get_last_parsed_transcript(self) -> Optional[ParsedTranscript]:
        """Get the most recent parsed transcript.

        Returns:
            Most recent ParsedTranscript or None
        """
        return self.last_parsed_transcript

    async def get_current_state(self):
        """Get current state."""
        async with self.lock:
            current_time = time()

            # Calculate remaining times
            remaining_transcription = 0
            if self.end_buffer > 0:
                remaining_transcription = max(0, round(current_time - self.beg_loop - self.end_buffer, 2))

            # Only calculate remaining_diarization if diarization is enabled
            remaining_diarization = 0
            if self.args.diarization and self.tokens:
                latest_end = max(self.end_buffer, self.tokens[-1].end if self.tokens else 0)
                remaining_diarization = max(0, round(latest_end - self.end_attributed_speaker, 2))

            return {
                "tokens": self.tokens.copy(),
                "buffer_transcription": self.buffer_transcription,
                "buffer_diarization": self.buffer_diarization,
                "end_buffer": self.end_buffer,
                "end_attributed_speaker": self.end_attributed_speaker,
                "sep": self.sep,
                "remaining_time_transcription": remaining_transcription,
                "remaining_time_diarization": remaining_diarization,
            }

    async def reset(self):
        """Reset all state variables to initial values."""
        async with self.lock:
            # SINGLE SOURCE OF TRUTH - reset global memory
            self.global_transcript = {
                "committed_tokens": [],
                "current_buffer": "",
                "parsed_content": "",
                "end_buffer": 0,
                "end_attributed_speaker": 0,
            }

            # Reset legacy fields for compatibility
            self.tokens.clear()
            self.buffer_transcription = self.buffer_diarization = ""
            self.end_buffer = self.end_attributed_speaker = 0
            self.full_transcription = self.last_response_content = ""
            self.beg_loop = time()

            # Reset parsed transcript data
            self.parsed_transcripts.clear()
            self.last_parsed_transcript = None
            # Reset final analysis flag
            self._final_analysis_generated = False

        # Reset transcription processor's incremental parsing state
        if self.transcription_processor:
            await self.transcription_processor.reset_parsing_state()

    async def process_audio(self, message):
        """Process incoming audio data."""
        # If already stopping or stdin is closed, ignore further audio, especially residual chunks.
        if self.is_stopping or (self.ffmpeg_processor and self.ffmpeg_processor.ffmpeg_process and self.ffmpeg_processor.ffmpeg_process.stdin and self.ffmpeg_processor.ffmpeg_process.stdin.closed):
            logger.warning(f"AudioProcessor is stopping or stdin is closed. Ignoring incoming audio message (length: {len(message)}).")
            if not message and self.ffmpeg_processor and self.ffmpeg_processor.ffmpeg_process and self.ffmpeg_processor.ffmpeg_process.stdin and not self.ffmpeg_processor.ffmpeg_process.stdin.closed:
                logger.info("Received empty message while already in stopping state; ensuring stdin is closed.")
                try:
                    self.ffmpeg_processor.ffmpeg_process.stdin.close()
                except Exception as e:
                    logger.warning(f"Error closing ffmpeg stdin on redundant stop signal during stopping state: {e}")
            return

        if not message:  # primary signal to start stopping
            logger.info("Empty audio message received, initiating stop sequence.")
            self.is_stopping = True
            if self.ffmpeg_processor and self.ffmpeg_processor.ffmpeg_process and self.ffmpeg_processor.ffmpeg_process.stdin and not self.ffmpeg_processor.ffmpeg_process.stdin.closed:
                try:
                    self.ffmpeg_processor.ffmpeg_process.stdin.close()
                    logger.info("FFmpeg stdin closed due to primary stop signal.")
                except Exception as e:
                    logger.warning(f"Error closing ffmpeg stdin on stop: {e}")
            return

        await self.ffmpeg_processor.process_audio_chunk(message)

    async def create_tasks(self):
        """Create async tasks for audio processing and result formatting."""
        # OPTIMIZATION: Track task creation timing for performance monitoring
        task_creation_start = time()

        self.all_tasks_for_cleanup = []  # Reset task list
        processing_tasks_for_watchdog = []

        # Initialize components on demand if not already created
        if not self.ffmpeg_processor:
            logger.info("🔧 Creating FFmpeg processor on demand")
            self.ffmpeg_processor = FFmpegProcessor(self.args)
            self.last_ffmpeg_activity = self.ffmpeg_processor.last_ffmpeg_activity

        # Initialize transcription components if needed - NOW WITH CORRECT LLM REFERENCE
        if self.args.transcription:
            if not self.transcription_processor:
                logger.info("🔧 Creating transcription processor on demand")
                models = AudioInsight()
                self.transcription_processor = TranscriptionProcessor(self.args, models.asr, models.tokenizer, coordinator=self)

            if not self.transcription_queue:
                self.transcription_queue = asyncio.Queue()

            # OPTIMIZATION: Start parser workers in parallel with other tasks
            parser_start_task = None
            if self.transcript_parser:
                parser_start_task = asyncio.create_task(self.transcription_processor.start_parser_worker())
            else:
                # If no parser instance but LLM features are enabled, create it
                logger.warning("Transcript parser not found - re-initializing...")
                self._initialize_llm_components()
                if self.transcript_parser:
                    parser_start_task = asyncio.create_task(self.transcription_processor.start_parser_worker())

            # FIXED: Now self.llm is properly initialized before creating transcription task
            self.transcription_task = asyncio.create_task(self.transcription_processor.process(self.transcription_queue, self.update_transcription, self.llm))
            self.all_tasks_for_cleanup.append(self.transcription_task)
            processing_tasks_for_watchdog.append(self.transcription_task)

            # Wait for parser to start if it was created
            if parser_start_task:
                await parser_start_task
                logger.info("Started parser worker from transcript parser")

        # Initialize diarization components if needed
        if self.args.diarization:
            if not self.diarization_processor:
                logger.info("🔧 Creating diarization processor on demand")
                models = AudioInsight()
                self.diarization_processor = DiarizationProcessor(self.args, models.diarization)

            if not self.diarization_queue:
                self.diarization_queue = asyncio.Queue()

            self.diarization_task = asyncio.create_task(self.diarization_processor.process(self.diarization_queue, self.get_current_state, self.update_diarization))
            self.all_tasks_for_cleanup.append(self.diarization_task)
            processing_tasks_for_watchdog.append(self.diarization_task)

        self.ffmpeg_reader_task = asyncio.create_task(self.ffmpeg_processor.read_audio_data(self.transcription_queue, self.diarization_queue))
        self.all_tasks_for_cleanup.append(self.ffmpeg_reader_task)
        processing_tasks_for_watchdog.append(self.ffmpeg_reader_task)

        # Start LLM inference processor monitoring if enabled - ALWAYS start monitoring
        llm_start_task = None
        if self.llm:
            llm_start_task = asyncio.create_task(self.llm.start_monitoring())
            self.all_tasks_for_cleanup.append(llm_start_task)
            logger.info("LLM inference processor monitoring task started")
        else:
            # If no LLM instance, check if we should create one
            if getattr(self.args, "llm_inference", False):
                logger.warning("LLM inference enabled but no LLM instance - re-initializing...")
                self._initialize_llm_components()
                if self.llm:
                    llm_start_task = asyncio.create_task(self.llm.start_monitoring())
                    self.all_tasks_for_cleanup.append(llm_start_task)
                    logger.info("LLM inference processor monitoring task started after re-initialization")

        # Monitor overall system health
        self.watchdog_task = asyncio.create_task(self.watchdog(processing_tasks_for_watchdog))
        self.all_tasks_for_cleanup.append(self.watchdog_task)

        # OPTIMIZATION: Track total task creation time
        task_creation_time = time() - task_creation_start
        logger.info(f"⚡ All processing tasks created in {task_creation_time:.3f}s")

        return self.results_formatter()

    async def watchdog(self, tasks_to_monitor):
        """Monitors the health of critical processing tasks."""
        while True:
            try:
                await asyncio.sleep(15)  # Check every 15 seconds instead of 10
                current_time = time()

                for i, task in enumerate(tasks_to_monitor):
                    if task.done():
                        exc = task.exception()
                        task_name = task.get_name() if hasattr(task, "get_name") else f"Task {i}"
                        if exc:
                            logger.error(f"{task_name} failed: {exc}")
                        else:
                            logger.info(f"{task_name} completed normally")

                # Sync FFmpeg activity timing
                if self.ffmpeg_processor:
                    self.last_ffmpeg_activity = self.ffmpeg_processor.last_ffmpeg_activity
                    ffmpeg_idle_time = current_time - self.last_ffmpeg_activity

                    # Only log idle warnings every 60 seconds and if significant
                    if not hasattr(self, "_last_idle_warning"):
                        self._last_idle_warning = 0

                    if ffmpeg_idle_time > 120 and current_time - self._last_idle_warning > 60.0:
                        logger.warning(f"FFmpeg idle for {ffmpeg_idle_time:.1f}s")
                        self._last_idle_warning = current_time

                    # Much more conservative restart policy:
                    # - Only restart if idle for more than 5 minutes (300 seconds)
                    # - AND we're not in stopping state
                    # - AND the FFmpeg process appears to be actually broken (not just idle)
                    if ffmpeg_idle_time > 300 and not self.is_stopping:
                        # Additional check: only restart if FFmpeg process is actually broken
                        if self.ffmpeg_processor.ffmpeg_process is None or self.ffmpeg_processor.ffmpeg_process.poll() is not None:
                            logger.error(f"FFmpeg idle for {ffmpeg_idle_time:.1f}s and process is broken, forcing restart")
                            await self.ffmpeg_processor.restart_ffmpeg()
                            # Update timing after restart
                            self.last_ffmpeg_activity = self.ffmpeg_processor.last_ffmpeg_activity
                        else:
                            # FFmpeg is idle but process is still healthy - this is normal when no audio is coming in
                            logger.debug(f"FFmpeg idle for {ffmpeg_idle_time:.1f}s but process is healthy - normal idle state")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Watchdog error: {e}")

    async def cleanup(self):
        """Clean up resources when processing is complete."""
        logger.info("Starting cleanup of AudioProcessor resources.")

        # Stop parser worker first
        if self.transcription_processor:
            await self.transcription_processor.stop_parser_worker()

        # Stop LLM inference processor to generate final inference
        if self.llm:
            await self.llm.stop_monitoring()
            logger.info("LLM inference processor stopped")

        for task in self.all_tasks_for_cleanup:
            if task and not task.done():
                task.cancel()

        created_tasks = [t for t in self.all_tasks_for_cleanup if t]
        if created_tasks:
            await asyncio.gather(*created_tasks, return_exceptions=True)
        logger.info("All processing tasks cancelled or finished.")

        # Clean up specialized processors
        if self.ffmpeg_processor:
            await asyncio.get_event_loop().run_in_executor(None, self.ffmpeg_processor.cleanup)

        if self.diarization_processor:
            self.diarization_processor.cleanup()

        logger.info("AudioProcessor cleanup complete.")

    async def force_reset(self):
        """Force reset all resources and clear memory for a fresh session.

        This is more aggressive than cleanup() and ensures no memory leaks between sessions.
        """
        logger.info("🧹 Starting force reset of AudioProcessor...")

        # Stop everything aggressively
        self.is_stopping = True

        # Cancel all tasks immediately
        for task in self.all_tasks_for_cleanup:
            if task and not task.done():
                task.cancel()

        # Wait for tasks to complete cancellation
        if self.all_tasks_for_cleanup:
            try:
                await asyncio.gather(*[t for t in self.all_tasks_for_cleanup if t], return_exceptions=True)
            except Exception as e:
                logger.warning(f"Error waiting for task cancellation: {e}")

        # Clear all queues aggressively
        if self.transcription_queue:
            while not self.transcription_queue.empty():
                try:
                    self.transcription_queue.get_nowait()
                    self.transcription_queue.task_done()
                except:
                    break

        if self.diarization_queue:
            while not self.diarization_queue.empty():
                try:
                    self.diarization_queue.get_nowait()
                    self.diarization_queue.task_done()
                except:
                    break

        # Force cleanup all processors with proper cleanup
        if self.ffmpeg_processor:
            try:
                # Ensure FFmpeg cleanup happens synchronously and completely
                await asyncio.get_event_loop().run_in_executor(None, self.ffmpeg_processor.cleanup)
                # Reset the shutdown flag for the next session
                self.ffmpeg_processor._is_shutting_down = False
                # Wait a moment for cleanup to complete
                await asyncio.sleep(0.5)
            except Exception as e:
                logger.warning(f"Error cleaning up FFmpeg processor: {e}")
            self.ffmpeg_processor = None

        if self.transcription_processor:
            self.transcription_processor = None

        if self.diarization_processor:
            try:
                self.diarization_processor.cleanup()
            except Exception as e:
                logger.warning(f"Error cleaning up diarization processor: {e}")
            self.diarization_processor = None

        # CRITICAL FIX: Stop LLM monitoring but preserve the LLM instance for re-use
        if self.llm:
            try:
                await self.llm.stop_monitoring()
                # DO NOT set self.llm = None - keep the instance for re-use
                logger.info("LLM monitoring stopped but instance preserved for re-use")
            except Exception as e:
                logger.warning(f"Error stopping LLM monitoring: {e}")
                # Only null the LLM if there was an error stopping it
                self.llm = None

        # Similarly preserve transcript parser
        if self.transcript_parser:
            try:
                await self.transcript_parser.stop_worker()
                logger.info("Parser workers stopped but instance preserved for re-use")
            except Exception as e:
                logger.warning(f"Error stopping parser workers: {e}")
                # Only null if there was an error
                self.transcript_parser = None

        # Clear all memory buffers and state
        async with self.lock:
            self.tokens.clear()
            self.buffer_transcription = ""
            self.buffer_diarization = ""
            self.full_transcription = ""
            self.end_buffer = 0
            self.end_attributed_speaker = 0
            self.last_response_content = ""
            if hasattr(self, "analyses"):
                self.analyses.clear()
            self._has_analyses = False
            # Clear parsed transcript data
            if hasattr(self, "parsed_transcripts"):
                self.parsed_transcripts.clear()
            self.last_parsed_transcript = None

        # Clear task references
        self.transcription_task = None
        self.diarization_task = None
        self.ffmpeg_reader_task = None
        self.watchdog_task = None
        self.llm_task = None
        self.all_tasks_for_cleanup.clear()

        # Reset queues to None - will be recreated on demand
        self.transcription_queue = None
        self.diarization_queue = None

        # Reset timing and state flags
        self.beg_loop = time()
        self.is_stopping = False
        self._final_analysis_generated = False  # Reset final analysis flag

        # CRITICAL FIX: Re-initialize LLM components if they were lost during reset
        if not self.llm or not self.transcript_parser:
            logger.info("🔧 Re-initializing LLM components after reset...")
            self._initialize_llm_components()

        logger.info("🧹 Force reset completed - components will be re-initialized on demand")

    async def results_formatter(self):
        """Format processing results for output."""
        while True:
            try:
                # Get current state
                state = await self.get_current_state()
                tokens = state["tokens"]
                buffer_transcription = state["buffer_transcription"]
                buffer_diarization = state["buffer_diarization"]
                end_attributed_speaker = state["end_attributed_speaker"]
                sep = state["sep"]

                # Add dummy tokens if needed
                if (not tokens or tokens[-1].is_dummy) and not self.args.transcription and self.args.diarization:
                    await self.add_dummy_token()
                    from time import sleep

                    sleep(0.5)
                    state = await self.get_current_state()
                    tokens = state["tokens"]

                # Check if we have a sentence tokenizer available
                has_sentence_tokenizer = self.transcription_processor and hasattr(self.transcription_processor, "online") and self.transcription_processor.online and hasattr(self.transcription_processor.online, "tokenize") and self.transcription_processor.online.tokenize is not None

                # Format output - segment by sentences if tokenizer available, otherwise by speaker
                if has_sentence_tokenizer and tokens:
                    lines = await self.formatter.format_by_sentences(tokens, sep, end_attributed_speaker, self.transcription_processor.online)
                else:
                    lines = await self.formatter.format_by_speaker(tokens, sep, end_attributed_speaker)

                # Handle undiarized text
                undiarized_text = []
                if self.args.diarization:
                    # Collect any undiarized tokens
                    for token in tokens:
                        if (token.speaker in [-1] or token.speaker is None) and token.end >= end_attributed_speaker:
                            undiarized_text.append(token.text)

                if undiarized_text:
                    combined = sep.join(undiarized_text)
                    if buffer_transcription:
                        combined += sep
                    await self.update_diarization(end_attributed_speaker, combined)
                    buffer_diarization = combined

                # Create response object with s2hk conversion applied to final output
                if not lines:
                    lines = [{"speaker": 0, "text": "", "beg": format_time(0), "end": format_time(tokens[-1].end if tokens else 0), "diff": 0}]

                # Apply s2hk conversion to final output only (reduces latency) - optimize with list comprehension
                final_lines = [{**line, "text": s2hk(line["text"]) if line["text"] else line["text"]} for line in lines]

                final_buffer_transcription = s2hk(buffer_transcription) if buffer_transcription else buffer_transcription
                final_buffer_diarization = s2hk(buffer_diarization) if buffer_diarization else buffer_diarization

                response = {"lines": final_lines, "buffer_transcription": final_buffer_transcription, "buffer_diarization": final_buffer_diarization, "remaining_time_transcription": state["remaining_time_transcription"], "remaining_time_diarization": state["remaining_time_diarization"], "diarization_enabled": self.args.diarization}

                # Add analyses if available - optimize by checking flag first and limiting frequency
                current_time = time()
                if self._has_analyses or (current_time - self._last_analysis_check > 5.0):  # OPTIMIZATION: Check every 5 seconds instead of 2
                    self._last_analysis_check = current_time
                    async with self.lock:
                        if self.analyses:
                            response["analyses"] = self.analyses.copy()
                            # Only log occasionally to reduce spam
                            if current_time - getattr(self, "_last_analysis_log", 0) > 60.0:  # Log every 60 seconds instead of 30
                                logger.info(f"Including {len(self.analyses)} analyses")
                                self._last_analysis_log = current_time
                            self._has_analyses = True
                        else:
                            self._has_analyses = False

                # Add LLM inference processor stats if available
                if self.llm:
                    llm_stats = self.llm.get_stats()
                    response["llm_stats"] = llm_stats
                    # Add adaptive frequency info for debugging/monitoring
                    response["llm_adaptive_frequency"] = {"current_hz": llm_stats.get("optimal_frequency_hz", 0), "cooldown_seconds": llm_stats.get("adaptive_cooldown", 0), "avg_processing_time": llm_stats.get("avg_processing_time", 0), "recent_times": llm_stats.get("recent_processing_times", [])}

                # Add parsed transcript data if available
                if self.transcript_parser and self._parser_enabled:
                    parser_stats = self.transcript_parser.get_stats()
                    response["transcript_parser"] = {"enabled": True, "stats": parser_stats, "last_parsed": self.last_parsed_transcript.model_dump() if self.last_parsed_transcript else None, "total_parsed": len(self.parsed_transcripts)}
                    # Add parser adaptive frequency info
                    response["parser_adaptive_frequency"] = {"current_hz": parser_stats.get("optimal_frequency_hz", 0), "cooldown_seconds": parser_stats.get("adaptive_cooldown", 0), "avg_processing_time": parser_stats.get("avg_processing_time", 0), "recent_times": parser_stats.get("recent_processing_times", [])}

                # Only yield if content has changed (use final converted content for comparison) - optimize string building
                response_content = " ".join(f"{line['speaker']} {line['text']}" for line in final_lines) + f" | {final_buffer_transcription} | {final_buffer_diarization}"

                # CRITICAL FIX: More aggressive yielding for real-time UI updates
                should_yield = False

                # Always yield if content has actually changed
                if response_content != self.last_response_content:
                    should_yield = True

                # Also yield periodically even if content hasn't changed (for progress updates)
                current_time = time()
                if not hasattr(self, "_last_yield_time"):
                    self._last_yield_time = 0

                # Force yield every 0.5 seconds even without content changes (for progress indicators)
                if current_time - self._last_yield_time > 0.5:
                    should_yield = True

                # Always yield if we have any content at all (even empty buffer for UI progress)
                if final_lines or final_buffer_transcription or final_buffer_diarization:
                    if should_yield:
                        yield response
                        self.last_response_content = response_content
                        self._last_yield_time = current_time

                # Check for termination condition
                if self.is_stopping:
                    all_processors_done = True
                    if self.args.transcription and self.transcription_task and not self.transcription_task.done():
                        all_processors_done = False
                    if self.args.diarization and self.diarization_task and not self.diarization_task.done():
                        all_processors_done = False

                    logger.info(f"🚨 DEBUG: is_stopping={self.is_stopping}, all_processors_done={all_processors_done}")
                    if self.args.transcription:
                        logger.info(f"🚨 DEBUG: transcription_task exists={self.transcription_task is not None}, done={self.transcription_task.done() if self.transcription_task else 'N/A'}")
                    if self.args.diarization:
                        logger.info(f"🚨 DEBUG: diarization_task exists={self.diarization_task is not None}, done={self.diarization_task.done() if self.diarization_task else 'N/A'}")

                    if all_processors_done:
                        logger.info("Results formatter: All upstream processors are done and in stopping state. Terminating.")

                        # Final processing section
                        async for final_response in self._process_final_results():
                            yield final_response
                        return

                await asyncio.sleep(0.05)  # CRITICAL FIX: Reduced from 0.2s to 0.05s for real-time responsiveness (20 updates/second)

            except Exception as e:
                logger.warning(f"Exception in results_formatter: {e}")
                logger.warning(f"Traceback: {traceback.format_exc()}")
                await asyncio.sleep(0.5)  # Back off on error

    async def _process_final_results(self):
        """Process final results when stopping."""
        # COMPREHENSIVE FINAL PROCESSING - Check all buffer sources
        logger.info("🔄 Comprehensive final processing - checking all buffer sources...")
        logger.info("🚨 DEBUG: Starting comprehensive final processing section")

        # Get current final state
        final_state = await self.get_current_state()

        # First check for any current buffer text that hasn't been processed
        current_buffer_text = final_state.get("buffer_transcription", "")
        if current_buffer_text and current_buffer_text.strip():
            logger.info(f"🔄 Found current buffer text: '{current_buffer_text[:50]}...'")

        # Force finish the ASR to get any remaining tokens
        final_transcript_from_asr = None
        if self.transcription_processor and self.transcription_processor.online:
            try:
                final_transcript_from_asr = self.transcription_processor.online.finish()
                if final_transcript_from_asr and final_transcript_from_asr.text and final_transcript_from_asr.text.strip():
                    logger.info(f"✅ Retrieved final transcript from ASR finish(): '{final_transcript_from_asr.text[:50]}...'")
                else:
                    logger.info("🔄 No additional content from ASR finish()")
            except Exception as e:
                logger.warning(f"Failed to finish ASR: {e}")

        # SINGLE SOURCE: Get all content from global memory atomically
        async with self.lock:
            # Get final content from global memory
            committed_text = self.sep.join([token.text for token in self.global_transcript["committed_tokens"] if token.text and token.text.strip()])
            buffer_text = self.global_transcript["current_buffer"]
            final_global_text = (committed_text + " " + buffer_text).strip() if buffer_text else committed_text

            logger.info(f"🔄 Final global memory content: '{final_global_text[:100]}...' ({len(final_global_text)} chars)")

        # Create additional tokens from remaining content (no duplication by design)
        additional_tokens_created = 0
        tokens_to_add = []

        # Get ASR finish() content
        asr_text = final_transcript_from_asr.text if final_transcript_from_asr and final_transcript_from_asr.text else ""

        # Simple check: only add ASR text if it's not already in global memory
        unique_asr_text = ""
        if asr_text.strip() and final_global_text:
            # Check if ASR text is already in global memory
            if asr_text.strip() not in final_global_text:
                unique_asr_text = asr_text.strip()
                logger.info(f"🔄 Found unique ASR text: '{unique_asr_text[:50]}...'")
            else:
                logger.info("🔄 ASR text already in global memory, skipping")
        elif asr_text.strip():
            unique_asr_text = asr_text.strip()
            logger.info(f"🔄 Using ASR text (no global memory): '{unique_asr_text[:50]}...'")

        # No complex deduplication needed - single source of truth prevents duplicates
        text_sources = []
        if unique_asr_text:
            text_sources.append(("asr", unique_asr_text))

        # Process unique texts - deduplication handled by work coordination
        for i, (source_name, unique_text) in enumerate(text_sources):
            last_end = final_state["tokens"][-1].end if final_state["tokens"] else 0
            if tokens_to_add:  # Use end time of last added token
                last_end = tokens_to_add[-1].end

            # Create token with estimated timing
            estimated_duration = max(2.0, len(unique_text) / 20)  # ~20 chars per second speaking rate
            final_token = ASRToken(start=last_end, end=last_end + estimated_duration, text=unique_text, speaker=0)  # Default speaker
            tokens_to_add.append(final_token)
            logger.info(f"🔄 Prepared final token {i+1} ({final_token.start:.1f}s-{final_token.end:.1f}s): '{final_token.text[:50]}...'")

        # Add all tokens at once to avoid race conditions
        if tokens_to_add:
            async with self.lock:
                self.tokens.extend(tokens_to_add)
                additional_tokens_created = len(tokens_to_add)
                logger.info(f"🔄 Added {additional_tokens_created} final tokens to committed transcript")

                # Update full transcription with any new content
                if hasattr(self, "full_transcription"):
                    # Calculate new full transcription from all tokens
                    all_token_texts = [token.text for token in self.tokens if token.text and token.text.strip()]
                    self.full_transcription = (final_state.get("sep", " ") or " ").join(all_token_texts)
                    logger.info(f"🔄 Updated full transcription (now {len(self.full_transcription)} chars)")

        # SINGLE SOURCE: All content is now in global memory
        logger.info(f"🔄 Using global memory for final processing: '{final_global_text[:100]}...' ({len(final_global_text)} chars)")

        # Update LLM with final global memory content
        if self.llm and final_global_text.strip():
            remaining_text_converted = s2hk(final_global_text)
            self.llm.update_transcription(remaining_text_converted, None)
            logger.info(f"Updated LLM with final global memory content: '{remaining_text_converted[:50]}...'")
        else:
            logger.info("No final content in global memory to process")

        # Generate final inference if needed - check before stopping monitoring
        async with self.lock:
            analyses_count = len(getattr(self, "analyses", []))

        logger.info(f"🚨 DEBUG: About to check final analysis generation, analyses_count={analyses_count}")

        # Generate final comprehensive analysis (avoid duplicates with single-path logic)
        final_analysis_generated = False
        should_generate_analysis = False  # Initialize flag

        logger.info(f"🔍 Checking final analysis generation conditions...")
        logger.info(f"🔍 self.llm exists: {self.llm is not None}")

        if self.llm:
            accumulated_length = len(self.llm.accumulated_data.strip()) if hasattr(self.llm, "accumulated_data") else 0
            logger.info(f"🔍 LLM accumulated_data length: {accumulated_length}")
            logger.info(f"🔍 LLM accumulated_data preview: '{self.llm.accumulated_data[:100]}...' " if hasattr(self.llm, "accumulated_data") and self.llm.accumulated_data else "No accumulated data")

            # Always try to populate with full transcript first
            if hasattr(self, "full_transcription") and self.full_transcription.strip():
                logger.info("🔧 Ensuring LLM has full transcription for final analysis...")
                self.llm.update_transcription(self.full_transcription, None)
                accumulated_length = len(self.llm.accumulated_data.strip()) if hasattr(self.llm, "accumulated_data") else 0
                logger.info(f"🔧 LLM now has {accumulated_length} chars of accumulated data")

            # Check if we have data to process
            if hasattr(self.llm, "accumulated_data") and self.llm.accumulated_data.strip():
                should_generate_analysis = True
                if analyses_count == 0:
                    logger.info("🔄 Generating final analysis (no analyses created during processing)...")
                else:
                    logger.info(f"🔄 Generating comprehensive final analysis (had {analyses_count} intermediate analyses)...")
            else:
                should_generate_analysis = False
                logger.warning("❌ Final analysis NOT generated - no content available after populating LLM")
        else:
            logger.warning("❌ Final analysis NOT generated - no LLM instance available")

        # SINGLE FORCE_INFERENCE CALL WITH DUPLICATE PREVENTION
        if should_generate_analysis:
            # Update with complete transcript to ensure comprehensive analysis
            complete_transcript = self.full_transcription
            if complete_transcript.strip():
                self.llm.update_transcription(complete_transcript, None)
                logger.info(f"🔄 Updated LLM with complete transcript ({len(complete_transcript)} chars) for final analysis")

            # Single force inference call
            await self.llm.force_inference()
            final_analysis_generated = True

            # Wait for the final inference to complete
            max_wait_time = 5.0  # Increased wait time for final analysis
            poll_interval = 0.3
            waited_time = 0.0
            initial_analysis_count = analyses_count

            while waited_time < max_wait_time:
                await asyncio.sleep(poll_interval)
                waited_time += poll_interval

                # Check if new analysis was added
                async with self.lock:
                    current_analysis_count = len(getattr(self, "analyses", []))

                if current_analysis_count > initial_analysis_count:
                    logger.info(f"✅ Final analysis added after {waited_time:.1f}s wait")
                    break

                if waited_time >= max_wait_time:
                    logger.warning(f"⚠️ Final analysis not added after {max_wait_time}s wait")
                    break

        # CRITICAL: Set final analysis flag AFTER final processing to prevent blocking final analysis
        self._final_analysis_generated = True
        logger.info("🛑 Set final analysis flag AFTER final processing completed")

        # Stop LLM monitoring after final processing is complete
        if self.llm:
            try:
                await self.llm.stop_monitoring()
                logger.info("🛑 Stopped LLM background monitoring after final processing")
            except Exception as e:
                logger.warning(f"Error stopping LLM monitoring: {e}")

        if not final_analysis_generated:
            logger.warning("❌ Final analysis NOT generated - LLM unavailable or no content")

        # Get updated final state after inference processing
        final_state = await self.get_current_state()

        # Format final lines and apply refinement and s2hk conversion
        final_lines_raw = await self.formatter.format_by_speaker(final_state["tokens"], final_state["sep"], final_state["end_attributed_speaker"])

        # Apply s2hk conversion to output (refined or original)
        final_lines_converted = [{**line, "text": s2hk(line["text"]) if line["text"] else line["text"]} for line in final_lines_raw]

        # Refine and include any remaining buffer text in the final response
        final_buffer_transcription = s2hk(final_state["buffer_transcription"]) if final_state["buffer_transcription"] else ""
        final_buffer_diarization = s2hk(final_state["buffer_diarization"]) if final_state["buffer_diarization"] else ""

        # Create final response with remaining buffer text included
        final_response = {"lines": final_lines_converted, "buffer_transcription": final_buffer_transcription, "buffer_diarization": final_buffer_diarization, "remaining_time_transcription": 0, "remaining_time_diarization": 0, "diarization_enabled": self.args.diarization}

        # Add existing analyses
        async with self.lock:
            if hasattr(self, "analyses") and self.analyses:
                final_response["analyses"] = self.analyses.copy()
                logger.info(f"🔄 Including {len(self.analyses)} final analyses in response")

        # Add LLM stats
        if self.llm:
            final_response["llm_stats"] = self.llm.get_stats()

        # Log final analysis of transcription
        total_lines = len(final_lines_converted)
        buffer_chars = len(final_buffer_transcription) + len(final_buffer_diarization)
        logger.info(f"📋 Final transcription: {total_lines} committed lines, {buffer_chars} buffer characters")

        yield final_response
