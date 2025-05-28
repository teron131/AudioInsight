import asyncio
import logging
import math
import re
import traceback
from datetime import timedelta
from time import sleep, time

import ffmpeg
import numpy as np
import opencc

from whisperlivekit.core import WhisperLiveKit
from whisperlivekit.llm import LLM, LLMTrigger
from whisperlivekit.timed_objects import ASRToken
from whisperlivekit.whisper_streaming.whisper_online import online_factory

# Set up logging once
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

SENTINEL = object()  # unique sentinel object for end of stream marker

# Cache OpenCC converter instance to avoid recreation
_s2hk_converter = None

# Pre-compile regex for sentence splitting
_sentence_split_regex = re.compile(r"[.!?]+")

# Cache timedelta formatting for common values
_cached_timedeltas = {}


def format_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS with caching for performance."""
    if seconds == 0:
        return "0:00:00"

    int_seconds = int(seconds)
    if int_seconds in _cached_timedeltas:
        return _cached_timedeltas[int_seconds]

    result = str(timedelta(seconds=int_seconds))

    # Cache up to 3600 entries (1 hour worth of seconds)
    if len(_cached_timedeltas) < 3600:
        _cached_timedeltas[int_seconds] = result

    return result


def s2hk(text: str) -> str:
    """Convert Simplified Chinese to Traditional Chinese with cached converter."""
    if not text:
        return text

    global _s2hk_converter
    if _s2hk_converter is None:
        _s2hk_converter = opencc.OpenCC("s2hk")

    return _s2hk_converter.convert(text)


class AudioProcessor:
    """
    Processes audio streams for transcription and diarization.
    Handles audio processing, state management, and result formatting.
    """

    def __init__(self):
        """Initialize the audio processor with configuration, models, and state."""

        models = WhisperLiveKit()

        # Audio processing settings - cache computed values
        self.args = models.args
        self.sample_rate = 16000
        self.channels = 1
        self.bytes_per_sample = 2

        # Pre-compute commonly used values
        self.samples_per_sec = int(self.sample_rate * self.args.min_chunk_size)
        self.bytes_per_sec = self.samples_per_sec * self.bytes_per_sample
        self.max_bytes_per_sec = 32000 * 5  # 5 seconds of audio at 32 kHz
        self.sample_rate_str = str(self.sample_rate)  # Cache string conversion

        # Pre-allocate buffers for better memory efficiency
        self.max_buffer_size = self.max_bytes_per_sec * 2  # Double buffer size for safety
        self.pcm_buffer = bytearray(self.max_buffer_size)
        self.pcm_buffer_length = 0  # Track actual data length

        # Pre-allocate numpy arrays to avoid repeated allocation
        self._temp_int16_array = np.empty(self.max_buffer_size // 2, dtype=np.int16)
        self._temp_float32_array = np.empty(self.max_buffer_size // 2, dtype=np.float32)

        # Timing settings
        self.last_ffmpeg_activity = time()
        self.ffmpeg_health_check_interval = 5
        self.ffmpeg_max_idle_time = 10

        # State management
        self.is_stopping = False
        self.tokens = []
        self.buffer_transcription = ""
        self.buffer_diarization = ""
        self.full_transcription = ""
        self.end_buffer = 0
        self.end_attributed_speaker = 0
        self.lock = asyncio.Lock()
        self.beg_loop = time()
        self.sep = " "  # Default separator
        self.last_response_content = ""
        self.summaries = []  # Initialize summaries list
        self._has_summaries = False  # Efficient flag to track summary availability
        self._last_summary_check = 0  # Timestamp of last summary check

        # Models and processing
        self.asr = models.asr
        self.tokenizer = models.tokenizer
        self.diarization = models.diarization
        self.ffmpeg_process = self.start_ffmpeg_decoder()
        self.transcription_queue = asyncio.Queue() if self.args.transcription else None
        self.diarization_queue = asyncio.Queue() if self.args.diarization else None

        # Task references
        self.transcription_task = None
        self.diarization_task = None
        self.ffmpeg_reader_task = None
        self.watchdog_task = None
        self.all_tasks_for_cleanup = []

        # Initialize transcription engine if enabled
        if self.args.transcription:
            self.online = online_factory(self.args, models.asr, models.tokenizer)

        # Initialize LLM inference processor if enabled
        self.llm = None
        self.llm_task = None
        if getattr(self.args, "llm_inference", False):
            try:
                trigger_config = LLMTrigger(
                    idle_time_seconds=getattr(self.args, "llm_trigger_time", 5.0),
                    conversation_trigger_count=getattr(self.args, "llm_conversation_trigger", 2),
                )
                self.llm = LLM(
                    model_id=getattr(self.args, "llm_model", "gpt-4.1-mini"),
                    trigger_config=trigger_config,
                )
                # Add callback to handle inference results
                self.llm.add_inference_callback(self._handle_inference_callback)
                logger.info(f"LLM inference enabled with model: {self.args.llm_model}")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM inference processor: {e}")
                self.llm = None

    def convert_pcm_to_float(self, pcm_data, length=None):
        """Convert PCM buffer in s16le format to normalized NumPy array with pre-allocated buffers."""
        if length is None:
            length = len(pcm_data)

        # Use pre-allocated arrays for better performance
        num_samples = length // 2
        if num_samples > len(self._temp_int16_array):
            # Resize if needed (rare case)
            self._temp_int16_array = np.empty(num_samples, dtype=np.int16)
            self._temp_float32_array = np.empty(num_samples, dtype=np.float32)

        # Copy data into pre-allocated buffer
        self._temp_int16_array[:num_samples] = np.frombuffer(pcm_data[:length], dtype=np.int16)

        # Convert to float32 in-place
        np.divide(self._temp_int16_array[:num_samples], 32768.0, out=self._temp_float32_array[:num_samples])

        # Return a copy of the needed portion
        return self._temp_float32_array[:num_samples].copy()

    def append_to_pcm_buffer(self, chunk):
        """Efficiently append audio chunk to PCM buffer."""
        chunk_len = len(chunk)
        new_length = self.pcm_buffer_length + chunk_len

        # Resize buffer if needed
        if new_length > len(self.pcm_buffer):
            new_size = max(new_length, len(self.pcm_buffer) * 2)
            new_buffer = bytearray(new_size)
            new_buffer[: self.pcm_buffer_length] = self.pcm_buffer[: self.pcm_buffer_length]
            self.pcm_buffer = new_buffer

        # Append new data
        self.pcm_buffer[self.pcm_buffer_length : new_length] = chunk
        self.pcm_buffer_length = new_length

    def get_pcm_data(self, max_bytes):
        """Get PCM data up to max_bytes and remove it from buffer."""
        actual_bytes = min(self.pcm_buffer_length, max_bytes)

        # Get data
        data = bytes(self.pcm_buffer[:actual_bytes])

        # Shift remaining data to front
        if actual_bytes < self.pcm_buffer_length:
            remaining = self.pcm_buffer_length - actual_bytes
            self.pcm_buffer[:remaining] = self.pcm_buffer[actual_bytes : self.pcm_buffer_length]
            self.pcm_buffer_length = remaining
        else:
            self.pcm_buffer_length = 0

        return data

    def start_ffmpeg_decoder(self):
        """Start FFmpeg process for WebM to PCM conversion."""
        return ffmpeg.input("pipe:0", format="webm").output("pipe:1", format="s16le", acodec="pcm_s16le", ac=self.channels, ar=self.sample_rate_str).run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)

    async def restart_ffmpeg(self):
        """Restart the FFmpeg process after failure."""
        logger.warning("Restarting FFmpeg process...")

        if self.ffmpeg_process:
            try:
                # we check if process is still running
                if self.ffmpeg_process.poll() is None:
                    logger.info("Terminating existing FFmpeg process")
                    self.ffmpeg_process.stdin.close()
                    self.ffmpeg_process.terminate()

                    # wait for termination with timeout
                    try:
                        await asyncio.wait_for(asyncio.get_event_loop().run_in_executor(None, self.ffmpeg_process.wait), timeout=5.0)
                    except asyncio.TimeoutError:
                        logger.warning("FFmpeg process did not terminate, killing forcefully")
                        self.ffmpeg_process.kill()
                        await asyncio.get_event_loop().run_in_executor(None, self.ffmpeg_process.wait)
            except Exception as e:
                logger.error(f"Error during FFmpeg process termination: {e}")
                logger.error(traceback.format_exc())

        # we start new process
        try:
            logger.info("Starting new FFmpeg process")
            self.ffmpeg_process = self.start_ffmpeg_decoder()
            self.pcm_buffer_length = 0  # Reset buffer length for new process
            self.last_ffmpeg_activity = time()
            logger.info("FFmpeg process restarted successfully")
        except Exception as e:
            logger.error(f"Failed to restart FFmpeg process: {e}")
            logger.error(traceback.format_exc())
            # try again after 5s
            await asyncio.sleep(5)
            try:
                self.ffmpeg_process = self.start_ffmpeg_decoder()
                self.pcm_buffer_length = 0  # Reset buffer length for new process
                self.last_ffmpeg_activity = time()
                logger.info("FFmpeg process restarted successfully on second attempt")
            except Exception as e2:
                logger.critical(f"Failed to restart FFmpeg process on second attempt: {e2}")
                logger.critical(traceback.format_exc())

    async def update_transcription(self, new_tokens, buffer, end_buffer, full_transcription, sep):
        """Thread-safe update of transcription with new data."""
        async with self.lock:
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
            self.tokens.append(ASRToken(start=current_time, end=current_time + 1, text=".", speaker=-1, is_dummy=True))

    async def _handle_inference_callback(self, inference_response, transcription_text):
        """Handle callback when a inference result is generated."""
        logger.info(f"📝 Inference result generated: {inference_response.summary[:50]}...")
        logger.info(f"🔑 Key points: {', '.join(inference_response.key_points[:2])}...")

        # Store inference result in the response for client access
        async with self.lock:
            if not hasattr(self, "summaries"):
                self.summaries = []

            # Check for duplicate results (same result text)
            new_result = {
                "timestamp": time(),
                "summary": inference_response.summary,
                "key_points": inference_response.key_points,
                "text_length": len(transcription_text),
            }

            # Only add if it's not a duplicate
            is_duplicate = any(existing["summary"] == new_result["summary"] for existing in self.summaries)

            if not is_duplicate:
                self.summaries.append(new_result)
                self._has_summaries = True  # Set efficient flag
                logger.info(f"✅ Added new inference result (total: {len(self.summaries)})")
            else:
                logger.info("⚠️ Duplicate inference result detected, not adding")

    async def get_current_state(self):
        """Get current state."""
        async with self.lock:
            current_time = time()

            # Calculate remaining times
            remaining_transcription = 0
            if self.end_buffer > 0:
                remaining_transcription = max(0, round(current_time - self.beg_loop - self.end_buffer, 2))

            remaining_diarization = 0
            if self.tokens:
                latest_end = max(self.end_buffer, self.tokens[-1].end if self.tokens else 0)
                remaining_diarization = max(0, round(latest_end - self.end_attributed_speaker, 2))

            return {"tokens": self.tokens.copy(), "buffer_transcription": self.buffer_transcription, "buffer_diarization": self.buffer_diarization, "end_buffer": self.end_buffer, "end_attributed_speaker": self.end_attributed_speaker, "sep": self.sep, "remaining_time_transcription": remaining_transcription, "remaining_time_diarization": remaining_diarization}

    async def reset(self):
        """Reset all state variables to initial values."""
        async with self.lock:
            self.tokens = []
            self.buffer_transcription = self.buffer_diarization = ""
            self.end_buffer = self.end_attributed_speaker = 0
            self.full_transcription = self.last_response_content = ""
            self.beg_loop = time()

    async def ffmpeg_stdout_reader(self):
        """Read audio data from FFmpeg stdout and process it."""
        loop = asyncio.get_event_loop()
        beg = time()

        while True:
            try:
                current_time = time()
                elapsed_time = math.floor((current_time - beg) * 10) / 10
                buffer_size = max(int(32000 * elapsed_time), 4096)
                beg = current_time

                # Detect idle state much more quickly
                if current_time - self.last_ffmpeg_activity > self.ffmpeg_max_idle_time:
                    logger.warning(f"FFmpeg process idle for {current_time - self.last_ffmpeg_activity:.2f}s. Restarting...")
                    await self.restart_ffmpeg()
                    beg = time()
                    self.last_ffmpeg_activity = time()
                    continue

                chunk = await loop.run_in_executor(None, self.ffmpeg_process.stdout.read, buffer_size)
                if chunk:
                    self.last_ffmpeg_activity = time()

                if not chunk:
                    logger.info("FFmpeg stdout closed, no more data to read.")
                    break

                self.append_to_pcm_buffer(chunk)

                # Send to diarization if enabled
                if self.args.diarization and self.diarization_queue:
                    await self.diarization_queue.put(self.convert_pcm_to_float(self.get_pcm_data(buffer_size)).copy())

                # Process when enough data
                if self.pcm_buffer_length >= self.bytes_per_sec:
                    if self.pcm_buffer_length > self.max_bytes_per_sec:
                        logger.warning(f"Audio buffer too large: {self.pcm_buffer_length / self.bytes_per_sec:.2f}s. " f"Consider using a smaller model.")

                    # Process audio chunk
                    pcm_array = self.convert_pcm_to_float(self.get_pcm_data(self.max_bytes_per_sec))

                    # Send to transcription if enabled
                    if self.args.transcription and self.transcription_queue:
                        await self.transcription_queue.put(pcm_array.copy())

                    # Sleep if no processing is happening
                    if not self.args.transcription and not self.args.diarization:
                        await asyncio.sleep(0.1)

            except Exception as e:
                logger.warning(f"Exception in ffmpeg_stdout_reader: {e}")
                logger.warning(f"Traceback: {traceback.format_exc()}")
                break

        logger.info("FFmpeg stdout processing finished. Signaling downstream processors.")
        if self.args.transcription and self.transcription_queue:
            await self.transcription_queue.put(SENTINEL)
            logger.debug("Sentinel put into transcription_queue.")
        if self.args.diarization and self.diarization_queue:
            await self.diarization_queue.put(SENTINEL)
            logger.debug("Sentinel put into diarization_queue.")

    async def transcription_processor(self):
        """Process audio chunks for transcription."""
        self.full_transcription = ""
        self.sep = self.online.asr.sep

        while True:
            try:
                pcm_array = await self.transcription_queue.get()
                if pcm_array is SENTINEL:
                    logger.debug("Transcription processor received sentinel. Finishing.")
                    self.transcription_queue.task_done()
                    break

                if not self.online:  # Should not happen if queue is used
                    logger.warning("Transcription processor: self.online not initialized.")
                    self.transcription_queue.task_done()
                    continue

                asr_internal_buffer_duration_s = len(self.online.audio_buffer) / self.online.SAMPLING_RATE
                transcription_lag_s = max(0.0, time() - self.beg_loop - self.end_buffer)

                logger.info(f"ASR processing: internal_buffer={asr_internal_buffer_duration_s:.2f}s, " f"lag={transcription_lag_s:.2f}s.")

                # Process transcription
                self.online.insert_audio_chunk(pcm_array)
                new_tokens = self.online.process_iter()

                if new_tokens:
                    self.full_transcription += self.sep.join([t.text for t in new_tokens])

                # Get buffer information
                _buffer = self.online.get_buffer()
                buffer = _buffer.text
                end_buffer = _buffer.end if _buffer.end else (new_tokens[-1].end if new_tokens else 0)

                # Avoid duplicating content
                if buffer in self.full_transcription:
                    buffer = ""

                await self.update_transcription(new_tokens, buffer, end_buffer, self.full_transcription, self.sep)

                # Update LLM inference processor with new transcription text
                if self.llm and new_tokens:
                    new_text = self.sep.join([t.text for t in new_tokens])
                    # Convert to Traditional Chinese for consistency
                    new_text_converted = s2hk(new_text) if new_text else new_text
                    # Get speaker info if available
                    speaker_info = None
                    if new_tokens and hasattr(new_tokens[0], "speaker") and new_tokens[0].speaker is not None:
                        speaker_info = {"speaker": new_tokens[0].speaker}
                    self.llm.update_transcription(new_text_converted, speaker_info)

                self.transcription_queue.task_done()

            except Exception as e:
                logger.warning(f"Exception in transcription_processor: {e}")
                logger.warning(f"Traceback: {traceback.format_exc()}")
                if "pcm_array" in locals() and pcm_array is not SENTINEL:  # Check if pcm_array was assigned from queue
                    self.transcription_queue.task_done()
        logger.info("Transcription processor task finished.")

    async def diarization_processor(self, diarization_obj):
        """Process audio chunks for speaker diarization."""
        buffer_diarization = ""

        while True:
            try:
                pcm_array = await self.diarization_queue.get()
                if pcm_array is SENTINEL:
                    logger.debug("Diarization processor received sentinel. Finishing.")
                    self.diarization_queue.task_done()
                    break

                # Process diarization
                await diarization_obj.diarize(pcm_array)

                # Get current state and update speakers
                state = await self.get_current_state()
                new_end = diarization_obj.assign_speakers_to_tokens(state["end_attributed_speaker"], state["tokens"])

                await self.update_diarization(new_end, buffer_diarization)
                self.diarization_queue.task_done()

            except Exception as e:
                logger.warning(f"Exception in diarization_processor: {e}")
                logger.warning(f"Traceback: {traceback.format_exc()}")
                if "pcm_array" in locals() and pcm_array is not SENTINEL:
                    self.diarization_queue.task_done()
        logger.info("Diarization processor task finished.")

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
                    sleep(0.5)
                    state = await self.get_current_state()
                    tokens = state["tokens"]

                # Check if we have a sentence tokenizer available
                has_sentence_tokenizer = hasattr(self, "online") and hasattr(self.online, "tokenize") and self.online.tokenize is not None

                # Format output - segment by sentences if tokenizer available, otherwise by speaker
                if has_sentence_tokenizer and tokens:
                    lines = await self._format_by_sentences(tokens, sep, end_attributed_speaker)
                else:
                    lines = await self._format_by_speaker(tokens, sep, end_attributed_speaker)

                # Handle undiarized text
                undiarized_text = []
                if self.args.diarization:
                    # Collect any undiarized tokens
                    for token in tokens:
                        if (token.speaker in [-1, 0]) and token.end >= end_attributed_speaker:
                            undiarized_text.append(token.text)

                if undiarized_text:
                    combined = sep.join(undiarized_text)
                    if buffer_transcription:
                        combined += sep
                    await self.update_diarization(end_attributed_speaker, combined)
                    buffer_diarization = combined

                # Create response object with s2hk conversion applied to final output
                if not lines:
                    lines = [{"speaker": 1, "text": "", "beg": format_time(0), "end": format_time(tokens[-1].end if tokens else 0), "diff": 0}]

                # Apply s2hk conversion to final output only (reduces latency) - optimize with list comprehension
                final_lines = [{**line, "text": s2hk(line["text"]) if line["text"] else line["text"]} for line in lines]

                final_buffer_transcription = s2hk(buffer_transcription) if buffer_transcription else buffer_transcription
                final_buffer_diarization = s2hk(buffer_diarization) if buffer_diarization else buffer_diarization

                response = {"lines": final_lines, "buffer_transcription": final_buffer_transcription, "buffer_diarization": final_buffer_diarization, "remaining_time_transcription": state["remaining_time_transcription"], "remaining_time_diarization": state["remaining_time_diarization"]}

                # Add summaries if available - optimize by checking flag first and limiting frequency
                current_time = time()
                if self._has_summaries or (current_time - self._last_summary_check > 2.0):  # Check every 2 seconds if no summaries
                    self._last_summary_check = current_time
                    async with self.lock:
                        if self.summaries:
                            response["summaries"] = self.summaries.copy()
                            logger.info(f"🔄 Including {len(self.summaries)} summaries in response")
                            self._has_summaries = True
                        else:
                            self._has_summaries = False

                # Add LLM inference processor stats if available
                if self.llm:
                    response["llm_stats"] = self.llm.get_stats()

                # Only yield if content has changed (use final converted content for comparison) - optimize string building
                response_content = " ".join(f"{line['speaker']} {line['text']}" for line in final_lines) + f" | {final_buffer_transcription} | {final_buffer_diarization}"

                if response_content != self.last_response_content and (final_lines or final_buffer_transcription or final_buffer_diarization):
                    yield response
                    self.last_response_content = response_content

                # Check for termination condition
                if self.is_stopping:
                    all_processors_done = True
                    if self.args.transcription and self.transcription_task and not self.transcription_task.done():
                        all_processors_done = False
                    if self.args.diarization and self.diarization_task and not self.diarization_task.done():
                        all_processors_done = False

                    if all_processors_done:
                        logger.info("Results formatter: All upstream processors are done and in stopping state. Terminating.")

                        # Get final state and create final response
                        final_state = await self.get_current_state()

                        # Force commit any remaining buffer text for final processing
                        final_buffer_text = final_state["buffer_transcription"]
                        if final_buffer_text and hasattr(self, "online") and self.online:
                            # Try to finish the transcription to get any remaining tokens
                            try:
                                remaining_transcript = self.online.finish()
                                if remaining_transcript and remaining_transcript.text:
                                    logger.info(f"Retrieved remaining transcript: '{remaining_transcript.text[:50]}...'")
                                    # Update LLM with any remaining text
                                    if self.llm:
                                        remaining_text_converted = s2hk(remaining_transcript.text) if remaining_transcript.text else ""
                                        if remaining_text_converted.strip():
                                            self.llm.update_transcription(remaining_text_converted, None)
                                            logger.info(f"Updated LLM with remaining text: '{remaining_text_converted[:50]}...'")
                            except Exception as e:
                                logger.warning(f"Failed to retrieve remaining transcript: {e}")

                        # Generate final inference if no summaries were created during processing
                        async with self.lock:
                            summaries_count = len(getattr(self, "summaries", []))

                        if self.llm and self.llm.accumulated_text.strip() and summaries_count == 0:
                            logger.info("🔄 Generating final inference (no summaries created during processing)...")
                            await self.llm.force_inference()
                            # Wait a moment for the callback to be processed
                            await asyncio.sleep(0.1)

                        # Get updated final state after inference processing
                        final_state = await self.get_current_state()

                        # Format final lines and apply s2hk conversion
                        final_lines_raw = await self._format_by_speaker(final_state["tokens"], final_state["sep"], final_state["end_attributed_speaker"])
                        final_lines_converted = [{**line, "text": s2hk(line["text"]) if line["text"] else line["text"]} for line in final_lines_raw]

                        # Include any remaining buffer text in the final response
                        final_buffer_transcription = s2hk(final_state["buffer_transcription"]) if final_state["buffer_transcription"] else ""
                        final_buffer_diarization = s2hk(final_state["buffer_diarization"]) if final_state["buffer_diarization"] else ""

                        # Create final response with remaining buffer text included
                        final_response = {"lines": final_lines_converted, "buffer_transcription": final_buffer_transcription, "buffer_diarization": final_buffer_diarization, "remaining_time_transcription": 0, "remaining_time_diarization": 0}

                        # Add existing summaries
                        async with self.lock:
                            if hasattr(self, "summaries") and self.summaries:
                                final_response["summaries"] = self.summaries.copy()
                                logger.info(f"🔄 Including {len(self.summaries)} final summaries in response")

                        # Add LLM stats
                        if self.llm:
                            final_response["llm_stats"] = self.llm.get_stats()

                        # Log final summary of transcription
                        total_lines = len(final_lines_converted)
                        buffer_chars = len(final_buffer_transcription) + len(final_buffer_diarization)
                        logger.info(f"📋 Final transcription: {total_lines} committed lines, {buffer_chars} buffer characters")

                        yield final_response
                        return

                await asyncio.sleep(0.1)  # Avoid overwhelming the client

            except Exception as e:
                logger.warning(f"Exception in results_formatter: {e}")
                logger.warning(f"Traceback: {traceback.format_exc()}")
                await asyncio.sleep(0.5)  # Back off on error

    async def _format_by_sentences(self, tokens, sep, end_attributed_speaker):
        """Format tokens by sentence boundaries using the sentence tokenizer."""
        if not tokens:
            return []

        # Build full text from all tokens - optimize by filtering first
        token_texts = [token.text for token in tokens if token.text and token.text.strip()]
        if not token_texts:
            return []

        full_text = sep.join(token_texts)

        try:
            # Use the sentence tokenizer to split into sentences
            if hasattr(self.online, "tokenize") and self.online.tokenize:
                try:
                    # MosesSentenceSplitter expects a list input
                    sentence_texts = self.online.tokenize([full_text])
                except Exception as e:
                    # Fallback for other tokenizers that might expect string input
                    try:
                        sentence_texts = self.online.tokenize(full_text)
                    except Exception as e2:
                        logger.warning(f"Sentence tokenization failed: {e2}. Falling back to speaker-based segmentation.")
                        return await self._format_by_speaker(tokens, sep, end_attributed_speaker)
            else:
                # No tokenizer, split by basic punctuation
                sentence_texts = _sentence_split_regex.split(full_text)
                sentence_texts = [s.strip() for s in sentence_texts if s.strip()]

            if not sentence_texts:
                sentence_texts = [full_text]

            # Map sentences back to tokens and create lines
            lines = []
            token_index = 0

            for sent_text in sentence_texts:
                sent_text = sent_text.strip()
                if not sent_text:
                    continue

                # Find tokens that make up this sentence
                sent_tokens = []
                accumulated = ""
                start_token_index = token_index

                # Accumulate tokens until we roughly match the sentence text
                while token_index < len(tokens) and len(accumulated) < len(sent_text):
                    token = tokens[token_index]
                    if token.text.strip():  # Only consider non-empty tokens
                        accumulated = (accumulated + " " + token.text).strip() if accumulated else token.text
                        sent_tokens.append(token)
                    token_index += 1

                # If we didn't get any tokens, try to get at least one
                if not sent_tokens and start_token_index < len(tokens):
                    sent_tokens = [tokens[start_token_index]]
                    token_index = start_token_index + 1

                if sent_tokens:
                    # Determine speaker (use most common speaker in the sentence) - optimize speaker detection
                    if self.args.diarization:
                        # Filter valid speakers once
                        valid_speakers = [t.speaker for t in sent_tokens if t.speaker not in {-1, 0}]
                        if valid_speakers:
                            # Use most frequent speaker with optimized counting
                            speaker = max(set(valid_speakers), key=valid_speakers.count)
                        else:
                            speaker = sent_tokens[0].speaker
                    else:
                        speaker = 1  # Default speaker when no diarization

                    # Create line for this sentence
                    line = {"speaker": speaker, "text": sent_text, "beg": format_time(sent_tokens[0].start), "end": format_time(sent_tokens[-1].end), "diff": 0}  # Not used in sentence mode
                    lines.append(line)

            return lines

        except Exception as e:
            logger.warning(f"Error in sentence-based formatting: {e}. Falling back to speaker-based segmentation.")
            return await self._format_by_speaker(tokens, sep, end_attributed_speaker)

    async def _format_by_speaker(self, tokens, sep, end_attributed_speaker):
        """Format tokens by speaker changes (original behavior)."""
        previous_speaker = -1
        lines = []
        last_end_diarized = 0
        undiarized_text = []

        # Process each token
        for token in tokens:
            speaker = token.speaker

            # Handle diarization - optimize with set membership checks
            if self.args.diarization:
                speaker_in_invalid_set = speaker in {-1, 0}
                if speaker_in_invalid_set and token.end >= end_attributed_speaker:
                    undiarized_text.append(token.text)
                    continue
                elif speaker_in_invalid_set and token.end < end_attributed_speaker:
                    speaker = previous_speaker
                if not speaker_in_invalid_set:
                    last_end_diarized = max(token.end, last_end_diarized)

            # Group by speaker
            if speaker != previous_speaker or not lines:
                lines.append({"speaker": speaker, "text": token.text, "beg": format_time(token.start), "end": format_time(token.end), "diff": round(token.end - last_end_diarized, 2)})
                previous_speaker = speaker
            elif token.text:  # Only append if text isn't empty
                lines[-1]["text"] += sep + token.text
                lines[-1]["end"] = format_time(token.end)
                lines[-1]["diff"] = round(token.end - last_end_diarized, 2)

        return lines

    async def create_tasks(self):
        """Create and start processing tasks."""
        self.all_tasks_for_cleanup = []
        processing_tasks_for_watchdog = []

        if self.args.transcription and self.online:
            self.transcription_task = asyncio.create_task(self.transcription_processor())
            self.all_tasks_for_cleanup.append(self.transcription_task)
            processing_tasks_for_watchdog.append(self.transcription_task)

        if self.args.diarization and self.diarization:
            self.diarization_task = asyncio.create_task(self.diarization_processor(self.diarization))
            self.all_tasks_for_cleanup.append(self.diarization_task)
            processing_tasks_for_watchdog.append(self.diarization_task)

        self.ffmpeg_reader_task = asyncio.create_task(self.ffmpeg_stdout_reader())
        self.all_tasks_for_cleanup.append(self.ffmpeg_reader_task)
        processing_tasks_for_watchdog.append(self.ffmpeg_reader_task)

        # Start LLM inference processor monitoring if enabled
        if self.llm:
            self.llm_task = asyncio.create_task(self.llm.start_monitoring())
            self.all_tasks_for_cleanup.append(self.llm_task)
            logger.info("LLM inference processor monitoring task started")

        # Monitor overall system health
        self.watchdog_task = asyncio.create_task(self.watchdog(processing_tasks_for_watchdog))
        self.all_tasks_for_cleanup.append(self.watchdog_task)

        return self.results_formatter()

    async def watchdog(self, tasks_to_monitor):
        """Monitors the health of critical processing tasks."""
        while True:
            try:
                await asyncio.sleep(10)
                current_time = time()

                for i, task in enumerate(tasks_to_monitor):
                    if task.done():
                        exc = task.exception()
                        task_name = task.get_name() if hasattr(task, "get_name") else f"Monitored Task {i}"
                        if exc:
                            logger.error(f"{task_name} unexpectedly completed with exception: {exc}")
                        else:
                            logger.info(f"{task_name} completed normally.")

                ffmpeg_idle_time = current_time - self.last_ffmpeg_activity
                if ffmpeg_idle_time > 15:
                    logger.warning(f"FFmpeg idle for {ffmpeg_idle_time:.2f}s - may need attention.")
                    if ffmpeg_idle_time > 30 and not self.is_stopping:
                        logger.error("FFmpeg idle for too long and not in stopping phase, forcing restart.")
                        await self.restart_ffmpeg()
            except asyncio.CancelledError:
                logger.info("Watchdog task cancelled.")
                break
            except Exception as e:
                logger.error(f"Error in watchdog task: {e}", exc_info=True)

    async def cleanup(self):
        """Clean up resources when processing is complete."""
        logger.info("Starting cleanup of AudioProcessor resources.")

        # Stop LLM inference processor first to generate final inference
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

        if self.ffmpeg_process:
            if self.ffmpeg_process.stdin and not self.ffmpeg_process.stdin.closed:
                try:
                    self.ffmpeg_process.stdin.close()
                except Exception as e:
                    logger.warning(f"Error closing ffmpeg stdin during cleanup: {e}")

            # Wait for ffmpeg process to terminate
            if self.ffmpeg_process.poll() is None:  # Check if process is still running
                logger.info("Waiting for FFmpeg process to terminate...")
                try:
                    # Run wait in executor to avoid blocking async loop
                    await asyncio.get_event_loop().run_in_executor(None, self.ffmpeg_process.wait, 5.0)  # 5s timeout
                except Exception as e:  # subprocess.TimeoutExpired is not directly caught by asyncio.wait_for with run_in_executor
                    logger.warning(f"FFmpeg did not terminate gracefully, killing. Error: {e}")
                    self.ffmpeg_process.kill()
                    await asyncio.get_event_loop().run_in_executor(None, self.ffmpeg_process.wait)  # Wait for kill
            logger.info("FFmpeg process terminated.")

        if self.args.diarization and hasattr(self, "diarization") and hasattr(self.diarization, "close"):
            self.diarization.close()
        logger.info("AudioProcessor cleanup complete.")

    async def process_audio(self, message):
        """Process incoming audio data."""
        # If already stopping or stdin is closed, ignore further audio, especially residual chunks.
        if self.is_stopping or (self.ffmpeg_process and self.ffmpeg_process.stdin and self.ffmpeg_process.stdin.closed):
            logger.warning(f"AudioProcessor is stopping or stdin is closed. Ignoring incoming audio message (length: {len(message)}).")
            if not message and self.ffmpeg_process and self.ffmpeg_process.stdin and not self.ffmpeg_process.stdin.closed:
                logger.info("Received empty message while already in stopping state; ensuring stdin is closed.")
                try:
                    self.ffmpeg_process.stdin.close()
                except Exception as e:
                    logger.warning(f"Error closing ffmpeg stdin on redundant stop signal during stopping state: {e}")
            return

        if not message:  # primary signal to start stopping
            logger.info("Empty audio message received, initiating stop sequence.")
            self.is_stopping = True
            if self.ffmpeg_process and self.ffmpeg_process.stdin and not self.ffmpeg_process.stdin.closed:
                try:
                    self.ffmpeg_process.stdin.close()
                    logger.info("FFmpeg stdin closed due to primary stop signal.")
                except Exception as e:
                    logger.warning(f"Error closing ffmpeg stdin on stop: {e}")
            return

        retry_count = 0
        max_retries = 3

        # Log periodic heartbeats showing ongoing audio proc
        current_time = time()
        if not hasattr(self, "_last_heartbeat") or current_time - self._last_heartbeat >= 10:
            logger.debug(f"Processing audio chunk, last FFmpeg activity: {current_time - self.last_ffmpeg_activity:.2f}s ago")
            self._last_heartbeat = current_time

        while retry_count < max_retries:
            try:
                if not self.ffmpeg_process or not hasattr(self.ffmpeg_process, "stdin") or self.ffmpeg_process.poll() is not None:
                    logger.warning("FFmpeg process not available, restarting...")
                    await self.restart_ffmpeg()

                loop = asyncio.get_running_loop()
                try:
                    await asyncio.wait_for(loop.run_in_executor(None, lambda: self.ffmpeg_process.stdin.write(message)), timeout=2.0)
                except asyncio.TimeoutError:
                    logger.warning("FFmpeg write operation timed out, restarting...")
                    await self.restart_ffmpeg()
                    retry_count += 1
                    continue

                try:
                    await asyncio.wait_for(loop.run_in_executor(None, self.ffmpeg_process.stdin.flush), timeout=2.0)
                except asyncio.TimeoutError:
                    logger.warning("FFmpeg flush operation timed out, restarting...")
                    await self.restart_ffmpeg()
                    retry_count += 1
                    continue

                self.last_ffmpeg_activity = time()
                return

            except (BrokenPipeError, AttributeError, OSError) as e:
                retry_count += 1
                logger.warning(f"Error writing to FFmpeg: {e}. Retry {retry_count}/{max_retries}...")

                if retry_count < max_retries:
                    await self.restart_ffmpeg()
                    await asyncio.sleep(0.5)
                else:
                    logger.error("Maximum retries reached for FFmpeg process")
                    await self.restart_ffmpeg()
                    return
