import asyncio
import math
from time import sleep, time

import ffmpeg
import numpy as np

from .base_processor import SENTINEL, BaseProcessor, logger


class FFmpegProcessor(BaseProcessor):
    """Handles FFmpeg process management and audio data conversion."""

    def __init__(self, args):
        super().__init__(args)
        self._is_shutting_down = False  # Track cleanup state to prevent restart loops
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
        self.ffmpeg_health_check_interval = 10  # Less frequent health checks
        self.ffmpeg_max_idle_time = 30  # Allow much longer idle time before restart

        # FFmpeg process
        self.ffmpeg_process = self.start_ffmpeg_decoder()

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
        try:
            # Close any existing stdout/stderr before creating new process
            if hasattr(self, "ffmpeg_process") and self.ffmpeg_process:
                try:
                    if self.ffmpeg_process.stdout and not self.ffmpeg_process.stdout.closed:
                        self.ffmpeg_process.stdout.close()
                    if self.ffmpeg_process.stderr and not self.ffmpeg_process.stderr.closed:
                        self.ffmpeg_process.stderr.close()
                except:
                    pass

            # Create new FFmpeg process with explicit error handling
            process = ffmpeg.input("pipe:0", format="webm").output("pipe:1", format="s16le", acodec="pcm_s16le", ac=self.channels, ar=self.sample_rate_str).run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True, quiet=True)  # Suppress FFmpeg output to avoid buffer issues

            # Verify the process started correctly
            if process.poll() is not None:
                raise RuntimeError("FFmpeg process failed to start")

            return process

        except Exception as e:
            logger.error(f"Failed to start FFmpeg process: {e}")
            raise

    async def restart_ffmpeg(self):
        """Restart the FFmpeg process after failure."""
        logger.warning("Restarting FFmpeg process...")

        # Force cleanup of existing process more aggressively
        if self.ffmpeg_process:
            try:
                # Close stdin immediately to signal shutdown
                if self.ffmpeg_process.stdin and not self.ffmpeg_process.stdin.closed:
                    try:
                        self.ffmpeg_process.stdin.close()
                    except:
                        pass

                # Close stdout and stderr
                if self.ffmpeg_process.stdout and not self.ffmpeg_process.stdout.closed:
                    try:
                        self.ffmpeg_process.stdout.close()
                    except:
                        pass

                if self.ffmpeg_process.stderr and not self.ffmpeg_process.stderr.closed:
                    try:
                        self.ffmpeg_process.stderr.close()
                    except:
                        pass

                # Terminate the process if it's still running
                if self.ffmpeg_process.poll() is None:
                    logger.info("Terminating existing FFmpeg process")
                    self.ffmpeg_process.terminate()

                    # Wait for termination with timeout
                    try:
                        await asyncio.wait_for(asyncio.get_event_loop().run_in_executor(None, self.ffmpeg_process.wait), timeout=3.0)
                    except asyncio.TimeoutError:
                        logger.warning("FFmpeg process did not terminate, killing forcefully")
                        self.ffmpeg_process.kill()
                        try:
                            await asyncio.wait_for(asyncio.get_event_loop().run_in_executor(None, self.ffmpeg_process.wait), timeout=2.0)
                        except asyncio.TimeoutError:
                            logger.error("FFmpeg process could not be killed")

            except Exception as e:
                logger.error(f"Error during FFmpeg process termination: {e}")

        # Clear the reference
        self.ffmpeg_process = None

        # Wait a moment before restarting to ensure cleanup
        await asyncio.sleep(0.5)

        # Start new process with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                logger.info(f"Starting new FFmpeg process (attempt {attempt + 1}/{max_retries})")
                self.ffmpeg_process = self.start_ffmpeg_decoder()
                self.pcm_buffer_length = 0  # Reset buffer length for new process
                self.last_ffmpeg_activity = time()
                self._is_shutting_down = False  # Reset shutdown flag on successful restart
                logger.info("FFmpeg process restarted successfully")
                return

            except Exception as e:
                logger.error(f"Failed to restart FFmpeg process (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    # Wait longer between retries
                    await asyncio.sleep(1.0 * (attempt + 1))
                else:
                    logger.error("Maximum retries reached for FFmpeg process - continuing without restart")
                    return

        logger.warning("All FFmpeg processing attempts failed but continuing")
        return

    async def read_audio_data(self, transcription_queue=None, diarization_queue=None):
        """Read audio data from FFmpeg stdout and process it."""
        loop = asyncio.get_event_loop()
        beg = time()

        while True:
            try:
                current_time = time()
                elapsed_time = math.floor((current_time - beg) * 10) / 10
                buffer_size = max(int(32000 * elapsed_time), 4096)
                beg = current_time

                # Detect idle state much more conservatively
                # Only restart if idle for more than 10 minutes (600 seconds) AND process is broken
                if current_time - self.last_ffmpeg_activity > 600.0:
                    # Check if process is actually broken before restarting
                    if self.ffmpeg_process is None or self.ffmpeg_process.poll() is not None or (self.ffmpeg_process.stdin and self.ffmpeg_process.stdin.closed):
                        logger.warning(f"FFmpeg process idle for {current_time - self.last_ffmpeg_activity:.2f}s and appears broken. Restarting...")
                    await self.restart_ffmpeg()
                    beg = time()
                    self.last_ffmpeg_activity = time()
                    continue
                else:
                    # Process is healthy but idle - this is normal, just update activity time
                    logger.debug(f"FFmpeg idle for {current_time - self.last_ffmpeg_activity:.2f}s but process is healthy")
                    self.last_ffmpeg_activity = current_time  # Reset the idle timer

                chunk = await loop.run_in_executor(None, self.ffmpeg_process.stdout.read, buffer_size)
                if chunk:
                    self.last_ffmpeg_activity = time()

                if not chunk:
                    logger.info("FFmpeg stdout closed, no more data to read.")
                    break

                self.append_to_pcm_buffer(chunk)

                # OPTIMIZATION: Increase minimum buffer for transcription to reduce processing frequency
                # Use larger buffer requirement to avoid VAD processing tiny chunks
                min_transcription_buffer = max(self.bytes_per_sec // 2, 8192)  # 0.5 seconds minimum (increased from 0.25s)
                if self.pcm_buffer_length >= min_transcription_buffer:
                    # Only log processing every 20 seconds and make it more concise
                    if not hasattr(self, "_ffmpeg_log_counter"):
                        self._ffmpeg_log_counter = 0
                        self._last_processing_log = 0
                    self._ffmpeg_log_counter += 1

                    current_log_time = time()
                    if current_log_time - self._last_processing_log > 20.0:  # Log every 20 seconds, less frequently
                        logger.info(f"🎵 Processing: {self._ffmpeg_log_counter} chunks, {self.pcm_buffer_length / self.bytes_per_sec:.1f}s buffered")
                        self._last_processing_log = current_log_time

                    if self.pcm_buffer_length > self.max_bytes_per_sec:
                        logger.warning(f"Audio buffer large: {self.pcm_buffer_length / self.bytes_per_sec:.2f}s")

                    # Process audio chunk - take larger chunks to reduce Whisper VAD calls
                    bytes_to_process = min(self.pcm_buffer_length, self.max_bytes_per_sec)
                    pcm_array = self.convert_pcm_to_float(self.get_pcm_data(bytes_to_process))

                    # Send to transcription if enabled
                    if self.args.transcription and transcription_queue:
                        await transcription_queue.put(pcm_array.copy())

                    # Send to diarization if enabled - use larger chunks less frequently to prevent backlog
                    if self.args.diarization and diarization_queue:
                        # Only send to diarization every 2 seconds worth of audio to prevent queue overload
                        if not hasattr(self, "_diarization_chunk_buffer"):
                            self._diarization_chunk_buffer = []
                            self._diarization_buffer_size = 0

                        self._diarization_chunk_buffer.append(pcm_array.copy())
                        self._diarization_buffer_size += len(pcm_array)

                        # Send larger chunks (2 seconds worth) to diarization to reduce processing frequency
                        diarization_chunk_threshold = self.bytes_per_sec * 2  # 2 seconds of audio
                        if self._diarization_buffer_size >= diarization_chunk_threshold:
                            # Check queue size to prevent overload
                            if diarization_queue.qsize() < 5:  # Limit queue to 5 items (10 seconds of audio)
                                # Concatenate buffered chunks
                                combined_audio = np.concatenate(self._diarization_chunk_buffer)
                                await diarization_queue.put(combined_audio)
                                # Remove verbose diarization logging entirely - only log errors
                            else:
                                # Only warn occasionally about queue being full
                                if not hasattr(self, "_last_queue_warning"):
                                    self._last_queue_warning = 0
                                if time() - self._last_queue_warning > 10.0:  # Warn every 10 seconds
                                    logger.warning(f"Diarization queue full ({diarization_queue.qsize()} items)")
                                    self._last_queue_warning = time()

                            # Reset buffer regardless of whether we sent data
                            self._diarization_chunk_buffer = []
                            self._diarization_buffer_size = 0

                    # Sleep if no processing is happening
                    if not self.args.transcription and not self.args.diarization:
                        await asyncio.sleep(0.1)

            except Exception as e:
                logger.warning(f"Exception in ffmpeg_stdout_reader: {e}")
                break

        logger.info("FFmpeg processing finished. Signaling downstream processors.")
        if self.args.transcription and transcription_queue:
            await transcription_queue.put(SENTINEL)
        if self.args.diarization and diarization_queue:
            await diarization_queue.put(SENTINEL)

    async def process_audio_chunk(self, message):
        """Process incoming audio data."""
        retry_count = 0
        max_retries = 1  # Reduce max retries to prevent restart loops

        # Check if we're in cleanup/shutdown mode - don't process if so
        if hasattr(self, "_is_shutting_down") and self._is_shutting_down:
            logger.debug("FFmpeg processor is shutting down, ignoring audio chunk")
            return

        while retry_count < max_retries:
            try:
                # Check if FFmpeg process is available and healthy
                if not self.ffmpeg_process or not hasattr(self.ffmpeg_process, "stdin") or self.ffmpeg_process.poll() is not None:
                    # Don't restart if we're shutting down
                    if hasattr(self, "_is_shutting_down") and self._is_shutting_down:
                        logger.debug("FFmpeg process unavailable during shutdown, not restarting")
                        return
                    logger.warning("FFmpeg process not available, restarting...")
                    await self.restart_ffmpeg()
                    retry_count += 1
                    continue

                # Check if stdin is still open
                if self.ffmpeg_process.stdin.closed:
                    # Don't restart if we're shutting down
                    if hasattr(self, "_is_shutting_down") and self._is_shutting_down:
                        logger.debug("FFmpeg stdin closed during shutdown, not restarting")
                        return
                    logger.warning("FFmpeg stdin is closed, restarting...")
                    await self.restart_ffmpeg()
                    retry_count += 1
                    continue

                loop = asyncio.get_running_loop()

                # Write operation with more generous timeout to prevent restarts
                write_timeout = 10.0 if retry_count > 0 else 8.0  # Much more generous timeouts
                try:
                    await asyncio.wait_for(loop.run_in_executor(None, lambda: self.ffmpeg_process.stdin.write(message)), timeout=write_timeout)
                except asyncio.TimeoutError:
                    logger.warning(f"FFmpeg write timeout ({write_timeout}s) - may be processing heavy load")
                    # Don't restart immediately on timeout - just log and continue
                    self.last_ffmpeg_activity = time()
                    return

                # Flush operation with more generous timeout
                flush_timeout = 6.0 if retry_count > 0 else 4.0  # Much more generous timeouts
                try:
                    await asyncio.wait_for(loop.run_in_executor(None, self.ffmpeg_process.stdin.flush), timeout=flush_timeout)
                except asyncio.TimeoutError:
                    logger.warning(f"FFmpeg flush timeout ({flush_timeout}s) - may be processing heavy load")
                    # Don't restart immediately on timeout - just log and continue
                    self.last_ffmpeg_activity = time()
                    return

                # Success - update activity time and return
                self.last_ffmpeg_activity = time()
                return

            except (BrokenPipeError, AttributeError, OSError) as e:
                retry_count += 1
                logger.warning(f"FFmpeg error: {e}. Retry {retry_count}/{max_retries}")

                if retry_count < max_retries:
                    await self.restart_ffmpeg()
                    await asyncio.sleep(1.0)  # Longer wait between retries
                else:
                    logger.error("Maximum retries reached for FFmpeg process - continuing without restart")
                    return

        logger.warning("All FFmpeg processing attempts failed but continuing")
        return

    def cleanup(self):
        """Clean up FFmpeg resources."""
        logger.info("Starting FFmpeg cleanup...")

        # Set shutdown flag to prevent restart loops during cleanup
        self._is_shutting_down = True

        if self.ffmpeg_process:
            try:
                # Close all file descriptors first
                if self.ffmpeg_process.stdin and not self.ffmpeg_process.stdin.closed:
                    try:
                        self.ffmpeg_process.stdin.close()
                        logger.debug("FFmpeg stdin closed")
                    except Exception as e:
                        logger.warning(f"Error closing ffmpeg stdin during cleanup: {e}")

                if self.ffmpeg_process.stdout and not self.ffmpeg_process.stdout.closed:
                    try:
                        self.ffmpeg_process.stdout.close()
                        logger.debug("FFmpeg stdout closed")
                    except Exception as e:
                        logger.warning(f"Error closing ffmpeg stdout during cleanup: {e}")

                if self.ffmpeg_process.stderr and not self.ffmpeg_process.stderr.closed:
                    try:
                        self.ffmpeg_process.stderr.close()
                        logger.debug("FFmpeg stderr closed")
                    except Exception as e:
                        logger.warning(f"Error closing ffmpeg stderr during cleanup: {e}")

                # Terminate the process if it's still running
                if self.ffmpeg_process.poll() is None:
                    logger.info("Terminating FFmpeg process during cleanup...")
                    try:
                        self.ffmpeg_process.terminate()
                        # Wait for termination with timeout
                        try:
                            self.ffmpeg_process.wait(timeout=3.0)
                            logger.debug("FFmpeg process terminated gracefully")
                        except:  # subprocess.TimeoutExpired and other exceptions
                            logger.warning("FFmpeg did not terminate gracefully, killing forcefully")
                            self.ffmpeg_process.kill()
                            try:
                                self.ffmpeg_process.wait(timeout=2.0)
                                logger.debug("FFmpeg process killed successfully")
                            except:
                                logger.error("FFmpeg process could not be killed")
                    except Exception as e:
                        logger.warning(f"Error terminating FFmpeg process during cleanup: {e}")
                else:
                    logger.debug("FFmpeg process already terminated")

            except Exception as e:
                logger.error(f"Error during FFmpeg process cleanup: {e}")
            finally:
                # Always clear the reference
                self.ffmpeg_process = None

        # Clear all memory buffers aggressively
        if hasattr(self, "pcm_buffer"):
            self.pcm_buffer = bytearray(self.max_buffer_size)
            self.pcm_buffer_length = 0

        # Clear pre-allocated arrays
        if hasattr(self, "_temp_int16_array"):
            self._temp_int16_array.fill(0)
        if hasattr(self, "_temp_float32_array"):
            self._temp_float32_array.fill(0)

        # Clear any diarization buffers
        if hasattr(self, "_diarization_chunk_buffer"):
            self._diarization_chunk_buffer.clear()
            self._diarization_buffer_size = 0

        # Reset timing
        self.last_ffmpeg_activity = time()

        logger.info("FFmpeg cleanup completed successfully")
