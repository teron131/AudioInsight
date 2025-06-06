import asyncio
import math
from time import time

from ..whisper_streaming.whisper_online import online_factory
from .base_processor import SENTINEL, BaseProcessor, _sentence_split_regex, logger, s2hk


class TranscriptionProcessor(BaseProcessor):
    """Handles speech-to-text transcription processing."""

    def __init__(self, args, asr, tokenizer, coordinator=None):
        super().__init__(args)
        self.coordinator = coordinator  # Reference to AudioProcessor for timing
        self.online = None
        self.full_transcription = ""
        self.sep = " "  # Default separator

        # No separate accumulation - will read from coordinator's global memory

        # Initialize transcription engine if enabled
        if self.args.transcription:
            self.online = online_factory(self.args, asr, tokenizer)
            self.sep = self.online.asr.sep

        # Incremental parsing optimization - track what's already been parsed
        self.last_parsed_text = ""  # Track what text has been parsed to avoid re-processing
        self.min_text_threshold = 100  # Variable: parse all if text < this many chars
        self.sentence_percentage = 0.4  # Variable: parse last 40% of sentences if text >= threshold

    async def start_parser_worker(self):
        """Start the parser worker task that processes queued text."""
        if self.coordinator and self.coordinator.transcript_parser:
            await self.coordinator.transcript_parser.start_worker()
            logger.info("Started parser worker from transcript parser")

    async def stop_parser_worker(self):
        """Stop the parser worker task."""
        if self.coordinator and self.coordinator.transcript_parser:
            await self.coordinator.transcript_parser.stop_worker()
            logger.info("Stopped parser worker from transcript parser")

    def _get_text_to_parse(self, current_text: str) -> str:
        """Get only the new text that needs parsing using intelligent sentence-based splitting.

        Args:
            current_text: The full accumulated text

        Returns:
            str: Only the new text that needs to be parsed, or empty string if nothing new
        """
        # If no previous parsing, handle based on text length
        if not self.last_parsed_text:
            if len(current_text) < self.min_text_threshold:
                # Parse all for short text
                logger.info(f"🔍 Incremental parsing: No previous parsing, text < {self.min_text_threshold} chars, parsing ALL")
                return current_text
            else:
                # Parse last 25% of sentences for longer text
                result = self._get_last_sentences_percentage(current_text)
                logger.info(f"🔍 Incremental parsing: No previous parsing, text >= {self.min_text_threshold} chars, parsing last 25% of sentences ({len(result)} chars)")
                return result

        # Find new text since last parsing
        if self.last_parsed_text in current_text:
            # Get the part after the last parsed text
            last_index = current_text.rfind(self.last_parsed_text)
            new_text_start = last_index + len(self.last_parsed_text)
            new_text = current_text[new_text_start:].strip()

            if not new_text:
                logger.info("🔍 Incremental parsing: No new text to parse")
                return ""  # No new text to parse

            # For incremental updates, always parse the new content
            logger.info(f"🔍 Incremental parsing: Found new text since last parsing ({len(new_text)} chars from {len(current_text)} total)")
            return new_text
        else:
            # Text doesn't contain last parsed content (maybe reset occurred)
            # Fall back to percentage-based parsing
            if len(current_text) < self.min_text_threshold:
                logger.info(f"🔍 Incremental parsing: Text doesn't contain last parsed content, < {self.min_text_threshold} chars, parsing ALL")
                return current_text
            else:
                result = self._get_last_sentences_percentage(current_text)
                logger.info(f"🔍 Incremental parsing: Text doesn't contain last parsed content, >= {self.min_text_threshold} chars, parsing last 25% ({len(result)} chars)")
                return result

    def _get_last_sentences_percentage(self, text: str) -> str:
        """Get the last 25% of sentences from the text (rounded up).

        Args:
            text: The text to split into sentences

        Returns:
            str: The last 25% of sentences joined together
        """
        # Split by common punctuation (sentences)
        sentences = _sentence_split_regex.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            return text

        # Calculate 25% rounded up
        num_sentences = len(sentences)
        percentage_count = max(1, math.ceil(num_sentences * self.sentence_percentage))

        # Get last N sentences
        last_sentences = sentences[-percentage_count:]
        result = ". ".join(last_sentences)

        logger.debug(f"Sentence-based parsing: {num_sentences} total sentences, processing last {percentage_count} sentences ({len(result)} chars)")
        return result

    async def process(self, transcription_queue, update_callback, llm=None):
        """Process audio chunks for transcription."""
        self.full_transcription = ""
        if self.online:
            self.sep = self.online.asr.sep

        logger.info("🎙️ Transcription processor started")

        while True:
            try:
                pcm_array = await transcription_queue.get()
                if pcm_array is SENTINEL:
                    transcription_queue.task_done()
                    break

                # OPTIMIZATION: Reduce logging frequency to prevent spam
                if not hasattr(self, "_transcription_log_counter"):
                    self._transcription_log_counter = 0
                    self._last_chunk_log = 0
                self._transcription_log_counter += 1

                # Log every 30 seconds instead of 120 seconds for better visibility
                current_log_time = time()
                if current_log_time - self._last_chunk_log > 30.0:
                    logger.info(f"🎵 Transcription: {self._transcription_log_counter} chunks processed")
                    self._last_chunk_log = current_log_time

                if not self.online:  # Should not happen if queue is used
                    logger.warning("Transcription processor: self.online not initialized.")
                    transcription_queue.task_done()
                    continue

                asr_internal_buffer_duration_s = len(self.online.audio_buffer) / self.online.SAMPLING_RATE

                # Calculate timing like the original - use coordinator references
                transcription_lag_s = 0.0
                if self.coordinator:
                    transcription_lag_s = max(0.0, time() - self.coordinator.beg_loop - self.coordinator.end_buffer)

                # OPTIMIZATION: Reduce ASR processing logs significantly
                if not hasattr(self, "_last_asr_log_time"):
                    self._last_asr_log_time = 0
                current_time = time()
                if current_time - self._last_asr_log_time > 15.0:  # Log every 15 seconds (reduced from 60s)
                    logger.info(f"ASR: buffer={asr_internal_buffer_duration_s:.1f}s, lag={transcription_lag_s:.1f}s")
                    self._last_asr_log_time = current_time

                # Process transcription
                self.online.insert_audio_chunk(pcm_array)
                new_tokens = self.online.process_iter()

                if new_tokens:
                    self.full_transcription += self.sep.join([t.text for t in new_tokens])
                    # Add minimal token generation logging for UI debugging
                    if len(new_tokens) > 0:
                        logger.debug(f"🎤 Generated {len(new_tokens)} new tokens: '{new_tokens[0].text[:30]}...'")

                    # Get buffer information
                    _buffer = self.online.get_buffer()
                    buffer = _buffer.text
                    end_buffer = _buffer.end if _buffer.end else (new_tokens[-1].end if new_tokens else 0)

                    # Buffer coordination now handled by work coordination system

                    await update_callback(new_tokens, buffer, end_buffer, self.full_transcription, self.sep)

                    # Work with GLOBAL MEMORY instead of local accumulation
                    if self.coordinator and new_tokens:
                        new_text = self.sep.join([t.text for t in new_tokens])
                        # Convert to Traditional Chinese for consistency
                        new_text_converted = s2hk(new_text) if new_text else new_text

                        if new_text_converted.strip():
                            # NON-BLOCKING: Update LLM with new transcription text in background
                            if self.coordinator.llm:
                                # Get speaker info for LLM context
                                speaker_info_dict = None
                                if new_tokens and hasattr(new_tokens[0], "speaker") and new_tokens[0].speaker is not None:
                                    speaker_info_dict = {"speaker": new_tokens[0].speaker}

                                # Make this completely non-blocking - fire and forget
                                try:
                                    self.coordinator.llm.update_transcription(new_text_converted, speaker_info_dict)
                                    logger.debug(f"🔄 Updated LLM with {len(new_text_converted)} chars: '{new_text_converted[:50]}...'")
                                except Exception as e:
                                    logger.debug(f"Non-critical LLM update error: {e}")

                            # ATOMIC: Get current global transcript content for parsing decisions
                            async with self.coordinator.lock:
                                # Build current full text from global memory
                                committed_text = self.coordinator.sep.join([token.text for token in self.coordinator.global_transcript["committed_tokens"] if token.text and token.text.strip()])
                                current_buffer = self.coordinator.global_transcript["current_buffer"]
                                full_current_text = (committed_text + " " + current_buffer).strip() if current_buffer else committed_text

                            # Get accumulated speaker info (use most recent speaker)
                            speaker_info = None
                            if new_tokens and hasattr(new_tokens[0], "speaker") and new_tokens[0].speaker is not None:
                                speaker_info = [{"speaker": new_tokens[0].speaker}]

                            # Use event-based Parser system reading from global memory
                            min_batch_size = 200  # Maintain threshold for batching

                            if len(full_current_text) >= min_batch_size:
                                # OPTIMIZATION: Use intelligent incremental parsing from global memory
                                text_to_parse = self._get_text_to_parse(full_current_text)

                                if text_to_parse and text_to_parse.strip() and self.coordinator.transcript_parser:
                                    # OPTIMIZATION: Smart batching to reduce API calls
                                    current_time = time()
                                    time_since_last_parse = current_time - getattr(self, "_last_parse_time", 0)
                                    min_parse_interval = 2.0  # Minimum 2 seconds between parsing requests

                                    should_parse_now = len(text_to_parse) >= 400 or time_since_last_parse >= min_parse_interval

                                    if should_parse_now:
                                        # CRITICAL: Make parser request completely non-blocking
                                        try:
                                            # Fire and forget - don't await the parser queue
                                            asyncio.create_task(self._queue_parser_non_blocking(text_to_parse, speaker_info))
                                            logger.info(f"✅ Queued {len(text_to_parse)} chars for non-blocking parser processing (from {len(full_current_text)} total in global memory)")

                                            # Update last parsed text to track progress
                                            self.last_parsed_text = full_current_text
                                            self._last_parse_time = current_time

                                        except Exception as e:
                                            logger.debug(f"Non-critical parser queue error: {e}")
                                    else:
                                        logger.debug(f"⏳ Delaying parse - batch size: {len(text_to_parse)}, time since last: {time_since_last_parse:.1f}s")
                                else:
                                    logger.debug("⚠️ No new text to parse from global memory")
                            else:
                                # Continue building text in global memory until batch size is reached
                                logger.debug(f"⏳ Building text in global memory: {len(full_current_text)}/{min_batch_size} chars")

                    # Get accumulated speaker info (use most recent speaker)
                    speaker_info = None
                    if new_tokens and hasattr(new_tokens[0], "speaker") and new_tokens[0].speaker is not None:
                        speaker_info = [{"speaker": new_tokens[0].speaker}]

                transcription_queue.task_done()

            except Exception as e:
                logger.warning(f"Exception in transcription_processor: {e}")
                if "pcm_array" in locals() and pcm_array is not SENTINEL:  # Check if pcm_array was assigned from queue
                    transcription_queue.task_done()
        logger.info("Transcription processor finished.")

    async def _update_coordinator_parser_async(self, coordinator, text, speaker_info):
        """Asynchronously update coordinator's transcript parser without blocking transcription."""
        try:
            if coordinator and hasattr(coordinator, "parse_and_store_transcript"):
                # Convert speaker_info to list format expected by parser
                speaker_list = [speaker_info] if speaker_info else None
                await coordinator.parse_and_store_transcript(text, speaker_list)
        except Exception as e:
            # Log errors but don't let them affect transcription
            logger.warning(f"Transcript parsing update failed (non-critical): {e}")

    async def _queue_parser_non_blocking(self, text_to_parse: str, speaker_info):
        """Queue parser request without blocking transcription processing."""
        try:
            if self.coordinator and self.coordinator.transcript_parser:
                success = await self.coordinator.transcript_parser.queue_parsing_request(text_to_parse, speaker_info, None)
                if success:
                    logger.debug(f"✅ Parser queue accepted {len(text_to_parse)} chars")
                else:
                    logger.debug(f"⏳ Parser queue busy, will retry later")
        except Exception as e:
            logger.debug(f"Parser queue error (non-critical): {e}")

    async def _get_end_buffer(self):
        """Get current end buffer value - to be implemented by coordinator."""
        return 0

    def finish_transcription(self):
        """Finish the transcription to get any remaining tokens."""
        # Work with global memory instead of local accumulation
        if self.coordinator:
            try:
                # ATOMIC: Get final content from global memory
                async def flush_global_memory():
                    async with self.coordinator.lock:
                        # Build final text from global memory
                        committed_text = self.coordinator.sep.join([token.text for token in self.coordinator.global_transcript["committed_tokens"] if token.text and token.text.strip()])
                        current_buffer = self.coordinator.global_transcript["current_buffer"]
                        final_text = (committed_text + " " + current_buffer).strip() if current_buffer else committed_text

                        # Clear buffer since we're finishing
                        self.coordinator.global_transcript["current_buffer"] = ""

                        return final_text

                # Get final text from global memory
                import asyncio

                try:
                    # Try to get event loop, create one if needed
                    loop = asyncio.get_running_loop()
                    final_text = loop.run_until_complete(flush_global_memory())
                except RuntimeError:
                    # No event loop running
                    final_text = asyncio.run(flush_global_memory())

                if final_text.strip():
                    # Update LLM with final text
                    if self.coordinator.llm:
                        self.coordinator.llm.update_transcription(final_text, None)
                        logger.info(f"🔄 Updated LLM with final text from global memory: {len(final_text)} chars")

                    # Queue final text for parsing
                    if self.coordinator.transcript_parser:
                        text_to_parse = self._get_text_to_parse(final_text)
                        if text_to_parse and text_to_parse.strip():
                            # Use the async queue method
                            asyncio.create_task(self.coordinator.transcript_parser.queue_parsing_request(text_to_parse, None, None))
                            self.last_parsed_text = final_text
                            logger.info(f"Queued final {len(text_to_parse)} chars for parsing from global memory")

            except Exception as e:
                logger.warning(f"Failed to process final text from global memory: {e}")

        # Finish the ASR engine
        if self.online:
            try:
                return self.online.finish()
            except Exception as e:
                logger.warning(f"Failed to finish transcription: {e}")
        return None

    async def reset_parsing_state(self):
        """Reset the incremental parsing state for fresh sessions."""
        self.last_parsed_text = ""
        # No local accumulation to reset - using global memory

        # Also reset parser state if available
        if self.coordinator and self.coordinator.transcript_parser:
            await self.coordinator.transcript_parser.reset_state()

        logger.debug("Reset incremental parsing state")
