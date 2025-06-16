import asyncio
from time import time

from ..whisper_streaming.whisper_online import online_factory
from .base_processor import SENTINEL, BaseProcessor, logger, s2hk


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

        # No incremental parsing state needed anymore

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
                    # Store tokens without s2hk conversion - conversion will be applied only at final step
                    self.full_transcription += self.sep.join([t.text for t in new_tokens])
                    # Add minimal token generation logging for UI debugging
                    if len(new_tokens) > 0:
                        logger.debug(f"🎤 Generated {len(new_tokens)} new tokens: '{new_tokens[0].text[:30]}...'")

                    # Get buffer information without s2hk conversion
                    _buffer = self.online.get_buffer()
                    buffer = _buffer.text if _buffer.text else _buffer.text
                    end_buffer = _buffer.end if _buffer.end else (new_tokens[-1].end if new_tokens else 0)

                    # Buffer coordination now handled by work coordination system

                    await update_callback(new_tokens, buffer, end_buffer, self.full_transcription, self.sep)

                    # Work with COMMITTED TRANSCRIPT MEMORY instead of local accumulation
                    if self.coordinator and new_tokens:
                        new_text = self.sep.join([t.text for t in new_tokens])

                        if new_text.strip():
                            # NON-BLOCKING: Update LLM with new transcription text using direct await
                            if self.coordinator.llm:
                                # Get speaker info for LLM context
                                speaker_info_dict = None
                                if new_tokens and hasattr(new_tokens[0], "speaker") and new_tokens[0].speaker is not None:
                                    speaker_info_dict = {"speaker": new_tokens[0].speaker}

                                # Make this completely non-blocking - fire and forget with direct await
                                try:
                                    # Get the entire committed transcript from global memory for LLM analysis
                                    async with self.coordinator.lock:
                                        committed_tokens = self.coordinator.committed_transcript.get("tokens", [])
                                        if committed_tokens:
                                            # Build the full committed transcript text (without s2hk conversion)
                                            full_committed_text = self.sep.join([token.text for token in committed_tokens if token.text])

                                            if full_committed_text and full_committed_text.strip():
                                                # Update LLM with full transcript using direct await (non-blocking)
                                                asyncio.create_task(self.coordinator.llm.update_transcription_direct(full_committed_text, speaker_info_dict))
                                                logger.debug(f"🔄 Updated LLM with {len(full_committed_text)} chars using direct await")
                                except Exception as e:
                                    logger.debug(f"Non-critical LLM update error: {e}")

                            # PARSER SHOULD WORK ON ENTIRE COMMITTED TRANSCRIPT using direct await
                            if self.coordinator and self.coordinator.transcript_parser:
                                # Get the entire committed transcript from global memory
                                try:
                                    async with self.coordinator.lock:
                                        committed_tokens = self.coordinator.committed_transcript.get("tokens", [])
                                        if committed_tokens:
                                            # Build the full committed transcript text (without s2hk conversion)
                                            full_committed_text = self.sep.join([token.text for token in committed_tokens if token.text])

                                            if full_committed_text and full_committed_text.strip():
                                                # Get speaker info from the most recent token
                                                speaker_info = None
                                                if committed_tokens and hasattr(committed_tokens[-1], "speaker") and committed_tokens[-1].speaker is not None:
                                                    speaker_info = [{"speaker": committed_tokens[-1].speaker}]

                                                logger.info(f"✅ Processing entire committed transcript with direct await: {len(full_committed_text)} chars")
                                                # Process the entire committed transcript using direct await (non-blocking)
                                                asyncio.create_task(self._parse_transcript_direct(full_committed_text, speaker_info))
                                except Exception as e:
                                    logger.debug(f"Non-critical parser processing error: {e}")

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

    async def _parse_transcript_direct(self, text_to_parse: str, speaker_info):
        """Parse transcript using direct await (simplified from queue-based processing)."""
        try:
            if self.coordinator and self.coordinator.transcript_parser:
                # Parse transcript using direct await
                parsed_result = await self.coordinator.transcript_parser.parse_transcript_direct(text_to_parse, speaker_info, None)
                if parsed_result:
                    logger.debug(f"✅ Parsed transcript with direct await: {len(text_to_parse)} -> {len(parsed_result.parsed_text)} chars")
                else:
                    logger.debug(f"⚠️ Transcript parsing returned no result")
        except Exception as e:
            logger.debug(f"Transcript parsing error (non-critical): {e}")

    async def _queue_parser_non_blocking(self, text_to_parse: str, speaker_info):
        """Legacy method - now redirects to direct await processing."""
        await self._parse_transcript_direct(text_to_parse, speaker_info)

    async def _get_end_buffer(self):
        """Get current end buffer value - to be implemented by coordinator."""
        return 0

    def finish_transcription(self):
        """Finish the transcription to get any remaining tokens."""
        # Finish the ASR engine first without s2hk conversion
        final_tokens = []
        if self.online:
            try:
                final_result = self.online.finish()
                # Handle both single Transcript object and list of tokens
                if final_result:
                    if hasattr(final_result, "text") and final_result.text:
                        # Single Transcript object - convert to token without s2hk
                        from ..timed_objects import ASRToken

                        final_token = ASRToken(start=getattr(final_result, "start", 0), end=getattr(final_result, "end", 0), text=final_result.text, speaker=getattr(final_result, "speaker", 0))
                        final_tokens = [final_token]
                    elif hasattr(final_result, "__iter__"):
                        # List of tokens without s2hk conversion
                        final_tokens = []
                        for token in final_result:
                            if hasattr(token, "text") and token.text:
                                final_tokens.append(token)
                    else:
                        final_tokens = []
            except Exception as e:
                logger.warning(f"Failed to finish transcription: {e}")
                final_tokens = []

        # FINAL PROCESSING FIX: Simplified approach without async complications
        # The final processing is now handled properly in audio_processor._process_final_results()
        # This method just needs to return the final tokens from ASR
        if self.coordinator and final_tokens:
            try:
                # Add final tokens to global memory synchronously
                # This will be handled by the audio processor's final processing
                logger.info(f"🔄 Returning {len(final_tokens)} final tokens for global memory processing")
            except Exception as e:
                logger.warning(f"Failed to process final tokens: {e}")

        return final_tokens

    async def reset_parsing_state(self):
        """Reset the incremental parsing state for fresh sessions."""
        # No local accumulation to reset - using global memory

        # Also reset parser state if available
        if self.coordinator and self.coordinator.transcript_parser:
            await self.coordinator.transcript_parser.reset_state()

        logger.debug("Reset incremental parsing state")
