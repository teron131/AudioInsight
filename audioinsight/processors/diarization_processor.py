from time import time

from .base_processor import SENTINEL, BaseProcessor, logger


class DiarizationProcessor(BaseProcessor):
    """Handles speaker diarization processing."""

    def __init__(self, args, diarization_obj):
        super().__init__(args)
        self.diarization_obj = diarization_obj

    async def process(self, diarization_queue, get_state_callback, update_callback):
        """Process audio chunks for speaker diarization."""
        buffer_diarization = ""
        processed_chunks = 0

        logger.info("🔊 Diarization processor started")

        while True:
            try:
                pcm_array = await diarization_queue.get()
                if pcm_array is SENTINEL:
                    diarization_queue.task_done()
                    break

                processed_chunks += 1

                # Only log every 120 seconds to reduce spam significantly
                if not hasattr(self, "_last_diarization_log"):
                    self._last_diarization_log = 0
                current_time = time()
                if current_time - self._last_diarization_log > 120.0:
                    logger.info(f"🔊 Diarization: {processed_chunks} chunks processed")
                    self._last_diarization_log = current_time

                # Process diarization
                await self.diarization_obj.diarize(pcm_array)

                # Get current state and update speakers
                state = await get_state_callback()
                new_end = self.diarization_obj.assign_speakers_to_tokens(state["end_attributed_speaker"], state["tokens"])

                await update_callback(new_end, buffer_diarization)
                diarization_queue.task_done()

            except Exception as e:
                logger.warning(f"Exception in diarization_processor: {e}")
                if "pcm_array" in locals() and pcm_array is not SENTINEL:
                    diarization_queue.task_done()
        logger.info("Diarization processor finished.")

    def cleanup(self):
        """Clean up diarization resources."""
        if self.diarization_obj and hasattr(self.diarization_obj, "close"):
            self.diarization_obj.close()
