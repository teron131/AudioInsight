import asyncio
import json
import os
from typing import AsyncGenerator

from fastapi import WebSocket, WebSocketDisconnect

from ..display_parser import get_display_parser
from ..logging_config import get_logger
from ..processors import AudioProcessor
from .utils import (
    calculate_streaming_params,
    cleanup_temp_file,
    log_progress,
    read_audio_chunks,
    setup_ffmpeg_process,
    stream_chunks_realtime,
)

logger = get_logger(__name__)

# Global audio processor to reuse between connections
_global_audio_processor = None
_processor_lock = asyncio.Lock()


async def handle_websocket_results(
    websocket: WebSocket,
    results_generator: AsyncGenerator,
) -> None:
    """Consume results from the audio processor and send them via WebSocket.

    Args:
        websocket: WebSocket connection
        results_generator: Generator yielding audio processing results
    """
    try:
        # Get display parser for text refinement
        display_parser = get_display_parser()

        async for response in results_generator:
            # Apply display text parsing if enabled and if we have lines with text
            if display_parser.is_enabled() and "lines" in response and response["lines"]:
                # Parse text for each line at display layer
                for line in response["lines"]:
                    if line.get("text") and len(line["text"].strip()) > 10:  # Only parse substantial text
                        try:
                            original_text = line["text"]
                            parsed_text = await display_parser.parse_for_display(original_text)
                            line["text"] = parsed_text

                            # Add a flag to indicate text was processed (for debugging)
                            if parsed_text != original_text:
                                line["_text_parsed"] = True

                        except Exception as e:
                            logger.warning(f"Display text parsing failed for line: {e}")
                            # Keep original text on failure

                # Also parse buffer text if available
                if response.get("buffer_transcription") and len(response["buffer_transcription"].strip()) > 10:
                    try:
                        original_buffer = response["buffer_transcription"]
                        parsed_buffer = await display_parser.parse_for_display(original_buffer)
                        response["buffer_transcription"] = parsed_buffer
                    except Exception as e:
                        logger.warning(f"Display text parsing failed for buffer: {e}")
                        # Keep original buffer on failure

            await websocket.send_json(response)

        # Signal completion when generator finishes
        logger.info("Results generator finished. Sending 'ready_to_stop' to client.")
        await websocket.send_json({"type": "ready_to_stop"})

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected while handling results (client likely closed connection).")
    except Exception as e:
        logger.warning(f"Error in WebSocket results handler: {e}")


async def process_file_through_websocket(
    file_path: str,
    duration: float,
    audio_processor: AudioProcessor,
) -> float:
    """Process an audio file through the same pipeline as live recording.

    This function streams the file in real-time to maintain identical processing.

    Args:
        file_path: Path to audio file
        duration: Duration of audio file in seconds
        audio_processor: AudioProcessor instance

    Returns:
        Total elapsed processing time

    Raises:
        Exception: If processing fails
    """
    logger.info(f"Starting unified file processing: {file_path}")

    try:
        # Set up FFmpeg process
        ffmpeg_process = setup_ffmpeg_process(file_path)

        # Read all audio chunks and calculate streaming parameters
        logger.info("Buffering audio data for real-time streaming...")
        chunks = read_audio_chunks(ffmpeg_process)
        total_bytes, bytes_per_second, chunk_interval = calculate_streaming_params(chunks, duration)

        # Stream chunks with real-time pacing
        total_elapsed = await stream_chunks_realtime(
            chunks=chunks,
            chunk_interval=chunk_interval,
            duration=duration,
            process_func=audio_processor.process_audio,
            progress_callback=log_progress,
        )

        logger.info(f"Finished unified file processing: {total_elapsed:.2f}s (target: {duration:.2f}s)")

        return total_elapsed

    finally:
        # Clean up temporary file automatically after processing
        cleanup_temp_file(file_path)


async def handle_websocket_connection(websocket: WebSocket) -> None:
    """Handle WebSocket connection for both live recording and file upload.

    Args:
        websocket: WebSocket connection
    """
    audio_processor = await get_or_create_audio_processor()

    await websocket.accept()
    logger.info("WebSocket connection opened.")

    # Start result handling task
    results_generator = await audio_processor.create_tasks()
    websocket_task = asyncio.create_task(handle_websocket_results(websocket, results_generator))

    try:
        while True:
            message = await websocket.receive()

            if "bytes" in message:
                # Live recording: direct audio data
                await audio_processor.process_audio(message["bytes"])

            elif "text" in message:
                # File upload: JSON message with file info
                try:
                    data = json.loads(message["text"])
                    if data.get("type") == "file_upload":
                        await _handle_file_upload_message(data, audio_processor, websocket)

                except json.JSONDecodeError:
                    logger.warning("Received invalid JSON in WebSocket message")
                except Exception as e:
                    logger.error(f"Error processing file upload message: {e}")
                    await websocket.send_json({"type": "error", "error": f"Error processing file: {str(e)}"})

    except KeyError as e:
        if "bytes" in str(e):
            logger.warning("Client has closed the connection.")
        else:
            logger.error(f"Unexpected KeyError in websocket_endpoint: {e}", exc_info=True)

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected by client during message receiving loop.")

    except RuntimeError as e:
        # Handle receive after disconnect gracefully
        if 'Cannot call "receive"' in str(e):
            logger.info("WebSocket disconnected; exiting receive loop.")
        else:
            logger.error(f"Unexpected RuntimeError in websocket_endpoint main loop: {e}", exc_info=True)

    except Exception as e:
        logger.error(f"Unexpected error in websocket_endpoint main loop: {e}", exc_info=True)

    finally:
        await _cleanup_websocket_connection(websocket_task, audio_processor)


async def _handle_file_upload_message(
    data: dict,
    audio_processor: AudioProcessor,
    websocket: WebSocket,
) -> None:
    """Handle file upload message in WebSocket.

    Args:
        data: File upload message data
        audio_processor: AudioProcessor instance
        websocket: WebSocket connection
    """
    file_path = data.get("file_path")
    duration = data.get("duration", 0)

    if file_path and os.path.exists(file_path):
        logger.info(f"Processing uploaded file via WebSocket: {file_path}")
        await process_file_through_websocket(file_path, duration, audio_processor)
    else:
        logger.warning(f"File not found for WebSocket processing: {file_path}")
        await websocket.send_json({"type": "error", "error": "File not found or invalid file path"})


async def _cleanup_websocket_connection(
    websocket_task: asyncio.Task,
    audio_processor: AudioProcessor,
) -> None:
    """Clean up WebSocket connection resources.

    Args:
        websocket_task: WebSocket results handling task
        audio_processor: AudioProcessor instance
    """
    logger.info("Cleaning up WebSocket endpoint...")

    if not websocket_task.done():
        websocket_task.cancel()

    try:
        await websocket_task
    except asyncio.CancelledError:
        logger.info("WebSocket results handler task was cancelled.")
    except Exception as e:
        logger.warning(f"Exception while awaiting websocket_task completion: {e}")

    await audio_processor.cleanup()
    logger.info("WebSocket endpoint cleaned up successfully.")


async def get_or_create_audio_processor() -> AudioProcessor:
    """Get or create a clean AudioProcessor instance.

    This function ensures we reuse the same processor instance but reset it between sessions
    to prevent memory leaks while maintaining performance.
    """
    global _global_audio_processor

    async with _processor_lock:
        if _global_audio_processor is None:
            logger.info("🔧 Creating new AudioProcessor instance")
            _global_audio_processor = AudioProcessor()
        else:
            # Reset the existing processor for a fresh session
            logger.info("🔄 Resetting existing AudioProcessor for new session")
            try:
                await _global_audio_processor.force_reset()
            except Exception as e:
                logger.warning(f"Error during processor reset, creating new instance: {e}")
                # If reset fails, create a completely new instance
                try:
                    await _global_audio_processor.cleanup()
                except:
                    pass
                _global_audio_processor = AudioProcessor()

        return _global_audio_processor


async def cleanup_global_processor():
    """Clean up the global audio processor completely.

    This should be called when the server is shutting down or for complete cleanup.
    """
    global _global_audio_processor

    async with _processor_lock:
        if _global_audio_processor is not None:
            logger.info("🧹 Cleaning up global AudioProcessor")
            try:
                await _global_audio_processor.cleanup()
            except Exception as e:
                logger.warning(f"Error during global processor cleanup: {e}")
            finally:
                _global_audio_processor = None
