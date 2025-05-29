import asyncio
import json
import logging
import os
from typing import AsyncGenerator

from fastapi import WebSocket, WebSocketDisconnect

from ..processors import AudioProcessor
from .utils import (
    calculate_streaming_params,
    cleanup_temp_file,
    get_audio_duration,
    log_progress,
    read_audio_chunks,
    setup_ffmpeg_process,
    stream_chunks_realtime,
)

logger = logging.getLogger(__name__)


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
        async for response in results_generator:
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
    audio_processor = AudioProcessor()

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
