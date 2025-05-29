import asyncio
import json
import time
from pathlib import Path
from typing import AsyncGenerator, Dict

from fastapi import HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from ..logging_config import get_logger
from ..processors import AudioProcessor
from .config import ALLOWED_AUDIO_TYPES, SSE_HEADERS
from .utils import (
    calculate_streaming_params,
    cleanup_temp_file,
    create_temp_file,
    get_audio_duration,
    log_progress,
    read_audio_chunks,
    setup_ffmpeg_process,
    stream_chunks_realtime,
    validate_file_type,
)

logger = get_logger(__name__)


async def handle_file_upload_for_websocket(file: UploadFile) -> Dict:
    """Upload file and return file info for WebSocket processing.

    Args:
        file: Uploaded audio file

    Returns:
        Dictionary with upload status and file information

    Raises:
        HTTPException: If file validation or processing fails
    """
    # Validate file type
    if not validate_file_type(file.content_type, ALLOWED_AUDIO_TYPES, file.filename):
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}. " f"Supported types: {', '.join(ALLOWED_AUDIO_TYPES)}")

    logger.info(f"Uploading file for WebSocket processing: {file.filename} ({file.content_type})")

    try:
        # Create temporary file
        content = await file.read()
        temp_file_path = create_temp_file(content, Path(file.filename).suffix)

        # Get audio duration
        duration = get_audio_duration(temp_file_path)

        return {
            "status": "success",
            "filename": file.filename,
            "file_path": temp_file_path,
            "duration": duration,
            "message": "File uploaded successfully. Use WebSocket to process.",
        }

    except Exception as e:
        # Clean up on error
        if "temp_file_path" in locals():
            cleanup_temp_file(temp_file_path)

        logger.error(f"Error uploading file {file.filename}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")


async def handle_file_upload_and_process(file: UploadFile) -> Dict:
    """Upload and process an audio file for transcription with real-time streaming simulation.

    Args:
        file: Audio file to transcribe

    Returns:
        JSON response with transcription results

    Raises:
        HTTPException: If file validation or processing fails
    """
    # Validate file type
    if not validate_file_type(file.content_type, ALLOWED_AUDIO_TYPES, file.filename):
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}. " f"Supported types: {', '.join(ALLOWED_AUDIO_TYPES)}")

    logger.info(f"Processing uploaded file: {file.filename} ({file.content_type})")

    temp_file_path = None
    try:
        # Create temporary file
        content = await file.read()
        temp_file_path = create_temp_file(content, Path(file.filename).suffix)

        # Get audio duration
        duration = get_audio_duration(temp_file_path)
        logger.info(f"Audio duration: {duration:.2f} seconds - will simulate real-time playback")

        # Process the file
        transcription_results = await _process_file_for_transcription(temp_file_path, duration)

        processing_time = transcription_results.get("processing_time", 0)
        results_count = len(transcription_results.get("transcription", []))

        logger.info(f"Successfully processed file: {file.filename}, results: {results_count} segments")

        return {
            "status": "success",
            "filename": file.filename,
            "transcription": transcription_results["transcription"],
            "message": f"File processed in real-time simulation ({processing_time:.1f}s for {duration:.1f}s audio)",
            "audio_duration": duration,
            "processing_time": processing_time,
        }

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error processing uploaded file {file.filename}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

    finally:
        # Clean up temporary file
        if temp_file_path:
            cleanup_temp_file(temp_file_path)


async def handle_file_upload_stream(file: UploadFile) -> StreamingResponse:
    """Upload and process an audio file with real-time streaming results using Server-Sent Events.

    Args:
        file: Audio file to transcribe

    Returns:
        StreamingResponse with SSE events containing transcription results
    """
    # Read file content before creating the async generator
    try:
        file_content = await file.read()
    except Exception as e:
        return _create_error_stream(f"Failed to read uploaded file: {str(e)}")

    async def event_stream() -> AsyncGenerator[str, None]:
        """Generate Server-Sent Events for file processing."""
        temp_file_path = None
        try:
            # Validate file type
            if not validate_file_type(file.content_type, ALLOWED_AUDIO_TYPES, file.filename):
                yield f"event: error\ndata: {{'error': 'Unsupported file type: {file.content_type}'}}\n\n"
                return

            logger.info(f"Processing uploaded file for streaming: {file.filename} ({file.content_type})")

            # Create temporary file
            temp_file_path = create_temp_file(file_content, Path(file.filename).suffix)

            # Get audio duration
            duration = get_audio_duration(temp_file_path)
            logger.info(f"Audio duration for streaming: {duration:.2f} seconds")

            # Send initial info
            yield f"event: start\ndata: {{'filename': '{file.filename}', 'duration': {duration}}}\n\n"

            # Process file and stream results
            async for event in _stream_file_processing_events(temp_file_path, duration):
                yield event

        except Exception as e:
            logger.error(f"Error in SSE upload stream: {str(e)}", exc_info=True)
            yield f"event: error\ndata: {{'error': 'Server error: {str(e)}'}}\n\n"

        finally:
            # Clean up temporary file
            if temp_file_path:
                cleanup_temp_file(temp_file_path)

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


async def handle_temp_file_cleanup(file_path: str) -> Dict:
    """Clean up temporary file after processing.

    Args:
        file_path: Path to temporary file

    Returns:
        Dictionary with cleanup status
    """
    if cleanup_temp_file(file_path):
        return {"status": "success", "message": "File cleaned up successfully"}
    else:
        return {"status": "not_found", "message": "File not found or invalid path"}


async def _process_file_for_transcription(file_path: str, duration: float) -> Dict:
    """Process audio file and collect transcription results.

    Args:
        file_path: Path to audio file
        duration: Duration in seconds

    Returns:
        Dictionary with transcription results and processing time
    """
    audio_processor = AudioProcessor()
    results_generator = await audio_processor.create_tasks()

    # Collect results
    transcription_results = []

    # Background task to collect transcription results
    async def collect_results():
        async for resp in results_generator:
            lines = resp.get("lines", [])
            caption = " ".join([line["text"] for line in lines])
            buffer = resp.get("buffer_transcription", "")
            full_caption = (caption + " " + buffer).strip()

            if full_caption:
                transcription_results.append(
                    {
                        "text": full_caption,
                        "lines": lines,
                        "buffer": buffer,
                        "timestamp": resp.get("timestamp"),
                    }
                )

    consumer_task = asyncio.create_task(collect_results())

    try:
        # Set up FFmpeg and process audio
        ffmpeg_process = setup_ffmpeg_process(file_path)
        chunks = read_audio_chunks(ffmpeg_process)
        total_bytes, bytes_per_second, chunk_interval = calculate_streaming_params(chunks, duration)

        # Stream chunks with real-time pacing
        processing_time = await stream_chunks_realtime(
            chunks=chunks,
            chunk_interval=chunk_interval,
            duration=duration,
            process_func=audio_processor.process_audio,
            progress_callback=log_progress,
        )

        # Wait for processing to complete
        await consumer_task

        return {
            "transcription": transcription_results,
            "processing_time": processing_time,
        }

    finally:
        # Cleanup audio processor
        await audio_processor.cleanup()


async def _stream_file_processing_events(file_path: str, duration: float) -> AsyncGenerator[str, None]:
    """Stream file processing events for Server-Sent Events.

    Args:
        file_path: Path to audio file
        duration: Duration in seconds

    Yields:
        Server-Sent Event strings
    """
    audio_processor = AudioProcessor()
    results_generator = await audio_processor.create_tasks()

    # Set up FFmpeg and prepare chunks
    ffmpeg_process = setup_ffmpeg_process(file_path)
    chunks = read_audio_chunks(ffmpeg_process)
    total_bytes, bytes_per_second, chunk_interval = calculate_streaming_params(chunks, duration)

    logger.info(f"SSE Streaming {total_bytes} bytes over {duration:.2f}s ({bytes_per_second:.0f} bytes/s)")

    # Create a queue to collect transcription results
    result_queue = asyncio.Queue()
    stream_start_time = time.time()
    last_progress_time = time.time()

    async def process_audio_and_collect_results():
        """Process audio chunks and collect results in background."""
        try:
            # Start consuming results in background
            async def collect_results():
                async for resp in results_generator:
                    lines = resp.get("lines", [])
                    caption = " ".join([line["text"] for line in lines])
                    buffer = resp.get("buffer_transcription", "")
                    full_caption = (caption + " " + buffer).strip()

                    if full_caption or lines:
                        data = {
                            "text": full_caption,
                            "lines": lines,
                            "buffer": buffer,
                            "timestamp": resp.get("timestamp"),
                        }
                        await result_queue.put(("transcription", data))

                # Signal completion
                await result_queue.put(("complete", None))

            # Start result collection
            result_task = asyncio.create_task(collect_results())

            # Process audio chunks with real-time pacing
            for i, chunk in enumerate(chunks):
                # Calculate target time for this chunk
                target_time = stream_start_time + (i * chunk_interval)
                current_time = time.time()

                # Sleep if we're ahead of schedule
                sleep_time = target_time - current_time
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

                # Send the chunk
                await audio_processor.process_audio(chunk)

            # Send end of stream signal
            await audio_processor.process_audio(b"")

            # Wait for result collection to complete
            await result_task

        except Exception as e:
            await result_queue.put(("error", str(e)))

    # Start audio processing in background
    audio_task = asyncio.create_task(process_audio_and_collect_results())

    try:
        # Stream results as they come in
        processing_complete = False
        while not processing_complete:
            try:
                # Wait for results with timeout
                event_type, data = await asyncio.wait_for(result_queue.get(), timeout=1.0)

                if event_type == "transcription":
                    yield f"event: transcription\ndata: {json.dumps(data)}\n\n"
                elif event_type == "complete":
                    processing_complete = True
                elif event_type == "error":
                    yield f"event: error\ndata: {{'error': 'Error in transcription streaming: {data}'}}\n\n"
                    processing_complete = True

            except asyncio.TimeoutError:
                # Send progress update during timeout
                current_time = time.time()
                if current_time - last_progress_time >= 2.0:
                    elapsed = current_time - stream_start_time
                    audio_progress = min(elapsed, duration)
                    progress_data = {
                        "progress": round(audio_progress, 1),
                        "total": round(duration, 1),
                        "elapsed": round(elapsed, 1),
                    }
                    yield f"event: progress\ndata: {json.dumps(progress_data)}\n\n"
                    last_progress_time = current_time

        # Wait for audio processing to complete
        await audio_task

        total_elapsed = time.time() - stream_start_time
        logger.info(f"Finished SSE streaming: {total_elapsed:.2f}s (target: {duration:.2f}s)")

        # Send completion event
        completion_data = {
            "status": "success",
            "processing_time": round(total_elapsed, 1),
            "message": "File processed successfully",
        }
        yield f"event: complete\ndata: {json.dumps(completion_data)}\n\n"

    finally:
        # Cleanup audio processor
        await audio_processor.cleanup()


def _create_error_stream(error_message: str) -> StreamingResponse:
    """Create an error stream response.

    Args:
        error_message: Error message to send

    Returns:
        StreamingResponse with error event
    """

    async def error_stream():
        yield f"event: error\ndata: {{'error': '{error_message}'}}\n\n"

    return StreamingResponse(
        error_stream(),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )
