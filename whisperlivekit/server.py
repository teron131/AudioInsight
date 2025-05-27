import asyncio
import json
import logging
import os
import subprocess
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import (
    FastAPI,
    File,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse

from whisperlivekit import WhisperLiveKit, parse_args
from whisperlivekit.audio_processor import AudioProcessor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger().setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Cache allowed audio types set for better performance
ALLOWED_AUDIO_TYPES = {"audio/mpeg", "audio/mp3", "audio/mp4", "audio/m4a", "audio/wav", "audio/flac", "audio/ogg", "audio/webm"}

kit = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global kit
    # Instantiate WhisperLiveKit with the same CLI arguments as the server entrypoint
    args = parse_args()
    kit = WhisperLiveKit(**vars(args))
    yield


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def get():
    return HTMLResponse(kit.web_interface())


async def handle_websocket_results(websocket, results_generator):
    """Consumes results from the audio processor and sends them via WebSocket."""
    try:
        async for response in results_generator:
            await websocket.send_json(response)
        # when the results_generator finishes it means all audio has been processed
        logger.info("Results generator finished. Sending 'ready_to_stop' to client.")
        await websocket.send_json({"type": "ready_to_stop"})
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected while handling results (client likely closed connection).")
    except Exception as e:
        logger.warning(f"Error in WebSocket results handler: {e}")


async def process_file_through_websocket(file_path: str, duration: float, audio_processor: AudioProcessor):
    """
    Process an audio file through the same pipeline as live recording.
    This function streams the file in real-time to maintain identical processing.
    """
    logger.info(f"Starting unified file processing: {file_path}")

    # Use FFmpeg to convert file to WebM format (same as live recording)
    ff = subprocess.Popen(["ffmpeg", "-i", file_path, "-f", "webm", "-c:a", "libopus", "-ar", "16000", "-ac", "1", "pipe:1"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Read all data first to calculate proper streaming rate
    logger.info("Buffering audio data for real-time streaming...")
    all_chunks = []
    chunk_size = 4096

    while True:
        chunk = ff.stdout.read(chunk_size)
        if not chunk:
            break
        all_chunks.append(chunk)

    # Calculate total bytes more efficiently and handle edge cases
    total_audio_bytes = sum(len(chunk) for chunk in all_chunks)
    if total_audio_bytes == 0:
        raise Exception("No audio data received from FFmpeg")
    if duration <= 0:
        raise Exception("Invalid duration for audio file")

    # Calculate bytes per second for real-time simulation
    bytes_per_second = total_audio_bytes / duration
    chunk_interval = chunk_size / bytes_per_second

    # Pre-calculate number of chunks for progress reporting optimization
    num_chunks = len(all_chunks)
    progress_log_interval = max(1, int(2.0 / chunk_interval)) if chunk_interval > 0 else 1

    logger.info(f"Streaming {total_audio_bytes} bytes over {duration:.2f}s ({bytes_per_second:.0f} bytes/s, {chunk_interval:.3f}s per chunk)")

    # Send chunks with real-time pacing (identical to live recording processing)
    stream_start_time = time.time()
    for i, chunk in enumerate(all_chunks):
        # Calculate target time for this chunk
        target_time = stream_start_time + (i * chunk_interval)
        current_time = time.time()

        # Sleep if we're ahead of schedule
        sleep_time = target_time - current_time
        if sleep_time > 0:
            await asyncio.sleep(sleep_time)

        # Process the chunk through the same AudioProcessor pipeline
        await audio_processor.process_audio(chunk)

        # Log progress periodically - use pre-calculated values
        if i % progress_log_interval == 0:
            elapsed = time.time() - stream_start_time
            audio_progress = (i / num_chunks) * duration
            logger.info(f"File streaming progress: {audio_progress:.1f}s/{duration:.1f}s ({elapsed:.1f}s elapsed)")

    # Send end of stream signal (same as live recording)
    await audio_processor.process_audio(b"")

    total_elapsed = time.time() - stream_start_time
    logger.info(f"Finished unified file processing: {total_elapsed:.2f}s (target: {duration:.2f}s)")

    # Clean up temporary file automatically after processing
    try:
        if os.path.exists(file_path) and "/tmp/" in file_path:
            os.unlink(file_path)
            logger.info(f"Automatically cleaned up temporary file: {file_path}")
    except Exception as e:
        logger.warning(f"Failed to auto-cleanup temporary file {file_path}: {e}")

    return total_elapsed


@app.websocket("/asr")
async def websocket_endpoint(websocket: WebSocket):
    """Unified WebSocket endpoint for both live recording and file upload."""
    audio_processor = AudioProcessor()

    await websocket.accept()
    logger.info("WebSocket connection opened.")

    results_generator = await audio_processor.create_tasks()
    websocket_task = asyncio.create_task(handle_websocket_results(websocket, results_generator))

    try:
        while True:
            message = await websocket.receive()

            # Handle different message types
            if "bytes" in message:
                # Live recording: direct audio data
                await audio_processor.process_audio(message["bytes"])
            elif "text" in message:
                # File upload: JSON message with file info
                try:
                    data = json.loads(message["text"])
                    if data.get("type") == "file_upload":
                        # Process uploaded file through unified pipeline
                        file_path = data.get("file_path")
                        duration = data.get("duration", 0)

                        if file_path and os.path.exists(file_path):
                            logger.info(f"Processing uploaded file via WebSocket: {file_path}")
                            await process_file_through_websocket(file_path, duration, audio_processor)
                        else:
                            logger.warning(f"File not found for WebSocket processing: {file_path}")
                            await websocket.send_json({"type": "error", "error": "File not found or invalid file path"})
                except json.JSONDecodeError:
                    logger.warning("Received invalid JSON in WebSocket message")
                except Exception as e:
                    logger.error(f"Error processing file upload message: {e}")
                    await websocket.send_json({"type": "error", "error": f"Error processing file: {str(e)}"})
    except KeyError as e:
        if "bytes" in str(e):
            logger.warning(f"Client has closed the connection.")
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


@app.post("/cleanup-file")
async def cleanup_temp_file(file_path: str):
    """
    Clean up temporary file after processing.
    This endpoint allows the client to request cleanup of temporary files.
    """
    try:
        if os.path.exists(file_path) and "/tmp/" in file_path:  # Safety check
            os.unlink(file_path)
            logger.info(f"Cleaned up temporary file: {file_path}")
            return {"status": "success", "message": "File cleaned up successfully"}
        else:
            return {"status": "not_found", "message": "File not found or invalid path"}
    except Exception as e:
        logger.error(f"Error cleaning up file {file_path}: {e}")
        return {"status": "error", "message": f"Error cleaning up file: {str(e)}"}


@app.post("/upload-file")
async def upload_file_for_websocket(file: UploadFile = File(...)):
    """
    Upload file and return file info for WebSocket processing.
    This endpoint prepares the file for unified WebSocket processing.
    """
    try:
        # Validate file type - cache allowed types set at module level for better performance
        if file.content_type not in ALLOWED_AUDIO_TYPES:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}. Supported types: {', '.join(ALLOWED_AUDIO_TYPES)}")

        logger.info(f"Uploading file for WebSocket processing: {file.filename} ({file.content_type})")

        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as temp_file:
            # Write uploaded file to temporary location
            content = await file.read()
            temp_file.write(content)
            temp_file.flush()
            temp_file_path = temp_file.name

        # Get audio duration using ffprobe
        logger.info(f"Getting duration for audio file: {temp_file_path}")
        duration_cmd = ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", temp_file_path]
        duration_result = subprocess.run(duration_cmd, capture_output=True, text=True)

        if duration_result.returncode != 0:
            logger.error(f"ffprobe failed: {duration_result.stderr}")
            os.unlink(temp_file_path)  # Clean up on error
            raise HTTPException(status_code=500, detail=f"Failed to get audio duration: {duration_result.stderr}")

        try:
            duration_str = duration_result.stdout.strip()
            if not duration_str:
                raise Exception("Empty duration result from ffprobe")
            duration = float(duration_str)
            if duration <= 0:
                raise Exception(f"Invalid duration: {duration}")
            logger.info(f"Audio duration: {duration:.2f} seconds")
        except (ValueError, IndexError) as e:
            os.unlink(temp_file_path)  # Clean up on error
            raise HTTPException(status_code=500, detail=f"Could not parse audio duration '{duration_result.stdout.strip()}': {e}")

        return {"status": "success", "filename": file.filename, "file_path": temp_file_path, "duration": duration, "message": "File uploaded successfully. Use WebSocket to process."}

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error uploading file {file.filename}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload and process an audio file for transcription with real-time streaming simulation.

    Args:
        file: Audio file to transcribe

    Returns:
        JSON response with transcription results
    """
    try:
        # Validate file type - use cached allowed types
        if file.content_type not in ALLOWED_AUDIO_TYPES:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}. Supported types: {', '.join(ALLOWED_AUDIO_TYPES)}")

        logger.info(f"Processing uploaded file: {file.filename} ({file.content_type})")

        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as temp_file:
            # Write uploaded file to temporary location
            content = await file.read()
            temp_file.write(content)
            temp_file.flush()  # Ensure data is written to disk
            temp_file_path = temp_file.name

        try:
            # Get audio duration first using ffprobe
            logger.info(f"Getting duration for audio file: {temp_file_path}")
            duration_cmd = ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", temp_file_path]
            duration_result = subprocess.run(duration_cmd, capture_output=True, text=True)

            if duration_result.returncode != 0:
                logger.error(f"ffprobe failed: {duration_result.stderr}")
                raise Exception(f"Failed to get audio duration: {duration_result.stderr}")

            try:
                duration_str = duration_result.stdout.strip()
                if not duration_str:
                    raise Exception("Empty duration result from ffprobe")
                duration = float(duration_str)
                if duration <= 0:
                    raise Exception(f"Invalid duration: {duration}")
                logger.info(f"Audio duration: {duration:.2f} seconds - will simulate real-time playback")
            except (ValueError, IndexError) as e:
                raise Exception(f"Could not parse audio duration '{duration_result.stdout.strip()}': {e}")

            # Process the file using AudioProcessor
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
                        transcription_results.append({"text": full_caption, "lines": lines, "buffer": buffer, "timestamp": resp.get("timestamp")})

            consumer_task = asyncio.create_task(collect_results())

            # Use FFmpeg to process the audio file
            logger.info(f"Starting real-time streaming simulation for: {temp_file_path}")
            ff = subprocess.Popen(["ffmpeg", "-i", temp_file_path, "-f", "webm", "-c:a", "libopus", "-ar", "16000", "-ac", "1", "pipe:1"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            # Real-time streaming simulation
            start_time = time.time()
            chunk_size = 4096
            total_bytes_sent = 0

            # Read all data first to calculate actual data rate
            logger.info("Buffering audio data to calculate streaming rate...")
            all_chunks = []
            while True:
                chunk = ff.stdout.read(chunk_size)
                if not chunk:
                    break
                all_chunks.append(chunk)

            total_audio_bytes = sum(len(chunk) for chunk in all_chunks)
            if total_audio_bytes == 0:
                raise Exception("No audio data received from FFmpeg")

            # Calculate bytes per second for real-time simulation
            bytes_per_second = total_audio_bytes / duration
            chunk_interval = chunk_size / bytes_per_second

            logger.info(f"Streaming {total_audio_bytes} bytes over {duration:.2f}s ({bytes_per_second:.0f} bytes/s, {chunk_interval:.3f}s per chunk)")

            # Send chunks with real-time pacing
            stream_start_time = time.time()
            for i, chunk in enumerate(all_chunks):
                # Calculate target time for this chunk
                target_time = stream_start_time + (i * chunk_interval)
                current_time = time.time()

                # Sleep if we're ahead of schedule
                sleep_time = target_time - current_time
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

                # Send the chunk
                await audio_processor.process_audio(chunk)
                total_bytes_sent += len(chunk)

                # Log progress every 2 seconds of audio
                if i % max(1, int(2.0 / chunk_interval)) == 0:
                    elapsed = time.time() - stream_start_time
                    audio_progress = (i / len(all_chunks)) * duration
                    logger.info(f"Streaming progress: {audio_progress:.1f}s/{duration:.1f}s ({elapsed:.1f}s elapsed)")

            # Send end of stream signal
            await audio_processor.process_audio(b"")

            total_elapsed = time.time() - stream_start_time
            logger.info(f"Finished real-time streaming: {total_elapsed:.2f}s (target: {duration:.2f}s)")

            # Wait for processing to complete
            await consumer_task

            # Cleanup audio processor
            await audio_processor.cleanup()

            logger.info(f"Successfully processed file: {file.filename}, results: {len(transcription_results)} segments")

            return {"status": "success", "filename": file.filename, "transcription": transcription_results, "message": f"File processed in real-time simulation ({total_elapsed:.1f}s for {duration:.1f}s audio)", "audio_duration": duration, "processing_time": total_elapsed}

        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                logger.debug(f"Cleaned up temporary file: {temp_file_path}")

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error processing uploaded file {file.filename}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")


@app.post("/upload-stream")
async def upload_file_stream(file: UploadFile = File(...)):
    """
    Upload and process an audio file with real-time streaming results using Server-Sent Events.

    Args:
        file: Audio file to transcribe

    Returns:
        StreamingResponse with SSE events containing transcription results
    """

    # Read file content before creating the async generator
    # (FastAPI UploadFile can only be read once)
    try:
        file_content = await file.read()
    except Exception as e:

        async def error_stream():
            yield f"event: error\ndata: {{'error': 'Failed to read uploaded file: {str(e)}'}}\n\n"

        return StreamingResponse(
            error_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
            },
        )

    async def event_stream():
        temp_file_path = None
        try:
            # Validate file type
            if file.content_type not in ALLOWED_AUDIO_TYPES:
                yield f"event: error\ndata: {{'error': 'Unsupported file type: {file.content_type}'}}\n\n"
                return

            logger.info(f"Processing uploaded file for streaming: {file.filename} ({file.content_type})")

            # Create temporary file using the pre-read content
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as temp_file:
                # Write uploaded file content to temporary location
                temp_file.write(file_content)
                temp_file.flush()  # Ensure data is written to disk
                temp_file_path = temp_file.name

            try:
                # Get audio duration first using ffprobe
                logger.info(f"Getting duration for streaming upload: {temp_file_path}")
                duration_cmd = ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", temp_file_path]
                duration_result = subprocess.run(duration_cmd, capture_output=True, text=True)

                if duration_result.returncode != 0:
                    logger.error(f"ffprobe failed: {duration_result.stderr}")
                    yield f"event: error\ndata: {{'error': 'Failed to get audio duration'}}\n\n"
                    return

                try:
                    duration_str = duration_result.stdout.strip()
                    if not duration_str:
                        raise Exception("Empty duration result from ffprobe")
                    duration = float(duration_str)
                    if duration <= 0:
                        raise Exception(f"Invalid duration: {duration}")
                    logger.info(f"Audio duration for streaming: {duration:.2f} seconds")

                    # Send initial info
                    yield f"event: start\ndata: {{'filename': '{file.filename}', 'duration': {duration}}}\n\n"

                except (ValueError, IndexError) as e:
                    yield f"event: error\ndata: {{'error': 'Could not parse audio duration: {e}'}}\n\n"
                    return

                # Process the file using AudioProcessor
                audio_processor = AudioProcessor()
                results_generator = await audio_processor.create_tasks()

                # Create tasks for audio processing and result streaming
                # Use FFmpeg to process the audio file in real-time
                logger.info(f"Starting real-time streaming for SSE: {temp_file_path}")
                ff = subprocess.Popen(["ffmpeg", "-i", temp_file_path, "-f", "webm", "-c:a", "libopus", "-ar", "16000", "-ac", "1", "pipe:1"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

                # Read all data first to calculate actual data rate
                logger.info("Buffering audio data for SSE streaming...")
                all_chunks = []
                while True:
                    chunk = ff.stdout.read(4096)
                    if not chunk:
                        break
                    all_chunks.append(chunk)

                total_audio_bytes = sum(len(chunk) for chunk in all_chunks)
                if total_audio_bytes == 0:
                    yield f"event: error\ndata: {{'error': 'No audio data received from FFmpeg'}}\n\n"
                    return

                # Calculate bytes per second for real-time simulation
                bytes_per_second = total_audio_bytes / duration
                chunk_interval = 4096 / bytes_per_second

                logger.info(f"SSE Streaming {total_audio_bytes} bytes over {duration:.2f}s ({bytes_per_second:.0f} bytes/s)")

                # Send chunks with real-time pacing and stream results
                stream_start_time = time.time()
                last_progress_time = time.time()

                # Create a queue to collect transcription results
                result_queue = asyncio.Queue()

                async def process_audio_and_collect_results():
                    """Process audio chunks with real-time pacing and collect results"""
                    try:
                        # Start consuming results in background
                        async def collect_results():
                            async for resp in results_generator:
                                lines = resp.get("lines", [])
                                caption = " ".join([line["text"] for line in lines])
                                buffer = resp.get("buffer_transcription", "")
                                full_caption = (caption + " " + buffer).strip()

                                if full_caption or lines:
                                    data = {"text": full_caption, "lines": lines, "buffer": buffer, "timestamp": resp.get("timestamp")}
                                    await result_queue.put(("transcription", data))

                            # Signal completion
                            await result_queue.put(("complete", None))

                        # Start result collection
                        result_task = asyncio.create_task(collect_results())

                        # Process audio chunks with real-time pacing
                        for i, chunk in enumerate(all_chunks):
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
                            yield f"event: progress\ndata: {{'progress': {audio_progress:.1f}, 'total': {duration:.1f}, 'elapsed': {elapsed:.1f}}}\n\n"
                            last_progress_time = current_time

                # Wait for audio processing to complete
                await audio_task

                total_elapsed = time.time() - stream_start_time
                logger.info(f"Finished SSE streaming: {total_elapsed:.2f}s (target: {duration:.2f}s)")

                # Cleanup audio processor
                await audio_processor.cleanup()

                # Send completion event
                yield f"event: complete\ndata: {{'status': 'success', 'processing_time': {total_elapsed:.1f}, 'message': 'File processed successfully'}}\n\n"

            finally:
                # Clean up temporary file
                if os.path.exists(temp_file_path):
                    os.unlink(temp_file_path)
                    logger.debug(f"Cleaned up temporary file: {temp_file_path}")

        except Exception as e:
            logger.error(f"Error in SSE upload stream: {str(e)}", exc_info=True)
            yield f"event: error\ndata: {{'error': 'Server error: {str(e)}'}}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*",
        },
    )


def main():
    """Entry point for the CLI command."""
    args = parse_args()

    uvicorn_kwargs = {
        "app": "whisperlivekit.server:app",
        "host": args.host,
        "port": args.port,
        "reload": False,
        "log_level": "info",
        "lifespan": "on",
    }

    ssl_kwargs = {}
    if args.ssl_certfile or args.ssl_keyfile:
        if not (args.ssl_certfile and args.ssl_keyfile):
            raise ValueError("Both --ssl-certfile and --ssl-keyfile must be specified together.")
        ssl_kwargs = {"ssl_certfile": args.ssl_certfile, "ssl_keyfile": args.ssl_keyfile}

    if ssl_kwargs:
        uvicorn_kwargs = {**uvicorn_kwargs, **ssl_kwargs}

    uvicorn.run(**uvicorn_kwargs)


if __name__ == "__main__":
    main()
