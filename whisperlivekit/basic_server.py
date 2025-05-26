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

kit = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global kit
    # Clear sys.argv to prevent argument parsing conflicts when used as a module
    import sys

    original_argv = sys.argv.copy()
    try:
        sys.argv = [sys.argv[0]]  # Keep only the script name
        kit = WhisperLiveKit(model="large-v3-turbo", diarization=False)
    finally:
        sys.argv = original_argv
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


@app.websocket("/asr")
async def websocket_endpoint(websocket: WebSocket):
    audio_processor = AudioProcessor()

    await websocket.accept()
    logger.info("WebSocket connection opened.")

    results_generator = await audio_processor.create_tasks()
    websocket_task = asyncio.create_task(handle_websocket_results(websocket, results_generator))

    try:
        while True:
            message = await websocket.receive_bytes()
            await audio_processor.process_audio(message)
    except KeyError as e:
        if "bytes" in str(e):
            logger.warning(f"Client has closed the connection.")
        else:
            logger.error(f"Unexpected KeyError in websocket_endpoint: {e}", exc_info=True)
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected by client during message receiving loop.")
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
        # Validate file type
        allowed_types = {"audio/mpeg", "audio/mp3", "audio/mp4", "audio/m4a", "audio/wav", "audio/flac", "audio/ogg", "audio/webm"}

        if file.content_type not in allowed_types:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file.content_type}. Supported types: {', '.join(allowed_types)}")

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
            allowed_types = {"audio/mpeg", "audio/mp3", "audio/mp4", "audio/m4a", "audio/wav", "audio/flac", "audio/ogg", "audio/webm"}

            if file.content_type not in allowed_types:
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
        "app": "whisperlivekit.basic_server:app",
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
