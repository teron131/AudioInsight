import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, File, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from . import AudioInsight, parse_args
from .server.config import CORS_SETTINGS
from .server.file_handlers import (
    handle_file_upload_and_process,
    handle_file_upload_for_websocket,
    handle_file_upload_stream,
    handle_temp_file_cleanup,
)
from .server.websocket_handlers import handle_websocket_connection

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.getLogger().setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Global AudioInsight kit instance
kit = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan and initialize AudioInsight."""
    global kit
    # Instantiate AudioInsight with the same CLI arguments as the server entrypoint
    args = parse_args()
    kit = AudioInsight(**vars(args))
    yield


# Create FastAPI application
app = FastAPI(lifespan=lifespan)

# Add CORS middleware
app.add_middleware(CORSMiddleware, **CORS_SETTINGS)


@app.get("/")
async def get_root():
    """Serve the web interface."""
    return HTMLResponse(kit.web_interface())


@app.websocket("/asr")
async def websocket_endpoint(websocket: WebSocket):
    """Unified WebSocket endpoint for both live recording and file upload."""
    await handle_websocket_connection(websocket)


@app.post("/upload-file")
async def upload_file_for_websocket(file: UploadFile = File(...)):
    """Upload file and return file info for WebSocket processing.

    This endpoint prepares the file for unified WebSocket processing.
    """
    return await handle_file_upload_for_websocket(file)


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload and process an audio file for transcription with real-time streaming simulation.

    Args:
        file: Audio file to transcribe

    Returns:
        JSON response with transcription results
    """
    return await handle_file_upload_and_process(file)


@app.post("/upload-stream")
async def upload_file_stream(file: UploadFile = File(...)):
    """Upload and process an audio file with real-time streaming results using Server-Sent Events.

    Args:
        file: Audio file to transcribe

    Returns:
        StreamingResponse with SSE events containing transcription results
    """
    return await handle_file_upload_stream(file)


@app.post("/cleanup-file")
async def cleanup_temp_file(file_path: str):
    """Clean up temporary file after processing.

    This endpoint allows the client to request cleanup of temporary files.
    """
    return await handle_temp_file_cleanup(file_path)


def main():
    """Entry point for the CLI command."""
    args = parse_args()

    uvicorn_kwargs = {
        "app": "audioinsight.app:app",
        "host": args.host,
        "port": args.port,
        "reload": False,
        "log_level": "info",
        "lifespan": "on",
    }

    # Handle SSL configuration
    ssl_kwargs = {}
    if args.ssl_certfile or args.ssl_keyfile:
        if not (args.ssl_certfile and args.ssl_keyfile):
            raise ValueError("Both --ssl-certfile and --ssl-keyfile must be specified together.")
        ssl_kwargs = {
            "ssl_certfile": args.ssl_certfile,
            "ssl_keyfile": args.ssl_keyfile,
        }

    if ssl_kwargs:
        uvicorn_kwargs = {**uvicorn_kwargs, **ssl_kwargs}

    uvicorn.run(**uvicorn_kwargs)
