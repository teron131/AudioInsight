import time

from fastapi import APIRouter, File, UploadFile, WebSocket
from fastapi.responses import HTMLResponse

from ..logging_config import get_logger
from ..server.file_handlers import (
    handle_file_upload_and_process,
    handle_file_upload_for_websocket,
    handle_file_upload_stream,
    handle_temp_file_cleanup,
)
from ..server.websocket_handlers import handle_websocket_connection

logger = get_logger(__name__)

router = APIRouter()

# Import global variables from server_app
import audioinsight.audioinsight_server as app


@router.get("/")
async def get_root():
    """Serve the web interface."""
    return HTMLResponse(app.kit.web_interface())


@router.get("/health")
@router.get("/api/health")
async def health_check():
    """Health check endpoint that includes backend readiness status."""
    health_status = {
        "status": "ok",
        "backend_ready": app.backend_ready,
        "kit_initialized": app.kit is not None,
        "timestamp": time.time(),
    }

    # Additional readiness checks
    if app.kit:
        try:
            # Check if key models are loaded
            health_status["whisper_loaded"] = hasattr(app.kit, "asr") and app.kit.asr is not None
            health_status["diarization_available"] = hasattr(app.kit, "diarization") and app.kit.diarization is not None
        except Exception as e:
            logger.warning(f"Error checking kit status: {e}")
            health_status["kit_status_error"] = str(e)

    return health_status


@router.websocket("/asr")
async def websocket_endpoint(websocket: WebSocket):
    """Unified WebSocket endpoint for both live recording and file upload."""
    # Check if backend is ready before proceeding
    if not app.backend_ready:
        logger.warning("❌ WebSocket connection attempted before backend is ready")
        await websocket.close(code=1013, reason="Backend not ready")
        return

    await handle_websocket_connection(websocket)


@router.post("/upload-file")
async def upload_file_for_websocket(file: UploadFile = File(...)):
    """Upload file and return file info for WebSocket processing.

    This endpoint prepares the file for unified WebSocket processing.
    """
    return await handle_file_upload_for_websocket(file)


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload and process an audio file for transcription with real-time streaming simulation.

    Args:
        file: Audio file to transcribe

    Returns:
        JSON response with transcription results
    """
    return await handle_file_upload_and_process(file)


@router.post("/upload-stream")
async def upload_file_stream(file: UploadFile = File(...)):
    """Upload and process an audio file with real-time streaming results using Server-Sent Events.

    Args:
        file: Audio file to transcribe

    Returns:
        StreamingResponse with SSE events containing transcription results
    """
    return await handle_file_upload_stream(file)


@router.post("/cleanup-file")
async def cleanup_temp_file(file_path: str):
    """Clean up temporary file after processing.

    This endpoint allows the client to request cleanup of temporary files.
    """
    return await handle_temp_file_cleanup(file_path)
