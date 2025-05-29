from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, File, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from . import AudioInsight, parse_args
from .display_parser import enable_display_parsing, get_display_parser
from .logging_config import get_logger, setup_logging
from .server.config import CORS_SETTINGS
from .server.file_handlers import (
    handle_file_upload_and_process,
    handle_file_upload_for_websocket,
    handle_file_upload_stream,
    handle_temp_file_cleanup,
)
from .server.websocket_handlers import (
    cleanup_global_processor,
    handle_websocket_connection,
)

# Ensure logging is initialized early
setup_logging()
logger = get_logger(__name__)

# Global AudioInsight kit instance
kit = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan and initialize AudioInsight."""
    global kit
    # Instantiate AudioInsight with the same CLI arguments as the server entrypoint
    args = parse_args()
    kit = AudioInsight(**vars(args))

    # Configure display text parsing with fast_llm model
    from .llm import ParserConfig

    fast_llm_model = getattr(args, "fast_llm", "google/gemini-flash-1.5-8b")
    display_config = ParserConfig(model_id=fast_llm_model)

    # Get the display parser and configure it
    display_parser = get_display_parser()
    display_parser.config = display_config

    # Enable display text parsing by default
    enable_display_parsing(True)
    logger.info(f"Display text parsing enabled with fast LLM model: {fast_llm_model}")

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


@app.post("/cleanup-session")
async def cleanup_session():
    """Force cleanup of all audio processing resources.

    This endpoint clears all memory, resets processors, and prepares for a fresh session.
    Useful to prevent memory leaks between file uploads or when the UI is refreshed.
    """
    global kit
    try:
        # First, clean up the global audio processor
        await cleanup_global_processor()
        logger.info("🧹 Global audio processor cleaned up")

        # Reset the AudioInsight singleton instance to clear all cached state
        if kit:
            kit.reset_instance()

        # Re-initialize with same arguments to ensure fresh state
        args = parse_args()
        kit = AudioInsight(**vars(args))

        logger.info("🧹 Session cleanup completed - all resources reset")
        return {"status": "success", "message": "Session cleaned up successfully"}

    except Exception as e:
        logger.error(f"Error during session cleanup: {e}")
        return {"status": "error", "message": f"Cleanup failed: {str(e)}"}


@app.post("/api/display-parser/enable")
async def enable_display_parser(enabled: bool = True):
    """Enable or disable display text parsing.

    Args:
        enabled: Whether to enable text parsing at display layer

    Returns:
        Status of the operation
    """
    try:
        display_parser = get_display_parser()
        display_parser.enable(enabled)

        status = "enabled" if enabled else "disabled"
        logger.info(f"Display text parser {status} via API")

        return {"status": "success", "message": f"Display text parser {status}", "enabled": enabled, "stats": display_parser.get_stats()}
    except Exception as e:
        logger.error(f"Error toggling display parser: {e}")
        return {"status": "error", "message": f"Error toggling display parser: {str(e)}", "enabled": False}


@app.get("/api/display-parser/status")
async def get_display_parser_status():
    """Get current status and statistics of display text parser.

    Returns:
        Display parser status and statistics
    """
    try:
        display_parser = get_display_parser()

        return {"status": "success", "enabled": display_parser.is_enabled(), "stats": display_parser.get_stats(), "config": {"model_id": display_parser.config.model_id if display_parser.config else "gpt-4o-mini", "cache_size": len(display_parser.parse_cache), "cache_max_size": display_parser.cache_max_size}}
    except Exception as e:
        logger.error(f"Error getting display parser status: {e}")
        return {"status": "error", "message": f"Error getting display parser status: {str(e)}", "enabled": False}


@app.post("/api/display-parser/clear-cache")
async def clear_display_parser_cache():
    """Clear the display text parser cache.

    Returns:
        Status of the operation
    """
    try:
        display_parser = get_display_parser()
        display_parser.clear_cache()

        return {"status": "success", "message": "Display text parser cache cleared", "stats": display_parser.get_stats()}
    except Exception as e:
        logger.error(f"Error clearing display parser cache: {e}")
        return {"status": "error", "message": f"Error clearing display parser cache: {str(e)}"}


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
