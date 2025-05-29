import time
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


# =============================================================================
# Configuration Management APIs
# =============================================================================


@app.get("/api/config/models")
async def get_model_config():
    """Get current model configuration."""
    try:
        global kit
        if not kit:
            return {"status": "error", "message": "AudioInsight not initialized"}

        # Get current configuration from the kit
        config = {
            "transcription_model": getattr(kit, "whisper_model", "openai/whisper-large-v3"),
            "diarization_enabled": getattr(kit, "diarization_enabled", True),
            "llm_analysis_enabled": getattr(kit, "llm_enabled", False),
            "fast_llm_model": getattr(kit.display_parser, "config", {}).get("model_id", "google/gemini-flash-1.5-8b") if hasattr(kit, "display_parser") else "google/gemini-flash-1.5-8b",
        }

        return {"status": "success", "config": config}
    except Exception as e:
        logger.error(f"Error getting model config: {e}")
        return {"status": "error", "message": f"Error getting model config: {str(e)}"}


@app.post("/api/config/processing")
async def update_processing_config(config: dict):
    """Update processing configuration.

    Args:
        config: Processing configuration settings

    Returns:
        Status of the operation
    """
    try:
        global kit
        if not kit:
            return {"status": "error", "message": "AudioInsight not initialized"}

        # Update configuration
        updated_fields = []

        if "diarization_enabled" in config:
            kit.diarization_enabled = config["diarization_enabled"]
            updated_fields.append("diarization_enabled")

        if "llm_analysis_enabled" in config:
            kit.llm_enabled = config["llm_analysis_enabled"]
            updated_fields.append("llm_analysis_enabled")

        if "fast_llm_model" in config and hasattr(kit, "display_parser"):
            from .llm import ParserConfig

            kit.display_parser.config = ParserConfig(model_id=config["fast_llm_model"])
            updated_fields.append("fast_llm_model")

        logger.info(f"Updated processing config: {updated_fields}")
        return {"status": "success", "message": f"Updated configuration: {', '.join(updated_fields)}", "updated_fields": updated_fields}
    except Exception as e:
        logger.error(f"Error updating processing config: {e}")
        return {"status": "error", "message": f"Error updating config: {str(e)}"}


# =============================================================================
# System Status and Health APIs
# =============================================================================


@app.get("/api/system/status")
async def get_system_status():
    """Get system status and health information."""
    try:
        import platform

        import psutil

        # System info
        system_info = {"platform": platform.system(), "python_version": platform.python_version(), "cpu_count": psutil.cpu_count(), "cpu_percent": psutil.cpu_percent(interval=1), "memory": {"total": psutil.virtual_memory().total, "available": psutil.virtual_memory().available, "percent": psutil.virtual_memory().percent}, "disk": {"total": psutil.disk_usage("/").total, "free": psutil.disk_usage("/").free, "percent": psutil.disk_usage("/").percent}}

        # AudioInsight status
        global kit
        audioinsight_status = {"initialized": kit is not None, "diarization_enabled": getattr(kit, "diarization_enabled", False) if kit else False, "llm_enabled": getattr(kit, "llm_enabled", False) if kit else False, "display_parser_enabled": get_display_parser().is_enabled() if kit else False}

        return {"status": "success", "system": system_info, "audioinsight": audioinsight_status, "timestamp": time.time()}
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        return {"status": "error", "message": f"Error getting system status: {str(e)}"}


@app.get("/api/system/health")
async def health_check():
    """Simple health check endpoint."""
    try:
        global kit
        health_status = "healthy" if kit is not None else "unhealthy"

        return {"status": health_status, "timestamp": time.time(), "service": "AudioInsight", "version": "1.0.0"}
    except Exception as e:
        return {"status": "error", "message": str(e)}


# =============================================================================
# Export APIs
# =============================================================================


@app.post("/api/export/transcript")
async def export_transcript(format: str = "txt", session_data: dict = None):
    """Export transcript in various formats.

    Args:
        format: Export format (txt, srt, vtt, json)
        session_data: Transcript data to export

    Returns:
        Exported transcript content
    """
    try:
        if not session_data or "lines" not in session_data:
            return {"status": "error", "message": "No transcript data provided"}

        lines = session_data["lines"]

        if format.lower() == "txt":
            content = "\n\n".join([f"Speaker {line.get('speaker', 'Unknown')}: {line.get('text', '')}" for line in lines if line.get("text", "").strip()])

        elif format.lower() == "srt":
            content = ""
            for i, line in enumerate(lines, 1):
                if line.get("text", "").strip():
                    start_time = _format_srt_time(line.get("beg", 0))
                    end_time = _format_srt_time(line.get("end", 0))
                    speaker = f"Speaker {line.get('speaker', 'Unknown')}"
                    text = line.get("text", "")

                    content += f"{i}\n{start_time} --> {end_time}\n{speaker}: {text}\n\n"

        elif format.lower() == "vtt":
            content = "WEBVTT\n\n"
            for line in lines:
                if line.get("text", "").strip():
                    start_time = _format_vtt_time(line.get("beg", 0))
                    end_time = _format_vtt_time(line.get("end", 0))
                    speaker = f"Speaker {line.get('speaker', 'Unknown')}"
                    text = line.get("text", "")

                    content += f"{start_time} --> {end_time}\n{speaker}: {text}\n\n"

        elif format.lower() == "json":
            import json

            content = json.dumps(session_data, indent=2)

        else:
            return {"status": "error", "message": f"Unsupported format: {format}"}

        return {"status": "success", "format": format, "content": content, "filename": f"transcript.{format.lower()}"}

    except Exception as e:
        logger.error(f"Error exporting transcript: {e}")
        return {"status": "error", "message": f"Error exporting transcript: {str(e)}"}


# =============================================================================
# Session Management APIs
# =============================================================================


@app.get("/api/sessions/current")
async def get_current_session():
    """Get current session information."""
    try:
        global kit
        if not kit:
            return {"status": "error", "message": "No active session"}

        # Get display parser stats if available
        display_stats = {}
        try:
            display_parser = get_display_parser()
            display_stats = display_parser.get_stats()
        except:
            pass

        session_info = {"session_id": getattr(kit, "session_id", "default"), "created_at": getattr(kit, "created_at", time.time()), "diarization_enabled": getattr(kit, "diarization_enabled", False), "llm_enabled": getattr(kit, "llm_enabled", False), "display_parser_stats": display_stats, "status": "active"}

        return {"status": "success", "session": session_info}
    except Exception as e:
        logger.error(f"Error getting current session: {e}")
        return {"status": "error", "message": f"Error getting session: {str(e)}"}


@app.post("/api/sessions/reset")
async def reset_session():
    """Reset current session with clean state."""
    try:
        # Use existing cleanup functionality
        result = await cleanup_session()

        if result.get("status") == "success":
            return {"status": "success", "message": "Session reset successfully", "timestamp": time.time()}
        else:
            return result

    except Exception as e:
        logger.error(f"Error resetting session: {e}")
        return {"status": "error", "message": f"Error resetting session: {str(e)}"}


# =============================================================================
# Analytics APIs
# =============================================================================


@app.get("/api/analytics/usage")
async def get_usage_analytics():
    """Get usage analytics and statistics."""
    try:
        # Get display parser stats
        display_parser = get_display_parser()
        display_stats = display_parser.get_stats()

        analytics = {"display_parser": {"total_requests": display_stats.get("total_requests", 0), "cache_hits": display_stats.get("cache_hits", 0), "cache_misses": display_stats.get("cache_misses", 0), "average_response_time": display_stats.get("average_response_time", 0), "cache_hit_rate": display_stats.get("cache_hit_rate", 0)}, "session": {"current_session_active": kit is not None, "uptime": time.time() - getattr(kit, "created_at", time.time()) if kit else 0}, "timestamp": time.time()}

        return {"status": "success", "analytics": analytics}
    except Exception as e:
        logger.error(f"Error getting usage analytics: {e}")
        return {"status": "error", "message": f"Error getting analytics: {str(e)}"}


# =============================================================================
# Helper Functions
# =============================================================================


def _format_srt_time(seconds: float) -> str:
    """Format seconds to SRT time format (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def _format_vtt_time(seconds: float) -> str:
    """Format seconds to WebVTT time format (HH:MM:SS.mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    milliseconds = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"


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
