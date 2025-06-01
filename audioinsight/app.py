import time
from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI, File, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from . import AudioInsight, parse_args
from .config import apply_parameter_updates, get_config, get_processing_parameters
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


# =============================================================================
# API Helper Functions
# =============================================================================


def success_response(message: str, data: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
    """Create a standardized success response."""
    response = {
        "status": "success",
        "message": message,
    }

    if data:
        response.update(data)

    response.update(kwargs)
    return response


def error_response(message: str, error: Optional[Exception] = None, log_error: bool = True, **kwargs) -> Dict[str, Any]:
    """Create a standardized error response."""
    if log_error and error:
        logger.error(f"{message}: {error}")
    elif log_error:
        logger.error(message)

    response = {
        "status": "error",
        "message": message,
    }

    response.update(kwargs)
    return response


def handle_api_exception(operation: str, error: Exception) -> Dict[str, Any]:
    """Handle API exceptions with consistent logging and response format."""
    return error_response(message=f"Error {operation}: {str(error)}", error=error, log_error=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan and initialize AudioInsight."""
    global kit
    # Instantiate AudioInsight with the same CLI arguments as the server entrypoint
    args = parse_args()
    kit = AudioInsight(**vars(args))

    # Configure display text parsing with fast_llm model
    from .llm import ParserConfig

    fast_llm_model = getattr(args, "fast_llm", "openai/gpt-4.1-nano")
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
    """Enable or disable display text parsing."""
    try:
        display_parser = get_display_parser()
        display_parser.enable(enabled)

        status = "enabled" if enabled else "disabled"
        return success_response(message=f"Display text parser {status}", enabled=enabled, stats=display_parser.get_stats())
    except Exception as e:
        return handle_api_exception("toggling display parser", e)


@app.get("/api/display-parser/status")
async def get_display_parser_status():
    """Get current status and statistics of display text parser."""
    try:
        display_parser = get_display_parser()

        return success_response(
            message="Display parser status retrieved",
            enabled=display_parser.is_enabled(),
            stats=display_parser.get_stats(),
            config={
                "model_id": display_parser.config.model_id if display_parser.config else "gpt-4o-mini",
                "cache_size": display_parser.parse_cache.size(),
                "cache_max_size": display_parser.cache_max_size,
            },
        )
    except Exception as e:
        return handle_api_exception("getting display parser status", e)


@app.post("/api/display-parser/clear-cache")
async def clear_display_parser_cache():
    """Clear the display text parser cache."""
    try:
        display_parser = get_display_parser()
        display_parser.clear_cache()

        return success_response(message="Display text parser cache cleared", stats=display_parser.get_stats())
    except Exception as e:
        return handle_api_exception("clearing display parser cache", e)


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
            "fast_llm_model": getattr(kit.display_parser, "config", {}).get("model_id", "openai/gpt-4.1-nano") if hasattr(kit, "display_parser") else "openai/gpt-4.1-nano",
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


# =============================================================================
# Model Management APIs
# =============================================================================


@app.get("/api/models/status")
async def get_models_status():
    """Get the status of all loaded models."""
    try:
        global kit
        if not kit:
            return {"status": "error", "message": "AudioInsight not initialized"}

        models_status = {
            "asr": {
                "loaded": kit._models_loaded,
                "model_name": getattr(kit.args, "model", "unknown"),
                "backend": getattr(kit.args, "backend", "unknown"),
                "language": getattr(kit.args, "lang", "auto"),
                "ready": kit.asr is not None if hasattr(kit, "asr") else False,
            },
            "diarization": {
                "loaded": kit._diarization_loaded,
                "enabled": getattr(kit.args, "diarization", False),
                "ready": kit.diarization is not None if hasattr(kit, "diarization") else False,
            },
            "llm": {
                "fast_model": getattr(kit.args, "fast_llm", "openai/gpt-4.1-nano"),
                "base_model": getattr(kit.args, "base_llm", "openai/gpt-4.1-mini"),
                "inference_enabled": getattr(kit.args, "llm_inference", True),
            },
        }

        return {"status": "success", "models": models_status}
    except Exception as e:
        logger.error(f"Error getting models status: {e}")
        return {"status": "error", "message": f"Error getting models status: {str(e)}"}


@app.post("/api/models/reload")
async def reload_models(model_type: str = "all"):
    """Reload specific models or all models."""
    try:
        global kit
        if not kit:
            return {"status": "error", "message": "AudioInsight not initialized"}

        reloaded = []

        if model_type in ["all", "asr"] and hasattr(kit, "_load_asr_models"):
            kit._models_loaded = False
            kit._load_asr_models()
            kit._models_loaded = True
            reloaded.append("asr")

        if model_type in ["all", "diarization"] and hasattr(kit, "_load_diarization"):
            kit._diarization_loaded = False
            kit._load_diarization()
            kit._diarization_loaded = True
            reloaded.append("diarization")

        logger.info(f"Reloaded models: {reloaded}")
        return {"status": "success", "message": f"Reloaded models: {', '.join(reloaded)}", "reloaded": reloaded}

    except Exception as e:
        logger.error(f"Error reloading models: {e}")
        return {"status": "error", "message": f"Error reloading models: {str(e)}"}


@app.post("/api/models/unload")
async def unload_models(model_type: str = "all"):
    """Unload specific models to free memory."""
    try:
        global kit
        if not kit:
            return {"status": "error", "message": "AudioInsight not initialized"}

        unloaded = []

        if model_type in ["all", "asr"]:
            kit.asr = None
            kit.tokenizer = None
            kit._models_loaded = False
            unloaded.append("asr")

        if model_type in ["all", "diarization"]:
            if hasattr(kit, "diarization") and kit.diarization:
                kit.diarization.close()
            kit.diarization = None
            kit._diarization_loaded = False
            unloaded.append("diarization")

        logger.info(f"Unloaded models: {unloaded}")
        return {"status": "success", "message": f"Unloaded models: {', '.join(unloaded)}", "unloaded": unloaded}

    except Exception as e:
        logger.error(f"Error unloading models: {e}")
        return {"status": "error", "message": f"Error unloading models: {str(e)}"}


# =============================================================================
# Audio Processing Control APIs
# =============================================================================


@app.post("/api/processing/parameters")
async def update_processing_parameters(parameters: dict):
    """Update audio processing parameters in real-time."""
    try:
        global kit
        if not kit:
            return {"status": "error", "message": "AudioInsight not initialized"}

        # Use domain-specific configuration system
        from .config import apply_runtime_updates

        updated_params = apply_runtime_updates(parameters)

        # Also update the kit's args for backward compatibility
        config = get_config()

        # Sync config back to kit.args (for legacy code compatibility)
        if hasattr(kit, "args"):
            # Update args with unified config values, mapping unified names to legacy names
            kit.args.host = config.server.host
            kit.args.port = config.server.port
            kit.args.model = config.model.model
            kit.args.backend = config.model.backend
            kit.args.lang = config.model.language  # Map unified 'language' to legacy 'lang'
            kit.args.task = config.model.task
            kit.args.min_chunk_size = config.processing.min_chunk_size
            kit.args.buffer_trimming = config.processing.buffer_trimming
            kit.args.buffer_trimming_sec = config.processing.buffer_trimming_sec
            kit.args.vac_chunk_size = config.processing.vac_chunk_size
            kit.args.transcription = config.features.transcription
            kit.args.diarization = config.features.diarization
            kit.args.vad = config.features.vad
            kit.args.vac = config.features.vac
            kit.args.confidence_validation = config.features.confidence_validation
            kit.args.llm_inference = config.features.llm_inference
            kit.args.fast_llm = config.llm.fast_llm
            kit.args.base_llm = config.llm.base_llm
            kit.args.llm_summary_interval = config.llm.llm_summary_interval
            kit.args.llm_new_text_trigger = config.llm.llm_new_text_trigger

        return success_response("Parameters updated successfully", {"updated_parameters": updated_params, "total_updates": sum(len(domain_updates) for domain_updates in updated_params.values())})

    except Exception as e:
        logger.error(f"Error updating processing parameters: {str(e)}")
        return {"status": "error", "message": f"Failed to update parameters: {str(e)}"}


@app.get("/api/processing/parameters")
async def get_processing_parameters():
    """Get current audio processing parameters with runtime/startup classification."""
    try:
        from .config import get_runtime_configurable_fields, get_startup_only_fields

        # Get all current parameters (backward compatibility)
        all_params = get_processing_parameters()

        # Get runtime vs startup classification
        runtime_fields = get_runtime_configurable_fields()
        startup_fields = get_startup_only_fields()

        return success_response(
            "Parameters retrieved successfully",
            {
                "parameters": all_params,  # Backward compatibility
                "runtime_configurable": runtime_fields,
                "startup_only": startup_fields,
            },
        )

    except Exception as e:
        logger.error(f"Error retrieving processing parameters: {str(e)}")
        return {"status": "error", "message": f"Failed to retrieve parameters: {str(e)}"}


# =============================================================================
# Configuration Presets APIs
# =============================================================================


@app.get("/api/presets")
async def get_configuration_presets():
    """Get available configuration presets."""
    try:
        presets = {
            "fast_transcription": {
                "name": "Fast Transcription",
                "description": "Optimized for speed with basic features",
                "config": {
                    "model": "base",
                    "diarization": False,
                    "llm_inference": False,
                    "min_chunk_size": 0.3,
                    "buffer_trimming": "segment",
                    "vad": True,
                    "vac": False,
                },
            },
            "high_accuracy": {
                "name": "High Accuracy",
                "description": "Maximum accuracy with all features enabled",
                "config": {
                    "model": "large-v3-turbo",
                    "diarization": True,
                    "llm_inference": True,
                    "min_chunk_size": 1.0,
                    "buffer_trimming": "sentence",
                    "vad": True,
                    "vac": True,
                    "confidence_validation": True,
                },
            },
            "meeting_recording": {
                "name": "Meeting Recording",
                "description": "Optimized for multi-speaker meetings",
                "config": {
                    "model": "large-v3",
                    "diarization": True,
                    "llm_inference": True,
                    "min_chunk_size": 0.8,
                    "buffer_trimming": "sentence",
                    "llm_conversation_trigger": 3,
                    "vad": True,
                },
            },
            "live_streaming": {
                "name": "Live Streaming",
                "description": "Real-time streaming with low latency",
                "config": {
                    "model": "medium",
                    "diarization": False,
                    "llm_inference": False,
                    "min_chunk_size": 0.2,
                    "buffer_trimming": "segment",
                    "vac": True,
                    "vad": True,
                },
            },
        }

        return {"status": "success", "presets": presets}

    except Exception as e:
        logger.error(f"Error getting configuration presets: {e}")
        return {"status": "error", "message": f"Error getting configuration presets: {str(e)}"}


@app.post("/api/presets/{preset_name}/apply")
async def apply_configuration_preset(preset_name: str):
    """Apply a configuration preset."""
    try:
        # Get available presets
        presets_response = await get_configuration_presets()
        if presets_response["status"] != "success":
            return presets_response

        presets = presets_response["presets"]

        if preset_name not in presets:
            return {"status": "error", "message": f"Preset '{preset_name}' not found"}

        preset_config = presets[preset_name]["config"]

        # Apply the preset configuration
        result = await update_processing_config(preset_config)

        if result["status"] == "success":
            logger.info(f"Applied configuration preset: {preset_name}")
            return {"status": "success", "message": f"Applied preset '{presets[preset_name]['name']}'", "preset": preset_name, "applied_config": preset_config}
        else:
            return result

    except Exception as e:
        logger.error(f"Error applying configuration preset: {e}")
        return {"status": "error", "message": f"Error applying preset: {str(e)}"}


# =============================================================================
# File Management APIs
# =============================================================================


@app.get("/api/files/uploaded")
async def get_uploaded_files():
    """Get list of uploaded files."""
    try:
        import os
        import tempfile
        from pathlib import Path

        temp_dir = Path(tempfile.gettempdir())
        uploaded_files = []

        # Look for audio files in temp directory (this is a simplified approach)
        audio_extensions = {".wav", ".mp3", ".mp4", ".m4a", ".flac", ".ogg", ".webm"}

        for file_path in temp_dir.glob("audioinsight_*"):
            if file_path.suffix.lower() in audio_extensions:
                stat = file_path.stat()
                uploaded_files.append(
                    {
                        "filename": file_path.name,
                        "path": str(file_path),
                        "size": stat.st_size,
                        "created": stat.st_ctime,
                        "modified": stat.st_mtime,
                    }
                )

        # Sort by creation time (newest first)
        uploaded_files.sort(key=lambda x: x["created"], reverse=True)

        return {"status": "success", "files": uploaded_files, "count": len(uploaded_files)}

    except Exception as e:
        logger.error(f"Error getting uploaded files: {e}")
        return {"status": "error", "message": f"Error getting uploaded files: {str(e)}"}


@app.delete("/api/files/{file_path:path}")
async def delete_uploaded_file(file_path: str):
    """Delete a specific uploaded file."""
    try:
        import tempfile
        from pathlib import Path

        # Security check - only allow deletion of files in temp directory
        temp_dir = Path(tempfile.gettempdir())
        full_path = Path(file_path)

        if not str(full_path).startswith(str(temp_dir)):
            return {"status": "error", "message": "Invalid file path"}

        if full_path.exists():
            full_path.unlink()
            logger.info(f"Deleted file: {file_path}")
            return {"status": "success", "message": f"Deleted file: {full_path.name}"}
        else:
            return {"status": "error", "message": "File not found"}

    except Exception as e:
        logger.error(f"Error deleting file: {e}")
        return {"status": "error", "message": f"Error deleting file: {str(e)}"}


@app.post("/api/files/cleanup")
async def cleanup_old_files(max_age_hours: int = 24):
    """Clean up old uploaded files."""
    try:
        import os
        import tempfile
        import time
        from pathlib import Path

        temp_dir = Path(tempfile.gettempdir())
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        deleted_files = []

        audio_extensions = {".wav", ".mp3", ".mp4", ".m4a", ".flac", ".ogg", ".webm"}

        for file_path in temp_dir.glob("audioinsight_*"):
            if file_path.suffix.lower() in audio_extensions:
                file_age = current_time - file_path.stat().st_mtime
                if file_age > max_age_seconds:
                    try:
                        file_path.unlink()
                        deleted_files.append(file_path.name)
                    except Exception as e:
                        logger.warning(f"Failed to delete {file_path}: {e}")

        logger.info(f"Cleaned up {len(deleted_files)} old files")
        return {"status": "success", "message": f"Cleaned up {len(deleted_files)} files older than {max_age_hours} hours", "deleted_files": deleted_files}

    except Exception as e:
        logger.error(f"Error cleaning up files: {e}")
        return {"status": "error", "message": f"Error cleaning up files: {str(e)}"}


# =============================================================================
# LLM Management APIs
# =============================================================================


@app.get("/api/llm/status")
async def get_llm_status():
    """Get LLM processing status and statistics."""
    try:
        # Get display parser stats
        display_parser = get_display_parser()
        display_stats = display_parser.get_stats()

        llm_status = {
            "display_parser": {
                "enabled": display_parser.is_enabled(),
                "model": display_parser.config.model_id if display_parser.config else "openai/gpt-4.1-nano",
                "stats": display_stats,
            },
            "inference": {
                "enabled": getattr(kit.args, "llm_inference", True) if kit else False,
                "fast_model": getattr(kit.args, "fast_llm", "openai/gpt-4.1-nano") if kit else None,
                "base_model": getattr(kit.args, "base_llm", "openai/gpt-4.1-mini") if kit else None,
            },
        }

        return {"status": "success", "llm_status": llm_status}

    except Exception as e:
        logger.error(f"Error getting LLM status: {e}")
        return {"status": "error", "message": f"Error getting LLM status: {str(e)}"}


@app.post("/api/llm/test")
async def test_llm_connection(model_id: str = None):
    """Test LLM connection and model availability."""
    try:
        from langchain.prompts import ChatPromptTemplate

        from .llm import LLMConfig, UniversalLLM

        # Use provided model or default
        test_model = model_id or "openai/gpt-4.1-nano"

        # Create test LLM client
        config = LLMConfig(model_id=test_model)
        llm_client = UniversalLLM(config)

        # Create simple test prompt
        prompt = ChatPromptTemplate.from_messages([("human", "Say 'Hello, AudioInsight!' if you can respond.")])

        # Test the connection
        start_time = time.time()
        result = await llm_client.invoke_text(prompt, {})
        response_time = time.time() - start_time

        return {"status": "success", "message": "LLM connection test successful", "model": test_model, "response": result, "response_time": response_time, "test_timestamp": time.time()}

    except Exception as e:
        logger.error(f"LLM connection test failed: {e}")
        return {"status": "error", "message": f"LLM connection test failed: {str(e)}", "model": test_model if "test_model" in locals() else model_id}


# =============================================================================
# Transcript Parser APIs
# =============================================================================


@app.get("/api/transcript-parser/status")
async def get_transcript_parser_status():
    """Get transcript parser status and statistics."""
    try:
        global current_processor
        if not current_processor or not hasattr(current_processor, "transcript_parser"):
            return {"status": "error", "message": "No active transcript parser"}

        parser = current_processor.transcript_parser
        if not parser:
            return {"status": "success", "enabled": False, "message": "Transcript parser not initialized"}

        return {
            "status": "success",
            "enabled": current_processor._parser_enabled,
            "stats": parser.get_stats(),
            "config": {
                "model_id": parser.config.model_id if parser.config else "openai/gpt-4.1-nano",
                "max_output_tokens": parser.config.max_output_tokens if parser.config else 33000,
            },
            "total_parsed": len(current_processor.parsed_transcripts),
            "last_parsed_available": current_processor.last_parsed_transcript is not None,
        }

    except Exception as e:
        logger.error(f"Error getting transcript parser status: {e}")
        return {"status": "error", "message": f"Error getting parser status: {str(e)}"}


@app.post("/api/transcript-parser/enable")
async def enable_transcript_parser(enabled: bool = True):
    """Enable or disable transcript parsing."""
    try:
        global current_processor
        if not current_processor or not hasattr(current_processor, "enable_transcript_parsing"):
            return {"status": "error", "message": "No active transcript parser"}

        current_processor.enable_transcript_parsing(enabled)
        status = "enabled" if enabled else "disabled"

        return {"status": "success", "message": f"Transcript parsing {status}", "enabled": enabled}

    except Exception as e:
        logger.error(f"Error toggling transcript parser: {e}")
        return {"status": "error", "message": f"Error toggling parser: {str(e)}"}


@app.get("/api/transcript-parser/transcripts")
async def get_parsed_transcripts(limit: int = 10):
    """Get parsed transcripts with optional limit."""
    try:
        global current_processor
        if not current_processor or not hasattr(current_processor, "get_parsed_transcripts"):
            return {"status": "error", "message": "No active transcript parser"}

        all_transcripts = current_processor.get_parsed_transcripts()

        # Apply limit
        if limit > 0:
            transcripts = all_transcripts[-limit:]
        else:
            transcripts = all_transcripts

        return {"status": "success", "transcripts": [t.model_dump() for t in transcripts], "total_count": len(all_transcripts), "returned_count": len(transcripts)}

    except Exception as e:
        logger.error(f"Error getting parsed transcripts: {e}")
        return {"status": "error", "message": f"Error getting transcripts: {str(e)}"}


@app.get("/api/transcript-parser/latest")
async def get_latest_parsed_transcript():
    """Get the most recent parsed transcript."""
    try:
        global current_processor
        if not current_processor or not hasattr(current_processor, "get_last_parsed_transcript"):
            return {"status": "error", "message": "No active transcript parser"}

        latest = current_processor.get_last_parsed_transcript()

        if not latest:
            return {"status": "success", "transcript": None, "message": "No parsed transcripts available"}

        return {"status": "success", "transcript": latest.model_dump(), "message": "Latest parsed transcript retrieved"}

    except Exception as e:
        logger.error(f"Error getting latest parsed transcript: {e}")
        return {"status": "error", "message": f"Error getting latest transcript: {str(e)}"}


# =============================================================================
# Batch Processing APIs
# =============================================================================


@app.post("/api/batch/process")
async def start_batch_processing(file_paths: list[str], processing_config: dict = None):
    """Start batch processing of multiple audio files."""
    try:
        import asyncio
        from pathlib import Path

        # Validate file paths
        valid_files = []
        for file_path in file_paths:
            path = Path(file_path)
            if path.exists() and path.suffix.lower() in {".wav", ".mp3", ".mp4", ".m4a", ".flac", ".ogg", ".webm"}:
                valid_files.append(str(path))
            else:
                logger.warning(f"Invalid or missing file: {file_path}")

        if not valid_files:
            return {"status": "error", "message": "No valid audio files found"}

        # Apply processing configuration if provided
        if processing_config:
            await update_processing_parameters(processing_config)

        # Start batch processing (this is a simplified implementation)
        batch_id = f"batch_{int(time.time())}"

        # In a real implementation, you would queue these for background processing
        # For now, we'll just return the batch information

        batch_info = {"batch_id": batch_id, "files": valid_files, "total_files": len(valid_files), "status": "queued", "created_at": time.time(), "config": processing_config or {}}

        logger.info(f"Started batch processing: {batch_id} with {len(valid_files)} files")

        return {"status": "success", "message": f"Batch processing started with {len(valid_files)} files", "batch": batch_info}

    except Exception as e:
        logger.error(f"Error starting batch processing: {e}")
        return {"status": "error", "message": f"Error starting batch processing: {str(e)}"}


@app.get("/api/batch/{batch_id}/status")
async def get_batch_status(batch_id: str):
    """Get status of a batch processing job."""
    try:
        # In a real implementation, you would track batch jobs in a database or cache
        # For now, we'll return a placeholder response

        batch_status = {"batch_id": batch_id, "status": "completed", "processed_files": 0, "total_files": 0, "failed_files": 0, "progress_percent": 100, "started_at": time.time() - 3600, "completed_at": time.time() - 600, "results": []}  # This would be dynamic in a real implementation  # Placeholder: 1 hour ago  # Placeholder: 10 minutes ago

        return {"status": "success", "batch": batch_status}

    except Exception as e:
        logger.error(f"Error getting batch status: {e}")
        return {"status": "error", "message": f"Error getting batch status: {str(e)}"}


# =============================================================================
# Audio Quality Analysis APIs
# =============================================================================


@app.post("/api/audio/analyze")
async def analyze_audio_quality(file_path: str):
    """Analyze audio quality and provide recommendations."""
    try:
        from pathlib import Path

        import librosa
        import numpy as np

        audio_path = Path(file_path)
        if not audio_path.exists():
            return {"status": "error", "message": "Audio file not found"}

        # Load audio file
        y, sr = librosa.load(str(audio_path), sr=None)
        duration = len(y) / sr

        # Basic audio analysis
        # RMS energy
        rms = librosa.feature.rms(y=y)[0]
        avg_rms = np.mean(rms)

        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        avg_zcr = np.mean(zcr)

        # Spectral features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        avg_spectral_centroid = np.mean(spectral_centroids)

        # Basic noise estimation (simplified)
        noise_estimate = np.std(y[: int(0.1 * sr)])  # First 100ms as noise reference

        # Calculate SNR estimate
        signal_power = np.mean(y**2)
        noise_power = noise_estimate**2
        snr_estimate = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float("inf")

        # Generate recommendations
        recommendations = []

        if avg_rms < 0.01:
            recommendations.append("Audio level is very low. Consider increasing input gain.")
        elif avg_rms > 0.8:
            recommendations.append("Audio level is very high. Risk of clipping.")

        if snr_estimate < 10:
            recommendations.append("High noise level detected. Consider noise reduction.")

        if sr < 16000:
            recommendations.append("Sample rate below 16kHz may affect transcription quality.")

        analysis = {
            "file_info": {
                "filename": audio_path.name,
                "duration": duration,
                "sample_rate": sr,
                "channels": 1,  # librosa loads as mono by default
            },
            "quality_metrics": {
                "average_rms": float(avg_rms),
                "average_zcr": float(avg_zcr),
                "average_spectral_centroid": float(avg_spectral_centroid),
                "estimated_snr": float(snr_estimate),
                "noise_estimate": float(noise_estimate),
            },
            "recommendations": recommendations,
            "overall_quality": "good" if snr_estimate > 15 and 0.01 < avg_rms < 0.8 else "fair" if snr_estimate > 5 else "poor",
        }

        return {"status": "success", "analysis": analysis}

    except Exception as e:
        logger.error(f"Error analyzing audio quality: {e}")
        return {"status": "error", "message": f"Error analyzing audio: {str(e)}"}


# =============================================================================
# Warmup Status Check API (for debugging cold start issues)
# =============================================================================


@app.get("/api/warmup/status")
async def get_warmup_status():
    """Get warmup status to debug cold start issues."""
    try:
        global kit
        if not kit:
            return {"status": "error", "message": "AudioInsight not initialized"}

        warmup_info = {
            "models_loaded": kit._models_loaded,
            "asr_instance": kit.asr is not None,
            "warmup_file_config": getattr(kit.args, "warmup_file", None),
            "backend": getattr(kit.args, "backend", "unknown"),
            "model": getattr(kit.args, "model", "unknown"),
        }

        # Try to check if the ASR model has been warmed up by checking its state
        if kit.asr:
            # Different backends might have different ways to check warmup status
            warmup_info.update(
                {
                    "asr_type": type(kit.asr).__name__,
                    "asr_ready": True,
                }
            )
        else:
            warmup_info.update(
                {
                    "asr_type": None,
                    "asr_ready": False,
                }
            )

        return {"status": "success", "warmup_info": warmup_info}

    except Exception as e:
        logger.error(f"Error getting warmup status: {e}")
        return {"status": "error", "message": f"Error getting warmup status: {str(e)}"}


@app.post("/api/warmup/force")
async def force_warmup():
    """Force a warmup of the ASR model to solve cold start issues."""
    try:
        global kit
        if not kit:
            return {"status": "error", "message": "AudioInsight not initialized"}

        if not kit.asr:
            return {"status": "error", "message": "ASR model not loaded"}

        # Import warmup function
        from .whisper_streaming.whisper_online import warmup_asr

        # Force warmup
        warmup_file = getattr(kit.args, "warmup_file", None)
        logger.info(f"🔥 Forcing warmup with file: {warmup_file}")

        success = warmup_asr(kit.asr, warmup_file)

        if success is False:
            return {"status": "warning", "message": "Warmup completed but may not have used warmup file"}

        return {"status": "success", "message": "ASR model warmup forced successfully", "warmup_file": warmup_file}

    except Exception as e:
        logger.error(f"Error forcing warmup: {e}")
        return {"status": "error", "message": f"Error forcing warmup: {str(e)}"}
