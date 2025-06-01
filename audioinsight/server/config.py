"""Server configuration with runtime vs startup settings."""

from typing import List

from ..config import get_config

# =============================================================================
# Legacy Constants and Compatibility
# =============================================================================


def get_server_settings():
    """Get server settings from main config."""
    config = get_config()
    return {
        "host": config.server.host,
        "port": config.server.port,
        "cors_origins": config.server.cors_origins,
        "cors_credentials": config.server.cors_credentials,
        "cors_methods": config.server.cors_methods,
        "cors_headers": config.server.cors_headers,
        "ssl_certfile": config.server.ssl_certfile,
        "ssl_keyfile": config.server.ssl_keyfile,
    }


def get_audio_settings():
    """Get audio settings from main config."""
    config = get_config()
    return {
        "allowed_types": list(config.audio.allowed_types),
        "chunk_size": config.audio.chunk_size,
        "progress_log_interval": config.audio.progress_log_interval,
        "ffmpeg_params": config.audio.ffmpeg_params,
    }


# Legacy compatibility constants
config = get_config()

# Audio file type validation
ALLOWED_AUDIO_TYPES = list(config.audio.allowed_types)

# Audio processing settings
CHUNK_SIZE = config.audio.chunk_size
PROGRESS_LOG_INTERVAL_SECONDS = config.audio.progress_log_interval

# FFmpeg settings
FFMPEG_AUDIO_PARAMS = config.audio.ffmpeg_params
FFPROBE_DURATION_CMD = ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0"]

# CORS settings - convert to dict format expected by FastAPI
CORS_SETTINGS = {
    "allow_origins": config.server.cors_origins,
    "allow_credentials": config.server.cors_credentials,
    "allow_methods": config.server.cors_methods,
    "allow_headers": config.server.cors_headers,
}

# Server-Sent Events settings
SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "Access-Control-Allow-Origin": "*",
}


# =============================================================================
# Domain-Specific Configuration Management
# =============================================================================


def get_runtime_settings() -> dict:
    """Get all runtime configurable server settings for settings page."""
    config = get_config()

    return {
        # CORS settings
        "cors_origins": config.server.cors_origins,
        "cors_credentials": config.server.cors_credentials,
        "cors_methods": config.server.cors_methods,
        "cors_headers": config.server.cors_headers,
        # Audio settings
        "allowed_types": list(config.audio.allowed_types),
        "chunk_size": config.audio.chunk_size,
        "progress_log_interval": config.audio.progress_log_interval,
    }


def get_startup_settings() -> dict:
    """Get startup-only server settings (not configurable at runtime)."""
    config = get_config()

    return {
        # Network settings
        "host": config.server.host,
        "port": config.server.port,
        "ssl_certfile": config.server.ssl_certfile,
        "ssl_keyfile": config.server.ssl_keyfile,
        # Audio processing
        "ffmpeg_params": config.audio.ffmpeg_params,
    }


def update_runtime_config(updates: dict) -> dict:
    """Update runtime server configuration and return the updated values."""
    config = get_config()
    updated = {}

    # Server-specific fields that can be updated at runtime
    server_fields = {"cors_origins", "cors_credentials", "cors_methods", "cors_headers"}

    audio_fields = {"allowed_types", "chunk_size", "progress_log_interval"}

    for key, value in updates.items():
        if key in server_fields and hasattr(config.server, key):
            setattr(config.server, key, value)
            updated[key] = value
        elif key in audio_fields and hasattr(config.audio, key):
            # Convert allowed_types list back to set for consistency
            if key == "allowed_types" and isinstance(value, list):
                value = set(value)
            setattr(config.audio, key, value)
            updated[key] = value

    return updated
