from typing import Set

# Audio file type validation
ALLOWED_AUDIO_TYPES: Set[str] = {
    "audio/mpeg",
    "audio/mp3",
    "audio/mp4",
    "audio/m4a",
    "audio/wav",
    "audio/flac",
    "audio/ogg",
    "audio/webm",
}

# Audio processing settings
CHUNK_SIZE = 4096
PROGRESS_LOG_INTERVAL_SECONDS = 2.0

# FFmpeg settings
FFMPEG_AUDIO_PARAMS = ["-f", "webm", "-c:a", "libopus", "-ar", "16000", "-ac", "1"]
FFPROBE_DURATION_CMD = ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0"]

# CORS settings
CORS_SETTINGS = {
    "allow_origins": ["*"],
    "allow_credentials": True,
    "allow_methods": ["*"],
    "allow_headers": ["*"],
}

# Server-Sent Events settings
SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "Access-Control-Allow-Origin": "*",
}
