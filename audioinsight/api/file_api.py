import os
import tempfile
import time
from pathlib import Path

from fastapi import APIRouter

from ..logging_config import get_logger
from ..audioinsight_server import error_response, success_response

logger = get_logger(__name__)

router = APIRouter()


@router.get("/api/files/uploaded")
async def get_uploaded_files():
    """Get list of uploaded files."""
    try:
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

        return success_response("Uploaded files retrieved", {"files": uploaded_files, "count": len(uploaded_files)})

    except Exception as e:
        logger.error(f"Error getting uploaded files: {e}")
        return error_response(f"Error getting uploaded files: {str(e)}")


@router.delete("/api/files/{file_path:path}")
async def delete_uploaded_file(file_path: str):
    """Delete a specific uploaded file."""
    try:
        temp_dir = Path(tempfile.gettempdir())
        full_path = Path(file_path)

        # Security check - only allow deletion of files in temp directory
        if not str(full_path).startswith(str(temp_dir)):
            return error_response("Invalid file path")

        if full_path.exists():
            full_path.unlink()
            logger.info(f"Deleted file: {file_path}")
            return success_response(f"Deleted file: {full_path.name}")
        else:
            return error_response("File not found")

    except Exception as e:
        logger.error(f"Error deleting file: {e}")
        return error_response(f"Error deleting file: {str(e)}")


@router.post("/api/files/cleanup")
async def cleanup_old_files(max_age_hours: int = 24):
    """Clean up old uploaded files."""
    try:
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
        return success_response(f"Cleaned up {len(deleted_files)} files older than {max_age_hours} hours", {"deleted_files": deleted_files})

    except Exception as e:
        logger.error(f"Error cleaning up files: {e}")
        return error_response(f"Error cleaning up files: {str(e)}")
