import time
from pathlib import Path

from fastapi import APIRouter

from ..audioinsight_server import error_response, success_response
from ..logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter()

# Import global variables from server_app
import audioinsight.audioinsight_server as app


@router.post("/api/batch/process")
async def start_batch_processing(file_paths: list[str], processing_config: dict = None):
    """Start batch processing of multiple audio files."""
    try:
        # Validate file paths
        valid_files = []
        for file_path in file_paths:
            path = Path(file_path)
            if path.exists() and path.suffix.lower() in {".wav", ".mp3", ".mp4", ".m4a", ".flac", ".ogg", ".webm"}:
                valid_files.append(str(path))
            else:
                logger.warning(f"Invalid or missing file: {file_path}")

        if not valid_files:
            return error_response("No valid audio files found")

        # Apply processing configuration if provided
        if processing_config:
            from .config_api import update_processing_parameters

            await update_processing_parameters(processing_config)

        # Start batch processing (this is a simplified implementation)
        batch_id = f"batch_{int(time.time())}"

        # In a real implementation, you would queue these for background processing
        # For now, we'll just return the batch information

        batch_info = {"batch_id": batch_id, "files": valid_files, "total_files": len(valid_files), "status": "queued", "created_at": time.time(), "config": processing_config or {}}

        logger.info(f"Started batch processing: {batch_id} with {len(valid_files)} files")

        return success_response(f"Batch processing started with {len(valid_files)} files", {"batch": batch_info})

    except Exception as e:
        logger.error(f"Error starting batch processing: {e}")
        return error_response(f"Error starting batch processing: {str(e)}")


@router.get("/api/batch/{batch_id}/status")
async def get_batch_status(batch_id: str):
    """Get status of a batch processing job."""
    try:
        # In a real implementation, you would track batch jobs in a database or cache
        # For now, we'll return a placeholder response

        batch_status = {"batch_id": batch_id, "status": "completed", "processed_files": 0, "total_files": 0, "failed_files": 0, "progress_percent": 100, "started_at": time.time() - 3600, "completed_at": time.time() - 600, "results": []}  # This would be dynamic in a real implementation  # Placeholder: 1 hour ago  # Placeholder: 10 minutes ago

        return success_response("Batch status retrieved", {"batch": batch_status})

    except Exception as e:
        logger.error(f"Error getting batch status: {e}")
        return error_response(f"Error getting batch status: {str(e)}")
