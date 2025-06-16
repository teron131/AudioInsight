import time

from fastapi import APIRouter

from .. import AudioInsight, parse_args
from ..audioinsight_server import error_response, success_response
from ..logging_config import get_logger
from ..server.websocket_handlers import cleanup_global_processor

logger = get_logger(__name__)

router = APIRouter()

# Import global variables from server_app
import audioinsight.audioinsight_server as app


@router.get("/api/sessions/current")
async def get_current_session():
    """Get current session information."""
    try:
        if not app.kit:
            return error_response("No active session")

        session_info = {"session_id": getattr(app.kit, "session_id", "default"), "created_at": getattr(app.kit, "created_at", time.time()), "diarization_enabled": getattr(app.kit, "diarization_enabled", False), "llm_enabled": getattr(app.kit, "llm_enabled", False), "status": "active"}

        return success_response("Current session retrieved", {"session": session_info})
    except Exception as e:
        logger.error(f"Error getting current session: {e}")
        return error_response(f"Error getting session: {str(e)}")


@router.post("/api/sessions/reset")
async def reset_session():
    """Reset current session with complete clean state."""
    try:
        logger.info("🧹 Starting comprehensive session reset...")

        # First, clean up the global audio processor
        await cleanup_global_processor()
        logger.info("🧹 Global audio processor cleaned up")

        # Reset the AudioInsight singleton instance to clear all cached state
        if app.kit:
            try:
                app.kit.reset_instance()
                logger.info("🧹 AudioInsight kit instance reset")
            except Exception as e:
                logger.warning(f"Failed to reset kit instance: {e}")

        # Re-initialize with same arguments to ensure fresh state
        args = parse_args()
        app.kit = AudioInsight(**vars(args))

        logger.info("🧹 Comprehensive session reset completed - all resources reset to fresh state")
        return success_response("Session reset completely - fresh state restored", {"timestamp": time.time(), "backend_ready": app.backend_ready})

    except Exception as e:
        logger.error(f"Error during comprehensive session reset: {e}")
        return error_response(f"Reset failed: {str(e)}")


@router.post("/cleanup-session")
async def cleanup_session():
    """Force cleanup of all audio processing resources.

    This endpoint clears all memory, resets processors, and prepares for a fresh session.
    Useful to prevent memory leaks between file uploads or when the UI is refreshed.
    """
    # Use the enhanced reset function
    return await reset_session()
