from fastapi import APIRouter

from ..audioinsight_server import error_response, success_response
from ..logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter()

# Import global variables from server_app
import audioinsight.audioinsight_server as app


@router.get("/api/models/status")
async def get_models_status():
    """Get the status of all loaded models."""
    try:
        if not app.kit:
            return error_response("AudioInsight not initialized")

        models_status = {
            "asr": {
                "loaded": app.kit._models_loaded,
                "model_name": getattr(app.kit.args, "model", "unknown"),
                "backend": getattr(app.kit.args, "backend", "unknown"),
                "language": getattr(app.kit.args, "lang", "auto"),
                "ready": app.kit.asr is not None if hasattr(app.kit, "asr") else False,
            },
            "diarization": {
                "loaded": app.kit._diarization_loaded,
                "enabled": getattr(app.kit.args, "diarization", False),
                "ready": app.kit.diarization is not None if hasattr(app.kit, "diarization") else False,
            },
            "llm": {
                "fast_model": getattr(app.kit.args, "fast_llm", "openai/gpt-4.1-nano"),
                "base_model": getattr(app.kit.args, "base_llm", "openai/gpt-4.1-mini"),
                "inference_enabled": getattr(app.kit.args, "llm_inference", True),
            },
        }

        return success_response("Models status retrieved", {"models": models_status})
    except Exception as e:
        logger.error(f"Error getting models status: {e}")
        return error_response(f"Error getting models status: {str(e)}")


@router.post("/api/models/reload")
async def reload_models(model_type: str = "all"):
    """Reload specific models or all models."""
    try:
        if not app.kit:
            return error_response("AudioInsight not initialized")

        reloaded = []

        if model_type in ["all", "asr"] and hasattr(app.kit, "_load_asr_models"):
            app.kit._models_loaded = False
            app.kit._load_asr_models()
            app.kit._models_loaded = True
            reloaded.append("asr")

        if model_type in ["all", "diarization"] and hasattr(app.kit, "_load_diarization"):
            app.kit._diarization_loaded = False
            app.kit._load_diarization()
            app.kit._diarization_loaded = True
            reloaded.append("diarization")

        logger.info(f"Reloaded models: {reloaded}")
        return success_response(f"Reloaded models: {', '.join(reloaded)}", {"reloaded": reloaded})

    except Exception as e:
        logger.error(f"Error reloading models: {e}")
        return error_response(f"Error reloading models: {str(e)}")


@router.post("/api/models/unload")
async def unload_models(model_type: str = "all"):
    """Unload specific models to free memory."""
    try:
        if not app.kit:
            return error_response("AudioInsight not initialized")

        unloaded = []

        if model_type in ["all", "asr"]:
            app.kit.asr = None
            app.kit.tokenizer = None
            app.kit._models_loaded = False
            unloaded.append("asr")

        if model_type in ["all", "diarization"]:
            if hasattr(app.kit, "diarization") and app.kit.diarization:
                app.kit.diarization.close()
            app.kit.diarization = None
            app.kit._diarization_loaded = False
            unloaded.append("diarization")

        logger.info(f"Unloaded models: {unloaded}")
        return success_response(f"Unloaded models: {', '.join(unloaded)}", {"unloaded": unloaded})

    except Exception as e:
        logger.error(f"Error unloading models: {e}")
        return error_response(f"Error unloading models: {str(e)}")


@router.get("/api/warmup/status")
async def get_warmup_status():
    """Get warmup status to debug cold start issues."""
    try:
        if not app.kit:
            return error_response("AudioInsight not initialized")

        warmup_info = {
            "models_loaded": app.kit._models_loaded,
            "asr_instance": app.kit.asr is not None,
            "warmup_file_config": getattr(app.kit.args, "warmup_file", None),
            "backend": getattr(app.kit.args, "backend", "unknown"),
            "model": getattr(app.kit.args, "model", "unknown"),
        }

        # Try to check if the ASR model has been warmed up by checking its state
        if app.kit.asr:
            # Different backends might have different ways to check warmup status
            warmup_info.update(
                {
                    "asr_type": type(app.kit.asr).__name__,
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

        return success_response("Warmup status retrieved", {"warmup_info": warmup_info})

    except Exception as e:
        logger.error(f"Error getting warmup status: {e}")
        return error_response(f"Error getting warmup status: {str(e)}")


@router.post("/api/warmup/force")
async def force_warmup():
    """Force a warmup of the ASR model to solve cold start issues."""
    try:
        if not app.kit:
            return error_response("AudioInsight not initialized")

        if not app.kit.asr:
            return error_response("ASR model not loaded")

        # Import warmup function
        from ..whisper_streaming.whisper_online import warmup_asr

        # Force warmup
        warmup_file = getattr(app.kit.args, "warmup_file", None)
        logger.info(f"🔥 Forcing warmup with file: {warmup_file}")

        success = warmup_asr(app.kit.asr, warmup_file)

        if success is False:
            return {"status": "warning", "message": "Warmup completed but may not have used warmup file"}

        return success_response("ASR model warmup forced successfully", {"warmup_file": warmup_file})

    except Exception as e:
        logger.error(f"Error forcing warmup: {e}")
        return error_response(f"Error forcing warmup: {str(e)}")
