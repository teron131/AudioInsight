from fastapi import APIRouter

from ..audioinsight_server import error_response, success_response
from ..config import (
    apply_runtime_updates,
    get_config,
    get_processing_parameters,
    get_runtime_configurable_fields,
    get_startup_only_fields,
)
from ..logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter()

# Import global variables from server_app
import audioinsight.audioinsight_server as app


@router.get("/api/config/models")
async def get_model_config():
    """Get current model configuration."""
    try:
        if not app.kit:
            return error_response("AudioInsight not initialized")

        # Get current configuration from the kit
        config = {
            "transcription_model": getattr(app.kit, "whisper_model", "openai/whisper-large-v3"),
            "diarization_enabled": getattr(app.kit, "diarization_enabled", True),
            "llm_analysis_enabled": getattr(app.kit, "llm_enabled", False),
            "fast_llm_model": "openai/gpt-4.1-nano",  # Default fast LLM model
        }

        return success_response("Model configuration retrieved", {"config": config})
    except Exception as e:
        logger.error(f"Error getting model config: {e}")
        return error_response(f"Error getting model config: {str(e)}")


@router.post("/api/config/processing")
async def update_processing_config(config: dict):
    """Update processing configuration.

    Args:
        config: Processing configuration settings

    Returns:
        Status of the operation
    """
    try:
        if not app.kit:
            return error_response("AudioInsight not initialized")

        # Update configuration
        updated_fields = []

        if "diarization_enabled" in config:
            app.kit.diarization_enabled = config["diarization_enabled"]
            updated_fields.append("diarization_enabled")

        if "llm_analysis_enabled" in config:
            app.kit.llm_enabled = config["llm_analysis_enabled"]
            updated_fields.append("llm_analysis_enabled")

        if "fast_llm_model" in config:
            # Fast LLM model configuration now handled by global atomic parser
            logger.info(f"Fast LLM model configuration: {config['fast_llm_model']}")
            updated_fields.append("fast_llm_model")

        logger.info(f"Updated processing config: {updated_fields}")
        return success_response(f"Updated configuration: {', '.join(updated_fields)}", {"updated_fields": updated_fields})
    except Exception as e:
        logger.error(f"Error updating processing config: {e}")
        return error_response(f"Error updating config: {str(e)}")


@router.post("/api/processing/parameters")
async def update_processing_parameters(parameters: dict):
    """Update audio processing parameters in real-time."""
    try:
        if not app.kit:
            return error_response("AudioInsight not initialized")

        # Use domain-specific configuration system
        updated_params = apply_runtime_updates(parameters)

        # Also update the kit's args for backward compatibility
        config = get_config()

        # Sync config back to kit.args (for legacy code compatibility)
        if hasattr(app.kit, "args"):
            # Update args with unified config values, mapping unified names to legacy names
            app.kit.args.host = config.server.host
            app.kit.args.port = config.server.port
            app.kit.args.model = config.model.model
            app.kit.args.backend = config.model.backend
            app.kit.args.lang = config.model.language  # Map unified 'language' to legacy 'lang'
            app.kit.args.task = config.model.task
            app.kit.args.min_chunk_size = config.processing.min_chunk_size
            app.kit.args.buffer_trimming = config.processing.buffer_trimming
            app.kit.args.buffer_trimming_sec = config.processing.buffer_trimming_sec
            app.kit.args.vac_chunk_size = config.processing.vac_chunk_size
            app.kit.args.transcription = config.features.transcription
            app.kit.args.diarization = config.features.diarization
            app.kit.args.vad = config.features.vad
            app.kit.args.vac = config.features.vac
            app.kit.args.confidence_validation = config.features.confidence_validation
            app.kit.args.llm_inference = config.features.llm_inference
            app.kit.args.fast_llm = config.llm.fast_llm
            app.kit.args.base_llm = config.llm.base_llm
            app.kit.args.llm_analysis_interval = config.llm.llm_analysis_interval
            app.kit.args.llm_new_text_trigger = config.llm.llm_new_text_trigger

        return success_response("Parameters updated successfully", {"updated_parameters": updated_params, "total_updates": sum(len(domain_updates) for domain_updates in updated_params.values())})

    except Exception as e:
        logger.error(f"Error updating processing parameters: {str(e)}")
        return error_response(f"Failed to update parameters: {str(e)}")


@router.get("/api/processing/parameters")
async def get_processing_parameters_endpoint():
    """Get current audio processing parameters with runtime/startup classification."""
    try:
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
        return error_response(f"Failed to retrieve parameters: {str(e)}")


@router.get("/api/presets")
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

        return success_response("Configuration presets retrieved", {"presets": presets})

    except Exception as e:
        logger.error(f"Error getting configuration presets: {e}")
        return error_response(f"Error getting configuration presets: {str(e)}")


@router.post("/api/presets/{preset_name}/apply")
async def apply_configuration_preset(preset_name: str):
    """Apply a configuration preset."""
    try:
        # Get available presets
        presets_response = await get_configuration_presets()
        if presets_response["status"] != "success":
            return presets_response

        presets = presets_response["presets"]

        if preset_name not in presets:
            return error_response(f"Preset '{preset_name}' not found")

        preset_config = presets[preset_name]["config"]

        # Apply the preset configuration
        result = await update_processing_config(preset_config)

        if result["status"] == "success":
            logger.info(f"Applied configuration preset: {preset_name}")
            return success_response(f"Applied preset '{presets[preset_name]['name']}'", {"preset": preset_name, "applied_config": preset_config})
        else:
            return result

    except Exception as e:
        logger.error(f"Error applying configuration preset: {e}")
        return error_response(f"Error applying preset: {str(e)}")
