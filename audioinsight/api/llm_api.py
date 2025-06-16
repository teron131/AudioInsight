import time

from fastapi import APIRouter

from ..audioinsight_server import error_response, success_response
from ..logging_config import get_logger

logger = get_logger(__name__)

router = APIRouter()

# Import global variables from server_app
import audioinsight.audioinsight_server as app


@router.get("/api/llm/status")
async def get_llm_status():
    """Get LLM processing status and statistics."""
    try:
        llm_status = {
            "inference": {
                "enabled": getattr(app.kit.args, "llm_inference", True) if app.kit else False,
                "fast_model": getattr(app.kit.args, "fast_llm", "openai/gpt-4.1-nano") if app.kit else None,
                "base_model": getattr(app.kit.args, "base_llm", "openai/gpt-4.1-mini") if app.kit else None,
            },
        }

        return success_response("LLM status retrieved", {"llm_status": llm_status})

    except Exception as e:
        logger.error(f"Error getting LLM status: {e}")
        return error_response(f"Error getting LLM status: {str(e)}")


@router.post("/api/llm/test")
async def test_llm_connection(model_id: str = None):
    """Test LLM connection and model availability."""
    try:
        from langchain.prompts import ChatPromptTemplate

        from ..llm import LLMConfig, UniversalLLM

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

        return success_response("LLM connection test successful", {"model": test_model, "response": result, "response_time": response_time, "test_timestamp": time.time()})

    except Exception as e:
        logger.error(f"LLM connection test failed: {e}")
        return error_response(f"LLM connection test failed: {str(e)}", model=test_model if "test_model" in locals() else model_id)


@router.get("/api/transcript-parser/status")
async def get_transcript_parser_status():
    """Get transcript parser status and statistics."""
    try:
        from ..server.websocket_handlers import _global_audio_processor

        current_processor = _global_audio_processor
        if not current_processor or not hasattr(current_processor, "transcript_parser"):
            return error_response("No active transcript parser")

        parser = current_processor.transcript_parser
        if not parser:
            return success_response("Transcript parser not initialized", {"enabled": False, "message": "Transcript parser not initialized"})

        return success_response(
            "Transcript parser status retrieved",
            {
                "enabled": current_processor._parser_enabled,
                "stats": parser.get_stats(),
                "config": {
                    "model_id": parser.config.model_id if parser.config else "openai/gpt-4.1-nano",
                    "max_output_tokens": parser.config.max_output_tokens if parser.config else 33000,
                },
                "total_parsed": len(current_processor.parsed_transcripts),
                "last_parsed_available": current_processor.last_parsed_transcript is not None,
            },
        )

    except Exception as e:
        logger.error(f"Error getting transcript parser status: {e}")
        return error_response(f"Error getting parser status: {str(e)}")


@router.post("/api/transcript-parser/enable")
async def enable_transcript_parser(enabled: bool = True):
    """Enable or disable transcript parsing."""
    try:
        from ..server.websocket_handlers import _global_audio_processor

        current_processor = _global_audio_processor
        if not current_processor or not hasattr(current_processor, "enable_transcript_parsing"):
            return error_response("No active transcript parser")

        current_processor.enable_transcript_parsing(enabled)
        status = "enabled" if enabled else "disabled"

        return success_response(f"Transcript parsing {status}", {"enabled": enabled})

    except Exception as e:
        logger.error(f"Error toggling transcript parser: {e}")
        return error_response(f"Error toggling parser: {str(e)}")


@router.get("/api/transcript-parser/transcripts")
async def get_parsed_transcripts(limit: int = 10):
    """Get parsed transcripts with optional limit."""
    try:
        from ..server.websocket_handlers import _global_audio_processor

        current_processor = _global_audio_processor
        if not current_processor or not hasattr(current_processor, "get_parsed_transcripts"):
            return error_response("No active transcript parser")

        all_transcripts = current_processor.get_parsed_transcripts()

        # Apply limit
        if limit > 0:
            transcripts = all_transcripts[-limit:]
        else:
            transcripts = all_transcripts

        return success_response("Parsed transcripts retrieved", {"transcripts": [t.model_dump() for t in transcripts], "total_count": len(all_transcripts), "returned_count": len(transcripts)})

    except Exception as e:
        logger.error(f"Error getting parsed transcripts: {e}")
        return error_response(f"Error getting transcripts: {str(e)}")


@router.get("/api/transcript-parser/latest")
async def get_latest_parsed_transcript():
    """Get the most recent parsed transcript."""
    try:
        from ..server.websocket_handlers import _global_audio_processor

        current_processor = _global_audio_processor
        if not current_processor or not hasattr(current_processor, "get_last_parsed_transcript"):
            return error_response("No active transcript parser")

        latest = current_processor.get_last_parsed_transcript()

        if not latest:
            return success_response("No parsed transcripts available", {"transcript": None, "message": "No parsed transcripts available"})

        return success_response("Latest parsed transcript retrieved", {"transcript": latest.model_dump(), "message": "Latest parsed transcript retrieved"})

    except Exception as e:
        logger.error(f"Error getting latest parsed transcript: {e}")
        return error_response(f"Error getting latest transcript: {str(e)}")
