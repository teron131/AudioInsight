from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import AudioInsight, parse_args
from .logging_config import get_logger, setup_logging
from .server.server_config import CORS_SETTINGS

# Ensure logging is initialized early
setup_logging()
logger = get_logger(__name__)

# Global AudioInsight kit instance
kit = None
backend_ready = False  # Flag to track if backend is fully warmed up


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
    global kit, backend_ready
    logger.info("🚀 Starting AudioInsight backend initialization...")

    # Instantiate AudioInsight with the same CLI arguments as the server entrypoint
    args = parse_args()
    kit = AudioInsight(**vars(args))

    # Warm up the AudioProcessor system to ensure workers are ready
    logger.info("🔥 Warming up AudioProcessor system...")
    try:
        from .processors import AudioProcessor

        await AudioProcessor.warm_up_system()

        # Create a test processor to pre-initialize components
        test_processor = AudioProcessor()
        logger.info("✅ Test AudioProcessor created - components pre-initialized")

        # Clean it up
        await test_processor.cleanup()
        logger.info("✅ Backend warmup completed - system ready for connections")
        backend_ready = True

    except Exception as e:
        logger.error(f"❌ Backend warmup failed: {e}")
        # Still set ready=True to allow connections, but log the issue
        backend_ready = True

    yield

    # Cleanup on shutdown
    backend_ready = False


# Create FastAPI application
app = FastAPI(lifespan=lifespan)

# Add CORS middleware
app.add_middleware(CORSMiddleware, **CORS_SETTINGS)

# Include API routers
from .api import (
    analytics_router,
    batch_router,
    config_router,
    core_router,
    file_router,
    llm_router,
    model_router,
    session_router,
)

app.include_router(core_router)
app.include_router(config_router)
app.include_router(session_router)
app.include_router(model_router)
app.include_router(file_router)
app.include_router(llm_router)
app.include_router(batch_router)
app.include_router(analytics_router)


def main():
    """Entry point for the CLI command."""
    args = parse_args()

    uvicorn_kwargs = {
        "app": app,
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


if __name__ == "__main__":
    main()
