"""
Centralized logging configuration for AudioInsight.
All modules should import get_logger from this module to ensure consistent logging.
"""

import logging
import os
from datetime import datetime
from pathlib import Path

# Global variables to store the logging configuration
_logging_initialized = False
_log_filename = None


def setup_logging():
    """Set up logging to both console and file with datetime-based filename."""
    global _logging_initialized, _log_filename

    if _logging_initialized:
        return _log_filename

    # Create logs directory if it doesn't exist
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)

    # Create datetime-based filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    _log_filename = logs_dir / f"audioinsight_{timestamp}.log"

    # Configure logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", handlers=[logging.FileHandler(_log_filename), logging.StreamHandler()])  # Console output

    # Set specific log levels for different modules
    logging.getLogger("audioinsight").setLevel(logging.INFO)
    logging.getLogger("uvicorn").setLevel(logging.WARNING)  # Reduce uvicorn verbosity

    _logging_initialized = True

    # Log the initialization
    logger = logging.getLogger(__name__)
    logger.info(f"📝 Centralized logging initialized - writing to {_log_filename}")

    return _log_filename


def get_logger(name=None):
    """Get a logger with the centralized configuration."""
    if not _logging_initialized:
        setup_logging()

    if name is None:
        name = __name__

    return logging.getLogger(name)


def get_log_filename():
    """Get the current log filename."""
    if not _logging_initialized:
        setup_logging()
    return _log_filename
