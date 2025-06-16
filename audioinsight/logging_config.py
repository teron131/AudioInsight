import logging
import os
from pathlib import Path

# Global variables to store the logging configuration
_logging_initialized = False
_log_filename = None


def setup_logging():
    """Set up logging to both console and file with fixed filename."""
    global _logging_initialized, _log_filename

    if _logging_initialized:
        return _log_filename

    # Get the main directory (current working directory)
    main_dir = Path(os.getcwd())

    # Create logs directory if it doesn't exist
    logs_dir = main_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    # Use fixed filename that overwrites previous logs
    _log_filename = logs_dir / "last_run.log"

    # Clear the log file by truncating it
    if _log_filename.exists():
        _log_filename.unlink()  # Delete the file
    _log_filename.touch()  # Create an empty file

    # Get the root logger and clear any existing handlers
    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    # Configure logging with mode='a' after clearing the file
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=[
            logging.FileHandler(_log_filename, mode="a"),  # Append mode
            logging.StreamHandler(),  # Console output
        ],
        force=True,  # Force reconfiguration
    )

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
