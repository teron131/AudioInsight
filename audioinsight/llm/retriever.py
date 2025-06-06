import time
from pathlib import Path
from typing import Optional

from ..logging_config import get_logger

logger = get_logger(__name__)

# RAG knowledge base cache
_rag_content_cache = None
_rag_cache_timestamp = 0
_cache_ttl = 300  # Cache for 5 minutes


class SimpleRetriever:
    """
    Simple file-based retriever that loads context from demo.txt.

    Features:
    - File-based caching with automatic reload on file changes
    - Configurable cache TTL
    - Graceful error handling
    - Formatted context preparation for LLM prompts
    """

    def __init__(self, knowledge_file: str = "demo.txt", cache_ttl: int = 300):
        """Initialize the retriever.

        Args:
            knowledge_file: Name of the knowledge file in data/ directory
            cache_ttl: Cache time-to-live in seconds
        """
        self.knowledge_file = knowledge_file
        self.cache_ttl = cache_ttl

        # Path to knowledge file in the data directory
        project_root = Path(__file__).parent.parent.parent
        self.file_path = project_root / "data" / knowledge_file

        logger.info(f"RAG Retriever initialized for file: {self.file_path}")

    def load_context(self, force_reload: bool = False) -> str:
        """Load RAG context from the knowledge file with caching.

        Args:
            force_reload: Force reload from file even if cached

        Returns:
            str: RAG context content or empty string if file not found
        """
        global _rag_content_cache, _rag_cache_timestamp

        current_time = time.time()

        # Check if we need to reload based on TTL or file modification
        try:
            file_mtime = self.file_path.stat().st_mtime
            cache_expired = (current_time - _rag_cache_timestamp) > self.cache_ttl
            file_modified = file_mtime > _rag_cache_timestamp

            if not force_reload and _rag_content_cache is not None and not cache_expired and not file_modified:
                logger.debug("Using cached RAG context")
                return _rag_content_cache

        except (FileNotFoundError, OSError) as e:
            logger.warning(f"Knowledge file not found: {self.file_path}")
            _rag_content_cache = ""
            _rag_cache_timestamp = current_time
            return ""

        # Load content from file
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()

            _rag_content_cache = content
            _rag_cache_timestamp = current_time

            logger.debug(f"Loaded RAG context: {len(content)} characters")
            return content

        except Exception as e:
            logger.error(f"Failed to load RAG context from {self.file_path}: {e}")
            _rag_content_cache = ""
            _rag_cache_timestamp = current_time
            return ""

    def prepare_context(self, include_separator: bool = True) -> str:
        """Prepare RAG context for inclusion in LLM prompts.

        Args:
            include_separator: Whether to include a separator line after context

        Returns:
            str: Formatted RAG context or empty string if no context available
        """
        rag_content = self.load_context()

        if not rag_content:
            return ""

        # Format as system context with clear separation
        if include_separator:
            return f"""

## System Context
{rag_content}

---

"""
        else:
            return f"""

## System Context
{rag_content}

"""

    def get_context_info(self) -> dict:
        """Get information about the current context.

        Returns:
            dict: Context information including file path, size, and cache status
        """
        global _rag_content_cache, _rag_cache_timestamp

        context = self.load_context()
        current_time = time.time()
        cache_age = current_time - _rag_cache_timestamp if _rag_cache_timestamp > 0 else 0

        try:
            file_exists = self.file_path.exists()
            file_size = self.file_path.stat().st_size if file_exists else 0
            file_mtime = self.file_path.stat().st_mtime if file_exists else 0
        except Exception:
            file_exists = False
            file_size = 0
            file_mtime = 0

        return {
            "file_path": str(self.file_path),
            "file_exists": file_exists,
            "file_size": file_size,
            "content_length": len(context),
            "cache_age_seconds": cache_age,
            "cache_ttl": self.cache_ttl,
            "is_cached": _rag_content_cache is not None,
            "last_modified": file_mtime,
        }

    def clear_cache(self):
        """Clear the RAG content cache."""
        global _rag_content_cache, _rag_cache_timestamp
        _rag_content_cache = None
        _rag_cache_timestamp = 0
        logger.info("RAG cache cleared")


# Global retriever instance for convenience
_default_retriever = None


def get_default_retriever() -> SimpleRetriever:
    """Get the default global retriever instance.

    Returns:
        SimpleRetriever: Default retriever instance
    """
    global _default_retriever
    if _default_retriever is None:
        _default_retriever = SimpleRetriever()
    return _default_retriever


def load_rag_context(force_reload: bool = False) -> str:
    """Convenience function to load RAG context using the default retriever.

    Args:
        force_reload: Force reload from file even if cached

    Returns:
        str: RAG context content or empty string if file not found
    """
    return get_default_retriever().load_context(force_reload)


def prepare_rag_context(include_separator: bool = True) -> str:
    """Convenience function to prepare RAG context using the default retriever.

    Args:
        include_separator: Whether to include a separator line after context

    Returns:
        str: Formatted RAG context or empty string if no context available
    """
    return get_default_retriever().prepare_context(include_separator)


def get_rag_info() -> dict:
    """Convenience function to get RAG context info using the default retriever.

    Returns:
        dict: Context information
    """
    return get_default_retriever().get_context_info()


def clear_rag_cache():
    """Convenience function to clear RAG cache using the default retriever."""
    get_default_retriever().clear_cache()
