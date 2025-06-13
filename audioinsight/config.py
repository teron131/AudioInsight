from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

# =============================================================================
# Core Configuration Classes
# =============================================================================


class ServerConfig(BaseModel):
    """Server configuration settings."""

    host: str = Field(default="localhost", description="Server bind address")
    port: int = Field(default=8080, ge=1, le=65535, description="Server port")
    ssl_certfile: Optional[str] = Field(default=None, description="SSL certificate file path")
    ssl_keyfile: Optional[str] = Field(default=None, description="SSL private key file path")

    # CORS settings
    cors_origins: List[str] = Field(default=["*"], description="Allowed CORS origins")
    cors_credentials: bool = Field(default=True, description="Allow CORS credentials")
    cors_methods: List[str] = Field(default=["*"], description="Allowed CORS methods")
    cors_headers: List[str] = Field(default=["*"], description="Allowed CORS headers")


class ModelConfig(BaseModel):
    """AI model configuration settings."""

    # Whisper ASR Configuration
    model: str = Field(default="large-v3-turbo", description="Whisper model size")
    backend: str = Field(default="faster-whisper", description="Whisper backend")
    language: str = Field(default="auto", description="Source language code or 'auto'")
    task: str = Field(default="transcribe", description="Processing task: transcribe or translate")
    model_cache_dir: Optional[str] = Field(default=None, description="Model cache directory")
    model_dir: Optional[str] = Field(default=None, description="Model directory override")
    warmup_file: Optional[str] = Field(default=None, description="Audio file for model warmup")


class ProcessingConfig(BaseModel):
    """Audio processing configuration settings."""

    min_chunk_size: float = Field(default=0.25, gt=0, description="Minimum audio chunk size in seconds")
    buffer_trimming: str = Field(default="segment", description="Buffer trimming strategy")
    buffer_trimming_sec: float = Field(default=8.0, gt=0, description="Buffer trimming threshold in seconds")
    vac_chunk_size: float = Field(default=0.04, gt=0, description="VAC sample size in seconds")

    # Derived values (computed automatically)
    samples_per_chunk: Optional[int] = Field(default=None, description="Samples per chunk (16kHz)")
    vac_samples_per_chunk: Optional[int] = Field(default=None, description="VAC samples per chunk (16kHz)")
    buffer_trimming_samples: Optional[int] = Field(default=None, description="Buffer trimming samples (16kHz)")


class FeatureConfig(BaseModel):
    """Feature flags configuration."""

    transcription: bool = Field(default=True, description="Enable transcription")
    diarization: bool = Field(default=False, description="Enable speaker diarization")
    vad: bool = Field(default=True, description="Enable Voice Activity Detection")
    vac: bool = Field(default=False, description="Enable Voice Activity Controller")
    confidence_validation: bool = Field(default=False, description="Enable confidence validation")
    llm_inference: bool = Field(default=True, description="Enable LLM-based inference")


class UIConfig(BaseModel):
    """User interface configuration settings."""

    show_lag_info: bool = Field(default=False, description="Show lag information in UI")
    show_speakers: bool = Field(default=False, description="Show speaker labels in transcript")


class LLMConfig(BaseSettings):
    """Language model configuration with environment variable support."""

    # Model identifiers
    fast_llm: str = Field(default="openai/gpt-4.1-nano", description="Fast LLM model for parsing")
    base_llm: str = Field(default="openai/gpt-4.1-mini", description="Base LLM model for analysis")

    # API Configuration
    api_key: Optional[str] = Field(default=None, env="OPENROUTER_API_KEY", description="LLM API key")
    temperature: float = Field(default=0.0, ge=0.0, le=2.0, description="Sampling temperature (fixed at 0.0)")
    timeout: float = Field(default=30.0, gt=0, le=300, description="Request timeout in seconds")

    # Trigger Configuration
    llm_analysis_interval: float = Field(default=1.0, gt=0, description="Minimum time between analyses")
    llm_new_text_trigger: int = Field(default=50, gt=0, description="Characters to trigger new analysis")

    # Parser Configuration
    parser_trigger_interval: float = Field(default=1.0, gt=0, description="Parser trigger interval")
    parser_output_tokens: int = Field(default=33000, gt=1000, le=100000, description="Parser max output tokens")

    # Analyzer Configuration
    analyzer_output_tokens: int = Field(default=4000, gt=100, le=8000, description="Analyzer max output tokens")
    analyzer_max_input_length: int = Field(default=2000000, gt=10000, description="Max input length for analyzer")

    class Config:
        env_file = ".env"
        env_prefix = "LLM_"
        extra = "ignore"

    def model_post_init(self, __context) -> None:
        """Ensure temperature is always 0.0 for consistent results."""
        object.__setattr__(self, "temperature", 0.0)


class AudioConfig(BaseModel):
    """Audio processing configuration."""

    # Supported audio types
    allowed_types: Set[str] = Field(
        default={
            "audio/wav",
            "audio/wave",
            "audio/x-wav",
            "audio/mpeg",
            "audio/mp3",
            "audio/flac",
            "audio/x-flac",
            "audio/mp4",
            "audio/x-m4a",
            "audio/ogg",
            "audio/webm",
        },
        description="Allowed audio file MIME types",
    )

    # Processing settings
    chunk_size: int = Field(default=2048, description="Audio chunk size for processing")
    progress_log_interval: float = Field(default=2.0, description="Progress logging interval in seconds")

    # FFmpeg settings
    ffmpeg_params: List[str] = Field(
        default=["-f", "webm", "-c:a", "libopus", "-ar", "16000", "-ac", "1"],
        description="FFmpeg audio conversion parameters",
    )


class LoggingConfig(BaseModel):
    """Logging configuration."""

    level: str = Field(default="DEBUG", description="Logging level")
    log_dir: Optional[str] = Field(default=None, description="Log directory")
    max_log_size: int = Field(default=10 * 1024 * 1024, description="Max log file size in bytes")
    backup_count: int = Field(default=5, description="Number of backup log files")


# =============================================================================
# Unified Configuration Class
# =============================================================================


class UnifiedConfig(BaseSettings):
    """Unified configuration for AudioInsight application."""

    server: ServerConfig = Field(default_factory=ServerConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    ui: UIConfig = Field(default_factory=UIConfig)

    class Config:
        env_file = ".env"
        extra = "ignore"

    def model_post_init(self, __context: Any) -> None:
        """Compute derived values after initialization."""
        # Compute derived processing values
        if self.processing.samples_per_chunk is None:
            self.processing.samples_per_chunk = int(16000 * self.processing.min_chunk_size)

        if self.processing.vac_samples_per_chunk is None:
            self.processing.vac_samples_per_chunk = int(16000 * self.processing.vac_chunk_size)

        if self.processing.buffer_trimming_samples is None:
            self.processing.buffer_trimming_samples = int(16000 * self.processing.buffer_trimming_sec)


# =============================================================================
# Configuration Constants and Defaults
# =============================================================================


# Default configuration dictionary for backward compatibility
DEFAULT_CONFIG = {
    "server": {
        "host": "localhost",
        "port": 8080,
    },
    "model": {
        "model": "large-v3-turbo",
        "backend": "faster-whisper",
        "language": "auto",  # Unified name (was 'lang' in backend)
        "task": "transcribe",
    },
    "processing": {
        "min_chunk_size": 0.5,
        "buffer_trimming": "segment",
        "buffer_trimming_sec": 15.0,
        "vac_chunk_size": 0.04,
    },
    "features": {
        "transcription": True,
        "diarization": False,
        "vad": True,  # Unified name (was 'vad_enabled' in frontend)
        "vac": False,  # Unified name (was 'vac_enabled' in frontend)
        "confidence_validation": False,
        "llm_inference": True,
    },
    "llm": {
        "fast_llm": "openai/gpt-4.1-nano",
        "base_llm": "openai/gpt-4.1-mini",
        "llm_analysis_interval": 1.0,
        "llm_new_text_trigger": 50,
        "parser_trigger_interval": 1.0,
        "parser_output_tokens": 33000,
    },
}

# Frontend/Backend field mapping for compatibility
FIELD_MAPPING = {
    # Frontend field -> Backend field
    "language": "lang",  # Frontend uses 'language', backend uses 'lang'
    "vad_enabled": "vad",  # Frontend uses 'vad_enabled', backend uses 'vad'
    "vac_enabled": "vac",  # Frontend uses 'vac_enabled', backend uses 'vac'
    # Backend field -> Frontend field (reverse mapping)
    "lang": "language",
    "vad": "vad_enabled",
    "vac": "vac_enabled",
}


# =============================================================================
# Global Configuration Instance
# =============================================================================


# Global configuration instance
_config: Optional[UnifiedConfig] = None


def get_config() -> UnifiedConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = UnifiedConfig()
    return _config


def reset_config() -> None:
    """Reset the global configuration instance."""
    global _config
    _config = None


def update_config(**kwargs) -> UnifiedConfig:
    """Update configuration with new values."""
    global _config
    if _config is None:
        _config = UnifiedConfig(**kwargs)
    else:
        # Update existing config
        for key, value in kwargs.items():
            if hasattr(_config, key):
                setattr(_config, key, value)
    return _config


# =============================================================================
# Compatibility Functions
# =============================================================================


def get_processing_parameters() -> Dict[str, Any]:
    """Get processing parameters in the format expected by the API."""
    config = get_config()

    return {
        # Server Configuration
        "host": config.server.host,
        "port": config.server.port,
        # Model Configuration
        "model": config.model.model,
        "backend": config.model.backend,
        "language": config.model.language,  # Frontend expects 'language'
        "task": config.model.task,
        "model_cache_dir": config.model.model_cache_dir,
        "model_dir": config.model.model_dir,
        # Processing Configuration
        "min_chunk_size": config.processing.min_chunk_size,
        "buffer_trimming": config.processing.buffer_trimming,
        "buffer_trimming_sec": config.processing.buffer_trimming_sec,
        "vac_chunk_size": config.processing.vac_chunk_size,
        "warmup_file": config.model.warmup_file,
        # Feature Configuration (frontend naming)
        "transcription": config.features.transcription,
        "diarization": config.features.diarization,
        "vad_enabled": config.features.vad,  # Frontend expects 'vad_enabled'
        "vac_enabled": config.features.vac,  # Frontend expects 'vac_enabled'
        "confidence_validation": config.features.confidence_validation,
        "llm_inference": config.features.llm_inference,
        # LLM Configuration
        "fast_llm": config.llm.fast_llm,
        "base_llm": config.llm.base_llm,
        "llm_analysis_interval": config.llm.llm_analysis_interval,
        "llm_new_text_trigger": config.llm.llm_new_text_trigger,
        "parser_trigger_interval": config.llm.parser_trigger_interval,
        "parser_output_tokens": config.llm.parser_output_tokens,
        # UI Configuration
        "show_lag_info": config.ui.show_lag_info,
        "show_speakers": config.ui.show_speakers,
    }


def apply_parameter_updates(parameters: Dict[str, Any]) -> List[str]:
    """Apply parameter updates and return list of updated fields."""
    config = get_config()
    updated_fields = []

    # Map frontend field names to backend field names
    for frontend_field, value in parameters.items():
        backend_field = FIELD_MAPPING.get(frontend_field, frontend_field)

        # Apply updates to appropriate config sections
        if backend_field in ["host", "port"]:
            setattr(config.server, backend_field, value)
            updated_fields.append(frontend_field)
        elif backend_field in ["model", "backend", "lang", "language", "task", "model_cache_dir", "model_dir", "warmup_file"]:
            field_name = "language" if backend_field == "lang" else backend_field
            setattr(config.model, field_name, value)
            updated_fields.append(frontend_field)
        elif backend_field in ["min_chunk_size", "buffer_trimming", "buffer_trimming_sec", "vac_chunk_size"]:
            setattr(config.processing, backend_field, value)
            updated_fields.append(frontend_field)
        elif backend_field in ["transcription", "diarization", "vad", "vac", "confidence_validation", "llm_inference"]:
            setattr(config.features, backend_field, value)
            updated_fields.append(frontend_field)
        elif backend_field in ["fast_llm", "base_llm", "llm_analysis_interval", "llm_new_text_trigger", "parser_trigger_interval", "parser_output_tokens"]:
            setattr(config.llm, backend_field, value)
            updated_fields.append(frontend_field)
        elif backend_field in ["show_lag_info", "show_speakers"]:
            setattr(config.ui, backend_field, value)
            updated_fields.append(frontend_field)

    # Recompute derived values
    config.model_post_init(None)

    return updated_fields


# =============================================================================
# Environment Setup
# =============================================================================


def setup_environment() -> None:
    """Setup environment and validate configuration."""
    from dotenv import load_dotenv

    # Load environment variables
    load_dotenv()

    # Validate paths exist
    config = get_config()

    # Create log directory if specified
    if config.logging.log_dir:
        Path(config.logging.log_dir).mkdir(parents=True, exist_ok=True)

    # Validate SSL configuration
    if config.server.ssl_certfile or config.server.ssl_keyfile:
        if not (config.server.ssl_certfile and config.server.ssl_keyfile):
            raise ValueError("Both ssl_certfile and ssl_keyfile must be provided together")

        cert_path = Path(config.server.ssl_certfile)
        key_path = Path(config.server.ssl_keyfile)

        if not cert_path.exists():
            raise FileNotFoundError(f"SSL certificate file not found: {cert_path}")
        if not key_path.exists():
            raise FileNotFoundError(f"SSL key file not found: {key_path}")


# Initialize configuration on import
setup_environment()


# =============================================================================
# Domain-Specific Configuration Coordination
# =============================================================================


def get_runtime_configurable_fields() -> dict:
    """Get all fields that can be configured at runtime via settings page.

    These settings can be changed without restarting the server and will take
    effect immediately for new processing requests.

    Returns:
        Dictionary mapping field names to their current values, grouped by domain.
    """
    from .llm.llm_config import get_runtime_settings as get_llm_runtime
    from .server.server_config import get_runtime_settings as get_server_runtime

    config = get_config()

    # Core processing settings (runtime configurable)
    processing_runtime = {
        # Feature flags
        "transcription": config.features.transcription,
        "diarization": config.features.diarization,
        "vad_enabled": config.features.vad,  # Frontend naming
        "vac_enabled": config.features.vac,  # Frontend naming
        "confidence_validation": config.features.confidence_validation,
        # Model selection
        "model": config.model.model,
        "backend": config.model.backend,
        "language": config.model.language,  # Frontend naming
        "task": config.model.task,
        # Processing parameters
        "min_chunk_size": config.processing.min_chunk_size,
        "buffer_trimming": config.processing.buffer_trimming,
        "buffer_trimming_sec": config.processing.buffer_trimming_sec,
        "vac_chunk_size": config.processing.vac_chunk_size,
        # Optional paths (can be changed at runtime)
        "model_cache_dir": config.model.model_cache_dir,
        "model_dir": config.model.model_dir,
        # UI Configuration
        "show_lag_info": config.ui.show_lag_info,
        "show_speakers": config.ui.show_speakers,
    }

    return {
        "processing": processing_runtime,
        "llm": get_llm_runtime(),
        "server": get_server_runtime(),
    }


def get_startup_only_fields() -> dict:
    """Get all fields that require server restart to change.

    These settings are only configurable via CLI arguments and environment
    variables. Changes require stopping and restarting the server.

    Returns:
        Dictionary mapping field names to their current values, grouped by domain.
    """
    from .llm.llm_config import get_startup_settings as get_llm_startup
    from .server.server_config import get_startup_settings as get_server_startup

    config = get_config()

    # Core startup-only settings
    startup_only = {
        # Audio system configuration (requires restart for model loading)
        "warmup_file": config.model.warmup_file,
        # Logging configuration (affects logging system initialization)
        "log_level": config.logging.level,
        "log_dir": config.logging.log_dir,
    }

    return {
        "core": startup_only,
        "llm": get_llm_startup(),
        "server": get_server_startup(),
    }


def apply_runtime_updates(updates: dict) -> dict:
    """Apply updates to runtime-configurable settings across all domains.

    Args:
        updates: Dictionary of field updates from settings page

    Returns:
        Dictionary of successfully applied updates
    """
    from .llm.llm_config import update_runtime_config as update_llm_runtime
    from .server.server_config import update_runtime_config as update_server_runtime

    all_updated = {}

    # Update core processing settings
    core_updated = apply_parameter_updates(updates)
    if core_updated:
        all_updated["processing"] = core_updated

    # Update LLM settings
    llm_updated = update_llm_runtime(updates)
    if llm_updated:
        all_updated["llm"] = llm_updated

    # Update server settings
    server_updated = update_server_runtime(updates)
    if server_updated:
        all_updated["server"] = server_updated

    return all_updated
