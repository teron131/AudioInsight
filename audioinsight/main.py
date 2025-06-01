from argparse import ArgumentParser, Namespace
from typing import Any, Dict, Optional

from .config import DEFAULT_CONFIG, UnifiedConfig, get_config, reset_config
from .logging_config import get_logger, setup_logging
from .whisper_streaming.whisper_online import backend_factory, warmup_asr

# Initialize centralized logging early
setup_logging()
logger = get_logger(__name__)

# Global cached parser to avoid recreation
_CACHED_PARSER: Optional[ArgumentParser] = None


def _get_argument_parser() -> ArgumentParser:
    """Get cached argument parser or create new one."""
    global _CACHED_PARSER

    if _CACHED_PARSER is not None:
        return _CACHED_PARSER

    parser = ArgumentParser(description="Whisper FastAPI Online Server")

    # Server configuration (STARTUP ONLY - requires restart)
    server_group = parser.add_argument_group("Server Configuration (Startup Only - Requires Restart)")
    server_group.add_argument("--host", type=str, default=DEFAULT_CONFIG["server"]["host"], help="The host address to bind the server to. [STARTUP ONLY]")
    server_group.add_argument("--port", type=int, default=DEFAULT_CONFIG["server"]["port"], help="The port number to bind the server to. [STARTUP ONLY]")
    server_group.add_argument("--ssl-certfile", type=str, default=None, help="Path to the SSL certificate file. [STARTUP ONLY]")
    server_group.add_argument("--ssl-keyfile", type=str, default=None, help="Path to the SSL private key file. [STARTUP ONLY]")

    # Model configuration (RUNTIME - can be changed in settings)
    model_group = parser.add_argument_group("Model Configuration (Runtime Configurable)")
    model_group.add_argument("--model", type=str, default=DEFAULT_CONFIG["model"]["model"], help="Name size of the Whisper model to use (default: tiny). Suggested values: tiny.en,tiny,base.en,base,small.en,small,medium.en,medium,large-v3-turbo. [RUNTIME CONFIGURABLE]")
    model_group.add_argument("--model_cache_dir", type=str, default=None, help="Overriding the default model cache dir where models downloaded from the hub are saved. [RUNTIME CONFIGURABLE]")
    model_group.add_argument("--model_dir", type=str, default=None, help="Model dir where model files are stored (for local models). [RUNTIME CONFIGURABLE]")
    model_group.add_argument("--backend", type=str, default=DEFAULT_CONFIG["model"]["backend"], choices=["faster-whisper", "openai-api"], help="The Whisper backend to use. [RUNTIME CONFIGURABLE]")
    model_group.add_argument("--language", type=str, default=DEFAULT_CONFIG["model"]["language"], help="Language to transcribe from (default: auto-detect). [RUNTIME CONFIGURABLE]")
    model_group.add_argument("--task", type=str, default=DEFAULT_CONFIG["model"]["task"], choices=["transcribe", "translate"], help="Task to perform (default: transcribe). [RUNTIME CONFIGURABLE]")
    model_group.add_argument("--warmup_file", type=str, default=None, help="Warmup file for the ASR model (default: None). [STARTUP ONLY]")

    # Processing configuration (RUNTIME - can be changed in settings)
    processing_group = parser.add_argument_group("Processing Configuration (Runtime Configurable)")
    processing_group.add_argument("--min_chunk_size", type=float, default=DEFAULT_CONFIG["processing"]["min_chunk_size"], help="Minimum chunk size for processing (in seconds). [RUNTIME CONFIGURABLE]")
    processing_group.add_argument("--buffer_trimming", type=str, default=DEFAULT_CONFIG["processing"]["buffer_trimming"], choices=["segment", "sentence"], help="Buffer trimming strategy. [RUNTIME CONFIGURABLE]")
    processing_group.add_argument("--buffer_trimming_sec", type=float, default=DEFAULT_CONFIG["processing"]["buffer_trimming_sec"], help="Buffer trimming time in seconds. [RUNTIME CONFIGURABLE]")
    processing_group.add_argument("--vac_chunk_size", type=float, default=DEFAULT_CONFIG["processing"]["vac_chunk_size"], help="Chunk size for VAC processing (in seconds). [RUNTIME CONFIGURABLE]")

    # Feature flags (RUNTIME - can be changed in settings)
    features_group = parser.add_argument_group("Feature Configuration (Runtime Configurable)")
    features_group.add_argument("--no_transcription", action="store_true", help="Disable transcription (for testing or diarization-only mode). [RUNTIME CONFIGURABLE]")
    features_group.add_argument("--diarization", action="store_true", help="Enable speaker diarization. [RUNTIME CONFIGURABLE]")
    features_group.add_argument("--no_vad", action="store_true", help="Disable Voice Activity Detection. [RUNTIME CONFIGURABLE]")
    features_group.add_argument("--vac", action="store_true", help="Enable Voice Activity Controller. [RUNTIME CONFIGURABLE]")
    features_group.add_argument("--confidence_validation", action="store_true", help="Enable confidence-based validation for faster streaming. [RUNTIME CONFIGURABLE]")

    # LLM configuration (RUNTIME - can be changed in settings)
    llm_group = parser.add_argument_group("LLM Configuration (Runtime Configurable)")
    llm_group.add_argument("--llm_inference", action="store_true", default=DEFAULT_CONFIG["llm"]["fast_llm"], help="Enable LLM-based transcript analysis. [RUNTIME CONFIGURABLE]")
    llm_group.add_argument("--fast_llm", type=str, default=DEFAULT_CONFIG["llm"]["fast_llm"], help="Fast LLM model for text parsing. [RUNTIME CONFIGURABLE]")
    llm_group.add_argument("--base_llm", type=str, default=DEFAULT_CONFIG["llm"]["base_llm"], help="Base LLM model for summarization. [RUNTIME CONFIGURABLE]")
    llm_group.add_argument("--llm_summary_interval", type=float, default=DEFAULT_CONFIG["llm"]["llm_summary_interval"], help="LLM summary trigger interval in seconds. [RUNTIME CONFIGURABLE]")
    llm_group.add_argument("--llm_new_text_trigger", type=int, default=DEFAULT_CONFIG["llm"]["llm_new_text_trigger"], help="Text length trigger for LLM processing. [RUNTIME CONFIGURABLE]")
    llm_group.add_argument("--parser_trigger_interval", type=float, default=DEFAULT_CONFIG["llm"]["parser_trigger_interval"], help="Parser trigger interval in seconds. [RUNTIME CONFIGURABLE]")
    llm_group.add_argument("--parser_output_tokens", type=int, default=DEFAULT_CONFIG["llm"]["parser_output_tokens"], help="Maximum parser output tokens. [RUNTIME CONFIGURABLE]")

    # Development and debugging (STARTUP ONLY)
    debug_group = parser.add_argument_group("Development & Debugging (Startup Only)")
    debug_group.add_argument("--log_level", type=str, default="DEBUG", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level. [STARTUP ONLY]")

    _CACHED_PARSER = parser
    return parser


def _validate_args(args: Namespace) -> None:
    """Validate argument combinations and values."""
    # Validate port range
    if not (1 <= args.port <= 65535):
        raise ValueError(f"Port must be between 1 and 65535, got {args.port}")

    # Validate chunk sizes
    if args.min_chunk_size <= 0:
        raise ValueError(f"min_chunk_size must be positive, got {args.min_chunk_size}")

    if args.vac_chunk_size <= 0:
        raise ValueError(f"vac_chunk_size must be positive, got {args.vac_chunk_size}")

    if args.buffer_trimming_sec <= 0:
        raise ValueError(f"buffer_trimming_sec must be positive, got {args.buffer_trimming_sec}")

    # Validate SSL configuration
    if (args.ssl_certfile is None) != (args.ssl_keyfile is None):
        raise ValueError("Both ssl_certfile and ssl_keyfile must be provided together or not at all")

    # Validate LLM configuration
    if args.llm_summary_interval <= 0:
        raise ValueError(f"llm_summary_interval must be positive, got {args.llm_summary_interval}")

    if args.llm_new_text_trigger <= 0:
        raise ValueError(f"llm_new_text_trigger must be positive, got {args.llm_new_text_trigger}")

    if args.parser_trigger_interval <= 0:
        raise ValueError(f"parser_trigger_interval must be positive, got {args.parser_trigger_interval}")

    if args.parser_output_tokens <= 0:
        raise ValueError(f"parser_output_tokens must be positive, got {args.parser_output_tokens}")

    if args.parser_output_tokens > 100000:
        raise ValueError(f"parser_output_tokens too high (max 100000), got {args.parser_output_tokens}")

    # Validate feature combinations
    if args.no_transcription and not args.diarization:
        raise ValueError("Cannot disable transcription without enabling diarization")


def _optimize_args(args: Namespace) -> Namespace:
    """Optimize and precompute derived values from arguments."""
    # Convert boolean flags efficiently - use setattr for better performance than delattr + recreation
    args.transcription = not getattr(args, "no_transcription", False)
    args.vad = not getattr(args, "no_vad", False)

    # Remove the temporary negative flags
    for attr in ["no_transcription", "no_vad"]:
        if hasattr(args, attr):
            delattr(args, attr)

    # Unify language field - maintain both 'lang' and 'language' for compatibility
    if hasattr(args, "lang") and not hasattr(args, "language"):
        args.language = args.lang
    elif hasattr(args, "language") and not hasattr(args, "lang"):
        args.lang = args.language  # Backend expects 'lang' field
    elif not hasattr(args, "lang") and not hasattr(args, "language"):
        # Set default if neither exists
        args.language = "auto"
        args.lang = "auto"

    # Pre-compute derived values for better performance (only if attributes exist)
    if hasattr(args, "min_chunk_size"):
        args.samples_per_chunk = int(16000 * args.min_chunk_size)  # 16kHz sample rate

    if hasattr(args, "vac_chunk_size"):
        args.vac_samples_per_chunk = int(16000 * args.vac_chunk_size)

    if hasattr(args, "buffer_trimming_sec"):
        args.buffer_trimming_samples = int(16000 * args.buffer_trimming_sec)

    # Create feature flags dictionary for easy access
    args.features = {
        "transcription": getattr(args, "transcription", True),
        "diarization": getattr(args, "diarization", False),
        "vad": getattr(args, "vad", True),
        "vac": getattr(args, "vac", False),
        "confidence_validation": getattr(args, "confidence_validation", False),
        "llm_inference": getattr(args, "llm_inference", True),
    }

    return args


def parse_args(argv=None) -> Namespace:
    """Parse and validate command line arguments with optimizations."""
    parser = _get_argument_parser()

    try:
        args = parser.parse_args(argv)
    except SystemExit:
        # If parsing fails, return defaults for testing
        if argv is None:
            # Re-raise the exception for normal usage
            raise
        else:
            # For testing with specific argv, return defaults
            args = Namespace()
            for category in DEFAULT_CONFIG.values():
                for key, value in category.items():
                    setattr(args, key, value)

    # Validate arguments
    _validate_args(args)

    # Optimize and precompute values
    args = _optimize_args(args)

    return args


def parse_args_safe(argv=None) -> Namespace:
    """Parse arguments safely for testing, returning defaults on failure."""
    try:
        return parse_args(argv)
    except (SystemExit, ValueError):
        # Return safe defaults for testing
        args = Namespace()
        for category in DEFAULT_CONFIG.values():
            for key, value in category.items():
                setattr(args, key, value)

        # Add required defaults
        required_defaults = {
            "warmup_file": None,
            "model_cache_dir": None,
            "model_dir": None,
            "ssl_certfile": None,
            "ssl_keyfile": None,
            "log_level": "DEBUG",
        }
        for key, value in required_defaults.items():
            setattr(args, key, value)

        return _optimize_args(args)


class AudioInsight:
    _instance = None
    _initialized = False
    _cached_html = None  # Cache for web interface HTML

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, **kwargs):
        if AudioInsight._initialized:
            return

        # Get unified configuration
        self.config = get_config()

        # Parse arguments and merge with config
        if kwargs:
            # If custom kwargs provided, create args from defaults and override
            args_dict = {}
            for category in DEFAULT_CONFIG.values():
                args_dict.update(category)
            args_dict.update(kwargs)

            # Add required defaults that might not be in DEFAULT_CONFIG
            required_defaults = {
                "warmup_file": None,
                "model_cache_dir": None,
                "model_dir": None,
                "ssl_certfile": None,
                "ssl_keyfile": None,
                "log_level": "DEBUG",
            }
            for key, value in required_defaults.items():
                if key not in args_dict:
                    args_dict[key] = value

            self.args = Namespace(**args_dict)

            # Apply optimizations to the args, but preserve explicit kwargs
            self.args = _optimize_args(self.args)

            # Restore explicit kwargs that might have been overridden by optimization
            for key, value in kwargs.items():
                setattr(self.args, key, value)
        else:
            # Parse arguments only when needed (no custom overrides)
            default_args = vars(parse_args())
            self.args = Namespace(**default_args)

        # Update unified config with args values
        self._sync_args_to_config()

        # Initialize components lazily
        self.asr = None
        self.tokenizer = None
        self.diarization = None
        self._models_loaded = False
        self._diarization_loaded = False

        # Load models only if transcription is enabled
        transcription_enabled = getattr(self.args, "transcription", True)
        if transcription_enabled and not self._models_loaded:
            self._load_asr_models()
            self._models_loaded = True

        # Load diarization only if enabled
        diarization_enabled = getattr(self.args, "diarization", False)
        if diarization_enabled and not self._diarization_loaded:
            self._load_diarization()
            self._diarization_loaded = True

        AudioInsight._initialized = True

    def _sync_args_to_config(self):
        """Sync args values to unified config."""
        # Update config with args values
        if hasattr(self.args, "host"):
            self.config.server.host = self.args.host
        if hasattr(self.args, "port"):
            self.config.server.port = self.args.port
        if hasattr(self.args, "ssl_certfile"):
            self.config.server.ssl_certfile = self.args.ssl_certfile
        if hasattr(self.args, "ssl_keyfile"):
            self.config.server.ssl_keyfile = self.args.ssl_keyfile

        # Model config - handle language/lang field mapping
        if hasattr(self.args, "model"):
            self.config.model.model = self.args.model
        if hasattr(self.args, "backend"):
            self.config.model.backend = self.args.backend
        if hasattr(self.args, "language"):
            self.config.model.language = self.args.language
            # Ensure args.lang exists for backend compatibility
            if not hasattr(self.args, "lang"):
                self.args.lang = self.args.language
        elif hasattr(self.args, "lang"):
            self.config.model.language = self.args.lang
            # Ensure args.language exists for frontend compatibility
            if not hasattr(self.args, "language"):
                self.args.language = self.args.lang
        if hasattr(self.args, "task"):
            self.config.model.task = self.args.task
        if hasattr(self.args, "warmup_file"):
            self.config.model.warmup_file = self.args.warmup_file

        # Processing config
        if hasattr(self.args, "min_chunk_size"):
            self.config.processing.min_chunk_size = self.args.min_chunk_size
        if hasattr(self.args, "buffer_trimming"):
            self.config.processing.buffer_trimming = self.args.buffer_trimming
        if hasattr(self.args, "buffer_trimming_sec"):
            self.config.processing.buffer_trimming_sec = self.args.buffer_trimming_sec
        if hasattr(self.args, "vac_chunk_size"):
            self.config.processing.vac_chunk_size = self.args.vac_chunk_size

        # Feature config
        if hasattr(self.args, "transcription"):
            self.config.features.transcription = self.args.transcription
        if hasattr(self.args, "diarization"):
            self.config.features.diarization = self.args.diarization
        if hasattr(self.args, "vad"):
            self.config.features.vad = self.args.vad
        if hasattr(self.args, "vac"):
            self.config.features.vac = self.args.vac
        if hasattr(self.args, "confidence_validation"):
            self.config.features.confidence_validation = self.args.confidence_validation
        if hasattr(self.args, "llm_inference"):
            self.config.features.llm_inference = self.args.llm_inference

        # LLM config
        if hasattr(self.args, "fast_llm"):
            self.config.llm.fast_llm = self.args.fast_llm
        if hasattr(self.args, "base_llm"):
            self.config.llm.base_llm = self.args.base_llm
        if hasattr(self.args, "llm_summary_interval"):
            self.config.llm.llm_summary_interval = self.args.llm_summary_interval
        if hasattr(self.args, "llm_new_text_trigger"):
            self.config.llm.llm_new_text_trigger = self.args.llm_new_text_trigger
        if hasattr(self.args, "parser_trigger_interval"):
            self.config.llm.parser_trigger_interval = self.args.parser_trigger_interval
        if hasattr(self.args, "parser_output_tokens"):
            self.config.llm.parser_output_tokens = self.args.parser_output_tokens

        # Recompute derived values
        self.config.model_post_init(None)

    def _load_asr_models(self):
        """Lazy loading of ASR models."""
        try:
            self.asr, self.tokenizer = backend_factory(self.args)
            warmup_asr(self.asr, self.args.warmup_file)
        except Exception as e:
            logger.error(f"Failed to load ASR models: {e}")
            raise

    def _load_diarization(self):
        """Lazy loading of diarization models."""
        try:
            from diart import SpeakerDiarizationConfig

            from .diarization.diarization_online import DiartDiarization

            # Very conservative configuration for high confidence diarization only
            config = SpeakerDiarizationConfig(
                step=2.0,  # Even slower processing = higher accuracy (was 1.0)
                latency=2.0,  # Allow more latency for better decisions (was 1.0)
                tau_active=0.8,  # Very high voice activity threshold = only strong voices (was 0.7)
                rho_update=0.05,  # Very low update rate = extremely stable speakers (was 0.1)
                delta_new=3.0,  # Very high threshold = very resistant to new speakers (was 1.8)
                gamma=15,  # Much higher gamma = very stable clustering (was 8)
                beta=30,  # Much higher beta = very stable processing (was 20)
                max_speakers=3,  # Conservative speaker limit (was 4)
            )

            self.diarization = DiartDiarization(config=config)
        except Exception as e:
            logger.error(f"Failed to load diarization models: {e}")
            raise

    @classmethod
    def get_instance(cls, **kwargs):
        """Get singleton instance with optional configuration override."""
        if cls._instance is None or kwargs:
            return cls(**kwargs)
        return cls._instance

    @classmethod
    def reset_instance(cls):
        """Reset singleton instance for testing purposes."""
        cls._instance = None
        cls._initialized = False
        cls._cached_html = None
        reset_config()  # Also reset unified config

    def reconfigure(self, **kwargs):
        """Reconfigure the instance with new parameters."""
        if not kwargs:
            return

        # Update configuration
        for key, value in kwargs.items():
            setattr(self.args, key, value)

        # Sync to unified config
        self._sync_args_to_config()

        # Reload models if necessary
        if "transcription" in kwargs and kwargs["transcription"] and not self._models_loaded:
            self._load_asr_models()
            self._models_loaded = True

        if "diarization" in kwargs and kwargs["diarization"] and not self._diarization_loaded:
            self._load_diarization()
            self._diarization_loaded = True

    def web_interface(self):
        """Get cached web interface HTML."""
        # Use cached HTML if available
        if AudioInsight._cached_html is not None:
            return AudioInsight._cached_html

        try:
            import pkg_resources

            html_path = pkg_resources.resource_filename("audioinsight", "frontend/ui.html")
            with open(html_path, "r", encoding="utf-8") as f:
                html = f.read()

            # Cache the HTML content
            AudioInsight._cached_html = html
            return html
        except Exception as e:
            logger.error(f"Failed to load web interface: {e}")
            # Return a minimal fallback HTML
            return """
            <html><head><title>AudioInsight</title></head>
            <body><h1>AudioInsight - Web Interface Error</h1>
            <p>Failed to load the web interface. Please check the installation.</p></body></html>
            """

    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of current configuration."""
        return {
            "model": {
                "name": self.config.model.model,
                "backend": self.config.model.backend,
                "language": self.config.model.language,
                "task": self.config.model.task,
            },
            "features": {
                "transcription": self.config.features.transcription,
                "diarization": self.config.features.diarization,
                "vad": self.config.features.vad,
                "vac": self.config.features.vac,
                "confidence_validation": self.config.features.confidence_validation,
                "llm_inference": self.config.features.llm_inference,
            },
            "processing": {
                "min_chunk_size": self.config.processing.min_chunk_size,
                "buffer_trimming": self.config.processing.buffer_trimming,
                "buffer_trimming_sec": self.config.processing.buffer_trimming_sec,
            },
            "server": {
                "host": self.config.server.host,
                "port": self.config.server.port,
            },
            "models_loaded": {
                "asr": self._models_loaded,
                "diarization": self._diarization_loaded,
            },
        }
