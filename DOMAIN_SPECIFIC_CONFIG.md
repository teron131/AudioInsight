# 🏗️ Domain-Specific Configuration System

AudioInsight now uses a sophisticated domain-specific configuration system that clearly separates **runtime configurable** settings from **startup-only** settings.

## 📋 Table of Contents

1. [Overview](#overview)
2. [Configuration Domains](#configuration-domains)
3. [Runtime vs Startup Settings](#runtime-vs-startup-settings)
4. [Configuration Files](#configuration-files)
5. [API Integration](#api-integration)
6. [Usage Examples](#usage-examples)

---

## 🔍 Overview

The new configuration system organizes settings into **domains** with clear separation between:

- **Runtime Configurable**: Settings that can be changed via the settings page without restarting the server
- **Startup Only**: Settings that require a server restart and are configured via CLI arguments

### 🎯 Key Benefits

- **Single Source of Truth**: Central coordination through `audioinsight/config.py`
- **Domain Separation**: LLM, server, and processing settings in dedicated modules
- **Runtime Safety**: Clear distinction between what can and cannot be changed at runtime
- **Type Safety**: Full Pydantic validation with proper field constraints
- **Backward Compatibility**: Existing APIs continue to work unchanged

---

## 🏗️ Configuration Domains

### 1. **LLM Domain** (`audioinsight/llm/config.py`)

**Runtime Configurable:**
- Model selection (`fast_llm`, `base_llm`)
- Processing triggers (`llm_summary_interval`, `parser_trigger_interval`)
- Output limits (`parser_output_tokens`, `summarizer_output_tokens`)
- Feature flags (`llm_inference`)
- Generation settings (`temperature`)

**Startup Only:**
- Worker pool settings (`max_workers`, `queue_size`)
- Timeout settings (`request_timeout`)
- API credentials (`openai_api_key`, `openrouter_api_key`)

### 2. **Server Domain** (`audioinsight/server/config.py`)

**Runtime Configurable:**
- CORS settings (`cors_origins`, `cors_credentials`)
- Request limits (`max_request_size`, `request_timeout`)
- Feature toggles (`enable_websockets`, `enable_file_upload`)

**Startup Only:**
- Network binding (`host`, `port`)
- SSL configuration (`ssl_certfile`, `ssl_keyfile`)
- Worker processes (`workers`)

### 3. **Processing Domain** (Central `audioinsight/config.py`)

**Runtime Configurable:**
- Audio processing (`model`, `language`, `vad_enabled`)
- Quality settings (`beam_size`, `best_of`)
- Feature flags (`enable_diarization`, `enable_vac`)

**Startup Only:**
- Hardware settings (`device`, `compute_type`)
- Model loading (`model_dir`)

---

## 📁 Configuration Files

### **Central Coordinator** (`audioinsight/config.py`)
```python
def get_runtime_configurable_fields() -> dict:
    """Get all runtime configurable settings grouped by domain."""
    
def apply_runtime_updates(updates: dict) -> dict:
    """Apply runtime updates across all domains."""
```

### **LLM Configuration** (`audioinsight/llm/config.py`)
```python
class LLMConfig(BaseModel):
    """Simple LLM config for UniversalLLM client."""
    
class LLMRuntimeConfig(BaseModel):
    """Runtime configurable LLM settings."""
    
class LLMStartupConfig(BaseModel):
    """Startup-only LLM settings."""

# Helper functions
def get_llm_config() -> DomainLLMConfig
def get_parser_config() -> ParserConfig
def get_summarizer_config() -> SummarizerConfig
```

### **Type Definitions** (`audioinsight/llm/types.py`)
```python
class ParsedTranscript(BaseModel):
    """Structured transcript response."""
    
class SummarizerResponse(BaseModel):
    """LLM inference response."""
    
class SummarizerStats:
    """Statistics tracking for LLM operations."""
```

**Note**: Configuration logic has been moved from `types.py` to `config.py` for better separation of concerns.

### **Server Configuration** (`audioinsight/server/config.py`)
```python
class ServerRuntimeConfig(BaseModel):
    """Runtime configurable server settings."""
    
class ServerStartupConfig(BaseModel):
    """Startup-only server settings."""
```

---

## 🔌 API Integration

### **Settings Page Integration**
```python
# Get all runtime configurable fields
@app.get("/api/settings/fields")
async def get_configurable_fields():
    from audioinsight.config import get_runtime_configurable_fields
    return get_runtime_configurable_fields()

# Update runtime settings
@app.post("/api/settings/update")
async def update_settings(updates: dict):
    from audioinsight.config import apply_runtime_updates
    return apply_runtime_updates(updates)
```

### **CLI Arguments**
```bash
# Startup-only settings (require restart)
python -m audioinsight --host 0.0.0.0 --port 8080 --workers 4

# Runtime settings (can be changed via API)
# These are handled through the settings page or API calls
```

---

## 💡 Usage Examples

### **Getting Domain-Specific Configuration**
```python
from audioinsight.llm.config import get_llm_config, get_parser_config
from audioinsight.server.config import get_server_config

# Get LLM configuration
llm_config = get_llm_config()
print(f"Fast LLM: {llm_config.runtime.fast_llm}")
print(f"Max workers: {llm_config.startup.max_workers}")

# Get parser-specific configuration
parser_config = get_parser_config()
print(f"Parser model: {parser_config.model_id}")
print(f"Max tokens: {parser_config.max_output_tokens}")
```

### **Runtime Configuration Updates**
```python
from audioinsight.config import apply_runtime_updates

# Update LLM settings
updates = {
    "fast_llm": "openai/gpt-4.1-turbo",
    "llm_summary_interval": 2.0,
    "parser_output_tokens": 50000
}

result = apply_runtime_updates(updates)
print(f"Updated: {result}")
```

### **Checking Runtime vs Startup Settings**
```python
from audioinsight.config import get_runtime_configurable_fields, get_startup_only_fields

# Get all runtime configurable settings
runtime_fields = get_runtime_configurable_fields()
print("Runtime configurable:")
for domain, fields in runtime_fields.items():
    print(f"  {domain}: {list(fields.keys())}")

# Get all startup-only settings
startup_fields = get_startup_only_fields()
print("Startup only:")
for domain, fields in startup_fields.items():
    print(f"  {domain}: {list(fields.keys())}")
```

---

## 🔧 Migration Guide

### **From Old Configuration**
```python
# OLD: Mixed configuration in types.py
from audioinsight.llm.types import LLMConfig, LLMTrigger

# NEW: Domain-specific configuration
from audioinsight.llm.config import get_llm_config, get_llm_trigger
```

### **For Component Initialization**
```python
# OLD: Manual configuration creation
config = LLMConfig(model_id="gpt-4", temperature=0.1)

# NEW: Use domain-specific helpers
from audioinsight.llm.config import get_parser_config
config = get_parser_config()  # Automatically configured from unified config
```

---

## ✅ Benefits Achieved

1. **🎯 Clear Separation**: Runtime vs startup settings are clearly distinguished
2. **🏗️ Domain Organization**: Each domain manages its own configuration
3. **🔒 Type Safety**: Full Pydantic validation with proper constraints
4. **🔄 Backward Compatibility**: Existing code continues to work
5. **📝 Self-Documenting**: Field descriptions and help text built-in
6. **🚀 Performance**: Optimized configuration loading and caching
7. **🧹 Clean Architecture**: Configuration logic separated from type definitions

The system now provides a robust, scalable foundation for managing configuration across the entire AudioInsight application while maintaining clear boundaries between what can be changed at runtime versus what requires a restart. 