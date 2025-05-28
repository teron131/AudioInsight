# Whisper Streaming: Technical Description

## Overview

WhisperLiveKit implements a sophisticated real-time streaming speech recognition system that transforms OpenAI's Whisper from a batch-processing model into a low-latency streaming ASR system. The codebase is organized into several key modules that work together to enable real-time processing while maintaining accuracy and coherence.

## Codebase Architecture Overview

The system is organized into the following core modules:

### **`core.py`** - System Initialization and Configuration Management
**Purpose**: Central coordinator and singleton that manages system-wide configuration, model initialization, and component orchestration.

**Key Components**:
- `WhisperLiveKit` class: Singleton pattern implementation that serves as the main entry point and configuration manager
- `parse_args()` function: Comprehensive command-line argument parsing with support for model selection, language settings, VAD configuration, buffer trimming options, and LLM summarization parameters
- Configuration propagation system that distributes settings across all system components
- Model loading orchestration through integration with `whisper_streaming/whisper_online.py`

**Responsibilities**:
- Parse and validate command-line arguments and configuration options
- Initialize ASR models, tokenizers, and diarization systems through backend factories
- Coordinate model warming through `warmup_asr()` function to ensure fast first-chunk processing
- Manage singleton instance lifecycle and ensure proper resource initialization
- Provide web interface HTML content through cached template serving
- Serve as the central configuration authority for all other system components

**Integration Points**: Called by `server.py` during FastAPI lifespan management, provides configuration to `audio_processor.py` and `whisper_streaming/` modules.

### **`audio_processor.py`** - Core Streaming Pipeline and Session Management
**Purpose**: Implements the sophisticated multi-stage asynchronous pipeline that handles real-time audio processing, coordinates between different processors, and manages per-session state.

**Key Components**:
- `AudioProcessor` class: Main orchestrator handling audio processing lifecycle for individual sessions
- FFmpeg integration system: `start_ffmpeg_decoder()`, `ffmpeg_stdout_reader()`, `restart_ffmpeg()` for robust audio format conversion
- Asynchronous task coordination: `create_tasks()`, `transcription_processor()`, `diarization_processor()`, `results_formatter()`
- Memory management: Efficient buffer operations, caching strategies, and resource optimization
- Error recovery: Comprehensive monitoring through `watchdog()` method and automatic recovery procedures

**Key Methods**:
- `process_audio()`: Entry point for incoming audio data with retry logic and FFmpeg health management
- `transcription_processor()`: Manages Whisper inference cycles and hypothesis buffer coordination
- `results_formatter()`: Aggregates transcription and diarization results for client delivery
- `ffmpeg_stdout_reader()`: Converts WebM/Opus streams to PCM format with real-time processing
- `update_transcription()`: Handles token state updates and coordinates with LocalAgreement algorithm

**Responsibilities**:
- Coordinate between FFmpeg audio conversion, Whisper transcription, and speaker diarization
- Manage asynchronous task lifecycle and inter-processor communication through queues
- Implement memory-efficient audio buffer management with caching and optimization strategies
- Provide comprehensive error recovery and health monitoring for all pipeline components
- Handle session-specific state isolation and resource cleanup
- Format and deliver real-time results to WebSocket clients

**Integration Points**: Used by `server.py` for each WebSocket connection, coordinates with `whisper_streaming/online_asr.py` for transcription, integrates with `diarization/` for speaker identification.

### **`server.py`** - FastAPI WebSocket Server and Multi-User Connection Handling
**Purpose**: FastAPI-based web server that handles WebSocket connections, manages multiple concurrent users, and provides both real-time streaming and file upload endpoints.

**Key Components**:
- FastAPI application with lifespan management and CORS configuration
- WebSocket endpoint `/asr` for unified real-time and file processing
- File upload endpoints: `/upload`, `/upload-stream`, `/upload-file` for different processing modes
- Connection lifecycle management with proper cleanup and resource allocation

**Key Methods**:
- `websocket_endpoint()`: Unified WebSocket handler for both live recording and file upload processing
- `handle_websocket_results()`: Manages result streaming from AudioProcessor to WebSocket clients
- `process_file_through_websocket()`: Processes uploaded files through the same pipeline as live recording
- `upload_file()`: HTTP endpoint for file upload with immediate processing and JSON response
- `upload_file_stream()`: Server-sent events endpoint for streaming file processing results

**Responsibilities**:
- Accept and manage WebSocket connections with proper session isolation
- Route incoming audio data (live or file-based) to appropriate AudioProcessor instances
- Implement fair resource allocation across multiple concurrent users
- Provide HTTP endpoints for file upload and processing with various response formats
- Handle connection termination and cleanup to prevent resource leaks
- Serve the web interface HTML and manage static file delivery

**Features**:
- Unified processing pipeline: Both live audio and uploaded files use the same AudioProcessor
- Real-time file simulation: Uploaded files are streamed at their original duration for consistent processing
- Multiple response formats: JSON, Server-Sent Events, and WebSocket for different client needs
- Automatic temporary file cleanup and resource management

**Integration Points**: Initializes `WhisperLiveKit` from `core.py`, creates `AudioProcessor` instances per connection, serves web interface from `web/` directory.

### **`whisper_streaming/`** - Core Streaming Algorithms and Hypothesis Buffer Management
**Purpose**: Contains the core streaming algorithms that enable real-time Whisper processing, including the LocalAgreement policy implementation and backend abstraction layer.

#### **`whisper_streaming/online_asr.py`** - LocalAgreement Algorithm and Buffer Management
**Key Components**:
- `HypothesisBuffer` class: Implements the sophisticated token validation state machine with three-phase token lifecycle
- `OnlineASRProcessor` class: Orchestrates streaming workflow with audio buffer management and context injection
- `VACOnlineASRProcessor` class: Voice Activity Controller wrapper with Silero VAD integration

**Core Algorithms**:
- `HypothesisBuffer.flush()`: Implements LocalAgreement-2 algorithm for stable token commitment
- `HypothesisBuffer.insert()`: Handles temporal alignment and n-gram overlap detection for new tokens
- `OnlineASRProcessor.process_iter()`: Main processing loop coordinating Whisper inference and hypothesis validation
- Context injection through `prompt()` method for maintaining conversational coherence
- Smart buffering with `chunk_completed_sentence()` and `chunk_completed_segment()` for memory management

#### **`whisper_streaming/backends.py`** - Backend Abstraction Layer
**Key Components**:
- `ASRBase` abstract class: Defines unified interface for all Whisper implementations
- `FasterWhisperASR` class: Integration with faster-whisper backend for local GPU processing
- `OpenAIAPIASR` class: Integration with OpenAI API for cloud-based processing

**Responsibilities**:
- Provide consistent interface across different Whisper implementations
- Handle backend-specific optimizations while maintaining compatibility
- Manage model loading, configuration, and inference coordination
- Support VAD integration and translation task configuration

#### **`whisper_streaming/whisper_online.py`** - Factory Functions and Initialization
**Functions**:
- `backend_factory()`: Creates appropriate ASR backend based on configuration
- `online_factory()`: Initializes OnlineASRProcessor or VACOnlineASRProcessor based on VAD settings
- `warmup_asr()`: Downloads and processes warmup audio to ensure fast first-chunk processing
- `create_tokenizer()`: Creates Moses sentence tokenizer for supported languages

### **`timed_objects.py`** - Data Structures and Temporal Representations
**Purpose**: Defines the fundamental data structures that carry temporal and content information throughout the streaming system.

**Key Classes**:
- `TimedText` base class: Common interface for all temporal text objects with start/end timestamps
- `ASRToken` class: Primary data structure representing individual transcribed words with confidence scores and speaker attribution
- `Sentence` and `Transcript` classes: Higher-level aggregations for complete thoughts and utterances
- `SpeakerSegment` class: Represents speaker identification segments with temporal boundaries

**Key Methods**:
- `ASRToken.with_offset()`: Critical method for temporal alignment during buffer management operations
- Dataclass implementations provide efficient memory usage and attribute access
- Optional fields with defaults handle missing information gracefully

**Responsibilities**:
- Provide consistent temporal representation across all system components
- Enable efficient timestamp calculations and offset operations
- Support speaker attribution and confidence score tracking
- Serve as the foundation for all text processing and buffer management operations

### **`diarization/`** - Speaker Identification and Attribution System
**Purpose**: Parallel processing system for real-time speaker identification and attribution using Diart and PyAnnote models.

#### **`diarization/diarization_online.py`** - Real-Time Speaker Diarization
**Key Components**:
- `DiartDiarization` class: Main coordinator for speaker identification using Diart pipeline
- `DiarizationObserver` class: Observer pattern implementation for collecting speaker segments
- `WebSocketAudioSource` class: Custom audio source for streaming audio data to diarization pipeline

**Key Methods**:
- `diarize()`: Processes PCM audio chunks for speaker identification
- `assign_speakers_to_tokens()`: Retrospectively assigns speaker labels to transcribed tokens based on temporal overlap
- `DiarizationObserver.on_next()`: Collects speaker segments and maintains temporal speaker model
- Threading synchronization for concurrent processing without blocking transcription

**Responsibilities**:
- Process audio independently of transcription pipeline to avoid latency impact
- Maintain temporal model of speaker activity across the streaming session
- Assign speaker labels to committed tokens based on timing overlap analysis
- Handle PyAnnote model integration and speaker segment extraction
- Provide thread-safe access to speaker segments with automatic cleanup of old data

### **`llm.py`** - LLM-Based Summarization and Conversation Analysis
**Purpose**: Intelligent conversation monitoring and summarization system that generates summaries based on activity patterns and conversation triggers.

**Key Components**:
- `LLM` class: Main coordinator for transcription monitoring and summary generation
- `SummaryTrigger` dataclass: Configuration for when to trigger summarization (idle time, conversation count, text length)
- `SummaryResponse` class: Structured response from LLM with summary and key points
- LangChain integration with OpenAI/OpenRouter models for structured output generation

**Key Methods**:
- `update_transcription()`: Monitors new transcription text and tracks conversation patterns
- `start_monitoring()`/`stop_monitoring()`: Async monitoring lifecycle management
- `_check_and_summarize()`: Evaluates trigger conditions and initiates summary generation
- `_generate_summary()`: LLM inference with structured output and language-aware prompting

**Features**:
- **Conversation Tracking**: Detects speaker turns and counts conversations for trigger evaluation
- **Idle Time Monitoring**: Tracks periods of inactivity to trigger summarization after conversation ends
- **Language-Aware Responses**: Automatically responds in the same language and script as the input transcription
- **Callback System**: Allows other components to receive summary notifications
- **Statistics Tracking**: Monitors summary generation patterns and performance metrics

**Trigger Conditions**:
- Idle time after speech activity (configurable seconds)
- Number of conversation turns (speaker changes)
- Maximum text length thresholds
- Manual force summarization capability

**Integration Points**: Called by `audio_processor.py` to receive transcription updates, provides callbacks for summary delivery to clients.

### **`web/`** - Frontend HTML/JavaScript Interface
**Purpose**: Complete web-based user interface for real-time transcription with support for both live recording and file upload.

#### **`web/live_transcription.html`** - Browser-Based Transcription Interface
**Key Features**:
- **MediaRecorder API Integration**: Captures browser microphone audio in WebM/Opus format
- **WebSocket Communication**: Real-time bidirectional communication for audio streaming and result display
- **File Upload Support**: Drag-and-drop and file selection for audio file processing
- **Real-Time Display**: Live transcription updates with speaker identification and confidence indicators
- **Responsive Design**: Modern UI with proper error handling and connection status indicators

**Components**:
- Audio recording controls with start/stop/pause functionality
- File upload interface with progress indicators and format validation
- Real-time transcription display with speaker attribution and timestamps
- Connection status monitoring and error handling
- Settings panel for configuration options

**Technical Implementation**:
- WebSocket handling with automatic reconnection and error recovery
- Efficient audio chunking and streaming to minimize latency
- Browser compatibility handling for different MediaRecorder implementations
- CSS styling for professional appearance and user experience

## Theoretical Foundation: The Streaming Challenge

### The Fundamental Problem

Traditional ASR models like Whisper are designed as encoder-decoder transformers that process complete audio sequences and generate full transcriptions. This architecture presents several challenges for real-time streaming:

1. **Sequence Dependency**: The model expects complete context to generate accurate transcriptions
2. **Output Stability**: Partial inputs can produce fluctuating and inconsistent outputs
3. **Context Management**: Maintaining conversational context across streaming chunks
4. **Latency vs. Accuracy Trade-off**: Balancing immediate response with transcription quality

### LocalAgreement Policy: The Core Innovation

The system implements the **LocalAgreement-2** streaming policy in `whisper_streaming/online_asr.py`, which solves the output stability problem through a sophisticated hypothesis validation mechanism. The core insight is that when consecutive processing iterations agree on the same tokens, those tokens are likely stable and can be safely committed as final output.

#### **Mathematical Foundation**

Given a streaming policy P that processes audio chunks c₁, c₂, ..., cₙ, the LocalAgreement-2 policy works as follows:

For each time step t, the system maintains:
- H^(t-1): Previous hypothesis from processing chunks c₁...cₜ₋₁  
- H^(t): Current hypothesis from processing chunks c₁...cₜ
- C^(t): Committed tokens up to time t

The policy commits the longest common prefix between H^(t-1) and H^(t):
```
C^(t) = C^(t-1) ∪ LongestCommonPrefix(H^(t-1), H^(t))
```

#### **LocalAgreement-2 Algorithm Implementation**

The core algorithm is implemented in `whisper_streaming/online_asr.py` within the `HypothesisBuffer.flush()` method:

```python
def flush(self) -> List[ASRToken]:
    """Find longest common prefix between consecutive hypotheses"""
    committed = []
    while self.new and self.buffer:
        if current_new.text == self.buffer[0].text:
            # Agreement found - commit this token
            committed.append(current_new)
            self.last_committed_time = current_new.end
            self.buffer.pop(0)
            self.new.pop(0)
        else:
            # Disagreement - stop committing
            break
    
    # Update buffers
    self.buffer = self.new
    self.new = []
    self.committed_in_buffer.extend(committed)
    return committed
```

## Core Architecture: Multi-Layer Streaming System

### 1. Data Structures and Token Management (`timed_objects.py`)

The foundation of the streaming system is built on carefully designed data structures that carry temporal and content information:

**ASRToken Class**: The primary data structure representing individual transcribed words with start/end timestamps, confidence scores, and speaker attribution. The `with_offset()` method is crucial for temporal alignment during buffer management operations.

**TimedText Base Class**: Provides the common interface for all temporal text objects, ensuring consistent timestamp handling across the system.

**Transcript and Sentence Classes**: Higher-level aggregations that represent complete thoughts or utterances, used in the buffer trimming and context management systems.

### 2. System Initialization and Configuration (`core.py`)

The `WhisperLiveKit` class in `core.py` serves as the central coordinator and singleton that manages:

**Model Loading and Backend Selection**: Coordinates with `whisper_streaming/whisper_online.py` to initialize the appropriate Whisper backend (faster-whisper, OpenAI API, etc.) through the `backend_factory()` function.

**Configuration Management**: Processes command-line arguments and configuration options, propagating settings throughout the system components including VAD settings, buffer trimming preferences, and model-specific parameters.

**Component Initialization**: Orchestrates the initialization of ASR models, tokenizers, and diarization systems, ensuring proper resource allocation and model warming through the `warmup_asr()` function.

### 3. Hypothesis Buffer Management System (`whisper_streaming/online_asr.py`)

The **HypothesisBuffer** class is the heart of the streaming mechanism, implementing a sophisticated state machine that manages token validation through three distinct phases:

#### **State Transitions and Token Lifecycle**

The buffer operates with three distinct token states that represent different stages of validation confidence:

**New Tokens Phase**: Fresh output from the latest Whisper inference represents the most recent hypothesis. The `insert()` method in `HypothesisBuffer` handles temporal alignment and n-gram overlap detection to properly correlate new tokens with existing state.

**Buffer Tokens Phase**: Previous iteration's hypothesis tokens that await validation against new incoming tokens. The system maintains these tokens as the baseline for agreement detection, representing the current "working hypothesis."

**Committed Tokens Phase**: Validated tokens that have achieved agreement between consecutive hypotheses. The `flush()` method implements the core LocalAgreement logic to promote buffer tokens to committed status.

#### **OnlineASRProcessor: Core Streaming Engine**

The `OnlineASRProcessor` class orchestrates the entire streaming workflow:

**Audio Buffer Management**: The `insert_audio_chunk()` method accumulates incoming audio while the `process_iter()` method triggers Whisper inference cycles and hypothesis updates.

**Context Injection**: The `prompt()` method extracts the last 200 characters of committed text to maintain conversational context across buffer management operations.

**Buffer Trimming Logic**: Methods like `chunk_completed_sentence()` and `chunk_completed_segment()` implement intelligent buffer trimming to prevent memory overflow while preserving semantic coherence.

### 4. Audio Processing Pipeline (`audio_processor.py`)

The `AudioProcessor` class implements the sophisticated multi-stage asynchronous pipeline that handles real-time audio processing:

#### **Asynchronous Task Management**

**FFmpeg Integration**: The `start_ffmpeg_decoder()` and `ffmpeg_stdout_reader()` methods manage the external FFmpeg process that converts WebM/Opus to PCM format. The system implements robust process monitoring and automatic restart capabilities in `restart_ffmpeg()`.

**Processing Pipeline Coordination**: The `create_tasks()` method spawns specialized processors:
- `transcription_processor()`: Manages Whisper inference and hypothesis buffer updates
- `diarization_processor()`: Handles speaker identification when enabled
- `results_formatter()`: Aggregates and formats output for client delivery

**Queue-Based Communication**: Uses asyncio queues for inter-processor coordination, with sentinel handling for graceful shutdown and backpressure management to prevent queue overflow.

#### **Memory Management and Optimization**

**Efficient Buffer Operations**: The `convert_pcm_to_float()` method implements optimized audio format conversion, while buffer management uses pre-allocated arrays and in-place operations to minimize memory pressure.

**Caching Strategies**: Global caches for expensive operations like time formatting (`_cached_timedeltas`) and text conversion (`_s2hk_converter`) reduce computational overhead during real-time processing.

**Error Recovery**: Comprehensive error handling with the `watchdog()` method monitoring system health and implementing automatic recovery procedures.

### 5. WebSocket Server and Connection Management (`server.py`)

The FastAPI-based server handles multiple concurrent users and WebSocket communication:

**Connection Lifecycle**: Each WebSocket connection spawns an isolated `AudioProcessor` instance, ensuring complete session isolation and preventing cross-user interference.

**Multi-User Session Management**: The server implements per-session resource allocation and cleanup, with proper handling of connection termination and resource deallocation.

**API Endpoints**: Provides both the WebSocket endpoint for real-time processing and HTTP endpoints for serving the web interface and system status.

### 6. Voice Activity Controller Integration (`whisper_streaming/online_asr.py`)

The `VACOnlineASRProcessor` class wraps the standard processor with intelligent voice activity detection:

**Silero VAD Integration**: Uses the Silero VAD model for high-resolution speech detection (40ms chunks), enabling responsive processing control and immediate finalization during silence periods.

**Adaptive Processing**: Implements state-driven processing that only triggers Whisper inference during detected speech activity, significantly reducing computational load and improving perceived latency.

**Utterance Boundary Detection**: Uses silence detection to identify natural speech boundaries, allowing immediate output finalization without waiting for standard agreement cycles.

### 7. Speaker Diarization System (`diarization/`)

The diarization subsystem operates in parallel to assign speaker labels:

**Independent Processing**: Runs asynchronously to avoid blocking primary transcription, with its own processing queue and temporal model of speaker activity.

**Retrospective Attribution**: Assigns speaker labels to committed tokens after transcription is complete, ensuring that speaker identification doesn't impact transcription latency.

**Integration with Results**: The `results_formatter()` method in `audio_processor.py` coordinates transcription and diarization results for unified output.

### 8. Backend Abstraction Layer (`whisper_streaming/backends.py`)

The system supports multiple Whisper implementations through a unified interface:

**ASRBase Abstract Class**: Defines the common interface that all backend implementations must support, ensuring streaming logic remains implementation-agnostic.

**FasterWhisperASR and OpenAIAPIASR**: Concrete implementations that handle the specifics of different Whisper backends while maintaining interface compatibility.

**Configuration Propagation**: The `backend_factory()` function in `whisper_streaming/whisper_online.py` applies common configurations (VAD, translation tasks) consistently across backends.

## Memory Management and Performance Optimization

### 1. Efficient Buffer Operations (`audio_processor.py`)

The system implements comprehensive memory optimization strategies:

**Pre-allocated Buffers**: Audio buffers are dimensioned based on expected processing patterns, with pre-computed values like `samples_per_sec` and `bytes_per_sec` reducing runtime calculations.

**Circular Buffer Patterns**: The `pcm_buffer` management uses efficient slicing operations to maintain rolling windows without excessive memory allocation.

**In-place Operations**: The `convert_pcm_to_float()` method uses numpy operations that minimize temporary array creation and memory copying.

### 2. Concurrency Control (`audio_processor.py`)

**Async Lock Coordination**: The system uses `asyncio.Lock()` for state consistency protection, with careful design to minimize blocking and prevent deadlock scenarios.

**Task Isolation**: Each async task (`transcription_task`, `diarization_task`, `ffmpeg_reader_task`) operates independently with proper exception handling and cleanup procedures.

## Error Recovery and Fault Tolerance

### 1. Process Health Monitoring (`audio_processor.py`)

**Watchdog System**: The `watchdog()` method continuously monitors task health, FFmpeg responsiveness, and resource utilization, implementing automatic recovery procedures when issues are detected.

**FFmpeg Process Management**: Sophisticated restart procedures in `restart_ffmpeg()` preserve as much processing state as possible while recovering from process failures.

**Graceful Degradation**: The system can continue core transcription functionality even when non-critical components (like diarization) fail.

### 2. State Recovery Mechanisms

**Incremental State Preservation**: Critical system state is maintained in formats that support recovery from failures, with the prompt injection mechanism serving dual purposes for both context preservation and state reconstruction.

**Buffer State Reconstruction**: The system can rebuild processing context from committed tokens and available audio history after component restarts.

## Multi-User Session Management (`server.py` + `audio_processor.py`)

### 1. Session Isolation Architecture

**Independent Processor Instantiation**: Each WebSocket connection in `server.py` creates an isolated `AudioProcessor` instance with dedicated resources, preventing cross-session interference.

**Resource Allocation**: Fair scheduling algorithms prevent any single session from monopolizing system resources, with adaptive allocation based on concurrent session count.

**Lifecycle Management**: Proper connection handling with coordinated shutdown sequences ensures clean resource deallocation and prevents resource leaks.

### 2. Load Balancing and Scalability

**Dynamic Resource Management**: The system adapts resource allocation based on system load, with the ability to implement quality trade-offs when necessary to maintain service availability.

**Memory Pressure Response**: Intelligent backpressure mechanisms prevent memory exhaustion while maintaining service quality across multiple concurrent users.

## Integration Points and Data Flow

### 1. Audio Flow Architecture

**Input Processing**: WebSocket binary data → `server.py` → `audio_processor.py` → FFmpeg conversion → PCM buffer → `whisper_streaming/online_asr.py`

**Transcription Pipeline**: Audio buffer → Whisper inference → `timed_objects.py` ASRToken creation → `HypothesisBuffer` validation → committed tokens

**Output Generation**: Committed tokens → `results_formatter()` → JSON response → WebSocket → client

### 2. Cross-Module Coordination

**Configuration Flow**: `core.py` → component initialization → runtime parameter propagation

**State Management**: `audio_processor.py` coordinates between streaming engine, diarization, and output formatting

**Error Propagation**: Failures are contained within modules but reported through the monitoring system for coordinated recovery

## Conclusion

WhisperLiveKit's codebase represents a sophisticated engineering solution where each module has clearly defined responsibilities that work together to solve the fundamental challenge of real-time speech recognition. The modular architecture enables independent development and optimization of components while maintaining system coherence through well-defined interfaces and coordination mechanisms.

The LocalAgreement algorithm in `whisper_streaming/online_asr.py` provides the theoretical foundation, while `audio_processor.py` orchestrates the complex real-time pipeline, `server.py` handles multi-user coordination, and supporting modules provide specialized functionality. This architecture enables future enhancements and optimizations without disrupting core functionality, making it a robust platform for real-time speech recognition applications. 