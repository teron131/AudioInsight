# Whisper Streaming: Technical Description

## Overview

AudioInsight implements a sophisticated real-time streaming speech recognition system that transforms OpenAI's Whisper from a batch-processing model into a low-latency streaming ASR system. The core innovation lies in the LocalAgreement-2 algorithm and modular processing pipeline that enable stable real-time transcription.

## Core Technical Architecture

### **Streaming Pipeline Components**

#### **`audioinsight/main.py`** - System Coordination
- `AudioInsight` class: Singleton pattern for model and configuration management
- Model loading orchestration and backend selection
- Configuration propagation across streaming components

#### **`audioinsight/processors.py`** - Real-Time Processing Pipeline
- `AudioProcessor`: Central coordinator managing shared state and inter-processor communication
- `FFmpegProcessor`: Audio format conversion and PCM data processing  
- `TranscriptionProcessor`: Whisper inference cycles and hypothesis buffer coordination
- `DiarizationProcessor`: Independent speaker identification processing
- `Formatter`: Result aggregation and output generation

#### **`audioinsight/whisper_streaming/online_asr.py`** - Core Streaming Algorithms
- `HypothesisBuffer`: Token validation state machine implementing LocalAgreement-2
- `OnlineASRProcessor`: Streaming workflow orchestration with audio buffer management
- `VACOnlineASRProcessor`: Voice Activity Controller wrapper with VAD integration

#### **`audioinsight/timed_objects.py`** - Temporal Data Structures
- `ASRToken`: Primary data structure representing transcribed words with timestamps
- `TimedText`: Base class for temporal text objects
- `SpeakerSegment`: Speaker identification segments with temporal boundaries

## Theoretical Foundation: The Streaming Challenge

### The Fundamental Problem

Traditional ASR models process complete audio sequences, presenting challenges for real-time streaming:

1. **Sequence Dependency**: Models expect complete context for accurate transcriptions
2. **Output Stability**: Partial inputs produce fluctuating and inconsistent outputs  
3. **Context Management**: Maintaining conversational context across streaming chunks
4. **Latency vs. Accuracy Trade-off**: Balancing immediate response with transcription quality

### LocalAgreement Policy: Core Innovation

The **LocalAgreement-2** streaming policy solves output stability through hypothesis validation. When consecutive processing iterations agree on the same tokens, those tokens are committed as stable output.

#### **Mathematical Foundation**

Given streaming policy P processing audio chunks c₁, c₂, ..., cₙ:

For each time step t, the system maintains:
- H^(t-1): Previous hypothesis from chunks c₁...cₜ₋₁  
- H^(t): Current hypothesis from chunks c₁...cₜ
- C^(t): Committed tokens up to time t

The policy commits the longest common prefix:
```
C^(t) = C^(t-1) ∪ LongestCommonPrefix(H^(t-1), H^(t))
```

#### **LocalAgreement-2 Algorithm Implementation**

Core algorithm in `HypothesisBuffer.flush()`:

```python
def flush(self) -> List[ASRToken]:
    """Returns committed tokens based on longest common prefix between hypotheses."""
    committed: List[ASRToken] = []
    commit_count = 0
    min_length = min(len(self.new), len(self.buffer))
    
    # Find longest matching prefix
    while commit_count < min_length:
        if self.new[commit_count].text == self.buffer[commit_count].text:
            committed.append(self.new[commit_count])
            commit_count += 1
        else:
            break

    # Efficiently update buffers
    if commit_count > 0:
        self.new = self.new[commit_count:]
        self.buffer = self.buffer[commit_count:] if len(self.buffer) >= commit_count else []

    self.buffer = self.new
    self.new = []
    return committed
```

**Performance Optimizations**:
- Batch processing to avoid repeated operations
- Efficient list slicing instead of O(n) pop operations
- Memory-efficient buffer management

## Hypothesis Buffer Management System

### Token Lifecycle States

The buffer operates with three validation phases:

1. **New Tokens**: Fresh output from latest Whisper inference
2. **Buffer Tokens**: Previous hypothesis awaiting validation  
3. **Committed Tokens**: Validated tokens achieving cross-hypothesis agreement

### Streaming Engine Components

**Audio Buffer Management**: 
- `insert_audio_chunk()`: Accumulates incoming audio
- `process_iter()`: Triggers Whisper inference cycles

**Context Injection**: 
- `prompt()`: Extracts last 200 characters for conversational context

**Buffer Trimming**: 
- `chunk_completed_sentence()`: Intelligent memory management preserving semantic coherence

## Memory Management and Performance

### Buffer Optimization Strategies

**Pre-allocated Buffers**: Audio buffers dimensioned for expected processing patterns

**Zero-Copy Operations**: Optimized data flow minimizing memory allocation

**Circular Buffer Management**: Efficient slicing operations for rolling windows

**Dynamic Sizing**: Exponential growth strategy minimizing reallocation overhead

### Processing Pipeline Efficiency

**Parallel Processing**: Transcription and diarization operate concurrently

**Queue Management**: Efficient inter-processor communication with backpressure handling

**Resource Pooling**: Shared timing and configuration coordination

## Voice Activity Detection Integration

**Silero VAD Integration**: High-resolution speech detection (40ms chunks)

**Adaptive Processing**: State-driven processing triggering Whisper only during speech

**Utterance Boundary Detection**: Silence-based natural speech boundary identification

## Speaker Diarization System

### Real-Time Speaker Identification

**`DiartDiarization`**: Speaker identification using Diart pipeline with PyAnnote models

**`DiarizationObserver`**: Observer pattern for collecting speaker segments with memory management

**Retrospective Attribution**: Speaker labels assigned to committed tokens based on temporal overlap

**Consistent Speaker Mapping**: First detected speaker gets ID 0 for stable identification

## Error Recovery and Fault Tolerance

**Process Health Monitoring**: Continuous task health monitoring with automatic recovery

**Graceful Degradation**: Core transcription continues even when auxiliary components fail

**State Recovery**: Critical system state maintained for recovery from failures

**Coordinated Recovery**: Central coordinator manages cross-processor recovery procedures

## Conclusion

AudioInsight's streaming architecture solves the fundamental challenge of real-time speech recognition through:

1. **LocalAgreement-2 Algorithm**: Ensures output stability through hypothesis validation
2. **Efficient Buffer Management**: Optimized memory operations minimize latency  
3. **Parallel Processing**: Independent transcription and diarization pipelines
4. **Context Preservation**: Intelligent buffer trimming maintains conversational coherence
5. **Voice Activity Detection**: Adaptive processing based on speech detection
6. **Fault Tolerance**: Robust error recovery and graceful degradation

This architecture enables production-grade real-time speech recognition with maintained accuracy and performance characteristics suitable for multi-user deployment scenarios.