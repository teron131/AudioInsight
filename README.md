# AudioInsight

> **Built on top of [WhisperLiveKit](https://github.com/QuentinFuxa/WhisperLiveKit) which built on top of [whisper_streaming](https://github.com/ufal/whisper_streaming).**

> **Real-time, Fully Local Speech-to-Text with Speaker Diarization and LLM-powered Transcript Analysis**

Transform speech into text instantly with AudioInsight - a production-ready streaming ASR system that runs entirely on your machine. Built on OpenAI's Whisper with advanced streaming algorithms for low-latency, accurate transcription, enhanced with intelligent LLM-powered conversation analysis and summarization.

---

## ⚡ Quick Start

Get up and running in seconds:

```bash
# Install AudioInsight
pip install audioinsight

# Start transcribing immediately  
audioinsight-server --model large-v3-turbo

# Open http://localhost:8001 and start speaking! 🎤
```

## 🎯 Why AudioInsight?

AudioInsight solves the fundamental challenge of real-time speech recognition by transforming OpenAI's batch-processing Whisper into a streaming system with **LocalAgreement** algorithms that ensure stable, coherent output. Enhanced with intelligent LLM-powered analysis for conversation understanding and summarization.

### ✨ Core Advantages

🔒 **100% Local Processing** - No data leaves your machine  
🎙️ **Real-time Streaming** - See words appear as you speak  
👥 **Multi-Speaker Support** - Identify different speakers automatically  
🧠 **LLM-Powered Analysis** - Intelligent conversation summarization and text parsing  
🌐 **Multi-User Ready** - Handle multiple sessions simultaneously  
⚡ **Ultra-Low Latency** - Optimized streaming algorithms  
🛠️ **Production Ready** - Built for real-world applications  
🎯 **Modular Architecture** - Specialized components for enhanced maintainability  
📁 **Comprehensive File Support** - Multiple processing modes for audio files  
🔄 **Unified Processing Pipeline** - Same engine for live and file processing  
📊 **Multiple Response Formats** - JSON, WebSocket, and Server-Sent Events  
🔍 **Intelligent Text Processing** - LLM-based transcript correction and enhancement

### 🏗️ Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Browser UI    │───▶│  FastAPI Server  │───▶│  Core Engine    │───▶│   LLM Analysis  │
│  (WebSocket)    │    │  (Multi-Module)  │    │  (ASR + Diart)  │    │  (Summarization)│
└─────────────────┘    └──────────────────┘    └─────────────────┘    └─────────────────┘
        │                        │                        │                        │
        ▼                        ▼                        ▼                        ▼
  Audio Capture           Modular Architecture      Real-time Processing    Conversation Analysis
  MediaRecorder          ┌──────────────────┐       LocalAgreement Policy   Text Parsing & Correction
                         │  server/config   │       Specialized Processors  Intelligent Summarization
                         │  server/handlers │       
                         │  server/websocket│       
                         │  server/utils    │       
                         └──────────────────┘       
```

**Four-Layer Modular Design:**
- **Frontend**: HTML5/JavaScript interface with WebSocket streaming and file upload
- **Server**: Modular FastAPI architecture with specialized components:
  - **Configuration Management** (`server/config.py`): CORS, audio validation, processing settings
  - **File Handlers** (`server/file_handlers.py`): Upload processing, SSE streaming, unified pipeline
  - **WebSocket Management** (`server/websocket_handlers.py`): Connection lifecycle, real-time processing
  - **Utilities** (`server/utils.py`): Audio processing, FFmpeg integration, streaming simulation
- **Core**: Advanced streaming algorithms with modular processors and speaker diarization
- **LLM Layer**: Intelligent conversation analysis, text parsing, and summarization using configurable LLM models

---

## 🚀 Installation & Setup

### Standard Installation

```bash
pip install audioinsight
```

### Development Installation

```bash
git clone https://github.com/teron131/AudioInsight
cd AudioInsight
pip install -e .
```

### System Requirements

**Required:**
```bash
# FFmpeg (audio processing)
sudo apt install ffmpeg        # Ubuntu/Debian
brew install ffmpeg           # macOS
# Windows: Download from https://ffmpeg.org
```

**Optional Enhancements:**
```bash
# Voice Activity Detection (recommended)
pip install torch

# Advanced sentence tokenization
pip install mosestokenizer

# Speaker diarization
pip install diart

# LLM inference capabilities (for conversation analysis)
pip install openai langchain langchain-google-genai

# Alternative Whisper backends
pip install audioinsight[whisper]   # Original Whisper
pip install audioinsight[openai]    # OpenAI API
```

### Speaker Diarization Setup

For multi-speaker identification, configure pyannote.audio:

1. Accept terms for required models:
   - [pyannote/segmentation](https://huggingface.co/pyannote/segmentation)
   - [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0)
   - [pyannote/embedding](https://huggingface.co/pyannote/embedding)

2. Login to Hugging Face:
   ```bash
   pip install huggingface_hub
   huggingface-cli login
   ```

### LLM Configuration

For intelligent conversation analysis, set up API keys:

```bash
# OpenAI API (for GPT models)
export OPENAI_API_KEY="your-openai-api-key"

# OpenRouter API (for access to multiple models)
export OPENROUTER_API_KEY="your-openrouter-api-key"

# Google AI API (for Gemini models)
export GOOGLE_API_KEY="your-google-api-key"
```

---

## 💡 Usage Guide

### Command Line Interface

**Basic Usage:**
```bash
# English transcription with default model
audioinsight-server

# Advanced configuration with LLM analysis
audioinsight-server \
  --model large-v3-turbo \
  --language auto \
  --diarization \
  --llm-inference \
  --fast-llm "openai/gpt-4.1-nano" \
  --base-llm "openai/gpt-4.1-mini" \
  --host 0.0.0.0 \
  --port 8001
```

**LLM-Enhanced Processing:**
```bash
# Enable conversation summarization with custom triggers
audioinsight-server \
  --model large-v3-turbo \
  --diarization \
  --llm-inference \
  --llm-trigger-time 3.0 \
  --llm-conversation-trigger 3 \
  --fast-llm "openai/gpt-4.1-nano" \
  --base-llm "anthropic/claude-3-haiku"
```

**SSL/HTTPS Support:**
```bash
audioinsight-server \
  --ssl-certfile cert.pem \
  --ssl-keyfile key.pem
# Access via https://localhost:8001
```

### Python Integration

**Basic Server with LLM Analysis:**
```python
from audioinsight import AudioInsight
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import asyncio

# Initialize core components with LLM analysis
app = FastAPI()
kit = AudioInsight(
    model="large-v3-turbo", 
    diarization=True,
    llm_inference=True,
    fast_llm="openai/gpt-4.1-nano",
    base_llm="openai/gpt-4.1-mini"
)

@app.get("/")
async def get_interface():
    return HTMLResponse(kit.web_interface())

@app.websocket("/asr")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # Create session-specific processor with LLM analysis
    from audioinsight.audio_processor import AudioProcessor
    processor = AudioProcessor()
    
    # Start processing pipeline
    results_generator = await processor.create_tasks()
    
    # Handle bidirectional communication
    async def send_results():
        async for result in results_generator:
            # Results now include LLM summaries and analysis
            if result.get('type') == 'summary':
                print(f"LLM Summary: {result['content']}")
            await websocket.send_json(result)
    
    send_task = asyncio.create_task(send_results())
    
    try:
        while True:
            # Receive audio data
            audio_data = await websocket.receive_bytes()
            await processor.process_audio(audio_data)
    except Exception as e:
        print(f"Connection closed: {e}")
    finally:
        send_task.cancel()
        await processor.cleanup()
```

**LLM Integration Example:**
```python
from audioinsight.llm import LLMSummarizer, Parser, UniversalLLM
from audioinsight.llm.types import LLMConfig, LLMTrigger
import asyncio

async def llm_transcript_analysis():
    """Example of using LLM components for transcript analysis."""
    
    # Configure LLM for summarization
    summarizer_config = LLMConfig(
        model="openai/gpt-4.1-mini",
        provider="openai",
        temperature=0.1
    )
    
    # Set up conversation trigger
    trigger = LLMTrigger(
        idle_time=5.0,
        conversation_count=2,
        max_text_length=1000
    )
    
    # Initialize summarizer
    summarizer = LLMSummarizer(
        config=summarizer_config,
        trigger=trigger
    )
    
    # Start monitoring
    await summarizer.start_monitoring()
    
    # Simulate transcript updates
    transcript_text = "Speaker 1: Hello, how are you today? Speaker 2: I'm doing well, thanks for asking!"
    await summarizer.update_transcription(transcript_text)
    
    # Text parsing example
    parser = Parser(
        config=LLMConfig(model="openai/gpt-4.1-nano", provider="google")
    )
    
    corrected_text = await parser.parse_text("This is a transcript with potential errors...")
    print(f"Corrected: {corrected_text}")
```

**File Processing Integration:**
```python
from audioinsight.server.file_handlers import handle_file_upload_and_process
from audioinsight.server.websocket_handlers import process_file_through_websocket
from fastapi import UploadFile
import asyncio

async def process_audio_file_with_llm(file_path: str):
    """Process audio file through unified pipeline with LLM analysis."""
    # Option 1: Direct file processing with JSON response including LLM summaries
    from pathlib import Path
    from audioinsight.server.utils import get_audio_duration
    
    # Get file duration for real-time simulation
    duration = get_audio_duration(file_path)
    
    # Process through same pipeline as live audio with LLM analysis
    from audioinsight.processors import AudioProcessor
    processor = AudioProcessor()
    
    # Stream file with temporal accuracy and LLM processing
    elapsed = await process_file_through_websocket(
        file_path, duration, processor
    )
    
    print(f"Processed {duration:.2f}s audio in {elapsed:.2f}s with LLM analysis")

# Option 2: Server-Sent Events for real-time progress with LLM summaries
@app.post("/upload-stream")
async def upload_file_stream(file: UploadFile):
    """Upload with real-time streaming results including LLM analysis."""
    from audioinsight.server.file_handlers import handle_file_upload_stream
    return await handle_file_upload_stream(file)
```

---

## ⚙️ Configuration Reference

### Core Parameters

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `--model` | Whisper model size | `large-v3-turbo` | `tiny`, `base`, `small`, `medium`, `large-v3-turbo` |
| `--language` | Source language | `auto` | Language codes such as `en`, `zh`, `ja`, etc. |
| `--task` | Processing task | `transcribe` | `transcribe`, `translate` |
| `--backend` | Whisper backend | `faster-whisper` | `faster-whisper`, `openai-api` |

### Advanced Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--diarization` | Enable speaker identification | `False` |
| `--confidence-validation` | Use confidence scores for faster output | `False` |
| `--vac` | Voice Activity Controller | `False` |
| `--no-vad` | Disable Voice Activity Detection | `False` |
| `--min-chunk-size` | Minimum audio chunk (seconds) | `0.5` |
| `--buffer-trimming` | Buffer management strategy | `segment` |
| `--buffer-trimming-sec` | Buffer trimming threshold (seconds) | `15.0` |
| `--vac-chunk-size` | VAC sample size (seconds) | `0.04` |

### LLM Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--llm-inference` | Enable LLM-based transcript analysis | `True` |
| `--fast-llm` | Fast LLM model for text parsing | `openai/gpt-4.1-nano` |
| `--base-llm` | Base LLM model for summarization | `openai/gpt-4.1-mini` |
| `--llm-trigger-time` | Idle time before LLM analysis (seconds) | `5.0` |
| `--llm-conversation-trigger` | Speaker turns before analysis | `2` |

> **Note:** If you are using OpenRouter, follow the format `model_name/model_id` for the model ID, e.g. `openai/openai/gpt-4.1-mini`.

### Server Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--host` | Server bind address | `localhost` |
| `--port` | Server port | `8001` |
| `--ssl-certfile` | SSL certificate path | `None` |
| `--ssl-keyfile` | SSL private key path | `None` |

### API Endpoints

AudioInsight provides comprehensive API endpoints for different processing modes:

| Endpoint | Method | Purpose | Response Format |
|----------|--------|---------|-----------------|
| `/` | GET | Web interface | HTML |
| `/asr` | WebSocket | Real-time transcription (live + file) with LLM analysis | WebSocket JSON |
| `/upload-file` | POST | Prepare file for WebSocket processing | JSON |
| `/upload` | POST | Direct file processing with LLM analysis | JSON |
| `/upload-stream` | POST | File processing with real-time updates | Server-Sent Events |
| `/cleanup-file` | POST | Clean up temporary files | JSON |

**Processing Modes:**
- **Live Recording**: Direct WebSocket connection with browser microphone and real-time LLM analysis
- **File Upload + WebSocket**: Unified processing through WebSocket with real-time simulation and LLM summarization
- **Direct File Processing**: Immediate processing with complete JSON response including LLM insights
- **Streaming File Processing**: Real-time progress updates via Server-Sent Events with LLM analysis

---

## 🔬 How It Works

### LocalAgreement Streaming Algorithm

AudioInsight's core innovation is the **LocalAgreement-2** policy that solves output stability in streaming ASR:

```python
# Simplified algorithm concept
def commit_tokens(previous_hypothesis, current_hypothesis):
    """Commit tokens that appear in both consecutive hypotheses"""
    committed = []
    for i, (prev_token, curr_token) in enumerate(zip(previous_hypothesis, current_hypothesis)):
        if prev_token.text == curr_token.text:
            committed.append(curr_token)
        else:
            break  # Stop at first disagreement
    return committed
```

### LLM-Powered Conversation Analysis

AudioInsight enhances raw transcription with intelligent analysis:

```python
# LLM analysis workflow
async def analyze_conversation(transcript_text, speaker_info):
    """Analyze conversation for insights and summaries"""
    # 1. Text parsing and correction
    corrected_text = await parser.parse_text(transcript_text)
    
    # 2. Conversation monitoring
    conversation_detected = monitor_speaker_turns(speaker_info)
    
    # 3. Triggered summarization
    if should_trigger_summary(idle_time, conversation_count):
        summary = await summarizer.generate_summary(corrected_text)
        return summary
    
    return corrected_text
```

**Key Benefits:**
- **Stability**: Prevents flickering text output
- **Accuracy**: Maintains Whisper's high-quality transcription
- **Low Latency**: Commits tokens as soon as they're stable
- **Context Preservation**: Maintains conversation flow
- **Intelligent Enhancement**: LLM-powered text correction and summarization
- **Conversation Understanding**: Automatic speaker turn detection and conversation analysis

### Processing Pipeline

1. **Audio Capture** → Browser MediaRecorder API captures audio
2. **Format Conversion** → FFmpeg converts WebM/Opus to PCM 
3. **Streaming Buffer** → LocalAgreement manages token validation
4. **Speaker Diarization** → Parallel speaker identification (optional)
5. **LLM Analysis** → Intelligent text processing and conversation summarization
6. **Real-time Output** → JSON responses via WebSocket with enhanced insights

---

## 🐋 Docker Deployment

### Quick Start

```bash
# Build image with LLM support
docker build -t audioinsight .

# Run with GPU support (recommended) and LLM analysis
docker run --gpus all -p 8001:8001 \
  -e OPENAI_API_KEY="your-key" \
  -e GOOGLE_API_KEY="your-key" \
  audioinsight --llm-inference

# CPU-only (slower but compatible)
docker run -p 8001:8001 \
  -e OPENAI_API_KEY="your-key" \
  audioinsight --llm-inference
```

### Custom Configuration

```bash
# Custom model and settings with LLM
docker run --gpus all -p 8001:8001 \
  -e OPENAI_API_KEY="your-key" \
  audioinsight \
  --model base \
  --diarization \
  --language auto \
  --llm-inference \
  --fast-llm "openai/gpt-4.1-nano" \
  --base-llm "openai/gpt-4.1-mini"
```

### Build Arguments

```bash
# Include additional features
docker build \
  --build-arg EXTRAS="whisper,diart,llm" \
  --build-arg HF_TOKEN="your_hf_token" \
  -t audioinsight-full .
```

---

## 🌐 Production Deployment

### Nginx Reverse Proxy

```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    # WebSocket support
    location / {
        proxy_pass http://localhost:8001;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_cache_bypass $http_upgrade;
    }
}
```

### Environment Variables

```bash
# LLM API Keys
export OPENAI_API_KEY="your-openai-key"
export OPENROUTER_API_KEY="your-openrouter-key"  
export GOOGLE_API_KEY="your-google-key"

# Model configurations
export WHISPER_MODEL="large-v3-turbo"
export FAST_LLM_MODEL="openai/gpt-4.1-nano"
export BASE_LLM_MODEL="openai/gpt-4.1-mini"
```

### Process Management

```bash
# Using systemd with LLM support
sudo systemctl enable audioinsight
sudo systemctl start audioinsight

# Using PM2 with environment variables
pm2 start "audioinsight-server --model large-v3-turbo --llm-inference" \
  --name audioinsight \
  --env OPENAI_API_KEY="your-key"

# Using Docker Compose
docker-compose up -d
```

### Scaling Considerations

- **Memory**: Larger models require more RAM (8GB+ recommended for large-v3)
- **CPU/GPU**: GPU acceleration highly recommended for real-time performance
- **Concurrent Users**: Each session requires ~500MB-2GB depending on model
- **Network**: WebSocket connections require persistent connections
- **LLM Usage**: Consider API rate limits and costs for LLM providers
- **API Keys**: Secure storage and rotation of LLM API credentials

---

## 🎯 Use Cases & Applications

### 🏢 Business Applications

**Meeting Transcription with AI Insights**
- Real-time meeting notes with speaker identification
- Automatic conversation summarization and key points extraction
- Action items detection and decision tracking
- Multi-language support for international teams

**Customer Support Analytics**
- Live call transcription for quality assurance
- Automated interaction summarization with sentiment analysis
- Conversation pattern detection and insights
- Compliance and training purposes with AI-enhanced analysis

**Content Creation with AI Enhancement** 
- Podcast and video transcription with automatic summaries
- Interview documentation with key insights extraction
- Live streaming captions with real-time content analysis
- Content optimization suggestions

### 🔬 Research & Development

**Academic Research with AI Analysis**
- Linguistic analysis of spoken data with automated insights
- Interview transcription for qualitative research with thematic analysis
- Conversation pattern detection and automated coding
- Accessibility tool development with intelligent text processing

**Healthcare with Clinical Insights**
- Clinical note-taking assistance with medical terminology correction
- Patient consultation documentation with automated summaries
- Telemedicine transcription with clinical decision support
- Medical conversation analysis and pattern detection

### 🛠️ Developer Integration

**API Integration with LLM Enhancement**
```python
# Embed in existing applications with AI analysis
from audioinsight import AudioInsight
from audioinsight.llm import LLMSummarizer

kit = AudioInsight(model="base", llm_inference=True)
summarizer = LLMSummarizer()

# Get enhanced transcription with AI insights
results = await kit.process_with_analysis(audio_data)
```

**Webhook Support with AI Insights**
```python
# Send enhanced transcriptions to external services
async def enhanced_transcription_webhook(result):
    if result.get('type') == 'summary':
        # POST AI summary to your API endpoint
        await post_summary_to_api(result)
    elif result.get('type') == 'transcription':
        # POST corrected transcription
        await post_transcription_to_api(result)

processor.add_callback(enhanced_transcription_webhook)
```

---

## 🔧 Troubleshooting

### Common Issues

**Installation Problems:**
```bash
# Missing FFmpeg
sudo apt update && sudo apt install ffmpeg

# LLM dependencies
pip install openai langchain langchain-google-genai

# Permission issues
pip install --user audioinsight

# M1 Mac compatibility
pip install audioinsight --no-deps
pip install torch torchvision torchaudio
```

**LLM Configuration Issues:**
```bash
# Check API keys
echo $OPENAI_API_KEY
echo $GOOGLE_API_KEY

# Test LLM connectivity
python -c "from audioinsight.llm import UniversalLLM; print('LLM ready')"

# Disable LLM if having issues
audioinsight-server --no-llm-inference
```

**Performance Issues:**
```bash
# Use smaller model for better speed
audioinsight-server --model base

# Enable GPU acceleration
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Reduce buffer size for lower latency
audioinsight-server --min-chunk-size 0.5

# Optimize LLM triggers
audioinsight-server --llm-trigger-time 10.0 --llm-conversation-trigger 5
```

**WebSocket Connection Issues:**
```bash
# Check firewall settings
sudo ufw allow 8001

# Test connection
curl -I http://localhost:8001

# Enable CORS for development
audioinsight-server --host 0.0.0.0
```
