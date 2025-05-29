# AudioInsight

> **Built on top of [WhisperLiveKit](https://github.com/QuentinFuxa/WhisperLiveKit) which built on top of [whisper_streaming](https://github.com/ufal/whisper_streaming).**

> **Real-time, Fully Local Speech-to-Text with Speaker Diarization**

Transform speech into text instantly with AudioInsight - a production-ready streaming ASR system that runs entirely on your machine. Built on OpenAI's Whisper with advanced streaming algorithms for low-latency, accurate transcription.

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

AudioInsight solves the fundamental challenge of real-time speech recognition by transforming OpenAI's batch-processing Whisper into a streaming system with **LocalAgreement** algorithms that ensure stable, coherent output.

### ✨ Core Advantages

🔒 **100% Local Processing** - No data leaves your machine  
🎙️ **Real-time Streaming** - See words appear as you speak  
👥 **Multi-Speaker Support** - Identify different speakers automatically  
🌐 **Multi-User Ready** - Handle multiple sessions simultaneously  
⚡ **Ultra-Low Latency** - Optimized streaming algorithms  
🛠️ **Production Ready** - Built for real-world applications  
🎯 **Modular Architecture** - Specialized components for enhanced maintainability  
📁 **Comprehensive File Support** - Multiple processing modes for audio files  
🔄 **Unified Processing Pipeline** - Same engine for live and file processing  
📊 **Multiple Response Formats** - JSON, WebSocket, and Server-Sent Events

### 🏗️ Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Browser UI    │───▶│  FastAPI Server  │───▶│  Core Engine    │
│  (WebSocket)    │    │  (Multi-Module)  │    │  (ASR + Diart)  │
└─────────────────┘    └──────────────────┘    └─────────────────┘
        │                        │                        │
        ▼                        ▼                        ▼
  Audio Capture           Modular Architecture      Real-time Processing
  MediaRecorder          ┌──────────────────┐       LocalAgreement Policy
                         │  server/config   │       Specialized Processors
                         │  server/handlers │       
                         │  server/websocket│       
                         │  server/utils    │       
                         └──────────────────┘       
```

**Three-Layer Modular Design:**
- **Frontend**: HTML5/JavaScript interface with WebSocket streaming and file upload
- **Server**: Modular FastAPI architecture with specialized components:
  - **Configuration Management** (`server/config.py`): CORS, audio validation, processing settings
  - **File Handlers** (`server/file_handlers.py`): Upload processing, SSE streaming, unified pipeline
  - **WebSocket Management** (`server/websocket_handlers.py`): Connection lifecycle, real-time processing
  - **Utilities** (`server/utils.py`): Audio processing, FFmpeg integration, streaming simulation
- **Core**: Advanced streaming algorithms with modular processors and speaker diarization

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

---

## 💡 Usage Guide

### Command Line Interface

**Basic Usage:**
```bash
# English transcription with default model
audioinsight-server

# Advanced configuration
audioinsight-server \
  --model large-v3-turbo \
  --language auto \
  --diarization \
  --host 0.0.0.0 \
  --port 8001
```

**SSL/HTTPS Support:**
```bash
audioinsight-server \
  --ssl-certfile cert.pem \
  --ssl-keyfile key.pem
# Access via https://localhost:8001
```

### Python Integration

**Basic Server:**
```python
from audioinsight import AudioInsight
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import asyncio

# Initialize core components
app = FastAPI()
kit = AudioInsight(model="large-v3-turbo", diarization=True)

@app.get("/")
async def get_interface():
    return HTMLResponse(kit.web_interface())

@app.websocket("/asr")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # Create session-specific processor
    from audioinsight.audio_processor import AudioProcessor
    processor = AudioProcessor()
    
    # Start processing pipeline
    results_generator = await processor.create_tasks()
    
    # Handle bidirectional communication
    async def send_results():
        async for result in results_generator:
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

**File Processing Integration:**
```python
from audioinsight.server.file_handlers import handle_file_upload_and_process
from audioinsight.server.websocket_handlers import process_file_through_websocket
from fastapi import UploadFile
import asyncio

async def process_audio_file(file_path: str):
    """Process audio file through unified pipeline."""
    # Option 1: Direct file processing with JSON response
    from pathlib import Path
    from audioinsight.server.utils import get_audio_duration
    
    # Get file duration for real-time simulation
    duration = get_audio_duration(file_path)
    
    # Process through same pipeline as live audio
    from audioinsight.processors import AudioProcessor
    processor = AudioProcessor()
    
    # Stream file with temporal accuracy
    elapsed = await process_file_through_websocket(
        file_path, duration, processor
    )
    
    print(f"Processed {duration:.2f}s audio in {elapsed:.2f}s")

# Option 2: Server-Sent Events for real-time progress
@app.post("/upload-stream")
async def upload_file_stream(file: UploadFile):
    """Upload with real-time streaming results."""
    from audioinsight.server.file_handlers import handle_file_upload_stream
    return await handle_file_upload_stream(file)
```

**Custom Processing Pipeline:**
```python
from audioinsight.main import AudioInsight
from audioinsight.processors import AudioProcessor
import asyncio

async def custom_transcription_pipeline():
    # Initialize with custom configuration
    kit = AudioInsight(
        model="large-v3-turbo",
        language="auto",
        diarization=True,
        confidence_validation=True
    )
    
    processor = AudioProcessor()
    
    # Custom result handler
    async def handle_results():
        async for result in await processor.create_tasks():
            # Process transcription result
            if result.get('type') == 'transcription':
                print(f"Speaker {result.get('speaker', 'Unknown')}: {result['text']}")
            elif result.get('type') == 'summary':
                print(f"Summary: {result['content']}")
    
    # Start processing
    asyncio.create_task(handle_results())
    
    # Simulate audio input (replace with real audio source)
    # await processor.process_audio(audio_bytes)
```

---

## ⚙️ Configuration Reference

### Core Parameters

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `--model` | Whisper model size | `large-v3-turbo` | `tiny`, `base`, `small`, `medium`, `large-v3`, `large-v3-turbo` |
| `--language` | Source language | `auto` | Language codes such as `en`, `zh`, `ja`, etc. |
| `--task` | Processing task | `transcribe` | `transcribe`, `translate` |
| `--backend` | Whisper backend | `faster-whisper` | `faster-whisper`, `openai-api`, `whisper` |

### Advanced Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--diarization` | Enable speaker identification | `False` |
| `--confidence-validation` | Use confidence scores for faster output | `False` |
| `--vac` | Voice Activity Controller | `False` |
| `--min-chunk-size` | Minimum audio chunk (seconds) | `1.0` |
| `--buffer-trimming` | Buffer management strategy | `segment` |

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
| `/asr` | WebSocket | Real-time transcription (live + file) | WebSocket JSON |
| `/upload-file` | POST | Prepare file for WebSocket processing | JSON |
| `/upload` | POST | Direct file processing | JSON |
| `/upload-stream` | POST | File processing with real-time updates | Server-Sent Events |
| `/cleanup-file` | POST | Clean up temporary files | JSON |

**Processing Modes:**
- **Live Recording**: Direct WebSocket connection with browser microphone
- **File Upload + WebSocket**: Unified processing through WebSocket with real-time simulation
- **Direct File Processing**: Immediate processing with complete JSON response
- **Streaming File Processing**: Real-time progress updates via Server-Sent Events

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

**Key Benefits:**
- **Stability**: Prevents flickering text output
- **Accuracy**: Maintains Whisper's high-quality transcription
- **Low Latency**: Commits tokens as soon as they're stable
- **Context Preservation**: Maintains conversation flow

### Processing Pipeline

1. **Audio Capture** → Browser MediaRecorder API captures audio
2. **Format Conversion** → FFmpeg converts WebM/Opus to PCM 
3. **Streaming Buffer** → LocalAgreement manages token validation
4. **Speaker Diarization** → Parallel speaker identification (optional)
5. **Real-time Output** → JSON responses via WebSocket

---

## 🐋 Docker Deployment

### Quick Start

```bash
# Build image
docker build -t audioinsight .

# Run with GPU support (recommended)
docker run --gpus all -p 8001:8001 audioinsight

# CPU-only (slower but compatible)
docker run -p 8001:8001 audioinsight
```

### Custom Configuration

```bash
# Custom model and settings
docker run --gpus all -p 8001:8001 audioinsight \
  --model base \
  --diarization \
  --language auto
```

### Build Arguments

```bash
# Include additional features
docker build \
  --build-arg EXTRAS="whisper,diart" \
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

### Process Management

```bash
# Using systemd
sudo systemctl enable audioinsight
sudo systemctl start audioinsight

# Using PM2
pm2 start "audioinsight-server --model large-v3-turbo" --name audioinsight

# Using Docker Compose
docker-compose up -d
```

### Scaling Considerations

- **Memory**: Larger models require more RAM (8GB+ recommended for large-v3)
- **CPU/GPU**: GPU acceleration highly recommended for real-time performance
- **Concurrent Users**: Each session requires ~500MB-2GB depending on model
- **Network**: WebSocket connections require persistent connections

---

## 🎯 Use Cases & Applications

### 🏢 Business Applications

**Meeting Transcription**
- Real-time meeting notes with speaker identification
- Action items and decision tracking
- Multi-language support for international teams

**Customer Support**
- Live call transcription for quality assurance
- Automated interaction summarization
- Compliance and training purposes

**Content Creation** 
- Podcast and video transcription
- Interview documentation
- Live streaming captions

### 🔬 Research & Development

**Academic Research**
- Linguistic analysis of spoken data
- Interview transcription for qualitative research
- Accessibility tool development

**Healthcare**
- Clinical note-taking assistance
- Patient consultation documentation
- Telemedicine transcription

### 🛠️ Developer Integration

**API Integration**
```python
# Embed in existing applications
from audioinsight import AudioInsight

kit = AudioInsight(model="base")
# Integrate with your WebSocket/audio pipeline
```

**Webhook Support**
```python
# Send transcriptions to external services
async def transcription_webhook(result):
    # POST to your API endpoint
    await post_to_api(result)

processor.add_callback(transcription_webhook)
```

---

## 🔧 Troubleshooting

### Common Issues

**Installation Problems:**
```bash
# Missing FFmpeg
sudo apt update && sudo apt install ffmpeg

# Permission issues
pip install --user audioinsight

# M1 Mac compatibility
pip install audioinsight --no-deps
pip install torch torchvision torchaudio
```

**Performance Issues:**
```bash
# Use smaller model for better speed
audioinsight-server --model base

# Enable GPU acceleration
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Reduce buffer size for lower latency
audioinsight-server --min-chunk-size 0.5
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
