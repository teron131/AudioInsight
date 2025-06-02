# AudioInsight UI

A modern Next.js frontend for the AudioInsight audio transcription and analysis system.

## Features

- **Real-time Audio Transcription**: Live microphone recording with WebSocket streaming
- **File Upload Support**: Drag-and-drop or browse to upload audio files
- **Speaker Diarization**: Automatic speaker identification and separation
- **AI-Powered Analysis**: Automatic summary, key points, and keyword extraction
- **Multiple Export Formats**: Export transcripts as TXT, SRT, VTT, or JSON
- **Modern UI**: Built with Next.js, TypeScript, and Tailwind CSS
- **Real-time Updates**: Live transcript updates with processing indicators

## Architecture

### Frontend Components

- **`useAudioInsight`**: Main hook that orchestrates WebSocket connections, audio recording, and file uploads
- **`AudioInsightWebSocket`**: WebSocket client for real-time communication with the backend
- **`AudioInsightAPI`**: REST API client for file uploads and exports
- **`TranscriptDisplay`**: Real-time transcript visualization with speaker colors
- **`AnalysisPanel`**: AI analysis results display (summary, key points, keywords)
- **`WaveformVisualization`**: Animated waveform during recording/processing
- **`ExportMenu`**: Multi-format transcript export functionality

### Backend Integration

The UI connects to the AudioInsight backend via:

- **WebSocket (`/asr`)**: Real-time audio streaming and transcript updates
- **REST API (`/upload-file`)**: File upload endpoint
- **Export API (`/api/export/transcript`)**: Transcript export in multiple formats

## Getting Started

### Prerequisites

- Node.js 18+ and npm/pnpm
- AudioInsight backend server running (see main project README)

### Installation

1. Install dependencies:
```bash
npm install
# or
pnpm install
```

2. Start the development server:
```bash
npm run dev
# or
pnpm dev
```

3. Open [http://localhost:3030](http://localhost:3030) in your browser

### Backend Setup

Make sure the AudioInsight backend is running on the same host. The UI will automatically connect to:
- WebSocket: `ws://localhost:8080/asr` (or current host)
- API: `http://localhost:8080` (or current host)

## Usage

### Live Recording

1. Click "Start Recording" to begin live transcription
2. Speak into your microphone
3. Watch real-time transcript appear with speaker identification
4. Click "Stop Recording" to finish and trigger AI analysis

### File Upload

1. Click "Upload Audio File" or drag-and-drop an audio file
2. Supported formats: MP3, WAV, M4A, and other common audio formats
3. The file will be processed in real-time simulation
4. AI analysis will be generated automatically

### Export Options

- **TXT**: Plain text with speaker labels
- **SRT**: SubRip subtitle format with timestamps
- **VTT**: WebVTT subtitle format
- **JSON**: Complete data including analysis and metadata

## Development

### Project Structure

```
audioinsight-ui/
├── app/                    # Next.js app router
│   ├── page.tsx           # Main transcription interface
│   ├── settings/          # Settings page
│   └── layout.tsx         # Root layout with theme provider
├── components/            # React components
│   ├── ui/               # shadcn/ui components
│   ├── analysis-panel.tsx
│   ├── transcript-display.tsx
│   ├── waveform-visualization.tsx
│   └── export-menu.tsx
├── hooks/                # Custom React hooks
│   ├── use-audioinsight.ts
│   ├── use-audio-recording.ts
│   └── use-toast.ts
├── lib/                  # Utilities
│   ├── websocket.ts      # WebSocket client
│   ├── api.ts           # REST API client
│   └── utils.ts         # General utilities
└── styles/              # Global styles
```

### Key Technologies

- **Next.js 15**: React framework with app router
- **TypeScript**: Type-safe development
- **Tailwind CSS**: Utility-first styling
- **shadcn/ui**: Modern component library
- **WebSocket API**: Real-time communication
- **MediaRecorder API**: Browser audio recording

## Configuration

The UI automatically adapts to the current host and protocol:
- Development: `ws://localhost:3030/asr`
- Production: Uses current window location

No additional configuration is required for basic usage.

## Troubleshooting

### WebSocket Connection Issues

- Ensure the AudioInsight backend is running
- Check browser console for connection errors
- Verify firewall settings allow WebSocket connections

### Audio Recording Issues

- Grant microphone permissions when prompted
- Check browser compatibility (modern browsers required)
- Ensure HTTPS in production for microphone access

### File Upload Issues

- Verify file format is supported
- Check file size limits
- Ensure backend storage is available

## Contributing

1. Follow the existing code style and patterns
2. Use TypeScript for all new code
3. Add proper error handling and loading states
4. Test with both live recording and file upload
5. Ensure responsive design works on mobile devices

## License

This project is part of the AudioInsight system. See the main project for license information.

To start both the frontend and backend with a single command:

```bash
npm run dev
```

This will start:
- **Frontend**: Next.js development server on http://localhost:3030 (cyan logs)
- **Backend**: AudioInsight Python server on http://localhost:8080 (magenta logs)

Individual commands:
- `npm run dev:frontend` - Start only the frontend
- `npm run dev:backend` - Start only the backend
- `npm run build` - Build the production frontend
- `npm run lint` - Run ESLint

### Recent Fixes

- **Buffer Text Commitment**: Fixed issue where the final portion of transcripts wasn't being committed to the UI when processing completed. The system now properly commits any remaining buffer text as a final transcript line when completion signals (`ready_to_stop`, `final=true`) are received. 