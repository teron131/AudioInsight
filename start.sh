#!/bin/bash

# Kill any existing processes on port 8001
echo "Checking for existing processes on port 8001..."
PID=$(lsof -ti:8001)
if [ ! -z "$PID" ]; then
    echo "Killing existing process on port 8001 (PID: $PID)"
    kill -9 $PID
    sleep 2
fi

# Start the audioinsight server
echo "Starting audioinsight server..."

audioinsight-server \
    --backend faster-whisper \
    --model large-v3-turbo \
    --base-llm "openai/gpt-4.1-mini" \
    --llm-trigger-time 5.0 \
    --fast-llm "openai/gpt-4.1-nano" \
    # --diarization