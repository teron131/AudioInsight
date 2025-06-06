#!/bin/bash

# AudioInsight Backend Server Startup Script
# Uses the port cleanup utility for consistent port management

BACKEND_PORT=8080
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "🚀 Starting AudioInsight Backend Server"
echo "======================================"

# Clean up backend port using the utility script
if [ -f "$SCRIPT_DIR/scripts/cleanup-ports.sh" ]; then
    echo "🧹 Cleaning up port $BACKEND_PORT..."
    "$SCRIPT_DIR/scripts/cleanup-ports.sh" $BACKEND_PORT
    
    if [ $? -ne 0 ]; then
        echo "❌ Failed to clean up port $BACKEND_PORT"
        exit 1
    fi
else
    # Fallback to original cleanup method if utility script is not available
    echo "🧹 Checking for existing processes on port $BACKEND_PORT..."
    PID=$(lsof -ti:$BACKEND_PORT)
    if [ ! -z "$PID" ]; then
        echo "📛 Killing existing process on port $BACKEND_PORT (PID: $PID)"
        kill -9 $PID
        sleep 2
    fi
fi

echo "🎯 Starting AudioInsight server on port $BACKEND_PORT..."
echo "📍 Access backend API at: http://localhost:$BACKEND_PORT"
echo "🔌 WebSocket endpoint: ws://localhost:$BACKEND_PORT/asr"
echo ""

audioinsight-server \
    --backend faster-whisper --model base \
    # --backend openai-api --model whisper-1 \
    --llm_inference \
    --base_llm "openai/gpt-4.1-mini" \
    --fast_llm "openai/gpt-4.1-nano" \
