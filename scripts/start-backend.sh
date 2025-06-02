#!/bin/bash

# Backend startup script with port cleanup
# Cleans up port 8080 and starts AudioInsight server

BACKEND_PORT=8080
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "🚀 Starting AudioInsight Backend"
echo "==============================="

# Clean up backend port
echo "🧹 Cleaning up port $BACKEND_PORT..."
"$SCRIPT_DIR/cleanup-ports.sh" $BACKEND_PORT

if [ $? -ne 0 ]; then
    echo "❌ Failed to clean up port $BACKEND_PORT"
    exit 1
fi

# Change to project root
cd "$PROJECT_ROOT" || {
    echo "❌ Failed to change to project directory"
    exit 1
}

# Start backend server
echo "🎯 Starting AudioInsight server on port $BACKEND_PORT..."
echo "📍 Access backend API at: http://localhost:$BACKEND_PORT"
echo "🔌 WebSocket endpoint: ws://localhost:$BACKEND_PORT/asr"
echo ""

# Start the audioinsight server
audioinsight-server \
    --backend faster-whisper \
    --model large-v3-turbo \
    --llm-inference \
    --base-llm "openai/gpt-4.1-mini" \
    --fast-llm "openai/gpt-4.1-nano" 