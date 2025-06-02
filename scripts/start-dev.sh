#!/bin/bash

# Unified development startup script
# Cleans up both frontend and backend ports and starts both services

FRONTEND_PORT=3030
BACKEND_PORT=8080
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "🚀 Starting AudioInsight Development Environment"
echo "==============================================="

# Clean up both ports
echo "🧹 Cleaning up development ports..."
"$SCRIPT_DIR/cleanup-ports.sh" $FRONTEND_PORT $BACKEND_PORT

if [ $? -ne 0 ]; then
    echo "❌ Failed to clean up some ports. Check the output above."
    exit 1
fi

# Change to project root
cd "$PROJECT_ROOT" || {
    echo "❌ Failed to change to project directory"
    exit 1
}

echo "🎯 Starting both frontend and backend services..."
echo "📍 Frontend will be available at: http://localhost:$FRONTEND_PORT"
echo "📍 Backend API will be available at: http://localhost:$BACKEND_PORT"
echo "🔌 WebSocket endpoint: ws://localhost:$BACKEND_PORT/asr"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Start both services using npm
npm run dev 