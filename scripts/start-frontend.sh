#!/bin/bash

# Frontend startup script with port cleanup
# Cleans up port 3030 and starts Next.js development server

FRONTEND_PORT=3030
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "🚀 Starting AudioInsight Frontend"
echo "================================"

# Clean up frontend port
echo "🧹 Cleaning up port $FRONTEND_PORT..."
"$SCRIPT_DIR/cleanup-ports.sh" $FRONTEND_PORT

if [ $? -ne 0 ]; then
    echo "❌ Failed to clean up port $FRONTEND_PORT"
    exit 1
fi

# Change to frontend directory
cd "$PROJECT_ROOT/audioinsight-ui" || {
    echo "❌ Failed to change to frontend directory"
    exit 1
}

# Start frontend development server
echo "🎯 Starting Next.js development server on port $FRONTEND_PORT..."
echo "📍 Access frontend at: http://localhost:$FRONTEND_PORT"
echo ""

# Use npm to start the frontend
npm run dev:frontend 