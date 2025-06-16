#!/bin/bash

echo "🚀 Starting AudioInsight backend and waiting for readiness..."

# Start the backend in the background
./start.sh &
BACKEND_PID=$!

echo "📡 Backend started (PID: $BACKEND_PID), waiting for readiness..."

# Function to check if backend is ready
check_backend_ready() {
    curl -s http://localhost:8080/health 2>/dev/null | grep -q '"backend_ready":true'
}

# Wait for backend to be ready
ATTEMPTS=0
MAX_ATTEMPTS=60  # 60 seconds max wait time
WAIT_INTERVAL=1  # Check every 1 second

while [ $ATTEMPTS -lt $MAX_ATTEMPTS ]; do
    if check_backend_ready; then
        echo "✅ Backend is ready! (took ${ATTEMPTS} seconds)"
        echo "🎯 Backend fully initialized and ready for connections"
        exit 0
    fi
    
    # Check if backend process is still running
    if ! kill -0 $BACKEND_PID 2>/dev/null; then
        echo "❌ Backend process died unexpectedly"
        exit 1
    fi
    
    echo "⏳ Waiting for backend readiness... (${ATTEMPTS}/${MAX_ATTEMPTS})"
    sleep $WAIT_INTERVAL
    ATTEMPTS=$((ATTEMPTS + 1))
done

echo "❌ Backend failed to become ready within ${MAX_ATTEMPTS} seconds"
echo "🔪 Killing backend process..."
kill $BACKEND_PID 2>/dev/null
exit 1 