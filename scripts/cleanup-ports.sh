#!/bin/bash

# Utility script to clean up processes on specified ports
# Usage: ./cleanup-ports.sh PORT1 PORT2 PORT3...

cleanup_port() {
    local port=$1
    echo "🔍 Checking for existing processes on port $port..."
    
    # Get PID of process using the port
    PID=$(lsof -ti:$port 2>/dev/null)
    
    if [ ! -z "$PID" ]; then
        echo "⚠️  Found process using port $port (PID: $PID)"
        
        # Get process name for better logging
        PROCESS_NAME=$(ps -p $PID -o comm= 2>/dev/null || echo "unknown")
        echo "📛 Killing process: $PROCESS_NAME (PID: $PID) on port $port"
        
        # Try graceful shutdown first
        kill $PID 2>/dev/null
        sleep 2
        
        # Check if process is still running
        if kill -0 $PID 2>/dev/null; then
            echo "⚡ Process still running, forcing kill..."
            kill -9 $PID 2>/dev/null
            sleep 1
        fi
        
        # Verify port is free
        if lsof -ti:$port >/dev/null 2>&1; then
            echo "❌ Failed to free port $port"
            return 1
        else
            echo "✅ Port $port is now free"
        fi
    else
        echo "✅ Port $port is already free"
    fi
    
    return 0
}

# Main execution
if [ $# -eq 0 ]; then
    echo "Usage: $0 PORT1 [PORT2] [PORT3] ..."
    echo "Example: $0 3030 8080"
    exit 1
fi

echo "🧹 AudioInsight Port Cleanup Utility"
echo "=================================="

# Clean up all specified ports
FAILED_PORTS=()
for port in "$@"; do
    if ! cleanup_port $port; then
        FAILED_PORTS+=($port)
    fi
    echo ""
done

# Report results
if [ ${#FAILED_PORTS[@]} -eq 0 ]; then
    echo "🎉 All ports cleaned successfully!"
    exit 0
else
    echo "❌ Failed to clean ports: ${FAILED_PORTS[*]}"
    exit 1
fi 