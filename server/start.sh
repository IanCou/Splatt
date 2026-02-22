#!/bin/bash
# GPU Server Startup Script
# ============================================================================
# Quick start script for the GPU Gaussian Splatting API server
# ============================================================================

set -e

echo "=========================================="
echo "GPU Gaussian Splatting Server"
echo "=========================================="

# Check if .env exists
if [ ! -f .env ]; then
    echo "⚠️  No .env file found!"
    echo "Creating from .env.example..."
    cp .env.example .env
    echo ""
    echo "⚠️  IMPORTANT: Edit .env and set your API key!"
    echo "   nano .env"
    echo ""
    read -p "Press Enter to continue (or Ctrl+C to abort)..."
fi

# Load environment
export $(cat .env | grep -v '^#' | xargs)

# Check Python environment
echo ""
echo "Checking Python environment..."
if [ -n "$NERF_PYTHON_PATH" ]; then
    PYTHON_BIN="$NERF_PYTHON_PATH"
else
    PYTHON_BIN="python"
fi

echo "Python: $PYTHON_BIN"
$PYTHON_BIN --version

# Check GPU
echo ""
echo "Checking GPU..."
$PYTHON_BIN -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" || echo "⚠️  Could not check GPU status"

# Check dependencies
echo ""
echo "Checking dependencies..."
$PYTHON_BIN -c "import fastapi, uvicorn, pydantic, requests" || {
    echo "❌ Missing dependencies!"
    echo "Install with: pip install -r requirements.txt"
    exit 1
}

# Create workspace directory
if [ -n "$WORKSPACE_ROOT" ]; then
    mkdir -p "$WORKSPACE_ROOT"
    echo "Workspace: $WORKSPACE_ROOT"
fi

# Start server
echo ""
echo "=========================================="
echo "Starting server..."
echo "Host: ${SERVER_HOST:-0.0.0.0}"
echo "Port: ${SERVER_PORT:-8080}"
echo "=========================================="
echo ""

# Run with uvicorn
$PYTHON_BIN -m uvicorn main:app \
    --host "${SERVER_HOST:-0.0.0.0}" \
    --port "${SERVER_PORT:-8080}" \
    --log-level info
