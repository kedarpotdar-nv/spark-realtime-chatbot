#!/bin/bash
# Launch script for ces-voice on Spark

# Default configuration - override with environment variables
export ASR_API_URL="${ASR_API_URL:-http://localhost:8000/v1/audio/transcriptions}"
export ASR_API_KEY="${ASR_API_KEY:-dummy-key}"
export ASR_MODEL="${ASR_MODEL:-Systran/faster-whisper-small.en}"

export LLM_SERVER_URL="${LLM_SERVER_URL:-http://localhost:8080/v1/chat/completions}"
export LLM_MODEL="${LLM_MODEL:-gpt-4x-local}"
export LLM_MAX_TOKENS="${LLM_MAX_TOKENS:-4096}"  # Increased for code generation
# Note: If using llama-server, start it first:
# ./llama-server --model gpt-oss-120b-mxfp4-00001-of-00003.gguf -ngl 99 --port 8080
# Then set LLM_MODEL to match your model name
# For longer code generation, increase LLM_MAX_TOKENS (e.g., export LLM_MAX_TOKENS=8192)

export KOKORO_LANG="${KOKORO_LANG:-a}"
export KOKORO_VOICE="${KOKORO_VOICE:-af_bella}"

# HuggingFace cache directory (defaults to /home/nvidia/hfcache if not set)
# Set HF_HOME to override the default ~/.cache/huggingface location
export HF_HOME="${HF_HOME:-/home/nvidia/hfcache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
# Create cache directory if it doesn't exist
mkdir -p "$HUGGINGFACE_HUB_CACHE"

# Port for this server (default 8001 to avoid conflict with faster-whisper on 8000)
PORT="${PORT:-8001}"

echo "=========================================="
echo "Launching ces-voice server"
echo "=========================================="
echo "ASR API URL: $ASR_API_URL"
echo "ASR Model: $ASR_MODEL"
echo "LLM URL: $LLM_SERVER_URL"
echo "LLM Model: $LLM_MODEL"
echo "LLM Max Tokens: $LLM_MAX_TOKENS"
echo "HF Cache: $HUGGINGFACE_HUB_CACHE"
echo "Port: $PORT"
echo "=========================================="
echo ""

# Check for --trtllm flag
if [ "$1" == "--trtllm" ]; then
    export LLM_BACKEND="trtllm"
    echo "TensorRT-LLM backend enabled"
fi

# Launch server
uvicorn server:app --host 0.0.0.0 --port "$PORT"

