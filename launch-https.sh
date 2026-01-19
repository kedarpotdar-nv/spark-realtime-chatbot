#!/bin/bash
# Launch script for spark-realtime-chatbot

# Default configuration - override with environment variables
export ASR_MODE="${ASR_MODE:-api}"  # "api" for server, "local" for in-process
export ASR_API_URL="${ASR_API_URL:-http://localhost:8000/v1/audio/transcriptions}"
export ASR_API_KEY="${ASR_API_KEY:-dummy-key}"
export ASR_MODEL="${ASR_MODEL:-Systran/faster-whisper-small.en}"
export ASR_DEVICE="${ASR_DEVICE:-cuda}"
export ASR_COMPUTE_TYPE="${ASR_COMPUTE_TYPE:-float16}"

export LLM_SERVER_URL="${LLM_SERVER_URL:-http://localhost:8080/v1/chat/completions}"
export LLM_MODEL="${LLM_MODEL:-qwen3-vl}"
export LLM_MAX_TOKENS="${LLM_MAX_TOKENS:-4096}"  # Increased for code generation

# Vision Language Model (Qwen3-VL via llama.cpp)
export VLM_SERVER_URL="${VLM_SERVER_URL:-http://localhost:8080/v1/chat/completions}"
export VLM_MODEL="${VLM_MODEL:-qwen3-vl}"
export VLM_MAX_TOKENS="${VLM_MAX_TOKENS:-150}"  # Lower for faster responses

export KOKORO_LANG="${KOKORO_LANG:-a}"
export KOKORO_VOICE="${KOKORO_VOICE:-af_bella}"
export TTS_OVERLAP="${TTS_OVERLAP:-false}"  # Overlap TTS with LLM streaming

# HuggingFace cache directory (defaults to /home/nvidia/hfcache if not set)
# Set HF_HOME to override the default ~/.cache/huggingface location
export HF_HOME="${HF_HOME:-/home/nvidia/hfcache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
# Create cache directory if it doesn't exist
mkdir -p "$HUGGINGFACE_HUB_CACHE"

# SSL certificate paths
SSL_KEY="${SSL_KEY:-key.pem}"
SSL_CERT="${SSL_CERT:-cert.pem}"

# Port for HTTPS server (default 8443)
PORT="${PORT:-8443}"

# Check if certificates exist
if [ ! -f "$SSL_KEY" ] || [ ! -f "$SSL_CERT" ]; then
    echo "=========================================="
    echo "SSL certificates not found!"
    echo "=========================================="
    echo "Generating self-signed certificate..."
    echo ""
    openssl req -x509 -newkey rsa:4096 \
        -keyout "$SSL_KEY" -out "$SSL_CERT" \
        -days 365 -nodes \
        -subj "/CN=localhost" 2>/dev/null
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to generate certificates. Install openssl or provide certificates manually."
        echo ""
        echo "To generate manually:"
        echo "  openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes"
        exit 1
    fi
    echo "✅ Certificates generated: $SSL_KEY, $SSL_CERT"
    echo ""
fi

echo "=========================================="
echo "Launching spark-realtime-chatbot (HTTPS)"
echo "=========================================="
echo "ASR Mode: $ASR_MODE"
if [ "$ASR_MODE" == "local" ]; then
    echo "ASR Device: $ASR_DEVICE"
    echo "ASR Compute Type: $ASR_COMPUTE_TYPE"
else
    echo "ASR API URL: $ASR_API_URL"
fi
echo "ASR Model: $ASR_MODEL"
echo "LLM URL: $LLM_SERVER_URL"
echo "LLM Model: $LLM_MODEL"
echo "LLM Max Tokens: $LLM_MAX_TOKENS"
echo "VLM URL: $VLM_SERVER_URL"
echo "VLM Model: $VLM_MODEL"
echo "VLM Max Tokens: $VLM_MAX_TOKENS"
echo "TTS Overlap: $TTS_OVERLAP"
echo "HF Cache: $HUGGINGFACE_HUB_CACHE"
echo "Port: $PORT (HTTPS)"
echo "SSL Key: $SSL_KEY"
echo "SSL Cert: $SSL_CERT"
echo "=========================================="
echo ""
echo "⚠️  Note: If using self-signed certificate,"
echo "   you'll need to accept browser security warning"
echo ""

# Parse command line flags
for arg in "$@"; do
    case $arg in
        --local-asr)
            export ASR_MODE="local"
            echo "✅ Local ASR enabled (in-process faster-whisper)"
            ;;
        --tts-overlap)
            export TTS_OVERLAP="true"
            echo "✅ TTS/LLM overlap enabled (parallel TTS generation)"
            ;;
        --trtllm)
            export LLM_BACKEND="trtllm"
            echo "✅ TensorRT-LLM backend enabled"
            ;;
    esac
done

# Launch server with SSL
# Use exec to replace shell process so signals (SIGTERM, SIGINT) go directly to uvicorn
exec uvicorn server:app --host 0.0.0.0 --port "$PORT" \
    --ssl-keyfile "$SSL_KEY" \
    --ssl-certfile "$SSL_CERT"

