#!/bin/bash
# Launch script for spark-realtime

# Default configuration - override with environment variables
export ASR_MODE="${ASR_MODE:-api}"  # "api" for server, "local" for in-process
export ASR_API_URL="${ASR_API_URL:-http://localhost:9000/v1/audio/transcriptions}"
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

export NEMOTRON_SERVER_URL="${NEMOTRON_SERVER_URL:-http://localhost:8005/v1/chat/completions}"
export NEMOTRON_MODEL="${NEMOTRON_MODEL:-nemotron-3-nano}"

export KOKORO_LANG="${KOKORO_LANG:-a}"
export KOKORO_VOICE="${KOKORO_VOICE:-af_bella}"
export TTS_OVERLAP="${TTS_OVERLAP:-false}"  # Overlap TTS with LLM streaming

# HuggingFace cache directory (defaults to /home/nvidia/hfcache if not set)
# Set HF_HOME to override the default ~/.cache/huggingface location
export HF_HOME="${HF_HOME:-/home/nvidia/hfcache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
# Create cache directory if it doesn't exist
mkdir -p "$HUGGINGFACE_HUB_CACHE"

# Server bind & SSL configuration
export HOST="${HOST:-0.0.0.0}"
export PORT="${PORT:-9443}"
export SSL_ENABLED="${SSL_ENABLED:-true}"
export SSL_KEY="${SSL_KEY:-${SSL_KEYFILE:-key.pem}}"
export SSL_CERT="${SSL_CERT:-${SSL_CERTFILE:-cert.pem}}"

# Hermes Agent (Nous Research) — lazy by default; set HERMES_URL or HERMES_ENABLED=true to warm at startup
export HERMES_URL="${HERMES_URL:-}"
export HERMES_MODEL="${HERMES_MODEL:-hermes-agent}"
export HERMES_TIMEOUT="${HERMES_TIMEOUT:-600}"
export HERMES_MAX_CONCURRENT="${HERMES_MAX_CONCURRENT:-4}"

# Personal filesystem / shell tools (Jarvis local mode on DGX host, no Docker sandbox)
export PERSONAL_FS_ENABLED="${PERSONAL_FS_ENABLED:-true}"
export PERSONAL_SHELL_ENABLED="${PERSONAL_SHELL_ENABLED:-true}"
export PERSONAL_SHELL_TIMEOUT="${PERSONAL_SHELL_TIMEOUT:-60}"
export PERSONAL_READ_MAX_BYTES="${PERSONAL_READ_MAX_BYTES:-204800}"
# Optional colon-separated allowlist; default is $HOME + Downloads/Documents/Desktop/Projects + workspace
# export PERSONAL_FS_ROOTS="${PERSONAL_FS_ROOTS:-}"

# Parse command line flags
while [[ $# -gt 0 ]]; do
    case "$1" in
        --local-asr)
            export ASR_MODE="local"
            echo "✅ Local ASR enabled (in-process faster-whisper)"
            shift
            ;;
        --tts-overlap)
            export TTS_OVERLAP="true"
            echo "✅ TTS/LLM overlap enabled (parallel TTS generation)"
            shift
            ;;
        --trtllm)
            export LLM_BACKEND="trtllm"
            echo "✅ TensorRT-LLM backend enabled"
            shift
            ;;
        --http|--no-ssl)
            export SSL_ENABLED="false"
            shift
            ;;
        --host)
            export HOST="$2"
            shift 2
            ;;
        --port)
            export PORT="$2"
            shift 2
            ;;
        --ssl-key|--ssl-keyfile)
            export SSL_KEY="$2"
            shift 2
            ;;
        --ssl-cert|--ssl-certfile)
            export SSL_CERT="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

# Check if certificates exist when SSL is enabled
if [ "$SSL_ENABLED" = "true" ]; then
    if [ ! -f "$SSL_KEY" ] || [ ! -f "$SSL_CERT" ]; then
        echo "=========================================="
        echo "SSL certificates not found ($SSL_KEY, $SSL_CERT)!"
        echo "=========================================="
        echo "Generating self-signed fallback certificate..."
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
        echo "✅ Self-signed certificates generated: $SSL_KEY, $SSL_CERT"
        echo ""
    fi
fi

SCHEME="HTTPS"
if [ "$SSL_ENABLED" != "true" ]; then
    SCHEME="HTTP"
fi

echo "=========================================="
echo "Launching spark-realtime ($SCHEME)"
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
echo "Nemotron URL: $NEMOTRON_SERVER_URL"
echo "Nemotron Model: $NEMOTRON_MODEL"
echo "TTS Overlap: $TTS_OVERLAP"
echo "HF Cache: $HUGGINGFACE_HUB_CACHE"
echo "Bind: ${HOST}:${PORT} ($SCHEME)"
if [ "$SSL_ENABLED" = "true" ]; then
    echo "SSL Key: $SSL_KEY"
    echo "SSL Cert: $SSL_CERT"
fi
echo "=========================================="
echo ""
if [ "$SSL_ENABLED" = "true" ]; then
    echo "ℹ️  Tip: To avoid browser security warnings on LAN, generate locally-trusted"
    echo "   certificates with mkcert: ./scripts/setup-mkcert-https.sh"
    echo ""
fi

# Launch server
# Use exec to replace shell process so signals (SIGTERM, SIGINT) go directly to uvicorn
if [ "$SSL_ENABLED" = "true" ]; then
    exec uvicorn server:app --host "$HOST" --port "$PORT" \
        --ssl-keyfile "$SSL_KEY" \
        --ssl-certfile "$SSL_CERT"
else
    exec uvicorn server:app --host "$HOST" --port "$PORT" \
        --proxy-headers --forwarded-allow-ips="127.0.0.1"
fi

