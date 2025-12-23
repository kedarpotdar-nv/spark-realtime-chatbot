#!/bin/bash
# Test script to compare llama-server vs trtllm-serve streaming APIs

echo "=========================================="
echo "Testing llama-server (standard OpenAI API)"
echo "=========================================="
echo ""
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4x-local",
    "messages": [{"role": "user", "content": "Say hello"}],
    "stream": true,
    "max_tokens": 50
  }' \
  --no-buffer 2>&1 | head -20

echo ""
echo ""
echo "=========================================="
echo "Testing trtllm-serve (TensorRT-LLM API)"
echo "=========================================="
echo ""
echo "Note: Update TRTLLM_URL and TRTLLM_MODEL below if needed"
TRTLLM_URL="${TRTLLM_URL:-http://localhost:8001/v1/chat/completions}"
TRTLLM_MODEL="${TRTLLM_MODEL:-your-model-name}"
echo "Using URL: $TRTLLM_URL"
echo "Using Model: $TRTLLM_MODEL"
echo ""
curl -X POST "$TRTLLM_URL" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$TRTLLM_MODEL\",
    \"messages\": [{\"role\": \"user\", \"content\": \"Say hello\"}],
    \"stream\": true,
    \"max_tokens\": 50
  }" \
  --no-buffer 2>&1 | head -30

echo ""
echo ""
echo "=========================================="
echo "Compare the output formats above"
echo "=========================================="

