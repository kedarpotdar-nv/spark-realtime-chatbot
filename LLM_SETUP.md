# LLM Server Setup

## Using llama-server

The ces-voice server expects an OpenAI-compatible API endpoint. If you're using `llama-server`, here's how to set it up:

### 1. Start llama-server

```bash
./llama-server --model gpt-oss-20b-mxfp4.gguf -ngl 99 --port 8080
```

**Parameters:**
- `--model`: Path to your model file
- `-ngl 99`: Number of GPU layers (adjust based on your GPU memory)
- `--port 8080`: Port for the server (default matches ces-voice config)

### 2. Configure ces-voice

The default configuration already points to `http://localhost:8080/v1/chat/completions`, which should work with llama-server.

**Set the model name** (if different from default):
```bash
export LLM_MODEL="gpt-oss-20b-mxfp4"  # Or whatever model name your server expects
```

### 3. Verify LLM Server

Test that the server is responding:
```bash
curl http://localhost:8080/v1/models
```

You should see a JSON response with available models.

### 4. Launch ces-voice

```bash
./launch.sh
```

Or with custom model:
```bash
LLM_MODEL="gpt-oss-20b-mxfp4" ./launch.sh
```

## Configuration Options

All LLM configuration can be set via environment variables:

```bash
export LLM_SERVER_URL="http://localhost:8080/v1/chat/completions"  # LLM server endpoint
export LLM_MODEL="gpt-oss-20b-mxfp4"                                 # Model name
export LLM_TEMP="0.7"                                                 # Temperature (0.0-2.0)
export LLM_MAX_TOKENS="256"                                           # Max tokens per response
export LLM_REASONING_EFFORT="low"                                      # For reasoning models: "low", "medium", "high", or "off"
```

## Troubleshooting

### LLM server not responding
- Verify llama-server is running: `curl http://localhost:8080/v1/models`
- Check the port matches: default is `8080`
- Check firewall/network settings

### Wrong model name
- Check what models llama-server reports: `curl http://localhost:8080/v1/models`
- Set `LLM_MODEL` to match one of the reported model names

### Connection errors
- Ensure llama-server is started before ces-voice
- Check that both are on the same machine or network
- Verify the URL format: `http://host:port/v1/chat/completions`

