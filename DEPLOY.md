# Remote Deployment Guide for DGX Spark

## Quick Start

### 1. Configure Environment Variables

Set these before launching (adjust ports/hosts as needed):

```bash
# ASR (faster-whisper server) - adjust if running on different host/port
export ASR_API_URL="http://localhost:8000/v1/audio/transcriptions"
export ASR_API_KEY="dummy-key"
export ASR_MODEL="Systran/faster-distil-whisper-small.en"

# LLM server (if different from default)
export LLM_SERVER_URL="http://localhost:8080/v1/chat/completions"
export LLM_MODEL="gpt-4x-local"

# Example: If using llama-server with a specific model:
# Start LLM server: ./llama-server --model gpt-oss-20b-mxfp4.gguf -ngl 99 --port 8080
# Then set: export LLM_MODEL="gpt-oss-20b-mxfp4" (or whatever model name the server expects)

# TTS (Kokoro)
export KOKORO_LANG="a"
export KOKORO_VOICE="af_nicole"
```

### 2. Launch the Server

```bash
uvicorn server:app --host 0.0.0.0 --port 8001
```

**Note:** Using port `8001` to avoid conflict with faster-whisper server on `8000`. Adjust as needed.

### 3. Access Remotely

⚠️ **IMPORTANT:** Browsers require **HTTPS** (or `localhost`) for microphone access!

#### Option A: SSH Port Forwarding (Recommended - Easiest)
From your local machine:
```bash
ssh -L 8001:localhost:8001 user@spark-hostname
```

Then access locally (works without HTTPS):
```
http://localhost:8001
```

✅ **Best option** - No HTTPS setup needed, browser treats it as localhost

#### Option B: HTTPS with SSL Certificates
Use the HTTPS launch script:
```bash
./launch-https.sh
```

Then access:
```
https://<spark-hostname-or-ip>:8443
```

⚠️ If using self-signed cert, accept browser security warning

#### Option C: Direct HTTP Access (Microphone Won't Work)
```
http://<spark-hostname-or-ip>:8001
```

❌ **Microphone will NOT work** - browser will block access

#### Option D: Using Spark's Built-in HTTPS
If Spark provides HTTPS URLs or reverse proxy, use those.

## Important Configuration Notes

### Faster-Whisper Server Location

If your faster-whisper server is running on:
- **Same machine**: Use `http://localhost:8000/v1/audio/transcriptions`
- **Different machine**: Use `http://<whisper-host>:<port>/v1/audio/transcriptions`
- **Different port**: Update `ASR_API_URL` accordingly

Example if faster-whisper is on port 9000:
```bash
export ASR_API_URL="http://localhost:9000/v1/audio/transcriptions"
```

### Port Conflicts

If port 8000 is already used by faster-whisper, use a different port:
```bash
uvicorn server:app --host 0.0.0.0 --port 8001
```

### HTTPS/WSS Support

If accessing via HTTPS, WebSockets will automatically use `wss://`. For local development, HTTP/WS is fine.

## Troubleshooting

### Connection Issues
- Verify faster-whisper server is running: `curl http://localhost:8000/v1/models` (or your ASR_API_URL)
- Check firewall rules allow the port
- Verify `--host 0.0.0.0` is set (not `127.0.0.1`)

### ASR Not Working
- Check `ASR_API_URL` points to correct faster-whisper server
- Verify faster-whisper server accepts the model name
- Check server logs for API errors

### WebSocket Connection Fails
- Ensure port forwarding is set up correctly
- Check browser console for connection errors
- Verify the server is accessible from your browser's network

## Production Considerations

For production deployment, consider:
- Using a reverse proxy (nginx/traefik) for HTTPS
- Setting up proper authentication
- Using environment-specific config files
- Monitoring and logging

