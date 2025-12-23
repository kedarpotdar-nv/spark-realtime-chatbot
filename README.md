# Voice Chat Assistant - Streaming Edition

An improved voice-to-voice chatbot with **streaming input** (real-time ASR), **streaming output** (streaming LLM responses + TTS), and **agent capabilities** including a coding assistant.

## Features

- 🎤 **Streaming ASR**: Real-time speech-to-text transcription as you speak
- 💬 **Streaming LLM**: Server-Sent Events (SSE) for streaming text responses
- 🔊 **Streaming TTS**: Audio generation and playback using Kokoro TTS
- 🤖 **Coding Assistant**: AI agent that generates and executes code in a sandbox
- 🛠️ **Tool Calling**: Extensible tool system (weather, home assistant, etc.)
- 💾 **Chat History**: Persistent conversation history with local storage
- ⌨️ **Multiple Input Methods**: Voice input (hold SPACE) or text input
- 🎨 **Modern UI**: Clean, responsive frontend with agent modals

## Architecture

```
ces-voice/
├── server.py          # FastAPI backend with streaming support
├── static/
│   └── index.html    # Frontend UI (separated)
├── audio_cache/       # Generated audio files (auto-created)
├── requirements.txt   # Python dependencies
├── launch.sh          # Launch script (HTTP)
├── launch-https.sh    # Launch script (HTTPS)
└── main-final.py      # Original reference implementation
```

## Prerequisites

1. **Python 3.8+**
2. **ffmpeg** (for audio decoding)
   - Ubuntu/Debian: `sudo apt install ffmpeg`
   - macOS: `brew install ffmpeg`
   - Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html)
3. **Docker** (for code execution sandbox)
   - Required for Coding Assistant agent
   - Install from [docker.com](https://www.docker.com/get-started)
4. **External Services**:
   - **ASR Server**: OpenAI-compatible ASR API (e.g., faster-whisper server)
   - **LLM Server**: OpenAI-compatible LLM API (e.g., llama-server)

## Setup

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

**Key dependencies:**
- `fastapi`, `uvicorn[standard]`, `websockets` - Web framework and real-time communication
- `kokoro-tts`, `kokoro` - Text-to-speech
- `torch`, `numpy`, `soundfile` - Audio processing
- `llm-sandbox[docker]` - Secure code execution sandbox
- `aiohttp` - Async HTTP client

**Note:** `podman` and `kubernetes` are optional dependencies for `llm-sandbox` but may be needed depending on your setup.

### 2. Start External Services

#### ASR Server (Speech-to-Text)

You need an OpenAI-compatible ASR API server running. Example with faster-whisper:

```bash
# Start faster-whisper server on port 8000
# (Adjust command based on your faster-whisper setup)
```

Default configuration expects ASR at: `http://localhost:8000/v1/audio/transcriptions`

#### LLM Server (Language Model)

You need an OpenAI-compatible LLM API server running. Example with llama-server:

```bash
# Example: Start llama-server
./llama-server --model gpt-oss-120b-mxfp4-00001-of-00003.gguf --port 8080 -ngl 99
```

Default configuration expects LLM at: `http://localhost:8080/v1/chat/completions`

### 3. Configure Environment Variables (Optional)

Set these before launching (or use defaults):

```bash
# ASR Configuration
export ASR_API_URL="http://localhost:8000/v1/audio/transcriptions"
export ASR_API_KEY="dummy-key"  # Can be any non-empty string
export ASR_MODEL="tiny.en"      # Model name for ASR API

# LLM Configuration
export LLM_SERVER_URL="http://localhost:8080/v1/chat/completions"
export LLM_MODEL="gpt-4x-local"  # Must match your LLM server's model name
export LLM_MAX_TOKENS="4096"      # Max tokens for responses (increased for code generation)
export LLM_TEMP="0.7"             # Temperature (0.0-1.0)
export LLM_REASONING_EFFORT="low" # "low", "medium", "high", or "off"

# TTS Configuration
export KOKORO_LANG="a"            # Language code (a = American English)
export KOKORO_VOICE="af_bella"    # Voice name
export KOKORO_SPEED="1.2"         # Speech speed multiplier

# HuggingFace Cache Configuration
export HF_HOME="/home/nvidia/hfcache"  # HF cache directory (defaults to /home/nvidia/hfcache)
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"  # Hub cache location

# Server Configuration
export PORT="8001"                # Port for ces-voice server (default: 8001)
export FFMPEG_PATH="ffmpeg"       # Path to ffmpeg binary
```

### 4. Launch the Server

#### Option A: Using Launch Script (Recommended)

```bash
# HTTP (for localhost)
./launch.sh

# HTTPS (for remote access - generates self-signed cert if needed)
./launch-https.sh
```

#### Option B: Manual Launch

```bash
uvicorn server:app --host 0.0.0.0 --port 8001
```

### 5. Access the Application

- **Localhost**: `http://localhost:8001`
- **HTTPS**: `https://localhost:8443` (if using launch-https.sh)
- **Remote**: Use SSH port forwarding or HTTPS (see [DEPLOY.md](DEPLOY.md))

⚠️ **Important**: Browsers require **HTTPS** (or `localhost`) for microphone access!

## Usage

1. **Connect**: Click "Connect" button or wait for auto-connect
2. **Enable Tools**: Check boxes in "Capabilities" and "Agents" sections
3. **Input Methods**:
   - **Voice**: Hold SPACE bar to record, release to send
   - **Text**: Type in the text input box and click "Send"
4. **Coding Assistant**: Enable "Coding Assistant" checkbox, then ask to write code. A modal will open showing code generation and execution.

## Features in Detail

### Capabilities (Basic Tools)
- **Weather**: Get current weather for any location
- **Home Assistant**: Control smart home devices (extensible)

### Agents (Complex Tools)
- **Coding Assistant**: 
  - Generates code based on your request
  - Executes code in a secure Docker sandbox
  - Automatically fixes errors and retries
  - Shows code and execution output in a modal UI
  - Stores generated code in chat history

### Chat History
- Conversations are saved to browser local storage
- Switch between multiple chats using the left sidebar
- Chat history persists across browser sessions

## API Endpoints

### WebSocket: `/ws/voice`
Main WebSocket endpoint for voice interaction.

**Messages:**
- Binary: Audio chunks (WebM format)
- JSON: `{"type": "set_tools", "tools": ["weather", "coding_assistant"]}` - Enable/disable tools
- JSON: `{"type": "text_message", "text": "..."}` - Send text message
- JSON: `{"type": "disconnect"}` - Disconnect

**Responses:**
- `{"type": "asr_partial", "text": "..."}` - Live transcription
- `{"type": "asr_final", "text": "..."}` - Final transcript
- `{"type": "transient_response", "text": "..."}` - Streaming LLM response
- `{"type": "final_response", "text": "..."}` - Final LLM response
- `{"type": "agent_started", ...}` - Agent tool activated
- `{"type": "agent_code_chunk", ...}` - Code generation streaming
- `{"type": "agent_code_complete", ...}` - Code generation finished
- Binary: Audio chunks for TTS

### POST: `/api/llm_stream`
Streaming LLM endpoint using Server-Sent Events (SSE).

### POST: `/api/tts_stream`
Streaming TTS endpoint.

## Troubleshooting

### Microphone Not Working
- **Problem**: Browser blocks microphone access
- **Solution**: Use `https://` or `http://localhost`. See [HTTPS_SETUP.md](HTTPS_SETUP.md) for HTTPS setup.

### ASR Not Working
- **Problem**: No transcription appearing
- **Solution**: 
  - Verify ASR server is running: `curl http://localhost:8000/v1/audio/transcriptions`
  - Check `ASR_API_URL` environment variable
  - Ensure `ffmpeg` is installed and accessible

### LLM Errors
- **Problem**: "Connection refused" or "Model not found"
- **Solution**:
  - Verify LLM server is running: `curl http://localhost:8080/v1/models`
  - Check `LLM_SERVER_URL` and `LLM_MODEL` match your server configuration
  - See [LLM_SETUP.md](LLM_SETUP.md) for LLM server setup

### Code Execution Not Working
- **Problem**: "Code execution not available" message
- **Solution**:
  - Install `llm-sandbox`: `pip install 'llm-sandbox[docker]'`
  - Ensure Docker is running: `docker ps`
  - Check Docker has permission to create containers

### TTS Errors
- **Problem**: No audio playback
- **Solution**:
  - Check browser console for errors
  - Verify Kokoro models are loaded (check server logs)
  - Try different `KOKORO_VOICE` values

### Port Conflicts
- **Problem**: "Address already in use"
- **Solution**: Change `PORT` environment variable or stop conflicting service

## Additional Documentation

- [DEPLOY.md](DEPLOY.md) - Remote deployment guide
- [HTTPS_SETUP.md](HTTPS_SETUP.md) - HTTPS/SSL configuration
- [LLM_SETUP.md](LLM_SETUP.md) - LLM server setup guide
- [TOOL_TESTING.md](TOOL_TESTING.md) - Testing tools and agents

## Development

### Project Structure
- `server.py` - Main FastAPI application with WebSocket handlers
- `static/index.html` - Frontend UI (HTML/CSS/JavaScript)
- `requirements.txt` - Python dependencies
- `launch.sh` / `launch-https.sh` - Launch scripts with environment setup

### Key Components
- **VoiceSession**: Manages WebSocket connections and conversation state
- **FasterWhisperASR**: Handles ASR API communication
- **LlamaCppClient**: Handles LLM API communication with streaming
- **KokoroTTS**: Text-to-speech synthesis
- **Tool System**: Extensible tool calling framework
- **Agent System**: Complex tools with specialized UIs (e.g., Coding Assistant)

## Notes

- Conversation history is stored in browser `localStorage` (client-side)
- Audio files are cached in `audio_cache/` directory
- Code execution uses Docker containers via `llm-sandbox`
- Requires external ASR and LLM servers (not included)
- Microphone access requires HTTPS or localhost (browser security requirement)
