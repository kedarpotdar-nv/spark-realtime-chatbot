# Webcam → VLM (Qwen3-VL)

Stream your browser's webcam to a Vision Language Model running on llama.cpp.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
# Default: VLM on localhost:8080, web UI on port 5000
./launch-https.sh

# Custom VLM server
VLM_SERVER=http://localhost:8080 ./launch-https.sh

# Custom port
PORT=5001 ./launch-https.sh
```

## Access

1. Open `https://<server-hostname>:5000` in your browser
2. Accept the self-signed certificate warning
3. Click **Start Camera** and allow camera access
4. Click **Analyze Frame** to send current frame to VLM

## Notes

- **HTTPS is required** for browser webcam access from remote machines
- Uses your **browser's webcam**, not a camera on the server
- Compatible with llama.cpp server running vision models (Qwen3-VL, LLaVA, etc.)

