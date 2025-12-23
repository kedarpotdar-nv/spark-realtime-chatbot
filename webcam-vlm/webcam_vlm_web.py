#!/usr/bin/env python3
"""
Web-based Webcam to VLM (Qwen3-VL) Streaming Test
Uses BROWSER webcam (client-side) - works over HTTPS.

Requirements:
    pip install flask requests

Usage:
    # Generate self-signed cert for HTTPS (required for webcam access)
    openssl req -x509 -newkey rsa:4096 -keyout key.pem -out cert.pem -days 365 -nodes -subj '/CN=localhost'
    
    # Run with HTTPS
    python webcam_vlm_web.py --port 5000 --vlm-server http://localhost:8080 --https
    
    Then open browser to: https://spark:5000 (accept the self-signed cert warning)
"""

import base64
import json
import threading
import time
import argparse
import ssl
import os
import requests
from flask import Flask, render_template_string, request, jsonify

app = Flask(__name__)

# Global state
vlm_server = "http://localhost:8080"
analyzing = False

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Webcam → Qwen3-VL</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #e0e0e0;
            min-height: 100vh;
            padding: 20px;
        }
        h1 {
            text-align: center;
            color: #00d4ff;
            margin-bottom: 20px;
            font-size: 1.8em;
        }
        .container {
            display: flex;
            gap: 20px;
            max-width: 1400px;
            margin: 0 auto;
            flex-wrap: wrap;
        }
        .panel {
            background: rgba(15, 15, 35, 0.8);
            border-radius: 12px;
            padding: 20px;
            border: 1px solid #333;
        }
        .video-panel {
            flex: 1;
            min-width: 400px;
        }
        .response-panel {
            flex: 1;
            min-width: 400px;
        }
        .panel-title {
            font-size: 1.1em;
            color: #00d4ff;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        #video-container {
            position: relative;
            width: 100%;
            max-width: 640px;
        }
        #webcam {
            width: 100%;
            border-radius: 8px;
            background: #000;
            transform: scaleX(-1); /* Mirror effect */
        }
        #canvas {
            display: none;
        }
        .controls {
            margin-top: 15px;
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
            align-items: center;
        }
        button {
            padding: 10px 20px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 600;
            font-size: 0.95em;
            transition: all 0.2s;
        }
        button:hover { transform: translateY(-1px); }
        button:active { transform: translateY(0); }
        .btn-start { background: #2ecc71; color: white; }
        .btn-start:hover { background: #27ae60; }
        .btn-analyze { background: #9b59b6; color: white; }
        .btn-analyze:hover { background: #8e44ad; }
        .btn-analyze:disabled { background: #555; cursor: not-allowed; }
        .btn-clear { background: #495057; color: white; }
        .btn-clear:hover { background: #5a6268; }
        .prompt-input {
            width: 100%;
            padding: 12px;
            margin-top: 15px;
            border: 1px solid #444;
            border-radius: 6px;
            background: #1a1a2e;
            color: #e0e0e0;
            font-size: 1em;
        }
        .prompt-input:focus {
            outline: none;
            border-color: #00d4ff;
        }
        #response-area {
            width: 100%;
            height: 400px;
            padding: 15px;
            background: #0a0a1a;
            border: 1px solid #333;
            border-radius: 8px;
            color: #e0e0e0;
            font-family: 'Consolas', monospace;
            font-size: 0.95em;
            overflow-y: auto;
            white-space: pre-wrap;
            word-wrap: break-word;
        }
        .status {
            margin-top: 10px;
            font-size: 0.85em;
            color: #6c757d;
        }
        .status.error { color: #e74c3c; }
        .status.success { color: #2ecc71; }
        .status.analyzing { color: #f39c12; }
        .continuous-label {
            display: flex;
            align-items: center;
            gap: 6px;
            color: #adb5bd;
            font-size: 0.9em;
        }
        .timestamp {
            color: #00d4ff;
            font-weight: bold;
        }
        .camera-status {
            padding: 8px 12px;
            border-radius: 6px;
            margin-bottom: 10px;
            font-size: 0.9em;
        }
        .camera-status.waiting { background: #f39c12; color: #000; }
        .camera-status.active { background: #2ecc71; color: #000; }
        .camera-status.error { background: #e74c3c; color: #fff; }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .analyzing-indicator {
            animation: pulse 1s infinite;
        }
    </style>
</head>
<body>
    <h1>📹 Webcam → 🤖 Qwen3-VL</h1>
    
    <div class="container">
        <div class="panel video-panel">
            <div class="panel-title">📷 Your Camera (Browser Webcam)</div>
            
            <div id="camera-status" class="camera-status waiting">
                ⏳ Click "Start Camera" to begin
            </div>
            
            <div id="video-container">
                <video id="webcam" autoplay playsinline></video>
                <canvas id="canvas"></canvas>
            </div>
            
            <div class="controls">
                <button class="btn-start" id="startBtn" onclick="startCamera()">
                    📷 Start Camera
                </button>
                <button class="btn-analyze" id="analyzeBtn" onclick="analyzeFrame()" disabled>
                    🔍 Analyze Frame
                </button>
                <label class="continuous-label">
                    <input type="checkbox" id="continuousMode" onchange="toggleContinuous()">
                    Continuous (every 3s)
                </label>
            </div>
            
            <input type="text" class="prompt-input" id="prompt" 
                   value="What do you see?"
                   placeholder="Enter your prompt...">
            
            <div class="status" id="status">Ready. Start camera and click "Analyze Frame".</div>
        </div>
        
        <div class="panel response-panel">
            <div class="panel-title">
                🤖 Model Response
                <button class="btn-clear" onclick="clearResponses()">Clear</button>
            </div>
            <div id="response-area"></div>
            <div class="status">VLM Server: {{ vlm_server }}</div>
        </div>
    </div>

    <script>
        let video = document.getElementById('webcam');
        let canvas = document.getElementById('canvas');
        let ctx = canvas.getContext('2d');
        let stream = null;
        let continuousInterval = null;
        let isAnalyzing = false;
        let cameraReady = false;
        
        async function startCamera() {
            const statusEl = document.getElementById('camera-status');
            const startBtn = document.getElementById('startBtn');
            const analyzeBtn = document.getElementById('analyzeBtn');
            
            statusEl.className = 'camera-status waiting';
            statusEl.textContent = '⏳ Checking camera access...';
            
            // Debug info
            console.log('Protocol:', location.protocol);
            console.log('Hostname:', location.hostname);
            console.log('mediaDevices:', navigator.mediaDevices);
            
            // Check if HTTPS or localhost
            if (location.protocol !== 'https:' && location.hostname !== 'localhost' && location.hostname !== '127.0.0.1') {
                statusEl.className = 'camera-status error';
                statusEl.innerHTML = '❌ HTTPS required!<br>Current URL: ' + location.href + '<br>Please use https:// instead of http://';
                return;
            }
            
            // Check if mediaDevices is available
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                statusEl.className = 'camera-status error';
                statusEl.innerHTML = '❌ Camera API not available.<br>Protocol: ' + location.protocol + '<br>Try: Open browser settings and allow camera for this site.';
                return;
            }
            
            statusEl.textContent = '⏳ Requesting camera permission...';
            
            try {
                stream = await navigator.mediaDevices.getUserMedia({
                    video: { 
                        width: { ideal: 512 },  // Reduced for faster processing
                        height: { ideal: 384 },
                        facingMode: 'user'
                    },
                    audio: false
                });
                
                video.srcObject = stream;
                await video.play();
                
                // Set canvas size to match video
                canvas.width = video.videoWidth || 640;
                canvas.height = video.videoHeight || 480;
                
                cameraReady = true;
                statusEl.className = 'camera-status active';
                statusEl.textContent = '✓ Camera active';
                startBtn.textContent = '📷 Camera Running';
                startBtn.disabled = true;
                analyzeBtn.disabled = false;
                
                document.getElementById('status').textContent = 'Camera ready. Click "Analyze Frame" to send to VLM.';
                
            } catch (err) {
                console.error('Camera error:', err);
                statusEl.className = 'camera-status error';
                
                if (err.name === 'NotAllowedError') {
                    statusEl.innerHTML = '❌ Camera access denied.<br>Click the camera icon in the address bar to allow access.';
                } else if (err.name === 'NotFoundError') {
                    statusEl.textContent = '❌ No camera found on this device.';
                } else if (err.name === 'NotReadableError') {
                    statusEl.textContent = '❌ Camera is in use by another application.';
                } else {
                    statusEl.innerHTML = '❌ Camera error: ' + err.message + '<br>Protocol: ' + location.protocol;
                }
            }
        }
        
        function captureFrame() {
            if (!cameraReady) return null;
            
            // Draw video frame to canvas
            ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
            
            // Get base64 JPEG (lower quality = smaller size = faster upload)
            return canvas.toDataURL('image/jpeg', 0.6).split(',')[1];
        }
        
        async function analyzeFrame() {
            if (!cameraReady || isAnalyzing) return;
            
            isAnalyzing = true;
            const btn = document.getElementById('analyzeBtn');
            const status = document.getElementById('status');
            const prompt = document.getElementById('prompt').value;
            
            btn.disabled = true;
            btn.innerHTML = '⏳ Analyzing...';
            status.className = 'status analyzing';
            status.innerHTML = '<span class="analyzing-indicator">🔄 Sending frame to Qwen3-VL...</span>';
            
            try {
                const imageBase64 = captureFrame();
                if (!imageBase64) {
                    throw new Error('Failed to capture frame');
                }
                
                const response = await fetch('/analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ 
                        prompt: prompt,
                        image: imageBase64
                    })
                });
                
                const data = await response.json();
                
                const responseArea = document.getElementById('response-area');
                const timestamp = new Date().toLocaleTimeString();
                
                if (data.success) {
                    responseArea.innerHTML += `<span class="timestamp">─── ${timestamp} ───</span>\n${data.response}\n\n`;
                    status.className = 'status success';
                    status.textContent = 'Analysis complete.';
                } else {
                    responseArea.innerHTML += `<span class="timestamp">─── ${timestamp} ───</span>\n❌ Error: ${data.error}\n\n`;
                    status.className = 'status error';
                    status.textContent = 'Error: ' + data.error;
                }
                
                responseArea.scrollTop = responseArea.scrollHeight;
                
            } catch (err) {
                document.getElementById('response-area').innerHTML += `❌ Request failed: ${err}\n\n`;
                status.className = 'status error';
                status.textContent = 'Request failed.';
            }
            
            btn.disabled = false;
            btn.innerHTML = '🔍 Analyze Frame';
            isAnalyzing = false;
        }
        
        function toggleContinuous() {
            const enabled = document.getElementById('continuousMode').checked;
            
            if (enabled && cameraReady) {
                analyzeFrame();
                continuousInterval = setInterval(analyzeFrame, 3000);
            } else {
                if (continuousInterval) {
                    clearInterval(continuousInterval);
                    continuousInterval = null;
                }
            }
        }
        
        function clearResponses() {
            document.getElementById('response-area').innerHTML = '';
        }
        
        // Auto-start camera on page load (will prompt user)
        // Uncomment if you want auto-start:
        // window.onload = startCamera;
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE, vlm_server=vlm_server)


@app.route('/analyze', methods=['POST'])
def analyze():
    global analyzing
    
    if analyzing:
        return jsonify({"success": False, "error": "Already analyzing"})
    
    analyzing = True
    
    try:
        data = request.json
        prompt = data.get('prompt', 'Describe what you see.')
        img_base64 = data.get('image')
        
        if not img_base64:
            return jsonify({"success": False, "error": "No image provided"})
        
        # Try OpenAI-compatible endpoint
        payload = {
            "model": "qwen3-vl",
            "messages": [
                {
                    "role": "system",
                    "content": "You are a concise visual assistant. Respond in 1-3 short sentences. Be direct and specific. No unnecessary details or explanations."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_base64}"
                            }
                        },
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            "max_tokens": 150,  # Reduced for faster response
            "temperature": 0.3,  # Lower = more deterministic/faster
            "stream": False
        }
        
        response = requests.post(
            f"{vlm_server}/v1/chat/completions",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]
                return jsonify({"success": True, "response": content})
            else:
                return jsonify({"success": False, "error": f"Unexpected response: {result}"})
        else:
            # Try legacy llama.cpp endpoint
            return try_legacy_endpoint(img_base64, prompt)
            
    except requests.exceptions.ConnectionError:
        return jsonify({"success": False, "error": f"Cannot connect to {vlm_server}"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)})
    finally:
        analyzing = False


def try_legacy_endpoint(img_base64, prompt):
    """Try legacy llama.cpp /completion endpoint"""
    try:
        payload = {
            "prompt": f"[img-1]Be concise. {prompt}",
            "image_data": [{"data": img_base64, "id": 1}],
            "n_predict": 150,  # Reduced for faster response
            "temperature": 0.3,
            "stream": False
        }
        
        response = requests.post(
            f"{vlm_server}/completion",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=120
        )
        
        if response.status_code == 200:
            result = response.json()
            content = result.get("content", result.get("response", str(result)))
            return jsonify({"success": True, "response": content})
        else:
            return jsonify({"success": False, "error": f"Server error {response.status_code}: {response.text[:200]}"})
            
    except Exception as e:
        return jsonify({"success": False, "error": f"Legacy endpoint error: {str(e)}"})


def generate_self_signed_cert():
    """Generate self-signed certificate if not exists"""
    cert_file = 'cert.pem'
    key_file = 'key.pem'
    
    if not os.path.exists(cert_file) or not os.path.exists(key_file):
        print("Generating self-signed SSL certificate...")
        os.system(f"openssl req -x509 -newkey rsa:2048 -keyout {key_file} -out {cert_file} -days 365 -nodes -subj '/CN=localhost' 2>/dev/null")
        print(f"✓ Generated {cert_file} and {key_file}")
    
    return cert_file, key_file


def main():
    global vlm_server
    
    parser = argparse.ArgumentParser(description="Web-based Webcam to VLM streaming (browser webcam)")
    parser.add_argument("--port", "-p", type=int, default=5000,
                       help="Web server port (default: 5000)")
    parser.add_argument("--vlm-server", "-v", default="http://localhost:8080",
                       help="llama.cpp VLM server URL (default: http://localhost:8080)")
    parser.add_argument("--host", default="0.0.0.0",
                       help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--https", action="store_true",
                       help="Enable HTTPS (required for browser webcam access)")
    args = parser.parse_args()
    
    vlm_server = args.vlm_server
    
    protocol = "https" if args.https else "http"
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║         Webcam → Qwen3-VL Web Interface                      ║
║         (Uses YOUR browser's webcam)                         ║
╠══════════════════════════════════════════════════════════════╣
║  Web UI:     {protocol}://<your-server>:{args.port}                     
║  VLM Server: {vlm_server}                      
╚══════════════════════════════════════════════════════════════╝
    """)
    
    if args.https:
        cert_file, key_file = generate_self_signed_cert()
        print(f"🔒 HTTPS enabled - accept the certificate warning in your browser")
        print(f"   Open: https://<server-ip>:{args.port}")
        
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(cert_file, key_file)
        
        app.run(host=args.host, port=args.port, threaded=True, debug=False, ssl_context=context)
    else:
        print("⚠️  Running without HTTPS - webcam may not work from remote browser!")
        print("   Use --https flag for remote access")
        print(f"   Open: http://<server-ip>:{args.port}")
        app.run(host=args.host, port=args.port, threaded=True, debug=False)


if __name__ == "__main__":
    main()
