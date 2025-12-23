import os
import uuid
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np
import torch
import soundfile as sf
import aiohttp
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from faster_whisper import WhisperModel
from kokoro import KPipeline


# -----------------------------
# Config
# -----------------------------

@dataclass
class ASRConfig:
    model_size: str = os.getenv("ASR_MODEL", "small.en")
    device: str = os.getenv("ASR_DEVICE", "cpu")  # override with ASR_DEVICE=cpu if needed
    compute_type: str = os.getenv("ASR_COMPUTE_TYPE", "int8")


@dataclass
class LLMConfig:
    base_url: str = os.getenv("LLM_SERVER_URL", "http://localhost:8080/v1/chat/completions")
    model: str = os.getenv("LLM_MODEL", "gpt-4x-local")
    temperature: float = float(os.getenv("LLM_TEMP", "0.7"))
    max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "256"))
    reasoning_effort: str = os.getenv("LLM_REASONING_EFFORT", "low")  # for GPT-OSS / reasoning models


@dataclass
class TTSConfig:
    lang_code: str = os.getenv("KOKORO_LANG", "a")          # 'a' = American English
    voice: str = os.getenv("KOKORO_VOICE", "af_nicole")
    speed: float = float(os.getenv("KOKORO_SPEED", "1.1"))  # slightly faster


AUDIO_DIR = Path("audio_cache")
AUDIO_DIR.mkdir(exist_ok=True)

SAMPLE_RATE = 16000


# -----------------------------
# ASR
# -----------------------------

class FasterWhisperASR:
    def __init__(self, cfg: ASRConfig):
        print(f"[ASR] Loading faster-whisper model '{cfg.model_size}' "
              f"on {cfg.device} ({cfg.compute_type})...")
        self.model = WhisperModel(
            cfg.model_size,
            device=cfg.device,
            compute_type=cfg.compute_type,
        )

    def transcribe(self, audio: np.ndarray) -> str:
        if audio.size == 0:
            return ""
        segments, info = self.model.transcribe(
            audio,
            beam_size=1,
            language="en",
            word_timestamps=False,
            vad_filter=True,
        )
        text = "".join([seg.text for seg in segments]).strip()
        return text


def decode_webm_to_pcm_f32(input_path: Path, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Non-streaming path: decode a single .webm file to float32 mono PCM.
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-ac", "1",
        "-ar", str(target_sr),
        "-f", "f32le",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        print("[ffmpeg stderr]\n", proc.stderr.decode("utf-8", errors="ignore"))
        raise RuntimeError("ffmpeg failed to decode audio")
    audio = np.frombuffer(proc.stdout, dtype=np.float32)
    return audio


def decode_webm_bytes_to_pcm_f32(data: bytes, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Streaming path: decode a growing audio/webm stream (as bytes) to float32 mono PCM using ffmpeg.
    We feed the entire accumulated WebM blob via stdin each time.
    """
    if not data:
        return np.zeros(0, dtype=np.float32)

    cmd = [
        "ffmpeg",
        "-y",
        "-i", "pipe:0",        # read from stdin
        "-ac", "1",
        "-ar", str(target_sr),
        "-f", "f32le",
        "pipe:1",
    ]
    proc = subprocess.run(
        cmd,
        input=data,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        print("[ffmpeg stderr]\n", proc.stderr.decode("utf-8", errors="ignore"))
        raise RuntimeError("ffmpeg failed to decode audio from bytes")
    audio = np.frombuffer(proc.stdout, dtype=np.float32)
    return audio


# -----------------------------
# LLM client
# -----------------------------

class LlamaCppClient:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg

    def _extract_final_channel(self, content: str) -> str:
        """
        For reasoning-style outputs like:
        <|channel|>analysis<|message|>...<|end|><|start|>assistant
        <|channel|>final<|message|>...<|end|>
        return only the 'final' message content.
        """
        if "<|channel|>final<|message|>" in content:
            content = content.split("<|channel|>final<|message|>", 1)[1]
        if "<|end|>" in content:
            content = content.split("<|end|>", 1)[0]
        return content.strip()

    async def complete(self, messages: List[Dict[str, Any]]) -> str:
        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
            "stream": False,
        }
        if self.cfg.reasoning_effort:
            payload["reasoning_effort"] = self.cfg.reasoning_effort

        async with aiohttp.ClientSession() as session:
            async with session.post(self.cfg.base_url, json=payload) as resp:
                data = await resp.json()

        raw = data["choices"][0]["message"]["content"]
        return self._extract_final_channel(raw)


# -----------------------------
# Kokoro TTS
# -----------------------------

class KokoroTTS:
    def __init__(self, cfg: TTSConfig):
        print(f"[TTS] Loading Kokoro pipeline (lang={cfg.lang_code}, voice={cfg.voice})...")
        self.cfg = cfg
        self.pipeline = KPipeline(lang_code=cfg.lang_code)

    def synth_to_file(self, text: str, out_path: Path) -> None:
        if not text.strip():
            sf.write(str(out_path), np.zeros(1600, dtype=np.float32), 16000)
            return

        generator = self.pipeline(
            text,
            voice=self.cfg.voice,
            speed=self.cfg.speed,
            split_pattern=r"\n+",
        )

        chunks = []
        for _, _, audio in generator:
            if isinstance(audio, torch.Tensor):
                audio = audio.detach().cpu().numpy()
            audio = audio.astype("float32")
            chunks.append(audio)

        if not chunks:
            sf.write(str(out_path), np.zeros(1600, dtype=np.float32), 16000)
            return

        audio = np.concatenate(chunks)
        sr = 24000
        sf.write(str(out_path), audio, sr, subtype="PCM_16")


# -----------------------------
# FastAPI app setup
# -----------------------------

app = FastAPI(title="Voice Chat (faster-whisper + llama.cpp + Kokoro)")

asr = FasterWhisperASR(ASRConfig())
llm = LlamaCppClient(LLMConfig())
tts = KokoroTTS(TTSConfig())

conversation_history: List[Dict[str, str]] = [
    {
        "role": "system",
        "content": (
            "You are a concise, helpful voice assistant. "
            "Answer in 1–2 short sentences, no internal reasoning or metadata in your reply."
        ),
    }
]

app.mount("/audio", StaticFiles(directory=str(AUDIO_DIR)), name="audio")


# -----------------------------
# HTML Frontend (SPACE hold + streaming ASR)
# -----------------------------

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8" />
  <title>Voice Chat (Whisper + Llama.cpp + Kokoro)</title>
  <style>
    body { font-family: sans-serif; max-width: 700px; margin: 2rem auto; }
    button { padding: 0.6rem 1rem; margin: 0.5rem 0; }
    #log { white-space: pre-wrap; background: #111; color: #eee; padding: 1rem; border-radius: 6px; }
    #user, #assistant { margin-top: 1rem; }
    .hint { color: #555; font-size: 0.9rem; }
  </style>
</head>
<body>
  <h1>Voice Chat (Streaming ASR)</h1>
  <p>
    Click <strong>Start Recording</strong> / <strong>Stop</strong>, or
    <strong>hold the SPACE bar</strong> to record.<br/>
    While you talk, partial ASR will stream in live. When you stop,
    the final transcript is sent to the LLM + TTS.
  </p>

  <button id="startBtn">Start Recording</button>
  <button id="stopBtn" disabled>Stop</button>
  <div class="hint">Tip: hold SPACE to talk, release to send.</div>

  <div id="status"></div>

  <div id="user">
    <h3>You said (live ASR):</h3>
    <div id="userText"></div>
  </div>

  <div id="assistant">
    <h3>Assistant:</h3>
    <div id="assistantText"></div>
    <audio id="assistantAudio" controls></audio>
  </div>

  <h3>Log</h3>
  <div id="log"></div>

<script>
let mediaRecorder = null;
let isRecording = false;
let ws = null;
let finalASRText = "";

const logEl = document.getElementById("log");
const statusEl = document.getElementById("status");
const userTextEl = document.getElementById("userText");
const assistantTextEl = document.getElementById("assistantText");
const assistantAudioEl = document.getElementById("assistantAudio");
const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");

function log(msg) {
  console.log(msg);
  logEl.textContent += msg + "\\n";
}

async function callLLMAndTTS(userText) {
  try {
    statusEl.textContent = "Thinking...";
    const resp = await fetch("/api/llm_tts", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user_text: userText }),
    });
    if (!resp.ok) {
      const text = await resp.text();
      log("LLM/TTS server error: " + text);
      statusEl.textContent = "LLM/TTS server error.";
      return;
    }
    const data = await resp.json();
    statusEl.textContent = "Done.";
    userTextEl.textContent = data.user_text || userText;
    assistantTextEl.textContent = data.assistant_text || "";

    if (data.audio_url) {
      assistantAudioEl.src = data.audio_url;
      assistantAudioEl.play().catch(e => log("Audio play error: " + e));
    }
  } catch (err) {
    log("Error calling LLM/TTS: " + err);
    statusEl.textContent = "Error calling LLM/TTS.";
  }
}

async function startRecording() {
  if (isRecording) return;

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    statusEl.textContent = "This page cannot access your microphone. " +
      "Open it via https or http://localhost instead.";
    log("MediaDevices.getUserMedia not available.");
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

    const wsProtocol = (location.protocol === "https:") ? "wss://" : "ws://";
    ws = new WebSocket(wsProtocol + location.host + "/ws/asr");

    ws.onopen = () => {
      log("ASR WebSocket connected.");
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      log("WS message: " + JSON.stringify(data));
      if (data.type === "partial") {
        userTextEl.textContent = data.text;
      } else if (data.type === "final") {
        finalASRText = data.text || "";
        userTextEl.textContent = finalASRText;
        statusEl.textContent = "ASR done. Calling LLM/TTS...";
        if (finalASRText.trim().length > 0) {
          callLLMAndTTS(finalASRText);
        } else {
          statusEl.textContent = "No speech detected.";
        }
      } else if (data.type === "error") {
        statusEl.textContent = "ASR error: " + data.error;
        log("ASR error: " + data.error);
      }
    };

    ws.onerror = (e) => {
      log("WebSocket error: " + e);
    };

    ws.onclose = () => {
      log("ASR WebSocket closed.");
    };

    // Don't force a mimeType; let the browser pick a supported one.
    mediaRecorder = new MediaRecorder(stream);

    mediaRecorder.ondataavailable = (e) => {
      log("ondataavailable, size=" + e.data.size);
      if (e.data.size > 0) {
        if (ws && ws.readyState === WebSocket.OPEN) {
          ws.send(e.data);
          log("Sent chunk over WebSocket");
        } else {
          log("WebSocket not open; readyState=" + (ws ? ws.readyState : "null"));
        }
      }
    };

    mediaRecorder.onstop = () => {
      log("MediaRecorder stopped.");
    };

    mediaRecorder.start(250); // ms timeslice
    isRecording = true;
    statusEl.textContent = "Recording (streaming ASR)...";
    startBtn.disabled = true;
    stopBtn.disabled = false;
    log("Recording started (streaming).");
  } catch (err) {
    log("Could not start recording: " + err);
  }
}

function stopRecording() {
  if (!isRecording) return;

  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
    statusEl.textContent = "Stopping...";
    log("Recording stopped.");
  }

  isRecording = false;
  startBtn.disabled = false;
  stopBtn.disabled = true;

  if (ws && ws.readyState === WebSocket.OPEN) {
    log("Sending 'end' to ASR WebSocket");
    ws.send("end");
  } else {
    log("WebSocket already closed or not open on stop.");
  }
}

startBtn.onclick = () => { startRecording(); };
stopBtn.onclick = () => { stopRecording(); };

document.addEventListener("keydown", (e) => {
  if (e.code === "Space" && !isRecording) {
    e.preventDefault();
    startRecording();
  }
});
document.addEventListener("keyup", (e) => {
  if (e.code === "Space" && isRecording) {
    e.preventDefault();
    stopRecording();
  }
});
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE


# -----------------------------
# Legacy non-streaming endpoint (still useful for debugging)
# -----------------------------

@app.post("/api/voice_chat")
async def voice_chat(audio: UploadFile = File(...)):
    tmp_id = uuid.uuid4().hex
    tmp_webm = AUDIO_DIR / f"{tmp_id}.webm"
    with open(tmp_webm, "wb") as f:
        f.write(await audio.read())

    try:
        pcm = decode_webm_to_pcm_f32(tmp_webm, target_sr=SAMPLE_RATE)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Failed to decode audio: {e}"},
        )
    finally:
        try:
            tmp_webm.unlink()
        except FileNotFoundError:
            pass

    user_text = asr.transcribe(pcm)
    if not user_text:
        return {"user_text": "", "assistant_text": "", "audio_url": ""}

    conversation_history.append({"role": "user", "content": user_text})
    assistant_text = await llm.complete(conversation_history)
    conversation_history.append({"role": "assistant", "content": assistant_text})

    out_wav = AUDIO_DIR / f"{tmp_id}.wav"
    tts.synth_to_file(assistant_text, out_wav)
    audio_url = f"/audio/{out_wav.name}"

    return {
        "user_text": user_text,
        "assistant_text": assistant_text,
        "audio_url": audio_url,
    }


# -----------------------------
# Optimized path: LLM + TTS only
# -----------------------------

@app.post("/api/llm_tts")
async def llm_tts(payload: Dict[str, Any]):
    user_text = (payload or {}).get("user_text", "") or ""
    user_text = user_text.strip()

    if not user_text:
        return {"user_text": "", "assistant_text": "", "audio_url": ""}

    tmp_id = uuid.uuid4().hex

    conversation_history.append({"role": "user", "content": user_text})
    assistant_text = await llm.complete(conversation_history)
    conversation_history.append({"role": "assistant", "content": assistant_text})

    out_wav = AUDIO_DIR / f"{tmp_id}.wav"
    tts.synth_to_file(assistant_text, out_wav)
    audio_url = f"/audio/{out_wav.name}"

    return {
        "user_text": user_text,
        "assistant_text": assistant_text,
        "audio_url": audio_url,
    }


# -----------------------------
# Streaming ASR over WebSockets
# -----------------------------

class StreamingASRSession:
    """
    Holds WebM bytes + PCM buffer for one WebSocket ASR session.
    We accumulate WebM bytes as they arrive, and whenever we want a partial
    transcript we re-decode the *full* WebM blob with ffmpeg and run ASR.
    """
    def __init__(self):
        self.webm_bytes = bytearray()
        self.pcm = np.zeros(0, dtype=np.float32)
        self.last_text = ""

    def append_chunk(self, chunk: bytes):
        if not chunk:
            return
        self.webm_bytes.extend(chunk)

    def decode_to_pcm(self):
        if not self.webm_bytes:
            self.pcm = np.zeros(0, dtype=np.float32)
            return
        self.pcm = decode_webm_bytes_to_pcm_f32(bytes(self.webm_bytes), target_sr=SAMPLE_RATE)


MIN_AUDIO_SECONDS = 0.5  # only start ASR after ~0.5 s of audio


@app.websocket("/ws/asr")
async def asr_stream(websocket: WebSocket):
    await websocket.accept()
    session = StreamingASRSession()

    try:
        while True:
            msg = await websocket.receive()

            if msg["type"] == "websocket.disconnect":
                print("[Streaming ASR] client disconnected")
                break

            # Binary = audio chunk
            if msg.get("bytes") is not None:
                chunk_bytes = msg["bytes"]
                session.append_chunk(chunk_bytes)
                print(f"[Streaming ASR] got chunk: {len(chunk_bytes)} bytes, total={len(session.webm_bytes)}")

                # small guard: wait for some bytes before decoding
                if len(session.webm_bytes) < 4000:
                    continue

                try:
                    session.decode_to_pcm()
                except Exception as e:
                    print("[Streaming ASR] decode error:", e)
                    await websocket.send_json({"type": "error", "error": str(e)})
                    continue

                if session.pcm.size < int(SAMPLE_RATE * MIN_AUDIO_SECONDS):
                    continue

                partial_text = asr.transcribe(session.pcm)
                print(f"[Streaming ASR] partial: '{partial_text}'")

                if partial_text and partial_text != session.last_text:
                    session.last_text = partial_text
                    await websocket.send_json({"type": "partial", "text": partial_text})

            # Text = control messages ("end")
            elif msg.get("text") is not None:
                text = msg["text"]
                print("[Streaming ASR] text message:", text)
                if text == "end":
                    try:
                        session.decode_to_pcm()
                        final_text = asr.transcribe(session.pcm) if session.pcm.size > 0 else ""
                    except Exception as e:
                        print("[Streaming ASR] final decode error:", e)
                        await websocket.send_json({"type": "error", "error": str(e)})
                        final_text = ""

                    print(f"[Streaming ASR] final: '{final_text}'")
                    await websocket.send_json({"type": "final", "text": final_text})
                    await websocket.close()
                    break

    except WebSocketDisconnect:
        print("[Streaming ASR] WebSocketDisconnect")
    except Exception as e:
        print("[Streaming ASR] unexpected error:", e)
        try:
            await websocket.send_json({"type": "error", "error": str(e)})
        except Exception:
            pass
        await websocket.close()

