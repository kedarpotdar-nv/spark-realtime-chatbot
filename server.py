"""spark-realtime-chatbot - Real-time Voice Chatbot Server.

A WebSocket-based voice assistant using:
- Faster-Whisper for ASR
- llama.cpp or TensorRT-LLM for LLM
- Kokoro for TTS
"""

import argparse
import asyncio
import json
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, File, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.websockets import WebSocketState

# Local modules
from config import (
    ASRConfig, LLMConfig, VLMConfig, NemotronConfig, TTSConfig,
    AUDIO_DIR, STATIC_DIR, SAMPLE_RATE, WORKSPACE_ROOT, FFMPEG_PATH
)
from audio import check_ffmpeg_available, decode_webm_bytes_to_pcm_f32
from tools import get_enabled_tools, execute_tool
from clients import (
    HTTPSessionManager,
    create_asr,
    LlamaCppClient,
    VLMClient,
    NemotronClient,
    KokoroTTS,
)
from clients.http_session import set_http_manager

# Import system prompt
from prompts import DEFAULT_SYSTEM_PROMPT


# -----------------------------
# FastAPI app setup
# -----------------------------

# Global models (initialized at startup)
asr = None  # FasterWhisperASR or LocalWhisperASR based on ASR_MODE
llm: LlamaCppClient = None
tts: KokoroTTS = None

# Conversation history (in-memory, per-session could be added later)
conversation_history: List[Dict[str, str]] = [
    {
        "role": "system",
        "content": (
            "You are a concise, helpful voice assistant. "
            "Answer in 1–2 short sentences, no internal reasoning or metadata in your reply."
        ),
    }
]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize models on startup."""
    global asr, llm, tts

    # Initialize shared HTTP session manager first (used by all clients)
    http_manager = HTTPSessionManager()
    set_http_manager(http_manager)

    # Check ffmpeg availability at startup
    if not check_ffmpeg_available():
        print(f"⚠️  WARNING: ffmpeg not found at '{FFMPEG_PATH}'")
        print(f"   Audio decoding will fail. Install ffmpeg:")
        print(f"   - Ubuntu/Debian: sudo apt install ffmpeg")
        print(f"   - macOS: brew install ffmpeg")
        print(f"   - Or set FFMPEG_PATH environment variable to ffmpeg location")
    else:
        print(f"✅ ffmpeg found at '{FFMPEG_PATH}'")

    asr = create_asr(ASRConfig())
    # Warmup ASR model to eliminate cold-start latency
    if hasattr(asr, 'warmup'):
        asr.warmup()
    llm = LlamaCppClient(LLMConfig())
    tts = KokoroTTS(TTSConfig())
    yield
    # Cleanup: close shared HTTP session
    if http_manager:
        await http_manager.close()


# Parse command-line arguments before creating app
def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='CES Voice Assistant Server')
    parser.add_argument('--trtllm', action='store_true', 
                       help='Use TensorRT-LLM backend (trtllm-serve) instead of standard OpenAI-compatible backend')
    # Only parse known args to avoid conflicts with uvicorn
    args, unknown = parser.parse_known_args()
    return args

# Check for LLM_BACKEND environment variable (set by launch scripts)
# This is the preferred method since uvicorn doesn't pass args to the app
if os.getenv("LLM_BACKEND") == "trtllm":
    print(f"[Server] TensorRT-LLM backend enabled via LLM_BACKEND environment variable")

app = FastAPI(title="Voice Chat (Streaming)", lifespan=lifespan)

# Serve static files (frontend)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/audio", StaticFiles(directory=str(AUDIO_DIR)), name="audio")


# -----------------------------
# Routes
# -----------------------------

@app.get("/health")
async def health_check():
    """Health check endpoint for Docker."""
    return {"status": "ok"}


@app.get("/api/default_prompt")
async def get_default_prompt():
    """Get the default system prompt from prompts.py."""
    from prompts import DEFAULT_SYSTEM_PROMPT
    return {"prompt": DEFAULT_SYSTEM_PROMPT}


# -----------------------------
# Face Recognition API
# -----------------------------

@app.post("/api/face/enroll")
async def enroll_face(request: Request):
    """Enroll a new face for recognition.

    Body: {"name": "Person Name", "image": "base64_image_data"}
    """
    try:
        from clients.face import get_face_recognizer
        data = await request.json()
        name = data.get("name")
        image_b64 = data.get("image")

        if not name or not image_b64:
            return {"success": False, "error": "Missing name or image"}

        recognizer = get_face_recognizer()
        success = recognizer.enroll_face(name, image_b64)

        if success:
            return {"success": True, "message": f"Enrolled {name}"}
        else:
            return {"success": False, "error": "No face detected in image"}
    except Exception as e:
        print(f"[Face API] Enroll error: {e}")
        return {"success": False, "error": str(e)}


@app.post("/api/face/recognize")
async def recognize_faces(request: Request):
    """Recognize faces in an image.

    Body: {"image": "base64_image_data"}
    """
    try:
        from clients.face import get_face_recognizer
        data = await request.json()
        image_b64 = data.get("image")

        if not image_b64:
            return {"success": False, "error": "Missing image"}

        recognizer = get_face_recognizer()
        faces = recognizer.recognize_faces(image_b64)

        return {"success": True, "faces": faces}
    except Exception as e:
        print(f"[Face API] Recognize error: {e}")
        return {"success": False, "error": str(e)}


@app.get("/api/face/list")
async def list_enrolled_faces():
    """List all enrolled faces."""
    try:
        from clients.face import get_face_recognizer
        recognizer = get_face_recognizer()
        names = recognizer.list_enrolled()
        return {"success": True, "faces": names}
    except Exception as e:
        print(f"[Face API] List error: {e}")
        return {"success": False, "error": str(e)}


@app.delete("/api/face/{name}")
async def delete_face(name: str):
    """Delete an enrolled face."""
    try:
        from clients.face import get_face_recognizer
        recognizer = get_face_recognizer()
        success = recognizer.delete_face(name)

        if success:
            return {"success": True, "message": f"Deleted {name}"}
        else:
            return {"success": False, "error": f"Face '{name}' not found"}
    except Exception as e:
        print(f"[Face API] Delete error: {e}")
        return {"success": False, "error": str(e)}


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the frontend HTML."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return index_path.read_text()
    return HTMLResponse("<h1>Frontend not found. Please create static/index.html</h1>")


# -----------------------------
# Persistent Voice WebSocket - Main Endpoint
# -----------------------------

class VoiceSession:
    """Manages state for a persistent voice session."""
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self.asr_webm_bytes = bytearray()
        self.asr_pcm = np.zeros(0, dtype=np.float32)
        self.asr_last_text = ""
        self.is_recording = False
        self.audio_context_initialized = False
        self.selected_voice = os.getenv("KOKORO_VOICE", "af_bella")  # Default voice
        self.enabled_tools = []  # Default: no tools enabled
        
        # Import system prompt from prompts.py
        from prompts import DEFAULT_SYSTEM_PROMPT
        self.system_prompt = DEFAULT_SYSTEM_PROMPT
        # Initialize conversation history AFTER system_prompt is defined
        self.conversation_history: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": self.system_prompt,
            }
        ]

    async def send_message(self, msg_type: str, data: Dict[str, Any] = None):
        """Send a JSON message to the client."""
        try:
            # Check if WebSocket is still connected
            if self.websocket.client_state != WebSocketState.CONNECTED:
                print(f"[Voice Session] Cannot send message '{msg_type}': WebSocket not connected (state: {self.websocket.client_state})")
                return False
            
            payload = {"type": msg_type}
            if data:
                payload.update(data)
            await self.websocket.send_json(payload)
            return True
        except (WebSocketDisconnect, Exception) as e:
            # Handle WebSocket disconnection gracefully
            error_type = type(e).__name__
            if "Disconnect" in error_type or "ConnectionClosed" in error_type or "ClientDisconnected" in error_type:
                print(f"[Voice Session] WebSocket disconnected while sending message '{msg_type}'")
            else:
                print(f"[Voice Session] Error sending message '{msg_type}': {e}")
            return False

    async def send_audio_chunk(self, audio_data: bytes):
        """Send binary audio chunk to the client."""
        try:
            # Check if WebSocket is still connected
            if self.websocket.client_state != WebSocketState.CONNECTED:
                print(f"[Voice Session] Cannot send audio chunk: WebSocket not connected (state: {self.websocket.client_state})")
                return False
            
            await self.websocket.send_bytes(audio_data)
            return True
        except (WebSocketDisconnect, Exception) as e:
            # Handle WebSocket disconnection gracefully
            error_type = type(e).__name__
            if "Disconnect" in error_type or "ConnectionClosed" in error_type or "ClientDisconnected" in error_type:
                print(f"[Voice Session] WebSocket disconnected while sending audio chunk")
            else:
                print(f"[Voice Session] Error sending audio chunk: {e}")
            return False

    async def process_asr_chunk(self, chunk_bytes: bytes):
        """Process an ASR audio chunk - streams directly to faster-whisper."""
        if not chunk_bytes:
            return
        
        self.asr_webm_bytes.extend(chunk_bytes)
        
        # Wait for some bytes before decoding (WebM needs headers)
        if len(self.asr_webm_bytes) < 4000:
            return
        
        # Optimize: Send larger chunks less frequently for better performance
        # Larger chunks = fewer API calls = faster overall
        STREAMING_CHUNK_SECONDS = 1.5  # Send chunks every 1.5 seconds (larger chunks)
        MIN_AUDIO_SECONDS = 1.0  # Minimum audio before first transcription
        
        try:
            # Decode accumulated WebM
            decoded_pcm = decode_webm_bytes_to_pcm_f32(bytes(self.asr_webm_bytes), target_sr=SAMPLE_RATE)
            
            # Check if we have enough audio
            if decoded_pcm.size < int(SAMPLE_RATE * MIN_AUDIO_SECONDS):
                return
            
            # Check if audio has actual signal
            audio_max = np.abs(decoded_pcm).max()
            if audio_max < 0.001:
                return
            
            # For streaming: only process new audio (sliding window)
            # This avoids re-processing the same audio
            if hasattr(self, '_last_processed_samples'):
                new_samples = decoded_pcm.size - self._last_processed_samples
                # Only process if we have enough new audio (larger chunks = fewer API calls)
                if new_samples < int(SAMPLE_RATE * STREAMING_CHUNK_SECONDS):
                    return
                # Extract only the new portion for processing
                audio_to_process = decoded_pcm[-new_samples:]
            else:
                # First chunk: process what we have
                audio_to_process = decoded_pcm
            
            # Update tracking
            self._last_processed_samples = decoded_pcm.size
            
            # Stream to faster-whisper (larger chunks = better performance)
            partial_text = await asr.transcribe(audio_to_process)
            
            if partial_text and partial_text != self.asr_last_text:
                self.asr_last_text = partial_text
                await self.send_message("asr_partial", {"text": partial_text})
                
        except Exception as e:
            print(f"[Voice Session] ASR decode error: {e}")
            import traceback
            traceback.print_exc()
            await self.send_message("error", {"error": f"ASR decode error: {e}"})
            return

    async def process_asr_final(self):
        """Process final ASR transcription - sends complete accumulated audio."""
        try:
            # Decode all accumulated audio
            self.asr_pcm = decode_webm_bytes_to_pcm_f32(bytes(self.asr_webm_bytes), target_sr=SAMPLE_RATE)
            
            # Check if audio has actual signal before transcribing
            if self.asr_pcm.size > 0:
                audio_max = np.abs(self.asr_pcm).max()
                if audio_max < 0.001:
                    print(f"[Voice Session] Final audio is silent (max amplitude: {audio_max:.6f}), skipping transcription")
                    final_text = ""
                else:
                    # For final transcription, send the complete audio
                    # This gives better accuracy than streaming chunks
                    final_text = await asr.transcribe(self.asr_pcm)
            else:
                final_text = ""
        except Exception as e:
            print(f"[Voice Session] Final ASR decode error: {e}")
            import traceback
            traceback.print_exc()
            await self.send_message("error", {"error": f"ASR error: {e}"})
            final_text = ""
        
        # Reset ASR state
        self.asr_webm_bytes = bytearray()
        self.asr_pcm = np.zeros(0, dtype=np.float32)
        self.asr_last_text = ""
        if hasattr(self, '_last_processed_samples'):
            delattr(self, '_last_processed_samples')
        
        if final_text and final_text.strip():
            await self.send_message("asr_final", {"text": final_text})
            return final_text
        else:
            print(f"[Voice Session] No transcription result (empty or silent audio)")
            return None

    async def send_transient_response(self, text: str):
        """Send a transient/interim response (e.g., 'on it', 'thinking...')."""
        await self.send_message("transient_response", {"text": text})
        # Also generate and stream TTS for transient response
        await self.stream_tts(text, is_transient=True)

    async def send_final_response(self, text: str):
        """Send final response and stream TTS sentence-by-sentence for lower latency."""
        print(f"[Voice Session] send_final_response called with: {text[:100]}...")
        await self.send_message("final_response", {"text": text})
        print(f"[Voice Session] final_response message sent, starting sentence-by-sentence TTS...")

        # Split into sentences for faster TTS start
        sentences, remaining = self._extract_complete_sentences(text)

        # Add any remaining fragment
        if remaining.strip():
            sentences.append(remaining.strip())

        if not sentences:
            # Fallback: use whole text if no sentences found
            sentences = [text]

        print(f"[Voice Session] TTS pipeline: {len(sentences)} sentence(s)")

        for i, sentence in enumerate(sentences):
            if sentence.strip():
                print(f"[Voice Session] TTS sentence {i+1}/{len(sentences)}: '{sentence[:40]}...'")
                await self.stream_tts(sentence.strip(), is_transient=False)

        print(f"[Voice Session] TTS completed ({len(sentences)} sentences)")

    async def stream_tts(self, text: str, is_transient: bool = False, voice: str = None):
        """Stream TTS audio chunks."""
        if not text or not text.strip():
            print(f"[Voice Session] stream_tts: empty text, skipping")
            return
        
        # Clean text - remove markers
        text = text.replace("<|channel|>analysis<|message|>", "")
        text = text.replace("<|channel|>final<|message|>", "")
        text = text.replace("<|end|>", "")
        text = text.replace("<|start|>assistant", "")
        text = text.strip()
        
        if not text:
            print(f"[Voice Session] stream_tts: text empty after cleaning")
            return
        
        print(f"[Voice Session] stream_tts: synthesizing '{text[:50]}...'")
        
        # Use provided voice or session default
        voice_to_use = voice or self.selected_voice
        print(f"[Voice Session] stream_tts: using voice '{voice_to_use}' (provided: {voice}, session default: {self.selected_voice})")
        
        # Send TTS start message - if it fails, abort early
        if not await self.send_message("tts_start", {"sample_rate": 24000, "is_transient": is_transient}):
            print(f"[Voice Session] stream_tts: failed to send tts_start, aborting")
            return
        
        try:
            chunk_count = 0
            for audio_data, sample_rate in tts.synth_stream_chunks(text, voice=voice_to_use):
                # Check if we can still send before processing more chunks
                if not await self.send_audio_chunk(audio_data):
                    print(f"[Voice Session] stream_tts: failed to send audio chunk {chunk_count + 1}, stopping")
                    break
                chunk_count += 1
                await asyncio.sleep(0.001)  # Small delay to prevent overwhelming
            
            print(f"[Voice Session] stream_tts: sent {chunk_count} audio chunks")
            # Try to send done message, but don't fail if connection is closed
            await self.send_message("tts_done", {"is_transient": is_transient})
        except (WebSocketDisconnect, Exception) as e:
            error_msg = str(e)
            print(f"[Voice Session] TTS error with voice '{voice_to_use}': {e}")
            
            # If voice file not found, try fallback voices
            if "404" in error_msg or "not found" in error_msg.lower() or "entry not found" in error_msg.lower():
                fallback_voices = ["af_heart", "af_nicole", "af_jessica"]
                print(f"[Voice Session] Trying fallback voices: {fallback_voices}")
                for fallback_voice in fallback_voices:
                    if fallback_voice == voice_to_use:
                        continue  # Skip if already tried
                    try:
                        print(f"[Voice Session] Trying fallback voice: {fallback_voice}")
                        # Re-send TTS start with fallback voice
                        if not await self.send_message("tts_start", {"sample_rate": 24000, "is_transient": is_transient}):
                            print(f"[Voice Session] Failed to send tts_start for fallback, aborting")
                            break
                        
                        chunk_count = 0
                        for audio_data, sample_rate in tts.synth_stream_chunks(text, voice=fallback_voice):
                            # Check if we can still send before processing more chunks
                            if not await self.send_audio_chunk(audio_data):
                                print(f"[Voice Session] Failed to send audio chunk with fallback voice, stopping")
                                break
                            chunk_count += 1
                            await asyncio.sleep(0.001)
                        print(f"[Voice Session] Fallback voice '{fallback_voice}' succeeded, sent {chunk_count} chunks")
                        await self.send_message("tts_done", {"is_transient": is_transient})
                        # Update session voice to the working fallback
                        self.selected_voice = fallback_voice
                        await self.send_message("voice_changed", {"voice": fallback_voice})
                        return
                    except (WebSocketDisconnect, Exception) as fallback_error:
                        error_type = type(fallback_error).__name__
                        if "Disconnect" in error_type or "ConnectionClosed" in error_type or "ClientDisconnected" in error_type:
                            print(f"[Voice Session] WebSocket disconnected during fallback voice attempt")
                            break  # Stop trying fallbacks if disconnected
                        print(f"[Voice Session] Fallback voice '{fallback_voice}' also failed: {fallback_error}")
                        continue
                
                # If all fallbacks failed, send error message (only if still connected)
                await self.send_message("error", {"error": f"TTS failed: Voice '{voice_to_use}' not available. Tried fallbacks: {fallback_voices}"})
            else:
                # For other errors, check if it's a WebSocket disconnection
                error_type = type(e).__name__
                if "Disconnect" in error_type or "ConnectionClosed" in error_type or "ClientDisconnected" in error_type:
                    print(f"[Voice Session] WebSocket disconnected during TTS streaming")
                    # Don't try to send error message if disconnected
                else:
                    # For other errors, just log and send error (only if still connected)
                    import traceback
                    traceback.print_exc()
                    await self.send_message("error", {"error": f"TTS error: {error_msg}"})

    def _extract_complete_sentences(self, text: str) -> tuple:
        """Extract complete sentences from text buffer.

        Returns (complete_sentences, remaining_buffer).
        A sentence is complete if it ends with . ! or ? followed by space or end of string.
        """
        if not text:
            return [], ""

        import re
        # Pattern: sentence ending punctuation followed by space or end
        # Avoid splitting on abbreviations like "Dr.", "Mr.", "etc."
        sentences = []

        # Simple approach: split on . ! ? followed by space
        # Keep the punctuation with the sentence
        pattern = r'([.!?])\s+'
        parts = re.split(pattern, text)

        # Reconstruct sentences (parts alternate: text, punct, text, punct, ...)
        current = ""
        for i, part in enumerate(parts):
            if i % 2 == 0:  # Text part
                current += part
            else:  # Punctuation part
                current += part
                sentences.append(current.strip())
                current = ""

        # Whatever is left is incomplete
        remaining = current.strip()

        return sentences, remaining

    async def stream_llm_with_tts(self, messages: list, tools: list = None) -> tuple:
        """Stream LLM response with overlapped TTS generation.

        Starts TTS generation for each sentence while LLM continues streaming.
        Uses a queue to process TTS in order while LLM runs in parallel.

        Returns: (status, tool_calls, full_response)
        """
        sentence_buffer = ""
        full_response = ""
        sentence_count = 0

        # Queue for sentences to be processed by TTS
        tts_queue = asyncio.Queue()
        tts_done = asyncio.Event()

        async def tts_worker():
            """Background worker that processes TTS queue in order."""
            while True:
                try:
                    # Wait for sentence or done signal
                    sentence = await asyncio.wait_for(tts_queue.get(), timeout=0.1)
                    if sentence is None:  # Poison pill - we're done
                        break
                    await self.stream_tts(sentence, is_transient=False)
                    tts_queue.task_done()
                except asyncio.TimeoutError:
                    if tts_done.is_set() and tts_queue.empty():
                        break
                    continue
                except Exception as e:
                    print(f"[TTS Worker] Error: {e}")
                    break

        # Start TTS worker
        tts_task = asyncio.create_task(tts_worker())

        try:
            async for chunk in llm.stream_complete(messages, tools=tools):
                if chunk.startswith("data: "):
                    try:
                        data = json.loads(chunk[6:])

                        # Handle tool calls - stop TTS and return for tool processing
                        if "tool_calls_complete" in data:
                            tts_done.set()
                            await tts_queue.put(None)  # Signal worker to stop
                            await tts_task
                            return ("tool_calls", data["tool_calls_complete"], full_response)

                        if "content" in data and data["content"]:
                            content = data["content"]
                            sentence_buffer += content
                            full_response += content

                            # Send transient response to show text as it streams
                            await self.send_message("transient_response", {"text": full_response})

                            # Check for complete sentences
                            sentences, sentence_buffer = self._extract_complete_sentences(sentence_buffer)

                            for sentence in sentences:
                                if sentence.strip():
                                    sentence_count += 1
                                    print(f"[TTS Pipeline] Queuing sentence {sentence_count}: '{sentence[:50]}...'")
                                    # Queue for TTS - doesn't block LLM streaming!
                                    await tts_queue.put(sentence)

                    except json.JSONDecodeError:
                        pass

            # Handle any remaining text in buffer
            if sentence_buffer.strip():
                sentence_count += 1
                print(f"[TTS Pipeline] Queuing final fragment {sentence_count}: '{sentence_buffer[:50]}...'")
                await tts_queue.put(sentence_buffer.strip())

            # Signal TTS worker we're done and wait for it to finish
            tts_done.set()
            await tts_queue.put(None)  # Poison pill
            await tts_task

            print(f"[TTS Pipeline] Completed: {sentence_count} sentences, {len(full_response)} chars total")
            return ("complete", None, full_response)

        except Exception as e:
            print(f"[TTS Pipeline] Error: {e}")
            tts_done.set()
            await tts_queue.put(None)
            await tts_task
            raise

    async def process_user_message(self, user_text: str):
        """Process user message through LLM pipeline."""
        if not user_text or not user_text.strip():
            return
        
        # Add user message to history
        self.conversation_history.append({"role": "user", "content": user_text})

        # Build messages for LLM
        messages_for_llm = list(self.conversation_history)

        # Stream LLM response
        final_response = ""
        chunk_count = 0
        raw_chunks = []
        try:
            # Get enabled tools for this session
            enabled_tool_defs = get_enabled_tools(self.enabled_tools)
            print(f"[Voice Session] Starting LLM stream with {len(messages_for_llm)} messages")
            print(f"[Voice Session] Enabled tools: {self.enabled_tools} -> {[t['function']['name'] for t in enabled_tool_defs]}")

            # Check if TTS/LLM overlap is enabled and we're in simple conversation mode (no tools)
            tts_config = TTSConfig()
            if tts_config.overlap_llm and not enabled_tool_defs:
                print(f"[Voice Session] Using TTS/LLM overlap pipeline (no tools enabled)")
                status, tool_calls, full_response = await self.stream_llm_with_tts(messages_for_llm, tools=None)

                if status == "complete" and full_response:
                    # Clean up response markers
                    full_response = llm._extract_final_channel(full_response)
                    full_response = full_response.replace("<|channel|>analysis<|message|>", "")
                    full_response = full_response.replace("<|channel|>final<|message|>", "")
                    full_response = full_response.replace("<|end|>", "")
                    full_response = full_response.replace("<|start|>assistant", "")
                    full_response = full_response.strip()

                    if full_response:
                        # Add to conversation history
                        self.conversation_history.append({"role": "assistant", "content": full_response})
                        # Send final text message (TTS already done in pipeline)
                        await self.send_message("final_response", {"text": full_response})
                        print(f"[Voice Session] Overlap pipeline complete: {len(full_response)} chars")
                        return

                print(f"[Voice Session] Overlap pipeline returned empty response, continuing...")

            async for chunk in llm.stream_complete(messages_for_llm, tools=enabled_tool_defs if enabled_tool_defs else None):
                chunk_count += 1
                raw_chunks.append(chunk[:100])  # Store first 100 chars for debugging
                
                if chunk.startswith("data: "):
                    try:
                        data = json.loads(chunk[6:])
                        # Handle tool calls completion
                        if "tool_calls_complete" in data:
                            tool_calls = data["tool_calls_complete"]
                            print(f"[Voice Session] ✅ TOOL CALLS DETECTED: {len(tool_calls)} tools")
                            for i, tc in enumerate(tool_calls):
                                func = tc.get("function", {})
                                print(f"  Tool {i+1}: {func.get('name', 'unknown')} with args: {func.get('arguments', '{}')}")
                            
                            # Determine feedback message based on tool type
                            is_agent_tool = False
                            for tc in tool_calls:
                                func = tc.get("function", {})
                                if func.get("name") in ["markdown_assistant", "reasoning_assistant"]:
                                    is_agent_tool = True
                                    break
                            
                            if is_agent_tool:
                                # Custom feedback for reasoning
                                for tc in tool_calls:
                                    if tc.get("function", {}).get("name") == "reasoning_assistant":
                                        feedback_msg = "Let me think through this..."
                                        break
                                else:
                                    feedback_msg = "On it."
                            else:
                                feedback_msg = "Looking that up for you."
                            
                            # Send conversational feedback
                            await self.send_message("tool_invocation", {"message": feedback_msg})
                            await self.stream_tts(feedback_msg, is_transient=True)
                            
                            # IMPORTANT: Add assistant message with tool_calls to conversation history FIRST
                            # The LLM server expects this format: assistant message with tool_calls, then tool results
                            assistant_message = {
                                "role": "assistant",
                                "content": None,  # No text content, only tool calls
                                "tool_calls": tool_calls
                            }
                            self.conversation_history.append(assistant_message)
                            print(f"[Voice Session] Added assistant message with {len(tool_calls)} tool call(s) to history")
                            
                            # Execute each tool call
                            tool_results = []
                            for tool_call in tool_calls:
                                tool_id = tool_call.get("id", "")
                                function = tool_call.get("function", {})
                                tool_name = function.get("name", "")
                                arguments_str = function.get("arguments", "{}")
                                
                                try:
                                    arguments = json.loads(arguments_str)
                                except json.JSONDecodeError:
                                    arguments = {}
                                
                                print(f"[Voice Session] Executing tool: {tool_name} with args: {arguments}")
                                
                                # Execute tool
                                tool_result = await execute_tool(tool_name, arguments)
                                print(f"[Voice Session] Tool '{tool_name}' returned: {tool_result[:100]}...")
                                
                                # Check if this is an agent tool that needs special handling
                                try:
                                    tool_result_data = json.loads(tool_result)
                                    if tool_result_data.get("agent_type"):
                                        # This is an agent tool - send signal to open UI
                                        await self.send_message("agent_started", {
                                            "agent_type": tool_result_data.get("agent_type"),
                                            "task": tool_result_data.get("task", ""),
                                            "codebase_path": tool_result_data.get("codebase_path", "")
                                        })
                                        print(f"[Voice Session] Agent '{tool_result_data.get('agent_type')}' started - UI should open")
                                except json.JSONDecodeError:
                                    pass  # Not JSON, treat as regular tool result
                                
                                tool_results.append({
                                    "tool_call_id": tool_id,
                                    "role": "tool",
                                    "name": tool_name,
                                    "content": tool_result
                                })
                            
                            # Add tool results to conversation history (after assistant message with tool_calls)
                            for tool_result in tool_results:
                                self.conversation_history.append(tool_result)
                            print(f"[Voice Session] Added {len(tool_results)} tool result(s) to history")
                            
                            # Get final response from LLM with tool results
                            print(f"[Voice Session] Getting final response after tool execution")
                            followup_messages = list(self.conversation_history)
                            
                            # Stream final response
                            tool_final_response = ""
                            enabled_tool_defs = get_enabled_tools(self.enabled_tools)
                            async for chunk in llm.stream_complete(followup_messages, tools=enabled_tool_defs if enabled_tool_defs else None):
                                if chunk.startswith("data: "):
                                    try:
                                        data = json.loads(chunk[6:])
                                        if "content" in data and data["content"]:
                                            tool_final_response += data["content"]
                                        elif "tool_calls_complete" in data:
                                            # Another tool call - handle recursively (for now, just break)
                                            print(f"[Voice Session] Another tool call detected, stopping recursion")
                                            break
                                    except json.JSONDecodeError:
                                        pass
                            
                            # Check if any of the executed tools were agents
                            is_agent_tool = False
                            agent_type = None
                            agent_task = None
                            for tool_result in tool_results:
                                try:
                                    result_data = json.loads(tool_result.get("content", "{}"))
                                    if result_data.get("agent_type"):
                                        is_agent_tool = True
                                        agent_type = result_data.get("agent_type")
                                        agent_task = result_data.get("task", "")
                                        break
                                except json.JSONDecodeError:
                                    pass
                            
                            if is_agent_tool and agent_type == "markdown_assistant":
                                # For markdown assistant, make a separate LLM call to generate markdown
                                print(f"[Voice Session] Making separate LLM call for markdown generation (task: {agent_task[:100]}...)")
                                
                                # Create a focused prompt for markdown generation
                                markdown_generation_messages = [
                                    {
                                        "role": "system",
                                        "content": """You are a documentation assistant. Your goal is to generate clear, well-structured markdown documents.

CRITICAL INSTRUCTIONS:
- Output ONLY markdown content. No explanations before or after.
- Do NOT think out loud or show your reasoning process.
- Generate the document directly without excessive analysis.
- Use proper markdown formatting: headers, lists, code blocks, tables as needed.
- Structure the document logically with clear sections.
- Be thorough but concise."""
                                    },
                                    {
                                        "role": "user",
                                        "content": f"Write the following document: {agent_task}"
                                    }
                                ]
                                
                                # Use medium reasoning for markdown generation
                                markdown_gen_config = LLMConfig()
                                markdown_gen_config.reasoning_effort = "medium"
                                markdown_gen_llm = LlamaCppClient(markdown_gen_config)
                                
                                # Stream markdown generation response
                                markdown_response = ""
                                reasoning_accumulated = ""
                                chunk_count = 0
                                
                                # Stream tokens directly to markdown editor as they arrive
                                await self.send_message("agent_markdown_chunk", {"content": "", "done": False})
                                
                                async for chunk in markdown_gen_llm.stream_complete(markdown_generation_messages, tools=None):
                                    chunk_count += 1
                                    if chunk.startswith("data: "):
                                        try:
                                            data = json.loads(chunk[6:])
                                            if "content" in data and data["content"]:
                                                content = data["content"]
                                                markdown_response += content
                                                # Stream to markdown editor immediately
                                                await self.send_message("agent_markdown_chunk", {"content": content, "done": False})
                                            elif "reasoning_content" in data and data["reasoning_content"]:
                                                reasoning_accumulated += data["reasoning_content"]
                                        except json.JSONDecodeError:
                                            pass
                                    elif chunk.strip() and not chunk.startswith("data: "):
                                        markdown_response += chunk
                                        await self.send_message("agent_markdown_chunk", {"content": chunk, "done": False})
                                
                                print(f"[Voice Session] Markdown generation complete: {len(markdown_response)} chars")
                                
                                # If no content but we have reasoning, use reasoning as fallback
                                if not markdown_response and reasoning_accumulated:
                                    markdown_response = reasoning_accumulated
                                
                                # Process markdown response
                                if markdown_response and markdown_response.strip():
                                    markdown_response = llm._extract_final_channel(markdown_response)
                                    markdown_response = markdown_response.replace("<|channel|>analysis<|message|>", "")
                                    markdown_response = markdown_response.replace("<|channel|>final<|message|>", "")
                                    markdown_response = markdown_response.replace("<|end|>", "")
                                    markdown_response = markdown_response.replace("<|start|>assistant", "")
                                    markdown_response = markdown_response.strip()
                                    
                                    if markdown_response:
                                        print(f"[Voice Session] Markdown generated: {len(markdown_response)} characters")
                                        
                                        # Signal completion
                                        await self.send_message("agent_markdown_chunk", {"content": "", "done": True})
                                        
                                        # Add to conversation history
                                        markdown_summary = f"Generated documentation for: {agent_task}\n\n{markdown_response[:500]}{'...' if len(markdown_response) > 500 else ''}"
                                        self.conversation_history.append({"role": "assistant", "content": markdown_summary})
                                        
                                        # Send a message to frontend to add markdown to conversation UI
                                        await self.send_message("agent_markdown_complete", {
                                            "task": agent_task,
                                            "markdown": markdown_response
                                        })
                                        return
                                    else:
                                        print(f"[Voice Session] Markdown response was empty after processing")
                                else:
                                    print(f"[Voice Session] No markdown response received")
                            
                            elif is_agent_tool and agent_type == "reasoning_assistant":
                                # For reasoning assistant, use Nemotron for deep analysis
                                agent_problem = ""
                                agent_context = ""
                                agent_analysis_type = "general"
                                
                                # Extract problem, context, and analysis_type from tool result
                                for tool_result in tool_results:
                                    try:
                                        result_data = json.loads(tool_result.get("content", "{}"))
                                        if result_data.get("agent_type") == "reasoning_assistant":
                                            agent_problem = result_data.get("problem", "")
                                            agent_context = result_data.get("context", "")
                                            agent_analysis_type = result_data.get("analysis_type", "general")
                                            break
                                    except json.JSONDecodeError:
                                        pass
                                
                                if agent_problem:
                                    print(f"[Voice Session] Executing reasoning agent for: {agent_problem[:100]}...")
                                    await self.execute_reasoning_agent(agent_problem, agent_context, agent_analysis_type)
                                    return
                                else:
                                    print(f"[Voice Session] No problem found for reasoning agent")
                            
                            # Process and send the final response from tool execution (for non-agent tools or fallback)
                            if tool_final_response and tool_final_response.strip():
                                tool_final_response = llm._extract_final_channel(tool_final_response)
                                tool_final_response = tool_final_response.replace("<|channel|>analysis<|message|>", "")
                                tool_final_response = tool_final_response.replace("<|channel|>final<|message|>", "")
                                tool_final_response = tool_final_response.replace("<|end|>", "")
                                tool_final_response = tool_final_response.replace("<|start|>assistant", "")
                                tool_final_response = tool_final_response.strip()
                                
                                if tool_final_response:
                                    print(f"[Voice Session] Tool execution final response: {tool_final_response[:100]}...")
                                    # Add to conversation history
                                    self.conversation_history.append({"role": "assistant", "content": tool_final_response})
                                    # Regular tool - send final response and TTS
                                    await self.send_final_response(tool_final_response)
                                    return  # Exit early since we've handled tool execution
                                else:
                                    print(f"[Voice Session] Tool execution response was empty after processing")
                            else:
                                print(f"[Voice Session] No response received after tool execution")
                            
                            # If we get here, tool execution didn't produce a valid response
                            # Break out of outer loop to handle error
                            break
                            
                        # Only accumulate actual content, not reasoning_content
                        # Reasoning models output reasoning first, then final content
                        elif "content" in data and data["content"]:
                            content_chunk = data["content"]
                            # Skip reasoning markers and their content
                            if "<|channel|>analysis" in content_chunk or "<|channel|>commentary" in content_chunk:
                                continue  # Skip reasoning/analysis content
                            final_response += content_chunk
                            # Note: No intermediate streaming - only final response is shown
                        # Note: reasoning_content is NOT shown - only actual content
                        elif "error" in data:
                            print(f"[Voice Session] LLM error: {data['error']}")
                    except json.JSONDecodeError:
                        # Chunk might not be JSON, try to extract content directly
                        if chunk.strip():
                            final_response += chunk
                elif chunk.strip() and not chunk.startswith("data: "):
                    # Some LLMs might return content directly without "data: " prefix
                    final_response += chunk
                    
            print(f"[Voice Session] LLM stream completed: {chunk_count} chunks, response length: {len(final_response)}")
            if chunk_count == 0:
                print(f"[Voice Session] WARNING: No chunks received from LLM stream")
            if len(final_response) == 0:
                print(f"[Voice Session] WARNING: Empty final_response after {chunk_count} chunks")
                print(f"[Voice Session] First few raw chunks: {raw_chunks[:5]}")
        except Exception as e:
            print(f"[Voice Session] Error streaming LLM response: {e}")
            import traceback
            traceback.print_exc()
            await self.send_message("error", {"error": f"LLM streaming error: {e}"})
            return
        
        # Extract final channel
        if not final_response or not final_response.strip():
            print(f"[Voice Session] Empty LLM response after streaming ({chunk_count} chunks received)")
            print(f"[Voice Session] Raw chunks preview: {raw_chunks[:10]}")
            print(f"[Voice Session] Full raw chunks: {raw_chunks}")
            await self.send_message("error", {"error": "No response from LLM"})
            return
            
        final_response = llm._extract_final_channel(final_response)
        # Clean up any remaining markers
        final_response = final_response.replace("<|channel|>analysis<|message|>", "")
        final_response = final_response.replace("<|channel|>final<|message|>", "")
        final_response = final_response.replace("<|end|>", "")
        final_response = final_response.replace("<|start|>assistant", "")
        final_response = final_response.strip()

        # Check if final_response is empty
        if not final_response or not final_response.strip():
            print(f"[Voice Session] Empty response after processing")
            await self.send_message("error", {"error": "Empty response after processing"})
            return

        # Send final response
        if final_response:
            print(f"[Voice Session] Sending response: {final_response[:100]}...")
            # Add to conversation history
            self.conversation_history.append({"role": "assistant", "content": final_response})
            # Send final response and TTS
            await self.send_final_response(final_response)
        else:
            print(f"[Voice Session] No final_response to send!")

    async def execute_markdown_agent(self, task: str, context: str = ""):
        """Execute the markdown assistant agent and stream results."""
        try:
            print(f"[Voice Session] Executing markdown agent: {task[:50]}...")
            
            # Signal agent started
            await self.send_message("agent_started", {"agent_type": "markdown_assistant", "task": task})
            
            # Build messages for markdown generation
            from prompts import MARKDOWN_ASSISTANT_PROMPT
            
            md_messages = [
                {"role": "system", "content": MARKDOWN_ASSISTANT_PROMPT},
                {"role": "user", "content": f"Task: {task}\n\nContext: {context}" if context else f"Task: {task}"}
            ]
            
            # Create LLM client for agent
            agent_llm = LlamaCppClient(LLMConfig())
            md_response = ""
            
            # Send initial chunk to signal start
            await self.send_message("agent_markdown_chunk", {"content": "", "done": False})
            
            async for chunk in agent_llm.stream_complete(md_messages, tools=None):
                if chunk.startswith("data: "):
                    try:
                        data = json.loads(chunk[6:])
                        if "content" in data and data["content"]:
                            content = data["content"]
                            md_response += content
                            await self.send_message("agent_markdown_chunk", {"content": content, "done": False})
                    except json.JSONDecodeError:
                        pass
                elif chunk.strip() and not chunk.startswith("data: "):
                    md_response += chunk
                    await self.send_message("agent_markdown_chunk", {"content": chunk, "done": False})
            
            # Clean up response
            if md_response:
                md_response = agent_llm._extract_final_channel(md_response)
            
            print(f"[Voice Session] Markdown generation complete: {len(md_response)} chars")
            
            # Signal completion
            await self.send_message("agent_markdown_chunk", {"content": "", "done": True})
            await self.send_message("agent_markdown_complete", {
                "task": task,
                "markdown": md_response
            })
            
        except Exception as e:
            print(f"[Voice Session] Markdown agent error: {e}")
            import traceback
            traceback.print_exc()
            await self.send_message("error", {"error": f"Markdown agent error: {str(e)}"})

    async def execute_html_agent(self, task: str, context: str = ""):
        """Execute the HTML assistant agent and stream results."""
        try:
            print(f"[Voice Session] Executing HTML agent: {task[:50]}...")
            
            # Signal agent started
            await self.send_message("agent_started", {"agent_type": "html_assistant", "task": task})
            
            # Build messages for HTML generation
            html_prompt = """You are an expert HTML, CSS, and JavaScript assistant. Generate clean, semantic, and functional web pages or components.

Guidelines:
- Generate complete HTML documents including <!DOCTYPE html>, <html>, <head>, and <body>.
- Use modern HTML5, CSS3, and vanilla JavaScript.
- For styling, use inline styles or a <style> block in the <head>.
- For interactivity, use a <script> block at the end of the <body>.
- Ensure the generated HTML is self-contained and runnable in a browser.

Output the complete HTML code."""
            
            html_messages = [
                {"role": "system", "content": html_prompt},
                {"role": "user", "content": f"Task: {task}\n\nContext: {context}" if context else f"Task: {task}"}
            ]
            
            # Create LLM client for agent
            agent_llm = LlamaCppClient(LLMConfig())
            html_response = ""
            
            # Send initial chunk to signal start
            await self.send_message("agent_html_chunk", {"content": "", "done": False})
            
            async for chunk in agent_llm.stream_complete(html_messages, tools=None):
                if chunk.startswith("data: "):
                    try:
                        data = json.loads(chunk[6:])
                        if "content" in data and data["content"]:
                            content = data["content"]
                            html_response += content
                            await self.send_message("agent_html_chunk", {"content": content, "done": False})
                    except json.JSONDecodeError:
                        pass
                elif chunk.strip() and not chunk.startswith("data: "):
                    html_response += chunk
                    await self.send_message("agent_html_chunk", {"content": chunk, "done": False})
            
            # Clean up response
            if html_response:
                html_response = agent_llm._extract_final_channel(html_response)
            
            print(f"[Voice Session] HTML generation complete: {len(html_response)} chars")
            
            # Signal completion
            await self.send_message("agent_html_chunk", {"content": "", "done": True})
            await self.send_message("agent_html_complete", {
                "task": task,
                "html": html_response
            })
            
        except Exception as e:
            print(f"[Voice Session] HTML agent error: {e}")
            import traceback
            traceback.print_exc()
            await self.send_message("error", {"error": f"HTML agent error: {str(e)}"})

    def load_demo_files(self) -> str:
        """Load demo files from demo_files/ folder for context injection."""
        demo_dir = Path(__file__).parent / "demo_files"
        if not demo_dir.exists():
            return ""
        
        context_parts = []
        context_parts.append("=== LOCAL DATA FILES ===\n")
        
        for file_path in sorted(demo_dir.iterdir()):
            if file_path.is_file() and file_path.suffix in ['.csv', '.txt', '.md']:
                try:
                    content = file_path.read_text()
                    context_parts.append(f"[{file_path.name}]")
                    context_parts.append(content)
                    context_parts.append("")  # Empty line between files
                except Exception as e:
                    print(f"[Demo Files] Error reading {file_path}: {e}")
        
        if len(context_parts) > 1:  # More than just the header
            return "\n".join(context_parts)
        return ""

    async def execute_reasoning_agent(self, problem: str, context: str = "", analysis_type: str = "general"):
        """Execute the Nemotron reasoning agent - shows thinking inline, then speaks conclusion."""
        try:
            print(f"[Voice Session] Executing reasoning agent: {problem[:80]}...")
            print(f"[Voice Session] Analysis type: {analysis_type}")
            
            # Load demo files and inject into context
            demo_context = self.load_demo_files()
            if demo_context:
                context = f"{context}\n\n{demo_context}" if context else demo_context
                print(f"[Voice Session] Injected {len(demo_context)} chars of demo file context")
            
            # Signal reasoning started (for inline display)
            await self.send_message("reasoning_started", {
                "problem": problem,
                "analysis_type": analysis_type
            })
            
            # Create Nemotron client
            nemotron = NemotronClient(NemotronConfig())
            
            thinking_response = ""
            content_response = ""
            chunk_count = 0
            
            # Stream the reasoning process
            async for chunk in nemotron.stream_reasoning(problem, context, analysis_type):
                chunk_count += 1
                if chunk_count <= 5:
                    print(f"[Voice Session] Reasoning chunk {chunk_count}: {chunk[:100]}...")
                    
                if chunk.startswith("data: "):
                    try:
                        data = json.loads(chunk[6:])
                        
                        if "thinking" in data:
                            # Stream thinking to show inline
                            thinking_chunk = data["thinking"]
                            thinking_response += thinking_chunk
                            await self.send_message("reasoning_thinking", {
                                "content": thinking_chunk
                            })
                            
                        elif "content" in data:
                            # Stream conclusion content
                            content_chunk = data["content"]
                            content_response += content_chunk
                            await self.send_message("reasoning_content", {
                                "content": content_chunk
                            })
                            
                        elif "done" in data:
                            print(f"[Voice Session] Reasoning done signal received")
                            break
                            
                        elif "error" in data:
                            await self.send_message("error", {"error": data["error"]})
                            return
                            
                    except json.JSONDecodeError:
                        pass
            
            print(f"[Voice Session] Reasoning complete: {len(thinking_response)} thinking chars, {len(content_response)} content chars")
            
            # Signal completion with full content
            await self.send_message("reasoning_complete", {
                "problem": problem,
                "thinking": thinking_response,
                "conclusion": content_response
            })
            
            # Add to conversation history
            self.conversation_history.append({
                "role": "assistant", 
                "content": content_response if content_response else thinking_response
            })
            
            # Speak the conclusion (it should already be TTS-friendly from the prompt)
            if content_response:
                await self.stream_tts(content_response)
            elif thinking_response:
                # If no separate conclusion, speak a summary of the thinking
                summary = self._extract_spoken_summary(thinking_response)
                if summary:
                    await self.stream_tts(summary)
            
        except Exception as e:
            print(f"[Voice Session] Reasoning agent error: {e}")
            import traceback
            traceback.print_exc()
            await self.send_message("error", {"error": f"Reasoning agent error: {str(e)}"})
    
    def _extract_spoken_summary(self, text: str, max_sentences: int = 3) -> str:
        """Extract a brief spoken summary from reasoning output."""
        # Clean up markdown and formatting
        import re
        
        # Remove headers, bullets, code blocks
        text = re.sub(r'^#+\s+.*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'^[-*]\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        text = re.sub(r'`[^`]+`', '', text)
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        sentences = [s.strip() for s in sentences if s.strip() and len(s) > 10]
        
        # Take first few meaningful sentences
        summary_sentences = sentences[:max_sentences]
        summary = " ".join(summary_sentences)
        
        # Clean up any remaining formatting
        summary = re.sub(r'\*+', '', summary)
        summary = re.sub(r'\s+', ' ', summary).strip()
        
        return summary if len(summary) > 20 else ""


MIN_AUDIO_SECONDS = 0.5


@app.websocket("/ws/voice")
async def voice_call(websocket: WebSocket):
    """Persistent voice call WebSocket - handles ASR, LLM, and TTS."""
    await websocket.accept()
    session = VoiceSession(websocket)
    
    print("[Voice Call] Client connected")
    try:
        await session.send_message("connected", {"status": "ready"})
        # Wait a moment for frontend to send initial voice selection
        await asyncio.sleep(0.2)
        # Send a short greeting (don't add to conversation history)
        # Pick a random short greeting for variety
        import random
        greetings = [
            "Hey! What's up?",
            "Hi there!",
            "Hey, I'm Spark!",
            "What can I help with?",
            "Hi! Ready when you are.",
        ]
        greeting = random.choice(greetings)
        print(f"[Voice Call] Sending greeting with voice: {session.selected_voice}")
        await session.send_message("final_response", {"text": greeting})
        await session.stream_tts(greeting, is_transient=False, voice=session.selected_voice)
    except Exception as e:
        print(f"[Voice Call] Error sending initial message: {e}")
        import traceback
        traceback.print_exc()
    
    try:
        while True:
            try:
                msg = await websocket.receive()
            except Exception as e:
                print(f"[Voice Call] Error receiving message: {e}")
                break
            
            if msg["type"] == "websocket.disconnect":
                print("[Voice Call] Client disconnected")
                break
            
            # Binary = audio chunk for ASR
            if msg.get("bytes") is not None:
                chunk_bytes = msg["bytes"]
                session.is_recording = True
                try:
                    await session.process_asr_chunk(chunk_bytes)
                except Exception as e:
                    print(f"[Voice Call] Error processing ASR chunk: {e}")
                    import traceback
                    traceback.print_exc()
            
            # Text = control messages
            elif msg.get("text") is not None:
                try:
                    data = json.loads(msg["text"])
                    msg_type = data.get("type")
                    
                    if msg_type == "asr_end":
                        # User finished speaking
                        session.is_recording = False
                        final_text = await session.process_asr_final()
                        if final_text:
                            # Process through LLM pipeline
                            await session.process_user_message(final_text)
                    
                    elif msg_type == "text_message":
                        # User sent a text message (typed)
                        text = data.get("text", "").strip()
                        if text:
                            await session.process_user_message(text)
                    
                    elif msg_type == "ping":
                        # Keep-alive ping
                        await session.send_message("pong")
                    
                    elif msg_type == "reset":
                        # Reset conversation
                        session.conversation_history = [
                            {
                                "role": "system",
                                "content": session.system_prompt,
                            }
                        ]
                        await session.send_message("reset_ack")
                    
                    elif msg_type == "set_voice":
                        # Change TTS voice
                        voice = data.get("voice")
                        if voice:
                            session.selected_voice = voice
                            await session.send_message("voice_changed", {"voice": voice})
                        else:
                            await session.send_message("error", {"error": "No voice specified"})
                    
                    elif msg_type == "set_system_prompt":
                        # Change system prompt
                        prompt = data.get("prompt")
                        if prompt:
                            session.system_prompt = prompt
                            # Update the system message in conversation history
                            if session.conversation_history and session.conversation_history[0].get("role") == "system":
                                session.conversation_history[0]["content"] = prompt
                            else:
                                session.conversation_history.insert(0, {"role": "system", "content": prompt})
                            print(f"[Voice Session] System prompt changed")
                            await session.send_message("system_prompt_changed", {"prompt": prompt})
                            # Also send the updated prompt so UI can sync
                            await session.send_message("system_prompt", {"prompt": prompt})
                        else:
                            await session.send_message("error", {"error": "No prompt specified"})
                    
                    elif msg_type == "set_tools":
                        # Enable/disable tools
                        enabled_tools = data.get("tools", [])
                        if isinstance(enabled_tools, list):
                            session.enabled_tools = enabled_tools
                            enabled_tool_defs = get_enabled_tools(enabled_tools)
                            print(f"[Voice Session] Tools updated: {enabled_tools} -> {[t['function']['name'] for t in enabled_tool_defs]}")
                            await session.send_message("tools_changed", {"tools": enabled_tools})
                        else:
                            await session.send_message("error", {"error": "Invalid tools format"})
                    
                    elif msg_type == "get_system_prompt":
                        # Get current system prompt
                        try:
                            await session.send_message("system_prompt", {"prompt": session.system_prompt})
                        except Exception as e:
                            print(f"[Voice Call] Error sending system prompt: {e}")
                            import traceback
                            traceback.print_exc()
                    
                    elif msg_type == "asr_audio":
                        # Voice call: audio only from VAD
                        print("[Voice Call] Received asr_audio")
                        audio_b64 = data.get("audio")
                        audio_format = data.get("format", "wav")
                        
                        if not audio_b64:
                            print("[Voice Call] No audio in payload")
                            await session.send_message("error", {"error": "No audio data"})
                            continue
                        
                        try:
                            # Decode audio
                            import base64
                            audio_bytes = base64.b64decode(audio_b64)
                            print(f"[Voice Call] Audio: {len(audio_bytes)} bytes")
                            
                            # Convert WAV to numpy array
                            import io
                            import soundfile as sf
                            audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))
                            print(f"[Voice Call] Audio decoded: {len(audio_data)} samples, {sample_rate}Hz")
                            
                            # Resample if needed
                            if sample_rate != SAMPLE_RATE:
                                ratio = SAMPLE_RATE / sample_rate
                                new_length = int(len(audio_data) * ratio)
                                resampled = np.interp(
                                    np.linspace(0, len(audio_data), new_length),
                                    np.arange(len(audio_data)),
                                    audio_data
                                )
                                audio_data = resampled.astype(np.float32)
                            
                            # Transcribe audio with streaming (uses global asr instance)
                            import time
                            asr_start = time.perf_counter()
                            transcription = ""

                            # Stream ASR segments to UI as they're recognized
                            async for partial_text in asr.transcribe_streaming(audio_data.astype(np.float32)):
                                transcription = partial_text
                                # Send partial result to frontend for live display
                                await session.send_message("asr_partial", {"text": partial_text})
                                print(f"[Voice Call] ASR partial: '{partial_text}'")

                            asr_elapsed = (time.perf_counter() - asr_start) * 1000
                            audio_duration = len(audio_data) / SAMPLE_RATE * 1000
                            print(f"[Voice Call] ⏱️ ASR: {asr_elapsed:.0f}ms for {audio_duration:.0f}ms audio (RTF: {asr_elapsed/audio_duration:.2f}x) → '{transcription}'")

                            if not transcription or not transcription.strip():
                                print("[Voice Call] Empty transcription, skipping")
                                await session.send_message("asr_result", {"text": ""})
                                continue

                            # Send final ASR result to frontend
                            await session.send_message("asr_result", {"text": transcription})

                            # Process with LLM and TTS
                            await session.process_user_message(transcription)
                            
                        except Exception as e:
                            print(f"[Voice Call] Error processing: {e}")
                            import traceback
                            traceback.print_exc()
                            await session.send_message("error", {"error": str(e)})
                    
                    elif msg_type == "video_call_data":
                        # Video call: audio + image from VAD/PTT
                        print("[Video Call] Received video_call_data")
                        audio_b64 = data.get("audio")
                        image_b64 = data.get("image")
                        audio_format = data.get("format", "wav")
                        custom_prompt = data.get("system_prompt")
                        
                        if not audio_b64:
                            print("[Video Call] No audio in payload")
                            await session.send_message("error", {"error": "No audio data"})
                            continue
                        
                        try:
                            # Decode audio
                            import base64
                            audio_bytes = base64.b64decode(audio_b64)
                            print(f"[Video Call] Audio: {len(audio_bytes)} bytes, format: {audio_format}")
                            
                            # Convert WAV to numpy array
                            import io
                            import soundfile as sf
                            audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes))
                            print(f"[Video Call] Audio decoded: {len(audio_data)} samples, {sample_rate}Hz")
                            
                            # Resample if needed
                            if sample_rate != SAMPLE_RATE:
                                ratio = SAMPLE_RATE / sample_rate
                                new_length = int(len(audio_data) * ratio)
                                resampled = np.interp(
                                    np.linspace(0, len(audio_data), new_length),
                                    np.arange(len(audio_data)),
                                    audio_data
                                )
                                audio_data = resampled.astype(np.float32)
                                print(f"[Video Call] Resampled to {SAMPLE_RATE}Hz: {len(audio_data)} samples")
                            
                            # Transcribe audio with streaming (uses global asr instance)
                            import time
                            asr_start = time.perf_counter()
                            transcription = ""

                            # Stream ASR segments to UI as they're recognized
                            async for partial_text in asr.transcribe_streaming(audio_data.astype(np.float32)):
                                transcription = partial_text
                                # Send partial result to frontend for live display
                                await session.send_message("asr_partial", {"text": partial_text})
                                print(f"[Video Call] ASR partial: '{partial_text}'")

                            asr_elapsed = (time.perf_counter() - asr_start) * 1000
                            audio_duration = len(audio_data) / SAMPLE_RATE * 1000
                            print(f"[Video Call] ⏱️ ASR: {asr_elapsed:.0f}ms for {audio_duration:.0f}ms audio (RTF: {asr_elapsed/audio_duration:.2f}x) → '{transcription}'")

                            if not transcription:
                                print("[Video Call] Empty transcription, skipping")
                                await session.send_message("asr_result", {"text": ""})
                                continue

                            # Send final ASR result to frontend
                            await session.send_message("asr_result", {"text": transcription})
                            
                            # Add to conversation history
                            session.conversation_history.append({
                                "role": "user",
                                "content": transcription
                            })
                            
                            # Build VLM request with image if available
                            if image_b64:
                                print(f"[Video Call] Image: {len(image_b64)} chars base64")

                                # Face recognition - recognize people in frame
                                face_context = ""
                                try:
                                    from clients.face import get_face_recognizer
                                    face_recognizer = get_face_recognizer()

                                    # Check for enrollment command: "remember my face as X" or "my name is X"
                                    lower_text = transcription.lower()
                                    if "remember my face as" in lower_text or "remember me as" in lower_text:
                                        # Extract name from command
                                        import re
                                        match = re.search(r'(?:remember (?:my face|me) as|my name is)\s+(\w+)', lower_text)
                                        if match:
                                            enroll_name = match.group(1).title()
                                            success = face_recognizer.enroll_face(enroll_name, image_b64)
                                            if success:
                                                await session.send_message("llm_final", {"text": f"Got it! I'll remember you as {enroll_name}."})
                                                await session.stream_tts(f"Got it! I'll remember you as {enroll_name}.")
                                                session.conversation_history.append({
                                                    "role": "assistant",
                                                    "content": f"Got it! I'll remember you as {enroll_name}."
                                                })
                                                continue  # Skip VLM call
                                            else:
                                                await session.stream_tts("I couldn't see your face clearly. Please try again.")
                                                continue

                                    # Recognize faces in frame
                                    recognized = face_recognizer.recognize_faces(image_b64)
                                    if recognized:
                                        face_context = face_recognizer.format_scene_description(recognized)
                                        print(f"[Video Call] Face recognition: {face_context}")
                                except Exception as e:
                                    print(f"[Video Call] Face recognition error (non-fatal): {e}")

                                # Use VLM for response
                                from prompts import VIDEO_CALL_PROMPT, DEFAULT_SYSTEM_PROMPT

                                # Combine personal context with video call prompt
                                system_prompt = custom_prompt or f"{DEFAULT_SYSTEM_PROMPT}\n\n{VIDEO_CALL_PROMPT}"

                                # Add face context to system prompt if we recognized anyone
                                if face_context:
                                    system_prompt = f"{system_prompt}\n\nCURRENT SCENE: {face_context}"

                                # Get recent conversation history (last 10 messages, excluding current)
                                # This gives VLM context of recent conversation
                                VLM_HISTORY_LIMIT = 10
                                recent_history = session.conversation_history[-VLM_HISTORY_LIMIT-1:-1] if len(session.conversation_history) > 1 else []
                                print(f"[Video Call] Including {len(recent_history)} history messages for VLM")

                                # Get VLM response
                                vlm_start = time.perf_counter()
                                vlm = VLMClient(VLMConfig())
                                enabled_tool_defs = get_enabled_tools(session.enabled_tools)

                                # Use streaming if no tools enabled, otherwise use non-streaming for tool support
                                if enabled_tool_defs:
                                    # Non-streaming mode with tool support
                                    vlm_result = await vlm.analyze_image(
                                        image_b64,
                                        transcription,
                                        system_prompt=system_prompt,
                                        tools=enabled_tool_defs,
                                        history=recent_history
                                    )
                                    vlm_elapsed = (time.perf_counter() - vlm_start) * 1000

                                    response_text = vlm_result.get("content", "")
                                    tool_calls = vlm_result.get("tool_calls", [])

                                    print(f"[Video Call] ⏱️ VLM: {vlm_elapsed:.0f}ms → {len(response_text)} chars")

                                    # Handle tool calls
                                    if tool_calls:
                                        print(f"[Video Call] Tool calls: {[tc.get('function', {}).get('name') for tc in tool_calls]}")
                                        for tool_call in tool_calls:
                                            func = tool_call.get("function", {})
                                            tool_name = func.get("name")
                                            try:
                                                args = json.loads(func.get("arguments", "{}"))
                                            except:
                                                args = {}

                                            print(f"[Video Call] Tool '{tool_name}' args: {args}")

                                            # Send acknowledgment
                                            await session.stream_tts("On it.", is_transient=True)

                                            # Execute tool
                                            if tool_name == "markdown_assistant":
                                                await session.execute_markdown_agent(args.get("task", ""), args.get("context", ""))
                                            elif tool_name == "html_assistant":
                                                await session.execute_html_agent(args.get("task", ""), args.get("context", ""))
                                            elif tool_name == "reasoning_assistant":
                                                await session.execute_reasoning_agent(
                                                    args.get("problem", ""),
                                                    args.get("context", ""),
                                                    args.get("analysis_type", "general")
                                                )
                                    else:
                                        # Regular response - speak it
                                        if response_text:
                                            session.conversation_history.append({
                                                "role": "assistant",
                                                "content": response_text
                                            })
                                            await session.send_message("llm_final", {"text": response_text})
                                            await session.stream_tts(response_text)
                                else:
                                    # Streaming mode (no tools) - stream text to UI as it arrives
                                    print("[Video Call] Using streaming VLM (no tools)")
                                    response_text = ""
                                    async for chunk in vlm.stream_analyze_image(
                                        image_b64,
                                        transcription,
                                        system_prompt=system_prompt,
                                        history=recent_history
                                    ):
                                        response_text += chunk
                                        # Send chunk to frontend for live display
                                        await session.send_message("transient_response", {"text": response_text})

                                    vlm_elapsed = (time.perf_counter() - vlm_start) * 1000
                                    print(f"[Video Call] ⏱️ VLM Stream: {vlm_elapsed:.0f}ms → {len(response_text)} chars")

                                    if response_text:
                                        session.conversation_history.append({
                                            "role": "assistant",
                                            "content": response_text
                                        })
                                        await session.send_message("llm_final", {"text": response_text})
                                        await session.stream_tts(response_text)
                            else:
                                # No image - use regular LLM
                                print("[Video Call] No image, using text LLM")
                                await session.process_user_message(transcription)
                                
                        except Exception as e:
                            print(f"[Video Call] Error processing: {e}")
                            import traceback
                            traceback.print_exc()
                            await session.send_message("error", {"error": str(e)})
                    
                    elif msg_type == "disconnect":
                        # Client requested disconnect
                        await session.send_message("disconnect_ack")
                        await websocket.close()
                        return
                    
                except json.JSONDecodeError as e:
                    print(f"[Voice Call] Invalid JSON: {e}")
                    await session.send_message("error", {"error": "Invalid message format"})
    
    except WebSocketDisconnect:
        print("[Voice Call] WebSocket disconnected")
    except Exception as e:
        print(f"[Voice Call] Error: {e}")
        import traceback
        traceback.print_exc()
        try:
            await session.send_message("error", {"error": str(e)})
        except:
            pass
    finally:
        print("[Voice Call] Session ended")


# -----------------------------
# Legacy endpoints (for backward compatibility)
# -----------------------------

@app.post("/api/voice_chat")
async def voice_chat(audio: UploadFile = File(...)):
    """Legacy non-streaming endpoint."""
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

    user_text = await asr.transcribe(pcm)
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
