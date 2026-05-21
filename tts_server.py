"""Streaming TTS HTTP server.

A small FastAPI wrapper around clients/tts.py so external apps (chatbot
front-ends, demos) can hit a stable HTTP endpoint instead of opening the
voice WebSocket or vendoring KPipeline themselves.

Endpoints:
  POST /tts
    body  : {"text": "...", "voice": "af_bella", "speed": 1.0}
    resp  : audio/wav, chunked. Streaming WAV header (data size = 0xFFFFFFFF)
            followed by int16 PCM @ TTS native sample rate. Yields one
            sentence at a time — first audio bytes arrive ~150-300 ms warm
            on CUDA Kokoro.

  GET  /health
    resp  : {"status": "ok", "engine": "...", "device": "...",
             "voice": "...", "sample_rate": 24000}

Run:
  python tts_server.py
  # listens on http://0.0.0.0:${TTS_PORT:-8002}

Picks engine/voice/device from TTSConfig (env vars: TTS_ENGINE, TTS_DEVICE,
KOKORO_VOICE, KOKORO_SPEED, etc.). On startup, loads the pipeline and runs
one dummy synth to JIT-warm CUDA kernels and load voice tensors. Subsequent
requests serialize on a single asyncio.Lock since the model is shared.
"""
from __future__ import annotations

import asyncio
import os
import struct
import time
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from clients.tts import create_tts
from config import TTSConfig


def streaming_wav_header(sample_rate: int, channels: int = 1,
                         sample_width: int = 2) -> bytes:
    """44-byte WAV header with placeholder sizes for streaming output.

    Most players (ffmpeg, vlc, <audio>, MediaSource) accept 0xFFFFFFFF as
    'unknown / stream until EOF' — we don't know the final length when we
    start sending.
    """
    byte_rate = sample_rate * channels * sample_width
    block_align = channels * sample_width
    bits = sample_width * 8
    return (
        b"RIFF" + struct.pack("<I", 0xFFFFFFFF) + b"WAVE"
        + b"fmt " + struct.pack("<IHHIIHH", 16, 1, channels, sample_rate,
                                 byte_rate, block_align, bits)
        + b"data" + struct.pack("<I", 0xFFFFFFFF)
    )


class TTSRequest(BaseModel):
    text: str
    voice: Optional[str] = None
    speed: Optional[float] = None  # accepted for forward-compat; clients/tts.py
                                   # currently honors only cfg.speed


app = FastAPI(title="Realtime2 TTS")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)
state: dict = {}
gpu_lock = asyncio.Lock()


@app.on_event("startup")
async def _startup() -> None:
    cfg = TTSConfig()
    print(f"[tts] loading engine={cfg.engine} device={cfg.device} "
          f"voice={cfg.voice}…")
    t0 = time.perf_counter()
    tts = create_tts(cfg)
    state["tts"] = tts
    state["cfg"] = cfg
    state["sample_rate"] = getattr(tts, "sample_rate", 24000)
    print(f"[tts] loaded in {(time.perf_counter()-t0)*1000:.0f} ms")

    # Warm the model: one short synth to JIT CUDA kernels and cache the voice.
    print("[tts] warming…")
    t0 = time.perf_counter()
    for _ in tts.synth_stream_chunks("Warmup.", voice=cfg.voice):
        pass
    print(f"[tts] warm in {(time.perf_counter()-t0)*1000:.0f} ms — ready")


@app.get("/health")
async def health():
    cfg: TTSConfig = state["cfg"]
    return {
        "status": "ok",
        "engine": cfg.engine,
        "device": cfg.device,
        "voice": cfg.voice,
        "sample_rate": state.get("sample_rate", 24000),
    }


@app.post("/tts")
async def tts(req: TTSRequest):
    """Stream synthesized speech as WAV (header upfront, then PCM chunks).

    synth_stream_chunks already splits on sentence boundaries and yields
    (pcm_bytes, sample_rate) tuples as each sentence finishes — we just
    forward them.
    """
    tts_engine = state["tts"]
    cfg: TTSConfig = state["cfg"]
    voice = req.voice or cfg.voice
    text = req.text.strip()

    async def gen():
        sr = state.get("sample_rate", 24000)
        yield streaming_wav_header(sr)
        if not text:
            return
        async with gpu_lock:
            t_start = time.perf_counter()
            first_chunk_at: Optional[float] = None
            total_audio_s = 0.0
            for pcm_bytes, chunk_sr in tts_engine.synth_stream_chunks(
                    text, voice=voice):
                total_audio_s += len(pcm_bytes) / 2 / chunk_sr  # int16 mono
                if first_chunk_at is None:
                    first_chunk_at = time.perf_counter() - t_start
                    print(f"[tts] TTFB {first_chunk_at*1000:.0f} ms "
                          f"({len(text)} chars)", flush=True)
                yield pcm_bytes
                # Cooperate with the event loop so the StreamingResponse
                # flushes between sentences.
                await asyncio.sleep(0)
            elapsed = time.perf_counter() - t_start
            if total_audio_s > 0:
                rtf = elapsed / total_audio_s
                print(f"[tts] done: {elapsed*1000:.0f} ms synth for "
                      f"{total_audio_s:.2f} s audio (RTF {rtf:.3f}x)",
                      flush=True)

    return StreamingResponse(gen(), media_type="audio/wav")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("TTS_PORT", "8002"))
    uvicorn.run("tts_server:app", host="0.0.0.0", port=port, log_level="info")
