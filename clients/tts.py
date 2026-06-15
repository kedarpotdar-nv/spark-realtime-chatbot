"""Text-to-Speech clients.

Two interchangeable backends with the same surface:
  - KokoroTTS       : default, CPU-friendly, sub-500ms TTFT on short utterances
  - ChatterboxTTS   : experimental, better voice quality, CUDA-only,
                       slower first-chunk (see bench/tts.json)

Both implement:
    synth_to_file(text, out_path)
    async synth_stream(text)         -> async generator of WAV bytes
    synth_stream_chunks(text, voice) -> sync generator of (pcm16, sr) tuples

Factory: create_tts(cfg) picks the right one based on cfg.engine.
"""

import io
import re
import time
from pathlib import Path
from typing import AsyncGenerator, Generator, Optional, Tuple

import numpy as np
import torch
import soundfile as sf
from kokoro import KPipeline

from config import TTSConfig


class KokoroTTS:
    def __init__(self, cfg: TTSConfig):
        print(f"[TTS] Loading Kokoro pipeline (lang={cfg.lang_code}, voice={cfg.voice})...")
        self.cfg = cfg

        # Force CPU for TTS - Blackwell GPU (sm_121) CUDA kernels not fully supported yet
        device = cfg.device if hasattr(cfg, 'device') else 'cpu'
        if device == 'cuda' and not self._check_cuda_support():
            print("[TTS] WARNING: CUDA not fully supported, falling back to CPU")
            device = 'cpu'

        self.pipeline = KPipeline(lang_code=cfg.lang_code, device=device)
        print(f"[TTS] Pipeline loaded on device: {device}")

    def _check_cuda_support(self) -> bool:
        """Check if CUDA is fully supported (no JIT compilation errors)."""
        try:
            # Quick test to see if basic CUDA ops work
            if not torch.cuda.is_available():
                return False
            x = torch.randn(10, device='cuda')
            y = x * 2
            del x, y
            torch.cuda.empty_cache()
            return True
        except Exception as e:
            print(f"[TTS] CUDA check failed: {e}")
            return False

    def synth_to_file(self, text: str, out_path: Path) -> None:
        """Synthesize text to audio file."""
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

    async def synth_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        """Stream audio as WAV file chunks."""
        if not text.strip():
            yield b""
            return

        # Generate full audio first (Kokoro generates in chunks anyway)
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
            yield b""
            return

        # Concatenate all chunks
        audio = np.concatenate(chunks)
        sr = 24000

        # Write to BytesIO as WAV
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, audio, sr, subtype="PCM_16", format="WAV")
        wav_data = wav_buffer.getvalue()
        
        # Stream in chunks
        chunk_size = 8192
        for i in range(0, len(wav_data), chunk_size):
            yield wav_data[i:i + chunk_size]

    def synth_stream_chunks(self, text: str, voice: str = None):
        """Stream audio chunks as they're generated (for WebSocket).
        Yields (audio_data: bytes, sample_rate: int) tuples.
        
        Args:
            text: Text to synthesize
            voice: Voice to use (defaults to self.cfg.voice)
        """
        if not text.strip():
            return

        sr = 24000
        voice_to_use = voice or self.cfg.voice
        
        # Split text into sentences for more incremental generation
        import re
        sentences = re.split(r'([.!?]\s+)', text)
        # Recombine sentences with their punctuation
        text_chunks = []
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                text_chunks.append(sentences[i] + sentences[i + 1])
            else:
                text_chunks.append(sentences[i])
        if len(sentences) % 2 == 1:
            text_chunks.append(sentences[-1])
        
        # If no sentence breaks, split by commas or just use whole text
        if len(text_chunks) == 1 and len(text) > 100:
            text_chunks = re.split(r'(,\s+)', text)
            text_chunks = [text_chunks[i] + (text_chunks[i+1] if i+1 < len(text_chunks) else '') 
                          for i in range(0, len(text_chunks), 2)]
        
        if not text_chunks:
            text_chunks = [text]
        
        print(f"[TTS] Splitting into {len(text_chunks)} chunks for streaming")
        
        import time

        # Generate and yield audio for each text chunk
        for chunk_idx, text_chunk in enumerate(text_chunks):
            if not text_chunk.strip():
                continue

            chunk_start = time.perf_counter()

            # Generate audio for this chunk
            generator = self.pipeline(
                text_chunk.strip(),
                voice=voice_to_use,
                speed=self.cfg.speed,
                split_pattern=r"\n+",
            )

            for _, _, audio in generator:
                if isinstance(audio, torch.Tensor):
                    audio = audio.detach().cpu().numpy()
                audio = audio.astype("float32")

                # Convert to int16 PCM
                audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
                audio_bytes = audio_int16.tobytes()

                elapsed_ms = (time.perf_counter() - chunk_start) * 1000
                audio_duration_ms = len(audio_int16) / sr * 1000
                rtf = elapsed_ms / audio_duration_ms if audio_duration_ms > 0 else 0
                print(f"[TTS] ⏱️ Chunk {chunk_idx + 1}: {elapsed_ms:.0f}ms → {audio_duration_ms:.0f}ms audio (RTF: {rtf:.2f}x) '{text_chunk[:30]}...'")

                # Yield audio data with sample rate immediately
                yield (audio_bytes, sr)


class ChatterboxTTS:
    """Experimental Chatterbox-Turbo backend. Better voice quality than Kokoro,
    but slower per-utterance on GB10 and no native streaming (generate() is
    one-shot). We wrap .generate() and fake-stream by splitting on sentence
    boundaries ourselves, same as the Kokoro streaming path.
    """

    def __init__(self, cfg: TTSConfig):
        print(f"[TTS] Loading Chatterbox-Turbo (device={cfg.device})...")
        self.cfg = cfg
        try:
            from chatterbox.tts import ChatterboxTTS as _Cb
        except ImportError as e:
            raise RuntimeError(
                "chatterbox-tts not installed. Install with `pip install chatterbox-tts` "
                "in an environment with a matching torch+torchaudio."
            ) from e
        device = cfg.device if torch.cuda.is_available() and cfg.device == "cuda" else "cpu"
        self._model = _Cb.from_pretrained(device=device)
        self.sample_rate = int(getattr(self._model, "sr", 24000))
        self._exag = cfg.chatterbox_exaggeration
        self._cfgw = cfg.chatterbox_cfg_weight
        self._device = device
        print(f"[TTS] Chatterbox loaded on {device}, sr={self.sample_rate}")

    def _generate(self, text: str) -> np.ndarray:
        wav = self._model.generate(text, exaggeration=self._exag, cfg_weight=self._cfgw)
        if hasattr(wav, "detach"):
            wav = wav.detach().cpu().numpy()
        if wav.ndim == 2:
            wav = wav[0] if wav.shape[0] == 1 else wav.mean(axis=0)
        return wav.astype(np.float32)

    def synth_to_file(self, text: str, out_path: Path) -> None:
        if not text.strip():
            sf.write(str(out_path), np.zeros(1600, dtype=np.float32), self.sample_rate)
            return
        wav = self._generate(text)
        sf.write(str(out_path), wav, self.sample_rate, subtype="PCM_16")

    async def synth_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        if not text.strip():
            yield b""
            return
        wav = self._generate(text)
        buf = io.BytesIO()
        sf.write(buf, wav, self.sample_rate, subtype="PCM_16", format="WAV")
        data = buf.getvalue()
        for i in range(0, len(data), 8192):
            yield data[i:i + 8192]

    def synth_stream_chunks(self, text: str, voice: Optional[str] = None):
        """Split on sentence boundaries and generate each separately for pseudo-streaming."""
        if not text.strip():
            return
        sentences = re.split(r'([.!?]\s+)', text)
        chunks = []
        for i in range(0, len(sentences) - 1, 2):
            chunks.append(sentences[i] + (sentences[i + 1] if i + 1 < len(sentences) else ""))
        if len(sentences) % 2 == 1 and sentences[-1].strip():
            chunks.append(sentences[-1])
        if not chunks:
            chunks = [text]
        print(f"[TTS/chatterbox] {len(chunks)} chunk(s)")
        sr = self.sample_rate
        for idx, chunk_text in enumerate(chunks):
            if not chunk_text.strip():
                continue
            t0 = time.perf_counter()
            wav = self._generate(chunk_text.strip())
            synth_ms = (time.perf_counter() - t0) * 1000
            pcm16 = (np.clip(wav, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
            audio_ms = (len(wav) / sr) * 1000
            rtf = synth_ms / audio_ms if audio_ms else 0
            print(f"[TTS/chatterbox] chunk {idx+1}: {synth_ms:.0f}ms → {audio_ms:.0f}ms (RTF {rtf:.2f})")
            yield (pcm16, sr)


def create_tts(cfg: TTSConfig):
    """Factory: pick the TTS backend based on cfg.engine."""
    engine = (cfg.engine or "kokoro").lower()
    if engine == "chatterbox":
        return ChatterboxTTS(cfg)
    if engine != "kokoro":
        print(f"[TTS] unknown engine {engine!r}, falling back to kokoro")
    return KokoroTTS(cfg)

