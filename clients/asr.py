"""Automatic Speech Recognition client - supports API server or in-process."""

import asyncio
import io
import os
import queue
import threading
from typing import AsyncGenerator

import aiohttp
import numpy as np
import soundfile as sf

from config import ASRConfig, SAMPLE_RATE
from .http_session import get_http_manager


class BaseASR:
    """Base ASR class with shared utilities."""

    def _clean_repetitive_text(self, text: str) -> str:
        """Remove repetitive patterns from ASR output (Whisper hallucination fix)."""
        if not text:
            return text

        words = text.split()
        if len(words) < 6:
            return text

        # Detect repeated words (e.g., "okay, okay, okay, okay")
        word_counts = {}
        for word in words:
            clean_word = word.lower().strip('.,!?')
            word_counts[clean_word] = word_counts.get(clean_word, 0) + 1

        # If any single word is more than 50% of the text, it's likely hallucination
        for word, count in word_counts.items():
            if count > len(words) * 0.5 and count > 5:
                print(f"[ASR] Detected repetitive hallucination: '{word}' repeated {count} times")
                first_occurrence = text.split(word)[0] + word
                return first_occurrence.strip(' ,')

        # Detect repeated phrases (e.g., "let's do that, let's do that")
        for pattern_len in range(2, 5):
            if len(words) >= pattern_len * 3:
                pattern = ' '.join(words[:pattern_len]).lower()
                pattern_count = text.lower().count(pattern)
                if pattern_count > 3:
                    print(f"[ASR] Detected repetitive phrase: '{pattern}' repeated {pattern_count} times")
                    return pattern.capitalize()

        return text

    def warmup(self):
        """Pre-load model to eliminate cold-start latency. Override in subclasses."""
        pass

    async def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe audio - to be implemented by subclasses."""
        raise NotImplementedError

    async def transcribe_streaming(self, audio: np.ndarray) -> AsyncGenerator[str, None]:
        """Stream transcription segments as they're recognized.

        Yields partial text as segments complete. Default implementation
        just yields the full transcription at once.
        """
        text = await self.transcribe(audio)
        if text:
            yield text


class FasterWhisperASR(BaseASR):
    """ASR client using OpenAI-compatible faster-whisper server."""

    def __init__(self, cfg: ASRConfig = None):
        self.cfg = cfg or ASRConfig()
        print(f"[ASR] Using faster-whisper server at '{self.cfg.api_url}'")
        print(f"[ASR] Model: {self.cfg.model}")
        self.streaming_enabled = os.getenv("ASR_STREAMING", "true").lower() == "true"

    def _audio_to_wav_bytes(self, audio: np.ndarray) -> bytes:
        """Convert numpy audio array to WAV file bytes."""
        audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
        wav_buffer = io.BytesIO()
        sf.write(wav_buffer, audio, SAMPLE_RATE, subtype="PCM_16", format="WAV")
        return wav_buffer.getvalue()

    async def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe audio using OpenAI-compatible API."""
        if audio.size == 0:
            return ""

        # Skip silent audio
        audio_max = np.abs(audio).max()
        if audio_max < 0.001:
            return ""

        # Limit audio length to prevent hallucination (max 30 seconds)
        max_samples = SAMPLE_RATE * 30
        if len(audio) > max_samples:
            print(f"[ASR] Trimming audio from {len(audio)/SAMPLE_RATE:.1f}s to 30s")
            audio = audio[:max_samples]

        wav_bytes = self._audio_to_wav_bytes(audio)

        # Prepare multipart form data
        data = aiohttp.FormData()
        wav_file = io.BytesIO(wav_bytes)
        wav_file.seek(0)
        data.add_field('file', wav_file, filename='audio.wav', content_type='audio/wav')
        data.add_field('model', self.cfg.model)
        data.add_field('language', 'en')

        import time
        start_time = time.perf_counter()
        headers = {'Authorization': f'Bearer {self.cfg.api_key}'}

        try:
            http_manager = get_http_manager()
            session = await http_manager.get_session()
            async with session.post(
                self.cfg.api_url,
                data=data,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as resp:
                if resp.status != 200:
                    error_text = await resp.text()
                    print(f"[ASR] API error {resp.status}: {error_text}")
                    return ""

                result = await resp.json()
                text = result.get('text', '').strip()
                text = self._clean_repetitive_text(text)

                elapsed = (time.perf_counter() - start_time) * 1000
                audio_duration = len(audio) / SAMPLE_RATE * 1000
                if text:
                    print(f"[ASR] ⏱️ {elapsed:.0f}ms for {audio_duration:.0f}ms audio (RTF: {elapsed/audio_duration:.2f}x) → '{text}'")
                return text
        except asyncio.TimeoutError:
            print("[ASR] Request timeout")
            return ""
        except Exception as e:
            print(f"[ASR] Error: {e}")
            return ""


class LocalWhisperASR(BaseASR):
    """In-process ASR using faster-whisper library directly."""

    def __init__(self, cfg: ASRConfig = None):
        self.cfg = cfg or ASRConfig()
        self._model = None
        print(f"[ASR] Using local faster-whisper (model={self.cfg.model}, device={self.cfg.device})")

    def warmup(self):
        """Pre-load model and run dummy inference to eliminate cold-start latency."""
        import time
        print("[ASR] Warming up model...")
        start = time.perf_counter()
        
        # Load the model
        model = self._get_model()
        
        # Run a dummy inference with 1 second of silence
        dummy_audio = np.zeros(SAMPLE_RATE, dtype=np.float32)
        segments, _ = model.transcribe(dummy_audio, language="en")
        # Consume the generator
        list(segments)
        
        elapsed = (time.perf_counter() - start) * 1000
        print(f"[ASR] ✅ Warmup complete ({elapsed:.0f}ms)")

    def _get_model(self):
        """Lazy-load the whisper model."""
        if self._model is None:
            from faster_whisper import WhisperModel

            device = self.cfg.device
            compute_type = self.cfg.compute_type

            # Try CUDA first, fall back to CPU if not available
            if device == "cuda":
                try:
                    import ctranslate2
                    cuda_types = ctranslate2.get_supported_compute_types("cuda")
                    if not cuda_types or "float16" not in cuda_types:
                        print(f"[ASR] CUDA not available in CTranslate2, falling back to CPU")
                        device = "cpu"
                        compute_type = "float32"  # float32 is safe fallback for CPU
                    else:
                        print(f"[ASR] CUDA compute types available: {cuda_types}")
                except Exception as e:
                    print(f"[ASR] Error checking CUDA support: {e}, falling back to CPU")
                    device = "cpu"
                    compute_type = "float32"

            print(f"[ASR] Loading model {self.cfg.model} on {device} ({compute_type})...")
            self._model = WhisperModel(
                self.cfg.model,
                device=device,
                compute_type=compute_type,
            )
            print(f"[ASR] Model loaded successfully")
        return self._model

    async def transcribe(self, audio: np.ndarray) -> str:
        """Transcribe audio using local faster-whisper."""
        if audio.size == 0:
            return ""

        # Skip silent audio
        audio_max = np.abs(audio).max()
        if audio_max < 0.001:
            return ""

        # Limit audio length to prevent hallucination (max 30 seconds)
        max_samples = SAMPLE_RATE * 30
        if len(audio) > max_samples:
            print(f"[ASR] Trimming audio from {len(audio)/SAMPLE_RATE:.1f}s to 30s")
            audio = audio[:max_samples]

        # Ensure audio is float32 and normalized
        audio = np.clip(audio, -1.0, 1.0).astype(np.float32)

        import time
        start_time = time.perf_counter()

        try:
            # Run transcription in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            text = await loop.run_in_executor(None, self._transcribe_sync, audio)
            text = self._clean_repetitive_text(text)

            elapsed = (time.perf_counter() - start_time) * 1000
            audio_duration = len(audio) / SAMPLE_RATE * 1000
            if text:
                print(f"[ASR] ⏱️ {elapsed:.0f}ms for {audio_duration:.0f}ms audio (RTF: {elapsed/audio_duration:.2f}x) → '{text}'")
            return text
        except Exception as e:
            print(f"[ASR] Error: {e}")
            return ""

    def _transcribe_sync(self, audio: np.ndarray) -> str:
        """Synchronous transcription (runs in thread pool)."""
        model = self._get_model()
        segments, info = model.transcribe(
            audio,
            language="en",
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )
        # Collect all segment texts
        text = " ".join(segment.text.strip() for segment in segments)
        return text.strip()

    def _transcribe_sync_streaming(self, audio: np.ndarray, segment_queue: queue.Queue):
        """Synchronous transcription that puts segments into a queue as they're recognized."""
        try:
            model = self._get_model()
            segments, info = model.transcribe(
                audio,
                language="en",
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
            )
            # Yield segments as they're transcribed
            accumulated_text = ""
            for segment in segments:
                segment_text = segment.text.strip()
                if segment_text:
                    accumulated_text += (" " if accumulated_text else "") + segment_text
                    # Put accumulated text so far (for incremental display)
                    segment_queue.put(("segment", accumulated_text))

            # Signal completion
            segment_queue.put(("done", accumulated_text))
        except Exception as e:
            segment_queue.put(("error", str(e)))

    async def transcribe_streaming(self, audio: np.ndarray) -> AsyncGenerator[str, None]:
        """Stream transcription segments as they're recognized."""
        if audio.size == 0:
            return

        # Skip silent audio
        audio_max = np.abs(audio).max()
        if audio_max < 0.001:
            return

        # Limit audio length
        max_samples = SAMPLE_RATE * 30
        if len(audio) > max_samples:
            print(f"[ASR] Trimming audio from {len(audio)/SAMPLE_RATE:.1f}s to 30s")
            audio = audio[:max_samples]

        # Ensure audio is float32 and normalized
        audio = np.clip(audio, -1.0, 1.0).astype(np.float32)

        # Use a queue to communicate between sync thread and async code
        segment_queue = queue.Queue()

        # Start transcription in background thread
        thread = threading.Thread(
            target=self._transcribe_sync_streaming,
            args=(audio, segment_queue)
        )
        thread.start()

        try:
            last_text = ""
            while True:
                # Check queue with timeout to allow async cooperation
                try:
                    # Use asyncio to periodically check the queue
                    msg_type, text = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: segment_queue.get(timeout=0.1)
                    )

                    if msg_type == "segment":
                        # Only yield if text changed
                        if text != last_text:
                            last_text = text
                            yield text
                    elif msg_type == "done":
                        # Final result - clean up and yield final text
                        final_text = self._clean_repetitive_text(text)
                        if final_text and final_text != last_text:
                            yield final_text
                        break
                    elif msg_type == "error":
                        print(f"[ASR] Streaming error: {text}")
                        break
                except queue.Empty:
                    # Queue empty, continue waiting
                    await asyncio.sleep(0.01)
        finally:
            thread.join(timeout=5.0)


def create_asr(cfg: ASRConfig = None) -> BaseASR:
    """Factory function to create the appropriate ASR client based on config."""
    cfg = cfg or ASRConfig()
    if cfg.mode == "local":
        return LocalWhisperASR(cfg)
    else:
        return FasterWhisperASR(cfg)
