"""Configuration dataclasses for spark-realtime-chatbot."""

import os
from dataclasses import dataclass
from pathlib import Path


# Unified endpoint — one Qwen3.6-35B-A3B served by Ollama handles text, vision, and reasoning.
OLLAMA_DEFAULT_URL = "http://localhost:11434/v1/chat/completions"
QWEN36_MODEL = "qwen3.6:35b-a3b"


@dataclass
class ASRConfig:
    """Automatic Speech Recognition configuration."""
    mode: str = os.getenv("ASR_MODE", "api")  # "api" for server, "local" for in-process
    api_url: str = os.getenv("ASR_API_URL", "http://localhost:8000/v1/audio/transcriptions")
    api_key: str = os.getenv("ASR_API_KEY", "dummy-key")
    model: str = os.getenv("ASR_MODEL", "Systran/faster-whisper-small.en")
    # Local mode settings
    device: str = os.getenv("ASR_DEVICE", "cuda")  # "cuda" or "cpu"
    compute_type: str = os.getenv("ASR_COMPUTE_TYPE", "float16")  # "float16", "int8", "float32"


@dataclass
class LLMConfig:
    """Language Model configuration (Qwen3.6-35B-A3B via Ollama, OpenAI-compatible)."""
    base_url: str = os.getenv("LLM_SERVER_URL", OLLAMA_DEFAULT_URL)
    model: str = os.getenv("LLM_MODEL", QWEN36_MODEL)
    temperature: float = float(os.getenv("LLM_TEMP", "0.7"))
    max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "4096"))
    # "none" disables <think> on Ollama → voice-path TTFT ~200ms. Set "high" for reasoning tasks.
    reasoning_effort: str = os.getenv("LLM_REASONING_EFFORT", "none")
    backend: str = os.getenv("LLM_BACKEND", "ollama")


@dataclass
class VLMConfig:
    """Vision Language Model configuration — same Qwen3.6 endpoint, vision is native."""
    base_url: str = os.getenv("VLM_SERVER_URL", OLLAMA_DEFAULT_URL)
    model: str = os.getenv("VLM_MODEL", QWEN36_MODEL)
    temperature: float = float(os.getenv("VLM_TEMP", "0.3"))
    max_tokens: int = int(os.getenv("VLM_MAX_TOKENS", "4000"))
    reasoning_effort: str = os.getenv("VLM_REASONING_EFFORT", "none")


@dataclass
class ReasoningConfig:
    """Deep-reasoning configuration — same Qwen3.6 model, reasoning_effort=high."""
    base_url: str = os.getenv("REASONING_SERVER_URL", OLLAMA_DEFAULT_URL)
    model: str = os.getenv("REASONING_MODEL", QWEN36_MODEL)
    temperature: float = float(os.getenv("REASONING_TEMP", "0.7"))
    max_tokens: int = int(os.getenv("REASONING_MAX_TOKENS", "4096"))
    reasoning_effort: str = os.getenv("REASONING_EFFORT", "high")


@dataclass
class TTSConfig:
    """Text-to-Speech configuration.

    engine="kokoro" is the default and current production choice on DGX Spark.
    engine="chatterbox" is supported as an experimental backend for A/B testing
    voice quality; slower TTFT on GB10 today (see bench/tts.json).
    """
    engine: str = os.getenv("TTS_ENGINE", "kokoro")  # "kokoro" | "chatterbox"
    lang_code: str = os.getenv("KOKORO_LANG", "a")
    voice: str = os.getenv("KOKORO_VOICE", "af_bella")
    speed: float = float(os.getenv("KOKORO_SPEED", "1.2"))
    overlap_llm: bool = os.getenv("TTS_OVERLAP", "false").lower() == "true"  # Overlap TTS with LLM streaming
    device: str = os.getenv("TTS_DEVICE", "cuda")  # "cuda" (default, ~70× realtime on GB10 with torch cu130) or "cpu"
    # Chatterbox-specific
    chatterbox_exaggeration: float = float(os.getenv("CHATTERBOX_EXAGGERATION", "0.5"))
    chatterbox_cfg_weight: float = float(os.getenv("CHATTERBOX_CFG_WEIGHT", "0.5"))


# Directory paths
AUDIO_DIR = Path("audio_cache")
AUDIO_DIR.mkdir(exist_ok=True)

STATIC_DIR = Path("static")
STATIC_DIR.mkdir(exist_ok=True)

# Audio settings
SAMPLE_RATE = 16000

# Workspace root for file operations
WORKSPACE_ROOT = Path(os.getenv("WORKSPACE_ROOT", Path.cwd())).resolve()

# FFmpeg path
FFMPEG_PATH = os.getenv("FFMPEG_PATH", "ffmpeg")
