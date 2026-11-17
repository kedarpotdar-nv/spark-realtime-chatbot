"""Configuration dataclasses for spark-realtime."""

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ASRConfig:
    """Automatic Speech Recognition configuration."""
    mode: str = os.getenv("ASR_MODE", "api")  # "api" for server, "local" for in-process
    api_url: str = os.getenv("ASR_API_URL", "http://localhost:9000/v1/audio/transcriptions")
    api_key: str = os.getenv("ASR_API_KEY", "dummy-key")
    model: str = os.getenv("ASR_MODEL", "Systran/faster-whisper-small.en")
    # Local mode settings
    device: str = os.getenv("ASR_DEVICE", "cuda")  # "cuda" or "cpu"
    compute_type: str = os.getenv("ASR_COMPUTE_TYPE", "float16")  # "float16", "int8", "float32"


@dataclass
class LLMConfig:
    """Language Model configuration."""
    base_url: str = os.getenv("LLM_SERVER_URL", "http://localhost:8080/v1/chat/completions")
    model: str = os.getenv("LLM_MODEL", "qwen3-vl")
    temperature: float = float(os.getenv("LLM_TEMP", "0.7"))
    max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "4096"))
    reasoning_effort: str = os.getenv("LLM_REASONING_EFFORT", "low")
    backend: str = os.getenv("LLM_BACKEND", "llama")


@dataclass
class VLMConfig:
    """Vision Language Model configuration (Qwen3-VL via llama.cpp)."""
    base_url: str = os.getenv("VLM_SERVER_URL", "http://localhost:8080/v1/chat/completions")
    model: str = os.getenv("VLM_MODEL", "qwen3-vl")
    temperature: float = float(os.getenv("VLM_TEMP", "0.3"))
    max_tokens: int = int(os.getenv("VLM_MAX_TOKENS", "4000"))


@dataclass
class NemotronConfig:
    """Nemotron 3 Nano configuration for deep reasoning tasks."""
    base_url: str = os.getenv("NEMOTRON_SERVER_URL", "http://localhost:8005/v1/chat/completions")
    model: str = os.getenv("NEMOTRON_MODEL", "nemotron-3-nano")
    temperature: float = float(os.getenv("NEMOTRON_TEMP", "1.0"))
    top_p: float = float(os.getenv("NEMOTRON_TOP_P", "1.0"))
    max_tokens: int = int(os.getenv("NEMOTRON_MAX_TOKENS", "512"))


@dataclass
class TTSConfig:
    """Text-to-Speech configuration."""
    engine: str = os.getenv("TTS_ENGINE", "kokoro")
    lang_code: str = os.getenv("KOKORO_LANG", "a")
    voice: str = os.getenv("KOKORO_VOICE", "af_bella")
    speed: float = float(os.getenv("KOKORO_SPEED", "1.2"))
    overlap_llm: bool = os.getenv("TTS_OVERLAP", "false").lower() == "true"  # Overlap TTS with LLM streaming
    device: str = os.getenv("TTS_DEVICE", "cpu")  # "cuda" or "cpu" - default CPU for GB10 compatibility


@dataclass
class ServerConfig:
    """HTTP / HTTPS server bind and TLS configuration."""
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "9443"))
    ssl_enabled: bool = os.getenv("SSL_ENABLED", "true").lower() == "true"
    ssl_keyfile: str = os.getenv("SSL_KEY", os.getenv("SSL_KEYFILE", "key.pem"))
    ssl_certfile: str = os.getenv("SSL_CERT", os.getenv("SSL_CERTFILE", "cert.pem"))


def _default_hermes_enabled() -> bool:
    raw = os.getenv("HERMES_ENABLED", "").strip().lower()
    if raw in ("false", "0", "no", "off"):
        return False
    if raw in ("true", "1", "yes", "on"):
        return True
    if os.getenv("HERMES_URL", "").strip() or os.getenv("HERMES_API_KEY", "").strip():
        return True
    cli = os.getenv("HERMES_CLI_PATH", "hermes").strip() or "hermes"
    if shutil.which(cli):
        return True
    return False


@dataclass
class HermesConfig:
    """Hermes Agent (Nous Research) background bridge configuration."""
    enabled: bool = field(default_factory=_default_hermes_enabled)
    url: str = field(default_factory=lambda: os.getenv("HERMES_URL", "").strip())
    api_key: str = field(default_factory=lambda: os.getenv("HERMES_API_KEY", os.getenv("API_SERVER_KEY", "")))
    model: str = field(default_factory=lambda: os.getenv("HERMES_MODEL", "hermes-agent"))
    provider: str = field(default_factory=lambda: os.getenv("HERMES_PROVIDER", ""))
    upstream_url: str = field(
        default_factory=lambda: os.getenv(
            "HERMES_UPSTREAM_URL",
            os.getenv("LLM_SERVER_URL", "http://localhost:8080/v1/chat/completions"),
        )
    )
    timeout: int = field(default_factory=lambda: int(os.getenv("HERMES_TIMEOUT", "600")))
    max_concurrent: int = field(default_factory=lambda: int(os.getenv("HERMES_MAX_CONCURRENT", "4")))
    cli_path: str = field(default_factory=lambda: os.getenv("HERMES_CLI_PATH", "hermes"))
    workspace_dir: str = field(
        default_factory=lambda: os.getenv(
            "HERMES_WORKSPACE",
            str(Path(os.getenv("WORKSPACE_ROOT", Path.cwd())).resolve() / "workspace"),
        )
    )


# Directory paths
AUDIO_DIR = Path("audio_cache")
AUDIO_DIR.mkdir(exist_ok=True)

STATIC_DIR = Path("static")
STATIC_DIR.mkdir(exist_ok=True)

# Audio settings
SAMPLE_RATE = 16000

# Workspace root for file operations
WORKSPACE_ROOT = Path(os.getenv("WORKSPACE_ROOT", Path.cwd())).resolve()

# Shared agent workspace directory for generated artifacts
AGENT_WORKSPACE_DIR = Path(os.getenv("HERMES_WORKSPACE", WORKSPACE_ROOT / "workspace")).resolve()
AGENT_WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)

# FFmpeg path
FFMPEG_PATH = os.getenv("FFMPEG_PATH", "ffmpeg")
