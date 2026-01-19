"""Client modules for external services."""

from .http_session import HTTPSessionManager
from .asr import FasterWhisperASR, LocalWhisperASR, create_asr
from .llm import LlamaCppClient
from .vlm import VLMClient
from .nemotron import NemotronClient
from .tts import KokoroTTS

__all__ = [
    "HTTPSessionManager",
    "FasterWhisperASR",
    "LocalWhisperASR",
    "create_asr",
    "LlamaCppClient",
    "VLMClient",
    "NemotronClient",
    "KokoroTTS",
]
