"""Client modules for external services."""

from .http_session import HTTPSessionManager
from .asr import FasterWhisperASR, LocalWhisperASR, create_asr
from .llm import LlamaCppClient, ReasoningClient
from .vlm import VLMClient
from .tts import KokoroTTS, ChatterboxTTS, create_tts

# Back-compat alias: NemotronClient was the old name for the deep-reasoning client.
NemotronClient = ReasoningClient

__all__ = [
    "HTTPSessionManager",
    "FasterWhisperASR",
    "LocalWhisperASR",
    "create_asr",
    "LlamaCppClient",
    "ReasoningClient",
    "NemotronClient",
    "VLMClient",
    "KokoroTTS",
    "ChatterboxTTS",
    "create_tts",
]
