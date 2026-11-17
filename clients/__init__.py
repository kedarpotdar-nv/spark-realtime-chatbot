"""Client modules for external services.

Heavy ML clients (ASR/TTS/etc.) are imported lazily so lightweight modules
like personal_fs and http_session work without the full dependency stack.
"""

from .http_session import HTTPSessionManager
from .personal_fs import PersonalFS, get_personal_fs

__all__ = [
    "HTTPSessionManager",
    "PersonalFS",
    "get_personal_fs",
    "FasterWhisperASR",
    "LocalWhisperASR",
    "create_asr",
    "LlamaCppClient",
    "VLMClient",
    "NemotronClient",
    "KokoroTTS",
]


def __getattr__(name: str):
    if name in ("FasterWhisperASR", "LocalWhisperASR", "create_asr"):
        from .asr import FasterWhisperASR, LocalWhisperASR, create_asr
        return {
            "FasterWhisperASR": FasterWhisperASR,
            "LocalWhisperASR": LocalWhisperASR,
            "create_asr": create_asr,
        }[name]
    if name == "LlamaCppClient":
        from .llm import LlamaCppClient
        return LlamaCppClient
    if name == "VLMClient":
        from .vlm import VLMClient
        return VLMClient
    if name == "NemotronClient":
        from .nemotron import NemotronClient
        return NemotronClient
    if name == "KokoroTTS":
        from .tts import KokoroTTS
        return KokoroTTS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
