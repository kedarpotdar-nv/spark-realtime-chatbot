"""Persistent Hermes gateway session identity (session key + transcript session id)."""

from __future__ import annotations

import json
import os
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from config import AGENT_WORKSPACE_DIR


def _default_session_store_path() -> Path:
    raw = os.getenv("HERMES_SESSION_STORE", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return (AGENT_WORKSPACE_DIR / ".hermes_sessions.json").resolve()


def _default_memory_scope_key(identity: str) -> str:
    explicit = os.getenv("HERMES_SESSION_KEY", "").strip()
    if explicit:
        return explicit[:256]
    user_id = os.getenv("SPARK_USER_ID", identity or "default").strip() or "default"
    return f"spark-realtime:{user_id}"[:256]


@dataclass
class HermesSessionIdentity:
    """Resolved Hermes session headers for one browser/user identity."""

    identity: str
    session_key: str
    hermes_session_id: str


class HermesSessionStore:
    """JSON-backed store mapping client identity -> stable Hermes session ids."""

    def __init__(self, path: Optional[Path] = None):
        self.path = path or _default_session_store_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._data: Dict[str, Any] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.is_file():
            self._data = {}
            return
        try:
            raw = self.path.read_text(encoding="utf-8")
            parsed = json.loads(raw)
            self._data = parsed if isinstance(parsed, dict) else {}
        except (OSError, json.JSONDecodeError):
            self._data = {}

    def _save(self) -> None:
        try:
            tmp = self.path.with_suffix(self.path.suffix + ".tmp")
            tmp.write_text(json.dumps(self._data, indent=2), encoding="utf-8")
            tmp.replace(self.path)
        except OSError as exc:
            print(f"[Hermes] Failed to persist session store {self.path}: {exc}")

    def get_identity(self, identity: str = "default") -> HermesSessionIdentity:
        key = (identity or "default").strip() or "default"
        with self._lock:
            entry = self._data.get(key)
            if not isinstance(entry, dict):
                entry = {}
            session_key = str(entry.get("session_key") or _default_memory_scope_key(key))[:256]
            hermes_session_id = str(entry.get("hermes_session_id") or "").strip()
            if not hermes_session_id:
                hermes_session_id = str(uuid.uuid4())
            self._data[key] = {
                "session_key": session_key,
                "hermes_session_id": hermes_session_id,
            }
            self._save()
            return HermesSessionIdentity(
                identity=key,
                session_key=session_key,
                hermes_session_id=hermes_session_id,
            )

    def set_session_key(self, identity: str, session_key: str) -> None:
        if not session_key:
            return
        key = (identity or "default").strip() or "default"
        with self._lock:
            entry = self._data.get(key)
            if not isinstance(entry, dict):
                entry = {}
            entry["session_key"] = session_key[:256]
            entry.setdefault("hermes_session_id", str(uuid.uuid4()))
            self._data[key] = entry
            self._save()

    def set_hermes_session_id(self, identity: str, hermes_session_id: str) -> None:
        if not hermes_session_id:
            return
        key = (identity or "default").strip() or "default"
        with self._lock:
            entry = self._data.get(key)
            if not isinstance(entry, dict):
                entry = {}
            entry.setdefault("session_key", _default_memory_scope_key(key)[:256])
            entry["hermes_session_id"] = hermes_session_id
            self._data[key] = entry
            self._save()


_store: Optional[HermesSessionStore] = None


def get_hermes_session_store() -> HermesSessionStore:
    global _store
    if _store is None:
        _store = HermesSessionStore()
    return _store