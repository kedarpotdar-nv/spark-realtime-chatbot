"""Unit tests for Hermes bridge session headers and memory helpers (mocked HTTP)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from agents.hermes import HermesAgentBridge
from agents.hermes_sessions import HermesSessionStore


@pytest.fixture
def temp_session_store(tmp_path: Path, monkeypatch):
    store_path = tmp_path / ".hermes_sessions.json"
    monkeypatch.setenv("HERMES_SESSION_STORE", str(store_path))
    monkeypatch.setenv("HERMES_SESSION_KEY", "spark-realtime:test-user")
    return HermesSessionStore(store_path)


def test_session_store_persists_ids(temp_session_store: HermesSessionStore):
    first = temp_session_store.get_identity("browser-a")
    second = temp_session_store.get_identity("browser-a")
    assert first.hermes_session_id == second.hermes_session_id
    assert first.session_key == "spark-realtime:test-user"

    other = temp_session_store.get_identity("browser-b")
    assert other.hermes_session_id != first.hermes_session_id


def test_headers_include_session_id_and_key(monkeypatch):
    monkeypatch.setenv("HERMES_API_KEY", "test-key")
    bridge = HermesAgentBridge()
    headers = bridge._headers("default")
    assert "X-Hermes-Session-Id" in headers
    assert "X-Hermes-Session-Key" in headers
    assert headers["Authorization"] == "Bearer test-key"


def test_merge_memory_into_system_prompt():
    bridge = HermesAgentBridge()
    base = "You are Spark."
    block = "---\n## Long-term memory (from Hermes)\nUser prefers dark mode.\n---"
    merged = bridge.merge_memory_into_system_prompt(base, block)
    assert "User prefers dark mode" in merged
    updated = bridge.merge_memory_into_system_prompt(merged, block.replace("dark", "light"))
    assert updated.count("Long-term memory (from Hermes)") == 1
    assert "light mode" in updated


def test_record_task_summary_keeps_last_three():
    bridge = HermesAgentBridge()
    for i in range(5):
        bridge._record_task_summary(f"summary-{i}")
    assert bridge._recent_task_summaries == ["summary-2", "summary-3", "summary-4"]


@pytest.mark.asyncio
async def test_ensure_hermes_api_session_accepts_201(monkeypatch):
    class FakeResponse:
        def __init__(self, status, payload):
            self.status = status
            self._payload = payload

        async def json(self):
            return self._payload

        async def text(self):
            return json.dumps(self._payload)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

    class FakeSession:
        def post(self, *args, **kwargs):
            return FakeResponse(201, {"id": "srv_sess_abc123"})

    class FakeManager:
        async def get_session(self):
            return FakeSession()

    monkeypatch.setattr("agents.hermes.get_http_manager", lambda: FakeManager())
    bridge = HermesAgentBridge()
    bridge._transport_mode = "gateway_chat"

    resolved = await bridge.ensure_hermes_api_session("default")
    assert resolved.hermes_session_id == "srv_sess_abc123"