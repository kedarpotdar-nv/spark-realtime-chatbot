"""Hermes Agent (Nous Research) asynchronous background bridge.

Integrates with `NousResearch/hermes-agent` via:
1. Hermes Gateway API Server (`http://127.0.0.1:8642` by default):
   - Uses `/v1/runs` + `/v1/runs/{id}/events` SSE when `run_submission` and
     `run_events_sse` capabilities are available, with `/v1/runs/{id}/stop`
     for server-side cancellation.
   - Uses `/v1/chat/completions` with `stream=True` (handling `hermes.tool.progress`
     SSE frames, `reasoning_content`, and `content` deltas) for OpenAI-compatible
     gateway mode or direct `llama-server` agentic fallback.
2. Hermes CLI (`hermes chat -q ...`) subprocess fallback when the binary is
   installed locally without a running gateway daemon.
3. Built-in local agentic execution loop against the configured OpenAI-compatible
   endpoint (`HERMES_UPSTREAM_URL` / `LLM_SERVER_URL`, e.g., `llama-server` on
   port 8080) when `HERMES_ENABLED=true` is set and the user wants local
   artifact-generating execution in the shared workspace.

All network traffic reuses the shared persistent `HTTPSessionManager` connection
pool (`aiohttp.TCPConnector` with keep-alive) for sub-millisecond dispatch.
"""

from __future__ import annotations

import asyncio
import json
import mimetypes
import os
import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import aiohttp

from clients.http_session import get_http_manager
from config import AGENT_WORKSPACE_DIR, HermesConfig
from .hermes_sessions import HermesSessionIdentity, get_hermes_session_store
from .base import (
    AgentBridge,
    AgentEvent,
    AgentTask,
    ArtifactInfo,
    EventCallback,
    TaskStatus,
)


PREVIEWABLE_EXTENSIONS = {
    ".html", ".htm", ".md", ".txt", ".json", ".py", ".js", ".ts", ".tsx",
    ".jsx", ".css", ".sh", ".yaml", ".yml", ".toml", ".csv", ".svg", ".xml",
    ".sql", ".rs", ".go", ".c", ".cpp", ".h", ".log",
    ".png", ".jpg", ".jpeg", ".webp", ".gif", ".mp3", ".wav", ".flac", ".ogg",
}

MEDIA_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".mp3", ".wav", ".flac", ".ogg"}

IGNORED_DIRS = {
    ".git", "__pycache__", "node_modules", ".venv", "venv", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", "dist", "build",
}


class HermesAgentBridge(AgentBridge):
    """Non-blocking background bridge to Nous Research's Hermes Agent."""

    agent_id = "hermes"
    display_name = "Hermes Agent"

    def __init__(self, cfg: Optional[HermesConfig] = None):
        self.cfg = cfg or HermesConfig()
        self.workspace_dir = Path(self.cfg.workspace_dir).resolve()
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

        self._tasks: Dict[str, AgentTask] = {}
        self._asyncio_tasks: Dict[str, asyncio.Task] = {}
        self._cancel_flags: Set[str] = set()
        self._semaphore = asyncio.Semaphore(max(1, self.cfg.max_concurrent))
        self._capabilities: Dict[str, Any] = {}
        self._transport_mode: str = "unconfigured"  # "gateway_runs", "gateway_chat", "cli", "local_openai", "unconfigured"
        self._last_probe_at: float = 0.0
        self._session_store = get_hermes_session_store()
        self._api_sessions_ready: Set[str] = set()
        self._memory_cache: Dict[str, str] = {}
        self._memory_cached_at: Dict[str, float] = {}
        self._memory_refresh_tasks: Dict[str, asyncio.Task] = {}
        self._recent_task_summaries: List[str] = []
        self._memory_section_marker = "## Long-term memory (from Hermes)"

    # ------------------------------------------------------------------
    # Configuration & Capability Discovery
    # ------------------------------------------------------------------

    def is_configured(self) -> bool:
        """Check whether Hermes is enabled or reachable via env/config/CLI."""
        raw_enabled = os.getenv("HERMES_ENABLED", "").strip().lower()
        if raw_enabled in ("false", "0", "no", "off"):
            return False
        if raw_enabled in ("true", "1", "yes", "on"):
            return True
        if os.getenv("HERMES_URL", "").strip() or os.getenv("HERMES_API_KEY", "").strip():
            return True
        if self.cfg.url or self.cfg.api_key:
            return True
        cli = os.getenv("HERMES_CLI_PATH", self.cfg.cli_path).strip() or "hermes"
        if shutil.which(cli):
            return True
        if self._transport_mode not in ("unconfigured", ""):
            return True
        return False

    def _base_url(self) -> str:
        raw = os.getenv("HERMES_URL", "").strip() or self.cfg.url or "http://127.0.0.1:8642"
        url = raw.rstrip("/")
        if url.endswith("/v1/chat/completions"):
            url = url[: -len("/v1/chat/completions")]
        elif url.endswith("/v1"):
            url = url[: -len("/v1")]
        return url

    def _resolve_identity(self, identity: str = "default") -> HermesSessionIdentity:
        return self._session_store.get_identity(identity or "default")

    def _headers(
        self,
        identity: str = "default",
        *,
        session_id: Optional[str] = None,
        session_key: Optional[str] = None,
    ) -> Dict[str, str]:
        resolved = self._resolve_identity(identity)
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "X-Hermes-Session-Id": session_id or resolved.hermes_session_id,
            "X-Hermes-Session-Key": session_key or resolved.session_key,
        }
        if self.cfg.api_key:
            headers["Authorization"] = f"Bearer {self.cfg.api_key}"
        return headers

    def _headers_for_task(self, agent_task: AgentTask) -> Dict[str, str]:
        identity = agent_task.hermes_identity or "default"
        return self._headers(
            identity,
            session_id=agent_task.session_id,
        )

    async def ensure_hermes_api_session(self, identity: str = "default") -> HermesSessionIdentity:
        """Create or reaffirm durable Hermes gateway session via POST /api/sessions (404 = skip)."""
        ident = (identity or "default").strip() or "default"
        resolved = self._resolve_identity(ident)
        if ident in self._api_sessions_ready:
            return resolved

        if self._transport_mode not in ("gateway_runs", "gateway_chat"):
            return resolved

        session = await get_http_manager().get_session()
        base = self._base_url()
        payload = {
            "session_id": resolved.hermes_session_id,
            "session_key": resolved.session_key,
        }
        try:
            async with session.post(
                f"{base}/api/sessions",
                headers=self._headers(ident),
                json=payload,
                timeout=aiohttp.ClientTimeout(total=6.0),
            ) as resp:
                if resp.status in (200, 201):
                    data = await resp.json()
                    new_id = data.get("id") or data.get("session_id")
                    if new_id:
                        self._session_store.set_hermes_session_id(ident, str(new_id))
                        resolved = self._resolve_identity(ident)
                elif resp.status != 404:
                    text = await resp.text()
                    print(f"[Hermes] POST /api/sessions returned {resp.status}: {text[:120]}")
        except Exception as exc:
            print(f"[Hermes] ensure_hermes_api_session warning: {exc}")

        self._api_sessions_ready.add(ident)
        return resolved

    async def warmup(self) -> None:
        """Probe Hermes gateway / upstream to establish a warm persistent session."""
        await self.check_status(force_probe=True)
        try:
            await self.ensure_hermes_api_session("default")
        except Exception as exc:
            print(f"[Hermes] warmup session notice: {exc}")

    async def ensure_ready(self) -> Dict[str, Any]:
        """Lazy-initialize Hermes on first use (delegate_task / REST).

        Skips aggressive probing at server startup; probes only when a task is
        actually delegated. Returns the status payload after probing.
        """
        status = await self.check_status(force_probe=True)
        if not self.is_configured() and self._transport_mode == "unconfigured":
            # Allow explicit HERMES_ENABLED=true to fall through to local_openai
            raw = os.getenv("HERMES_ENABLED", "").strip().lower()
            if raw in ("true", "1", "yes", "on"):
                self._transport_mode = "local_openai"
                status = self._build_status_payload(available=True, configured=True)
        return status

    async def check_status(self, force_probe: bool = False) -> Dict[str, Any]:
        """Probe Hermes availability and return detailed status for the UI."""
        now = time.time()
        configured = self.is_configured()

        if not configured and not force_probe:
            self._transport_mode = "unconfigured"
            return self._build_status_payload(available=False, configured=False)

        if force_probe or (now - self._last_probe_at > 15.0) or self._transport_mode == "unconfigured":
            self._last_probe_at = now
            await self._probe_backend()

        available = self._transport_mode != "unconfigured"
        return self._build_status_payload(available=available, configured=configured or available)

    async def _probe_backend(self) -> None:
        """Detect the fastest available Hermes execution transport."""
        if os.getenv("HERMES_ENABLED", "").lower() in ("false", "0", "no", "off"):
            self._transport_mode = "unconfigured"
            return

        base = self._base_url()
        session = await get_http_manager().get_session()

        # 1. Probe Hermes Gateway (/v1/capabilities and /health)
        try:
            async with session.get(
                f"{base}/v1/capabilities",
                headers=self._headers(),
                timeout=aiohttp.ClientTimeout(total=2.5, connect=1.0),
            ) as resp:
                if resp.status == 200:
                    caps = await resp.json()
                    if isinstance(caps, dict) and caps.get("platform") == "hermes-agent":
                        self._capabilities = caps.get("features", {})
                        if self._capabilities.get("run_submission") and self._capabilities.get("run_events_sse"):
                            self._transport_mode = "gateway_runs"
                        else:
                            self._transport_mode = "gateway_chat"
                        return
        except Exception:
            pass

        # 2. Check /health or /v1/models on HERMES_URL
        has_explicit_url = bool(os.getenv("HERMES_URL", "").strip() or self.cfg.url)
        try:
            async with session.get(
                f"{base}/health",
                headers=self._headers(),
                timeout=aiohttp.ClientTimeout(total=2.0, connect=1.0),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    if isinstance(data, dict) and data.get("platform") == "hermes-agent":
                        self._transport_mode = "gateway_chat"
                        return
                    if isinstance(data, dict) and data.get("status") == "ok" and has_explicit_url:
                        self._transport_mode = "gateway_chat"
                        return
        except Exception:
            pass

        if has_explicit_url:
            try:
                async with session.get(
                    f"{base}/v1/models",
                    headers=self._headers(),
                    timeout=aiohttp.ClientTimeout(total=2.0, connect=1.0),
                ) as resp:
                    if resp.status == 200:
                        self._transport_mode = "gateway_chat"
                        return
            except Exception:
                pass

        # 3. Check if `hermes` CLI binary is installed
        cli = os.getenv("HERMES_CLI_PATH", self.cfg.cli_path).strip() or "hermes"
        if shutil.which(cli):
            self._transport_mode = "cli"
            return

        # 4. If HERMES_ENABLED=true or HERMES_URL is explicitly configured, allow local OpenAI-compatible agent loop
        if os.getenv("HERMES_ENABLED", "").strip().lower() in ("true", "1", "yes", "on") or has_explicit_url:
            self._transport_mode = "local_openai"
            return

        self._transport_mode = "unconfigured"

    def _build_status_payload(self, available: bool, configured: bool) -> Dict[str, Any]:
        active_tasks = [
            t.to_dict() for t in self.list_tasks()
            if t.status in (TaskStatus.QUEUED, TaskStatus.RUNNING)
        ]
        return {
            "agent_id": self.agent_id,
            "display_name": self.display_name,
            "configured": configured,
            "available": available,
            "transport": self._transport_mode,
            "url": self._base_url(),
            "model": self.cfg.model,
            "provider": self.cfg.provider or "default",
            "workspace_dir": str(self.workspace_dir),
            "max_concurrent": self.cfg.max_concurrent,
            "timeout": self.cfg.timeout,
            "active_count": len(active_tasks),
            "total_tasks": len(self._tasks),
            "capabilities": self._capabilities,
        }

    # ------------------------------------------------------------------
    # Task Management & Dispatch
    # ------------------------------------------------------------------

    def get_task(self, task_id: str) -> Optional[AgentTask]:
        return self._tasks.get(task_id)

    def list_tasks(self) -> List[AgentTask]:
        return sorted(self._tasks.values(), key=lambda t: t.created_at, reverse=True)

    async def submit_task(
        self,
        task: str,
        context: str = "",
        category: str = "general",
        identity: str = "default",
        image_b64: Optional[str] = None,
        session_key: Optional[str] = None,
        on_event: Optional[EventCallback] = None,
    ) -> AgentTask:
        """Create and schedule a background Hermes task without blocking the caller."""
        # Lazy probe / activate transport on first delegation
        await self.ensure_ready()
        ident = (identity or "default").strip() or "default"
        session_identity = await self.ensure_hermes_api_session(ident)

        agent_task = AgentTask.create(
            agent_id=self.agent_id,
            task=task,
            context=context,
            category=category,
            workspace_dir=str(self.workspace_dir),
        )
        agent_task.hermes_identity = ident
        agent_task.session_id = session_identity.hermes_session_id
        if image_b64:
            agent_task.image_b64 = image_b64.strip()
        if session_key:
            self._session_store.set_session_key(ident, session_key)
        self._tasks[agent_task.task_id] = agent_task

        bg_task = asyncio.create_task(
            self._execute_task_wrapper(agent_task, on_event=on_event)
        )
        self._asyncio_tasks[agent_task.task_id] = bg_task
        bg_task.add_done_callback(lambda _: self._asyncio_tasks.pop(agent_task.task_id, None))
        return agent_task

    async def cancel_task(self, task_id: str) -> bool:
        """Cancel a queued or running Hermes task."""
        agent_task = self._tasks.get(task_id)
        if not agent_task:
            return False

        if agent_task.status not in (TaskStatus.QUEUED, TaskStatus.RUNNING):
            return False

        self._cancel_flags.add(task_id)

        # If running on Hermes Gateway /v1/runs, send stop signal
        if agent_task.remote_run_id:
            try:
                session = await get_http_manager().get_session()
                base = self._base_url()
                await session.post(
                    f"{base}/v1/runs/{agent_task.remote_run_id}/stop",
                    headers=self._headers_for_task(agent_task),
                    timeout=aiohttp.ClientTimeout(total=5.0),
                )
            except Exception as e:
                print(f"[Hermes] Remote stop warning for {agent_task.remote_run_id}: {e}")

        bg = self._asyncio_tasks.get(task_id)
        if bg and not bg.done():
            bg.cancel()

        agent_task.status = TaskStatus.CANCELLED
        agent_task.finished_at = time.time()
        agent_task.error = "Task cancelled by user."
        return True

    # ------------------------------------------------------------------
    # Background Execution Pipeline
    # ------------------------------------------------------------------

    async def _emit(
        self,
        on_event: Optional[EventCallback],
        agent_task: AgentTask,
        event_type: str,
        **data: Any,
    ) -> None:
        event = AgentEvent(task_id=agent_task.task_id, event_type=event_type, data=data)
        if on_event:
            try:
                await on_event(event)
            except Exception as e:
                print(f"[Hermes] Event callback error ({event_type}): {e}")

    async def _execute_task_wrapper(
        self,
        agent_task: AgentTask,
        on_event: Optional[EventCallback] = None,
    ) -> None:
        """Acquire concurrency slot, run with timeout, detect artifacts, and emit lifecycle events."""
        await self._emit(
            on_event,
            agent_task,
            "task.queued",
            task=agent_task.to_dict(),
        )

        async with self._semaphore:
            if agent_task.task_id in self._cancel_flags:
                agent_task.status = TaskStatus.CANCELLED
                agent_task.finished_at = time.time()
                await self._emit(on_event, agent_task, "task.cancelled", task=agent_task.to_dict())
                return

            agent_task.status = TaskStatus.RUNNING
            agent_task.started_at = time.time()

            # Snapshot workspace files before execution to detect new/modified artifacts
            before_files = self._snapshot_workspace()

            # Ensure transport is probed
            if self._transport_mode == "unconfigured":
                await self._probe_backend()

            await self._emit(
                on_event,
                agent_task,
                "task.started",
                task=agent_task.to_dict(),
                transport=self._transport_mode,
            )

            try:
                await asyncio.wait_for(
                    self._dispatch_to_backend(agent_task, on_event=on_event),
                    timeout=float(self.cfg.timeout),
                )

                if agent_task.task_id in self._cancel_flags:
                    agent_task.status = TaskStatus.CANCELLED
                    agent_task.finished_at = time.time()
                    await self._emit(on_event, agent_task, "task.cancelled", task=agent_task.to_dict())
                    return

                # Extract any code blocks with filenames from the response and materialize in workspace
                created_from_output = self._materialize_artifacts_from_output(
                    agent_task.output, agent_task.task_id
                )

                # Compare workspace snapshot to find newly created or modified files
                after_files = self._snapshot_workspace()
                changed_paths = [
                    rel_path for rel_path, mtime in after_files.items()
                    if rel_path not in before_files or mtime > before_files[rel_path]
                ]
                for p in created_from_output:
                    if p not in changed_paths:
                        changed_paths.append(p)

                agent_task.artifacts = sorted(changed_paths)
                for artifact_path in agent_task.artifacts:
                    await self._emit(
                        on_event,
                        agent_task,
                        "artifact.created",
                        artifact=artifact_path,
                    )

                agent_task.status = TaskStatus.COMPLETED
                agent_task.finished_at = time.time()
                agent_task.spoken_summary = self.build_spoken_summary(agent_task)
                self._record_task_summary(agent_task.spoken_summary)
                self.invalidate_memory_cache(agent_task.hermes_identity)

                await self._emit(
                    on_event,
                    agent_task,
                    "task.completed",
                    task=agent_task.to_dict(),
                    spoken_summary=agent_task.spoken_summary,
                )

            except asyncio.TimeoutError:
                agent_task.status = TaskStatus.TIMED_OUT
                agent_task.finished_at = time.time()
                agent_task.error = f"Task timed out after {self.cfg.timeout}s"
                agent_task.spoken_summary = "The background task timed out before finishing."
                await self._emit(
                    on_event,
                    agent_task,
                    "task.failed",
                    task=agent_task.to_dict(),
                    error=agent_task.error,
                    spoken_summary=agent_task.spoken_summary,
                )
            except asyncio.CancelledError:
                agent_task.status = TaskStatus.CANCELLED
                agent_task.finished_at = time.time()
                agent_task.error = "Task cancelled"
                await self._emit(
                    on_event,
                    agent_task,
                    "task.cancelled",
                    task=agent_task.to_dict(),
                )
            except Exception as exc:
                agent_task.status = TaskStatus.FAILED
                agent_task.finished_at = time.time()
                agent_task.error = str(exc)
                agent_task.spoken_summary = f"Hermes encountered an error while working on the task."
                print(f"[Hermes] Task {agent_task.task_id} failed: {exc}")
                await self._emit(
                    on_event,
                    agent_task,
                    "task.failed",
                    task=agent_task.to_dict(),
                    error=agent_task.error,
                    spoken_summary=agent_task.spoken_summary,
                )

    async def _dispatch_to_backend(
        self,
        agent_task: AgentTask,
        on_event: Optional[EventCallback] = None,
    ) -> None:
        """Route execution to the active Hermes transport."""
        has_image = bool((agent_task.image_b64 or "").strip())

        if has_image and self._transport_mode in ("gateway_runs", "gateway_chat"):
            try:
                await self._run_via_chat_completions(
                    agent_task,
                    endpoint_url=f"{self._base_url()}/v1/chat/completions",
                    on_event=on_event,
                    attach_camera=True,
                )
                return
            except Exception as e:
                print(f"[Hermes] Multimodal chat failed ({e}), falling back")

        if self._transport_mode == "gateway_runs" and not has_image:
            try:
                await self._run_via_gateway_runs(agent_task, on_event=on_event)
                return
            except Exception as e:
                print(f"[Hermes] /v1/runs failed ({e}), falling back to /v1/chat/completions")

        if self._transport_mode in ("gateway_runs", "gateway_chat"):
            try:
                await self._run_via_chat_completions(
                    agent_task,
                    endpoint_url=f"{self._base_url()}/v1/chat/completions",
                    on_event=on_event,
                )
                return
            except Exception as e:
                print(f"[Hermes] Gateway chat completions failed ({e}), trying fallback")

        if self._transport_mode == "cli" or shutil.which(self.cfg.cli_path):
            await self._run_via_cli(agent_task, on_event=on_event)
            return

        # Local OpenAI-compatible endpoint (e.g., llama-server on :8080) or configured HERMES_URL
        target_url = os.getenv("HERMES_UPSTREAM_URL") or self.cfg.upstream_url
        if (os.getenv("HERMES_URL") or self.cfg.url) and self._transport_mode != "local_openai":
            base = self._base_url()
            target_url = f"{base}/v1/chat/completions"

        await self._run_via_chat_completions(
            agent_task,
            endpoint_url=target_url,
            on_event=on_event,
            is_local_fallback=(self._transport_mode == "local_openai"),
        )

    # ------------------------------------------------------------------
    # Transport 1: Hermes Gateway `/v1/runs` + `/v1/runs/{id}/events`
    # ------------------------------------------------------------------

    async def _run_via_gateway_runs(
        self,
        agent_task: AgentTask,
        on_event: Optional[EventCallback] = None,
    ) -> None:
        session = await get_http_manager().get_session()
        base = self._base_url()
        headers = self._headers_for_task(agent_task)
        headers["Idempotency-Key"] = f"spark-{agent_task.task_id}"

        prompt_text = self._build_agent_prompt(agent_task)
        payload: Dict[str, Any] = {
            "input": prompt_text,
            "instructions": self._build_system_instructions(agent_task),
            "session_id": agent_task.session_id,
        }
        if self.cfg.model:
            payload["model"] = self.cfg.model
        if self.cfg.provider:
            payload["provider"] = self.cfg.provider

        async with session.post(
            f"{base}/v1/runs",
            headers=headers,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=15.0),
        ) as resp:
            if resp.status not in (200, 201, 202):
                err_text = await resp.text()
                raise RuntimeError(f"Hermes /v1/runs returned {resp.status}: {err_text[:200]}")
            run_data = await resp.json()

        run_id = run_data.get("run_id")
        if not run_id:
            raise RuntimeError("Hermes /v1/runs did not return a run_id")

        agent_task.remote_run_id = run_id
        self._record_step(agent_task, "run.started", f"Hermes run {run_id} admitted")
        await self._emit(
            on_event,
            agent_task,
            "step.progress",
            step={"type": "run.started", "title": f"Hermes run {run_id}", "status": "running"},
            task=agent_task.to_dict(),
        )

        # Subscribe to SSE stream `/v1/runs/{run_id}/events`
        async with session.get(
            f"{base}/v1/runs/{run_id}/events",
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=float(self.cfg.timeout), sock_read=120.0),
        ) as sse_resp:
            if sse_resp.status != 200:
                err_text = await sse_resp.text()
                raise RuntimeError(f"Hermes /v1/runs/{run_id}/events returned {sse_resp.status}: {err_text[:200]}")

            current_event = "message"
            buffer = ""
            async for raw_line in sse_resp.content:
                if agent_task.task_id in self._cancel_flags:
                    return
                line_decoded = raw_line.decode("utf-8", errors="ignore")
                buffer += line_decoded
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.rstrip("\r")
                    if not line:
                        current_event = "message"
                        continue
                    if line.startswith(":"):
                        # SSE keepalive comment
                        continue
                    if line.startswith("event:"):
                        current_event = line[6:].strip()
                        continue
                    if line.startswith("data:"):
                        data_str = line[5:].strip()
                        if data_str == "[DONE]":
                            return
                        try:
                            data = json.loads(data_str)
                        except json.JSONDecodeError:
                            continue

                        await self._handle_hermes_sse_event(
                            agent_task, current_event, data, on_event
                        )
                        if current_event in ("run.completed", "run.failed", "run.cancelled", "run.interrupted", "done"):
                            return

        # If stream closed without final output, poll `/v1/runs/{run_id}` once
        if not agent_task.output:
            async with session.get(
                f"{base}/v1/runs/{run_id}",
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10.0),
            ) as status_resp:
                if status_resp.status == 200:
                    state = await status_resp.json()
                    if state.get("output"):
                        agent_task.output = state["output"]
                    if state.get("status") == "failed":
                        raise RuntimeError(state.get("error") or "Hermes run failed")

    async def _handle_hermes_sse_event(
        self,
        agent_task: AgentTask,
        event_name: str,
        data: Dict[str, Any],
        on_event: Optional[EventCallback],
    ) -> None:
        """Process a single event from Hermes `/v1/runs/{id}/events` or `/api/sessions/{id}/chat/stream`."""
        evt_type = data.get("type") or event_name

        if evt_type in ("tool.started", "hermes.tool.progress", "subagent.start"):
            tool_name = data.get("tool") or data.get("tool_name") or data.get("name") or "tool"
            preview = data.get("preview") or data.get("delta") or ""
            if isinstance(data.get("args"), dict):
                preview = preview or json.dumps(data["args"])[:160]
            step = self._record_step(agent_task, "tool.started", tool_name, detail=str(preview)[:200])
            await self._emit(
                on_event,
                agent_task,
                "tool.started",
                tool=tool_name,
                preview=str(preview)[:200],
                step=step,
                task=agent_task.to_dict(),
            )

        elif evt_type in ("tool.completed", "tool.failed", "subagent.complete"):
            tool_name = data.get("tool") or data.get("tool_name") or "tool"
            is_err = bool(data.get("error")) or evt_type == "tool.failed"
            preview = data.get("preview") or data.get("summary") or ""
            duration = data.get("duration")
            step = self._record_step(
                agent_task,
                "tool.failed" if is_err else "tool.completed",
                tool_name,
                detail=str(preview)[:200],
                duration=duration,
            )
            await self._emit(
                on_event,
                agent_task,
                "tool.failed" if is_err else "tool.completed",
                tool=tool_name,
                preview=str(preview)[:200],
                duration=duration,
                step=step,
                task=agent_task.to_dict(),
            )

        elif evt_type in ("message.delta", "assistant.delta", "response.output_text.delta"):
            delta = data.get("delta") or data.get("text") or data.get("content") or ""
            if delta:
                agent_task.output += delta
                await self._emit(
                    on_event,
                    agent_task,
                    "message.delta",
                    delta=delta,
                    output_length=len(agent_task.output),
                )

        elif evt_type in ("message.interim", "assistant.commentary"):
            text = data.get("text") or ""
            if text:
                step = self._record_step(agent_task, "commentary", "Agent commentary", detail=text[:200])
                await self._emit(
                    on_event,
                    agent_task,
                    "step.progress",
                    step=step,
                    task=agent_task.to_dict(),
                )

        elif evt_type in ("assistant.completed", "run.completed"):
            content = data.get("content") or data.get("output") or ""
            if content and not agent_task.output:
                agent_task.output = content

        elif evt_type in ("run.failed", "error"):
            err_msg = data.get("error") or data.get("message") or "Hermes run failed"
            raise RuntimeError(str(err_msg))

    # ------------------------------------------------------------------
    # Transport 2: OpenAI-Compatible `/v1/chat/completions` (SSE)
    # ------------------------------------------------------------------

    async def _run_via_chat_completions(
        self,
        agent_task: AgentTask,
        endpoint_url: str,
        on_event: Optional[EventCallback] = None,
        is_local_fallback: bool = False,
        attach_camera: bool = False,
    ) -> None:
        """Stream from Hermes `/v1/chat/completions` (or upstream OpenAI server)."""
        session = await get_http_manager().get_session()
        headers = self._headers_for_task(agent_task)

        model_name = self.cfg.model
        if is_local_fallback and model_name == "hermes-agent":
            model_name = os.getenv("LLM_MODEL", "qwen3-vl")

        user_prompt = self._build_agent_prompt(agent_task)
        image_b64 = (agent_task.image_b64 or "").strip()
        if attach_camera and image_b64:
            if image_b64.startswith("data:"):
                image_url = image_b64
            else:
                image_url = f"data:image/jpeg;base64,{image_b64}"
            user_content: Any = [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]
        else:
            user_content = user_prompt

        messages = [
            {"role": "system", "content": self._build_system_instructions(agent_task)},
            {"role": "user", "content": user_content},
        ]

        payload: Dict[str, Any] = {
            "model": model_name,
            "messages": messages,
            "stream": True,
            "temperature": 0.4,
            "max_tokens": 4096,
        }
        if self.cfg.provider and not is_local_fallback:
            payload["provider"] = self.cfg.provider

        step_label = "Hermes Gateway stream" if not is_local_fallback else "Local model stream"
        step = self._record_step(agent_task, "stream.started", step_label, detail=endpoint_url)
        await self._emit(on_event, agent_task, "step.progress", step=step, task=agent_task.to_dict())

        async with session.post(
            endpoint_url,
            headers=headers,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=float(self.cfg.timeout), sock_read=120.0),
        ) as resp:
            if resp.status != 200:
                err_text = await resp.text()
                raise RuntimeError(f"Endpoint {endpoint_url} returned {resp.status}: {err_text[:200]}")

            current_event = "message"
            buffer = ""
            reasoning_chars = 0

            async for raw_line in resp.content:
                if agent_task.task_id in self._cancel_flags:
                    return
                buffer += raw_line.decode("utf-8", errors="ignore")
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.rstrip("\r")
                    if not line:
                        current_event = "message"
                        continue
                    if line.startswith(":"):
                        continue
                    if line.startswith("event:"):
                        current_event = line[6:].strip()
                        continue
                    if not line.startswith("data:"):
                        continue

                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        return

                    try:
                        data = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue

                    # Hermes custom tool progress event in Chat Completions SSE
                    if current_event == "hermes.tool.progress" or data.get("object") == "hermes.tool.progress":
                        tool_name = data.get("tool") or data.get("tool_name") or "tool"
                        preview = data.get("preview") or data.get("status") or ""
                        t_step = self._record_step(agent_task, "tool.started", tool_name, detail=str(preview)[:200])
                        await self._emit(
                            on_event,
                            agent_task,
                            "tool.started",
                            tool=tool_name,
                            preview=str(preview)[:200],
                            step=t_step,
                            task=agent_task.to_dict(),
                        )
                        continue

                    choices = data.get("choices") or []
                    if not choices:
                        continue
                    choice = choices[0]
                    delta = choice.get("delta") or {}
                    finish_reason = choice.get("finish_reason")

                    reasoning_delta = delta.get("reasoning_content") or delta.get("reasoning") or ""
                    if reasoning_delta:
                        if reasoning_chars == 0:
                            r_step = self._record_step(agent_task, "thinking", "Analyzing & planning")
                            await self._emit(on_event, agent_task, "step.progress", step=r_step, task=agent_task.to_dict())
                        reasoning_chars += len(reasoning_delta)
                        await self._emit(
                            on_event,
                            agent_task,
                            "thinking.delta",
                            delta=reasoning_delta,
                        )

                    content_delta = delta.get("content") or ""
                    if content_delta:
                        # Strip channel tags if present
                        if "<|channel|>" not in content_delta:
                            agent_task.output += content_delta
                            await self._emit(
                                on_event,
                                agent_task,
                                "message.delta",
                                delta=content_delta,
                                output_length=len(agent_task.output),
                            )

                    if finish_reason in ("error", "cancelled"):
                        err_msg = (data.get("error") or {}).get("message") or f"Stream ended with {finish_reason}"
                        raise RuntimeError(err_msg)

    # ------------------------------------------------------------------
    # Transport 3: Hermes CLI (`hermes chat -q ...`)
    # ------------------------------------------------------------------

    async def _run_via_cli(
        self,
        agent_task: AgentTask,
        on_event: Optional[EventCallback] = None,
    ) -> None:
        """Execute task via the local `hermes` CLI in non-interactive mode."""
        prompt = self._build_agent_prompt(agent_task)
        cmd = [self.cfg.cli_path, "chat", "-q", prompt]
        if self.cfg.model and self.cfg.model != "hermes-agent":
            cmd.extend(["-m", self.cfg.model])

        step = self._record_step(agent_task, "cli.started", "Running hermes CLI", detail=" ".join(cmd[:4]))
        await self._emit(on_event, agent_task, "step.progress", step=step, task=agent_task.to_dict())

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(self.workspace_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            assert proc.stdout is not None
            while True:
                if agent_task.task_id in self._cancel_flags:
                    proc.terminate()
                    return
                line = await proc.stdout.readline()
                if not line:
                    break
                decoded = line.decode("utf-8", errors="ignore")
                agent_task.output += decoded
                await self._emit(
                    on_event,
                    agent_task,
                    "message.delta",
                    delta=decoded,
                    output_length=len(agent_task.output),
                )

            await proc.wait()
            if proc.returncode != 0 and not agent_task.output.strip():
                stderr_text = ""
                if proc.stderr:
                    stderr_text = (await proc.stderr.read()).decode("utf-8", errors="ignore")
                raise RuntimeError(f"hermes CLI exited with code {proc.returncode}: {stderr_text[:200]}")
        finally:
            if proc.returncode is None:
                try:
                    proc.kill()
                except ProcessLookupError:
                    pass

    # ------------------------------------------------------------------
    # Long-term memory context (for local realtime model)
    # ------------------------------------------------------------------

    def _record_task_summary(self, summary: str) -> None:
        text = (summary or "").strip()
        if not text:
            return
        self._recent_task_summaries.append(text)
        self._recent_task_summaries = self._recent_task_summaries[-3:]

    def invalidate_memory_cache(self, identity: str = "default") -> None:
        ident = (identity or "default").strip() or "default"
        self._memory_cache.pop(ident, None)
        self._memory_cached_at.pop(ident, None)

    def get_cached_memory_context(self, identity: str = "default") -> str:
        """Return cached Hermes memory block (empty if stale/unavailable). Never blocks."""
        if not self.is_configured():
            return ""
        ident = (identity or "default").strip() or "default"
        now = time.time()
        ttl = float(self.cfg.memory_ttl)
        cached = self._memory_cache.get(ident, "")
        cached_at = self._memory_cached_at.get(ident, 0.0)
        if cached and (now - cached_at) < ttl:
            return self._format_memory_block(cached, use_recent=True)
        if self._recent_task_summaries:
            return self._format_memory_block("", use_recent=True)
        return ""

    def schedule_memory_refresh(self, identity: str = "default") -> None:
        """Refresh memory context in the background without blocking callers."""
        if not self.is_configured():
            return
        ident = (identity or "default").strip() or "default"
        existing = self._memory_refresh_tasks.get(ident)
        if existing and not existing.done():
            return
        self._memory_refresh_tasks[ident] = asyncio.create_task(self._refresh_memory_context(ident))

    async def get_memory_context(self, identity: str = "default") -> str:
        """Cached memory context; triggers background refresh when stale."""
        ident = (identity or "default").strip() or "default"
        cached = self.get_cached_memory_context(ident)
        now = time.time()
        ttl = float(self.cfg.memory_ttl)
        cached_at = self._memory_cached_at.get(ident, 0.0)
        if not self._memory_cache.get(ident) or (now - cached_at) >= ttl:
            self.schedule_memory_refresh(ident)
        return cached

    async def _refresh_memory_context(self, identity: str) -> None:
        if self._transport_mode == "unconfigured":
            await self._probe_backend()
        if self._transport_mode not in ("gateway_runs", "gateway_chat"):
            return

        ident = (identity or "default").strip() or "default"
        await self.ensure_hermes_api_session(ident)
        session = await get_http_manager().get_session()
        base = self._base_url()
        headers = dict(self._headers(ident))
        headers["Accept"] = "application/json"

        messages = [
            {
                "role": "system",
                "content": (
                    "Summarize what you remember about the user and outcomes of recent delegated tasks. "
                    "Respond in plain text, at most 150 words, no markdown headers."
                ),
            },
            {
                "role": "user",
                "content": "Provide a concise memory summary for the Spark realtime voice assistant.",
            },
        ]
        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "stream": False,
            "max_tokens": 320,
            "temperature": 0.2,
        }
        if self.cfg.provider:
            payload["provider"] = self.cfg.provider

        try:
            async with session.post(
                f"{base}/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=float(self.cfg.memory_fetch_timeout)),
            ) as resp:
                if resp.status != 200:
                    return
                data = await resp.json()
        except Exception as exc:
            print(f"[Hermes] memory context fetch failed: {exc}")
            return

        choices = data.get("choices") or []
        if not choices:
            return
        message = choices[0].get("message") or {}
        content = (message.get("content") or "").strip()
        if not content:
            return
        words = content.split()
        if len(words) > 150:
            content = " ".join(words[:150])
        self._memory_cache[ident] = content
        self._memory_cached_at[ident] = time.time()

    def _format_memory_block(self, hermes_summary: str, *, use_recent: bool = True) -> str:
        lines = [f"---\n{self._memory_section_marker}"]
        if hermes_summary:
            lines.append(hermes_summary.strip())
        if use_recent and self._recent_task_summaries:
            lines.append("\nRecent delegated tasks:")
            for item in self._recent_task_summaries[-3:]:
                lines.append(f"- {item}")
        lines.append("---")
        return "\n".join(lines)

    def merge_memory_into_system_prompt(self, base_prompt: str, memory_block: str) -> str:
        if not memory_block:
            return base_prompt
        marker = self._memory_section_marker
        pattern = re.compile(
            r"\n---\n" + re.escape(marker) + r"[\s\S]*?\n---",
            re.MULTILINE,
        )
        stripped = pattern.sub("", base_prompt).rstrip()
        return stripped + "\n\n" + memory_block

    # ------------------------------------------------------------------
    # Prompt & Artifact Helpers
    # ------------------------------------------------------------------

    def _build_system_instructions(self, agent_task: AgentTask) -> str:
        return (
            "You are Hermes Agent, an autonomous software engineering and research agent "
            "running alongside Spark Realtime on an NVIDIA DGX Spark. "
            f"Your shared workspace directory is: {self.workspace_dir}\n"
            "When generating files, web apps, scripts, or reports:\n"
            "1. If you have file-writing tools available, write generated files directly into the workspace directory.\n"
            "2. Also include any generated files as fenced code blocks with an explicit filename header "
            "in the format ```language filename=\"path/to/file.ext\" so the workspace artifact extractor "
            "can index and preview them immediately.\n"
            "3. End your response with a concise 1-2 sentence `## Summary` section suitable for reading aloud."
        )

    def _build_agent_prompt(self, agent_task: AgentTask) -> str:
        parts = [f"Task ({agent_task.category}): {agent_task.task}"]
        if agent_task.context:
            parts.append(f"\nContext:\n{agent_task.context}")
        parts.append(f"\nWorkspace Path: {self.workspace_dir}")
        return "\n".join(parts)

    def _record_step(
        self,
        agent_task: AgentTask,
        step_type: str,
        title: str,
        detail: str = "",
        duration: Optional[float] = None,
    ) -> Dict[str, Any]:
        step = {
            "id": f"step_{ len(agent_task.steps) + 1 }",
            "type": step_type,
            "title": title,
            "detail": detail,
            "timestamp": time.time(),
        }
        if duration is not None:
            step["duration"] = duration
        agent_task.steps.append(step)
        return step

    def _snapshot_workspace(self) -> Dict[str, float]:
        """Return mapping of relative file path -> mtime in workspace_dir."""
        snapshot: Dict[str, float] = {}
        if not self.workspace_dir.exists():
            return snapshot
        for root, dirs, files in os.walk(self.workspace_dir):
            dirs[:] = [d for d in dirs if d not in IGNORED_DIRS and not d.startswith(".")]
            for fname in files:
                if fname.startswith("."):
                    continue
                full = Path(root) / fname
                try:
                    rel = str(full.relative_to(self.workspace_dir))
                    snapshot[rel] = full.stat().st_mtime
                except OSError:
                    continue
        return snapshot

    def _materialize_artifacts_from_output(self, output: str, task_id: str) -> List[str]:
        """Extract fenced code blocks with filenames (or standalone HTML/Markdown documents) into workspace."""
        if not output or not output.strip():
            return []

        created: List[str] = []

        # Pattern 1: ```lang filename="foo.py" or ```lang foo.py
        pattern = re.compile(
            r'```([a-zA-Z0-9_+-]*)\s+(?:filename=["\']?([^"\'\n`]+)["\']?|file=["\']?([^"\'\n`]+)["\']?|([a-zA-Z0-9_./-]+\.[a-zA-Z0-9]+))\s*\n([\s\S]*?)```'
        )
        for match in pattern.finditer(output):
            lang, fn1, fn2, fn3, code = match.groups()
            raw_name = (fn1 or fn2 or fn3 or "").strip()
            if not raw_name:
                continue
            safe_rel = self._sanitize_rel_path(raw_name)
            if not safe_rel:
                continue
            target = (self.workspace_dir / safe_rel).resolve()
            if not str(target).startswith(str(self.workspace_dir)):
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(code, encoding="utf-8")
            created.append(str(target.relative_to(self.workspace_dir)))

        # Pattern 2: If no named blocks were found, check if there is a complete HTML block or save a task report
        if not created:
            html_match = re.search(r"```html\s*\n([\s\S]*?)```", output, re.IGNORECASE)
            if html_match and "<html" in html_match.group(1).lower():
                rel_name = f"{task_id}_app.html"
                target = self.workspace_dir / rel_name
                target.write_text(html_match.group(1), encoding="utf-8")
                created.append(rel_name)

            # Always save a markdown report for completed non-empty tasks so it's inspectable in artifacts
            if len(output.strip()) > 60:
                report_name = f"{task_id}_report.md"
                report_path = self.workspace_dir / report_name
                report_path.write_text(output.strip(), encoding="utf-8")
                created.append(report_name)

        return created

    @staticmethod
    def _sanitize_rel_path(raw_path: str) -> Optional[str]:
        cleaned = raw_path.strip().lstrip("/\\")
        if not cleaned or ".." in Path(cleaned).parts:
            return None
        return cleaned

    def list_artifacts(self) -> List[ArtifactInfo]:
        """List all files in the shared workspace directory sorted by modification time (newest first)."""
        artifacts: List[ArtifactInfo] = []
        if not self.workspace_dir.exists():
            return artifacts

        # Map artifact path -> task_id if known
        artifact_to_task: Dict[str, str] = {}
        for t in self._tasks.values():
            for a in t.artifacts:
                artifact_to_task[a] = t.task_id

        for root, dirs, files in os.walk(self.workspace_dir):
            dirs[:] = [d for d in dirs if d not in IGNORED_DIRS and not d.startswith(".")]
            for fname in files:
                if fname.startswith("."):
                    continue
                full = Path(root) / fname
                try:
                    stat = full.stat()
                    rel = str(full.relative_to(self.workspace_dir))
                    ext = full.suffix.lower()
                    mime, _ = mimetypes.guess_type(fname)
                    artifacts.append(
                        ArtifactInfo(
                            path=rel,
                            name=fname,
                            size=stat.st_size,
                            modified_at=stat.st_mtime,
                            mime_type=mime or "text/plain",
                            previewable=ext in PREVIEWABLE_EXTENSIONS or (mime or "").startswith("text/"),
                            task_id=artifact_to_task.get(rel),
                        )
                    )
                except OSError:
                    continue

        artifacts.sort(key=lambda a: a.modified_at, reverse=True)
        return artifacts

    def read_artifact(self, rel_path: str, max_bytes: int = 250_000) -> Optional[Dict[str, Any]]:
        """Safely read an artifact file from the shared workspace for UI preview."""
        safe = self._sanitize_rel_path(rel_path)
        if not safe:
            return None
        target = (self.workspace_dir / safe).resolve()
        if not str(target).startswith(str(self.workspace_dir)) or not target.is_file():
            return None

        stat = target.stat()
        ext = target.suffix.lower()
        mime, _ = mimetypes.guess_type(target.name)
        if ext in MEDIA_EXTENSIONS or (mime or "").startswith(("image/", "audio/")):
            return {
                "path": safe,
                "name": target.name,
                "size": stat.st_size,
                "modified_at": stat.st_mtime,
                "mime_type": mime or "application/octet-stream",
                "extension": ext,
                "binary": True,
                "content": "",
                "workspace_url": f"/workspace/{safe}",
                "truncated": False,
            }
        raw = target.read_bytes()[:max_bytes]
        content = raw.decode("utf-8", errors="replace")
        return {
            "path": safe,
            "name": target.name,
            "size": stat.st_size,
            "modified_at": stat.st_mtime,
            "mime_type": mime or "text/plain",
            "extension": ext,
            "content": content,
            "truncated": stat.st_size > max_bytes,
        }

    @staticmethod
    def build_spoken_summary(agent_task: AgentTask) -> str:
        """Extract a clean, 1-2 sentence TTS-friendly summary from the agent's output."""
        text = (agent_task.output or "").strip()
        if not text:
            if agent_task.artifacts:
                return f"Hermes finished the task and created {len(agent_task.artifacts)} file{'s' if len(agent_task.artifacts) != 1 else ''} in the workspace."
            return "Hermes finished working on your task."

        # Check if there's an explicit ## Summary section
        summary_match = re.search(
            r"(?:^|\n)#+\s*Summary\s*\n+([\s\S]+?)(?:\n#+|\Z)",
            text,
            re.IGNORECASE,
        )
        candidate = summary_match.group(1).strip() if summary_match else text

        # Strip code blocks, headers, markdown syntax
        candidate = re.sub(r"```[\s\S]*?```", "", candidate)
        candidate = re.sub(r"^#+\s+.*$", "", candidate, flags=re.MULTILINE)
        candidate = re.sub(r"^[-*+]\s+", "", candidate, flags=re.MULTILINE)
        candidate = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", candidate)
        candidate = re.sub(r"[*_`~>#]", "", candidate)
        candidate = re.sub(r"\s+", " ", candidate).strip()

        if not candidate:
            if agent_task.artifacts:
                return f"Task complete. I generated {agent_task.artifacts[0]} in the workspace."
            return "Hermes finished the task. You can view the full output in the activity panel."

        sentences = re.split(r"(?<=[.!?])\s+", candidate)
        sentences = [s.strip() for s in sentences if len(s.strip()) > 8]
        spoken = " ".join(sentences[:2]).strip()
        if len(spoken) > 240:
            spoken = spoken[:237].rsplit(" ", 1)[0] + "."
        return spoken or "Hermes finished the task."


# Global singleton instance
_hermes_bridge: Optional[HermesAgentBridge] = None


def get_hermes_bridge() -> HermesAgentBridge:
    """Get or create the singleton HermesAgentBridge."""
    global _hermes_bridge
    if _hermes_bridge is None:
        _hermes_bridge = HermesAgentBridge()
    return _hermes_bridge
