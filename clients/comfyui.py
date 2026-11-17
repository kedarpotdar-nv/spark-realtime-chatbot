"""Async ComfyUI API client (prompt queue, WebSocket progress, output download)."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional
from urllib.parse import urlparse, urlunparse

import aiohttp

from clients.http_session import get_http_manager

ProgressCallback = Callable[[Dict[str, Any]], Awaitable[None]]


@dataclass
class ComfyUIJobResult:
    prompt_id: str
    outputs: Dict[str, Any]
    images: List[Dict[str, str]] = field(default_factory=list)
    audio: List[Dict[str, str]] = field(default_factory=list)


class ComfyUIClient:
    def __init__(self, base_url: str, *, timeout_seconds: float = 600.0, ws_timeout_seconds: float = 600.0):
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.ws_timeout_seconds = ws_timeout_seconds
        parsed = urlparse(self.base_url)
        ws_scheme = "wss" if parsed.scheme == "https" else "ws"
        self.ws_base = urlunparse((ws_scheme, parsed.netloc, "/ws", "", "", ""))

    def _timeout(self) -> aiohttp.ClientTimeout:
        return aiohttp.ClientTimeout(total=self.timeout_seconds, connect=15, sock_connect=15)

    async def system_stats(self) -> Dict[str, Any]:
        session = await get_http_manager().get_session()
        async with session.get(f"{self.base_url}/system_stats", timeout=self._timeout()) as resp:
            resp.raise_for_status()
            return await resp.json()

    async def is_reachable(self) -> bool:
        try:
            await self.system_stats()
            return True
        except Exception:
            return False

    async def queue_prompt(self, prompt: Dict[str, Any], *, client_id: Optional[str] = None) -> str:
        client_id = client_id or str(uuid.uuid4())
        payload = {"prompt": prompt, "client_id": client_id}
        session = await get_http_manager().get_session()
        async with session.post(
            f"{self.base_url}/prompt",
            json=payload,
            timeout=self._timeout(),
        ) as resp:
            body = await resp.text()
            if resp.status >= 400:
                raise RuntimeError(f"ComfyUI /prompt failed ({resp.status}): {body[:500]}")
            data = json.loads(body)
            prompt_id = data.get("prompt_id")
            if not prompt_id:
                raise RuntimeError(f"ComfyUI returned no prompt_id: {body[:500]}")
            return str(prompt_id)

    async def interrupt(self) -> None:
        session = await get_http_manager().get_session()
        try:
            async with session.post(f"{self.base_url}/interrupt", timeout=self._timeout()) as resp:
                await resp.read()
        except Exception:
            pass

    async def get_history(self, prompt_id: str) -> Dict[str, Any]:
        session = await get_http_manager().get_session()
        async with session.get(f"{self.base_url}/history/{prompt_id}", timeout=self._timeout()) as resp:
            if resp.status == 404:
                return {}
            resp.raise_for_status()
            data = await resp.json()
            return data.get(prompt_id, data if prompt_id in str(data) else data)

    async def upload_image(self, filename: str, data: bytes, *, subfolder: str = "", overwrite: bool = True) -> str:
        session = await get_http_manager().get_session()
        form = aiohttp.FormData()
        form.add_field("image", data, filename=filename, content_type="image/png")
        form.add_field("overwrite", "true" if overwrite else "false")
        if subfolder:
            form.add_field("subfolder", subfolder)
        async with session.post(f"{self.base_url}/upload/image", data=form, timeout=self._timeout()) as resp:
            resp.raise_for_status()
            payload = await resp.json()
            return str(payload.get("name") or filename)

    async def download_view(
        self,
        *,
        filename: str,
        subfolder: str = "",
        folder_type: str = "output",
    ) -> bytes:
        session = await get_http_manager().get_session()
        params = {"filename": filename, "subfolder": subfolder, "type": folder_type}
        async with session.get(f"{self.base_url}/view", params=params, timeout=self._timeout()) as resp:
            resp.raise_for_status()
            return await resp.read()

    async def _ws_progress_loop(
        self,
        client_id: str,
        prompt_id: str,
        on_progress: Optional[ProgressCallback],
        done: asyncio.Event,
    ) -> None:
        if not on_progress:
            return
        url = f"{self.ws_base}?clientId={client_id}"
        try:
            timeout = aiohttp.ClientWSTimeout(ws_receive=self.ws_timeout_seconds)
            session = await get_http_manager().get_session()
            async with session.ws_connect(url, timeout=timeout, heartbeat=30) as ws:
                while not done.is_set():
                    try:
                        msg = await asyncio.wait_for(ws.receive(), timeout=2.0)
                    except asyncio.TimeoutError:
                        continue
                    if msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                        break
                    if msg.type != aiohttp.WSMsgType.TEXT:
                        continue
                    try:
                        evt = json.loads(msg.data)
                    except json.JSONDecodeError:
                        continue
                    etype = evt.get("type")
                    data = evt.get("data") or {}
                    if etype == "progress" and data.get("prompt_id") == prompt_id:
                        await on_progress({
                            "event": "comfy_progress",
                            "prompt_id": prompt_id,
                            "value": data.get("value"),
                            "max": data.get("max"),
                        })
                    elif etype == "executing" and data.get("prompt_id") == prompt_id:
                        if data.get("node") is None:
                            done.set()
                            break
        except Exception as exc:
            await on_progress({"event": "comfy_progress", "prompt_id": prompt_id, "detail": f"ws: {exc}"})

    async def wait_for_outputs(
        self,
        prompt_id: str,
        *,
        client_id: Optional[str] = None,
        on_progress: Optional[ProgressCallback] = None,
        poll_interval: float = 1.5,
    ) -> ComfyUIJobResult:
        client_id = client_id or str(uuid.uuid4())
        done = asyncio.Event()
        ws_task: Optional[asyncio.Task] = None
        if on_progress:
            ws_task = asyncio.create_task(self._ws_progress_loop(client_id, prompt_id, on_progress, done))

        history: Dict[str, Any] = {}
        deadline = time.monotonic() + self.timeout_seconds
        while time.monotonic() < deadline:
            history = await self.get_history(prompt_id)
            if history and history.get("outputs"):
                done.set()
                break
            if done.is_set() and history:
                break
            await asyncio.sleep(poll_interval)
            if on_progress:
                await on_progress({
                    "event": "comfy_progress",
                    "prompt_id": prompt_id,
                    "detail": "waiting",
                })

        if ws_task:
            ws_task.cancel()
            try:
                await ws_task
            except asyncio.CancelledError:
                pass

        if not history or not history.get("outputs"):
            raise TimeoutError(f"ComfyUI job {prompt_id} timed out after {self.timeout_seconds}s")

        images: List[Dict[str, str]] = []
        audio: List[Dict[str, str]] = []
        for node_out in (history.get("outputs") or {}).values():
            for img in node_out.get("images") or []:
                images.append({
                    "filename": img.get("filename", ""),
                    "subfolder": img.get("subfolder", ""),
                    "type": img.get("type", "output"),
                })
            for aud in node_out.get("audio") or []:
                audio.append({
                    "filename": aud.get("filename", ""),
                    "subfolder": aud.get("subfolder", ""),
                    "type": aud.get("type", "output"),
                })
            for gif in node_out.get("gifs") or []:
                audio.append({
                    "filename": gif.get("filename", ""),
                    "subfolder": gif.get("subfolder", ""),
                    "type": gif.get("type", "output"),
                })

        return ComfyUIJobResult(
            prompt_id=prompt_id,
            outputs=history.get("outputs") or {},
            images=images,
            audio=audio,
        )

    async def run_prompt(
        self,
        prompt: Dict[str, Any],
        *,
        on_progress: Optional[ProgressCallback] = None,
    ) -> ComfyUIJobResult:
        client_id = str(uuid.uuid4())
        prompt_id = await self.queue_prompt(prompt, client_id=client_id)
        if on_progress:
            await on_progress({"event": "comfy_progress", "prompt_id": prompt_id, "status": "queued"})
        return await self.wait_for_outputs(prompt_id, client_id=client_id, on_progress=on_progress)


_client: Optional[ComfyUIClient] = None
_enabled: Optional[bool] = None


def get_comfyui_client() -> ComfyUIClient:
    global _client
    if _client is None:
        from config import ComfyUIConfig

        cfg = ComfyUIConfig()
        _client = ComfyUIClient(cfg.url, timeout_seconds=cfg.timeout_seconds)
    return _client


async def comfyui_available() -> bool:
    global _enabled
    from config import ComfyUIConfig

    cfg = ComfyUIConfig()
    if not cfg.enabled:
        _enabled = False
        return False
    if _enabled is not None:
        return _enabled
    try:
        _enabled = await get_comfyui_client().is_reachable()
    except Exception:
        _enabled = False
    return bool(_enabled)