"""ComfyUI job runner with WebSocket progress + workspace artifacts."""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set

from config import AGENT_WORKSPACE_DIR
from clients.comfy_media import (
    decode_b64_image,
    edit_image_comfy,
    generate_image_comfy,
    generate_music_comfy,
)

NotifyFn = Callable[[str, Dict[str, Any]], Awaitable[None]]

_comfy_jobs: Dict[str, Dict[str, Any]] = {}


def list_comfy_jobs() -> List[Dict[str, Any]]:
    return sorted(_comfy_jobs.values(), key=lambda j: j.get("created_at", 0), reverse=True)


def get_comfy_job(job_id: str) -> Optional[Dict[str, Any]]:
    return _comfy_jobs.get(job_id)


def _new_job(kind: str, label: str) -> str:
    job_id = f"comfy-{uuid.uuid4().hex[:10]}"
    _comfy_jobs[job_id] = {
        "job_id": job_id,
        "kind": kind,
        "label": label,
        "status": "queued",
        "progress": None,
        "artifact": None,
        "error": None,
        "created_at": time.time(),
    }
    if len(_comfy_jobs) > 40:
        oldest = sorted(_comfy_jobs, key=lambda k: _comfy_jobs[k]["created_at"])[:10]
        for k in oldest:
            _comfy_jobs.pop(k, None)
    return job_id


async def run_comfy_job(
    *,
    kind: str,
    params: Dict[str, Any],
    notify: NotifyFn,
    resolve_edit_source: Callable[[str], Awaitable[tuple[bytes, str]]],
) -> Dict[str, Any]:
    job_id = _new_job(kind, params.get("prompt") or params.get("tags") or kind)
    job = _comfy_jobs[job_id]

    async def on_progress(evt: Dict[str, Any]) -> None:
        val, mx = evt.get("value"), evt.get("max")
        if val is not None and mx:
            job["progress"] = {"value": val, "max": mx, "percent": round(100 * val / mx, 1)}
        job["status"] = "running"
        await notify("comfy_progress", {**job, **evt})

    job["status"] = "running"
    await notify("comfy_progress", dict(job))

    try:
        if kind == "image":
            result = await generate_image_comfy(
                prompt=params["prompt"],
                width=params.get("width"),
                height=params.get("height"),
                seed=params.get("seed"),
                steps=params.get("steps"),
                on_progress=on_progress,
            )
        elif kind == "edit":
            src = str(params.get("source") or "camera")
            img_bytes, fname = await resolve_edit_source(src)
            result = await edit_image_comfy(
                prompt=params["prompt"],
                source_bytes=img_bytes,
                source_filename=fname,
                width=params.get("width"),
                height=params.get("height"),
                seed=params.get("seed"),
                steps=params.get("steps"),
                on_progress=on_progress,
            )
        elif kind == "music":
            result = await generate_music_comfy(
                tags=params["tags"],
                lyrics=params.get("lyrics") or "",
                duration_seconds=params.get("duration_seconds"),
                seed=params.get("seed"),
                on_progress=on_progress,
            )
        else:
            raise ValueError(f"Unknown kind: {kind}")

        if not result.get("success"):
            job["status"] = "failed"
            job["error"] = result.get("error", "ComfyUI failed")
            await notify("comfy_progress", dict(job))
            return {"success": False, "job_id": job_id, **result}

        job["status"] = "completed"
        job["artifact"] = result.get("path")
        job["result"] = result
        await notify("comfy_progress", dict(job))
        await notify("media_ready", {
            "job_id": job_id,
            "path": result.get("path"),
            "kind": kind,
            "result": result,
        })
        return {"success": True, "job_id": job_id, **result}
    except Exception as exc:
        job["status"] = "failed"
        job["error"] = str(exc)
        await notify("comfy_progress", dict(job))
        return {"success": False, "job_id": job_id, "error": str(exc)}


async def resolve_edit_source_for_session(
    session: Any,
    source: str,
) -> tuple[bytes, str]:
    src = (source or "camera").strip()
    low = src.lower()
    if low in ("camera", "webcam", "video"):
        if not session.latest_camera_frame_b64:
            raise ValueError("No camera frame available. Turn on video or pick another source.")
        return decode_b64_image(session.latest_camera_frame_b64), "camera_frame.png"
    if low in ("last_image", "last", "previous"):
        path = getattr(session, "last_generated_image_path", None)
        if not path:
            raise ValueError("No previous generated image in this session.")
        full = (AGENT_WORKSPACE_DIR / path).resolve()
        if not full.is_file():
            raise ValueError(f"Last image not found: {path}")
        return full.read_bytes(), full.name
    rel = src.lstrip("/")
    full = (AGENT_WORKSPACE_DIR / rel).resolve()
    if not str(full).startswith(str(AGENT_WORKSPACE_DIR)) or not full.is_file():
        raise ValueError(f"Workspace image not found: {source}")
    return full.read_bytes(), full.name