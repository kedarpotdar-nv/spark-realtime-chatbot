"""High-level ComfyUI media generation (save to workspace/media)."""

from __future__ import annotations

import base64
import re
import time
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, Optional

from config import AGENT_WORKSPACE_DIR, ComfyUIConfig
from clients.comfyui import ComfyUIClient, comfyui_available, get_comfyui_client
from clients.comfyui_workflow import (
    inject_ace_music,
    inject_flux_edit,
    inject_flux_t2i,
    load_workflow,
    random_seed,
)

ProgressCallback = Callable[[Dict[str, Any]], Awaitable[None]]

MEDIA_DIR = AGENT_WORKSPACE_DIR / "media"


def _ensure_media_dir() -> Path:
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    return MEDIA_DIR


def _slug(text: str, limit: int = 48) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", (text or "")[:limit]).strip("_").lower()
    return s or "output"


async def _unavailable_message() -> Dict[str, Any]:
    return {
        "success": False,
        "error": (
            "ComfyUI is not reachable. Start ComfyUI on this host (default http://127.0.0.1:8188) "
            "with FLUX.2 Klein and ACE-Step 1.5, or set COMFYUI_ENABLED=false to hide these tools."
        ),
    }


async def generate_image_comfy(
    *,
    prompt: str,
    width: Optional[int] = None,
    height: Optional[int] = None,
    seed: Optional[int] = None,
    steps: Optional[int] = None,
    on_progress: Optional[ProgressCallback] = None,
) -> Dict[str, Any]:
    if not await comfyui_available():
        return await _unavailable_message()

    cfg = ComfyUIConfig()
    wf = load_workflow(Path(cfg.workflow_t2i))
    seed_val = seed if seed is not None else random_seed()
    injected = inject_flux_t2i(
        wf,
        positive=prompt,
        negative="",
        width=width or cfg.default_width,
        height=height or cfg.default_height,
        seed=seed_val,
        steps=steps or cfg.default_steps,
        model_overrides=cfg.model_overrides_flux(edit=False),
    )
    client = get_comfyui_client()
    result = await client.run_prompt(injected, on_progress=on_progress)
    if not result.images:
        return {"success": False, "error": "ComfyUI finished but returned no images."}
    img_meta = result.images[0]
    data = await client.download_view(
        filename=img_meta["filename"],
        subfolder=img_meta.get("subfolder", ""),
        folder_type=img_meta.get("type", "output"),
    )
    out_dir = _ensure_media_dir()
    fname = f"{int(time.time())}_{_slug(prompt)}_{uuid.uuid4().hex[:8]}.png"
    out_path = out_dir / fname
    out_path.write_bytes(data)
    rel = str(out_path.relative_to(AGENT_WORKSPACE_DIR))
    return {
        "success": True,
        "path": rel,
        "absolute_path": str(out_path),
        "prompt_id": result.prompt_id,
        "seed": seed_val,
        "kind": "image",
    }


async def edit_image_comfy(
    *,
    prompt: str,
    source_bytes: bytes,
    source_filename: str = "edit_input.png",
    width: Optional[int] = None,
    height: Optional[int] = None,
    seed: Optional[int] = None,
    steps: Optional[int] = None,
    on_progress: Optional[ProgressCallback] = None,
) -> Dict[str, Any]:
    if not await comfyui_available():
        return await _unavailable_message()

    cfg = ComfyUIConfig()
    client = get_comfyui_client()
    uploaded = await client.upload_image(source_filename, source_bytes)
    wf = load_workflow(Path(cfg.workflow_edit))
    seed_val = seed if seed is not None else random_seed()
    injected = inject_flux_edit(
        wf,
        positive=prompt,
        uploaded_image_name=uploaded,
        width=width or cfg.default_width,
        height=height or cfg.default_height,
        seed=seed_val,
        steps=steps or cfg.default_steps,
        model_overrides=cfg.model_overrides_flux(edit=True),
    )
    result = await client.run_prompt(injected, on_progress=on_progress)
    if not result.images:
        return {"success": False, "error": "ComfyUI edit finished but returned no images."}
    img_meta = result.images[0]
    data = await client.download_view(
        filename=img_meta["filename"],
        subfolder=img_meta.get("subfolder", ""),
        folder_type=img_meta.get("type", "output"),
    )
    out_dir = _ensure_media_dir()
    fname = f"{int(time.time())}_{_slug(prompt)}_edit_{uuid.uuid4().hex[:8]}.png"
    out_path = out_dir / fname
    out_path.write_bytes(data)
    rel = str(out_path.relative_to(AGENT_WORKSPACE_DIR))
    return {
        "success": True,
        "path": rel,
        "absolute_path": str(out_path),
        "prompt_id": result.prompt_id,
        "seed": seed_val,
        "kind": "edit",
    }


async def generate_music_comfy(
    *,
    tags: str,
    lyrics: str = "",
    duration_seconds: Optional[float] = None,
    seed: Optional[int] = None,
    on_progress: Optional[ProgressCallback] = None,
) -> Dict[str, Any]:
    if not await comfyui_available():
        return await _unavailable_message()

    cfg = ComfyUIConfig()
    duration = duration_seconds if duration_seconds is not None else 60.0
    wf = load_workflow(Path(cfg.workflow_music))
    seed_val = seed if seed is not None else random_seed()
    injected = inject_ace_music(
        wf,
        tags=tags,
        lyrics=lyrics,
        duration_seconds=duration,
        seed=seed_val,
        model_overrides=cfg.model_overrides_ace(),
    )
    client = get_comfyui_client()
    result = await client.run_prompt(injected, on_progress=on_progress)
    files = result.audio or result.images
    if not files:
        return {"success": False, "error": "ComfyUI music job finished but returned no audio."}
    meta = files[0]
    data = await client.download_view(
        filename=meta["filename"],
        subfolder=meta.get("subfolder", ""),
        folder_type=meta.get("type", "output"),
    )
    ext = Path(meta["filename"]).suffix or ".mp3"
    out_dir = _ensure_media_dir()
    fname = f"{int(time.time())}_{_slug(tags)}_{uuid.uuid4().hex[:8]}{ext}"
    out_path = out_dir / fname
    out_path.write_bytes(data)
    rel = str(out_path.relative_to(AGENT_WORKSPACE_DIR))
    return {
        "success": True,
        "path": rel,
        "absolute_path": str(out_path),
        "prompt_id": result.prompt_id,
        "seed": seed_val,
        "kind": "music",
        "duration_seconds": duration,
    }


def decode_b64_image(image_b64: str) -> bytes:
    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]
    return base64.b64decode(image_b64)