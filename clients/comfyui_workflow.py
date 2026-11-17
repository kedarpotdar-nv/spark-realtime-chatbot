"""Load ComfyUI workflow JSON and inject parameters by node title / class_type."""

from __future__ import annotations

import copy
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WORKFLOW_DIR = REPO_ROOT / "workflows" / "comfyui"

# _meta.title convention (case-insensitive). Users can export API workflows with these titles.
TITLE_ALIASES: Dict[str, Tuple[str, ...]] = {
    "PROMPT": ("PROMPT", "CLIP Text Encode (Positive Prompt)", "Positive Prompt"),
    "NEGATIVE": ("NEGATIVE", "CLIP Text Encode (Negative Prompt)", "Negative Prompt"),
    "SEED": ("SEED", "RandomNoise", "Int (Seed)"),
    "WIDTH": ("WIDTH", "Width"),
    "HEIGHT": ("HEIGHT", "Height"),
    "STEPS": ("STEPS", "Flux2Scheduler"),
    "INPUT_IMAGE": ("INPUT_IMAGE", "LoadImage"),
    "TAGS": ("TAGS", "Tags"),
    "LYRICS": ("LYRICS", "Lyrics"),
    "DURATION": ("DURATION", "Float (Duration)", "Duration"),
    "UNET": ("UNET", "UNETLoader"),
    "CLIP": ("CLIP", "CLIPLoader", "DualCLIPLoader"),
    "VAE": ("VAE", "VAELoader"),
    "SAVE_IMAGE": ("SAVE_IMAGE", "SaveImage"),
    "SAVE_AUDIO": ("SAVE_AUDIO", "SaveAudioAdvanced", "SaveAudio"),
}


def _norm_title(title: str) -> str:
    return re.sub(r"\s+", " ", (title or "").strip()).casefold()


def _node_title(node: Dict[str, Any]) -> str:
    meta = node.get("_meta") or {}
    return str(meta.get("title") or node.get("title") or "")


def _matches_slot(title: str, class_type: str, slot: str) -> bool:
    nt = _norm_title(title)
    ct = (class_type or "").casefold()
    for alias in TITLE_ALIASES.get(slot, (slot,)):
        a = _norm_title(alias)
        if nt == a or a in nt:
            return True
        if slot == "SEED" and class_type in ("RandomNoise", "KSampler") and "seed" not in nt:
            if class_type == "RandomNoise":
                return True
        if slot == "STEPS" and class_type == "Flux2Scheduler":
            return True
        if slot == "UNET" and class_type == "UNETLoader":
            return True
        if slot == "CLIP" and class_type in ("CLIPLoader", "DualCLIPLoader"):
            return True
        if slot == "VAE" and class_type == "VAELoader":
            return True
        if slot == "INPUT_IMAGE" and class_type == "LoadImage":
            return True
        if slot == "DURATION" and class_type in ("EmptyAceStep1.5LatentAudio", "PrimitiveNode", "PrimitiveInt"):
            if "duration" in nt or class_type.startswith("EmptyAceStep"):
                return True
        if slot == "TAGS" and class_type == "TextEncodeAceStepAudio1.5" and "lyrics" not in nt:
            return title.upper().startswith("TAGS") or nt == "tags"
        if slot == "LYRICS" and class_type == "TextEncodeAceStepAudio1.5":
            return "lyrics" in nt or title.upper().startswith("LYRICS")
        if slot == "SAVE_AUDIO" and "saveaudio" in ct:
            return True
        if slot == "SAVE_IMAGE" and class_type == "SaveImage":
            return True
        if slot in ("PROMPT", "NEGATIVE") and class_type == "CLIPTextEncode":
            if slot == "NEGATIVE" and ("negative" in nt or "negative" in title.casefold()):
                return True
            if slot == "PROMPT" and ("positive" in nt or "prompt" in nt) and "negative" not in nt:
                return True
    return False


def find_nodes_by_slot(prompt: Dict[str, Any], slot: str) -> list[Tuple[str, Dict[str, Any]]]:
    out: list[Tuple[str, Dict[str, Any]]] = []
    for nid, node in prompt.items():
        if not isinstance(node, dict):
            continue
        title = _node_title(node)
        ctype = str(node.get("class_type") or "")
        if _matches_slot(title, ctype, slot):
            out.append((nid, node))
    return out


def load_workflow(path: Path) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "prompt" in data and isinstance(data["prompt"], dict):
        return copy.deepcopy(data["prompt"])
    if isinstance(data, dict) and all(isinstance(v, dict) and "class_type" in v for v in data.values()):
        return copy.deepcopy(data)
    raise ValueError(f"Workflow {path} is not ComfyUI API format (expected node id -> class_type map)")


def resolve_workflow_path(kind: str, cfg: Any) -> Path:
    env_map = {
        "t2i": "COMFYUI_FLUX_T2I_WORKFLOW",
        "edit": "COMFYUI_FLUX_EDIT_WORKFLOW",
        "music": "COMFYUI_ACE_WORKFLOW",
    }
    default_map = {
        "t2i": DEFAULT_WORKFLOW_DIR / "flux2_klein_t2i.json",
        "edit": DEFAULT_WORKFLOW_DIR / "flux2_klein_edit.json",
        "music": DEFAULT_WORKFLOW_DIR / "ace_step_1_5_music.json",
    }
    raw = os.getenv(env_map[kind], "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    override = getattr(cfg, f"workflow_{kind}", None) or getattr(cfg, "workflow_paths", {}).get(kind)
    if override:
        return Path(override).expanduser().resolve()
    return default_map[kind]


def _set_unet_name(node: Dict[str, Any], filename: str) -> None:
    inputs = node.setdefault("inputs", {})
    if "unet_name" in inputs or node.get("class_type") == "UNETLoader":
        inputs["unet_name"] = filename


def _set_clip_name(node: Dict[str, Any], filename: str, dual: bool = False) -> None:
    inputs = node.setdefault("inputs", {})
    if dual:
        # DualCLIP: only override first clip if env provides single file
        if "clip_name1" in inputs:
            inputs["clip_name1"] = filename
    else:
        inputs["clip_name"] = filename


def _set_vae_name(node: Dict[str, Any], filename: str) -> None:
    node.setdefault("inputs", {})["vae_name"] = filename


def inject_flux_t2i(
    prompt: Dict[str, Any],
    *,
    positive: str,
    negative: str = "",
    width: int,
    height: int,
    seed: Optional[int],
    steps: Optional[int],
    model_overrides: Dict[str, str],
) -> Dict[str, Any]:
    wf = copy.deepcopy(prompt)
    for _, node in find_nodes_by_slot(wf, "PROMPT"):
        node.setdefault("inputs", {})["text"] = positive
    for _, node in find_nodes_by_slot(wf, "NEGATIVE"):
        node.setdefault("inputs", {})["text"] = negative
    for _, node in wf.items():
        if node.get("class_type") != "EmptyFlux2LatentImage":
            continue
        ins = node.setdefault("inputs", {})
        ins["width"] = width
        ins["height"] = height
    for _, node in find_nodes_by_slot(wf, "WIDTH"):
        if node.get("class_type") == "EmptyFlux2LatentImage":
            continue
        node.setdefault("inputs", {})["value"] = width
    for _, node in find_nodes_by_slot(wf, "HEIGHT"):
        if node.get("class_type") == "EmptyFlux2LatentImage":
            continue
        node.setdefault("inputs", {})["value"] = height
    if seed is not None:
        for _, node in find_nodes_by_slot(wf, "SEED"):
            ctype = node.get("class_type")
            if ctype == "RandomNoise":
                node.setdefault("inputs", {})["noise_seed"] = seed
            elif ctype == "KSampler":
                node.setdefault("inputs", {})["seed"] = seed
    if steps is not None:
        for _, node in find_nodes_by_slot(wf, "STEPS"):
            if node.get("class_type") == "Flux2Scheduler":
                node.setdefault("inputs", {})["steps"] = steps
    if model_overrides.get("unet"):
        for _, node in find_nodes_by_slot(wf, "UNET"):
            _set_unet_name(node, model_overrides["unet"])
    if model_overrides.get("clip"):
        for _, node in find_nodes_by_slot(wf, "CLIP"):
            _set_clip_name(node, model_overrides["clip"], dual=node.get("class_type") == "DualCLIPLoader")
    if model_overrides.get("vae"):
        for _, node in find_nodes_by_slot(wf, "VAE"):
            _set_vae_name(node, model_overrides["vae"])
    return wf


def inject_flux_edit(
    prompt: Dict[str, Any],
    *,
    positive: str,
    uploaded_image_name: str,
    width: int,
    height: int,
    seed: Optional[int],
    steps: Optional[int],
    model_overrides: Dict[str, str],
) -> Dict[str, Any]:
    wf = inject_flux_t2i(
        prompt,
        positive=positive,
        negative="",
        width=width,
        height=height,
        seed=seed,
        steps=steps,
        model_overrides=model_overrides,
    )
    for _, node in find_nodes_by_slot(wf, "INPUT_IMAGE"):
        node.setdefault("inputs", {})["image"] = uploaded_image_name
    return wf


def inject_ace_music(
    prompt: Dict[str, Any],
    *,
    tags: str,
    lyrics: str = "",
    duration_seconds: float,
    seed: Optional[int],
    model_overrides: Dict[str, str],
) -> Dict[str, Any]:
    wf = copy.deepcopy(prompt)
    dur = max(1.0, float(duration_seconds))
    for _, node in find_nodes_by_slot(wf, "DURATION"):
        ctype = node.get("class_type", "")
        if ctype.startswith("EmptyAceStep"):
            node.setdefault("inputs", {})["seconds"] = dur
        else:
            node.setdefault("inputs", {})["value"] = dur
    for _, node in find_nodes_by_slot(wf, "TAGS"):
        if node.get("class_type") == "TextEncodeAceStepAudio1.5":
            wv = node.get("inputs", {}).get("widgets")  # fallback
            node.setdefault("inputs", {})
            # TextEncodeAceStepAudio1.5 uses tags as first widget in API exports
            node["inputs"]["tags"] = tags
            if lyrics:
                node["inputs"]["lyrics"] = lyrics
            elif "lyrics" not in node["inputs"]:
                node["inputs"]["lyrics"] = ""
    # Also match untitled TextEncode nodes
    for nid, node in wf.items():
        if node.get("class_type") == "TextEncodeAceStepAudio1.5":
            ins = node.setdefault("inputs", {})
            if "tags" not in ins:
                ins["tags"] = tags
            if lyrics and "lyrics" not in ins:
                ins["lyrics"] = lyrics
    if seed is not None:
        for _, node in find_nodes_by_slot(wf, "SEED"):
            ctype = node.get("class_type")
            if ctype == "KSampler":
                node.setdefault("inputs", {})["seed"] = seed
            elif ctype in ("RandomNoise", "PrimitiveInt"):
                node.setdefault("inputs", {})["value"] = seed
    if model_overrides.get("unet"):
        for _, node in find_nodes_by_slot(wf, "UNET"):
            _set_unet_name(node, model_overrides["unet"])
    if model_overrides.get("vae"):
        for _, node in find_nodes_by_slot(wf, "VAE"):
            _set_vae_name(node, model_overrides["vae"])
    return wf


def pick_output_image_nodes(prompt: Dict[str, Any]) -> Iterable[Tuple[str, Dict[str, Any]]]:
    for nid, node in prompt.items():
        if node.get("class_type") == "SaveImage":
            yield nid, node


def pick_output_audio_nodes(prompt: Dict[str, Any]) -> Iterable[Tuple[str, Dict[str, Any]]]:
    for nid, node in prompt.items():
        ct = (node.get("class_type") or "").casefold()
        if "saveaudio" in ct:
            yield nid, node


def random_seed() -> int:
    return random.randint(0, 2**53 - 1)