#!/usr/bin/env python3
"""Validate Spark ComfyUI API workflows against a ComfyUI source tree (and optional live server)."""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WORKFLOWS = REPO_ROOT / "workflows" / "comfyui"

MODEL_INPUT_KEYS = {
    "UNETLoader": ("unet_name", "diffusion_models"),
    "CLIPLoader": ("clip_name", "text_encoders"),
    "DualCLIPLoader": (("clip_name1", "clip_name2"), "text_encoders"),
    "VAELoader": ("vae_name", "vae"),
}

WORKFLOW_FILES = (
    "flux2_klein_t2i.json",
    "flux2_klein_edit.json",
    "ace_step_1_5_music.json",
)


def _collect_class_types(comfy_root: Path) -> Set[str]:
    names: Set[str] = set()
    nodes_py = comfy_root / "nodes.py"
    if nodes_py.is_file():
        text = nodes_py.read_text(encoding="utf-8", errors="replace")
        block_match = re.search(r"NODE_CLASS_MAPPINGS\s*=\s*\{([^}]+)\}", text, re.S)
        if block_match:
            for m in re.finditer(r'"([^"]+)"\s*:', block_match.group(1)):
                names.add(m.group(1))
    for path in comfy_root.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for m in re.finditer(r'node_id="([^"]+)"', text):
            names.add(m.group(1))
        for m in re.finditer(
            r'"([A-Za-z][A-Za-z0-9_]*)"\s*:\s*([A-Za-z][A-Za-z0-9_]*)',
            text,
        ):
            if m.group(1)[0].isupper():
                names.add(m.group(1))
    return names


def _load_workflows(workflow_dir: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for name in WORKFLOW_FILES:
        path = workflow_dir / name
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "prompt" in data:
            data = data["prompt"]
        out[name] = data
    return out


def _is_link(value: Any) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[0], str)
        and isinstance(value[1], int)
    )


def _iter_model_filenames(node: Dict[str, Any]) -> Iterable[Tuple[str, str]]:
    ctype = str(node.get("class_type") or "")
    spec = MODEL_INPUT_KEYS.get(ctype)
    if not spec:
        return
    keys, subdir = spec
    inputs = node.get("inputs") or {}
    if isinstance(keys, tuple) and len(keys) == 2 and ctype == "DualCLIPLoader":
        for key in keys:
            val = inputs.get(key)
            if isinstance(val, str) and val:
                yield subdir, val
    elif isinstance(keys, str):
        val = inputs.get(keys)
        if isinstance(val, str) and val:
            yield subdir, val


def _validate_links(prompt: Dict[str, Any], errors: List[str], *, prefix: str) -> None:
    for nid, node in prompt.items():
        if not isinstance(node, dict):
            errors.append(f"{prefix}node {nid}: not an object")
            continue
        for key, value in (node.get("inputs") or {}).items():
            if not _is_link(value):
                continue
            target_id, out_idx = value
            if target_id not in prompt:
                errors.append(f"{prefix}node {nid} input {key}: missing link target {target_id}")
            if out_idx < 0 or out_idx > 32:
                errors.append(f"{prefix}node {nid} input {key}: suspicious output index {out_idx}")


def _validate_models(prompt: Dict[str, Any], comfy_root: Path, errors: List[str], *, prefix: str) -> None:
    models = comfy_root / "models"
    for nid, node in prompt.items():
        for subdir, filename in _iter_model_filenames(node):
            path = models / subdir / filename
            if not path.is_file():
                errors.append(f"{prefix}node {nid}: model file missing: models/{subdir}/{filename}")


def _fetch_object_info(base_url: str) -> Dict[str, Any]:
    url = base_url.rstrip("/") + "/object_info"
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.loads(resp.read().decode())


def _validate_with_object_info(
    prompt: Dict[str, Any],
    object_info: Dict[str, Any],
    errors: List[str],
    *,
    prefix: str,
) -> None:
    for nid, node in prompt.items():
        ctype = node.get("class_type")
        if not ctype:
            errors.append(f"{prefix}node {nid}: missing class_type")
            continue
        info = object_info.get(ctype)
        if not info:
            errors.append(f"{prefix}node {nid}: unknown class_type {ctype} (object_info)")
            continue
        required = (info.get("input") or {}).get("required") or {}
        inputs = node.get("inputs") or {}
        for req_key in required:
            if req_key not in inputs:
                errors.append(f"{prefix}node {nid} ({ctype}): missing required input '{req_key}'")
        for key, value in inputs.items():
            if not _is_link(value):
                continue
            target_id, out_idx = value
            target = prompt.get(target_id)
            if not target:
                continue
            ttype = target.get("class_type")
            tinfo = object_info.get(ttype or "")
            if not tinfo:
                continue
            outputs = tinfo.get("output") or []
            if out_idx >= len(outputs):
                errors.append(
                    f"{prefix}node {nid} input {key}: link [{target_id}, {out_idx}] "
                    f"but {ttype} has {len(outputs)} output(s)"
                )


def validate(
    *,
    comfy_root: Path,
    workflow_dir: Path,
    server_url: Optional[str],
) -> Tuple[bool, List[str]]:
    errors: List[str] = []
    warnings: List[str] = []

    class_types = _collect_class_types(comfy_root)
    if not class_types:
        errors.append(f"Could not collect any NODE_CLASS_MAPPINGS from {comfy_root}")

    object_info: Optional[Dict[str, Any]] = None
    if server_url:
        try:
            object_info = _fetch_object_info(server_url)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            warnings.append(f"Live validation skipped: could not fetch object_info from {server_url}: {exc}")

    workflows = _load_workflows(workflow_dir)
    for name, prompt in workflows.items():
        prefix = f"{name}: "
        for nid, node in prompt.items():
            ctype = node.get("class_type")
            if not ctype:
                errors.append(f"{prefix}node {nid}: missing class_type")
                continue
            if ctype not in class_types:
                errors.append(f"{prefix}node {nid}: unknown class_type '{ctype}'")
        _validate_links(prompt, errors, prefix=prefix)
        _validate_models(prompt, comfy_root, errors, prefix=prefix)
        if object_info:
            _validate_with_object_info(prompt, object_info, errors, prefix=prefix)

    for w in warnings:
        print(f"warning: {w}")
    return len(errors) == 0, errors


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "comfy_root",
        type=Path,
        help="Path to ComfyUI installation (read-only; used for nodes + models/)",
    )
    parser.add_argument(
        "--workflows",
        type=Path,
        default=DEFAULT_WORKFLOWS,
        help=f"Directory containing API workflows (default: {DEFAULT_WORKFLOWS})",
    )
    parser.add_argument(
        "--url",
        default="",
        help="Optional ComfyUI base URL for GET /object_info validation",
    )
    args = parser.parse_args(argv)

    comfy_root = args.comfy_root.expanduser().resolve()
    if not comfy_root.is_dir():
        print(f"ComfyUI root not found: {comfy_root}", file=sys.stderr)
        return 2

    ok, errors = validate(
        comfy_root=comfy_root,
        workflow_dir=args.workflows.resolve(),
        server_url=args.url.strip() or None,
    )
    if ok:
        print(f"OK — {len(WORKFLOW_FILES)} workflows validated against {comfy_root}")
        return 0
    for err in errors:
        print(f"error: {err}", file=sys.stderr)
    print(f"FAILED — {len(errors)} error(s)", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())