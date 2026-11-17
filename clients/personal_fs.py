"""Personal filesystem and shell tools for Spark (Jarvis local mode).

Runs directly on the DGX Spark host with root allowlisting — no Docker sandbox.
"""

from __future__ import annotations

import asyncio
import mimetypes
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import PersonalFSConfig


_TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".json", ".jsonl", ".csv", ".tsv", ".py", ".js",
    ".ts", ".tsx", ".jsx", ".css", ".html", ".htm", ".xml", ".yaml", ".yml",
    ".toml", ".ini", ".cfg", ".conf", ".sh", ".bash", ".zsh", ".sql", ".rs",
    ".go", ".c", ".cpp", ".h", ".hpp", ".java", ".kt", ".swift", ".rb", ".php",
    ".r", ".log", ".env", ".gitignore", ".dockerfile", ".makefile", ".rst",
    ".tex", ".svg",
}

_BINARY_PREVIEW_NOTE = "[binary file — content not shown as text]"


class PersonalFS:
    """Execute personal filesystem and shell operations with root gating."""

    def __init__(self, cfg: Optional[PersonalFSConfig] = None):
        self.cfg = cfg or PersonalFSConfig()

    # ------------------------------------------------------------------
    # Path helpers
    # ------------------------------------------------------------------

    def resolve(self, path: str) -> Path:
        return self.cfg.resolve_allowed(path)

    def roots_summary(self) -> List[str]:
        return [str(r) for r in self.cfg.roots]

    # ------------------------------------------------------------------
    # list_directory
    # ------------------------------------------------------------------

    def list_directory(self, path: str = "~", max_entries: Optional[int] = None) -> Dict[str, Any]:
        if not self.cfg.enabled:
            return {"success": False, "error": "Personal filesystem tools are disabled"}
        try:
            target = self.resolve(path or str(Path.home()))
        except ValueError as exc:
            return {"success": False, "error": str(exc)}

        if not target.exists():
            return {"success": False, "error": f"Path does not exist: {target}"}
        if not target.is_dir():
            return {"success": False, "error": f"Not a directory: {target}"}

        limit = max_entries or self.cfg.list_max_entries
        entries: List[Dict[str, Any]] = []
        try:
            children = sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except PermissionError:
            return {"success": False, "error": f"Permission denied: {target}"}

        truncated = False
        for child in children:
            if len(entries) >= limit:
                truncated = True
                break
            try:
                stat = child.stat()
                entries.append({
                    "name": child.name,
                    "path": str(child),
                    "type": "dir" if child.is_dir() else "file",
                    "size": stat.st_size if child.is_file() else None,
                    "modified_at": stat.st_mtime,
                })
            except OSError:
                entries.append({
                    "name": child.name,
                    "path": str(child),
                    "type": "unknown",
                    "size": None,
                    "modified_at": None,
                })

        return {
            "success": True,
            "path": str(target),
            "count": len(entries),
            "truncated": truncated,
            "entries": entries,
        }

    # ------------------------------------------------------------------
    # read_file
    # ------------------------------------------------------------------

    def read_file(self, path: str, max_bytes: Optional[int] = None) -> Dict[str, Any]:
        if not self.cfg.enabled:
            return {"success": False, "error": "Personal filesystem tools are disabled"}
        try:
            target = self.resolve(path)
        except ValueError as exc:
            return {"success": False, "error": str(exc)}

        if not target.exists():
            return {"success": False, "error": f"File does not exist: {target}"}
        if not target.is_file():
            return {"success": False, "error": f"Not a file: {target}"}

        limit = max_bytes or self.cfg.read_max_bytes
        mime, _ = mimetypes.guess_type(str(target))
        suffix = target.suffix.lower()
        is_text = (
            suffix in _TEXT_EXTENSIONS
            or (mime or "").startswith("text/")
            or mime in ("application/json", "application/xml", "application/javascript")
        )

        try:
            size = target.stat().st_size
            if not is_text:
                # Peek for null bytes to confirm binary
                with open(target, "rb") as fh:
                    sample = fh.read(min(4096, size))
                if b"\x00" in sample:
                    return {
                        "success": True,
                        "path": str(target),
                        "size": size,
                        "mime_type": mime or "application/octet-stream",
                        "binary": True,
                        "content": _BINARY_PREVIEW_NOTE,
                        "truncated": False,
                    }

            with open(target, "rb") as fh:
                raw = fh.read(limit + 1)
            truncated = len(raw) > limit
            raw = raw[:limit]
            try:
                content = raw.decode("utf-8")
            except UnicodeDecodeError:
                content = raw.decode("utf-8", errors="replace")

            return {
                "success": True,
                "path": str(target),
                "size": size,
                "mime_type": mime or "text/plain",
                "binary": False,
                "content": content,
                "truncated": truncated,
            }
        except PermissionError:
            return {"success": False, "error": f"Permission denied: {target}"}
        except OSError as exc:
            return {"success": False, "error": f"Failed to read file: {exc}"}

    # ------------------------------------------------------------------
    # write_file
    # ------------------------------------------------------------------

    def write_file(self, path: str, content: str, create_dirs: bool = True) -> Dict[str, Any]:
        if not self.cfg.enabled:
            return {"success": False, "error": "Personal filesystem tools are disabled"}
        try:
            target = self.resolve(path)
        except ValueError as exc:
            return {"success": False, "error": str(exc)}

        try:
            if create_dirs:
                target.parent.mkdir(parents=True, exist_ok=True)
            if not self.cfg.is_under_roots(target.parent):
                return {"success": False, "error": f"Parent directory outside allowed roots: {target.parent}"}
            data = content if isinstance(content, str) else str(content)
            target.write_text(data, encoding="utf-8")
            return {
                "success": True,
                "path": str(target),
                "bytes_written": len(data.encode("utf-8")),
            }
        except PermissionError:
            return {"success": False, "error": f"Permission denied: {target}"}
        except OSError as exc:
            return {"success": False, "error": f"Failed to write file: {exc}"}

    # ------------------------------------------------------------------
    # search_files
    # ------------------------------------------------------------------

    def search_files(
        self,
        query: str,
        root: str = "~",
        max_results: Optional[int] = None,
    ) -> Dict[str, Any]:
        if not self.cfg.enabled:
            return {"success": False, "error": "Personal filesystem tools are disabled"}
        if not query or not query.strip():
            return {"success": False, "error": "Search query is required"}

        try:
            base = self.resolve(root or str(Path.home()))
        except ValueError as exc:
            return {"success": False, "error": str(exc)}

        if not base.exists():
            return {"success": False, "error": f"Search root does not exist: {base}"}

        limit = max_results or self.cfg.search_max_results
        needle = query.strip().lower()
        matches: List[Dict[str, Any]] = []
        truncated = False
        ignored = {".git", "__pycache__", "node_modules", ".venv", "venv", ".cache", "dist", "build"}

        walk_root = base if base.is_dir() else base.parent
        for dirpath, dirnames, filenames in os.walk(walk_root):
            dirnames[:] = [d for d in dirnames if d not in ignored and not d.startswith(".")]
            for name in filenames:
                if needle in name.lower():
                    full = Path(dirpath) / name
                    try:
                        if not self.cfg.is_under_roots(full):
                            continue
                        matches.append({
                            "name": name,
                            "path": str(full.resolve()),
                            "size": full.stat().st_size,
                        })
                    except OSError:
                        continue
                    if len(matches) >= limit:
                        truncated = True
                        break
            if truncated:
                break
            # Also match directory names
            for name in list(dirnames):
                if needle in name.lower():
                    full = Path(dirpath) / name
                    try:
                        if self.cfg.is_under_roots(full):
                            matches.append({
                                "name": name,
                                "path": str(full.resolve()),
                                "size": None,
                                "type": "dir",
                            })
                    except OSError:
                        pass
                    if len(matches) >= limit:
                        truncated = True
                        break
            if truncated:
                break

        return {
            "success": True,
            "query": query,
            "root": str(base),
            "count": len(matches),
            "truncated": truncated,
            "matches": matches,
        }

    # ------------------------------------------------------------------
    # run_command
    # ------------------------------------------------------------------

    async def run_command(self, command: str, cwd: Optional[str] = None) -> Dict[str, Any]:
        if not self.cfg.enabled:
            return {"success": False, "error": "Personal filesystem tools are disabled"}
        if not self.cfg.shell_enabled:
            return {"success": False, "error": "Personal shell tools are disabled"}
        if not command or not str(command).strip():
            return {"success": False, "error": "Command is required"}

        workdir = Path.home()
        if cwd:
            try:
                workdir = self.resolve(cwd)
            except ValueError as exc:
                return {"success": False, "error": str(exc)}
        else:
            try:
                workdir = self.cfg.resolve_allowed(str(Path.home()))
            except ValueError:
                workdir = Path.home()

        if not workdir.is_dir():
            return {"success": False, "error": f"Working directory is not a directory: {workdir}"}

        timeout = max(1, int(self.cfg.shell_timeout))
        max_out = self.cfg.command_max_output_bytes

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                cwd=str(workdir),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, "TERM": "dumb"},
            )
            try:
                stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                proc.kill()
                try:
                    await proc.wait()
                except Exception:
                    pass
                return {
                    "success": False,
                    "error": f"Command timed out after {timeout}s",
                    "cwd": str(workdir),
                    "command": command,
                }

            def _decode_trunc(data: bytes) -> tuple[str, bool]:
                truncated = len(data) > max_out
                chunk = data[:max_out]
                try:
                    text = chunk.decode("utf-8")
                except UnicodeDecodeError:
                    text = chunk.decode("utf-8", errors="replace")
                return text, truncated

            stdout, out_trunc = _decode_trunc(stdout_b or b"")
            stderr, err_trunc = _decode_trunc(stderr_b or b"")
            return {
                "success": proc.returncode == 0,
                "exit_code": proc.returncode,
                "cwd": str(workdir),
                "command": command,
                "stdout": stdout,
                "stderr": stderr,
                "stdout_truncated": out_trunc,
                "stderr_truncated": err_trunc,
            }
        except Exception as exc:
            return {"success": False, "error": f"Failed to run command: {exc}", "cwd": str(workdir)}

    # ------------------------------------------------------------------
    # open / reveal (host desktop)
    # ------------------------------------------------------------------

    def open_path(self, path: str, reveal: bool = False) -> Dict[str, Any]:
        """Open a file/folder with the host desktop opener (xdg-open / open)."""
        try:
            target = self.resolve(path)
        except ValueError as exc:
            return {"success": False, "error": str(exc)}

        if not target.exists():
            return {"success": False, "error": f"Path does not exist: {target}"}

        open_target = target
        if reveal and target.is_file():
            open_target = target.parent

        opener = shutil.which("xdg-open") or shutil.which("open")
        if not opener:
            return {
                "success": False,
                "error": "No desktop opener found (xdg-open / open)",
                "path": str(target),
            }

        try:
            subprocess.Popen(
                [opener, str(open_target)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            return {
                "success": True,
                "path": str(target),
                "opened": str(open_target),
                "reveal": reveal,
                "opener": opener,
            }
        except OSError as exc:
            return {"success": False, "error": f"Failed to open path: {exc}", "path": str(target)}

    # ------------------------------------------------------------------
    # LLM-friendly string summaries
    # ------------------------------------------------------------------

    def format_for_llm(self, result: Dict[str, Any], tool_name: str) -> str:
        """Compact JSON-ish summary suitable for tool results in chat history."""
        import json

        if not result.get("success", False) and "error" in result:
            return json.dumps(result)

        if tool_name == "list_directory":
            lines = [f"Directory: {result.get('path')} ({result.get('count')} entries)"]
            for entry in result.get("entries", [])[:40]:
                kind = entry.get("type", "?")
                name = entry.get("name", "")
                if kind == "dir":
                    lines.append(f"  [dir]  {name}/")
                else:
                    size = entry.get("size")
                    size_s = f" ({size} bytes)" if size is not None else ""
                    lines.append(f"  [file] {name}{size_s}")
            if result.get("truncated"):
                lines.append("  ... (truncated)")
            return "\n".join(lines)

        if tool_name == "read_file":
            header = f"File: {result.get('path')} ({result.get('size')} bytes)"
            if result.get("binary"):
                return f"{header}\n{result.get('content')}"
            content = result.get("content", "")
            note = "\n... (truncated)" if result.get("truncated") else ""
            return f"{header}\n{content}{note}"

        if tool_name == "search_files":
            lines = [
                f"Search '{result.get('query')}' under {result.get('root')}: "
                f"{result.get('count')} match(es)"
            ]
            for m in result.get("matches", [])[:30]:
                lines.append(f"  {m.get('path')}")
            if result.get("truncated"):
                lines.append("  ... (truncated)")
            return "\n".join(lines)

        if tool_name == "run_command":
            parts = [
                f"$ {result.get('command')}  (cwd={result.get('cwd')}, exit={result.get('exit_code')})",
            ]
            if result.get("stdout"):
                parts.append("--- stdout ---\n" + result["stdout"])
            if result.get("stderr"):
                parts.append("--- stderr ---\n" + result["stderr"])
            if result.get("stdout_truncated") or result.get("stderr_truncated"):
                parts.append("(output truncated)")
            if result.get("error"):
                parts.append("Error: " + result["error"])
            return "\n".join(parts)

        return json.dumps(result, ensure_ascii=False)


# Module-level singleton
_personal_fs: Optional[PersonalFS] = None


def get_personal_fs() -> PersonalFS:
    global _personal_fs
    if _personal_fs is None:
        _personal_fs = PersonalFS()
    return _personal_fs
