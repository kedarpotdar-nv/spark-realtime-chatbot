"""Tool definitions and execution for OpenAI-compatible tool calling.

Two categories of tools:

1. **Inline tools** — executed server-side, return a string result that is appended
   as a `tool` message and fed back into the LLM. The model decides what to do
   with the output (read_file, write_file, list_files, run_python, web_search,
   remember_fact, recall_fact).

2. **UI agent tools** — return a sentinel JSON `{"agent_type": "...", "status":
   "initiated"}`. server.py detects these and opens the corresponding streaming
   UI panel (markdown editor, reasoning panel). These are kept for back-compat
   with the existing frontend.
"""

from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import quote_plus

import aiohttp

from config import WORKSPACE_ROOT


# ---------------------------------------------------------------------------
# Tool schemas (OpenAI format)
# ---------------------------------------------------------------------------

ALL_TOOLS: Dict[str, Dict[str, Any]] = {
    # ----- inline tools -----
    "read_file": {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 text file from the workspace. Use for inspecting code, notes, CSVs, or any user file before acting on it.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to the workspace root."},
                    "max_bytes": {"type": "integer", "description": "Max bytes to read (default 64000).", "default": 64000},
                },
                "required": ["path"],
            },
        },
    },
    "write_file": {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create or overwrite a UTF-8 text file in the workspace. Use when the user asks to save, generate, or modify a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to the workspace root."},
                    "content": {"type": "string", "description": "File contents."},
                },
                "required": ["path", "content"],
            },
        },
    },
    "list_files": {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files and subdirectories at a workspace path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path relative to workspace root. Defaults to '.'.", "default": "."},
                },
                "required": [],
            },
        },
    },
    "run_python": {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": "Execute a short Python snippet in a sandboxed interpreter. Use for math, data crunching, quick plots → returned as text (stdout). Do NOT use for persistent file I/O — use write_file instead.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python source to run. Anything printed to stdout is returned."},
                    "timeout_s": {"type": "integer", "description": "Max run time in seconds (default 15).", "default": 15},
                },
                "required": ["code"],
            },
        },
    },
    "web_search": {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web and return the top results (title, URL, snippet). Use for current events, unfamiliar terms, or anything time-sensitive.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query."},
                    "max_results": {"type": "integer", "description": "Number of results to return (default 5).", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    "remember_fact": {
        "type": "function",
        "function": {
            "name": "remember_fact",
            "description": "Store a small fact about the user or session for later recall (e.g. name, preference, project detail). Persists across sessions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "Short identifier (e.g. 'user_name', 'favorite_lang')."},
                    "value": {"type": "string", "description": "The fact to remember."},
                },
                "required": ["key", "value"],
            },
        },
    },
    "recall_fact": {
        "type": "function",
        "function": {
            "name": "recall_fact",
            "description": "Recall a previously remembered fact by key, or list all stored facts if no key is given.",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "The key to recall. If omitted, returns all facts."},
                },
                "required": [],
            },
        },
    },

    # ----- UI agent tools (kept for back-compat with existing frontend) -----
    "markdown_assistant": {
        "type": "function",
        "function": {
            "name": "markdown_assistant",
            "description": "Open the streaming markdown editor for long-form documentation. Use when the user asks to write a README, guide, or any substantial markdown document.",
            "parameters": {
                "type": "object",
                "properties": {
                    "task": {"type": "string", "description": "The documentation task."},
                    "context": {"type": "string", "description": "Optional project/topic context."},
                },
                "required": ["task"],
            },
        },
    },
    "reasoning_assistant": {
        "type": "function",
        "function": {
            "name": "reasoning_assistant",
            "description": "ONLY for customer feedback / feature prioritization questions against local demo_files. Opens the streaming reasoning panel with thinking + conclusion.",
            "parameters": {
                "type": "object",
                "properties": {
                    "problem": {"type": "string", "description": "The question to reason about."},
                    "context": {"type": "string", "description": "Optional extra context (e.g. a whiteboard description)."},
                    "analysis_type": {
                        "type": "string",
                        "enum": ["general", "comparison", "prioritization", "planning"],
                    },
                },
                "required": ["problem"],
            },
        },
    },
}


def get_enabled_tools(enabled_tool_ids: List[str]) -> List[Dict[str, Any]]:
    """Get list of tool definitions for enabled tool IDs."""
    return [ALL_TOOLS[tid] for tid in enabled_tool_ids if tid in ALL_TOOLS]


# ---------------------------------------------------------------------------
# Workspace path safety
# ---------------------------------------------------------------------------

def _resolve_safe(rel_path: str) -> Path:
    """Resolve a path inside WORKSPACE_ROOT, refusing to escape."""
    p = (WORKSPACE_ROOT / rel_path).resolve()
    try:
        p.relative_to(WORKSPACE_ROOT)
    except ValueError as e:
        raise ValueError(f"path escapes workspace root: {rel_path}") from e
    return p


def _truncate(s: str, limit: int) -> str:
    if len(s) <= limit:
        return s
    return s[:limit] + f"\n\n[...truncated {len(s) - limit} chars...]"


# ---------------------------------------------------------------------------
# Inline tool implementations
# ---------------------------------------------------------------------------

async def _tool_read_file(args: Dict[str, Any]) -> str:
    path = args.get("path", "")
    max_bytes = int(args.get("max_bytes", 64000))
    try:
        p = _resolve_safe(path)
        if not p.exists():
            return json.dumps({"error": f"file not found: {path}"})
        if p.is_dir():
            return json.dumps({"error": f"path is a directory: {path}"})
        data = p.read_bytes()[:max_bytes]
        return json.dumps({
            "path": str(p.relative_to(WORKSPACE_ROOT)),
            "size_bytes": p.stat().st_size,
            "content": _truncate(data.decode("utf-8", errors="replace"), max_bytes),
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


async def _tool_write_file(args: Dict[str, Any]) -> str:
    path = args.get("path", "")
    content = args.get("content", "")
    try:
        p = _resolve_safe(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return json.dumps({
            "path": str(p.relative_to(WORKSPACE_ROOT)),
            "size_bytes": p.stat().st_size,
            "ok": True,
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


async def _tool_list_files(args: Dict[str, Any]) -> str:
    path = args.get("path", ".")
    try:
        p = _resolve_safe(path)
        if not p.exists():
            return json.dumps({"error": f"path not found: {path}"})
        if p.is_file():
            return json.dumps({"error": f"path is a file: {path}"})
        entries = []
        for child in sorted(p.iterdir())[:200]:
            entries.append({
                "name": child.name,
                "type": "dir" if child.is_dir() else "file",
                "size_bytes": child.stat().st_size if child.is_file() else None,
            })
        return json.dumps({"path": str(p.relative_to(WORKSPACE_ROOT)) or ".", "entries": entries})
    except Exception as e:
        return json.dumps({"error": str(e)})


async def _tool_run_python(args: Dict[str, Any]) -> str:
    code = args.get("code", "")
    timeout_s = int(args.get("timeout_s", 15))
    try:
        # Prefer llm-sandbox if available (Docker-backed, safer). Fall back to a
        # subprocess with a short timeout if llm-sandbox isn't importable (dev).
        try:
            from llm_sandbox import SandboxSession
            return await asyncio.to_thread(_run_in_sandbox, code, timeout_s)
        except ImportError:
            return await _run_subprocess_python(code, timeout_s)
    except Exception as e:
        return json.dumps({"error": f"run_python failed: {e}"})


def _run_in_sandbox(code: str, timeout_s: int) -> str:
    from llm_sandbox import SandboxSession
    started = time.perf_counter()
    with SandboxSession(lang="python", verbose=False) as sess:
        result = sess.run(code)
    elapsed_ms = (time.perf_counter() - started) * 1000
    stdout = getattr(result, "stdout", "") or ""
    stderr = getattr(result, "stderr", "") or ""
    exit_code = getattr(result, "exit_code", None)
    return json.dumps({
        "stdout": _truncate(stdout, 8000),
        "stderr": _truncate(stderr, 2000),
        "exit_code": exit_code,
        "elapsed_ms": round(elapsed_ms, 1),
        "backend": "llm-sandbox",
    })


async def _run_subprocess_python(code: str, timeout_s: int) -> str:
    proc = await asyncio.create_subprocess_exec(
        "python3", "-I", "-c", code,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    started = time.perf_counter()
    try:
        stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except asyncio.TimeoutError:
        proc.kill()
        return json.dumps({"error": f"run_python timed out after {timeout_s}s", "backend": "subprocess"})
    return json.dumps({
        "stdout": _truncate(stdout_b.decode("utf-8", errors="replace"), 8000),
        "stderr": _truncate(stderr_b.decode("utf-8", errors="replace"), 2000),
        "exit_code": proc.returncode,
        "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
        "backend": "subprocess",
    })


async def _tool_web_search(args: Dict[str, Any]) -> str:
    """Web search — DuckDuckGo HTML (POST) with Wikipedia fallback. No API key."""
    query = args.get("query", "")
    max_results = int(args.get("max_results", 5))
    if not query.strip():
        return json.dumps({"error": "empty query"})

    ua = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
          "Chrome/120.0.0.0 Safari/537.36")
    headers = {
        "User-Agent": ua,
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "en-US,en;q=0.9",
    }

    html = None
    err = None
    try:
        async with aiohttp.ClientSession() as sess:
            # DDG lite endpoint is more forgiving than the main /html/
            async with sess.post(
                "https://html.duckduckgo.com/html/",
                data={"q": query, "kl": "us-en"},
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as r:
                if r.status == 200:
                    html = await r.text()
                else:
                    err = f"DDG HTTP {r.status}"
    except Exception as e:
        err = f"DDG failed: {e}"

    if html:
        try:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
            results = []
            for block in soup.select(".result")[:max_results * 2]:
                a = block.select_one(".result__a")
                snip = block.select_one(".result__snippet")
                if not a:
                    continue
                results.append({
                    "title": a.get_text(strip=True),
                    "url": a.get("href", ""),
                    "snippet": _truncate(snip.get_text(" ", strip=True) if snip else "", 280),
                })
                if len(results) >= max_results:
                    break
            if results:
                return json.dumps({"query": query, "results": results, "source": "duckduckgo"})
        except ImportError:
            pass  # fall through to wikipedia fallback

    # Fallback: Wikipedia REST opensearch — always works, no auth
    try:
        async with aiohttp.ClientSession() as sess:
            async with sess.get(
                "https://en.wikipedia.org/w/api.php",
                params={"action": "opensearch", "search": query, "limit": str(max_results), "format": "json"},
                headers={"User-Agent": ua},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as r:
                if r.status != 200:
                    return json.dumps({"error": f"wikipedia HTTP {r.status}", "ddg_error": err})
                data = await r.json()
        titles, descs, urls = data[1], data[2], data[3]
        results = [
            {"title": t, "url": u, "snippet": _truncate(d, 280)}
            for t, d, u in zip(titles, descs, urls)
        ]
        return json.dumps({"query": query, "results": results, "source": "wikipedia", "note": err})
    except Exception as e:
        return json.dumps({"error": f"search failed: ddg={err} wiki={e}"})


# ---------------------------------------------------------------------------
# Memory tool (sqlite)
# ---------------------------------------------------------------------------

_MEMORY_DB_PATH = WORKSPACE_ROOT / "audio_cache" / "memory.db"


def _memory_conn() -> sqlite3.Connection:
    _MEMORY_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_MEMORY_DB_PATH))
    conn.execute("""CREATE TABLE IF NOT EXISTS facts (
        key     TEXT PRIMARY KEY,
        value   TEXT NOT NULL,
        updated REAL NOT NULL
    )""")
    return conn


async def _tool_remember_fact(args: Dict[str, Any]) -> str:
    key = (args.get("key") or "").strip()
    value = args.get("value") or ""
    if not key:
        return json.dumps({"error": "key required"})
    def _work():
        with _memory_conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO facts(key, value, updated) VALUES (?, ?, ?)",
                (key, value, time.time()),
            )
    await asyncio.to_thread(_work)
    return json.dumps({"ok": True, "key": key})


async def _tool_recall_fact(args: Dict[str, Any]) -> str:
    key = (args.get("key") or "").strip()
    def _work():
        with _memory_conn() as conn:
            if key:
                row = conn.execute("SELECT value FROM facts WHERE key = ?", (key,)).fetchone()
                return {"key": key, "value": row[0] if row else None}
            rows = conn.execute("SELECT key, value FROM facts ORDER BY key").fetchall()
            return {"facts": [{"key": k, "value": v} for k, v in rows]}
    out = await asyncio.to_thread(_work)
    return json.dumps(out)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_INLINE_DISPATCH = {
    "read_file":     _tool_read_file,
    "write_file":    _tool_write_file,
    "list_files":    _tool_list_files,
    "run_python":    _tool_run_python,
    "web_search":    _tool_web_search,
    "remember_fact": _tool_remember_fact,
    "recall_fact":   _tool_recall_fact,
}


def is_agent_tool(tool_name: str) -> bool:
    """UI-dispatched agent tools return a sentinel payload; loop should not re-prompt."""
    return tool_name in ("markdown_assistant", "reasoning_assistant")


async def execute_tool(tool_name: str, arguments: Dict[str, Any]) -> str:
    """Execute a tool and return its result as a JSON string."""
    if tool_name in _INLINE_DISPATCH:
        try:
            return await _INLINE_DISPATCH[tool_name](arguments or {})
        except Exception as e:
            return json.dumps({"error": f"{tool_name} failed: {e}"})

    # UI agent tools — sentinel response, server.py opens the UI
    if tool_name == "markdown_assistant":
        return json.dumps({
            "agent_type": "markdown_assistant",
            "task": arguments.get("task", ""),
            "context": arguments.get("context", ""),
            "status": "initiated",
        })
    if tool_name == "reasoning_assistant":
        return json.dumps({
            "agent_type": "reasoning_assistant",
            "problem": arguments.get("problem", ""),
            "context": arguments.get("context", ""),
            "analysis_type": arguments.get("analysis_type", "general"),
            "status": "initiated",
        })

    return json.dumps({"error": f"unknown tool: {tool_name}"})
