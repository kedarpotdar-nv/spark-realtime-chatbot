#!/usr/bin/env python3
"""E2E test: simulate the server's agent loop against live Ollama + real tools.

Proves: Qwen3.6 emits tool_calls → we execute real tools → feed result back →
model synthesizes a final answer. Measures latency for each iteration.

Run: python3 bench/test_agent_loop.py
"""
import asyncio
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools import execute_tool, ALL_TOOLS  # noqa: E402

OLLAMA_URL = "http://localhost:11434/v1/chat/completions"
MODEL = "qwen3.6:35b-a3b"


def _post(payload, timeout=60):
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


async def run_agent(user_msg, enabled_tool_names, max_iter=4):
    tools = [ALL_TOOLS[n] for n in enabled_tool_names]
    messages = [
        {"role": "system", "content": "You are a helpful assistant with tools. Use them when useful. Answer briefly."},
        {"role": "user", "content": user_msg},
    ]
    timings = []
    for i in range(max_iter):
        t0 = time.perf_counter()
        resp = _post({
            "model": MODEL,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "stream": False,
            "max_tokens": 800,
            "reasoning_effort": "none",
        })
        elapsed = (time.perf_counter() - t0) * 1000
        msg = resp["choices"][0]["message"]
        tool_calls = msg.get("tool_calls") or []
        content = msg.get("content") or ""
        timings.append({"iter": i + 1, "llm_ms": round(elapsed, 1), "tool_calls": len(tool_calls), "content_chars": len(content)})
        print(f"  iter {i+1}: llm={elapsed:.0f}ms, tool_calls={len(tool_calls)}, content={len(content)}ch")
        if not tool_calls:
            return {"final": content, "iters": i + 1, "timings": timings}
        # Append assistant turn
        messages.append({"role": "assistant", "content": None, "tool_calls": tool_calls})
        # Execute tools in parallel
        async def _run(tc):
            name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"].get("arguments") or "{}")
            except Exception:
                args = {}
            t0 = time.perf_counter()
            result = await execute_tool(name, args)
            return tc, name, result, (time.perf_counter() - t0) * 1000
        executed = await asyncio.gather(*[_run(tc) for tc in tool_calls])
        for tc, name, result, ms in executed:
            print(f"    tool {name}: {ms:.0f}ms, {len(result)}ch")
            messages.append({
                "role": "tool", "tool_call_id": tc["id"], "name": name, "content": result,
            })
    return {"final": "[hit max_iter]", "iters": max_iter, "timings": timings}


async def main():
    cases = [
        ("list + read", "What files are in the current workspace? Then tell me the first line of README.md.",
         ["list_files", "read_file"]),
        ("math via run_python", "Use the python tool to compute 17 * 42 and tell me the result.",
         ["run_python"]),
        ("memory", "Please remember that my favorite color is teal, then immediately recall it back to me.",
         ["remember_fact", "recall_fact"]),
    ]
    all_ok = True
    for title, prompt, tool_names in cases:
        print(f"\n=== {title} ===")
        t0 = time.perf_counter()
        r = await run_agent(prompt, tool_names)
        total_ms = (time.perf_counter() - t0) * 1000
        print(f"  final ({r['iters']} iter, {total_ms:.0f}ms): {r['final'][:200]}")
        if not r["final"] or r["final"] == "[hit max_iter]":
            all_ok = False
    print("\n" + ("OK" if all_ok else "FAIL"))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    asyncio.run(main())
