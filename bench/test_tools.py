#!/usr/bin/env python3
"""E2E smoke test for tools.py. Hits each inline tool once and reports status.

Run: python3 bench/test_tools.py
"""
import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools import execute_tool, ALL_TOOLS  # noqa: E402


async def check(name, args, expect_keys=None, expect_ok=True):
    t0 = asyncio.get_event_loop().time()
    raw = await execute_tool(name, args)
    elapsed = (asyncio.get_event_loop().time() - t0) * 1000
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {"_raw": raw}
    ok = True
    reasons = []
    if expect_ok and "error" in data:
        ok = False
        reasons.append(f"error: {data['error']}")
    for k in (expect_keys or []):
        if k not in data:
            ok = False
            reasons.append(f"missing key {k!r}")
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name} ({elapsed:.0f}ms) {'; '.join(reasons) if reasons else ''}")
    if not ok or name in ("run_python", "web_search"):
        preview = json.dumps(data)[:300]
        print(f"        payload: {preview}")
    return ok


async def main():
    registered = list(ALL_TOOLS.keys())
    print(f"tools registered: {registered}\n")

    results = []
    results.append(await check("list_files", {"path": "."}, ["entries"]))
    results.append(await check("read_file", {"path": "README.md", "max_bytes": 200}, ["content", "size_bytes"]))
    results.append(await check("write_file", {"path": "audio_cache/_test_write.txt", "content": "hello"}, ["ok"]))
    results.append(await check("read_file", {"path": "audio_cache/_test_write.txt"}, ["content"]))
    results.append(await check("run_python", {"code": "print(2+2)", "timeout_s": 10}, ["stdout"]))
    results.append(await check("remember_fact", {"key": "_test_key", "value": "hello"}, ["ok"]))
    results.append(await check("recall_fact", {"key": "_test_key"}, ["value"]))
    results.append(await check("recall_fact", {}, ["facts"]))
    # web_search: optional network, don't fail hard
    await check("web_search", {"query": "DGX Spark", "max_results": 3}, ["results"])
    # agent tool sentinels
    results.append(await check("markdown_assistant", {"task": "x"}, ["agent_type", "status"]))
    results.append(await check("reasoning_assistant", {"problem": "x"}, ["agent_type", "status"]))

    passed = sum(1 for r in results if r)
    total = len(results)
    print(f"\n{passed}/{total} required checks passed")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    asyncio.run(main())
