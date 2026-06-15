#!/usr/bin/env python3
"""Perf bench for the Qwen3.6 single-model pipeline.

Hits Ollama directly (no project-venv deps). Measures:
  - voice_turn        text-only, reasoning_effort=none, short system+user
  - video_turn        image + text, reasoning_effort=none, small system+user
  - reasoning_turn    text-only, reasoning_effort=high, multi-step problem
  - tool_call_turn    OpenAI tool-call roundtrip (1 tool declared, model calls it)

Each metric: N trials, reports median + p90 TTFT and total. Writes JSON to stdout
and optionally to --out. Compare across runs with bench/diff.py.

Run:
  python3 bench/bench.py --trials 5 --out bench/baseline.json
  python3 bench/bench.py --trials 5 --out bench/after.json
"""

import argparse
import base64
import json
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

OLLAMA_URL = "http://localhost:11434/v1/chat/completions"
MODEL = "qwen3.6:35b-a3b"
REPO_ROOT = Path(__file__).resolve().parent.parent


def _post_stream(payload, timeout=120):
    """POST JSON, return (ttft_ms, total_ms, first_content_chunk, reasoning_chars, content_chars, tool_calls)."""
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    ttft_ms = None
    first_content = None
    reasoning_chars = 0
    content_chars = 0
    tool_calls = []
    acc_tool_calls = {}
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            try:
                j = json.loads(body)
            except json.JSONDecodeError:
                continue
            choices = j.get("choices") or []
            if not choices:
                continue
            d = choices[0].get("delta") or {}
            if d.get("reasoning"):
                reasoning_chars += len(d["reasoning"])
            if d.get("content"):
                if ttft_ms is None:
                    ttft_ms = (time.perf_counter() - t0) * 1000
                    first_content = d["content"]
                content_chars += len(d["content"])
            tc = d.get("tool_calls") or []
            for part in tc:
                idx = part.get("index", 0)
                slot = acc_tool_calls.setdefault(idx, {"name": "", "args": ""})
                fn = part.get("function") or {}
                if fn.get("name"):
                    slot["name"] = fn["name"]
                if fn.get("arguments"):
                    slot["args"] += fn["arguments"]
            fr = choices[0].get("finish_reason")
            if fr == "tool_calls" and ttft_ms is None:
                # Tool-only response: mark TTFT at first-seen tool chunk time
                ttft_ms = (time.perf_counter() - t0) * 1000
    total_ms = (time.perf_counter() - t0) * 1000
    tool_calls = [acc_tool_calls[i] for i in sorted(acc_tool_calls)]
    return {
        "ttft_ms": ttft_ms,
        "total_ms": total_ms,
        "first_content": first_content,
        "reasoning_chars": reasoning_chars,
        "content_chars": content_chars,
        "tool_calls": tool_calls,
    }


def _summary(values, key):
    v = [x[key] for x in values if x.get(key) is not None]
    if not v:
        return None
    return {
        "n": len(v),
        "median": round(statistics.median(v), 1),
        "p90": round(statistics.quantiles(v, n=10)[-1], 1) if len(v) >= 3 else round(max(v), 1),
        "min": round(min(v), 1),
        "max": round(max(v), 1),
    }


def _warmup():
    """One tiny request to load the model into Ollama cache before we measure anything."""
    print("[bench] warming up model cache...", file=sys.stderr)
    t0 = time.perf_counter()
    _post_stream({
        "model": MODEL,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
        "max_tokens": 8,
        "reasoning_effort": "none",
    })
    print(f"[bench] warmup {(time.perf_counter()-t0)*1000:.0f}ms", file=sys.stderr)


def bench_voice_turn(trials):
    results = []
    for i in range(trials):
        r = _post_stream({
            "model": MODEL,
            "messages": [
                {"role": "system", "content": "You are a concise voice assistant. Answer in one short sentence."},
                {"role": "user", "content": "What's the capital of France?"},
            ],
            "stream": True,
            "max_tokens": 60,
            "reasoning_effort": "none",
        })
        results.append(r)
        print(f"  voice_turn[{i+1}/{trials}] ttft={r['ttft_ms']:.0f}ms total={r['total_ms']:.0f}ms", file=sys.stderr)
    return {
        "ttft_ms": _summary(results, "ttft_ms"),
        "total_ms": _summary(results, "total_ms"),
    }


def bench_video_turn(trials):
    img_path = REPO_ROOT / "demo.png"
    if not img_path.exists():
        return {"skipped": "demo.png not found"}
    b64 = base64.b64encode(img_path.read_bytes()).decode()
    results = []
    for i in range(trials):
        r = _post_stream({
            "model": MODEL,
            "messages": [
                {"role": "system", "content": "You are a concise visual assistant. Answer in one short sentence."},
                {"role": "user", "content": [
                    {"type": "text", "text": "what's in this image?"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ]},
            ],
            "stream": True,
            "max_tokens": 80,
            "reasoning_effort": "none",
        }, timeout=180)
        results.append(r)
        print(f"  video_turn[{i+1}/{trials}] ttft={r['ttft_ms']:.0f}ms total={r['total_ms']:.0f}ms", file=sys.stderr)
    return {
        "ttft_ms": _summary(results, "ttft_ms"),
        "total_ms": _summary(results, "total_ms"),
    }


def bench_reasoning_turn(trials):
    results = []
    for i in range(trials):
        r = _post_stream({
            "model": MODEL,
            "messages": [
                {"role": "system", "content": "Think step-by-step, then give the final answer in one sentence."},
                {"role": "user", "content": "A farmer has 17 sheep. All but 9 die. How many are left?"},
            ],
            "stream": True,
            "max_tokens": 2048,
            "reasoning_effort": "high",
        }, timeout=300)
        results.append(r)
        print(f"  reasoning_turn[{i+1}/{trials}] total={r['total_ms']:.0f}ms reasoning={r['reasoning_chars']} content={r['content_chars']}", file=sys.stderr)
    return {
        "total_ms": _summary(results, "total_ms"),
        "reasoning_chars": _summary(results, "reasoning_chars"),
        "content_chars": _summary(results, "content_chars"),
    }


def bench_agent_loop_turn(trials):
    """Measure a full agent loop: LLM call → tool exec → LLM call → final content.

    TTFT here = time from request start to the first content token AFTER any
    tool-call iterations, i.e. the latency the user actually perceives before
    hearing the answer. Uses run_python (cheap local tool) to avoid network.
    """
    import sys as _sys
    if str(REPO_ROOT) not in _sys.path:
        _sys.path.insert(0, str(REPO_ROOT))
    import tools as tools_mod
    # Guard: tools must be importable standalone
    try:
        tool_def = tools_mod.ALL_TOOLS["run_python"]
    except Exception as e:
        return {"skipped": f"tools import failed: {e}"}
    results = []
    for i in range(trials):
        messages = [
            {"role": "system", "content": "Use the run_python tool for math, then answer briefly."},
            {"role": "user", "content": "Use run_python to compute 137 * 42, then tell me the result in one short sentence."},
        ]
        t0 = time.perf_counter()
        ttft_ms = None
        iterations = 0
        final_text = ""
        while iterations < 4:
            iterations += 1
            resp = urllib.request.Request(
                OLLAMA_URL,
                data=json.dumps({
                    "model": MODEL,
                    "messages": messages,
                    "tools": [tool_def],
                    "tool_choice": "auto",
                    "stream": True,
                    "max_tokens": 400,
                    "reasoning_effort": "none",
                }).encode(),
                headers={"Content-Type": "application/json"},
            )
            accum_tool_calls = {}
            content_piece = ""
            with urllib.request.urlopen(resp, timeout=60) as r:
                for raw in r:
                    line = raw.decode("utf-8", errors="ignore").strip()
                    if not line.startswith("data:"):
                        continue
                    body = line[5:].strip()
                    if body == "[DONE]":
                        break
                    try:
                        j = json.loads(body)
                    except json.JSONDecodeError:
                        continue
                    choices = j.get("choices") or []
                    if not choices:
                        continue
                    d = choices[0].get("delta") or {}
                    if d.get("content"):
                        if ttft_ms is None:
                            ttft_ms = (time.perf_counter() - t0) * 1000
                        content_piece += d["content"]
                    for part in d.get("tool_calls") or []:
                        idx = part.get("index", 0)
                        slot = accum_tool_calls.setdefault(idx, {"id": part.get("id", ""), "name": "", "args": ""})
                        fn = part.get("function") or {}
                        if fn.get("name"):
                            slot["name"] = fn["name"]
                        if fn.get("arguments"):
                            slot["args"] += fn["arguments"]
            if not accum_tool_calls:
                final_text = content_piece
                break
            # Execute tool calls inline (subprocess path, fast)
            messages.append({
                "role": "assistant", "content": None,
                "tool_calls": [
                    {"id": s["id"], "type": "function",
                     "function": {"name": s["name"], "arguments": s["args"]}}
                    for s in accum_tool_calls.values()
                ],
            })
            for s in accum_tool_calls.values():
                # Inline execute — not using asyncio to keep bench simple
                import asyncio as _asy
                loop = _asy.new_event_loop()
                try:
                    result = loop.run_until_complete(
                        __import__('tools').execute_tool(s["name"], json.loads(s["args"] or "{}")))
                finally:
                    loop.close()
                messages.append({
                    "role": "tool", "tool_call_id": s["id"], "name": s["name"], "content": result,
                })
        total_ms = (time.perf_counter() - t0) * 1000
        results.append({
            "ttft_ms": ttft_ms,
            "total_ms": total_ms,
            "iterations": iterations,
            "content_chars": len(final_text),
        })
        print(f"  agent_loop_turn[{i+1}/{trials}] iters={iterations} ttft={ttft_ms:.0f}ms total={total_ms:.0f}ms text={final_text[:60]!r}", file=sys.stderr)
    return {
        "ttft_ms": _summary(results, "ttft_ms"),
        "total_ms": _summary(results, "total_ms"),
        "iterations": _summary(results, "iterations"),
    }


def bench_tool_call_turn(trials):
    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a city.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }]
    results = []
    for i in range(trials):
        r = _post_stream({
            "model": MODEL,
            "messages": [
                {"role": "system", "content": "Use tools when needed."},
                {"role": "user", "content": "What's the weather like in Tokyo right now?"},
            ],
            "tools": tools,
            "tool_choice": "auto",
            "stream": True,
            "max_tokens": 300,
            "reasoning_effort": "none",
        })
        got = bool(r["tool_calls"])
        results.append(r)
        print(f"  tool_call_turn[{i+1}/{trials}] ttft={r['ttft_ms']:.0f}ms total={r['total_ms']:.0f}ms tool_called={got}", file=sys.stderr)
    return {
        "ttft_ms": _summary(results, "ttft_ms"),
        "total_ms": _summary(results, "total_ms"),
        "tool_call_rate": round(sum(bool(r["tool_calls"]) for r in results) / len(results), 2),
    }


def git_sha():
    try:
        return subprocess.check_output(["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def git_dirty():
    try:
        out = subprocess.check_output(["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
                                      stderr=subprocess.DEVNULL).decode()
        return bool(out.strip())
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--only", choices=["voice", "video", "reasoning", "tool", "agent"], default=None)
    args = ap.parse_args()

    _warmup()

    out = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_sha": git_sha(),
        "git_dirty": git_dirty(),
        "model": MODEL,
        "ollama_url": OLLAMA_URL,
        "trials": args.trials,
    }

    if args.only in (None, "voice"):
        print("[bench] voice_turn", file=sys.stderr)
        out["voice_turn"] = bench_voice_turn(args.trials)
    if args.only in (None, "video"):
        print("[bench] video_turn", file=sys.stderr)
        out["video_turn"] = bench_video_turn(args.trials)
    if args.only in (None, "reasoning"):
        print("[bench] reasoning_turn", file=sys.stderr)
        out["reasoning_turn"] = bench_reasoning_turn(max(2, args.trials // 2))
    if args.only in (None, "tool"):
        print("[bench] tool_call_turn", file=sys.stderr)
        out["tool_call_turn"] = bench_tool_call_turn(args.trials)
    if args.only in (None, "agent"):
        print("[bench] agent_loop_turn", file=sys.stderr)
        out["agent_loop_turn"] = bench_agent_loop_turn(max(3, args.trials))

    print(json.dumps(out, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"[bench] wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
