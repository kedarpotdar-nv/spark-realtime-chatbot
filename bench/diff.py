#!/usr/bin/env python3
"""Diff two bench JSON files. Usage: python3 bench/diff.py baseline.json after.json"""
import json
import sys


def fmt(v):
    return f"{v:6.1f}" if isinstance(v, (int, float)) else f"{v}"


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    a = json.loads(open(sys.argv[1]).read())
    b = json.loads(open(sys.argv[2]).read())
    print(f"{sys.argv[1]} (sha={a.get('git_sha')}) → {sys.argv[2]} (sha={b.get('git_sha')})\n")
    rows = []
    for section in ["voice_turn", "video_turn", "reasoning_turn", "tool_call_turn", "agent_loop_turn"]:
        for metric in ["ttft_ms", "total_ms"]:
            an = (a.get(section) or {}).get(metric) or {}
            bn = (b.get(section) or {}).get(metric) or {}
            if "median" not in an or "median" not in bn:
                continue
            delta = bn["median"] - an["median"]
            pct = (delta / an["median"] * 100) if an["median"] else 0
            arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "=")
            rows.append((section, metric, an["median"], bn["median"], delta, pct, arrow))

    if not rows:
        print("No comparable metrics.")
        return
    print(f"{'section':<16}{'metric':<12}{'before':>10}{'after':>10}{'Δ':>10}{'%':>9}  ")
    print("-" * 68)
    for section, metric, a_med, b_med, delta, pct, arrow in rows:
        print(f"{section:<16}{metric:<12}{fmt(a_med):>10}{fmt(b_med):>10}{fmt(delta):>10}{pct:>7.1f}%  {arrow}")


if __name__ == "__main__":
    main()
