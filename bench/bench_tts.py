#!/usr/bin/env python3
"""TTS bench: Kokoro (CPU + CUDA) vs Chatterbox-Turbo (CUDA) on GB10.

Measures per-utterance TTFT (time until the first audio sample appears) and
realtime factor (synth_ms / audio_ms). The voice-assistant experience is
bounded by TTFT, not by full-synthesis time — a TTS that starts speaking in
50ms but finishes in 2s feels faster than one that finishes in 300ms but
returns nothing until done.

Run inside realtime2-tts image:
  docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
      -v /home/nvidia/hfcache:/root/.cache/huggingface \
      -v /home/nvidia/realtime2:/workspace/realtime2 \
      realtime2-tts:latest \
      python3 bench/bench_tts.py --out bench/tts.json
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

SENTENCES = [
    "Hey, what's up?",
    "The capital of France is Paris.",
    "I computed it using the Python tool and the answer is seven hundred and fourteen.",
    "Let me take a closer look. Based on what you showed me, the architecture has three clear layers, an input adapter, a reasoning core, and a renderer, and the renderer is doing too much work.",
]


def _stat(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return {
        "n": len(vals),
        "median": round(statistics.median(vals), 1),
        "p90": round(statistics.quantiles(vals, n=10)[-1], 1) if len(vals) >= 3 else round(max(vals), 1),
        "min": round(min(vals), 1),
        "max": round(max(vals), 1),
    }


def bench_kokoro(device: str, voice: str, trials: int):
    import numpy as np
    from kokoro import KPipeline
    print(f"[kokoro] device={device} voice={voice}", file=sys.stderr)
    pipe = KPipeline(lang_code="a", device=device)
    # Warmup
    list(pipe("Hi.", voice=voice, speed=1.2))
    rows = []
    for s in SENTENCES:
        for _ in range(trials):
            t0 = time.perf_counter()
            ttft = None
            total_samples = 0
            sr = 24000
            # KPipeline 0.9+ yields Result objects; older returns (graphemes, phonemes, audio)
            for out in pipe(s, voice=voice, speed=1.2):
                if hasattr(out, "audio"):
                    audio = out.audio
                elif isinstance(out, tuple):
                    audio = out[-1]
                else:
                    audio = out
                if audio is None:
                    continue
                a = audio.detach().cpu().numpy() if hasattr(audio, "detach") else np.asarray(audio)
                if a.size == 0:
                    continue
                if ttft is None:
                    ttft = (time.perf_counter() - t0) * 1000
                total_samples += a.size
            total_ms = (time.perf_counter() - t0) * 1000
            audio_ms = (total_samples / sr) * 1000 if total_samples else 0
            rtf = (total_ms / audio_ms) if audio_ms else None
            rows.append({"text_len": len(s), "ttft_ms": ttft, "total_ms": total_ms, "audio_ms": audio_ms, "rtf": rtf})
            print(f"  kokoro/{device} len={len(s):3d} ttft={ttft:.0f}ms synth={total_ms:.0f}ms audio={audio_ms:.0f}ms rtf={rtf:.3f}" if rtf else f"  (no audio)", file=sys.stderr)
    return {
        "backend": f"kokoro-{device}",
        "voice": voice,
        "ttft_ms": _stat([r["ttft_ms"] for r in rows]),
        "total_ms": _stat([r["total_ms"] for r in rows]),
        "rtf": _stat([r["rtf"] for r in rows]),
        "rows": rows,
    }


def bench_chatterbox(device: str, trials: int):
    import torch
    from chatterbox.tts import ChatterboxTTS
    print(f"[chatterbox] device={device}", file=sys.stderr)
    model = ChatterboxTTS.from_pretrained(device=device)
    sr = model.sr
    # Warmup
    _ = model.generate("Hi.", exaggeration=0.5, cfg_weight=0.5)
    rows = []
    for s in SENTENCES:
        for _ in range(trials):
            t0 = time.perf_counter()
            ttft = None
            # Chatterbox's public API is one-shot — no native streaming.
            # We still measure total (to compare against Kokoro's total) and
            # note that TTFT ≈ total in this mode.
            wav = model.generate(s, exaggeration=0.5, cfg_weight=0.5)
            total_ms = (time.perf_counter() - t0) * 1000
            if hasattr(wav, "detach"):
                wav = wav.detach().cpu()
            samples = wav.numel() if hasattr(wav, "numel") else wav.size
            audio_ms = (samples / sr) * 1000 if sr else 0
            rtf = (total_ms / audio_ms) if audio_ms else None
            # TTFT is not distinguishable from total in one-shot mode
            ttft = total_ms
            rows.append({"text_len": len(s), "ttft_ms": ttft, "total_ms": total_ms, "audio_ms": audio_ms, "rtf": rtf})
            print(f"  chatterbox/{device} len={len(s):3d} synth={total_ms:.0f}ms audio={audio_ms:.0f}ms rtf={rtf:.3f}" if rtf else f"  (no audio)", file=sys.stderr)
    return {
        "backend": f"chatterbox-{device}",
        "ttft_ms": _stat([r["ttft_ms"] for r in rows]),
        "total_ms": _stat([r["total_ms"] for r in rows]),
        "rtf": _stat([r["rtf"] for r in rows]),
        "rows": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--only", choices=["kokoro_cpu", "kokoro_cuda", "chatterbox"], default=None)
    ap.add_argument("--voice", default="af_bella")
    args = ap.parse_args()

    out = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "trials": args.trials, "results": {}}

    try:
        import torch
        out["torch"] = torch.__version__
        out["cuda"] = torch.cuda.is_available()
        out["device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        pass

    if args.only in (None, "kokoro_cpu"):
        try:
            out["results"]["kokoro_cpu"] = bench_kokoro("cpu", args.voice, args.trials)
        except Exception as e:
            out["results"]["kokoro_cpu"] = {"error": str(e)}
            print(f"[kokoro cpu] ERROR: {e}", file=sys.stderr)
    if args.only in (None, "kokoro_cuda"):
        try:
            out["results"]["kokoro_cuda"] = bench_kokoro("cuda", args.voice, args.trials)
        except Exception as e:
            out["results"]["kokoro_cuda"] = {"error": str(e)}
            print(f"[kokoro cuda] ERROR: {e}", file=sys.stderr)
    if args.only in (None, "chatterbox"):
        try:
            out["results"]["chatterbox_cuda"] = bench_chatterbox("cuda", args.trials)
        except Exception as e:
            out["results"]["chatterbox_cuda"] = {"error": str(e)}
            print(f"[chatterbox cuda] ERROR: {e}", file=sys.stderr)

    print(json.dumps(out, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
        print(f"[bench] wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
