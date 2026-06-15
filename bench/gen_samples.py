#!/usr/bin/env python3
"""Generate same-sentence samples from whatever backends are installed
in the current container. Saves WAVs to bench/samples/<backend>/.
"""
import os, sys, time, json
from pathlib import Path

SENTENCES = {
    "short":  "Hey, what's up?",
    "mid":    "The capital of France is Paris.",
    "long":   "I computed it using the Python tool and the answer is seven hundred and fourteen.",
    "xlong":  "Let me take a closer look. Based on what you showed me, the architecture has three clear layers, an input adapter, a reasoning core, and a renderer, and the renderer is doing too much work.",
}

OUT = Path("/workspace/realtime2/bench/samples")
OUT.mkdir(parents=True, exist_ok=True)


def save_wav(path, samples, sr):
    import numpy as np, soundfile as sf
    if hasattr(samples, "detach"):
        samples = samples.detach().cpu().numpy()
    samples = np.asarray(samples, dtype="float32").reshape(-1)
    sf.write(str(path), samples, sr, subtype="PCM_16")


def run_kokoro(device):
    try:
        from kokoro import KPipeline
    except ImportError as e:
        print(f"[kokoro] skipped: {e}")
        return
    name = f"kokoro_{device}"
    d = OUT / name; d.mkdir(parents=True, exist_ok=True)
    print(f"[{name}] loading...")
    pipe = KPipeline(lang_code="a", device=device)
    # warm
    for _ in pipe("Hi.", voice="af_bella", speed=1.2):
        break
    rows = {}
    for label, s in SENTENCES.items():
        t0 = time.perf_counter()
        import numpy as np
        chunks = []
        for out in pipe(s, voice="af_bella", speed=1.2):
            a = out.audio if hasattr(out, "audio") else (out[-1] if isinstance(out, tuple) else out)
            if a is None: continue
            arr = a.detach().cpu().numpy() if hasattr(a, "detach") else np.asarray(a)
            chunks.append(arr)
        synth_ms = (time.perf_counter() - t0) * 1000
        if not chunks: continue
        audio = np.concatenate(chunks).astype("float32")
        sr = 24000
        audio_ms = len(audio)/sr*1000
        path = d / f"{label}.wav"
        save_wav(path, audio, sr)
        rows[label] = {"synth_ms": round(synth_ms,1), "audio_ms": round(audio_ms,1), "rtf": round(synth_ms/audio_ms,3), "path": str(path)}
        print(f"  {label:<6} synth={synth_ms:.0f}ms audio={audio_ms:.0f}ms rtf={synth_ms/audio_ms:.3f}")
    return rows


def run_chatterbox(device):
    try:
        from chatterbox.tts import ChatterboxTTS
    except ImportError as e:
        print(f"[chatterbox] skipped: {e}")
        return
    name = f"chatterbox_{device}"
    d = OUT / name; d.mkdir(parents=True, exist_ok=True)
    print(f"[{name}] loading...")
    model = ChatterboxTTS.from_pretrained(device=device)
    sr = int(getattr(model, "sr", 24000))
    # warm
    _ = model.generate("Hi.", exaggeration=0.5, cfg_weight=0.5)
    rows = {}
    for label, s in SENTENCES.items():
        t0 = time.perf_counter()
        wav = model.generate(s, exaggeration=0.5, cfg_weight=0.5)
        synth_ms = (time.perf_counter() - t0) * 1000
        if hasattr(wav, "detach"):
            wav = wav.detach().cpu().numpy().reshape(-1)
        audio_ms = len(wav)/sr*1000
        path = d / f"{label}.wav"
        save_wav(path, wav, sr)
        rows[label] = {"synth_ms": round(synth_ms,1), "audio_ms": round(audio_ms,1), "rtf": round(synth_ms/audio_ms,3), "path": str(path)}
        print(f"  {label:<6} synth={synth_ms:.0f}ms audio={audio_ms:.0f}ms rtf={synth_ms/audio_ms:.3f}")
    return rows


def main():
    results = {}
    backend = sys.argv[1] if len(sys.argv) > 1 else None
    if backend == "kokoro" or backend is None:
        try:
            import torch
            if torch.cuda.is_available():
                results["kokoro_cuda"] = run_kokoro("cuda")
        except Exception as e: print(f"[kokoro cuda] error: {e}")
        results["kokoro_cpu"] = run_kokoro("cpu")
    if backend == "chatterbox" or backend is None:
        try:
            import torch
            if torch.cuda.is_available():
                results["chatterbox_cuda"] = run_chatterbox("cuda")
        except Exception as e: print(f"[chatterbox cuda] error: {e}")
    # Write or merge
    out_json = OUT.parent / "samples_index.json"
    existing = {}
    if out_json.exists():
        try: existing = json.loads(out_json.read_text())
        except Exception: existing = {}
    existing.update({k: v for k, v in results.items() if v})
    out_json.write_text(json.dumps(existing, indent=2))
    print(f"[index] wrote {out_json}")


if __name__ == "__main__":
    main()
