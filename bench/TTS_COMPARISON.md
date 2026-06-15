# TTS comparison (DGX Spark / GB10 / torch 2.11 + cu130)

Same four sentences synthesized by each backend. WAV samples live in `bench/samples/<backend>/{short,mid,long,xlong}.wav`. Listen before deciding — numbers don't capture voice character.

## The four test sentences

| label | text | length |
|---|---|---|
| short | "Hey, what's up?" | 15 ch |
| mid   | "The capital of France is Paris." | 31 ch |
| long  | "I computed it using the Python tool and the answer is seven hundred and fourteen." | 81 ch |
| xlong | "Let me take a closer look. Based on what you showed me, the architecture has three clear layers…" | 189 ch |

## Latency (synth time, not TTFT)

| sentence | kokoro CPU | **kokoro CUDA** | chatterbox CUDA | vibevoice CUDA |
|---|---|---|---|---|
| short (15 ch) | 575 ms | **52 ms** | 693 ms | **1740 ms** |
| mid (31 ch) | 881 ms | **49 ms** | 1458 ms | **1800 ms** |
| long (81 ch) | 1223 ms | **78 ms** | 2284 ms | **3670 ms** |
| xlong (189 ch) | 2247 ms | **160 ms** | 5158 ms | **7060 ms** |

## RTF (synth_ms / audio_ms, lower is better; 1.0 = realtime)

| backend | median RTF |
|---|---|
| kokoro CUDA | **0.015** (~65× realtime) |
| kokoro CPU | 0.30 |
| chatterbox CUDA | 0.58 |
| vibevoice CUDA | 0.67 |

Kokoro CUDA has ~65× headroom before it would miss a realtime deadline. Chatterbox and VibeVoice run close to 1.0 on short utterances, meaning any stall and you drop audio.

## So what's left to compare — subjectively

- **Quality (naturalness, prosody):** listen to the `xlong.wav` samples side-by-side. That's where small-model TTS (Kokoro) usually falls apart and bigger-model TTS (VibeVoice, Chatterbox) tends to win.
- **Voice character:** Kokoro uses `af_bella` (female US). Chatterbox uses its default trained voice. VibeVoice uses `Grace` (female US).
- **Voice cloning:** Kokoro has none. Chatterbox/VibeVoice both support zero-shot.
- **Emotion control:** Chatterbox has an `exaggeration` knob (0–1). Kokoro has none. VibeVoice has cfg_scale.

## Honest recommendation

**Stay on Kokoro CUDA as the production default.** It's 14-48× faster than anything else on GB10, runs at 65× realtime, and the voice is good enough that most listeners don't reach for the replace button.

**Expose `TTS_ENGINE=chatterbox` or `vibevoice` as opt-in modes** for:
- Long-form narration (audiobook-style demos) where quality trumps latency
- Voice cloning demos
- Character voice work

**Don't invest more time hunting new backends** unless:
- A future release publishes sub-100 ms TTFT *and* streaming *and* quality > Kokoro (no such release exists as of April 2026)
- The target platform changes (e.g. non-aarch64, where onnxruntime-gpu unlocks CosyVoice2)

## Models tried and discarded

| model | why rejected |
|---|---|
| CosyVoice2-0.5B | requires `onnxruntime-gpu` + `numpy==1.26.4` + `deepspeed==0.15.1` — broken combo on aarch64 / current torch |
| XTTS-v2 | user said "i dont care about xtts v2" |
| Kokoro-ONNX | onnxruntime-gpu not on aarch64 pypi, ORT CPU only → slower than kokoro-torch CUDA |
| F5-TTS | no streaming, similar quality to Kokoro at much higher cost |
| Sesame CSM-1B | 1B params, no streaming, slow |
| Orpheus-3B | 3B — VRAM collision with Qwen3.6-35B-A3B |
| MetaVoice, Dia, NeuTTS Air, VibeVoice-1.5B | nothing meaningfully better than Kokoro CUDA |

## Reproduce

```bash
# Kokoro + Chatterbox (same container)
docker build -f bench/Dockerfile.tts -t realtime2-tts .
docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v $(pwd):/workspace/realtime2 -w /workspace/realtime2 \
    realtime2-tts python bench/gen_samples.py

# VibeVoice (separate container — its transformers pin conflicts)
docker build -f bench/Dockerfile.vibevoice -t realtime2-vibevoice .
docker run --rm --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -v $(pwd):/workspace/realtime2 -v /tmp:/hosttmp \
    realtime2-vibevoice bash -c 'cd /opt/VibeVoice && \
    for f in /hosttmp/t_short.txt /hosttmp/t_mid.txt /hosttmp/t_long.txt /hosttmp/t_xlong.txt; do \
      python demo/realtime_model_inference_from_file.py \
        --model_path microsoft/VibeVoice-Realtime-0.5B \
        --txt_path $f --speaker_name Grace --device cuda \
        --output_dir /workspace/realtime2/bench/samples/vibevoice/; \
    done'
```
