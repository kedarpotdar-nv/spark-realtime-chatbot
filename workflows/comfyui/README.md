# ComfyUI workflow templates (API format)

These JSON files are **ComfyUI API prompts** (flattened, no subgraph proxies). They were rebuilt from ComfyUI v0.37 blueprint subgraphs:

| File | Source blueprint |
|------|------------------|
| `flux2_klein_t2i.json` | `Image Edit (Flux.2 Klein 4B)` without reference-image path (Klein loaders, euler, 20 steps, cfg 5) |
| `flux2_klein_edit.json` | `Image Edit (Flux.2 Klein 4B)` with `Reference Conditioning` inlined |
| `ace_step_1_5_music.json` | `Text to Audio (ACE-Step 1.5)` |

Default checkpoint filenames match a typical local install (`flux-2-klein-4b-fp8 (1).safetensors`, `qwen_3_4b (1).safetensors`, `flux2-vae (1).safetensors`, `acestep_v1.5_turbo.safetensors`, etc.) and are overridable via env (see below).

Validate against your ComfyUI tree:

```bash
python scripts/validate_comfy_workflows.py "/path/to/ComfyUI"
python scripts/validate_comfy_workflows.py "/path/to/ComfyUI" --url http://127.0.0.1:8188
```

If your ComfyUI build or custom nodes differ, export your own workflow via **Save (API format)** and set `COMFYUI_FLUX_T2I_WORKFLOW`, `COMFYUI_FLUX_EDIT_WORKFLOW`, or `COMFYUI_ACE_WORKFLOW`.

## Node title convention

Spark injects parameters by `_meta.title` (case-insensitive) and/or `class_type`. Use these titles on the nodes you want Spark to drive:

| Title | Purpose |
|-------|---------|
| `PROMPT` | Positive text (also matches "CLIP Text Encode (Positive Prompt)") |
| `NEGATIVE` | Negative prompt |
| `SEED` | `RandomNoise` or `KSampler` seed |
| `WIDTH` / `HEIGHT` | Latent size (`EmptyFlux2LatentImage` or `PrimitiveInt`) |
| `STEPS` | `Flux2Scheduler` step count |
| `INPUT_IMAGE` | `LoadImage` for edit workflows |
| `TAGS` | ACE-Step style tags (`TextEncodeAceStepAudio1.5`) |
| `LYRICS` | Optional lyrics on `TextEncodeAceStepAudio1.5` |
| `DURATION` | `EmptyAceStep1.5LatentAudio` or duration primitive |
| `UNET` / `CLIP` / `VAE` | Model loaders (optional; filenames overridable via env) |
| `SAVE_IMAGE` / `SAVE_AUDIO` | Output nodes |

Model filenames can also be overridden with `COMFYUI_FLUX_UNET`, `COMFYUI_FLUX_EDIT_UNET`, `COMFYUI_FLUX_CLIP`, `COMFYUI_FLUX_VAE`, `COMFYUI_ACE_UNET`, and `COMFYUI_ACE_VAE`.