# ComfyUI workflow templates (API format)

These JSON files are **ComfyUI API prompts** (not UI canvas exports). They were derived from [Comfy-Org/workflow_templates](https://github.com/Comfy-Org/workflow_templates):

| File | Source template |
|------|-----------------|
| `flux2_klein_t2i.json` | `image_flux2_klein_text_to_image` (FLUX.2 Klein 4B Distilled subgraph) |
| `flux2_klein_edit.json` | `image_flux2_klein_image_edit_4b_distilled` |
| `ace_step_1_5_music.json` | `audio_ace_step1_5_xl_turbo` |

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