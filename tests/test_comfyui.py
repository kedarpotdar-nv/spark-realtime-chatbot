"""ComfyUI client, workflow injection, and runner tests (mocked HTTP — no live ComfyUI)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from clients.comfyui import ComfyUIClient
from clients.comfyui_workflow import inject_flux_t2i, inject_ace_music, load_workflow
from clients.comfy_runner import run_comfy_job


WORKFLOW_DIR = Path(__file__).resolve().parent.parent / "workflows" / "comfyui"


@pytest.fixture
def flux_prompt():
    return load_workflow(WORKFLOW_DIR / "flux2_klein_t2i.json")


def test_inject_flux_t2i_by_title(flux_prompt):
    injected = inject_flux_t2i(
        flux_prompt,
        positive="A red bicycle",
        negative="",
        width=768,
        height=512,
        seed=42,
        steps=6,
        model_overrides={"unet": "flux-2-klein-4b.safetensors", "clip": "qwen_3_4b.safetensors", "vae": "flux2-vae.safetensors"},
    )
    prompts = [n for n in injected.values() if n.get("class_type") == "CLIPTextEncode"]
    assert any("red bicycle" in (n.get("inputs", {}).get("text") or "") for n in prompts)
    latent = next(n for n in injected.values() if n.get("class_type") == "EmptyFlux2LatentImage")
    assert latent["inputs"]["width"] == 768
    assert latent["inputs"]["height"] == 512


def test_inject_ace_music_tags():
    ace = load_workflow(WORKFLOW_DIR / "ace_step_1_5_music.json")
    injected = inject_ace_music(
        ace,
        tags="Chill Lofi, 80 BPM",
        lyrics="",
        duration_seconds=45,
        seed=7,
        model_overrides={"unet": "acestep_v1.5_xl_turbo_bf16.safetensors", "vae": "ace_1.5_vae.safetensors"},
    )
    enc = next(n for n in injected.values() if n.get("class_type") == "TextEncodeAceStepAudio1.5")
    assert "Chill Lofi" in (enc.get("inputs", {}).get("tags") or "")


@pytest.mark.asyncio
async def test_queue_prompt_and_history_poll(monkeypatch, flux_prompt):
    client = ComfyUIClient("http://127.0.0.1:8188", timeout_seconds=30)

    class FakeResp:
        def __init__(self, status=200, payload=None, body=b""):
            self.status = status
            self._payload = payload or {}
            self._body = body

        async def json(self):
            return self._payload

        async def text(self):
            return self._body.decode() if isinstance(self._body, bytes) else str(self._body)

        async def read(self):
            return self._body if isinstance(self._body, bytes) else b""

        def raise_for_status(self):
            if self.status >= 400:
                raise RuntimeError(self.status)

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

    history_calls = {"n": 0}

    class FakeRespJson(FakeResp):
        async def text(self):
            return json.dumps(self._payload)

    class FakeSession:
        def post(self, url, **kwargs):
            assert url.endswith("/prompt")
            return FakeRespJson(payload={"prompt_id": "pid-test-1"})

        def get(self, url, **kwargs):
            if "/history/" in url:
                history_calls["n"] += 1
                if history_calls["n"] < 2:
                    return FakeResp(payload={"pid-test-1": {}})
                return FakeResp(
                    payload={
                        "pid-test-1": {
                            "outputs": {
                                "99": {
                                    "images": [
                                        {
                                            "filename": "spark_00001.png",
                                            "subfolder": "",
                                            "type": "output",
                                        }
                                    ]
                                }
                            }
                        }
                    }
                )
            return FakeResp(payload={})

    class FakeMgr:
        async def get_session(self):
            return FakeSession()

    monkeypatch.setattr("clients.comfyui.get_http_manager", lambda: FakeMgr())

    injected = inject_flux_t2i(
        flux_prompt,
        positive="test",
        negative="",
        width=1024,
        height=1024,
        seed=1,
        steps=4,
        model_overrides={},
    )
    pid = await client.queue_prompt(injected, client_id="c1")
    assert pid == "pid-test-1"
    result = await client.wait_for_outputs(pid, client_id="c1", on_progress=None, poll_interval=0.01)
    assert result.images[0]["filename"] == "spark_00001.png"


@pytest.mark.asyncio
async def test_download_view(monkeypatch):
    client = ComfyUIClient("http://127.0.0.1:8188")

    class FakeResp:
        status = 200

        async def read(self):
            return b"\x89PNG\r\n"

        def raise_for_status(self):
            return None

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

    class FakeSession:
        def get(self, url, **kwargs):
            assert kwargs.get("params", {}).get("filename") == "spark.png"
            return FakeResp()

    class FakeMgr:
        async def get_session(self):
            return FakeSession()

    monkeypatch.setattr("clients.comfyui.get_http_manager", lambda: FakeMgr())
    data = await client.download_view(filename="spark.png")
    assert data.startswith(b"\x89PNG")


@pytest.mark.asyncio
async def test_run_comfy_job_image_mock(monkeypatch, tmp_path):
    """End-to-end runner with Comfy API mocked (no socket I/O)."""
    monkeypatch.setenv("HERMES_WORKSPACE", str(tmp_path))

    async def fake_image(**kwargs):
        rel = Path("media/out.png")
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes(b"png")
        return {"success": True, "path": str(rel), "prompt_id": "p1", "kind": "image"}

    monkeypatch.setattr("clients.comfy_runner.generate_image_comfy", fake_image)

    events = []

    async def notify(t, d):
        events.append((t, d))

    result = await run_comfy_job(
        kind="image",
        params={"prompt": "cat"},
        notify=notify,
        resolve_edit_source=lambda s: (_ for _ in ()).throw(RuntimeError("unused")),
    )
    assert result["success"]
    assert any(t == "media_ready" for t, _ in events)