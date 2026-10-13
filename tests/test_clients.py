import base64
import io
import json

import httpx
import pytest
from PIL import Image

from gvc.captioner import (
    FADE_NOTE,
    LlamaServerCaptioner,
    OllamaCaptioner,
    build_request,
    clean_caption,
    encode_image,
    llama_server_cmd,
)
from gvc.config import CaptionConfig, GenerateConfig
from gvc.generator import SdServerGenerator, generation_size, sd_server_cmd
from gvc.service import is_up

from .conftest import png


def test_generation_size_keeps_aspect_and_multiples_of_64():
    assert generation_size(1920, 1080, 512) == (704, 384)
    assert generation_size(1080, 1920, 512) == (384, 704)
    assert generation_size(100, 100, 512) == (512, 512)


def test_clean_caption():
    assert clean_caption('  Description: "a red  car,\n parked" ') == "a red car, parked"


def _capture(responses):
    requests = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return responses(request)

    return requests, httpx.MockTransport(handler)


def _frames(tmp_path):
    cur, prev = tmp_path / "cur.png", tmp_path / "prev.png"
    cur.write_bytes(png((0, 0, 255), (1920, 1080)))
    prev.write_bytes(png((255, 0, 0), (1920, 1080)))
    return cur, prev


def _decode(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64)))


def test_encode_image_scales_longest_side_and_uses_jpeg(tmp_path):
    cur, _ = _frames(tmp_path)
    small = _decode(encode_image(cur, 512))
    assert small.format == "JPEG" and small.size == (512, 288)
    assert _decode(encode_image(cur, 0)).size == (1920, 1080)
    assert _decode(encode_image(cur, 4000)).size == (1920, 1080)  # never upscales


def test_ollama_context_message(tmp_path):
    cur, prev = _frames(tmp_path)
    requests, transport = _capture(lambda r: httpx.Response(200, json={"message": {"content": "Prompt: a dog"}}))
    captioner = OllamaCaptioner("http://ollama", "m", image_size=256, transport=transport)

    assert captioner.describe(cur, ["a cat"], prev) == "a dog"
    body = json.loads(requests[0].content)
    user = body["messages"][1]
    assert "a cat" in user["content"]
    # Previous frame first, then the current one, both scaled down.
    first, second = (_decode(b) for b in user["images"])
    assert first.getpixel((0, 0))[0] > 200 and second.getpixel((0, 0))[2] > 200
    assert second.size == (256, 144)
    assert body["options"]["seed"] == 42

    captioner.describe(cur, [])
    user = json.loads(requests[1].content)["messages"][1]
    assert len(user["images"]) == 1


def test_fade_note_only_for_faded_frames(tmp_path):
    cur, prev = _frames(tmp_path)
    for history, prev_image in (([], None), (["a cat"], None), (["a cat"], prev)):
        assert FADE_NOTE not in build_request(cur, history, prev_image, 64)[0]
        assert build_request(cur, history, prev_image, 64, faded=True)[0].endswith(FADE_NOTE)


def test_llama_server_openai_message(tmp_path):
    cur, prev = _frames(tmp_path)
    requests, transport = _capture(
        lambda r: httpx.Response(200, json={"choices": [{"message": {"content": "a dog"}}]})
    )
    captioner = LlamaServerCaptioner("http://llama", seed=3, transport=transport)
    assert captioner.describe(cur, ["a cat"], prev) == "a dog"
    assert requests[0].url.path == "/v1/chat/completions"
    body = json.loads(requests[0].content)
    parts = body["messages"][1]["content"]
    assert [p["type"] for p in parts] == ["image_url", "image_url", "text"]
    assert parts[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert "a cat" in parts[2]["text"]
    assert body["seed"] == 3 and body["max_tokens"] == 180


def test_llama_server_cmd():
    cmd = llama_server_cmd(CaptionConfig(parallel=3, llama_url="http://0.0.0.0:9001"))
    assert cmd[:5] == ["llama-server", "--host", "0.0.0.0", "--port", "9001"]
    assert cmd[cmd.index("--parallel") + 1] == "3"
    assert cmd[cmd.index("--hf-repo") + 1] == "ggml-org/Qwen2.5-VL-7B-Instruct-GGUF:Q4_K_M"

    cmd = llama_server_cmd(CaptionConfig(llama_model="m.gguf", llama_mmproj="p.gguf", llama_server_args=["-fa", "on"]))
    assert "--hf-repo" not in cmd
    assert cmd[cmd.index("--model") + 1] == "m.gguf" and cmd[cmd.index("--mmproj") + 1] == "p.gguf"
    assert cmd[-2:] == ["-fa", "on"]


def test_is_up_waits_for_loading_server(monkeypatch):
    status = {"code": 503}
    monkeypatch.setattr(httpx, "get", lambda url, timeout: httpx.Response(status["code"]))
    assert not is_up("http://llama/health")
    status["code"] = 200
    assert is_up("http://llama/health")


def _sd(cfg: GenerateConfig):
    image = base64.b64encode(png((1, 2, 3), (8, 8))).decode()
    requests, transport = _capture(lambda r: httpx.Response(200, json={"images": [image]}))
    return requests, SdServerGenerator(cfg, transport=transport)


def test_sd_txt2img_and_img2img():
    requests, gen = _sd(GenerateConfig())
    gen.generate("a cat", 64, 64, 5)
    gen.generate("a cat", 64, 64, 5, init=b"init", strength=0.4)
    assert requests[0].url.path == "/sdapi/v1/txt2img"
    assert requests[1].url.path == "/sdapi/v1/img2img"
    body = json.loads(requests[1].content)
    assert body["denoising_strength"] == 0.4
    assert body["init_images"] == [base64.b64encode(b"init").decode()]
    assert body["seed"] == 5


def test_sd_reference_ip_adapter_and_ref_images(monkeypatch):
    monkeypatch.setattr("gvc.generator.POLL_INTERVAL", 0)
    image = base64.b64encode(png((1, 2, 3), (8, 8))).decode()
    polls = iter(["queued", "generating", "completed"])

    def native(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(202, json={"id": "job_1", "status": "queued", "poll_url": "/sdcpp/v1/jobs/job_1"})
        status = next(polls)
        result = {"images": [{"index": 0, "b64_json": image}]} if status == "completed" else None
        return httpx.Response(200, json={"status": status, "result": result, "error": None})

    requests, transport = _capture(native)
    gen = SdServerGenerator(GenerateConfig(init="chain-ref", ref_strength=0.6), transport=transport)
    assert gen.generate("a cat", 64, 64, 5, reference=b"ref") == base64.b64decode(image)
    # Native API: the image is a JSON field, never part of the prompt.
    assert requests[0].url.path == "/sdcpp/v1/img_gen"
    body = json.loads(requests[0].content)
    assert body["prompt"] == "a cat"
    assert body["ip_adapter_image"] == base64.b64encode(b"ref").decode()
    assert body["ip_adapter_strength"] == 0.6
    assert body["seed"] == 5 and body["sample_params"]["sample_steps"] == GenerateConfig().steps
    assert [r.url.path for r in requests[1:]] == ["/sdcpp/v1/jobs/job_1"] * 3

    requests, gen = _sd(GenerateConfig(init="chain-ref", ref_method="ref-images"))
    gen.generate("a cat", 64, 64, 5, reference=b"ref")
    body = json.loads(requests[0].content)
    assert body["prompt"] == "a cat"
    assert body["extra_images"] == [base64.b64encode(b"ref").decode()]


def test_sd_server_cmd_loads_ip_adapter_only_when_needed():
    assert "--ip-adapter" not in sd_server_cmd(GenerateConfig(init="chain"))
    cmd = sd_server_cmd(GenerateConfig(init="chain-ref", server_url="http://0.0.0.0:9000", server_args=["--diffusion-fa"]))
    assert cmd[:5] == ["sd-server", "-l", "0.0.0.0", "--listen-port", "9000"]
    assert "--ip-adapter" in cmd and "--clip_vision" in cmd
    assert cmd[-1] == "--diffusion-fa"


def test_sd_native_job_failure_raises(monkeypatch):
    monkeypatch.setattr("gvc.generator.POLL_INTERVAL", 0)

    def native(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(202, json={"id": "j", "status": "queued", "poll_url": "/sdcpp/v1/jobs/j"})
        return httpx.Response(200, json={"status": "failed", "result": None, "error": {"message": "boom"}})

    _, transport = _capture(native)
    gen = SdServerGenerator(GenerateConfig(init="chain-ref"), transport=transport)
    with pytest.raises(RuntimeError, match="boom"):
        gen.generate("a cat", 64, 64, 5, reference=b"ref")
