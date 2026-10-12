import base64
import json

import httpx

from gvc.captioner import OllamaCaptioner, clean_caption
from gvc.config import GenerateConfig
from gvc.generator import SdServerGenerator, generation_size, sd_server_cmd

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


def test_ollama_context_message(tmp_path):
    cur, prev = tmp_path / "cur.png", tmp_path / "prev.png"
    cur.write_bytes(b"current")
    prev.write_bytes(b"previous")
    requests, transport = _capture(lambda r: httpx.Response(200, json={"message": {"content": "Prompt: a dog"}}))
    captioner = OllamaCaptioner("http://ollama", "m", transport=transport)

    assert captioner.describe(cur, ["a cat"], prev) == "a dog"
    body = json.loads(requests[0].content)
    user = body["messages"][1]
    assert "a cat" in user["content"]
    assert user["images"] == [base64.b64encode(b"previous").decode(), base64.b64encode(b"current").decode()]
    assert body["options"]["seed"] == 42

    captioner.describe(cur, [])
    user = json.loads(requests[1].content)["messages"][1]
    assert user["images"] == [base64.b64encode(b"current").decode()]


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


def test_sd_reference_ip_adapter_and_ref_images():
    requests, gen = _sd(GenerateConfig(init="chain-ref", ref_strength=0.6))
    gen.generate("a cat", 64, 64, 5, reference=b"ref")
    body = json.loads(requests[0].content)
    assert requests[0].url.path == "/sdapi/v1/txt2img"
    prompt, extra = body["prompt"].split(" <sd_cpp_extra_args>")
    assert prompt == "a cat"
    extra = json.loads(extra.removesuffix("</sd_cpp_extra_args>"))
    assert extra == {"ip_adapter_image": base64.b64encode(b"ref").decode(), "ip_adapter_strength": 0.6}

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
