"""Text (+ optional image) -> image, using stable-diffusion.cpp's sd-server."""

from __future__ import annotations

import base64
import math
import time
from typing import Protocol

import httpx

from .config import GenerateConfig
from .service import host_port


# Seconds between job status checks on sd-server's native API.
POLL_INTERVAL = 0.1


class Generator(Protocol):
    def generate(
        self,
        prompt: str,
        width: int,
        height: int,
        seed: int,
        init: bytes | None = None,
        strength: float = 0.0,
        reference: bytes | None = None,
    ) -> bytes:
        """Return a PNG. `init` makes it img2img; `reference` conditions on an image."""
        ...


def generation_size(width: int, height: int, size: int) -> tuple[int, int]:
    """Source aspect ratio, about size*size pixels, both sides multiples of 64."""
    aspect = width / height
    h = math.sqrt(size * size / aspect)
    w = h * aspect

    def snap(v: float) -> int:
        return max(64, int(round(v / 64)) * 64)

    return snap(w), snap(h)


def build_prompt(caption: str, style: str) -> str:
    return f"{caption}, {style}" if style else caption


def sd_server_cmd(cfg: GenerateConfig) -> list[str]:
    host, port = host_port(cfg.server_url, 7860)
    cmd = ["sd-server", "-l", host, "--listen-port", str(port), "-m", cfg.model]
    if cfg.vae:
        cmd += ["--vae", cfg.vae]
    if cfg.init == "chain-ref" and cfg.ref_method == "ip-adapter":
        cmd += ["--clip_vision", cfg.clip_vision, "--ip-adapter", cfg.ip_adapter]
    return cmd + list(cfg.server_args)


class SdServerGenerator:
    def __init__(self, cfg: GenerateConfig, transport: httpx.BaseTransport | None = None):
        self.cfg = cfg
        self.client = httpx.Client(base_url=cfg.server_url, timeout=cfg.timeout, transport=transport)

    def generate(
        self,
        prompt: str,
        width: int,
        height: int,
        seed: int,
        init: bytes | None = None,
        strength: float = 0.0,
        reference: bytes | None = None,
    ) -> bytes:
        body: dict = {
            "negative_prompt": self.cfg.negative_prompt,
            "width": width,
            "height": height,
            "steps": self.cfg.steps,
            "cfg_scale": self.cfg.cfg_scale,
            "seed": seed,
            "sampler_name": self.cfg.sampler,
            "batch_size": 1,
        }
        if reference is not None:
            ref = base64.b64encode(reference).decode()
            if self.cfg.ref_method == "ip-adapter":
                return self._native_ip_adapter(prompt, width, height, seed, ref)
            body["extra_images"] = [ref]
        body["prompt"] = prompt

        endpoint = "/sdapi/v1/txt2img"
        if init is not None:
            endpoint = "/sdapi/v1/img2img"
            body["init_images"] = [base64.b64encode(init).decode()]
            body["denoising_strength"] = strength

        r = self.client.post(endpoint, json=body)
        r.raise_for_status()
        images = r.json().get("images") or []
        if not images:
            raise RuntimeError(f"sd-server returned no image: {r.text[:500]}")
        return base64.b64decode(images[0])

    def _native_ip_adapter(self, prompt: str, width: int, height: int, seed: int, ref: str) -> bytes:
        """txt2img with an IP-Adapter image through sd-server's native job API.

        sdapi has no IP-Adapter field, and smuggling the image in through the prompt's
        <sd_cpp_extra_args> makes the server's std::regex overflow the stack and crash.
        """
        body = {
            "prompt": prompt,
            "negative_prompt": self.cfg.negative_prompt,
            "width": width,
            "height": height,
            "seed": seed,
            "batch_count": 1,
            "ip_adapter_image": ref,
            "ip_adapter_strength": self.cfg.ref_strength,
            "sample_params": {
                "sample_method": self.cfg.sampler,
                "sample_steps": self.cfg.steps,
                "guidance": {"txt_cfg": self.cfg.cfg_scale},
            },
            "output_format": "png",
        }
        r = self.client.post("/sdcpp/v1/img_gen", json=body)
        r.raise_for_status()
        poll_url = r.json()["poll_url"]
        deadline = time.monotonic() + self.cfg.timeout
        while True:
            job = self.client.get(poll_url)
            job.raise_for_status()
            data = job.json()
            if data["status"] == "completed":
                return base64.b64decode(data["result"]["images"][0]["b64_json"])
            if data["status"] in ("failed", "cancelled"):
                raise RuntimeError(f"sd-server job {data['status']}: {data.get('error')}")
            if time.monotonic() > deadline:
                self.client.post(f"{poll_url}/cancel")
                raise RuntimeError(f"sd-server job did not finish within {self.cfg.timeout:.0f}s")
            time.sleep(POLL_INTERVAL)
