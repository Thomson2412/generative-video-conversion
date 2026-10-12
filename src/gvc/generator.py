"""Text (+ optional image) -> image, using stable-diffusion.cpp's sd-server."""

from __future__ import annotations

import base64
import json
import math
from typing import Protocol

import httpx

from .config import GenerateConfig
from .service import host_port


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
                # sdapi has no IP-Adapter field; native sd.cpp parameters ride along
                # inside the prompt and are stripped by the server before generation.
                extra = {"ip_adapter_image": ref, "ip_adapter_strength": self.cfg.ref_strength}
                prompt = f"{prompt} <sd_cpp_extra_args>{json.dumps(extra)}</sd_cpp_extra_args>"
            else:
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
