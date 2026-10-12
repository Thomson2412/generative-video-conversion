"""Frame -> text, using a vision model served by Ollama or llama.cpp's llama-server."""

from __future__ import annotations

import base64
import io
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Protocol

import httpx
from PIL import Image

from .config import CaptionConfig
from .service import host_port, managed_service

SYSTEM_PROMPT = """\
You write prompts for a text-to-image model. Describe the image as one line of \
comma-separated visual phrases: main subjects and their appearance, pose and action, \
setting, composition and camera angle or shot type, lighting, dominant colours, and \
visual style or medium. Be concrete and visual. Do not speculate, tell a story, or \
mention "the image", "the frame" or "the scene". At most {max_words} words."""

CONTEXT_PROMPT = """\
Descriptions of the previous frames, oldest first:
{history}

Describe this new frame. Keep the wording of the latest description wherever it still \
applies and change only what is different, so consecutive descriptions stay consistent."""

CONTEXT_IMAGE_PROMPT = """\
The first image is the previous frame, the second image is the current frame.
Descriptions of the previous frames, oldest first:
{history}

Describe the current (second) image. Keep the wording of the latest description \
wherever it still applies and change only what differs between the two images."""

FRESH_PROMPT = "Describe this image."

JPEG_QUALITY = 90


class Captioner(Protocol):
    def describe(self, image: Path, history: list[str], prev_image: Path | None = None) -> str: ...


def encode_image(path: Path, max_size: int) -> str:
    """Base64 JPEG, longest side scaled down to max_size (0 keeps the original size).

    Vision models spend tokens per image patch, so a smaller image is the
    biggest single speed-up for captioning.
    """
    with Image.open(path) as img:
        img = img.convert("RGB")
        if max_size and max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return base64.b64encode(buf.getvalue()).decode()


def clean_caption(text: str) -> str:
    text = re.sub(r"^\s*(description|prompt)\s*:\s*", "", text.strip(), flags=re.IGNORECASE)
    text = " ".join(text.split())
    return text.strip("\"' ")


def build_request(
    image: Path, history: list[str], prev_image: Path | None, max_size: int
) -> tuple[str, list[str]]:
    """User prompt text and the base64 images to send with it, in order."""
    if not history:
        return FRESH_PROMPT, [encode_image(image, max_size)]
    numbered = "\n".join(f"- {h}" for h in history)
    if prev_image is None:
        return CONTEXT_PROMPT.format(history=numbered), [encode_image(image, max_size)]
    return CONTEXT_IMAGE_PROMPT.format(history=numbered), [
        encode_image(prev_image, max_size),
        encode_image(image, max_size),
    ]


class OllamaCaptioner:
    def __init__(
        self,
        url: str,
        model: str,
        max_words: int = 60,
        temperature: float = 0.2,
        seed: int = 42,
        image_size: int = 512,
        timeout: float = 600.0,
        transport: httpx.BaseTransport | None = None,
    ):
        self.model = model
        self.max_words = max_words
        self.image_size = image_size
        self.options = {
            "temperature": temperature,
            "seed": seed,
            # Roughly 2 tokens per word plus slack; stops rambling models.
            "num_predict": max_words * 3,
        }
        self.client = httpx.Client(base_url=url, timeout=timeout, transport=transport)

    def ensure_model(self) -> None:
        """Pull the model if Ollama does not have it yet."""
        r = self.client.post("/api/show", json={"model": self.model})
        if r.status_code == 404:
            self.client.post("/api/pull", json={"model": self.model, "stream": False}, timeout=None).raise_for_status()
        else:
            r.raise_for_status()

    def describe(self, image: Path, history: list[str], prev_image: Path | None = None) -> str:
        text, images = build_request(image, history, prev_image, self.image_size)
        r = self.client.post(
            "/api/chat",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT.format(max_words=self.max_words)},
                    {"role": "user", "content": text, "images": images},
                ],
                "stream": False,
                "options": self.options,
            },
        )
        r.raise_for_status()
        return clean_caption(r.json()["message"]["content"])

    def unload(self) -> None:
        """Free the model's VRAM so the image generator can use it."""
        try:
            self.client.post("/api/generate", json={"model": self.model, "keep_alive": 0})
        except httpx.HTTPError:
            pass


class LlamaServerCaptioner:
    """llama.cpp's llama-server through its OpenAI-compatible chat endpoint."""

    def __init__(
        self,
        url: str,
        max_words: int = 60,
        temperature: float = 0.2,
        seed: int = 42,
        image_size: int = 512,
        timeout: float = 600.0,
        transport: httpx.BaseTransport | None = None,
    ):
        self.max_words = max_words
        self.image_size = image_size
        self.params = {"temperature": temperature, "seed": seed, "max_tokens": max_words * 3}
        self.client = httpx.Client(base_url=url, timeout=timeout, transport=transport)

    def describe(self, image: Path, history: list[str], prev_image: Path | None = None) -> str:
        text, images = build_request(image, history, prev_image, self.image_size)
        content = [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}} for b64 in images]
        content.append({"type": "text", "text": text})
        r = self.client.post(
            "/v1/chat/completions",
            json={
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT.format(max_words=self.max_words)},
                    {"role": "user", "content": content},
                ],
                **self.params,
            },
        )
        r.raise_for_status()
        return clean_caption(r.json()["choices"][0]["message"]["content"])


def llama_server_cmd(cfg: CaptionConfig) -> list[str]:
    host, port = host_port(cfg.llama_url, 8080)
    cmd = [
        "llama-server", "--host", host, "--port", str(port),
        "--n-gpu-layers", "999",
        # One slot per concurrent request, each with its own 4k context; without this
        # the KV cache would be sized for the model's full (128k) context.
        "--parallel", str(cfg.parallel), "--kv-unified-per-slot", "4096",
    ]
    if cfg.llama_model.endswith(".gguf"):
        cmd += ["--model", cfg.llama_model]
        if cfg.llama_mmproj:
            cmd += ["--mmproj", cfg.llama_mmproj]
    else:
        cmd += ["--hf-repo", cfg.llama_model]
    return cmd + list(cfg.llama_server_args)


@contextmanager
def captioner_service(cfg: CaptionConfig, log_dir: Path) -> Iterator[Captioner]:
    """A ready captioner for the configured backend; its VRAM is freed on exit."""
    if cfg.backend == "llama-server":
        with managed_service(
            "llama-server",
            f"{cfg.llama_url.rstrip('/')}/health",
            llama_server_cmd(cfg),
            log_dir / "llama-server.log",
            cfg.manage_server,
            # The first start may download several GB from Hugging Face.
            startup_timeout=3600.0,
        ):
            yield LlamaServerCaptioner(cfg.llama_url, cfg.max_words, cfg.temperature, cfg.seed, cfg.image_size, cfg.timeout)
        return

    host, port = host_port(cfg.ollama_url, 11434)
    with managed_service(
        "ollama",
        cfg.ollama_url,
        ["ollama", "serve"],
        log_dir / "ollama.log",
        cfg.manage_server,
        # Only applies to a server gvc starts; a running one keeps its own setting.
        env={"OLLAMA_HOST": f"{host}:{port}", "OLLAMA_NUM_PARALLEL": str(cfg.parallel)},
    ):
        captioner = OllamaCaptioner(
            cfg.ollama_url, cfg.model, cfg.max_words, cfg.temperature, cfg.seed, cfg.image_size, cfg.timeout
        )
        captioner.ensure_model()
        try:
            yield captioner
        finally:
            captioner.unload()
