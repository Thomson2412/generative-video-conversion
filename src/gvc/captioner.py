"""Frame -> text, using a vision model served by Ollama."""

from __future__ import annotations

import base64
import re
from pathlib import Path
from typing import Protocol

import httpx

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


class Captioner(Protocol):
    def describe(self, image: Path, history: list[str], prev_image: Path | None = None) -> str: ...


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode()


def clean_caption(text: str) -> str:
    text = re.sub(r"^\s*(description|prompt)\s*:\s*", "", text.strip(), flags=re.IGNORECASE)
    text = " ".join(text.split())
    return text.strip("\"' ")


def build_messages(
    image: Path, history: list[str], prev_image: Path | None, max_words: int
) -> list[dict]:
    if history:
        numbered = "\n".join(f"- {h}" for h in history)
        template = CONTEXT_IMAGE_PROMPT if prev_image is not None else CONTEXT_PROMPT
        content = template.format(history=numbered)
    else:
        content = FRESH_PROMPT
    images = ([_b64(prev_image)] if prev_image is not None and history else []) + [_b64(image)]
    return [
        {"role": "system", "content": SYSTEM_PROMPT.format(max_words=max_words)},
        {"role": "user", "content": content, "images": images},
    ]


class OllamaCaptioner:
    def __init__(
        self,
        url: str,
        model: str,
        max_words: int = 60,
        temperature: float = 0.2,
        seed: int = 42,
        timeout: float = 600.0,
        transport: httpx.BaseTransport | None = None,
    ):
        self.model = model
        self.max_words = max_words
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
        r = self.client.post(
            "/api/chat",
            json={
                "model": self.model,
                "messages": build_messages(image, history, prev_image, self.max_words),
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
