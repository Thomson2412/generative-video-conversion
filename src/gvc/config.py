"""Configuration: built-in defaults <- preset <- gvc.toml <- CLI flags."""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

CAPTION_BACKENDS = ("ollama", "llama-server")
CAPTION_CONTEXTS = ("none", "text", "text+image")
GENERATE_INITS = ("none", "chain", "chain-ref", "source")
REF_METHODS = ("ip-adapter", "ref-images")

# A preset only picks the two propagation axes; everything else stays configurable.
PRESETS: dict[str, dict[str, dict[str, Any]]] = {
    "literal": {"caption": {"context": "none"}, "generate": {"init": "none"}},
    "stable-text": {"caption": {"context": "text"}, "generate": {"init": "none"}},
    "chain": {"caption": {"context": "none"}, "generate": {"init": "chain"}},
    "full": {"caption": {"context": "text"}, "generate": {"init": "chain"}},
    "full-ref": {"caption": {"context": "text"}, "generate": {"init": "chain-ref"}},
    "guided": {"caption": {"context": "text"}, "generate": {"init": "source"}},
}
DEFAULT_PRESET = "full"


def runtime(default: Any = None, *, factory: Any = None) -> Any:
    """A field that does not influence the output (URLs, process management).

    Runtime fields are excluded from stage fingerprints, so changing them never
    invalidates cached work.
    """
    if factory is not None:
        return field(default_factory=factory, metadata={"runtime": True})
    return field(default=default, metadata={"runtime": True})


@dataclass
class CaptionConfig:
    # ollama | llama-server
    backend: str = "ollama"
    # Ollama model name (backend = "ollama").
    model: str = "qwen2.5vl:7b"
    # backend = "llama-server": a Hugging Face "repo[:quant]" (downloaded by llama-server,
    # vision projector included) or a path to a local .gguf (then also set llama_mmproj).
    llama_model: str = "ggml-org/Qwen2.5-VL-7B-Instruct-GGUF:Q4_K_M"
    llama_mmproj: str = ""
    # Caption propagation: none | text (previous caption) | text+image (+ previous source frame)
    context: str = "text"
    # How many previous captions to pass when context != none.
    window: int = 1
    max_words: int = 60
    temperature: float = 0.2
    seed: int = 42
    # Longest side of the image sent to the LLM (0 = original). Fewer pixels means
    # fewer image tokens, which dominates captioning time.
    image_size: int = 512
    # A context caption is treated as "stuck" and redone without context when it is
    # at least this similar to the previous caption while the frame changed by at
    # least stale_diff (mean absolute pixel difference, 0..1).
    stale_similarity: float = 0.97
    stale_diff: float = 0.08
    # Concurrent caption requests. Only used when context = "none", because context
    # captions depend on the previous one.
    parallel: int = runtime(4)
    ollama_url: str = runtime("http://127.0.0.1:11434")
    llama_url: str = runtime("http://127.0.0.1:8080")
    # Extra llama-server flags, e.g. ["--image-max-tokens", "256"].
    llama_server_args: list[str] = runtime(factory=list)
    # Start the backend server when nothing is listening on its URL.
    manage_server: bool = runtime(True)
    timeout: float = runtime(600.0)


@dataclass
class GenerateConfig:
    # Image propagation:
    #   none      txt2img from the caption only
    #   chain     img2img, previous generated frame is the starting image
    #   chain-ref txt2img, previous generated frame is a reference (see ref_method)
    #   source    img2img, original frame is the starting image
    init: str = "chain"
    # img2img denoising strength: 0 keeps the init image, 1 ignores it.
    strength: float = 0.55
    # chain-ref: ip-adapter (SD 1.5/SDXL + IP-Adapter weights) or ref-images (edit models
    # such as FLUX Kontext / Qwen-Image-Edit, configured through server_args).
    ref_method: str = "ip-adapter"
    ref_strength: float = 0.7
    seed: int = 42
    steps: int = 4
    cfg_scale: float = 1.0
    sampler: str = "euler_a"
    # Generation resolution: width*height ~= size*size, source aspect ratio, multiples of 64.
    size: int = 512
    # Appended to every caption, e.g. "watercolor painting, soft pastel colours".
    style: str = ""
    negative_prompt: str = ""
    model: str = "models/sd_xl_turbo_1.0_fp16.safetensors"
    vae: str = "models/sdxl_vae-fp16-fix.safetensors"
    # Only loaded for init = "chain-ref" with ref_method = "ip-adapter".
    ip_adapter: str = "models/ip-adapter_sdxl_vit-h.safetensors"
    clip_vision: str = "models/clip_vision_h.safetensors"
    server_url: str = runtime("http://127.0.0.1:7860")
    # Start `sd-server` when nothing is listening on server_url.
    manage_server: bool = runtime(True)
    # Extra sd-server flags, e.g. ["--diffusion-fa", "--vae-tiling"].
    server_args: list[str] = runtime(factory=list)
    timeout: float = runtime(600.0)


@dataclass
class DedupeConfig:
    # Max perceptual-hash (64 bit) Hamming distance for frames to count as duplicates.
    # -1 disables dedupe.
    max_distance: int = 2


@dataclass
class ResetConfig:
    # Mean absolute pixel difference to the previous frame (0..1) that counts as a scene cut.
    scene_threshold: float = 0.25
    # Additionally break both chains every N frames to limit drift (0 = off).
    every_n: int = 0


@dataclass
class FadeConfig:
    # Fades to and from black are applied after generation: the generator draws the
    # content at full brightness and the result is dimmed to match the source.
    enabled: bool = True
    # Mean brightness (0..1) at or below which a source frame counts as black. Black
    # frames are not captioned or generated, they stay black.
    black_level: float = 0.03
    # Minimum brightness rise per frame (relative) for a frame next to black to be
    # part of a fade ramp rather than the shot itself.
    min_rise: float = 0.005


@dataclass
class OutputConfig:
    keep_audio: bool = True
    codec: str = "libx264"
    crf: int = 18
    # Scale generated frames back to the source resolution.
    scale_to_source: bool = True


@dataclass
class Config:
    preset: str = DEFAULT_PRESET
    caption: CaptionConfig = field(default_factory=CaptionConfig)
    generate: GenerateConfig = field(default_factory=GenerateConfig)
    dedupe: DedupeConfig = field(default_factory=DedupeConfig)
    reset: ResetConfig = field(default_factory=ResetConfig)
    fade: FadeConfig = field(default_factory=FadeConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def validate(self) -> None:
        if self.caption.backend not in CAPTION_BACKENDS:
            raise ConfigError(f"caption.backend must be one of {CAPTION_BACKENDS}, got {self.caption.backend!r}")
        if self.caption.parallel < 1:
            raise ConfigError("caption.parallel must be >= 1")
        if self.caption.context not in CAPTION_CONTEXTS:
            raise ConfigError(f"caption.context must be one of {CAPTION_CONTEXTS}, got {self.caption.context!r}")
        if self.generate.init not in GENERATE_INITS:
            raise ConfigError(f"generate.init must be one of {GENERATE_INITS}, got {self.generate.init!r}")
        if self.caption.window < 1:
            raise ConfigError("caption.window must be >= 1")
        if self.generate.ref_method not in REF_METHODS:
            raise ConfigError(f"generate.ref_method must be one of {REF_METHODS}, got {self.generate.ref_method!r}")
        if not 0.0 <= self.generate.strength <= 1.0:
            raise ConfigError("generate.strength must be between 0 and 1")
        if self.generate.size < 64:
            raise ConfigError("generate.size must be >= 64")
        if not 0.0 <= self.fade.black_level < 1.0:
            raise ConfigError("fade.black_level must be between 0 and 1")


class ConfigError(ValueError):
    pass


SECTIONS = {
    "caption": CaptionConfig,
    "generate": GenerateConfig,
    "dedupe": DedupeConfig,
    "reset": ResetConfig,
    "fade": FadeConfig,
    "output": OutputConfig,
}


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _build(data: dict[str, Any]) -> Config:
    unknown = set(data) - set(SECTIONS) - {"preset"}
    if unknown:
        raise ConfigError(f"unknown config keys: {sorted(unknown)}")
    sections = {}
    for name, cls in SECTIONS.items():
        values = data.get(name, {})
        known = {f.name for f in fields(cls)}
        bad = set(values) - known
        if bad:
            raise ConfigError(f"unknown keys in [{name}]: {sorted(bad)}")
        sections[name] = cls(**values)
    config = Config(preset=data.get("preset", DEFAULT_PRESET), **sections)
    config.validate()
    return config


def load_config(
    path: Path | None = None,
    preset: str | None = None,
    overrides: dict[str, dict[str, Any]] | None = None,
) -> Config:
    """Load config with precedence: defaults <- preset <- file <- overrides.

    `preset` (from the CLI) beats a preset named in the file. Explicit settings in
    the file still beat the preset, so a file can say `preset = "full"` and then
    tweak `caption.context`.
    """
    file_data: dict[str, Any] = {}
    if path is not None:
        with path.open("rb") as f:
            file_data = tomllib.load(f)

    preset_name = preset or file_data.get("preset", DEFAULT_PRESET)
    if preset_name not in PRESETS:
        raise ConfigError(f"unknown preset {preset_name!r}, choose from {sorted(PRESETS)}")

    data: dict[str, Any] = {"preset": preset_name}
    data = _merge(data, PRESETS[preset_name])
    data = _merge(data, {k: v for k, v in file_data.items() if k != "preset"})
    data = _merge(data, overrides or {})
    return _build(data)


def fingerprint(*parts: Any) -> str:
    """Stable hash of config sections, ignoring runtime-only fields."""

    def strip(obj: Any) -> Any:
        if hasattr(obj, "__dataclass_fields__"):
            return {
                f.name: strip(getattr(obj, f.name))
                for f in fields(obj)
                if not f.metadata.get("runtime")
            }
        return obj

    payload = json.dumps([strip(p) for p in parts], sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def as_dict(config: Config) -> dict[str, Any]:
    return asdict(config)
