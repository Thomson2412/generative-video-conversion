"""Download the image model files the current config needs."""

from __future__ import annotations

from pathlib import Path

import httpx
from rich.console import Console
from rich.progress import BarColumn, DownloadColumn, Progress, TextColumn, TransferSpeedColumn

from .config import GenerateConfig

HF = "https://huggingface.co"

# Known downloads, keyed by file name so configured paths can live anywhere.
KNOWN_FILES = {
    "sd_xl_turbo_1.0_fp16.safetensors": f"{HF}/stabilityai/sdxl-turbo/resolve/main/sd_xl_turbo_1.0_fp16.safetensors",
    "sd_turbo.safetensors": f"{HF}/stabilityai/sd-turbo/resolve/main/sd_turbo.safetensors",
    "sdxl_vae-fp16-fix.safetensors": f"{HF}/madebyollin/sdxl-vae-fp16-fix/resolve/main/sdxl_vae.safetensors",
    "ip-adapter_sdxl_vit-h.safetensors": f"{HF}/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter_sdxl_vit-h.safetensors",
    "ip-adapter-plus_sdxl_vit-h.safetensors": f"{HF}/h94/IP-Adapter/resolve/main/sdxl_models/ip-adapter-plus_sdxl_vit-h.safetensors",
    "ip-adapter_sd15.safetensors": f"{HF}/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15.safetensors",
    "clip_vision_h.safetensors": f"{HF}/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors",
}


def required_files(cfg: GenerateConfig) -> list[Path]:
    paths = [cfg.model]
    if cfg.vae:
        paths.append(cfg.vae)
    if cfg.init == "chain-ref" and cfg.ref_method == "ip-adapter":
        paths += [cfg.ip_adapter, cfg.clip_vision]
    return [Path(p) for p in paths]


def download(url: str, dest: Path, console: Console) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    with httpx.stream("GET", url, headers=headers, follow_redirects=True, timeout=60.0) as r:
        r.raise_for_status()
        if r.status_code != 206:
            offset = 0  # server ignored the range; start over
        total = int(r.headers.get("content-length", 0)) + offset
        with (
            partial.open("ab" if offset else "wb") as f,
            Progress(
                TextColumn(f"[bold]{dest.name}"), BarColumn(), DownloadColumn(), TransferSpeedColumn(), console=console
            ) as progress,
        ):
            task = progress.add_task("download", total=total or None, completed=offset)
            for chunk in r.iter_bytes(1 << 20):
                f.write(chunk)
                progress.advance(task, len(chunk))
    partial.rename(dest)


def pull(cfg: GenerateConfig, console: Console) -> list[Path]:
    """Download missing files. Returns the ones that are missing and unknown."""
    unknown = []
    for path in required_files(cfg):
        if path.exists():
            console.print(f"[green]present[/] {path}")
        elif path.name in KNOWN_FILES:
            download(KNOWN_FILES[path.name], path, console)
        else:
            unknown.append(path)
    return unknown
