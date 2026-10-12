"""Command line interface."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Optional

import typer
from rich.console import Console
from rich.table import Table

from . import models, video
from .config import CAPTION_CONTEXTS, GENERATE_INITS, PRESETS, REF_METHODS, Config, ConfigError, load_config
from .pipeline import Pipeline
from .service import ServiceError

app = typer.Typer(no_args_is_help=True, help="Generative video conversion: frames -> captions -> new frames -> video.")
console = Console()

DEFAULT_CONFIG = Path("gvc.toml")

Input = Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="Source video.")]
ConfigOpt = Annotated[Optional[Path], typer.Option("--config", "-c", help="Config file (default: ./gvc.toml if present).")]
PresetOpt = Annotated[Optional[str], typer.Option("--preset", "-p", help=f"One of: {', '.join(PRESETS)}.")]
WorkdirOpt = Annotated[Optional[Path], typer.Option("--workdir", "-w", help="Work directory (default: work/<video name>).")]
MaxFramesOpt = Annotated[Optional[int], typer.Option("--max-frames", "-n", help="Only process the first N frames (for tests).")]
ContextOpt = Annotated[Optional[str], typer.Option("--caption-context", help=f"Caption propagation: {' | '.join(CAPTION_CONTEXTS)}.")]
WindowOpt = Annotated[Optional[int], typer.Option("--caption-window", help="Previous captions passed as context.")]
CaptionModelOpt = Annotated[Optional[str], typer.Option("--caption-model", help="Ollama vision model.")]
InitOpt = Annotated[Optional[str], typer.Option("--init", help=f"Image propagation: {' | '.join(GENERATE_INITS)}.")]
StrengthOpt = Annotated[Optional[float], typer.Option("--strength", help="img2img denoising strength (chain, source).")]
RefMethodOpt = Annotated[Optional[str], typer.Option("--ref-method", help=f"chain-ref method: {' | '.join(REF_METHODS)}.")]
RefStrengthOpt = Annotated[Optional[float], typer.Option("--ref-strength", help="Reference strength (chain-ref).")]
SeedOpt = Annotated[Optional[int], typer.Option("--seed", help="Image generation seed.")]
StyleOpt = Annotated[Optional[str], typer.Option("--style", help="Text appended to every prompt.")]


def _config(
    config: Path | None,
    preset: str | None,
    caption_context: str | None = None,
    caption_window: int | None = None,
    caption_model: str | None = None,
    init: str | None = None,
    strength: float | None = None,
    ref_method: str | None = None,
    ref_strength: float | None = None,
    seed: int | None = None,
    style: str | None = None,
) -> Config:
    overrides: dict[str, dict[str, Any]] = {
        "caption": {"context": caption_context, "window": caption_window, "model": caption_model},
        "generate": {
            "init": init,
            "strength": strength,
            "ref_method": ref_method,
            "ref_strength": ref_strength,
            "seed": seed,
            "style": style,
        },
    }
    overrides = {s: {k: v for k, v in values.items() if v is not None} for s, values in overrides.items()}
    if config is None and DEFAULT_CONFIG.exists():
        config = DEFAULT_CONFIG
    try:
        return load_config(config, preset, overrides)
    except ConfigError as e:
        raise typer.BadParameter(str(e)) from e


def _pipeline(input: Path, cfg: Config, workdir: Path | None, max_frames: int | None) -> Pipeline:
    workdir = workdir or Path("work") / input.stem
    p = Pipeline(cfg, input, workdir, max_frames=max_frames, console=console)
    console.print(
        f"preset [bold]{cfg.preset}[/]: caption.context=[bold]{cfg.caption.context}[/] "
        f"generate.init=[bold]{cfg.generate.init}[/]  workdir {workdir}"
    )
    return p


def _guard(fn):
    try:
        return fn()
    except (ServiceError, video.FFmpegError, RuntimeError) as e:
        console.print(f"[red]error:[/] {e}")
        raise typer.Exit(1)


@app.command()
def probe(input: Input) -> None:
    """Show what gvc sees in a video."""
    info = _guard(lambda: video.probe(input))
    for key, value in info.to_dict().items():
        console.print(f"{key:>12}: {value}")


@app.command()
def extract(
    input: Input,
    config: ConfigOpt = None,
    preset: PresetOpt = None,
    workdir: WorkdirOpt = None,
    max_frames: MaxFramesOpt = None,
) -> None:
    """Extract and analyze frames."""
    p = _pipeline(input, _config(config, preset), workdir, max_frames)
    _guard(p.analyze)


@app.command()
def caption(
    input: Input,
    config: ConfigOpt = None,
    preset: PresetOpt = None,
    workdir: WorkdirOpt = None,
    max_frames: MaxFramesOpt = None,
    caption_context: ContextOpt = None,
    caption_window: WindowOpt = None,
    caption_model: CaptionModelOpt = None,
) -> None:
    """Run the pipeline up to and including captioning."""
    cfg = _config(config, preset, caption_context, caption_window, caption_model)
    p = _pipeline(input, cfg, workdir, max_frames)
    _guard(p.caption)


@app.command()
def generate(
    input: Input,
    config: ConfigOpt = None,
    preset: PresetOpt = None,
    workdir: WorkdirOpt = None,
    max_frames: MaxFramesOpt = None,
    caption_context: ContextOpt = None,
    caption_window: WindowOpt = None,
    caption_model: CaptionModelOpt = None,
    init: InitOpt = None,
    strength: StrengthOpt = None,
    ref_method: RefMethodOpt = None,
    ref_strength: RefStrengthOpt = None,
    seed: SeedOpt = None,
    style: StyleOpt = None,
) -> None:
    """Run the pipeline up to and including image generation."""
    cfg = _config(config, preset, caption_context, caption_window, caption_model, init, strength, ref_method, ref_strength, seed, style)
    p = _pipeline(input, cfg, workdir, max_frames)
    _guard(p.generate)


@app.command()
def run(
    input: Input,
    output: Annotated[Optional[Path], typer.Option("--output", "-o", help="Output video (default: <name>_gvc.mp4).")] = None,
    config: ConfigOpt = None,
    preset: PresetOpt = None,
    workdir: WorkdirOpt = None,
    max_frames: MaxFramesOpt = None,
    caption_context: ContextOpt = None,
    caption_window: WindowOpt = None,
    caption_model: CaptionModelOpt = None,
    init: InitOpt = None,
    strength: StrengthOpt = None,
    ref_method: RefMethodOpt = None,
    ref_strength: RefStrengthOpt = None,
    seed: SeedOpt = None,
    style: StyleOpt = None,
) -> None:
    """Run the whole pipeline and write the output video."""
    cfg = _config(config, preset, caption_context, caption_window, caption_model, init, strength, ref_method, ref_strength, seed, style)
    p = _pipeline(input, cfg, workdir, max_frames)
    _guard(lambda: p.run(output or Path(f"{input.stem}_gvc.mp4")))


@app.command()
def presets() -> None:
    """List the presets."""
    table = Table("preset", "caption.context", "generate.init")
    for name, values in PRESETS.items():
        table.add_row(name, values["caption"]["context"], values["generate"]["init"])
    console.print(table)


@app.command("models")
def models_cmd(
    config: ConfigOpt = None,
    preset: PresetOpt = None,
    init: InitOpt = None,
) -> None:
    """Download the image model files the config needs (caption models are pulled by Ollama on first use)."""
    cfg = _config(config, preset, init=init)
    unknown = models.pull(cfg.generate, console)
    for path in unknown:
        console.print(f"[yellow]missing, no known download:[/] {path}")
    if unknown:
        raise typer.Exit(1)
