"""Stage orchestration with a resumable work directory.

Layout of work/<video>/:
    frames/          extracted source frames 000001.png ...
    analysis.json    duplicates, frame differences, scene resets
    captions.jsonl   one caption per frame, appended as they finish
    gen/             generated frames 000001.png ...
    manifest.json    fingerprint + completion state per stage
    logs/            output of managed servers

Every stage's fingerprint includes the previous stage's fingerprint, so changing a
setting re-runs exactly the stages it affects. A stage whose fingerprint still
matches resumes from whatever output it already has.
"""

from __future__ import annotations

import difflib
import io
import json
import os
import shutil
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator

from PIL import Image
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn

from . import video
from .analysis import Analysis, analyze
from .captioner import Captioner, OllamaCaptioner
from .config import Config, fingerprint
from .generator import Generator, SdServerGenerator, build_prompt, generation_size, sd_server_cmd
from .service import host_port, managed_service
from .video import VideoInfo, frame_name


@dataclass
class Workdir:
    root: Path

    @property
    def frames(self) -> Path:
        return self.root / "frames"

    @property
    def gen(self) -> Path:
        return self.root / "gen"

    @property
    def analysis(self) -> Path:
        return self.root / "analysis.json"

    @property
    def captions(self) -> Path:
        return self.root / "captions.jsonl"

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.json"

    @property
    def logs(self) -> Path:
        return self.root / "logs"


class Manifest:
    def __init__(self, path: Path):
        self.path = path
        self.data: dict = json.loads(path.read_text()) if path.exists() else {}

    def save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.data, indent=2))
        os.replace(tmp, self.path)

    def matches(self, stage: str, fp: str) -> bool:
        return self.data.get(stage, {}).get("fp") == fp

    def complete(self, stage: str, fp: str) -> bool:
        entry = self.data.get(stage, {})
        return entry.get("fp") == fp and entry.get("complete", False)

    def begin(self, stage: str, fp: str) -> None:
        self.data[stage] = {"fp": fp, "complete": False}
        self.save()

    def finish(self, stage: str, **extra) -> None:
        self.data[stage].update(complete=True, **extra)
        self.save()


def _write_atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _reset_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def load_captions(path: Path) -> dict[int, dict]:
    captions: dict[int, dict] = {}
    if not path.exists():
        return captions
    for line in path.read_text().splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            # A line cut off by a crash; that frame is simply captioned again.
            continue
        captions[record["i"]] = record
    return captions


def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def _progress(console: Console) -> Progress:
    return Progress(
        TextColumn("[bold]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    )


class Pipeline:
    def __init__(
        self,
        config: Config,
        input_path: Path,
        workdir: Path,
        max_frames: int | None = None,
        console: Console | None = None,
    ):
        self.cfg = config
        self.input = input_path.resolve()
        self.wd = Workdir(workdir)
        self.wd.root.mkdir(parents=True, exist_ok=True)
        self.max_frames = max_frames
        self.console = console or Console()
        self.manifest = Manifest(self.wd.manifest)
        self.info: VideoInfo = video.probe(self.input)

    # -- fingerprints -----------------------------------------------------------

    def _fp_extract(self) -> str:
        stat = self.input.stat()
        return fingerprint(str(self.input), stat.st_size, stat.st_mtime_ns, self.max_frames)

    def _fp_analyze(self) -> str:
        return fingerprint(self._fp_extract(), self.cfg.dedupe, self.cfg.reset)

    def _fp_caption(self) -> str:
        return fingerprint(self._fp_analyze(), self.cfg.caption)

    def _fp_generate(self) -> str:
        return fingerprint(self._fp_caption(), self.cfg.generate, self.gen_size)

    @property
    def frame_count(self) -> int:
        return self.manifest.data["extract"]["frame_count"]

    @property
    def gen_size(self) -> tuple[int, int]:
        return generation_size(self.info.width, self.info.height, self.cfg.generate.size)

    # -- stages -------------------------------------------------------------------

    def extract(self) -> None:
        fp = self._fp_extract()
        if self.manifest.complete("extract", fp):
            return
        _reset_path(self.wd.frames)
        self.manifest.begin("extract", fp)
        self.console.print(f"Extracting frames from {self.input.name} ({self.info.frame_count} frames @ {self.info.fps} fps)")
        count = video.extract_frames(self.input, self.wd.frames, self.max_frames)
        self.manifest.finish("extract", frame_count=count)

    def analyze(self) -> Analysis:
        self.extract()
        fp = self._fp_analyze()
        if self.manifest.complete("analyze", fp):
            return Analysis.load(self.wd.analysis)
        self.manifest.begin("analyze", fp)
        result = analyze(self.wd.frames, self.frame_count, self.cfg.dedupe, self.cfg.reset)
        result.save(self.wd.analysis)
        unique = sum(1 for i, d in enumerate(result.dup_of) if i == d)
        self.console.print(f"Analyzed {self.frame_count} frames: {unique} unique, {sum(result.reset)} chain resets")
        self.manifest.finish("analyze")
        return result

    def caption(self, captioner: Captioner | None = None) -> None:
        analysis = self.analyze()
        fp = self._fp_caption()
        if self.manifest.complete("caption", fp):
            return
        if not self.manifest.matches("caption", fp):
            _reset_path(self.wd.captions)
            self.manifest.begin("caption", fp)
        if captioner is not None:
            self._caption_frames(analysis, captioner)
        else:
            with self._ollama() as ollama:
                self._caption_frames(analysis, ollama)
        self.manifest.finish("caption")

    def generate(self, generator: Generator | None = None) -> None:
        self.caption()
        analysis = Analysis.load(self.wd.analysis)
        fp = self._fp_generate()
        if self.manifest.complete("generate", fp):
            return
        if not self.manifest.matches("generate", fp):
            _reset_path(self.wd.gen)
            self.manifest.begin("generate", fp)
        self.wd.gen.mkdir(parents=True, exist_ok=True)
        if generator is not None:
            self._generate_frames(analysis, generator)
        else:
            with self._sd_server() as sd:
                self._generate_frames(analysis, sd)
        self.manifest.finish("generate")

    def assemble(self, output: Path) -> Path:
        self.generate()
        out = self.cfg.output
        size = (self.info.width, self.info.height) if out.scale_to_source else None
        audio = self.input if out.keep_audio and self.info.has_audio else None
        self.console.print(f"Assembling {self.frame_count} frames @ {self.info.fps} fps -> {output}")
        video.assemble(self.wd.gen, output, self.info.fps, self.frame_count, size, audio, out.codec, out.crf)
        written = video.count_frames(output)
        if written != self.frame_count:
            raise RuntimeError(f"frame count mismatch: expected {self.frame_count}, output has {written}")
        self.console.print(f"[green]Done:[/] {output} ({written} frames, same as input)")
        return output

    # -- services -------------------------------------------------------------------

    @contextmanager
    def _ollama(self) -> Iterator[OllamaCaptioner]:
        c = self.cfg.caption
        host, port = host_port(c.ollama_url, 11434)
        with managed_service(
            "ollama",
            c.ollama_url,
            ["ollama", "serve"],
            self.wd.logs / "ollama.log",
            c.manage_server,
            env={"OLLAMA_HOST": f"{host}:{port}"},
        ):
            captioner = OllamaCaptioner(c.ollama_url, c.model, c.max_words, c.temperature, c.seed, c.timeout)
            captioner.ensure_model()
            try:
                yield captioner
            finally:
                captioner.unload()

    @contextmanager
    def _sd_server(self) -> Iterator[SdServerGenerator]:
        g = self.cfg.generate
        with managed_service(
            "sd-server",
            f"{g.server_url.rstrip('/')}/sdapi/v1/sd-models",
            sd_server_cmd(g),
            self.wd.logs / "sd-server.log",
            g.manage_server,
        ):
            yield SdServerGenerator(g)

    # -- per-frame loops --------------------------------------------------------------

    def _frame(self, i: int) -> Path:
        return self.wd.frames / frame_name(i)

    def _gen(self, i: int) -> Path:
        return self.wd.gen / frame_name(i)

    def _loop(self, label: str, done: Callable[[int], bool]) -> Iterator[int]:
        """Yield the indices still to do, with a progress bar over all frames."""
        n = self.frame_count
        with _progress(self.console) as progress:
            task = progress.add_task(label, total=n)
            for i in range(n):
                if not done(i):
                    yield i
                progress.advance(task)

    def _caption_frames(self, analysis: Analysis, captioner: Captioner) -> None:
        c = self.cfg.caption
        captions = load_captions(self.wd.captions)
        with self.wd.captions.open("a") as out:
            for i in self._loop("Captioning", lambda i: i in captions):
                canonical = analysis.dup_of[i]
                if canonical != i:
                    record = {"i": i, "caption": captions[canonical]["caption"], "kind": "dup"}
                else:
                    record = self._caption_one(i, analysis, captions, captioner, c)
                captions[i] = record
                out.write(json.dumps(record) + "\n")
                out.flush()

    def _caption_one(self, i, analysis: Analysis, captions: dict[int, dict], captioner: Captioner, c) -> dict:
        use_context = c.context != "none" and not analysis.reset[i]
        if not use_context:
            return {"i": i, "caption": captioner.describe(self._frame(i), []), "kind": "fresh"}

        # Previous captions since the last chain reset, collapsing duplicate-frame repeats.
        start = analysis.last_reset(i)
        history: list[str] = []
        for j in range(start, i):
            text = captions[j]["caption"]
            if not history or history[-1] != text:
                history.append(text)
        history = history[-c.window:]
        prev_image = self._frame(i - 1) if c.context == "text+image" else None

        text = captioner.describe(self._frame(i), history, prev_image)
        if similarity(text, history[-1]) >= c.stale_similarity and analysis.diff[i] >= c.stale_diff:
            # The model just echoed the previous caption although the picture changed.
            return {"i": i, "caption": captioner.describe(self._frame(i), []), "kind": "stale-retry"}
        return {"i": i, "caption": text, "kind": "context"}

    def _generate_frames(self, analysis: Analysis, generator: Generator) -> None:
        g = self.cfg.generate
        w, h = self.gen_size
        captions = load_captions(self.wd.captions)
        prompts = [build_prompt(captions[i]["caption"], g.style) for i in range(self.frame_count)]

        # txt2img with a fixed seed is deterministic, so an identical prompt can reuse
        # an earlier image. Every other mode depends on the previous frame.
        first_with_prompt: dict[str, int] = {}

        def done(i: int) -> bool:
            if self._gen(i).exists():
                first_with_prompt.setdefault(prompts[i], i)
                return True
            return False

        for i in self._loop("Generating", done):
            prompt = prompts[i]
            fresh = analysis.reset[i]
            if g.init == "none" or (fresh and g.init in ("chain", "chain-ref")):
                if g.init == "none" and prompt in first_with_prompt:
                    data = self._gen(first_with_prompt[prompt]).read_bytes()
                else:
                    data = generator.generate(prompt, w, h, g.seed)
            elif g.init == "chain":
                data = generator.generate(prompt, w, h, g.seed, init=self._gen(i - 1).read_bytes(), strength=g.strength)
            elif g.init == "chain-ref":
                data = generator.generate(prompt, w, h, g.seed, reference=self._gen(i - 1).read_bytes())
            else:  # source
                data = generator.generate(prompt, w, h, g.seed, init=self._source_init(i, w, h), strength=g.strength)
            _write_atomic(self._gen(i), data)
            first_with_prompt.setdefault(prompt, i)

    def _source_init(self, i: int, w: int, h: int) -> bytes:
        with Image.open(self._frame(i)) as img:
            buf = io.BytesIO()
            img.convert("RGB").resize((w, h), Image.Resampling.LANCZOS).save(buf, format="PNG")
            return buf.getvalue()

    def run(self, output: Path) -> Path:
        return self.assemble(output)
