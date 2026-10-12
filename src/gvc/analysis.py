"""Per-frame analysis on the extracted source frames: duplicates and scene cuts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import imagehash
from PIL import Image, ImageChops, ImageStat

from .config import DedupeConfig, ResetConfig
from .video import frame_name

THUMB_SIZE = (64, 64)


@dataclass
class Analysis:
    # dup_of[i] == i for a unique frame, else the index of the first frame of its run.
    dup_of: list[int]
    # Mean absolute pixel difference to the previous frame, 0..1 (0.0 for frame 0).
    diff: list[float]
    # True where both propagation chains start fresh.
    reset: list[bool]

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self)))

    @classmethod
    def load(cls, path: Path) -> Analysis:
        return cls(**json.loads(path.read_text()))

    def last_reset(self, index: int) -> int:
        while index > 0 and not self.reset[index]:
            index -= 1
        return index


def _thumb(image: Image.Image) -> Image.Image:
    return image.convert("RGB").resize(THUMB_SIZE, Image.Resampling.BILINEAR)


def mean_abs_diff(a: Image.Image, b: Image.Image) -> float:
    # RGB, not grayscale: a cut between equally bright but differently coloured shots must register.
    stat = ImageStat.Stat(ImageChops.difference(_thumb(a), _thumb(b)))
    return sum(stat.mean) / len(stat.mean) / 255.0


def analyze(frames_dir: Path, frame_count: int, dedupe: DedupeConfig, reset: ResetConfig) -> Analysis:
    dup_of: list[int] = []
    diff: list[float] = []
    hashes: list[imagehash.ImageHash] = []
    prev: Image.Image | None = None

    for i in range(frame_count):
        with Image.open(frames_dir / frame_name(i)) as img:
            img.load()
            hashes.append(imagehash.phash(img))
            diff.append(0.0 if prev is None else mean_abs_diff(prev, img))
            prev = img

        canonical = i
        if i > 0 and dedupe.max_distance >= 0:
            # Compare against the first frame of the previous run, not the previous
            # frame, so slow fades cannot chain into one endless duplicate run.
            candidate = dup_of[i - 1]
            if hashes[i] - hashes[candidate] <= dedupe.max_distance:
                canonical = candidate
        dup_of.append(canonical)

    resets = [
        i == 0
        or diff[i] >= reset.scene_threshold
        or (reset.every_n > 0 and i % reset.every_n == 0)
        for i in range(frame_count)
    ]
    return Analysis(dup_of=dup_of, diff=diff, reset=resets)
