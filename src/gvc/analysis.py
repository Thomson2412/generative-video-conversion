"""Per-frame analysis on the extracted source frames: duplicates, scene cuts and fades."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import imagehash
from PIL import Image, ImageChops, ImageStat

from .config import DedupeConfig, FadeConfig, ResetConfig
from .video import frame_name

THUMB_SIZE = (64, 64)
# Fade levels at or above this count as full brightness.
FULL_LEVEL = 0.97


@dataclass
class Analysis:
    # dup_of[i] == i for a unique frame, else the index of the first frame of its run.
    dup_of: list[int]
    # Mean absolute pixel difference to the previous frame, 0..1 (0.0 for frame 0).
    diff: list[float]
    # True where both propagation chains start fresh.
    reset: list[bool]
    # Mean brightness, 0..1.
    luma: list[float]
    # Source frame is (near) black.
    black: list[bool]
    # Brightness relative to the shot during a fade to or from black: 1.0 outside fades.
    fade: list[float]

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


def mean_luma(image: Image.Image) -> float:
    return ImageStat.Stat(_thumb(image).convert("L")).mean[0] / 255.0


def fade_levels(luma: list[float], black: list[bool], cut: list[bool], min_rise: float) -> list[float]:
    """Brightness relative to full for the fade ramps next to each run of black frames.

    A ramp starts at the frame next to black and continues away from it while the
    brightness keeps rising by at least min_rise per frame; its last frame is taken
    as full brightness. cut[i] means frame i starts a new shot.
    """
    n = len(luma)
    level = [1.0] * n

    def ramp(start: int, step: int) -> None:
        idx = [start]
        while True:
            nxt = idx[-1] + step
            if not 0 <= nxt < n or black[nxt] or cut[max(nxt, idx[-1])]:
                break
            if luma[nxt] < luma[idx[-1]] * (1 + min_rise):
                break
            idx.append(nxt)
        full = luma[idx[-1]]
        for j in idx:
            lv = luma[j] / full
            level[j] = 1.0 if lv >= FULL_LEVEL else round(lv, 4)

    for i in range(n):
        if black[i]:
            continue
        if i > 0 and black[i - 1]:
            ramp(i, 1)  # fade in
        if i + 1 < n and black[i + 1]:
            ramp(i, -1)  # fade out
    return level


def analyze(
    frames_dir: Path, frame_count: int, dedupe: DedupeConfig, reset: ResetConfig, fade: FadeConfig
) -> Analysis:
    dup_of: list[int] = []
    diff: list[float] = []
    luma: list[float] = []
    hashes: list[imagehash.ImageHash] = []
    prev: Image.Image | None = None

    for i in range(frame_count):
        with Image.open(frames_dir / frame_name(i)) as img:
            img.load()
            hashes.append(imagehash.phash(img))
            diff.append(0.0 if prev is None else mean_abs_diff(prev, img))
            luma.append(mean_luma(img))
            prev = img

        canonical = i
        if i > 0 and dedupe.max_distance >= 0:
            # Compare against the first frame of the previous run, not the previous
            # frame, so slow fades cannot chain into one endless duplicate run.
            candidate = dup_of[i - 1]
            if hashes[i] - hashes[candidate] <= dedupe.max_distance:
                canonical = candidate
        dup_of.append(canonical)

    cuts = [i == 0 or diff[i] >= reset.scene_threshold for i in range(frame_count)]
    if fade.enabled:
        black = [v <= fade.black_level for v in luma]
        levels = fade_levels(luma, black, cuts, fade.min_rise)
    else:
        black, levels = [False] * frame_count, [1.0] * frame_count
    resets = [
        cuts[i]
        or (reset.every_n > 0 and i % reset.every_n == 0)
        # Nothing to carry over out of black.
        or (i > 0 and black[i - 1] and not black[i])
        for i in range(frame_count)
    ]
    return Analysis(dup_of=dup_of, diff=diff, reset=resets, luma=luma, black=black, fade=levels)
