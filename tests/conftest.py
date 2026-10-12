from __future__ import annotations

import io
import subprocess
from pathlib import Path

import pytest
from PIL import Image

FPS = "30000/1001"
STATIC_FRAMES = 6  # identical red frames -> one dedupe run
MOVING_FRAMES = 8  # moving test pattern after a hard cut


@pytest.fixture(scope="session")
def sample_video(tmp_path_factory) -> Path:
    """14 frames at 29.97 fps: 6 identical red frames, a hard cut, 8 moving frames, plus audio."""
    path = tmp_path_factory.mktemp("video") / "sample.mp4"
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y",
            "-f", "lavfi", "-i", f"color=c=red:s=160x90:r={FPS}",
            "-f", "lavfi", "-i", f"testsrc2=s=160x90:r={FPS}",
            "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100",
            "-filter_complex",
            f"[0:v]trim=end_frame={STATIC_FRAMES}[a];[1:v]trim=end_frame={MOVING_FRAMES}[b];"
            "[a][b]concat=n=2:v=1:a=0,setpts=N/(30000/1001)/TB[v]",
            "-map", "[v]", "-map", "2:a", "-shortest",
            "-c:v", "libx264", "-qp", "0", "-pix_fmt", "yuv444p", "-r", FPS,
            str(path),
        ],
        check=True,
    )
    return path


def png(color: tuple[int, int, int], size: tuple[int, int]) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


class FakeCaptioner:
    def __init__(self, fail_after: int | None = None):
        self.calls: list[dict] = []
        self.fail_after = fail_after

    def describe(self, image: Path, history: list[str], prev_image: Path | None = None) -> str:
        if self.fail_after is not None and len(self.calls) >= self.fail_after:
            raise RuntimeError("simulated crash")
        self.calls.append({"image": image.name, "history": list(history), "prev_image": prev_image})
        return f"a picture of {image.stem}"


class FakeGenerator:
    def __init__(self):
        self.calls: list[dict] = []

    def generate(self, prompt, width, height, seed, init=None, strength=0.0, reference=None) -> bytes:
        self.calls.append(
            {"prompt": prompt, "size": (width, height), "seed": seed, "init": init, "strength": strength, "reference": reference}
        )
        shade = len(self.calls) * 10 % 256
        return png((shade, 0, 255 - shade), (width, height))


@pytest.fixture
def fake_captioner():
    return FakeCaptioner()


@pytest.fixture
def fake_generator():
    return FakeGenerator()
