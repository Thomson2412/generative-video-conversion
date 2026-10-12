"""ffprobe / ffmpeg wrappers."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

FRAME_PATTERN = "%06d.png"


def frame_name(index: int) -> str:
    """File name of a 0-based frame index (ffmpeg numbers from 1)."""
    return f"{index + 1:06d}.png"


class FFmpegError(RuntimeError):
    pass


def _run(cmd: list[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise FFmpegError(f"{cmd[0]} failed ({result.returncode}): {result.stderr.strip()}")
    return result.stdout


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    width: int
    height: int
    fps: Fraction
    frame_count: int
    duration: float
    has_audio: bool

    def to_dict(self) -> dict:
        return {
            "path": str(self.path),
            "width": self.width,
            "height": self.height,
            "fps": str(self.fps),
            "frame_count": self.frame_count,
            "duration": self.duration,
            "has_audio": self.has_audio,
        }


def _fraction(value: str | None) -> Fraction | None:
    if not value or value in ("0/0", "0"):
        return None
    return Fraction(value)


def count_frames(path: Path) -> int:
    """Exact number of video frames (counts packets, no decoding)."""
    out = _run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
        "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(path),
    ])
    return int(out.strip().split(",")[0])


def probe(path: Path) -> VideoInfo:
    out = _run([
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_streams", "-show_format", str(path),
    ])
    data = json.loads(out)
    video = next((s for s in data["streams"] if s["codec_type"] == "video"), None)
    if video is None:
        raise FFmpegError(f"{path} has no video stream")
    # avg_frame_rate is frames/duration, which keeps the total duration right even
    # for variable frame rate input; r_frame_rate is the fallback.
    fps = _fraction(video.get("avg_frame_rate")) or _fraction(video.get("r_frame_rate"))
    if fps is None:
        raise FFmpegError(f"cannot determine frame rate of {path}")
    duration = float(video.get("duration") or data["format"].get("duration") or 0.0)
    return VideoInfo(
        path=path,
        width=int(video["width"]),
        height=int(video["height"]),
        fps=fps,
        frame_count=count_frames(path),
        duration=duration,
        has_audio=any(s["codec_type"] == "audio" for s in data["streams"]),
    )


def extract_frames(path: Path, out_dir: Path, max_frames: int | None = None) -> int:
    """Write every decoded frame as a PNG, without dropping or duplicating frames."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(path), "-map", "0:v:0", "-fps_mode", "passthrough"]
    if max_frames:
        cmd += ["-frames:v", str(max_frames)]
    cmd.append(str(out_dir / FRAME_PATTERN))
    _run(cmd)
    return len(list(out_dir.glob("*.png")))


def assemble(
    frames_dir: Path,
    out_path: Path,
    fps: Fraction,
    frame_count: int,
    size: tuple[int, int] | None,
    audio_from: Path | None,
    codec: str = "libx264",
    crf: int = 18,
) -> None:
    """Encode numbered PNGs into a video at exactly `fps`, optionally muxing source audio."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-v", "error", "-y",
        "-framerate", str(fps), "-start_number", "1", "-i", str(frames_dir / FRAME_PATTERN),
    ]
    if audio_from is not None:
        cmd += ["-i", str(audio_from), "-map", "0:v:0", "-map", "1:a:0?", "-c:a", "aac", "-b:a", "192k",
                # Trim audio to the video length; -shortest could cut video frames instead.
                "-t", f"{float(frame_count / fps):.6f}"]
    filters = []
    if size is not None:
        # Even dimensions keep yuv420p encoders happy.
        w, h = size[0] - size[0] % 2, size[1] - size[1] % 2
        filters.append(f"scale={w}:{h}:flags=lanczos")
    filters.append("format=yuv420p")
    cmd += [
        "-vf", ",".join(filters),
        "-frames:v", str(frame_count),
        "-c:v", codec, "-crf", str(crf),
        str(out_path),
    ]
    _run(cmd)
