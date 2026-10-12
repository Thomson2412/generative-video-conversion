"""Compare caption backends and image sizes on frames of a real video."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from pathlib import Path

from .captioner import captioner_service
from .pipeline import Pipeline
from .video import frame_name


@dataclass
class BenchResult:
    backend: str
    image_size: int
    sequential: float  # seconds per frame, one request at a time
    parallel: float | None  # seconds per frame with `workers` concurrent requests
    workers: int
    sample: str


def sample_frames(dup_of: list[int], count: int) -> list[int]:
    """Up to `count` unique frames spread evenly over the video."""
    unique = [i for i, d in enumerate(dup_of) if i == d]
    if len(unique) <= count:
        return unique
    step = len(unique) / count
    return [unique[int(k * step)] for k in range(count)]


def _timed(fn) -> float:
    start = time.perf_counter()
    fn()
    return time.perf_counter() - start


def bench(
    pipeline: Pipeline,
    backends: list[str],
    image_sizes: list[int],
    frames: int,
    workers: int,
) -> list[BenchResult]:
    analysis = pipeline.analyze()
    # Every measurement gets its own frames: servers cache the processed prompt, so
    # re-sending a frame would time a cache hit instead of real work.
    rounds = len(image_sizes) * (2 if workers > 1 else 1)
    picked = [pipeline.wd.frames / frame_name(i) for i in sample_frames(analysis.dup_of, frames * rounds + 1)]
    warmup, pool_paths = picked[0], picked[1:]
    batches: list[list[Path]] = [pool_paths[k::rounds] for k in range(rounds)]
    if any(not b for b in batches):
        raise RuntimeError(f"not enough unique frames for {rounds} measurements, lower --frames")
    results = []
    for backend in backends:
        cfg = replace(pipeline.cfg.caption, backend=backend, parallel=workers)
        pipeline.console.print(f"[bold]{backend}[/]: starting, the first start may download the model")
        with captioner_service(cfg, pipeline.wd.logs) as captioner:
            # Warm-up: loads the model into VRAM, which would otherwise count as caption time.
            captioner.describe(warmup, [])
            batch = iter(batches)
            for size in image_sizes:
                captioner.image_size = size
                captions: list[str] = []
                seq_paths = next(batch)
                seq = _timed(lambda: captions.extend(captioner.describe(p, []) for p in seq_paths)) / len(seq_paths)
                par = None
                if workers > 1:
                    par_paths = next(batch)
                    with ThreadPoolExecutor(workers) as pool:
                        par = _timed(lambda: list(pool.map(lambda p: captioner.describe(p, []), par_paths))) / len(par_paths)
                results.append(BenchResult(backend, size, seq, par, workers, captions[0]))
                pipeline.console.print(f"  image_size={size or 'original'}: {seq:.2f} s/frame sequential")
    return results
