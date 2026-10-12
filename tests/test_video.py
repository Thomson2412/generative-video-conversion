from fractions import Fraction

from gvc import video
from gvc.analysis import analyze
from gvc.config import DedupeConfig, ResetConfig

from .conftest import MOVING_FRAMES, STATIC_FRAMES, png

TOTAL = STATIC_FRAMES + MOVING_FRAMES


def test_probe(sample_video):
    info = video.probe(sample_video)
    assert (info.width, info.height) == (160, 90)
    assert info.fps == Fraction(30000, 1001)
    assert info.frame_count == TOTAL
    assert info.has_audio


def test_extract_and_assemble_keep_frame_count_and_rate(sample_video, tmp_path):
    frames = tmp_path / "frames"
    assert video.extract_frames(sample_video, frames) == TOTAL

    # Different size frames, as produced by the generator.
    gen = tmp_path / "gen"
    gen.mkdir()
    for i in range(TOTAL):
        (gen / video.frame_name(i)).write_bytes(png((i * 15, 0, 0), (192, 128)))

    out = tmp_path / "out.mp4"
    video.assemble(gen, out, Fraction(30000, 1001), TOTAL, (160, 90), sample_video)
    info = video.probe(out)
    assert info.frame_count == TOTAL
    assert info.fps == Fraction(30000, 1001)
    assert (info.width, info.height) == (160, 90)
    assert info.has_audio


def test_extract_max_frames(sample_video, tmp_path):
    assert video.extract_frames(sample_video, tmp_path, max_frames=4) == 4


def test_analysis_finds_duplicates_and_cut(sample_video, tmp_path):
    video.extract_frames(sample_video, tmp_path)
    a = analyze(tmp_path, TOTAL, DedupeConfig(), ResetConfig())
    assert a.dup_of[:STATIC_FRAMES] == [0] * STATIC_FRAMES
    assert a.dup_of[STATIC_FRAMES] == STATIC_FRAMES
    assert [i for i, r in enumerate(a.reset) if r] == [0, STATIC_FRAMES]
    assert a.last_reset(TOTAL - 1) == STATIC_FRAMES


def test_analysis_every_n_and_dedupe_off(sample_video, tmp_path):
    video.extract_frames(sample_video, tmp_path)
    a = analyze(tmp_path, TOTAL, DedupeConfig(max_distance=-1), ResetConfig(scene_threshold=2.0, every_n=5))
    assert a.dup_of == list(range(TOTAL))
    assert [i for i, r in enumerate(a.reset) if r] == [0, 5, 10]
