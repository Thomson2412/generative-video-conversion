from dataclasses import replace
from pathlib import Path

import pytest
from rich.console import Console

from gvc import video
from gvc.analysis import Analysis
from gvc.config import load_config
from gvc.pipeline import Pipeline, load_captions
from gvc.video import frame_name

from .conftest import MOVING_FRAMES, STATIC_FRAMES, FakeCaptioner, FakeGenerator

TOTAL = STATIC_FRAMES + MOVING_FRAMES


def make(sample_video, tmp_path, preset="full", **overrides) -> Pipeline:
    cfg = load_config(preset=preset, overrides=overrides)
    return Pipeline(cfg, sample_video, tmp_path / "work", console=Console(quiet=True))


def run(p: Pipeline, captioner, generator, tmp_path) -> Path:
    p.caption(captioner)
    p.generate(generator)
    return p.assemble(tmp_path / "out.mp4")


def analysis(p: Pipeline) -> Analysis:
    return Analysis.load(p.wd.analysis)


def gen_bytes(p: Pipeline, i: int) -> bytes:
    return (p.wd.gen / frame_name(i)).read_bytes()


def test_literal(sample_video, tmp_path, fake_captioner, fake_generator):
    p = make(sample_video, tmp_path, "literal")
    out = run(p, fake_captioner, fake_generator, tmp_path)
    a = analysis(p)
    unique = [i for i in range(TOTAL) if a.dup_of[i] == i]

    assert video.count_frames(out) == TOTAL
    assert sorted(c["image"] for c in fake_captioner.calls) == [frame_name(i) for i in unique]
    assert all(c["history"] == [] for c in fake_captioner.calls)
    # Duplicate frames share caption and seed, so their image is reused, not regenerated.
    assert len(fake_generator.calls) == len(unique)
    assert all(c["init"] is None and c["reference"] is None for c in fake_generator.calls)
    assert gen_bytes(p, STATIC_FRAMES - 1) == gen_bytes(p, 0)


def test_full_propagates_text_and_images(sample_video, tmp_path, fake_captioner, fake_generator):
    p = make(sample_video, tmp_path, "full", generate={"strength": 0.4})
    run(p, fake_captioner, fake_generator, tmp_path)
    a = analysis(p)

    by_image = {c["image"]: c for c in fake_captioner.calls}
    assert by_image[frame_name(0)]["history"] == []
    assert by_image[frame_name(STATIC_FRAMES)]["history"] == []  # scene cut resets the text chain
    for i in range(STATIC_FRAMES + 1, TOTAL):
        if a.dup_of[i] == i:
            # Only captions since the cut, never the red frames before it.
            history = by_image[frame_name(i)]["history"]
            assert history and all(int(h.split()[-1]) > STATIC_FRAMES for h in history)

    assert len(fake_generator.calls) == TOTAL  # the chain advances even on duplicate frames
    for i, call in enumerate(fake_generator.calls):
        if a.reset[i]:
            assert call["init"] is None
        else:
            assert call["init"] == gen_bytes(p, i - 1)
            assert call["strength"] == 0.4


def test_text_history_is_previous_caption(sample_video, tmp_path, fake_captioner):
    p = make(sample_video, tmp_path, "stable-text", dedupe={"max_distance": -1})
    p.caption(fake_captioner)
    for i, call in enumerate(fake_captioner.calls):
        if i in (0, STATIC_FRAMES):
            assert call["history"] == []
        else:
            assert call["history"] == [f"a picture of {i:06d}"]  # frame_name(i - 1) stem
            assert call["prev_image"] is None


def test_text_image_context_and_window(sample_video, tmp_path, fake_captioner):
    p = make(sample_video, tmp_path, "full", caption={"context": "text+image", "window": 3}, dedupe={"max_distance": -1})
    p.caption(fake_captioner)
    last = fake_captioner.calls[-1]
    assert last["prev_image"].name == frame_name(TOTAL - 2)
    assert last["history"] == [f"a picture of {s:06d}" for s in range(TOTAL - 3, TOTAL)]


def test_chain_ref_passes_previous_frame_as_reference(sample_video, tmp_path, fake_captioner, fake_generator):
    p = make(sample_video, tmp_path, "full-ref")
    run(p, fake_captioner, fake_generator, tmp_path)
    a = analysis(p)
    for i, call in enumerate(fake_generator.calls):
        assert call["init"] is None
        if a.reset[i]:
            assert call["reference"] is None
        else:
            assert call["reference"] == gen_bytes(p, i - 1)


def test_source_uses_original_frame(sample_video, tmp_path, fake_captioner, fake_generator):
    p = make(sample_video, tmp_path, "guided", generate={"size": 128})
    run(p, fake_captioner, fake_generator, tmp_path)
    assert all(c["init"] is not None and c["size"] == p.gen_size for c in fake_generator.calls)
    assert p.gen_size == (192, 128)


def test_caption_resume_after_crash(sample_video, tmp_path):
    p = make(sample_video, tmp_path, "full")
    crashing = FakeCaptioner(fail_after=3)
    with pytest.raises(RuntimeError):
        p.caption(crashing)
    assert not p.manifest.complete("caption", p._fp_caption())

    resumed = FakeCaptioner()
    p = make(sample_video, tmp_path, "full")
    p.caption(resumed)
    a = analysis(p)
    unique = [frame_name(i) for i in range(TOTAL) if a.dup_of[i] == i]
    assert [c["image"] for c in crashing.calls + resumed.calls] == unique
    assert sorted(load_captions(p.wd.captions)) == list(range(TOTAL))


def test_changing_generate_settings_keeps_captions(sample_video, tmp_path, fake_captioner, fake_generator):
    p = make(sample_video, tmp_path, "full")
    p.caption(fake_captioner)
    p.generate(fake_generator)
    calls = len(fake_captioner.calls)

    p = make(sample_video, tmp_path, "full", generate={"strength": 0.8})
    second = FakeGenerator()
    p.caption(FakeCaptioner(fail_after=0))  # would raise if captioning re-ran
    p.generate(second)
    assert len(second.calls) == TOTAL
    assert len(fake_captioner.calls) == calls

    # Changing the caption settings re-runs captioning (and therefore generation).
    p = make(sample_video, tmp_path, "literal", generate={"init": "chain", "strength": 0.8})
    recaption = FakeCaptioner()
    p.caption(recaption)
    assert recaption.calls


def test_stale_caption_is_redone_without_context(sample_video, tmp_path):
    class Echo(FakeCaptioner):
        def describe(self, image, history, prev_image=None):
            super().describe(image, history, prev_image)
            return history[-1] if history else f"fresh {image.stem}"

    p = make(sample_video, tmp_path, "stable-text", caption={"stale_diff": 0.0}, dedupe={"max_distance": -1})
    echo = Echo()
    p.caption(echo)
    records = load_captions(p.wd.captions)
    assert records[1]["kind"] == "stale-retry"
    assert records[1]["caption"] == "fresh 000002"


def test_parallel_captions_without_context(sample_video, tmp_path):
    p = make(sample_video, tmp_path, "literal", caption={"parallel": 4})
    captioner = FakeCaptioner()
    p.caption(captioner)
    a = analysis(p)
    unique = sorted(frame_name(i) for i in range(TOTAL) if a.dup_of[i] == i)
    assert sorted(c["image"] for c in captioner.calls) == unique
    records = load_captions(p.wd.captions)
    assert sorted(records) == list(range(TOTAL))
    assert all(records[i]["caption"] == records[a.dup_of[i]]["caption"] for i in range(TOTAL))


def test_parallel_caption_failure_keeps_finished_captions(sample_video, tmp_path):
    p = make(sample_video, tmp_path, "literal", caption={"parallel": 2}, dedupe={"max_distance": -1})
    with pytest.raises(RuntimeError):
        p.caption(FakeCaptioner(fail_after=5))
    done = load_captions(p.wd.captions)
    assert 0 < len(done) < TOTAL

    resumed = FakeCaptioner()
    p.caption(resumed)
    assert len(resumed.calls) == TOTAL - len(done)


def test_bench_sample_frames():
    from gvc.bench import sample_frames

    assert sample_frames([0, 0, 2, 3, 3, 5], 10) == [0, 2, 3, 5]
    assert sample_frames(list(range(100)), 4) == [0, 25, 50, 75]
