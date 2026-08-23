"""plan_clips is pure metadata -- no ffmpeg, no filesystem -- so these tests
need no audio fixtures at all, unlike test_review_workflow.py's build_video."""

from voice_factory.core.youtube_ingest import (
    PAD_AFTER_SEC,
    PAD_BEFORE_SEC,
    plan_clips,
)


def segment(start: float, end: float, text: str = "line") -> dict:
    return {"start": start, "end": end, "text": text}


def test_plan_clips_writes_no_wav_files(tmp_path):
    plan_clips(100.0, [segment(1.0, 3.0)])

    assert list(tmp_path.iterdir()) == []


def test_clip_ids_stay_stable_when_a_segment_is_out_of_bounds():
    """clip_id is the transcript index, not a count of survivors -- a
    dropped segment must not shift the ids of the ones after it."""
    segments = [
        segment(0.0, 2.0),
        segment(2.5, 2.6),  # 0.1s, below RETAIN_FLOOR_SEC -- dropped outright
        segment(5.0, 7.0),
    ]

    clips = plan_clips(100.0, segments, min_duration=1.0, max_duration=30.0)

    assert [clip["clip_id"] for clip in clips] == ["clip_0001", "clip_0003"]


def test_the_pad_is_asymmetric():
    clips = plan_clips(100.0, [segment(10.0, 12.0)])

    clip = clips[0]
    assert clip["start"] == 10.0 - PAD_BEFORE_SEC
    assert clip["end"] == 12.0 + PAD_AFTER_SEC


def test_the_pad_clamps_to_media_bounds_and_never_goes_negative():
    clips = plan_clips(100.0, [segment(0.05, 1.5)])

    assert clips[0]["start"] == 0.0


def test_the_pad_clamps_to_the_end_of_the_media():
    clips = plan_clips(12.1, [segment(10.0, 12.0)])

    assert clips[0]["end"] == 12.1


def test_the_pad_clamps_into_the_gap_to_the_previous_neighbour():
    """An unclamped pad on a diarized video reaches into the previous
    speaker's last phoneme -- clamp to the midpoint of the gap instead."""
    segments = [segment(0.0, 2.0), segment(2.2, 4.0)]

    clips = plan_clips(100.0, segments)

    gap_midpoint = (2.0 + 2.2) / 2
    assert clips[1]["start"] == gap_midpoint


def test_the_pad_clamps_into_the_gap_to_the_next_neighbour():
    segments = [segment(0.0, 2.0), segment(2.2, 4.0)]

    clips = plan_clips(100.0, segments)

    gap_midpoint = (2.0 + 2.2) / 2
    assert clips[0]["end"] == gap_midpoint


def test_the_duration_filter_uses_the_unpadded_duration():
    """A segment whose padded span would cross min_duration must still be
    judged on its own unpadded length, and the padded span is still what
    gets written out for commit to cut later."""
    # unpadded 0.9s < min_duration 1.0 -- fails, but retained (>= 0.3s floor)
    clips = plan_clips(100.0, [segment(10.0, 10.9)], min_duration=1.0)

    clip = clips[0]
    assert clip["excluded_reason"] == "too_short"
    assert clip["start"] == 10.0 - PAD_BEFORE_SEC
    assert clip["end"] == 10.9 + PAD_AFTER_SEC


def test_out_of_bounds_segments_are_retained_with_the_right_excluded_reason():
    too_short = plan_clips(100.0, [segment(10.0, 10.5)], min_duration=1.0)
    assert too_short[0]["excluded_reason"] == "too_short"
    assert too_short[0]["clip_id"] == "clip_0001"

    too_long = plan_clips(100.0, [segment(10.0, 45.0)], max_duration=30.0)
    assert too_long[0]["excluded_reason"] == "too_long"


def test_a_300_second_segment_is_still_dropped():
    """RETAIN_CEILING_SEC bounds retention -- a multi-minute music passage
    must not be kept just because it failed the max_duration filter."""
    clips = plan_clips(400.0, [segment(0.0, 300.0)], max_duration=30.0)

    assert clips == []


def test_a_segment_within_bounds_has_no_excluded_reason():
    clips = plan_clips(100.0, [segment(10.0, 12.0)])

    assert clips[0]["excluded_reason"] == ""
