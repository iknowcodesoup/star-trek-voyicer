from voice_factory.core.diarization import assign_speakers, count_by_speaker


def clip(start: float, end: float, clip_id: str = "clip_0001") -> dict:
    return {
        "clip_id": clip_id,
        "start": start,
        "end": end,
        "duration": end - start,
        "text": "some words",
    }


def turn(start: float, end: float, speaker: str) -> dict:
    return {"start": start, "end": end, "speaker": speaker}


def test_clip_inside_one_turn_gets_that_speaker():
    result = assign_speakers([clip(2.0, 4.0)], [turn(0.0, 10.0, "SPEAKER_00")])

    assert result[0]["speaker_label"] == "SPEAKER_00"
    assert result[0]["speaker_coverage"] == 1.0


def test_clip_straddling_two_speakers_still_reports_the_dominant_one():
    # 2s clip split evenly, so neither speaker reaches the 0.9 default -- the
    # label is still reported so a reviewer can see and override it; only
    # min_coverage-aware callers (review, count_by_speaker) treat this as
    # rejected
    result = assign_speakers(
        [clip(4.0, 6.0)],
        [turn(0.0, 5.0, "SPEAKER_00"), turn(5.0, 10.0, "SPEAKER_01")],
    )

    assert result[0]["speaker_label"] == "SPEAKER_00"
    assert result[0]["speaker_coverage"] == 0.5


def test_clip_with_no_covering_turn_gets_no_label():
    # the gap between turns is music or silence -- this is the VAD behaviour,
    # and there is nothing to report because no turn overlaps at all
    result = assign_speakers(
        [clip(6.0, 8.0)],
        [turn(0.0, 5.0, "SPEAKER_00"), turn(9.0, 12.0, "SPEAKER_01")],
    )

    assert result[0]["speaker_label"] is None
    assert result[0]["speaker_coverage"] == 0.0


def test_dominant_speaker_wins_above_the_coverage_floor():
    # 95% SPEAKER_00, 5% SPEAKER_01 -- clears the default 0.9
    result = assign_speakers(
        [clip(0.0, 10.0)],
        [turn(0.0, 9.5, "SPEAKER_00"), turn(9.5, 20.0, "SPEAKER_01")],
    )

    assert result[0]["speaker_label"] == "SPEAKER_00"
    assert result[0]["speaker_coverage"] == 0.95


def test_count_by_speaker_min_coverage_is_configurable():
    labelled = assign_speakers(
        [clip(4.0, 6.0)],
        [turn(0.0, 5.0, "SPEAKER_00"), turn(5.0, 10.0, "SPEAKER_01")],
    )

    relaxed = count_by_speaker(labelled, min_coverage=0.5)

    assert relaxed == {"SPEAKER_00": 1}


def test_coverage_never_exceeds_one_when_turns_overlap():
    # pyannote can emit slightly overlapping turns for the same speaker; the
    # summed overlap must still read as a fraction
    result = assign_speakers(
        [clip(0.0, 4.0)],
        [turn(0.0, 3.0, "SPEAKER_00"), turn(2.0, 6.0, "SPEAKER_00")],
    )

    assert result[0]["speaker_coverage"] == 1.0
    assert result[0]["speaker_label"] == "SPEAKER_00"


def test_assign_speakers_does_not_mutate_its_input():
    clips = [clip(2.0, 4.0)]

    assign_speakers(clips, [turn(0.0, 10.0, "SPEAKER_00")])

    assert "speaker_label" not in clips[0]


def test_original_clip_fields_survive():
    result = assign_speakers([clip(2.0, 4.0)], [turn(0.0, 10.0, "SPEAKER_00")])

    assert result[0]["clip_id"] == "clip_0001"
    assert result[0]["text"] == "some words"
    assert result[0]["duration"] == 2.0


def test_count_by_speaker_buckets_rejected_clips_together():
    labelled = assign_speakers(
        [clip(0.0, 2.0, "a"), clip(2.0, 4.0, "b"), clip(20.0, 22.0, "c")],
        [turn(0.0, 4.0, "SPEAKER_00")],
    )

    assert count_by_speaker(labelled) == {"SPEAKER_00": 2, "rejected": 1}
