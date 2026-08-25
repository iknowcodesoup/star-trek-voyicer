import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from voice_factory.core.audio_slicing import TARGET_RATE
from voice_factory.core.review_workflow import (
    SpeakerMapConflict,
    clips_for_character,
    compile_dataset_for,
    write_speaker_map,
)
from voice_factory.repositories.review_csv_repository import write_review_csv


def build_video(youtube_dir: Path, video_id: str, rows: list[dict]) -> Path:
    video_dir = youtube_dir / video_id
    clips_dir = video_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        (clips_dir / f"{row['clip_id']}.wav").write_bytes(b"RIFF-not-real-audio")
    write_review_csv(video_dir / "review.csv", rows)
    return video_dir


def build_video_with_full_wav(
    youtube_dir: Path, video_id: str, rows: list[dict], num_seconds: float = 10.0
) -> Path:
    """A video with full.wav present, no clips/*.wav -- the shape every
    video takes once plan_clips replaces chunk_clips."""
    video_dir = youtube_dir / video_id
    video_dir.mkdir(parents=True, exist_ok=True)
    ramp = np.arange(round(num_seconds * TARGET_RATE), dtype=np.int16)
    sf.write(video_dir / "full.wav", ramp, TARGET_RATE, subtype="PCM_16")
    write_review_csv(video_dir / "review.csv", rows)
    return video_dir


def row(
    clip_id: str, keep: str = "1", speaker_label: str = "", assigned_voice: str = ""
) -> dict:
    return {
        "clip_id": clip_id,
        "keep": keep,
        "quality_score": 22.0,
        "flagged": 0,
        "speaker_label": speaker_label,
        "speaker_coverage": 1.0,
        "assigned_voice": assigned_voice,
        "duration_sec": 3.0,
        "start_sec": 0.0,
        "end_sec": 3.0,
        "text": f"line for {clip_id}",
    }


def read_metadata(dataset_dir: Path) -> list[str]:
    path = dataset_dir / "metadata.csv"
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8").splitlines()


def wav_names(dataset_dir: Path) -> list[str]:
    wavs_dir = dataset_dir / "wavs"
    if not wavs_dir.exists():
        return []
    return sorted(path.name for path in wavs_dir.iterdir())


# --- gathering: which clips a compile picks up ---------------------------


def test_compile_gathers_only_kept_clips_assigned_to_this_character(tmp_path):
    youtube_dir = tmp_path / "youtube"
    build_video(
        youtube_dir,
        "vid1",
        [
            row("clip_0001", assigned_voice="janeway"),
            row("clip_0002", keep="0", assigned_voice="janeway"),
            row("clip_0003", assigned_voice="chakotay"),
            row("clip_0004"),
        ],
    )
    dataset_dir = tmp_path / "dataset"

    result = compile_dataset_for("janeway", youtube_dir, dataset_dir)

    assert result.clip_count == 1
    assert read_metadata(dataset_dir) == ["yt_vid1_clip_0001|line for clip_0001"]


def test_compile_gathers_one_voice_across_several_videos(tmp_path):
    """The whole reason a voice's dataset cannot be built per run: a voice is
    made of clips spread over many videos."""
    youtube_dir = tmp_path / "youtube"
    build_video(youtube_dir, "vid1", [row("clip_0001", assigned_voice="janeway")])
    build_video(youtube_dir, "vid2", [row("clip_0002", assigned_voice="janeway")])
    build_video(youtube_dir, "vid3", [row("clip_0003", assigned_voice="chakotay")])
    dataset_dir = tmp_path / "dataset"

    result = compile_dataset_for("janeway", youtube_dir, dataset_dir)

    assert result.clip_count == 2
    assert wav_names(dataset_dir) == [
        "yt_vid1_clip_0001.wav",
        "yt_vid2_clip_0002.wav",
    ]


def test_compile_ignores_the_speaker_map(tmp_path):
    """assigned_voice is the only thing that routes a clip into a dataset.
    speaker_map.json answers who diarization heard, which is a different
    question from who the clip is for."""
    youtube_dir = tmp_path / "youtube"
    video_dir = build_video(
        youtube_dir, "vid1", [row("clip_0001", speaker_label="SPEAKER_00")]
    )
    write_speaker_map(video_dir, {"SPEAKER_00": "janeway"})
    dataset_dir = tmp_path / "dataset"

    result = compile_dataset_for("janeway", youtube_dir, dataset_dir)

    assert result.clip_count == 0
    assert read_metadata(dataset_dir) == []


def test_compile_writes_an_empty_dataset_when_nothing_is_assigned(tmp_path):
    youtube_dir = tmp_path / "youtube"
    build_video(youtube_dir, "vid1", [row("clip_0001", assigned_voice="chakotay")])
    dataset_dir = tmp_path / "dataset"

    result = compile_dataset_for("janeway", youtube_dir, dataset_dir)

    assert result.clip_count == 0
    assert read_metadata(dataset_dir) == []
    assert (dataset_dir / "wavs").is_dir()


def test_compile_handles_a_youtube_dir_that_does_not_exist(tmp_path):
    dataset_dir = tmp_path / "dataset"

    result = compile_dataset_for("janeway", tmp_path / "missing", dataset_dir)

    assert result.clip_count == 0
    assert (dataset_dir / "wavs").is_dir()


def test_compile_skips_a_video_with_no_review_csv(tmp_path):
    youtube_dir = tmp_path / "youtube"
    (youtube_dir / "vid1").mkdir(parents=True)
    build_video(youtube_dir, "vid2", [row("clip_0001", assigned_voice="janeway")])
    dataset_dir = tmp_path / "dataset"

    result = compile_dataset_for("janeway", youtube_dir, dataset_dir)

    assert result.clip_count == 1


def test_clips_for_character_orders_videos_by_name(tmp_path):
    """Two compiles of the same decisions must write the same metadata.csv."""
    youtube_dir = tmp_path / "youtube"
    build_video(youtube_dir, "vid_b", [row("clip_0002", assigned_voice="janeway")])
    build_video(youtube_dir, "vid_a", [row("clip_0001", assigned_voice="janeway")])

    gathered = clips_for_character(youtube_dir, "janeway")

    assert [video_dir.name for video_dir, _row in gathered] == ["vid_a", "vid_b"]


# --- rebuild-from-scratch: what makes decisions reversible ---------------


def test_compile_is_idempotent(tmp_path):
    youtube_dir = tmp_path / "youtube"
    build_video(youtube_dir, "vid1", [row("clip_0001", assigned_voice="janeway")])
    dataset_dir = tmp_path / "dataset"

    compile_dataset_for("janeway", youtube_dir, dataset_dir)
    first = read_metadata(dataset_dir)
    second_result = compile_dataset_for("janeway", youtube_dir, dataset_dir)

    assert second_result.clip_count == 1
    assert read_metadata(dataset_dir) == first
    assert wav_names(dataset_dir) == ["yt_vid1_clip_0001.wav"]


def test_un_keeping_a_clip_removes_it_from_the_next_compile(tmp_path):
    """No delete path: the clip is gone because it was not gathered."""
    youtube_dir = tmp_path / "youtube"
    rows = [
        row("clip_0001", assigned_voice="janeway"),
        row("clip_0002", assigned_voice="janeway"),
    ]
    video_dir = build_video(youtube_dir, "vid1", rows)
    dataset_dir = tmp_path / "dataset"
    compile_dataset_for("janeway", youtube_dir, dataset_dir)

    rows[0]["keep"] = "0"
    write_review_csv(video_dir / "review.csv", rows)
    result = compile_dataset_for("janeway", youtube_dir, dataset_dir)

    assert result.clip_count == 1
    assert wav_names(dataset_dir) == ["yt_vid1_clip_0002.wav"]
    assert read_metadata(dataset_dir) == ["yt_vid1_clip_0002|line for clip_0002"]


def test_clearing_a_clip_back_to_undecided_removes_it(tmp_path):
    youtube_dir = tmp_path / "youtube"
    rows = [row("clip_0001", assigned_voice="janeway")]
    video_dir = build_video(youtube_dir, "vid1", rows)
    dataset_dir = tmp_path / "dataset"
    compile_dataset_for("janeway", youtube_dir, dataset_dir)

    rows[0]["keep"] = ""
    write_review_csv(video_dir / "review.csv", rows)
    result = compile_dataset_for("janeway", youtube_dir, dataset_dir)

    assert result.clip_count == 0
    assert wav_names(dataset_dir) == []


def test_reassigning_a_clip_moves_it_between_two_voices(tmp_path):
    youtube_dir = tmp_path / "youtube"
    rows = [row("clip_0001", assigned_voice="janeway")]
    video_dir = build_video(youtube_dir, "vid1", rows)
    janeway_dir = tmp_path / "janeway" / "dataset"
    chakotay_dir = tmp_path / "chakotay" / "dataset"
    compile_dataset_for("janeway", youtube_dir, janeway_dir)
    assert wav_names(janeway_dir) == ["yt_vid1_clip_0001.wav"]

    rows[0]["assigned_voice"] = "chakotay"
    write_review_csv(video_dir / "review.csv", rows)
    compile_dataset_for("janeway", youtube_dir, janeway_dir)
    compile_dataset_for("chakotay", youtube_dir, chakotay_dir)

    assert wav_names(janeway_dir) == []
    assert wav_names(chakotay_dir) == ["yt_vid1_clip_0001.wav"]


def test_compile_replaces_a_dataset_left_by_an_earlier_run(tmp_path):
    """A wav no compile would produce must not survive into training."""
    youtube_dir = tmp_path / "youtube"
    build_video(youtube_dir, "vid1", [row("clip_0001", assigned_voice="janeway")])
    dataset_dir = tmp_path / "dataset"
    (dataset_dir / "wavs").mkdir(parents=True)
    (dataset_dir / "wavs" / "stale.wav").write_bytes(b"stale")
    (dataset_dir / "metadata.csv").write_text("stale|stale line\n", encoding="utf-8")

    compile_dataset_for("janeway", youtube_dir, dataset_dir)

    assert wav_names(dataset_dir) == ["yt_vid1_clip_0001.wav"]
    assert read_metadata(dataset_dir) == ["yt_vid1_clip_0001|line for clip_0001"]


def test_compile_clears_a_staging_directory_left_by_a_crash(tmp_path):
    youtube_dir = tmp_path / "youtube"
    build_video(youtube_dir, "vid1", [row("clip_0001", assigned_voice="janeway")])
    dataset_dir = tmp_path / "dataset"
    staging_dir = dataset_dir.with_name("dataset.compiling")
    (staging_dir / "wavs").mkdir(parents=True)
    (staging_dir / "wavs" / "half_written.wav").write_bytes(b"partial")

    compile_dataset_for("janeway", youtube_dir, dataset_dir)

    assert wav_names(dataset_dir) == ["yt_vid1_clip_0001.wav"]
    assert not staging_dir.exists()


# --- audio: where each clip's wav comes from -----------------------------


def test_compile_cuts_from_full_wav_at_the_reviewed_bounds(tmp_path):
    youtube_dir = tmp_path / "youtube"
    build_video_with_full_wav(
        youtube_dir, "vid1", [row("clip_0001", assigned_voice="janeway")]
    )
    dataset_dir = tmp_path / "dataset"

    result = compile_dataset_for("janeway", youtube_dir, dataset_dir)

    assert result.clip_count == 1
    samples, rate = sf.read(
        dataset_dir / "wavs" / "yt_vid1_clip_0001.wav", dtype="int16"
    )
    assert rate == TARGET_RATE
    # row("clip_0001") carries start_sec=0.0, end_sec=3.0
    assert len(samples) == round(3.0 * TARGET_RATE)


def test_compile_falls_back_to_a_pre_cut_clip_when_there_is_no_full_wav(tmp_path):
    youtube_dir = tmp_path / "youtube"
    build_video(youtube_dir, "vid1", [row("clip_0001", assigned_voice="janeway")])
    dataset_dir = tmp_path / "dataset"

    result = compile_dataset_for("janeway", youtube_dir, dataset_dir)

    assert result.clip_count == 1
    assert (dataset_dir / "wavs" / "yt_vid1_clip_0001.wav").read_bytes() == (
        b"RIFF-not-real-audio"
    )


def test_compile_prefers_full_wav_over_a_stale_pre_cut_clip(tmp_path):
    """A reviewer's trim only ever changes review.csv's bounds, so a stale
    pre-cut clips/*.wav must never win -- the fix would never reach the
    dataset and nothing would error."""
    youtube_dir = tmp_path / "youtube"
    video_dir = build_video(
        youtube_dir, "vid1", [row("clip_0001", assigned_voice="janeway")]
    )
    ramp = np.arange(round(10.0 * TARGET_RATE), dtype=np.int16)
    sf.write(video_dir / "full.wav", ramp, TARGET_RATE, subtype="PCM_16")
    dataset_dir = tmp_path / "dataset"

    compile_dataset_for("janeway", youtube_dir, dataset_dir)

    samples, _ = sf.read(
        dataset_dir / "wavs" / "yt_vid1_clip_0001.wav", dtype="int16"
    )
    # sliced from full.wav's ramp, not the stale "RIFF-not-real-audio" bytes
    assert samples[0] == 0
    assert len(samples) == round(3.0 * TARGET_RATE)


def test_compile_re_slices_a_clip_whose_bounds_changed(tmp_path):
    """A trim after an earlier compile takes effect with no re-slice path:
    the next compile cuts from full.wav at the current bounds."""
    youtube_dir = tmp_path / "youtube"
    rows = [row("clip_0001", assigned_voice="janeway")]
    video_dir = build_video_with_full_wav(youtube_dir, "vid1", rows)
    dataset_dir = tmp_path / "dataset"
    compile_dataset_for("janeway", youtube_dir, dataset_dir)

    rows[0]["end_sec"] = 5.0
    write_review_csv(video_dir / "review.csv", rows)
    compile_dataset_for("janeway", youtube_dir, dataset_dir)

    samples, _ = sf.read(
        dataset_dir / "wavs" / "yt_vid1_clip_0001.wav", dtype="int16"
    )
    assert len(samples) == round(5.0 * TARGET_RATE)


def test_compile_clamps_bounds_past_eof(tmp_path):
    youtube_dir = tmp_path / "youtube"
    clip_row = row("clip_0001", assigned_voice="janeway")
    clip_row["end_sec"] = 500.0  # past the 10s fixture
    build_video_with_full_wav(youtube_dir, "vid1", [clip_row])
    dataset_dir = tmp_path / "dataset"

    result = compile_dataset_for("janeway", youtube_dir, dataset_dir)

    assert result.clip_count == 1
    samples, _ = sf.read(
        dataset_dir / "wavs" / "yt_vid1_clip_0001.wav", dtype="int16"
    )
    assert len(samples) == round(10.0 * TARGET_RATE)


def test_compile_skips_a_row_that_clamps_to_zero_frames(tmp_path):
    youtube_dir = tmp_path / "youtube"
    clip_row = row("clip_0001", assigned_voice="janeway")
    clip_row["start_sec"] = 500.0  # past EOF, so start clamps to the same
    clip_row["end_sec"] = 600.0  # frame end does, leaving nothing to write
    build_video_with_full_wav(youtube_dir, "vid1", [clip_row])
    dataset_dir = tmp_path / "dataset"

    result = compile_dataset_for("janeway", youtube_dir, dataset_dir)

    assert result.clip_count == 0
    assert result.skipped_count == 1
    assert read_metadata(dataset_dir) == []
    assert wav_names(dataset_dir) == []


def test_compile_skips_a_row_with_no_audio_at_all(tmp_path):
    youtube_dir = tmp_path / "youtube"
    video_dir = youtube_dir / "vid1"
    video_dir.mkdir(parents=True)
    write_review_csv(
        video_dir / "review.csv", [row("clip_0001", assigned_voice="janeway")]
    )

    result = compile_dataset_for("janeway", youtube_dir, tmp_path / "dataset")

    assert result.clip_count == 0
    assert result.skipped_count == 1


# --- speaker map: unchanged, still the diarization record ----------------


def test_speaker_map_round_trips(tmp_path):
    written = write_speaker_map(tmp_path, {"SPEAKER_00": "janeway", "SPEAKER_01": None})

    assert json.loads(written.read_text(encoding="utf-8")) == {
        "SPEAKER_00": "janeway",
        "SPEAKER_01": None,
    }


def test_write_speaker_map_merges_instead_of_overwriting(tmp_path):
    """A second character's claim on a shared video must not erase the first
    character's earlier speaker assignments."""
    write_speaker_map(tmp_path, {"SPEAKER_00": "janeway"})

    written = write_speaker_map(tmp_path, {"SPEAKER_01": "chakotay"})

    assert json.loads(written.read_text(encoding="utf-8")) == {
        "SPEAKER_00": "janeway",
        "SPEAKER_01": "chakotay",
    }


def test_write_speaker_map_allows_resubmitting_the_same_value(tmp_path):
    write_speaker_map(tmp_path, {"SPEAKER_00": "janeway"})

    written = write_speaker_map(tmp_path, {"SPEAKER_00": "janeway"})

    assert json.loads(written.read_text(encoding="utf-8")) == {"SPEAKER_00": "janeway"}


def test_write_speaker_map_rejects_reassigning_an_already_mapped_speaker(tmp_path):
    """A second character's claim must not silently move a speaker's clips
    away from the character an earlier claim already assigned it to."""
    write_speaker_map(tmp_path, {"SPEAKER_00": "janeway"})

    with pytest.raises(SpeakerMapConflict) as exc_info:
        write_speaker_map(tmp_path, {"SPEAKER_00": "chakotay"})

    assert exc_info.value.conflicting_labels == ["SPEAKER_00"]
    # the whole request is rejected, not partially applied
    assert json.loads((tmp_path / "speaker_map.json").read_text(encoding="utf-8")) == {
        "SPEAKER_00": "janeway"
    }
