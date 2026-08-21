import json
from pathlib import Path

import pytest

from voice_factory.core.review_workflow import (
    SpeakerMapConflict,
    commit_reviewed_clips,
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


def row(clip_id: str, keep: str = "1", speaker_label: str = "") -> dict:
    return {
        "clip_id": clip_id,
        "keep": keep,
        "quality_score": 22.0,
        "flagged": 0,
        "speaker_label": speaker_label,
        "speaker_coverage": 1.0,
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


def test_commit_without_a_map_keeps_the_original_behaviour(tmp_path):
    youtube_dir = tmp_path / "youtube"
    build_video(youtube_dir, "vid1", [row("clip_0001"), row("clip_0002", keep="0")])
    out_dir = tmp_path / "dataset"

    result = commit_reviewed_clips(youtube_dir, out_dir)

    assert result.newly_committed == 1
    assert result.already_committed == 0
    assert read_metadata(out_dir) == ["yt_vid1_clip_0001|line for clip_0001"]
    assert (out_dir / "wavs" / "yt_vid1_clip_0001.wav").exists()


def test_commit_is_idempotent(tmp_path):
    youtube_dir = tmp_path / "youtube"
    build_video(youtube_dir, "vid1", [row("clip_0001")])
    out_dir = tmp_path / "dataset"

    commit_reviewed_clips(youtube_dir, out_dir)
    second = commit_reviewed_clips(youtube_dir, out_dir)

    assert second.newly_committed == 0
    assert second.already_committed == 1
    assert len(read_metadata(out_dir)) == 1


def test_speaker_map_routes_clips_to_separate_characters(tmp_path):
    youtube_dir = tmp_path / "youtube"
    video_dir = build_video(
        youtube_dir,
        "vid1",
        [
            row("clip_0001", speaker_label="SPEAKER_00"),
            row("clip_0002", speaker_label="SPEAKER_01"),
            row("clip_0003", speaker_label="SPEAKER_02"),
        ],
    )
    write_speaker_map(
        video_dir,
        {"SPEAKER_00": "janeway", "SPEAKER_01": "chakotay", "SPEAKER_02": None},
    )
    work_dir = tmp_path / "work"

    def dataset_dir_for(character: str) -> Path:
        return work_dir / character / "dataset"

    result = commit_reviewed_clips(
        youtube_dir, dataset_dir_for("janeway"), dataset_dir_for
    )

    assert result.newly_committed == 2
    assert read_metadata(dataset_dir_for("janeway")) == [
        "yt_vid1_clip_0001|line for clip_0001"
    ]
    assert read_metadata(dataset_dir_for("chakotay")) == [
        "yt_vid1_clip_0002|line for clip_0002"
    ]
    # SPEAKER_02 mapped to null, so it is discarded
    assert read_metadata(dataset_dir_for("tuvok")) == []
    assert result.committed_by_target == {
        dataset_dir_for("janeway"): 1,
        dataset_dir_for("chakotay"): 1,
    }


def test_unmapped_speaker_stays_uncommittable_until_the_map_is_fixed(tmp_path):
    youtube_dir = tmp_path / "youtube"
    video_dir = build_video(
        youtube_dir, "vid1", [row("clip_0001", speaker_label="SPEAKER_07")]
    )
    write_speaker_map(video_dir, {"SPEAKER_00": "janeway"})
    work_dir = tmp_path / "work"

    def dataset_dir_for(character: str) -> Path:
        return work_dir / character / "dataset"

    first = commit_reviewed_clips(
        youtube_dir, dataset_dir_for("janeway"), dataset_dir_for
    )
    assert first.newly_committed == 0

    # correcting the map must let the clip through on the next run
    write_speaker_map(video_dir, {"SPEAKER_07": "tuvok"})
    second = commit_reviewed_clips(
        youtube_dir, dataset_dir_for("janeway"), dataset_dir_for
    )

    assert second.newly_committed == 1
    assert read_metadata(dataset_dir_for("tuvok")) == [
        "yt_vid1_clip_0001|line for clip_0001"
    ]


def test_each_video_uses_its_own_speaker_map(tmp_path):
    # SPEAKER_00 in one video is a different person from SPEAKER_00 in the next
    youtube_dir = tmp_path / "youtube"
    first_video = build_video(
        youtube_dir, "vid1", [row("clip_0001", speaker_label="SPEAKER_00")]
    )
    second_video = build_video(
        youtube_dir, "vid2", [row("clip_0001", speaker_label="SPEAKER_00")]
    )
    write_speaker_map(first_video, {"SPEAKER_00": "janeway"})
    write_speaker_map(second_video, {"SPEAKER_00": "tuvok"})
    work_dir = tmp_path / "work"

    def dataset_dir_for(character: str) -> Path:
        return work_dir / character / "dataset"

    commit_reviewed_clips(youtube_dir, dataset_dir_for("janeway"), dataset_dir_for)

    assert read_metadata(dataset_dir_for("janeway")) == [
        "yt_vid1_clip_0001|line for clip_0001"
    ]
    assert read_metadata(dataset_dir_for("tuvok")) == [
        "yt_vid2_clip_0001|line for clip_0001"
    ]


def test_undiarized_row_falls_back_to_the_primary_dataset(tmp_path):
    youtube_dir = tmp_path / "youtube"
    video_dir = build_video(youtube_dir, "vid1", [row("clip_0001", speaker_label="")])
    write_speaker_map(video_dir, {"SPEAKER_00": "chakotay"})
    work_dir = tmp_path / "work"

    def dataset_dir_for(character: str) -> Path:
        return work_dir / character / "dataset"

    result = commit_reviewed_clips(
        youtube_dir, dataset_dir_for("janeway"), dataset_dir_for
    )

    assert result.newly_committed == 1
    assert read_metadata(dataset_dir_for("janeway")) == [
        "yt_vid1_clip_0001|line for clip_0001"
    ]


def test_review_csv_written_before_diarization_still_commits(tmp_path):
    # a review.csv with no speaker columns at all, as written by the old code
    youtube_dir = tmp_path / "youtube"
    video_dir = youtube_dir / "vid1"
    (video_dir / "clips").mkdir(parents=True)
    (video_dir / "clips" / "clip_0001.wav").write_bytes(b"RIFF-not-real-audio")
    (video_dir / "review.csv").write_text(
        "clip_id,keep,quality_score,flagged,duration_sec,start_sec,end_sec,text\n"
        "clip_0001,1,22.0,0,3.0,0.0,3.0,an older line\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "dataset"

    result = commit_reviewed_clips(youtube_dir, out_dir)

    assert result.newly_committed == 1
    assert read_metadata(out_dir) == ["yt_vid1_clip_0001|an older line"]


def test_out_dir_none_skips_unmapped_and_undiarized_rows(tmp_path):
    """The batched, multi-character commit route has no single "committing
    character" to fall back to. out_dir=None must leave those rows
    uncommitted instead of guessing -- never silently route them anywhere."""
    youtube_dir = tmp_path / "youtube"
    # SPEAKER_00 is mapped, SPEAKER_09 is not, and clip_0003 was never diarized
    video_dir = build_video(
        youtube_dir,
        "vid1",
        [
            row("clip_0001", speaker_label="SPEAKER_00"),
            row("clip_0002", speaker_label="SPEAKER_09"),
            row("clip_0003", speaker_label=""),
        ],
    )
    write_speaker_map(video_dir, {"SPEAKER_00": "janeway"})
    work_dir = tmp_path / "work"

    def dataset_dir_for(character: str) -> Path:
        return work_dir / character / "dataset"

    result = commit_reviewed_clips(youtube_dir, None, dataset_dir_for)

    assert result.newly_committed == 1
    assert read_metadata(dataset_dir_for("janeway")) == [
        "yt_vid1_clip_0001|line for clip_0001"
    ]
    # neither the unmapped speaker nor the undiarized row landed anywhere,
    # and neither counted as committed
    committed_text = (youtube_dir / "vid1" / "committed.csv").read_text()
    assert "clip_0001" in committed_text
    assert "clip_0002" not in committed_text
    assert "clip_0003" not in committed_text


def test_out_dir_none_with_no_map_at_all_skips_every_row(tmp_path):
    """A video with no speaker_map.json and dataset_dir_for set is already
    skipped at the video level (see the shared-scan test above). out_dir=None
    is the second guard: even if that video-level skip were ever bypassed,
    the row-level fallback must not guess either."""
    youtube_dir = tmp_path / "youtube"
    build_video(youtube_dir, "vid1", [row("clip_0001")])
    work_dir = tmp_path / "work"

    def dataset_dir_for(character: str) -> Path:
        return work_dir / character / "dataset"

    result = commit_reviewed_clips(youtube_dir, None, dataset_dir_for)

    assert result.newly_committed == 0
    assert result.committed_by_target == {}


def test_speaker_map_round_trips(tmp_path):
    written = write_speaker_map(tmp_path, {"SPEAKER_00": "janeway", "SPEAKER_01": None})

    assert json.loads(written.read_text(encoding="utf-8")) == {
        "SPEAKER_00": "janeway",
        "SPEAKER_01": None,
    }


def test_write_speaker_map_merges_instead_of_overwriting(tmp_path):
    """A second character's claim on a shared video must not erase the first
    character's earlier speaker assignments (round-1 finding)."""
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
    away from the character an earlier claim already assigned it to -- the
    same corruption class patch_clips's 409 guards against, one layer up."""
    write_speaker_map(tmp_path, {"SPEAKER_00": "janeway"})

    with pytest.raises(SpeakerMapConflict) as exc_info:
        write_speaker_map(tmp_path, {"SPEAKER_00": "chakotay"})

    assert exc_info.value.conflicting_labels == ["SPEAKER_00"]
    # the whole request is rejected, not partially applied
    assert json.loads((tmp_path / "speaker_map.json").read_text(encoding="utf-8")) == {
        "SPEAKER_00": "janeway"
    }


def test_commit_skips_another_characters_unmapped_video_on_a_shared_scan(tmp_path):
    """The round-1 regression test: this story moves every character's
    ingested videos into one shared work/youtube/ directory. Before this
    fix, commit_reviewed_clips fell back to "assume this video is mine" for
    any video with no speaker_map.json -- correct when youtube_dir held only
    one character's videos, unsafe once it is shared. stage_youtube_review
    auto-writes review.csv with keep=1 rows before any human review or
    speaker_map.json exists, so a character committing while an unrelated
    video sits mid-ingest for someone else must never sweep it in.
    """
    youtube_dir = tmp_path / "youtube"
    # janeway's video: reviewed and explicitly mapped to janeway
    janeway_video = build_video(
        youtube_dir, "vid_janeway", [row("clip_0001", speaker_label="SPEAKER_00")]
    )
    write_speaker_map(janeway_video, {"SPEAKER_00": "janeway"})
    # chakotay's video: auto-written keep=1 rows from stage_youtube_review,
    # but chakotay has not reviewed it or written a speaker_map.json yet
    build_video(youtube_dir, "vid_chakotay", [row("clip_0001")])
    work_dir = tmp_path / "work"

    def dataset_dir_for(character: str) -> Path:
        return work_dir / character / "dataset"

    result = commit_reviewed_clips(
        youtube_dir, dataset_dir_for("janeway"), dataset_dir_for
    )

    assert read_metadata(dataset_dir_for("janeway")) == [
        "yt_vid_janeway_clip_0001|line for clip_0001"
    ]
    # the unmapped video's clip must never land in janeway's dataset, and
    # nothing marks it committed either -- chakotay's own later commit run
    # must still see it as pending
    assert read_metadata(dataset_dir_for("chakotay")) == []
    assert not (youtube_dir / "vid_chakotay" / "committed.csv").exists()
    assert result.newly_committed == 1


def test_commit_discards_clips_whose_map_routes_every_speaker_to_null(tmp_path):
    """An all-null speaker_map means the reviewer explicitly discarded every
    speaker in this video. That must not fall back to "assume it's mine"
    either -- same unsafe pattern as the missing-map case above, one row at a
    time instead of one video at a time."""
    youtube_dir = tmp_path / "youtube"
    video_dir = build_video(
        youtube_dir, "vid1", [row("clip_0001", speaker_label="SPEAKER_00")]
    )
    write_speaker_map(video_dir, {"SPEAKER_00": None})
    work_dir = tmp_path / "work"

    def dataset_dir_for(character: str) -> Path:
        return work_dir / character / "dataset"

    result = commit_reviewed_clips(
        youtube_dir, dataset_dir_for("janeway"), dataset_dir_for
    )

    assert result.newly_committed == 0
    assert read_metadata(dataset_dir_for("janeway")) == []


def test_a_second_characters_later_commit_still_sees_the_previously_skipped_video(
    tmp_path,
):
    """Once chakotay reviews and maps their own video, their own commit run
    picks it up -- the earlier skip only deferred it, it did not lose it."""
    youtube_dir = tmp_path / "youtube"
    build_video(youtube_dir, "vid_chakotay", [row("clip_0001")])
    work_dir = tmp_path / "work"

    def dataset_dir_for(character: str) -> Path:
        return work_dir / character / "dataset"

    # janeway's commit runs first, while chakotay's video is still unmapped
    commit_reviewed_clips(youtube_dir, dataset_dir_for("janeway"), dataset_dir_for)
    assert read_metadata(dataset_dir_for("chakotay")) == []

    # chakotay reviews and maps it, then commits
    write_speaker_map(youtube_dir / "vid_chakotay", {"SPEAKER_00": "chakotay"})
    commit_reviewed_clips(youtube_dir, dataset_dir_for("chakotay"), dataset_dir_for)

    assert read_metadata(dataset_dir_for("chakotay")) == [
        "yt_vid_chakotay_clip_0001|line for clip_0001"
    ]
