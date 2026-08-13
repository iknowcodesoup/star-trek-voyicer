import json
from pathlib import Path

from review import commit_reviewed_clips, write_review_csv, write_speaker_map


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


def test_speaker_map_round_trips(tmp_path):
    written = write_speaker_map(tmp_path, {"SPEAKER_00": "janeway", "SPEAKER_01": None})

    assert json.loads(written.read_text(encoding="utf-8")) == {
        "SPEAKER_00": "janeway",
        "SPEAKER_01": None,
    }
