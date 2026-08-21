"""Tests for the speaker-map routes in routes/speaker_map.py."""

from pathlib import Path

from voice_factory.repositories.review_csv_repository import write_review_csv


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


def build_video(work_dir: Path, video_id: str, rows: list[dict]) -> Path:
    video_dir = work_dir / "youtube" / video_id
    clips_dir = video_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    for clip_row in rows:
        (clips_dir / f"{clip_row['clip_id']}.wav").write_bytes(b"RIFF-not-real-audio")
    write_review_csv(video_dir / "review.csv", rows)
    return video_dir


def test_put_speaker_map_writes_the_shared_file(client, work_dir):
    build_video(work_dir, "vid1", [row("clip_0001", speaker_label="SPEAKER_00")])

    response = client.put(
        "/videos/vid1/speaker-map",
        json={"speaker_map": {"SPEAKER_00": "janeway", "SPEAKER_01": None}},
    )

    assert response.status_code == 200
    written = work_dir / "youtube" / "vid1" / "speaker_map.json"
    assert written.exists()
    assert client.get("/videos/vid1/clips").json()["speaker_map"] == {
        "SPEAKER_00": "janeway",
        "SPEAKER_01": None,
    }


def test_put_speaker_map_merges_instead_of_overwriting(client, work_dir):
    """A second character's claim must not erase the first character's
    earlier speaker assignments on the same shared video."""
    build_video(work_dir, "vid1", [row("clip_0001", speaker_label="SPEAKER_00")])
    client.put(
        "/videos/vid1/speaker-map", json={"speaker_map": {"SPEAKER_00": "janeway"}}
    )

    response = client.put(
        "/videos/vid1/speaker-map", json={"speaker_map": {"SPEAKER_01": "chakotay"}}
    )

    assert response.status_code == 200
    assert client.get("/videos/vid1/clips").json()["speaker_map"] == {
        "SPEAKER_00": "janeway",
        "SPEAKER_01": "chakotay",
    }


def test_put_speaker_map_rejects_reassigning_an_already_mapped_speaker(
    client, work_dir
):
    """A second character's claim must not silently move an already-mapped
    speaker's clips to a different character (round-1 finding, mirrors
    patch_clips's 409 on an already-recorded clip speaker_label)."""
    build_video(work_dir, "vid1", [row("clip_0001", speaker_label="SPEAKER_00")])
    client.put(
        "/videos/vid1/speaker-map", json={"speaker_map": {"SPEAKER_00": "janeway"}}
    )

    response = client.put(
        "/videos/vid1/speaker-map", json={"speaker_map": {"SPEAKER_00": "chakotay"}}
    )

    assert response.status_code == 409
    assert client.get("/videos/vid1/clips").json()["speaker_map"] == {
        "SPEAKER_00": "janeway"
    }


def test_post_videos_commit_routes_two_videos_to_two_characters(client, work_dir):
    """FR14: one call, two ingested videos, three characters between them."""
    build_video(
        work_dir,
        "vid1",
        [
            row("clip_0001", speaker_label="SPEAKER_00"),
            row("clip_0002", speaker_label="SPEAKER_01"),
        ],
    )
    build_video(work_dir, "vid2", [row("clip_0001", speaker_label="SPEAKER_00")])

    response = client.post(
        "/videos/commit",
        json={
            "assignments": {
                "vid1": {"SPEAKER_00": "janeway", "SPEAKER_01": "chakotay"},
                "vid2": {"SPEAKER_00": "tuvok"},
            }
        },
    )

    assert response.status_code == 200
    assert response.json() == {"committed": {"chakotay": 1, "janeway": 1, "tuvok": 1}}
    assert (
        work_dir / "youtube" / "vid1" / "speaker_map.json"
    ).exists() and (work_dir / "youtube" / "vid2" / "speaker_map.json").exists()


def test_post_videos_commit_rejects_a_conflicting_speaker_map_entry(client, work_dir):
    build_video(work_dir, "vid1", [row("clip_0001", speaker_label="SPEAKER_00")])
    client.put(
        "/videos/vid1/speaker-map", json={"speaker_map": {"SPEAKER_00": "janeway"}}
    )

    response = client.post(
        "/videos/commit",
        json={"assignments": {"vid1": {"SPEAKER_00": "chakotay"}}},
    )

    assert response.status_code == 409
    # the whole payload is rejected, and nothing committed
    assert not (work_dir / "youtube" / "vid1" / "committed.csv").exists()
    assert client.get("/videos/vid1/clips").json()["speaker_map"] == {
        "SPEAKER_00": "janeway"
    }


def test_post_videos_commit_leaves_no_video_written_when_one_conflicts(
    client, work_dir
):
    """A conflict on one video in the payload must not half-apply the rest."""
    build_video(work_dir, "vid1", [row("clip_0001", speaker_label="SPEAKER_00")])
    build_video(work_dir, "vid2", [row("clip_0001", speaker_label="SPEAKER_00")])
    client.put(
        "/videos/vid1/speaker-map", json={"speaker_map": {"SPEAKER_00": "janeway"}}
    )

    response = client.post(
        "/videos/commit",
        json={
            "assignments": {
                "vid1": {"SPEAKER_00": "chakotay"},
                "vid2": {"SPEAKER_00": "tuvok"},
            }
        },
    )

    assert response.status_code == 409
    assert client.get("/videos/vid2/clips").json()["speaker_map"] == {}


def test_post_videos_commit_404s_for_an_unknown_video(client, work_dir):
    build_video(work_dir, "vid1", [row("clip_0001", speaker_label="SPEAKER_00")])

    response = client.post(
        "/videos/commit",
        json={"assignments": {"nope": {"SPEAKER_00": "janeway"}}},
    )

    assert response.status_code == 404
    # vid1 was never named in the payload, but this also proves nothing at
    # all was written before the unknown video was rejected
    assert client.get("/videos/vid1/clips").json()["speaker_map"] == {}
