"""Tests for the video-scoped routes in api.py.

api.py is safe to import directly here: unlike main.py, it never imports
chatterbox-tts, whisper, or piper_phonemize at module load time, so these
tests need no GPU and no heavy model download.

Every test points api.WORK_DIR at a tmp_path, so nothing here touches the
real work/ directory these stages write to outside of tests.
"""

import sys
from pathlib import Path

# api.py lives one directory up from tests/, and is not itself a package --
# the same problem conftest.py already solves for src/, one level further out.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from fastapi.testclient import TestClient

import api
from review import write_review_csv, write_speaker_map
from youtube_ingest import DIARIZATION_NAME


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


@pytest.fixture
def work_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(api, "WORK_DIR", tmp_path)
    return tmp_path


@pytest.fixture
def client():
    with TestClient(api.app) as test_client:
        yield test_client


def test_get_clips_reads_the_shared_video_path_with_no_character_segment(
    client, work_dir
):
    """FR12: the URL that reads a video's clips carries no character at all,
    so a second character claiming the same video hits the same route."""
    build_video(work_dir, "vid1", [row("clip_0001"), row("clip_0002", keep="0")])

    response = client.get("/videos/vid1/clips")

    assert response.status_code == 200
    body = response.json()
    assert body["video_id"] == "vid1"
    assert [clip["clip_id"] for clip in body["clips"]] == ["clip_0001", "clip_0002"]
    assert body["clips"][0]["keep"] is True
    assert body["clips"][1]["keep"] is False


def test_get_clips_404s_for_an_unknown_video(client, work_dir):
    response = client.get("/videos/nope/clips")

    assert response.status_code == 404


def test_patch_clips_writes_through_to_review_csv(client, work_dir):
    build_video(work_dir, "vid1", [row("clip_0001")])

    response = client.patch(
        "/videos/vid1/clips",
        json={"decisions": [{"clip_id": "clip_0001", "keep": False}]},
    )

    assert response.status_code == 200
    assert response.json() == {"updated": 1}
    clips = client.get("/videos/vid1/clips").json()["clips"]
    assert clips[0]["keep"] is False


def test_patch_clips_allows_repeated_keep_toggling(client, work_dir):
    """SM-4: a reviewer changing their mind about a clip must never be
    blocked -- only a speaker_label reassignment is guarded (see below)."""
    build_video(work_dir, "vid1", [row("clip_0001", speaker_label="SPEAKER_00")])

    for keep in (False, True, False, True):
        response = client.patch(
            "/videos/vid1/clips",
            json={"decisions": [{"clip_id": "clip_0001", "keep": keep}]},
        )
        assert response.status_code == 200

    clips = client.get("/videos/vid1/clips").json()["clips"]
    assert clips[0]["keep"] is True


def test_patch_clips_rejects_reassigning_an_already_labelled_clip(client, work_dir):
    """A clip's speaker_label decides which character's dataset it lands in
    (see commit_reviewed_clips). Silently moving it once it is already
    labelled is the cross-character corruption this story guards against."""
    build_video(work_dir, "vid1", [row("clip_0001", speaker_label="SPEAKER_00")])

    response = client.patch(
        "/videos/vid1/clips",
        json={"decisions": [{"clip_id": "clip_0001", "speaker_label": "SPEAKER_01"}]},
    )

    assert response.status_code == 409
    clips = client.get("/videos/vid1/clips").json()["clips"]
    assert clips[0]["speaker_label"] == "SPEAKER_00"


def test_patch_clips_allows_resubmitting_the_same_label(client, work_dir):
    build_video(work_dir, "vid1", [row("clip_0001", speaker_label="SPEAKER_00")])

    response = client.patch(
        "/videos/vid1/clips",
        json={"decisions": [{"clip_id": "clip_0001", "speaker_label": "SPEAKER_00"}]},
    )

    assert response.status_code == 200


def test_patch_clips_allows_labelling_a_previously_unlabelled_clip(client, work_dir):
    build_video(work_dir, "vid1", [row("clip_0001", speaker_label="")])

    response = client.patch(
        "/videos/vid1/clips",
        json={"decisions": [{"clip_id": "clip_0001", "speaker_label": "SPEAKER_00"}]},
    )

    assert response.status_code == 200
    clips = client.get("/videos/vid1/clips").json()["clips"]
    assert clips[0]["speaker_label"] == "SPEAKER_00"


def test_patch_clips_rejects_conflicting_labels_within_the_same_request(
    client, work_dir
):
    """_reassigns_a_recorded_label only checks one decision against the row's
    persisted state, so two decisions for the same clip in one payload must
    be caught separately -- otherwise the last one silently wins."""
    build_video(work_dir, "vid1", [row("clip_0001", speaker_label="")])

    response = client.patch(
        "/videos/vid1/clips",
        json={
            "decisions": [
                {"clip_id": "clip_0001", "speaker_label": "SPEAKER_00"},
                {"clip_id": "clip_0001", "speaker_label": "SPEAKER_01"},
            ]
        },
    )

    assert response.status_code == 409
    clips = client.get("/videos/vid1/clips").json()["clips"]
    assert clips[0]["speaker_label"] is None


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


def test_get_clip_audio_streams_the_shared_clip(client, work_dir):
    build_video(work_dir, "vid1", [row("clip_0001")])

    response = client.get("/videos/vid1/clips/clip_0001/audio")

    assert response.status_code == 200
    assert response.content == b"RIFF-not-real-audio"


def test_get_clip_audio_404s_for_an_unknown_clip(client, work_dir):
    build_video(work_dir, "vid1", [row("clip_0001")])

    response = client.get("/videos/vid1/clips/clip_9999/audio")

    assert response.status_code == 404


def test_list_videos_reports_diarization_and_review_status(client, work_dir):
    build_video(work_dir, "vid1", [row("clip_0001"), row("clip_0002")])
    (work_dir / "youtube" / "vid1" / DIARIZATION_NAME).write_text("{}")
    # ingested but not yet reviewed
    (work_dir / "youtube" / "vid2" / "clips").mkdir(parents=True)

    response = client.get("/videos")

    assert response.status_code == 200
    videos = {video["video_id"]: video for video in response.json()["videos"]}
    assert videos["vid1"]["diarized"] is True
    assert videos["vid1"]["reviewed"] is True
    assert videos["vid1"]["clip_count"] == 2
    assert videos["vid2"]["diarized"] is False
    assert videos["vid2"]["reviewed"] is False
    assert videos["vid2"]["clip_count"] == 0


def test_list_videos_is_empty_before_anything_is_ingested(client, work_dir):
    response = client.get("/videos")

    assert response.status_code == 200
    assert response.json() == {"videos": []}


def test_get_video_speakers_groups_by_label_and_counts_clips(client, work_dir):
    build_video(
        work_dir,
        "vid1",
        [
            row("clip_0001", speaker_label="SPEAKER_00"),
            row("clip_0002", speaker_label="SPEAKER_00", keep="0"),
            row("clip_0003", speaker_label="SPEAKER_01"),
            row("clip_0004", speaker_label=""),
        ],
    )

    response = client.get("/videos/vid1/speakers")

    assert response.status_code == 200
    speakers = response.json()["speakers"]
    assert [speaker["speaker_label"] for speaker in speakers] == [
        "SPEAKER_00",
        "SPEAKER_01",
        None,
    ]
    assert speakers[0]["clip_count"] == 2
    assert speakers[0]["kept_count"] == 1
    assert speakers[2]["clip_count"] == 1


def test_get_video_speakers_404s_for_an_unknown_video(client, work_dir):
    response = client.get("/videos/nope/speakers")

    assert response.status_code == 404


def test_a_video_ingested_once_serves_every_character_that_claims_it(client, work_dir):
    """The whole point of the move: no character ever appears in these URLs,
    so the same ingested video answers identically for every character that
    claims it, and nothing here ever re-downloads or re-diarizes it."""
    build_video(work_dir, "vid1", [row("clip_0001", speaker_label="SPEAKER_00")])
    write_speaker_map(
        work_dir / "youtube" / "vid1", {"SPEAKER_00": "janeway", "SPEAKER_01": None}
    )

    for _ in range(2):  # janeway's dashboard, then chakotay's -- same call
        response = client.get("/videos/vid1/clips")
        assert response.status_code == 200
        assert response.json()["clips"][0]["clip_id"] == "clip_0001"


def test_build_command_needs_no_character_for_youtube_ingest_stages():
    """FR12/decoupling: starting an ingest job for a video needs no
    character, since the artifacts it writes are not scoped to one.

    Calls _build_command directly rather than POST /jobs, which would spawn a
    real main.py subprocess -- this is a pure function, no process needed.
    """
    command = api._build_command(
        api.JobRequest(stage="youtube-download", youtube_url="https://example.com/v")
    )

    assert "--stage" in command
    assert command[command.index("--stage") + 1] == "youtube-download"


def test_build_command_still_needs_a_character_for_youtube_commit():
    with pytest.raises(api.HTTPException) as exc_info:
        api._build_command(api.JobRequest(stage="youtube-commit"))

    assert exc_info.value.status_code == 422
