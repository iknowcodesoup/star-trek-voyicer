"""Tests for the video-scoped routes in routes/videos.py."""

from pathlib import Path

from voice_factory.core.youtube_ingest import DIARIZATION_NAME
from voice_factory.repositories.review_csv_repository import write_review_csv
from voice_factory.repositories.video_meta_repository import write_video_meta_file


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


def build_video(
    work_dir: Path, video_id: str, rows: list[dict], meta: dict | None = None
) -> Path:
    """One ingested video. meta stays optional, because a video ingested
    before meta.json existed must keep working with no backfill."""
    video_dir = work_dir / "youtube" / video_id
    clips_dir = video_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    for clip_row in rows:
        (clips_dir / f"{clip_row['clip_id']}.wav").write_bytes(b"RIFF-not-real-audio")
    write_review_csv(video_dir / "review.csv", rows)
    if meta is not None:
        write_video_meta_file(video_dir, meta)
    return video_dir


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


def test_patch_video_renames_it_and_returns_the_video(client, work_dir):
    build_video(
        work_dir,
        "vid1",
        [row("clip_0001")],
        meta={"title": "Auto title", "channel": "Voyager", "url": "http://y/vid1"},
    )

    response = client.patch("/videos/vid1", json={"title": "Corrected by hand"})

    assert response.status_code == 200
    assert response.json()["title"] == "Corrected by hand"
    assert client.get("/videos").json()["videos"][0]["title"] == "Corrected by hand"


def test_patch_video_keeps_the_fields_it_was_not_given(client, work_dir):
    """A rename merges over meta.json. url, channel and ingested_at are
    yt-dlp's answers and dropping them would cost a re-ingest to recover."""
    build_video(
        work_dir,
        "vid1",
        [row("clip_0001")],
        meta={"title": "Auto", "channel": "Voyager", "url": "http://y/vid1"},
    )

    client.patch("/videos/vid1", json={"title": "Renamed"})

    video = client.get("/videos").json()["videos"][0]
    assert video["channel"] == "Voyager"
    assert video["url"] == "http://y/vid1"


def test_patch_video_names_a_video_that_had_no_meta(client, work_dir):
    """video_summary falls back to the id when meta.json is absent, so a
    rename is also how an old video gets a name for the first time."""
    build_video(work_dir, "vid1", [row("clip_0001")])

    response = client.patch("/videos/vid1", json={"title": "Named at last"})

    assert response.status_code == 200
    assert response.json()["title"] == "Named at last"


def test_patch_video_reports_404_for_an_unknown_video(client, work_dir):
    response = client.patch("/videos/vid_missing", json={"title": "Nope"})

    assert response.status_code == 404


def test_patch_video_rejects_a_blank_title(client, work_dir):
    """A blank name would hide the video in every list."""
    build_video(work_dir, "vid1", [row("clip_0001")])

    assert client.patch("/videos/vid1", json={"title": "   "}).status_code == 422
    assert client.patch("/videos/vid1", json={"title": ""}).status_code == 422


def test_patch_clips_writes_through_to_review_csv(client, work_dir):
    build_video(work_dir, "vid1", [row("clip_0001")])

    response = client.patch(
        "/videos/vid1/clips",
        json={"decisions": [{"clip_id": "clip_0001", "keep": False}]},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["updated"] == 1
    # the new state comes back with the write, so a caller never has to ask
    assert body["clips"][0]["clip_id"] == "clip_0001"
    assert body["clips"][0]["keep"] is False
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
    """reassigns_a_recorded_label only checks one decision against the row's
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
    assert set(videos["vid1"]) == {
        "video_id",
        "diarized",
        "reviewed",
        "clip_count",
        "title",
        "url",
        "duration_sec",
        "channel",
        "ingested_at",
    }
    assert videos["vid1"]["diarized"] is True
    assert videos["vid1"]["reviewed"] is True
    assert videos["vid1"]["clip_count"] == 2
    assert videos["vid2"]["diarized"] is False
    assert videos["vid2"]["reviewed"] is False
    assert videos["vid2"]["clip_count"] == 0


def test_list_videos_names_a_video_from_its_meta_json(client, work_dir):
    """The factory owns the title, so every character that claims this video
    reads the same name from the same file."""
    build_video(
        work_dir,
        "vid1",
        [row("clip_0001")],
        meta={
            "video_id": "vid1",
            "title": "The Best of Both Worlds",
            "url": "https://www.youtube.com/watch?v=vid1",
            "duration_sec": 2730.0,
            "channel": "Star Trek",
            "ingested_at": "2026-08-12T19:42:00+00:00",
        },
    )

    video = client.get("/videos").json()["videos"][0]

    assert video["title"] == "The Best of Both Worlds"
    assert video["url"] == "https://www.youtube.com/watch?v=vid1"
    assert video["duration_sec"] == 2730.0
    assert video["channel"] == "Star Trek"
    assert video["ingested_at"] == "2026-08-12T19:42:00+00:00"


def test_list_videos_falls_back_to_the_video_id_without_meta_json(client, work_dir):
    """A video ingested before meta.json existed must keep working, so an
    absent file gives null fields and the id stands in for the title."""
    build_video(work_dir, "vid1", [row("clip_0001")])

    video = client.get("/videos").json()["videos"][0]

    assert video["title"] == "vid1"
    assert video["url"] is None
    assert video["duration_sec"] is None
    assert video["channel"] is None
    assert video["ingested_at"] is None


def test_list_videos_is_empty_before_anything_is_ingested(client, work_dir):
    """A WORK_DIR with no youtube/ under it is a fresh install, not a fault.
    Contrast with the missing-WORK_DIR test below."""
    response = client.get("/videos")

    assert response.status_code == 200
    assert response.json() == {"videos": []}


def test_list_videos_500s_when_work_dir_is_missing(client, missing_work_dir):
    """The 2026-08-20 outage: a bad WORK_DIR answered 200 with an empty list,
    and the dashboard rendered its own stale copy over the top of it."""
    response = client.get("/videos")

    assert response.status_code == 500
    assert str(missing_work_dir) in response.json()["detail"]


def test_get_characters_500s_when_work_dir_is_missing(client, missing_work_dir):
    response = client.get("/characters")

    assert response.status_code == 500
    assert str(missing_work_dir) in response.json()["detail"]


def test_health_still_answers_when_work_dir_is_missing(client, missing_work_dir):
    """The route that reports WORK_DIR is the one that must not raise, or
    there is no way to see what the path resolved to."""
    response = client.get("/health")

    assert response.status_code == 200


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
    from voice_factory.core.review_workflow import write_speaker_map

    build_video(work_dir, "vid1", [row("clip_0001", speaker_label="SPEAKER_00")])
    write_speaker_map(
        work_dir / "youtube" / "vid1", {"SPEAKER_00": "janeway", "SPEAKER_01": None}
    )

    for _ in range(2):  # janeway's dashboard, then chakotay's -- same call
        response = client.get("/videos/vid1/clips")
        assert response.status_code == 200
        assert response.json()["clips"][0]["clip_id"] == "clip_0001"
