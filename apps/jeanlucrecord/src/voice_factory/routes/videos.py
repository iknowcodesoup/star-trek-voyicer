"""Video-scoped routes: list ingested videos, their speakers and clips.

None of these take a character. A video is ingested once and shared across
every character that later claims it -- see /videos/{video_id}/speakers and
the clip routes below, all keyed on video_id alone.
"""

import json

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse

from voice_factory.core import clip_review
from voice_factory.infrastructure.filesystem_layout import (
    check_name,
    require_work_dir,
)
from voice_factory.repositories.review_csv_repository import (
    read_review_csv,
    write_review_csv,
)
from voice_factory.repositories.speaker_map_repository import (
    SPEAKER_MAP_FILENAME,
)
from voice_factory.repositories.video_meta_repository import (
    read_video_meta,
    write_video_meta_file,
)
from voice_factory.schemas import ClipDecisionRequest, VideoRenameRequest

router = APIRouter(tags=["Videos"])


@router.get("/videos")
async def get_videos() -> dict:
    """Every ingested video, independent of any character.

    Lets the dashboard offer a video for a second character without asking
    the factory to ingest it again -- see /videos/{video_id}/speakers and the
    four clip routes below, none of which take a character either.
    """
    # An absent WORK_DIR is a broken deployment and raises. An absent
    # youtube/ under a WORK_DIR that is there is a fresh install, which
    # really has no videos yet.
    youtube_dir = require_work_dir() / "youtube"
    if not youtube_dir.exists():
        return {"videos": []}
    videos = [
        clip_review.video_summary(video_dir)
        for video_dir in sorted(youtube_dir.iterdir())
        if video_dir.is_dir()
    ]
    return {"videos": videos}


@router.patch("/videos/{video_id}")
async def patch_video(video_id: str, rename_request: VideoRenameRequest) -> dict:
    """Rename a video.

    The title belongs to the video, not to any run that claims it, so a rename
    is visible to every character that shares it - the same reason meta.json
    sits beside the clips rather than in a caller's database.

    Merged over the existing meta, never written on top of it: url, channel,
    and ingested_at are yt-dlp's answers and a rename must not drop them.
    ensure_video_meta already refuses to overwrite "a title a person corrected
    by hand", and this is the route that does the correcting.
    """
    video_directory = clip_review.video_dir(video_id)
    if not video_directory.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No video {video_id}")

    meta = read_video_meta(video_directory)
    write_video_meta_file(
        video_directory, {**meta, "title": rename_request.title}
    )
    return clip_review.video_summary(video_directory)


@router.get("/videos/{video_id}/speakers")
async def get_video_speakers(video_id: str) -> dict:
    rows = read_review_csv(clip_review.review_path(video_id))
    grouped: dict[str | None, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row.get("speaker_label") or None, []).append(row)

    speakers = [
        {
            "speaker_label": speaker_label,
            "clip_count": len(group),
            "kept_count": sum(1 for row in group if row["keep"] == "1"),
        }
        # None (undiarized/rejected) sorts last, same order the dashboard
        # already expects from the run-scoped speaker board
        for speaker_label, group in sorted(
            grouped.items(), key=lambda item: (item[0] is None, item[0] or "")
        )
    ]
    return {"video_id": video_id, "speakers": speakers}


@router.get("/videos/{video_id}/clips")
async def get_clips(video_id: str) -> dict:
    rows = read_review_csv(clip_review.review_path(video_id))
    map_path = clip_review.video_dir(video_id) / SPEAKER_MAP_FILENAME
    speaker_map = (
        json.loads(map_path.read_text(encoding="utf-8")) if map_path.exists() else {}
    )
    return {
        "video_id": video_id,
        "speaker_map": speaker_map,
        "clips": [clip_review.clip_from_row(row) for row in rows],
    }


@router.patch("/videos/{video_id}/clips")
async def patch_clips(video_id: str, decisions_request: ClipDecisionRequest) -> dict:
    """Apply keep/speaker-label decisions to review.csv.

    review.csv is shared now: once a video has more than one claimant, two
    characters' runs can both reach the same clip. Re-keeping or rejecting a
    clip is always safe -- it never changes which character the clip belongs
    to -- so `keep` is applied unconditionally, exactly as before this story.
    Reassigning `speaker_label` is different: that is how a clip moves from
    one character's dataset to another's, so silently overwriting an
    already-recorded label with a different one is exactly the
    cross-character corruption this story guards against. A conflicting
    reassignment is rejected with 409 instead of applied. Full multi-claimant
    routing is Story 2.2's job -- this is the narrow stopgap.

    `assigned_voice` is neither of those. It is the reviewer's per-clip answer
    to "who is this for", it is expected to change as they work, and it never
    conflicts: one clip carries one assignment and only this route writes it.
    """
    review_path = clip_review.review_path(video_id)
    rows = read_review_csv(review_path)
    by_clip_id = {row["clip_id"]: row for row in rows}

    unknown = [
        decision.clip_id
        for decision in decisions_request.decisions
        if decision.clip_id not in by_clip_id
    ]
    if unknown:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, f"Unknown clip ids: {', '.join(unknown)}"
        )

    against_recorded = {
        decision.clip_id
        for decision in decisions_request.decisions
        if decision.speaker_label is not None
        and clip_review.reassigns_a_recorded_label(
            by_clip_id[decision.clip_id], decision.speaker_label
        )
    }
    within_request = clip_review.conflicting_labels_within_request(
        decisions_request.decisions
    )
    conflicts = sorted(against_recorded | within_request)
    if conflicts:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "Clip(s) already carry a different recorded speaker_label, and a "
            "shared video must not silently move them to another character's "
            f"dataset: {', '.join(conflicts)}. Update the video's speaker map "
            "instead, or resubmit without reassigning these clips.",
        )

    for decision in decisions_request.decisions:
        row = by_clip_id[decision.clip_id]
        if decision.keep is not None:
            row["keep"] = "1" if decision.keep else "0"
        if decision.speaker_label is not None:
            row["speaker_label"] = decision.speaker_label
        if decision.assigned_voice is not None:
            row["assigned_voice"] = decision.assigned_voice
        if decision.text is not None:
            row["text"] = decision.text

    write_review_csv(
        review_path, [clip_review.fill_missing_fields(row) for row in rows]
    )
    # The clips as they now stand, not just how many changed: a caller that
    # edited them needs the new state, and a count makes it go and ask again
    # for what this call already knows.
    return {
        "video_id": video_id,
        "updated": len(decisions_request.decisions),
        "clips": [
            clip_review.clip_from_row(by_clip_id[decision.clip_id])
            for decision in decisions_request.decisions
        ],
    }


@router.get("/videos/{video_id}/clips/{clip_id}/audio")
async def get_clip_audio(video_id: str, clip_id: str) -> FileResponse:
    check_name(clip_id, "clip_id")
    clip_path = clip_review.video_dir(video_id) / "clips" / f"{clip_id}.wav"
    if not clip_path.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No clip {clip_id}")
    return FileResponse(clip_path, media_type="audio/wav")
