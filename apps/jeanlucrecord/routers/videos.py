"""Video-scoped routes: list ingested videos, their speakers and clips.

None of these take a character. A video is ingested once and shared across
every character that later claims it -- see /videos/{video_id}/speakers and
the clip routes below, all keyed on video_id alone.
"""

import json

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import FileResponse

import fs_paths
from models import ClipDecision, ClipDecisionRequest
from review import SPEAKER_MAP_FILENAME, read_review_csv, write_review_csv

router = APIRouter(tags=["Videos"])


@router.get("/videos")
async def get_videos() -> dict:
    """Every ingested video, independent of any character.

    Lets the dashboard offer a video for a second character without asking
    the factory to ingest it again -- see /videos/{video_id}/speakers and the
    four clip routes below, none of which take a character either.
    """
    youtube_dir = fs_paths.WORK_DIR / "youtube"
    if not youtube_dir.exists():
        return {"videos": []}
    videos = [
        fs_paths._video_summary(video_dir)
        for video_dir in sorted(youtube_dir.iterdir())
        if video_dir.is_dir()
    ]
    return {"videos": videos}


@router.get("/videos/{video_id}/speakers")
async def get_video_speakers(video_id: str) -> dict:
    rows = read_review_csv(fs_paths._review_path(video_id))
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
    rows = read_review_csv(fs_paths._review_path(video_id))
    map_path = fs_paths._video_dir(video_id) / SPEAKER_MAP_FILENAME
    speaker_map = (
        json.loads(map_path.read_text(encoding="utf-8")) if map_path.exists() else {}
    )
    return {
        "video_id": video_id,
        "speaker_map": speaker_map,
        "clips": [fs_paths._clip_from_row(row) for row in rows],
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
    """
    review_path = fs_paths._review_path(video_id)
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
        and _reassigns_a_recorded_label(
            by_clip_id[decision.clip_id], decision.speaker_label
        )
    }
    within_request = _conflicting_labels_within_request(decisions_request.decisions)
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

    write_review_csv(review_path, [fs_paths._fill_missing_fields(row) for row in rows])
    return {"updated": len(decisions_request.decisions)}


def _reassigns_a_recorded_label(row: dict, new_speaker_label: str) -> bool:
    # empty means never labelled (undiarized, or diarization left it blank),
    # so the first label a decision gives it is never a reassignment
    current = row.get("speaker_label") or ""
    return bool(current) and current != new_speaker_label


def _conflicting_labels_within_request(decisions: list[ClipDecision]) -> set[str]:
    """Clip ids that two decisions in the same request disagree about.

    _reassigns_a_recorded_label only ever sees one decision against the row's
    persisted state, so two decisions for the same clip_id in one payload --
    e.g. an unlabelled clip -- both pass that check independently and the
    last one applied would silently win. This catches that within the
    request itself, before anything is written.
    """
    labels_by_clip: dict[str, set[str]] = {}
    for decision in decisions:
        if decision.speaker_label is not None:
            labels_by_clip.setdefault(decision.clip_id, set()).add(
                decision.speaker_label
            )
    return {clip_id for clip_id, labels in labels_by_clip.items() if len(labels) > 1}


@router.get("/videos/{video_id}/clips/{clip_id}/audio")
async def get_clip_audio(video_id: str, clip_id: str) -> FileResponse:
    fs_paths._check_name(clip_id, "clip_id")
    clip_path = fs_paths._video_dir(video_id) / "clips" / f"{clip_id}.wav"
    if not clip_path.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No clip {clip_id}")
    return FileResponse(clip_path, media_type="audio/wav")
