"""Video-scoped routes: list ingested videos, their speakers and clips.

None of these take a character. A video is ingested once and shared across
every character that later claims it -- see /videos/{video_id}/speakers and
the clip routes below, all keyed on video_id alone.
"""

import asyncio
import io
import json
import shutil

import soundfile as sf
from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import FileResponse, Response

from voice_factory.core import clip_review
from voice_factory.core.audio_slicing import TARGET_RATE, read_slice
from voice_factory.core.quality import clip_quality_score
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

# Mirrors plan_clips's own defaults (core/youtube_ingest.py). A trim has no
# access to whatever --min/--max-clip-duration the original ingest ran with,
# so re-deriving excluded_reason here uses the same defaults every ingest
# uses unless overridden -- the same trade-off RETAIN_FLOOR_SEC and
# RETAIN_CEILING_SEC already make as fixed constants rather than per-video
# settings.
MIN_CLIP_DURATION_SEC = 1.0
MAX_CLIP_DURATION_SEC = 30.0

MAX_PAD_SEC = 10.0


def _length_excluded_reason(duration_sec: float) -> str:
    if duration_sec < MIN_CLIP_DURATION_SEC:
        return "too_short"
    if duration_sec > MAX_CLIP_DURATION_SEC:
        return "too_long"
    return ""


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


@router.delete("/videos/{video_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_video(video_id: str) -> None:
    """Remove one video's directory: its audio, clips, and review.csv.

    Irreversible - there is no trash. clip_review.video_dir already checks the
    id is filesystem-safe and 404s when the video does not exist, so this is
    the delete counterpart of every route above that reads through it.
    """
    video_directory = clip_review.video_dir(video_id)
    shutil.rmtree(video_directory)


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

    `start_sec`/`end_sec` is the trim bar's write. Trimming does not touch
    `keep` -- the two are orthogonal, a reviewer's choice either way. Both
    bounds must be given together; end_sec <= start_sec or either being
    negative is 422. A bounds change recomputes duration_sec, re-derives the
    length-based excluded_reason (so extending a too_short clip clears it),
    and rescores quality_score from the new slice -- a stale score would
    misreport the flag badge and the sort in review.csv.
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

    for decision in decisions_request.decisions:
        if decision.start_sec is None and decision.end_sec is None:
            continue
        if decision.start_sec is None or decision.end_sec is None:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                f"Clip {decision.clip_id}: start_sec and end_sec must be given together.",
            )
        if decision.start_sec < 0 or decision.end_sec < 0:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                f"Clip {decision.clip_id}: start_sec and end_sec must not be negative.",
            )
        if decision.end_sec <= decision.start_sec:
            raise HTTPException(
                status.HTTP_422_UNPROCESSABLE_ENTITY,
                f"Clip {decision.clip_id}: end_sec must be greater than start_sec.",
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

    video_directory = clip_review.video_dir(video_id)
    full_wav = video_directory / "full.wav"

    for decision in decisions_request.decisions:
        row = by_clip_id[decision.clip_id]
        if decision.keep is not None:
            row["keep"] = {"kept": "1", "excluded": "0", "none": ""}[decision.keep]
        if decision.speaker_label is not None:
            row["speaker_label"] = decision.speaker_label
        if decision.assigned_voice is not None:
            row["assigned_voice"] = decision.assigned_voice
        if decision.text is not None:
            row["text"] = decision.text
        if decision.start_sec is not None and decision.end_sec is not None:
            row["start_sec"] = decision.start_sec
            row["end_sec"] = decision.end_sec
            row["duration_sec"] = decision.end_sec - decision.start_sec
            row["excluded_reason"] = _length_excluded_reason(
                decision.end_sec - decision.start_sec
            )
            if full_wav.exists():
                samples = await asyncio.to_thread(
                    read_slice, full_wav, decision.start_sec, decision.end_sec
                )
                row["quality_score"] = clip_quality_score(samples)

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
async def get_clip_audio(
    video_id: str,
    clip_id: str,
    pad_sec: float = Query(0.0, ge=0.0, le=MAX_PAD_SEC),
) -> Response:
    """Stream one clip's audio, sliced on the fly from full.wav.

    full.wav wins over a pre-cut clips/{id}.wav whenever both exist -- a
    reviewer's trim only ever changes review.csv's bounds, and the old cut
    would silently outlive it otherwise (see review_workflow's commit
    precedence, which makes the same call). The pre-cut file is the fallback
    for a video an operator already reclaimed full.wav's disk space from,
    after review is long done.

    One route, not a second "with padding" endpoint: both would duplicate
    this same row lookup and precedence, and the pre-cut fallback has no
    audio outside its own bounds to honour a pad from anyway. No response
    header describes the granted window either -- that would need
    Access-Control-Expose-Headers on the orchestrator's CORS middleware, a
    silent-failure step. The client derives it instead:
    windowStart = max(0, startSec - padSec), windowEnd = windowStart + duration.
    """
    check_name(clip_id, "clip_id")
    video_directory = clip_review.video_dir(video_id)
    full_wav = video_directory / "full.wav"

    if full_wav.exists():
        rows = read_review_csv(clip_review.review_path(video_id))
        row = next((r for r in rows if r["clip_id"] == clip_id), None)
        if row is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"No clip {clip_id}")
        start_sec = float(row["start_sec"]) - pad_sec
        end_sec = float(row["end_sec"]) + pad_sec
        samples = read_slice(full_wav, start_sec, end_sec)
        buffer = io.BytesIO()
        sf.write(buffer, samples, TARGET_RATE, format="WAV", subtype="PCM_16")
        return Response(
            content=buffer.getvalue(),
            media_type="audio/wav",
            headers={"Accept-Ranges": "none"},
        )

    clip_path = video_directory / "clips" / f"{clip_id}.wav"
    if not clip_path.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No clip {clip_id}")
    return FileResponse(clip_path, media_type="audio/wav")
