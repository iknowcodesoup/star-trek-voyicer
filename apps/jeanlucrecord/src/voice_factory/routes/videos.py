"""Video-scoped routes: list ingested videos, their speakers and clips.

None of these take a character. A video is ingested once and shared across
every character that later claims it -- see /videos/{video_id}/speakers and
the clip routes below, all keyed on video_id alone.
"""

import io
import json
import shutil

import soundfile as sf
from fastapi import APIRouter, HTTPException, Query, status
from fastapi.responses import FileResponse, Response

from voice_factory.core import clip_review
from voice_factory.core.audio_slicing import TARGET_RATE, read_slice
from voice_factory.infrastructure.filesystem_layout import (
    check_name,
    require_work_dir,
)
from voice_factory.repositories.review_csv_repository import (
    read_review_csv,
)
from voice_factory.repositories.speaker_map_repository import (
    SPEAKER_MAP_FILENAME,
)
from voice_factory.repositories.video_meta_repository import (
    read_video_meta,
    write_video_meta_file,
)
from voice_factory.schemas import VideoRenameRequest

router = APIRouter(tags=["Videos"])

# Mirrors plan_clips's own defaults (core/youtube_ingest.py). A trim has no
# access to whatever --min/--max-clip-duration the original ingest ran with,
# so re-deriving excluded_reason here uses the same defaults every ingest
# uses unless overridden -- the same trade-off RETAIN_FLOOR_SEC and
# RETAIN_CEILING_SEC already make as fixed constants rather than per-video
# settings.
MIN_CLIP_DURATION_SEC = 1.0
MAX_CLIP_DURATION_SEC = 30.0

MAX_PAD_SEC = 300.0


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


# PATCH /videos/{video_id}/clips is gone. Keep, text and bounds are the
# orchestrator's now - it holds them in Postgres and is the only writer. A
# route here that still edited review.csv would be a second writer to a file
# nothing reads after ingest, and the two copies would part ways the first
# time anyone trimmed a clip.


@router.get("/videos/{video_id}/clips/{clip_id}/audio")
async def get_clip_audio(
    video_id: str,
    clip_id: str,
    pad_sec: float = Query(0.0, ge=0.0, le=MAX_PAD_SEC),
    bounds: str | None = Query(None, pattern=r"^\d+(\.\d+)?-\d+(\.\d+)?$"),
) -> Response:
    """Stream one clip's audio, sliced on the fly from full.wav.

    `bounds` is "start-end" in seconds, and it wins when given. The reviewer's
    trim lives in the orchestrator's database, not in review.csv here, so the
    caller sends the window it wants rather than this host looking up bounds
    it no longer maintains. Without it, the clip's ingest bounds are used --
    which is what review.csv still holds and what an untrimmed clip is.

    full.wav wins over a pre-cut clips/{id}.wav whenever both exist -- a
    reviewer's trim changes the bounds and never the file, and the old cut
    would silently outlive it otherwise. The pre-cut file is the fallback for
    a video an operator already reclaimed full.wav's disk space from, after
    review is long done.

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
        window = _requested_bounds(bounds) or _ingest_bounds(video_id, clip_id)
        if window is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"No clip {clip_id}")
        start_sec = window[0] - pad_sec
        end_sec = window[1] + pad_sec
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


@router.get("/videos/{video_id}/transcript_text")
async def get_transcript_text(
    video_id: str,
    start_sec: float = Query(..., ge=0.0),
    end_sec: float = Query(..., ge=0.0),
) -> dict:
    """The video's own transcript, joined over one time window.

    Backs the orchestrator's resize-fills-the-text behaviour: a clip's text
    tracks its transcript until a reviewer types over it, and this is what
    it tracks against. No transcript.json answers "" rather than 404 - a
    caller that gets nothing back leaves the clip's text exactly as it was.
    """
    text = clip_review.transcript_text_for_range(video_id, start_sec, end_sec)
    return {"text": text}


def _requested_bounds(bounds: str | None) -> tuple[float, float] | None:
    """The caller's window, or None when it named none.

    The pattern on the query parameter already rejects anything that is not
    two numbers, so this only has to split them.
    """
    if not bounds:
        return None
    start_text, end_text = bounds.split("-", 1)
    return float(start_text), float(end_text)


def _ingest_bounds(video_id: str, clip_id: str) -> tuple[float, float] | None:
    """Where ingest cut this clip, from review.csv.

    review.csv is the record of what ingest produced and nothing writes to it
    afterwards, so these bounds are the clip's original ones. A caller that
    wants a trimmed window sends it - see `bounds` above.
    """
    rows = read_review_csv(clip_review.review_path(video_id))
    row = next((r for r in rows if r["clip_id"] == clip_id), None)
    if row is None:
        return None
    return float(row["start_sec"]), float(row["end_sec"])
