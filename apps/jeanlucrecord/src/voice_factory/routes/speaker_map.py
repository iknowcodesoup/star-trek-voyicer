"""Speaker assignment routes: route a diarized video's speakers to characters.

Kept as its own small router, separate from videos.py, because "where does
speaker assignment happen in jeanlucrecord" should be a one-file answer.
"""

from fastapi import APIRouter, HTTPException, status

from voice_factory.core import clip_review
from voice_factory.core.review_workflow import (
    SpeakerMapConflict,
    write_speaker_map,
)
from voice_factory.infrastructure.filesystem_layout import check_name
from voice_factory.schemas import SpeakerMapRequest

router = APIRouter(tags=["Speaker Map"])


@router.put("/videos/{video_id}/speaker-map")
async def put_speaker_map(video_id: str, map_request: SpeakerMapRequest) -> dict:
    for target in map_request.speaker_map.values():
        if target is not None:
            check_name(target, "character")
    try:
        written = write_speaker_map(
            clip_review.video_dir(video_id), map_request.speaker_map
        )
    except SpeakerMapConflict as error:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "Speaker(s) already carry a different recorded assignment, and a "
            "shared video must not silently move them to another character's "
            f"dataset: {', '.join(error.conflicting_labels)}. Confirm the "
            "existing assignment instead, or resubmit without these speakers.",
        ) from error
    return {"speaker_map": map_request.speaker_map, "path": str(written)}
