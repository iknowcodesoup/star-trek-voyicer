"""Speaker assignment routes: route a diarized video's speakers to characters.

Kept as its own small router, separate from videos.py, because "where does
speaker assignment happen in jeanlucrecord" should be a one-file answer.
"""

import asyncio

from fastapi import APIRouter, HTTPException, status

from voice_factory.core import clip_review, speaker_assignment
from voice_factory.core.review_workflow import (
    SpeakerMapConflict,
    commit_reviewed_clips,
    write_speaker_map,
)
from voice_factory.infrastructure import filesystem_layout
from voice_factory.infrastructure.filesystem_layout import check_name
from voice_factory.schemas import CommitRequest, SpeakerMapRequest

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


@router.post("/videos/commit")
async def post_videos_commit(commit_request: CommitRequest) -> dict:
    """Write every named video's speaker-map entries, then run one commit
    pass across the whole shared work/youtube/ directory (FR14).

    stage_youtube_commit already fans one commit call out to every character
    any video's map names, so the only thing this route adds is a
    payload-driven way to write those maps for several videos at once,
    without naming a single "committing character". Every video is validated
    and checked for a conflicting speaker-map entry before anything is
    written, so a conflict anywhere in the payload leaves every video's map
    untouched -- the same all-or-nothing guarantee PUT /speaker-map gives one
    video at a time.
    """
    video_dirs = {
        video_id: clip_review.video_dir(video_id)
        for video_id in commit_request.assignments
    }
    for speaker_map in commit_request.assignments.values():
        for character in speaker_map.values():
            if character is not None:
                check_name(character, "character")

    conflicts = speaker_assignment.check_conflicts_across_videos(
        video_dirs, commit_request.assignments
    )
    if conflicts:
        detail = "; ".join(
            f"{video_id}: {', '.join(labels)}" for video_id, labels in conflicts.items()
        )
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "Speaker(s) already carry a different recorded assignment, and a "
            "shared video must not silently move them to another character's "
            f"dataset: {detail}. Confirm the existing assignment instead, or "
            "resubmit without these speakers.",
        )

    try:
        speaker_assignment.write_assignments(video_dirs, commit_request.assignments)
    except SpeakerMapConflict as error:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "Speaker(s) already carry a different recorded assignment, and a "
            "shared video must not silently move them to another character's "
            f"dataset: {', '.join(error.conflicting_labels)}. Confirm the "
            "existing assignment instead, or resubmit without these speakers.",
        ) from error

    # out_dir=None: a batched, multi-character call has no single "committing
    # character" to fall back to, so an unmapped or undiarized clip is left
    # uncommitted rather than guessed at (see commit_reviewed_clips).
    #
    # Run in a thread: commit now decodes and re-encodes audio per clip
    # across every video (see commit_reviewed_clips), which would otherwise
    # block the event loop for the whole call.
    result = await asyncio.to_thread(
        commit_reviewed_clips,
        filesystem_layout.WORK_DIR / "youtube",
        None,
        clip_review.dataset_dir_for,
    )
    committed = {
        target.parent.name: count
        for target, count in sorted(
            result.committed_by_target.items(), key=lambda item: item[0].parent.name
        )
    }
    return {"committed": committed}
