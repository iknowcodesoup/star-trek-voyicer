"""Speaker assignment: routes a diarized video's speakers to characters.

Kept as its own small router, separate from videos.py, because "where does
speaker assignment happen in jeanlucrecord" should be a one-file answer.
"""

from fastapi import APIRouter, HTTPException, status

import fs_paths
from models import CommitRequest, SpeakerMapRequest
from review import (
    SpeakerMapConflict,
    commit_reviewed_clips,
    read_speaker_map,
    speaker_map_conflicts,
    write_speaker_map,
)

router = APIRouter(tags=["Speaker Map"])


@router.put("/videos/{video_id}/speaker-map")
async def put_speaker_map(video_id: str, map_request: SpeakerMapRequest) -> dict:
    for target in map_request.speaker_map.values():
        if target is not None:
            fs_paths._check_name(target, "character")
    try:
        written = write_speaker_map(
            fs_paths._video_dir(video_id), map_request.speaker_map
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
        video_id: fs_paths._video_dir(video_id)
        for video_id in commit_request.assignments
    }
    for speaker_map in commit_request.assignments.values():
        for character in speaker_map.values():
            if character is not None:
                fs_paths._check_name(character, "character")

    # read each video's existing map once here, and reuse that same snapshot
    # for both the conflict check and (below) the merge in write_speaker_map
    # -- two separate reads could each see a different file if something else
    # writes to it in between, letting a conflict check pass against a
    # snapshot the merge itself never actually used
    existing_maps = {
        video_id: read_speaker_map(video_dirs[video_id])
        for video_id in commit_request.assignments
    }
    conflicts: dict[str, list[str]] = {
        video_id: video_conflicts
        for video_id, speaker_map in commit_request.assignments.items()
        if (
            video_conflicts := speaker_map_conflicts(
                existing_maps[video_id], speaker_map
            )
        )
    }
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

    # the pre-check above only narrows the race window, it does not close it:
    # a concurrent write between the check and here can still make
    # write_speaker_map raise. Videos already written earlier in this same
    # loop are not rolled back if a later one conflicts -- full transactional
    # rollback across videos is out of scope for this story.
    try:
        for video_id, speaker_map in commit_request.assignments.items():
            write_speaker_map(video_dirs[video_id], speaker_map)
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
    result = commit_reviewed_clips(
        fs_paths.WORK_DIR / "youtube", None, fs_paths._dataset_dir_for
    )
    committed = {
        target.parent.name: count
        for target, count in sorted(
            result.committed_by_target.items(), key=lambda item: item[0].parent.name
        )
    }
    return {"committed": committed}
