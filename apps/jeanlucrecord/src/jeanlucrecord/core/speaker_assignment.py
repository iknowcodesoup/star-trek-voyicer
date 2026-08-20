"""Speaker assignment: routes a diarized video's speakers to characters.

Kept as its own small module, separate from clip_review.py, because "where
does speaker assignment happen in jeanlucrecord" should be a one-file answer.
Orchestrates review_workflow.py's per-video conflict/merge rules across a
batch of videos for routes/speaker_map.py's two routes.
"""

from pathlib import Path

from jeanlucrecord.core.review_workflow import (
    SpeakerMapConflict,
    read_speaker_map,
    speaker_map_conflicts,
    write_speaker_map,
)


def check_conflicts_across_videos(
    video_dirs: dict[str, Path],
    assignments: dict[str, dict[str, str | None]],
) -> dict[str, list[str]]:
    """Every video's conflicts against its own existing speaker_map.json.

    Reads each video's existing map once here, and the caller reuses that
    same snapshot for the merge in write_speaker_map -- two separate reads
    could each see a different file if something else writes to it in
    between, letting a conflict check pass against a snapshot the merge
    itself never actually used.
    """
    existing_maps = {
        video_id: read_speaker_map(video_dirs[video_id]) for video_id in assignments
    }
    return {
        video_id: video_conflicts
        for video_id, speaker_map in assignments.items()
        if (
            video_conflicts := speaker_map_conflicts(
                existing_maps[video_id], speaker_map
            )
        )
    }


def write_assignments(
    video_dirs: dict[str, Path],
    assignments: dict[str, dict[str, str | None]],
) -> None:
    """Write every named video's speaker-map entries.

    The pre-check in check_conflicts_across_videos only narrows the race
    window, it does not close it: a concurrent write between the check and
    here can still make write_speaker_map raise. Videos already written
    earlier in this same loop are not rolled back if a later one conflicts --
    full transactional rollback across videos is out of scope for this story.
    Propagates SpeakerMapConflict; the caller turns it into a 409.
    """
    for video_id, speaker_map in assignments.items():
        write_speaker_map(video_dirs[video_id], speaker_map)


__all__ = [
    "SpeakerMapConflict",
    "check_conflicts_across_videos",
    "write_assignments",
]
