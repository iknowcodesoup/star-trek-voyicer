"""Domain rules for committing reviewed clips and merging speaker maps.

Calls into repositories/review_csv_repository.py and
repositories/speaker_map_repository.py for the actual file I/O; owns the
conflict-detection and commit-routing rules on top of it.
"""

import shutil
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path
from typing import NamedTuple, TextIO

from voice_factory.repositories.review_csv_repository import (
    REVIEW_CSV_NAME,
    read_review_csv,
)
from voice_factory.repositories.speaker_map_repository import (
    SPEAKER_MAP_FILENAME,
    read_speaker_map,
    write_speaker_map_file,
)


class CommitResult(NamedTuple):
    newly_committed: int
    already_committed: int
    # how many clips each dataset directory gained -- callers need this to mark
    # every character that received clips, not just the primary one
    committed_by_target: dict[Path, int]


def load_committed(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    committed = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        clip_id, dataset_id = line.split("|", 1)
        committed[clip_id] = dataset_id
    return committed


def commit_reviewed_clips(
    youtube_dir: Path,
    out_dir: Path | None = None,
    dataset_dir_for: Callable[[str], Path] | None = None,
) -> CommitResult:
    """Merge keep=1 rows from every work/<character>/youtube/<video_id>/review.csv
    into a dataset directory, skipping rows already recorded in that video's
    committed.csv ledger.

    dataset_dir_for resolves a character name to that character's dataset
    directory. Supply it to honour each video's speaker_map.json, so one diarized
    video can seed several characters at once. None sends every clip to out_dir,
    which is the behaviour from before diarization existed.

    out_dir is the fallback destination for a clip with no single mapped
    character: an unmapped video (dataset_dir_for is None), an undiarized row,
    or a speaker absent from its video's map. Pass None to skip those clips
    instead of guessing -- the batched, multi-character commit route has no
    one "committing character" to fall back to, so a guess there would risk
    routing a clip to the wrong dataset.

    The map is read per video, because SPEAKER_00 in one video is not the same
    person as SPEAKER_00 in the next. A labelled clip whose speaker is absent
    from its video's map stays uncommitted, so a later run with a corrected map
    can still pick it up.
    """
    if out_dir is not None:
        (out_dir / "wavs").mkdir(parents=True, exist_ok=True)

    newly_committed = 0
    already_committed = 0
    committed_by_target: dict[Path, int] = {}

    with ExitStack() as open_files:
        metadata_files: dict[Path, TextIO] = {}

        def metadata_file_for(target: Path) -> TextIO:
            if target not in metadata_files:
                (target / "wavs").mkdir(parents=True, exist_ok=True)
                metadata_files[target] = open_files.enter_context(
                    open(target / "metadata.csv", "a", encoding="utf-8")
                )
            return metadata_files[target]

        for video_dir in sorted(youtube_dir.glob("*")):
            review_path = video_dir / REVIEW_CSV_NAME
            if not review_path.exists():
                continue
            video_id = video_dir.name
            speaker_targets = load_speaker_targets(video_dir, dataset_dir_for)

            # dataset_dir_for is only ever omitted by tests exercising the
            # no-routing case. Every real caller (stage_youtube_commit) passes
            # it, and youtube_dir is shared across every character since this
            # story, so a video with no speaker_map.json here is unclaimed --
            # not "assume it's mine" the way a character-scoped scan used to
            # read it. Skipping before touching committed.csv is what keeps
            # this commit run from marking someone else's clips as handled.
            if dataset_dir_for is not None and speaker_targets is None:
                continue

            committed_path = video_dir / "committed.csv"
            committed = load_committed(committed_path)

            with open(committed_path, "a", encoding="utf-8") as committed_file:
                for row in read_review_csv(review_path):
                    clip_id = row["clip_id"]
                    if clip_id in committed:
                        already_committed += 1
                        continue
                    if row["keep"] != "1":
                        continue

                    target = _resolve_target(row, out_dir, speaker_targets)
                    if target is None:
                        continue

                    dataset_id = f"yt_{video_id}_{clip_id}"
                    # opens the metadata file and creates target/wavs, so it has
                    # to happen before the copy below
                    metadata_file = metadata_file_for(target)
                    shutil.copy(
                        video_dir / "clips" / f"{clip_id}.wav",
                        target / "wavs" / f"{dataset_id}.wav",
                    )
                    metadata_file.write(f"{dataset_id}|{row['text']}\n")
                    metadata_file.flush()
                    committed_file.write(f"{clip_id}|{dataset_id}\n")
                    committed_file.flush()
                    newly_committed += 1
                    committed_by_target[target] = committed_by_target.get(target, 0) + 1

    return CommitResult(newly_committed, already_committed, committed_by_target)


def load_speaker_targets(
    video_dir: Path,
    dataset_dir_for: Callable[[str], Path] | None,
) -> dict[str, Path] | None:
    """Read video_dir/speaker_map.json into {speaker_label: dataset_dir}.

    Returns None when there is no map to apply, which sends every clip to the
    default destination. A speaker mapped to null is deliberately dropped: that
    is how the reviewer says "discard this voice".
    """
    if dataset_dir_for is None:
        return None
    map_path = video_dir / SPEAKER_MAP_FILENAME
    if not map_path.exists():
        return None
    speaker_map = read_speaker_map(video_dir)
    return {
        speaker_label: dataset_dir_for(character)
        for speaker_label, character in speaker_map.items()
        if character
    }


class SpeakerMapConflict(Exception):
    """A merge would silently overwrite an already-recorded, different
    assignment for one or more speaker labels.

    Same corruption class patch_clips's 409 already guards a clip's
    speaker_label against, one layer up: a video's speaker_map.json can now
    carry assignments from more than one character's claim, so silently
    replacing an earlier claim's non-null value is never safe. app.py turns
    this into a 409 instead of applying the write.
    """

    def __init__(self, conflicting_labels: list[str]) -> None:
        self.conflicting_labels = conflicting_labels
        super().__init__(
            "speaker(s) already carry a different recorded assignment: "
            + ", ".join(conflicting_labels)
        )


def speaker_map_conflicts(
    existing: dict[str, str | None], speaker_map: dict[str, str | None]
) -> list[str]:
    """Speaker labels merging speaker_map into an already-loaded existing map
    would raise on, without writing anything.

    Pure, no I/O: takes the existing map a caller already read, rather than
    reading it again itself. That is what lets a caller which must write
    several videos' maps in one request -- the batched multi-character commit
    route -- check every video for a conflict, and then write it, off the
    same single read, instead of two reads that could see a different file if
    something else writes to it in between.
    """
    return sorted(
        speaker_label
        for speaker_label, current in existing.items()
        if current is not None
        and speaker_label in speaker_map
        and speaker_map[speaker_label] != current
    )


def write_speaker_map(video_dir: Path, speaker_map: dict[str, str | None]) -> Path:
    """Merge speaker_map into video_dir/speaker_map.json, not replace it.

    A video's map used to belong to one character for its whole lifetime, so
    overwriting it outright was safe. Now that a second character can claim
    the same shared video later, replacing the file would erase the first
    character's assignments -- this merges the new keys in instead, so an
    earlier claim's speaker->character routing survives a later one's.

    Raises SpeakerMapConflict, and writes nothing, when the request would
    change an existing non-null value for a speaker label to something else.
    A new label, or the same value resubmitted, always succeeds.
    """
    existing = read_speaker_map(video_dir)
    conflicts = speaker_map_conflicts(existing, speaker_map)
    if conflicts:
        raise SpeakerMapConflict(conflicts)
    merged = {**existing, **speaker_map}
    return write_speaker_map_file(video_dir, merged)


def _resolve_target(
    row: dict,
    out_dir: Path | None,
    speaker_targets: dict[str, Path] | None,
) -> Path | None:
    # None means no map at all -- the pre-diarization behaviour, send
    # everything to out_dir. An empty dict is different: it means a map
    # exists and every speaker in it was explicitly discarded (mapped to
    # null), so falling back to out_dir here would be the same unsafe
    # "assume it's mine" guess this story removes from the video-level scan
    # above, just one row at a time instead of one video at a time.
    #
    # Either way, out_dir itself may now be None: the batched, multi-character
    # commit route has no single "committing character" to fall back to, so it
    # passes None here on purpose. Returning None is exactly what the caller
    # already treats as "leave this row uncommitted".
    if speaker_targets is None:
        return out_dir
    speaker_label = row.get("speaker_label")
    if not speaker_label:
        # an undiarized row in an otherwise mapped video -- the primary
        # character is the only sensible destination, when there is one
        return out_dir
    return speaker_targets.get(speaker_label)
