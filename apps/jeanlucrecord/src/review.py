import csv
import json
import shutil
from collections.abc import Callable
from contextlib import ExitStack
from pathlib import Path
from typing import NamedTuple, TextIO

SPEAKER_MAP_FILENAME = "speaker_map.json"
REVIEW_CSV_NAME = "review.csv"

# speaker_label and speaker_coverage are only filled in when ingest ran with
# --diarize. csv.DictReader returns None for them on a review.csv written before
# diarization existed, so old files still commit.
REVIEW_FIELDS = [
    "clip_id",
    "keep",
    "quality_score",
    "flagged",
    "speaker_label",
    "speaker_coverage",
    "duration_sec",
    "start_sec",
    "end_sec",
    "text",
]


class CommitResult(NamedTuple):
    newly_committed: int
    already_committed: int
    # how many clips each dataset directory gained -- callers need this to mark
    # every character that received clips, not just the primary one
    committed_by_target: dict[Path, int]


def write_review_csv(path: Path, rows: list[dict]) -> None:
    """Comma-delimited, header row, Excel-openable -- deliberately different
    from the pipe-delimited LJSpeech metadata.csv. Sorted ascending by
    quality_score (worst/noisiest first) so manual attention goes where
    it's most needed."""
    # float(), not the raw value: rows read back from an existing review.csv
    # carry quality_score as a string, which would sort lexicographically
    ordered = sorted(rows, key=lambda r: float(r["quality_score"]))
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REVIEW_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ordered)


def read_review_csv(path: Path) -> list[dict]:
    with open(path, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


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
    out_dir: Path,
    dataset_dir_for: Callable[[str], Path] | None = None,
) -> CommitResult:
    """Merge keep=1 rows from every work/<character>/youtube/<video_id>/review.csv
    into a dataset directory, skipping rows already recorded in that video's
    committed.csv ledger.

    dataset_dir_for resolves a character name to that character's dataset
    directory. Supply it to honour each video's speaker_map.json, so one diarized
    video can seed several characters at once. None sends every clip to out_dir,
    which is the behaviour from before diarization existed.

    The map is read per video, because SPEAKER_00 in one video is not the same
    person as SPEAKER_00 in the next. A labelled clip whose speaker is absent
    from its video's map stays uncommitted, so a later run with a corrected map
    can still pick it up.
    """
    wav_dir = out_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)

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
            committed_path = video_dir / "committed.csv"
            committed = load_committed(committed_path)
            speaker_targets = load_speaker_targets(video_dir, dataset_dir_for)

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
    speaker_map = json.loads(map_path.read_text(encoding="utf-8"))
    return {
        speaker_label: dataset_dir_for(character)
        for speaker_label, character in speaker_map.items()
        if character
    }


def write_speaker_map(video_dir: Path, speaker_map: dict[str, str | None]) -> Path:
    map_path = video_dir / SPEAKER_MAP_FILENAME
    map_path.parent.mkdir(parents=True, exist_ok=True)
    map_path.write_text(json.dumps(speaker_map, indent=2), encoding="utf-8")
    return map_path


def _resolve_target(
    row: dict,
    out_dir: Path,
    speaker_targets: dict[str, Path] | None,
) -> Path | None:
    if not speaker_targets:
        return out_dir
    speaker_label = row.get("speaker_label")
    if not speaker_label:
        # an undiarized row in an otherwise mapped video -- the primary
        # character is the only sensible destination
        return out_dir
    return speaker_targets.get(speaker_label)
