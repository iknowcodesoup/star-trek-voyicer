"""Domain rules for building a character's dataset and merging speaker maps.

Calls into repositories/speaker_map_repository.py for the actual file I/O;
owns the conflict-detection and routing rules on top of it.

compile_dataset_for is the one path that builds training audio. It rebuilds
work/<character>/dataset/ from scratch on every call, out of whatever the
orchestrator currently says the voice is made of, so the folder is always
exactly the reviewer's live set of decisions. That is what removes the delete
paths an incremental merge would need: un-keeping a clip, un-assigning it, or
moving it to another voice all take effect by simply not being gathered on
the next compile.

The decisions themselves are not here any more. They live in the
orchestrator's Postgres, and this host owns the audio - see
infrastructure/orchestrator_gateway.py. So a compile is told which slices to
cut rather than scanning for them.
"""

import shutil
from pathlib import Path
from typing import NamedTuple

import soundfile as sf

from voice_factory.core.audio_slicing import write_slice_from
from voice_factory.repositories.speaker_map_repository import (
    read_speaker_map,
    write_speaker_map_file,
)


class CompileResult(NamedTuple):
    """What one compile pass wrote into work/<character>/dataset/."""

    clip_count: int
    # rows skipped because their bounds clamp to zero frames, or because the
    # video holds neither a full.wav to slice nor a pre-cut clip to copy. A
    # caller reports this: a compile that silently drops half a dataset and
    # still says "ok" is how a training run starts on the wrong audio.
    skipped_count: int


def clips_to_compile(
    youtube_dir: Path, dataset_clips: list[dict]
) -> list[tuple[Path, dict]]:
    """Pair each assigned clip with the video directory holding its audio.

    Returns (video_dir, clip) pairs in a stable order -- videos sorted by
    directory name, clips by start time -- so two compiles of the same
    decisions write the same metadata.csv byte for byte.

    A clip whose video directory is gone is left out. That is an operator who
    reclaimed the disk, not a corrupt dataset, and the caller reports it as a
    skip rather than failing the whole compile.
    """
    gathered: list[tuple[Path, dict]] = []
    for clip in dataset_clips:
        video_dir = youtube_dir / str(clip["video_id"])
        if not video_dir.exists():
            continue
        gathered.append((video_dir, clip))
    gathered.sort(key=lambda pair: (pair[0].name, float(pair[1]["start_sec"])))
    return gathered


def compile_dataset_for(
    youtube_dir: Path,
    dataset_dir: Path,
    dataset_clips: list[dict],
) -> CompileResult:
    """Rebuild dataset_dir from the clips the orchestrator says to use.

    dataset_clips arrives already filtered to kept and assigned, so this only
    turns decisions into audio. Passing it in rather than fetching it here is
    what keeps the file work testable with no HTTP in the way - and it is why
    the voice's name is no longer a parameter: the caller resolved the name
    into clips before getting here.

    Destructive by design: dataset_dir/wavs/ and dataset_dir/metadata.csv are
    replaced, not appended to. Anything the reviewer has since un-kept or
    reassigned disappears because it is not gathered, which is the whole
    reason this runs at training start rather than at assignment time.

    Written to a sibling directory and swapped in at the end, so a crash or a
    read error partway through leaves the previous dataset intact rather than
    a half-built one a training run would happily read.
    """
    rows = clips_to_compile(youtube_dir, dataset_clips)

    staging_dir = dataset_dir.with_name(dataset_dir.name + ".compiling")
    if staging_dir.exists():
        shutil.rmtree(staging_dir)
    (staging_dir / "wavs").mkdir(parents=True)

    clip_count = 0
    skipped_count = 0
    metadata_lines: list[str] = []

    # group by video so each video's full.wav opens once, not once per clip
    for video_dir in _ordered_video_dirs(rows):
        video_rows = [row for source_dir, row in rows if source_dir == video_dir]
        # full.wav wins over a stale pre-cut clips/{id}.wav: a trim only ever
        # changes review.csv's bounds, and the old cut would outlive it under
        # the reverse precedence
        full_wav = video_dir / "full.wav"
        source_reader = sf.SoundFile(full_wav) if full_wav.exists() else None
        try:
            for row in video_rows:
                written = _write_clip(
                    video_dir, row, staging_dir, source_reader
                )
                if written is None:
                    skipped_count += 1
                    continue
                metadata_lines.append(written)
                clip_count += 1
        finally:
            if source_reader is not None:
                source_reader.close()

    (staging_dir / "metadata.csv").write_text(
        "".join(metadata_lines), encoding="utf-8"
    )

    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    dataset_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir.replace(dataset_dir)
    return CompileResult(clip_count, skipped_count)


def _ordered_video_dirs(rows: list[tuple[Path, dict]]) -> list[Path]:
    """The video directories in rows, first-seen order, no duplicates."""
    ordered: list[Path] = []
    for video_dir, _row in rows:
        if video_dir not in ordered:
            ordered.append(video_dir)
    return ordered


def _write_clip(
    video_dir: Path,
    row: dict,
    staging_dir: Path,
    source_reader: sf.SoundFile | None,
) -> str | None:
    """Write one clip's wav into staging_dir. Returns its metadata.csv line,
    or None when the row produced no usable audio."""
    dataset_id = f"yt_{video_dir.name}_{row['clip_id']}"
    out_wav = staging_dir / "wavs" / f"{dataset_id}.wav"

    if source_reader is not None:
        frame_count = write_slice_from(
            source_reader,
            out_wav,
            float(row["start_sec"]),
            float(row["end_sec"]),
        )
        if frame_count == 0:
            # bounds clamp to zero frames -- a 0-frame wav fails downstream
            # preprocessing far from the cause, so drop the row instead
            out_wav.unlink()
            return None
    else:
        pre_cut_clip = video_dir / "clips" / f"{row['clip_id']}.wav"
        if not pre_cut_clip.exists():
            return None
        shutil.copy(pre_cut_clip, out_wav)

    return f"{dataset_id}|{row['text']}\n"


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
