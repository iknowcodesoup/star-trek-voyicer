"""Business-flavored filesystem helpers for reviewing ingested video clips.

Split out of the old flat fs_paths.py: these functions know CSV field names
and raise domain-specific 404s, unlike infrastructure/filesystem_layout.py's
pure path/layout facts. Routes call into this module directly instead of
reaching past core into raw path helpers, keeping the
routes -> core -> repositories -> infrastructure layering direction.
"""

from pathlib import Path

from fastapi import HTTPException, status

from voice_factory.core.training_log_reader import parse_optional_float
from voice_factory.infrastructure import filesystem_layout
from voice_factory.infrastructure.filesystem_layout import check_name
from voice_factory.repositories.review_csv_repository import (
    REVIEW_CSV_NAME,
    REVIEW_FIELDS,
    read_review_csv,
)
from voice_factory.repositories.video_meta_repository import read_video_meta
from voice_factory.schemas import ClipDecision


def video_dir(video_id: str) -> Path:
    # video artifacts are shared across every character, so one video id names
    # one directory regardless of who ingested it or who claims it next
    check_name(video_id, "video_id")
    directory = filesystem_layout.WORK_DIR / "youtube" / video_id
    if not directory.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No ingested video {video_id}")
    return directory


def dataset_dir_for(character: str) -> Path:
    # mirrors cli.py's dataset_dir_for. Duplicated rather than imported,
    # because cli.py loads chatterbox-tts, whisper, and piper_phonemize at
    # module scope, and app.py must stay importable with none of them.
    return filesystem_layout.WORK_DIR / character / "dataset"


def review_path(video_id: str) -> Path:
    path = video_dir(video_id) / REVIEW_CSV_NAME
    if not path.exists():
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, f"No review.csv for {video_id}, ingest first"
        )
    return path


def video_summary(video_directory: Path) -> dict:
    # deferred import: youtube_ingest depends on nothing here, but importing
    # it at module scope would make every clip_review caller pay for its
    # subprocess-tooling constants too
    from voice_factory.core.youtube_ingest import DIARIZATION_NAME

    path = video_directory / REVIEW_CSV_NAME
    clip_count = len(read_review_csv(path)) if path.exists() else 0
    video_id = video_directory.name
    # a video ingested before meta.json existed has no title, and must keep
    # working with no backfill -- the id is the name until someone re-ingests
    meta = read_video_meta(video_directory)
    return {
        "video_id": video_id,
        "diarized": (video_directory / DIARIZATION_NAME).exists(),
        "reviewed": path.exists(),
        "clip_count": clip_count,
        "title": meta.get("title") or video_id,
        "url": meta.get("url"),
        "duration_sec": meta.get("duration_sec"),
        "channel": meta.get("channel"),
        "thumbnail_url": meta.get("thumbnail_url"),
        "ingested_at": meta.get("ingested_at"),
    }


def clip_from_row(row: dict) -> dict:
    return {
        "clip_id": row["clip_id"],
        "keep": row["keep"] == "1",
        "quality_score": parse_optional_float(row.get("quality_score")),
        "flagged": row.get("flagged") == "1",
        "speaker_label": row.get("speaker_label") or None,
        "speaker_coverage": parse_optional_float(row.get("speaker_coverage")),
        "assigned_voice": row.get("assigned_voice") or None,
        "duration_sec": parse_optional_float(row.get("duration_sec")),
        "start_sec": parse_optional_float(row.get("start_sec")),
        "end_sec": parse_optional_float(row.get("end_sec")),
        "text": row.get("text", ""),
        "excluded_reason": row.get("excluded_reason") or "",
    }


def fill_missing_fields(row: dict) -> dict:
    # a review.csv written before diarization has no speaker columns; DictWriter
    # needs every field present
    return {field: row.get(field, "") for field in REVIEW_FIELDS}


def reassigns_a_recorded_label(row: dict, new_speaker_label: str) -> bool:
    # empty means never labelled (undiarized, or diarization left it blank),
    # so the first label a decision gives it is never a reassignment
    current = row.get("speaker_label") or ""
    return bool(current) and current != new_speaker_label


def conflicting_labels_within_request(decisions: list[ClipDecision]) -> set[str]:
    """Clip ids that two decisions in the same request disagree about.

    reassigns_a_recorded_label only ever sees one decision against the row's
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
