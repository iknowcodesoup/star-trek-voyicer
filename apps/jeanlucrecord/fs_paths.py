"""Filesystem layout shared by every router.

Owns WORK_DIR as its source of truth. Every other module reads
fs_paths.WORK_DIR at call time instead of importing the value directly, so
tests/test_api.py's work_dir fixture -- which monkeypatches this module's
attribute -- reaches every code path that touches the filesystem, including
ones several routers away from api.py itself.
"""

import re
from pathlib import Path

from fastapi import HTTPException, status

from review import REVIEW_CSV_NAME, REVIEW_FIELDS, read_review_csv
from services.log_parsing import _as_float
from youtube_ingest import DIARIZATION_NAME

APP_DIR = Path(__file__).resolve().parent
WORK_DIR = APP_DIR / "work"
JOB_STATE_PATH = WORK_DIR / "_jobs.json"

# character and video ids reach the filesystem, so keep them to characters that
# cannot escape work/ -- no dots, no separators
SAFE_NAME = re.compile(r"^[A-Za-z0-9_-]+$")


def _check_name(value: str | None, label: str) -> str:
    if not value or not SAFE_NAME.match(value):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            f"{label} must match {SAFE_NAME.pattern}",
        )
    return value


def _video_dir(video_id: str) -> Path:
    # video artifacts are shared across every character, so one video id names
    # one directory regardless of who ingested it or who claims it next
    _check_name(video_id, "video_id")
    video_dir = WORK_DIR / "youtube" / video_id
    if not video_dir.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No ingested video {video_id}")
    return video_dir


def _dataset_dir_for(character: str) -> Path:
    # mirrors main.py's dataset_dir_for. Duplicated rather than imported,
    # because main.py loads chatterbox-tts, whisper, and piper_phonemize at
    # module scope, and api.py must stay importable with none of them.
    return WORK_DIR / character / "dataset"


def _review_path(video_id: str) -> Path:
    review_path = _video_dir(video_id) / REVIEW_CSV_NAME
    if not review_path.exists():
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, f"No review.csv for {video_id}, ingest first"
        )
    return review_path


def _video_summary(video_dir: Path) -> dict:
    review_path = video_dir / REVIEW_CSV_NAME
    clip_count = len(read_review_csv(review_path)) if review_path.exists() else 0
    return {
        "video_id": video_dir.name,
        "diarized": (video_dir / DIARIZATION_NAME).exists(),
        "reviewed": review_path.exists(),
        "clip_count": clip_count,
    }


def _clip_from_row(row: dict) -> dict:
    return {
        "clip_id": row["clip_id"],
        "keep": row["keep"] == "1",
        "quality_score": _as_float(row.get("quality_score")),
        "flagged": row.get("flagged") == "1",
        "speaker_label": row.get("speaker_label") or None,
        "speaker_coverage": _as_float(row.get("speaker_coverage")),
        "duration_sec": _as_float(row.get("duration_sec")),
        "start_sec": _as_float(row.get("start_sec")),
        "end_sec": _as_float(row.get("end_sec")),
        "text": row.get("text", ""),
    }


def _fill_missing_fields(row: dict) -> dict:
    # a review.csv written before diarization has no speaker columns; DictWriter
    # needs every field present
    return {field: row.get(field, "") for field in REVIEW_FIELDS}
