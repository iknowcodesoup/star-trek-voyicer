"""Persistence for review.csv: comma-delimited clip review records.

Pure I/O, no domain rules -- see core/review_workflow.py for the
conflict/commit logic that reads and writes through these functions.
"""

import csv
from pathlib import Path

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
