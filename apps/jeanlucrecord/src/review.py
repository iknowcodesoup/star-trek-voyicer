import csv
import shutil
from pathlib import Path

REVIEW_FIELDS = ["clip_id", "keep", "quality_score", "flagged", "duration_sec", "start_sec", "end_sec", "text"]


def write_review_csv(path: Path, rows: list[dict]) -> None:
    """Comma-delimited, header row, Excel-openable -- deliberately different
    from the pipe-delimited LJSpeech metadata.csv. Sorted ascending by
    quality_score (worst/noisiest first) so manual attention goes where
    it's most needed."""
    ordered = sorted(rows, key=lambda r: r["quality_score"])
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REVIEW_FIELDS)
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


def commit_reviewed_clips(youtube_dir: Path, out_dir: Path) -> tuple[int, int]:
    """Merge keep=1 rows from every work/<character>/youtube/<video_id>/review.csv
    into out_dir/{wavs,metadata.csv}, skipping rows already recorded in that
    video's committed.csv ledger. Returns (newly_committed, already_committed)."""
    wav_dir = out_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = out_dir / "metadata.csv"

    newly_committed = 0
    already_committed = 0

    with open(metadata_path, "a", encoding="utf-8") as metadata_file:
        for video_dir in sorted(youtube_dir.glob("*")):
            review_path = video_dir / "review.csv"
            if not review_path.exists():
                continue
            video_id = video_dir.name
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

                    dataset_id = f"yt_{video_id}_{clip_id}"
                    shutil.copy(video_dir / "clips" / f"{clip_id}.wav", wav_dir / f"{dataset_id}.wav")
                    metadata_file.write(f"{dataset_id}|{row['text']}\n")
                    metadata_file.flush()
                    committed_file.write(f"{clip_id}|{dataset_id}\n")
                    committed_file.flush()
                    newly_committed += 1

    return newly_committed, already_committed
