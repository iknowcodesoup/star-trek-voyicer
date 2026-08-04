from pathlib import Path

from resample import normalize_ref_wav


def load_metadata(path: Path) -> list[tuple[str, str]]:
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        clip_id, text = line.split("|", 1)
        entries.append((clip_id, text))
    return entries


def find_source_wav(source_dir: Path, clip_id: str) -> Path | None:
    for candidate in (
        source_dir / f"{clip_id}.wav",
        source_dir / "wavs" / f"{clip_id}.wav",
        source_dir / "wav" / f"{clip_id}.wav",
    ):
        if candidate.exists():
            return candidate
    return None


def import_dataset(source_dir: Path, out_dir: Path) -> None:
    """Import an externally-prepared id|text metadata.csv + matching wavs
    (e.g. samples/cena/) directly into work/<character>/dataset/, skipping
    Chatterbox synthesis. Rows with no matching wav are dropped with a
    warning, not treated as a fatal error. Idempotent: ids already present
    in out_dir/metadata.csv are skipped without touching their wav."""
    source_metadata = source_dir / "metadata.csv"
    entries = load_metadata(source_metadata)

    wav_dir = out_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    dest_metadata = out_dir / "metadata.csv"
    existing = {clip_id for clip_id, _ in load_metadata(dest_metadata)} if dest_metadata.exists() else set()

    imported = 0
    dropped = 0
    with open(dest_metadata, "a", encoding="utf-8") as metadata_file:
        for clip_id, text in entries:
            if clip_id in existing:
                continue

            src_wav = find_source_wav(source_dir, clip_id)
            if src_wav is None:
                print(f"  WARNING: no wav for '{clip_id}', dropping row")
                dropped += 1
                continue

            normalize_ref_wav(src_wav, wav_dir / f"{clip_id}.wav")
            metadata_file.write(f"{clip_id}|{text}\n")
            metadata_file.flush()
            imported += 1

    print(f"Imported {imported} clip(s) from {source_dir}, dropped {dropped} (no matching wav).")
