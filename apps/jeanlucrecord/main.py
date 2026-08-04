import argparse
import shutil
import subprocess
import sys
import threading
from pathlib import Path

# corpus text and Whisper transcripts can contain characters outside the
# Windows console's default cp1252 codepage
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from corpus import load_corpus, load_validation_sentences  # noqa: E402
from generate_dataset import generate_dataset  # noqa: E402
from import_dataset import import_dataset  # noqa: E402
from quality import FLAG_THRESHOLD_DB, clip_quality_score, is_flagged  # noqa: E402
from resample import normalize_ref_wav, resample_dir  # noqa: E402
from review import commit_reviewed_clips, write_review_csv  # noqa: E402
from youtube_ingest import (
    chunk_clips,
    download_audio,
    resolve_video_id,
    transcribe,
)  # noqa: E402

APP_DIR = Path(__file__).resolve().parent
DOCKER_IMAGE = "jeanlucrecord-trainer"

CORPUS_SIZE_DEFAULT = 1300
BASE_CHECKPOINT = "checkpoints/ljspeech-2000.ckpt"
MAX_EPOCHS = 3000
# 12 -> 14: docs/dataloader-perf-spec.md profiling showed training is GPU-compute-bound
# (DataLoader wait was <1% of total time), so a bigger batch amortizes per-step
# forward/backward overhead over more samples. Kept to a smaller increment than the
# usual 16 since headroom was thin (~1GB free of 8151MiB) at batch size 12 --
# watch nvidia-smi memory.used on the next run regardless.
BATCH_SIZE = 14
# every_n_epochs for piper_train's ModelCheckpoint. 1 (checkpoint every epoch) gave
# the finest-grained crash recovery but, combined with CHECKPOINT_KEEP below,
# meant the retained checkpoints were consecutive epochs -- nearly identical
# in quality, useless for picking a meaningfully different one by ear. 20 spaces the
# retained window out to 200 epochs of real training progress; a crash now loses at
# most 19 epochs instead of <1, which is the trade-off for that.
CHECKPOINT_EPOCHS = 20
# Dockerfile's ModelCheckpoint is save_top_k=-1 (keeps every checkpoint -- Lightning
# can't rank "top k" without a monitored metric, see Dockerfile), so retention is
# bounded here instead: a background thread prunes down to the most recent
# CHECKPOINT_KEEP while training runs, so a long run doesn't fill the disk
# (~1GB/checkpoint here, saved every CHECKPOINT_EPOCHS epochs).
CHECKPOINT_KEEP = 10
VALIDATION_SENTENCES = 8

STAGES = [
    "all",
    "dataset",
    "resample",
    "preprocess",
    "smoketest",
    "train",
    "export",
    "sample",
    "import",
    "youtube-ingest",
    "youtube-commit",
]


def run_docker(*args: str) -> None:
    # calling docker directly (no intermediate shell script) sidesteps host bash
    # path-translation issues -- "bash" on PATH here resolves to the WSL launcher,
    # which doesn't understand MSYS-style paths like docker/run.sh would need
    subprocess.run(
        [
            "docker",
            "build",
            "-t",
            DOCKER_IMAGE,
            "-f",
            str(APP_DIR / "Dockerfile"),
            str(APP_DIR),
        ],
        check=True,
    )
    subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--gpus",
            "all",
            # num_workers=8 DataLoader workers hand batches back to the main process
            # over /dev/shm; Docker's 64MB default is enough for num_workers=1 but
            # overflows with 8, killing workers with a "bus error" mid-epoch
            "--shm-size",
            "8g",
            "-v",
            f"{APP_DIR.as_posix()}:/app",
            "-w",
            "/app",
            DOCKER_IMAGE,
            *args,
        ],
        check=True,
    )


def external_source_marker(character: str) -> Path:
    return APP_DIR / "work" / character / "dataset" / ".external_source"


def mark_external_source(character: str, note: str) -> None:
    # tells stage_dataset (Chatterbox synthesis) that this dataset was populated
    # from real audio (import/youtube-commit), not the corpus, so a later
    # --stage all/dataset shouldn't try to extend it with synthetic clips
    marker = external_source_marker(character)
    marker.parent.mkdir(parents=True, exist_ok=True)
    with open(marker, "a", encoding="utf-8") as f:
        f.write(f"{note}\n")


def stage_dataset(character: str, corpus_size: int, retry_failed: bool = False) -> None:
    # generate_dataset() resumes internally from work/<character>/dataset/metadata.csv
    # and failed.csv, so it's always safe (and cheap when already complete) to call
    # this again
    marker = external_source_marker(character)
    if marker.exists():
        print(
            f"work/{character}/dataset/ was populated from real audio (see {marker}), "
            f"skipping Chatterbox synthesis. Delete {marker.name} in that folder to force "
            f"synthesis on top of it."
        )
        return
    out_dir = APP_DIR / "work" / character / "dataset"
    raw_ref_wavs = sorted((APP_DIR / "samples" / character).glob("*.wav"))
    if not raw_ref_wavs:
        raise SystemExit(f"No reference wavs found in samples/{character}/")
    ref_dir = APP_DIR / "work" / character / "ref"
    ref_wavs = [normalize_ref_wav(p, ref_dir / p.name) for p in raw_ref_wavs]
    phrases = load_corpus(corpus_size)
    generate_dataset(character, ref_wavs, phrases, out_dir, retry_failed=retry_failed)


def stage_import(character: str, import_dir: Path) -> None:
    # alternative to stage_dataset: bring an already-clipped-and-transcribed
    # dataset (e.g. samples/cena/, id|text metadata.csv + matching wavs) straight
    # into work/<character>/dataset/, skipping Chatterbox synthesis entirely.
    # import_dataset() is idempotent, so rerunning after adding more rows to the
    # source folder only imports what's new.
    if not import_dir.exists():
        raise SystemExit(f"Import directory not found: {import_dir}")
    if not (import_dir / "metadata.csv").exists():
        raise SystemExit(f"No metadata.csv found in {import_dir}")
    out_dir = APP_DIR / "work" / character / "dataset"
    import_dataset(import_dir, out_dir)
    mark_external_source(character, f"imported from {import_dir}")


def stage_youtube_ingest(
    character: str,
    url: str,
    whisper_model: str,
    min_duration: float,
    max_duration: float,
    quality_flag_threshold: float,
) -> None:
    # writes candidate clips + a review.csv for manual accept/reject -- nothing
    # here touches work/<character>/dataset/ directly, see stage_youtube_commit.
    video_id = resolve_video_id(url)
    video_dir = APP_DIR / "work" / character / "youtube" / video_id
    review_path = video_dir / "review.csv"
    if review_path.exists():
        print(f"{url} already ingested, review at {review_path}")
        return

    full_wav = download_audio(url, video_dir / "full.wav")
    segments = transcribe(full_wav, whisper_model)
    if not segments:
        raise SystemExit(f"No speech segments found in {url}")

    clips = chunk_clips(
        full_wav, segments, video_dir / "clips", min_duration, max_duration
    )
    if not clips:
        print(
            f"No clips survived duration filtering for {url} -- adjust --min/--max-clip-duration and retry."
        )
        return

    rows = []
    for clip in clips:
        score = clip_quality_score(video_dir / "clips" / f"{clip['clip_id']}.wav")
        flagged = is_flagged(score, quality_flag_threshold)
        rows.append(
            {
                "clip_id": clip["clip_id"],
                "keep": "0" if flagged else "1",
                "quality_score": round(score, 2),
                "flagged": int(flagged),
                "duration_sec": round(clip["duration"], 2),
                "start_sec": round(clip["start"], 2),
                "end_sec": round(clip["end"], 2),
                "text": clip["text"],
            }
        )
    write_review_csv(review_path, rows)

    flagged_count = sum(r["flagged"] for r in rows)
    print(f"\n{len(rows)} clip(s) ready for review at {review_path}")
    print(
        f"{flagged_count} flagged as likely low quality (keep=0 by default, worst-scoring first)."
    )
    print(
        f"Listen to clips under {video_dir / 'clips'}, edit the 'keep' column, then run:"
    )
    print(f"  uv run python main.py {character} --stage youtube-commit")


def stage_youtube_commit(character: str) -> None:
    youtube_dir = APP_DIR / "work" / character / "youtube"
    if not youtube_dir.exists():
        raise SystemExit(
            f"No ingested YouTube videos found for {character} under {youtube_dir}"
        )
    out_dir = APP_DIR / "work" / character / "dataset"
    newly_committed, already_committed = commit_reviewed_clips(youtube_dir, out_dir)
    if newly_committed:
        mark_external_source(
            character, f"{newly_committed} clip(s) committed from {youtube_dir}"
        )
    print(
        f"Committed {newly_committed} new clip(s), {already_committed} already committed."
    )


def stage_resample(character: str) -> None:
    # cheap CPU-only step (no TTS/Whisper) -- always rerun so it reflects whatever
    # the dataset stage currently has, rather than risk mirroring a stale dataset
    dataset_dir = APP_DIR / "work" / character / "dataset"
    resampled_dir = APP_DIR / "work" / character / "resampled"
    resampled_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(dataset_dir / "metadata.csv", resampled_dir / "metadata.csv")
    resample_dir(dataset_dir / "wavs", resampled_dir / "wavs")


def stage_preprocess(character: str) -> None:
    training_dir = APP_DIR / "work" / character / "training"
    if (training_dir / "config.json").exists():
        print(f"Preprocessed training data already exists at {training_dir}, skipping.")
        return
    run_docker(
        "python3",
        "-m",
        "piper_train.preprocess",
        "--language",
        "en-us",
        "--input-dir",
        f"work/{character}/resampled",
        "--output-dir",
        f"work/{character}/training",
        "--dataset-format",
        "ljspeech",
        "--single-speaker",
        "--sample-rate",
        "22050",
    )


def stage_smoketest() -> None:
    run_docker(
        "python3",
        "-c",
        "import torch; "
        "print('cuda available:', torch.cuda.is_available()); "
        "print('capability:', torch.cuda.get_device_capability(0)); "
        "print(torch.randn(4, 4, device='cuda') @ torch.randn(4, 4, device='cuda'))",
    )


def find_all_checkpoints(character: str) -> list[Path]:
    training_dir = APP_DIR / "work" / character / "training"
    return sorted(training_dir.glob("**/*.ckpt"), key=lambda p: p.stat().st_mtime)


def find_latest_checkpoint(character: str) -> str | None:
    checkpoints = find_all_checkpoints(character)
    if not checkpoints:
        return None
    return checkpoints[-1].relative_to(APP_DIR).as_posix()


def prune_checkpoints(character: str, keep: int) -> None:
    for stale in find_all_checkpoints(character)[:-keep]:
        stale.unlink(missing_ok=True)


def stage_train(character: str) -> None:
    # resume fine-tuning from this character's own last checkpoint if a previous
    # training run got partway through and crashed -- otherwise every rerun would
    # silently restart from the base LJSpeech checkpoint and lose that progress
    checkpoint = find_latest_checkpoint(character) or BASE_CHECKPOINT
    print(f"Resuming training from checkpoint: {checkpoint}")
    # piper_train's ModelCheckpoint saves every checkpoint (Dockerfile's
    # save_top_k=-1) since Lightning can't bound retention itself without a
    # monitored metric -- prune on this side instead, concurrently with training,
    # so checkpoints don't pile up over a 3000-epoch run.
    stop_pruning = threading.Event()

    def prune_loop() -> None:
        while not stop_pruning.wait(60):
            prune_checkpoints(character, CHECKPOINT_KEEP)

    pruner = threading.Thread(target=prune_loop, daemon=True)
    pruner.start()
    try:
        run_docker(
            "python3",
            "-m",
            "piper_train",
            "--dataset-dir",
            f"work/{character}/training",
            "--accelerator",
            "gpu",
            "--devices",
            "1",
            "--batch-size",
            str(BATCH_SIZE),
            "--validation-split",
            "0.0",
            "--num-test-examples",
            "0",
            "--max_epochs",
            str(MAX_EPOCHS),
            "--resume_from_checkpoint",
            checkpoint,
            "--checkpoint-epochs",
            str(CHECKPOINT_EPOCHS),
            "--quality",
            "high",
            # bf16 breaks training: piper_train's mel-spectrogram step calls torch.stft
            # (cuFFT) outside any autocast(enabled=False) block, and cuFFT has no bf16
            # kernel at all -- fails on the very first batch with
            # "RuntimeError: cuFFT doesn't support tensor of type: BFloat16"
            "--precision",
            "32",
        )
    finally:
        stop_pruning.set()
        pruner.join()
        prune_checkpoints(character, CHECKPOINT_KEEP)


def stage_export(character: str, checkpoint: str | None = None) -> None:
    # checkpoint lets you export a specific retained checkpoint (e.g. one picked by
    # ear from stage_sample's output) instead of always the most recent by mtime --
    # a later epoch isn't guaranteed to sound better.
    run_docker("bash", "docker/export.sh", character, checkpoint or "")
    print_handoff(character)


def stage_sample(character: str, num_sentences: int) -> None:
    # Exports every currently-retained checkpoint (see CHECKPOINT_KEEP above)
    # to its own ONNX model and synthesizes the same fixed held-out sentences
    # (corpus.load_validation_sentences) from each, so they can be compared by ear
    # under work/<character>/checkpoint_samples/<checkpoint-name>/*.wav before
    # deciding which checkpoint to hand to stage_export.
    if not find_all_checkpoints(character):
        raise SystemExit(
            f"No checkpoints found for {character} under work/{character}/training/"
        )
    sentences_path = APP_DIR / "work" / character / "validation_sentences.txt"
    sentences_path.write_text(
        "\n".join(load_validation_sentences(num_sentences)) + "\n", encoding="utf-8"
    )
    run_docker("bash", "docker/sample_checkpoints.sh", character)


def print_handoff(character: str) -> None:
    upper = character.upper()
    print(f"""
Copy these files:
  copy output\\{character}.onnx      ..\\janewav\\src\\models\\{character}.onnx
  copy output\\{character}.onnx.json ..\\janewav\\src\\models\\{character}.onnx.json

Then in apps/janewav/.env, set in MODELS:
  "{upper}": "/models/{character}.onnx"
""")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a fine-tuned Piper voice model for a character."
    )
    parser.add_argument(
        "character", help="Character name, matching samples/<character>/"
    )
    parser.add_argument("--corpus-size", type=int, default=CORPUS_SIZE_DEFAULT)
    parser.add_argument("--stage", choices=STAGES, default="all")
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry phrases previously recorded in failed.csv instead of skipping them "
        "(e.g. after tuning exaggeration or swapping the reference wav).",
    )
    parser.add_argument(
        "--checkpoint",
        help="For --stage export: path to a specific .ckpt to export (default: most "
        "recent by mtime). Use after --stage sample to export whichever checkpoint "
        "sounded best rather than assuming the latest one is.",
    )
    parser.add_argument(
        "--num-validation-sentences",
        type=int,
        default=VALIDATION_SENTENCES,
        help=f"For --stage sample: how many held-out sentences to synthesize per "
        f"checkpoint (default: {VALIDATION_SENTENCES}).",
    )
    parser.add_argument(
        "--import-dir",
        help="For --stage import: folder with metadata.csv (id|text) + matching wavs "
        "(flat, or in wavs/ or wav/) to import directly into work/<character>/dataset/, "
        "skipping Chatterbox synthesis (e.g. samples/cena).",
    )
    parser.add_argument(
        "--youtube-url",
        help="For --stage youtube-ingest: video URL to download, transcribe, and chunk "
        "into candidate clips for manual review.",
    )
    parser.add_argument(
        "--whisper-model",
        default="medium",
        help="For --stage youtube-ingest: faster-whisper model size "
        "(tiny/base/small/medium/large-v3, default: medium).",
    )
    parser.add_argument(
        "--min-clip-duration",
        type=float,
        default=1.0,
        help="For --stage youtube-ingest: drop transcript segments shorter than this "
        "many seconds (default: 1.0).",
    )
    parser.add_argument(
        "--max-clip-duration",
        type=float,
        default=30.0,
        help="For --stage youtube-ingest: drop transcript segments longer than this "
        "many seconds (default: 30.0).",
    )
    parser.add_argument(
        "--quality-flag-threshold",
        type=float,
        default=FLAG_THRESHOLD_DB,
        help="For --stage youtube-ingest: clips scoring below this default to keep=0 "
        f"in review.csv (default: {FLAG_THRESHOLD_DB}).",
    )
    args = parser.parse_args()

    if args.stage == "import" and not args.import_dir:
        parser.error("--stage import requires --import-dir")
    if args.stage == "youtube-ingest" and not args.youtube_url:
        parser.error("--stage youtube-ingest requires --youtube-url")

    if args.stage in ("all", "dataset"):
        stage_dataset(args.character, args.corpus_size, args.retry_failed)
    if args.stage == "import":
        stage_import(args.character, Path(args.import_dir))
    if args.stage == "youtube-ingest":
        stage_youtube_ingest(
            args.character,
            args.youtube_url,
            args.whisper_model,
            args.min_clip_duration,
            args.max_clip_duration,
            args.quality_flag_threshold,
        )
    if args.stage == "youtube-commit":
        stage_youtube_commit(args.character)
    if args.stage in ("all", "resample"):
        stage_resample(args.character)
    if args.stage in ("all", "preprocess"):
        stage_preprocess(args.character)
    if args.stage == "smoketest":
        stage_smoketest()
    if args.stage in ("all", "train"):
        stage_train(args.character)
    if args.stage == "sample":
        stage_sample(args.character, args.num_validation_sentences)
    if args.stage in ("all", "export"):
        stage_export(args.character, args.checkpoint)


if __name__ == "__main__":
    main()
