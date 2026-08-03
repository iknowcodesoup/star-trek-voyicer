import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# corpus text and Whisper transcripts can contain characters outside the
# Windows console's default cp1252 codepage
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from corpus import load_corpus, load_validation_sentences  # noqa: E402
from generate_dataset import generate_dataset  # noqa: E402
from resample import normalize_ref_wav, resample_dir  # noqa: E402

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
# the finest-grained crash recovery but, combined with the Dockerfile's save_top_k=10,
# meant the 10 retained checkpoints were 10 *consecutive* epochs -- nearly identical
# in quality, useless for picking a meaningfully different one by ear. 20 spaces the
# retained window out to 200 epochs of real training progress; a crash now loses at
# most 19 epochs instead of <1, which is the trade-off for that.
CHECKPOINT_EPOCHS = 20
VALIDATION_SENTENCES = 8

STAGES = ["all", "dataset", "resample", "preprocess", "smoketest", "train", "export", "sample"]


def run_docker(*args: str) -> None:
    # calling docker directly (no intermediate shell script) sidesteps host bash
    # path-translation issues -- "bash" on PATH here resolves to the WSL launcher,
    # which doesn't understand MSYS-style paths like docker/run.sh would need
    subprocess.run(
        ["docker", "build", "-t", DOCKER_IMAGE, "-f", str(APP_DIR / "Dockerfile"), str(APP_DIR)],
        check=True,
    )
    subprocess.run(
        [
            "docker", "run", "--rm", "--gpus", "all",
            # num_workers=8 DataLoader workers hand batches back to the main process
            # over /dev/shm; Docker's 64MB default is enough for num_workers=1 but
            # overflows with 8, killing workers with a "bus error" mid-epoch
            "--shm-size", "8g",
            "-v", f"{APP_DIR.as_posix()}:/app", "-w", "/app",
            DOCKER_IMAGE, *args,
        ],
        check=True,
    )


def stage_dataset(character: str, corpus_size: int, retry_failed: bool = False) -> None:
    # generate_dataset() resumes internally from work/<character>/dataset/metadata.csv
    # and failed.csv, so it's always safe (and cheap when already complete) to call
    # this again
    out_dir = APP_DIR / "work" / character / "dataset"
    raw_ref_wavs = sorted((APP_DIR / "samples" / character).glob("*.wav"))
    if not raw_ref_wavs:
        raise SystemExit(f"No reference wavs found in samples/{character}/")
    ref_dir = APP_DIR / "work" / character / "ref"
    ref_wavs = [normalize_ref_wav(p, ref_dir / p.name) for p in raw_ref_wavs]
    phrases = load_corpus(corpus_size)
    generate_dataset(character, ref_wavs, phrases, out_dir, retry_failed=retry_failed)


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
        "python3", "-m", "piper_train.preprocess",
        "--language", "en-us",
        "--input-dir", f"work/{character}/resampled",
        "--output-dir", f"work/{character}/training",
        "--dataset-format", "ljspeech",
        "--single-speaker",
        "--sample-rate", "22050",
    )


def stage_smoketest() -> None:
    run_docker(
        "python3", "-c",
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


def stage_train(character: str) -> None:
    # resume fine-tuning from this character's own last checkpoint if a previous
    # training run got partway through and crashed -- otherwise every rerun would
    # silently restart from the base LJSpeech checkpoint and lose that progress
    checkpoint = find_latest_checkpoint(character) or BASE_CHECKPOINT
    print(f"Resuming training from checkpoint: {checkpoint}")
    run_docker(
        "python3", "-m", "piper_train",
        "--dataset-dir", f"work/{character}/training",
        "--accelerator", "gpu",
        "--devices", "1",
        "--batch-size", str(BATCH_SIZE),
        "--validation-split", "0.0",
        "--num-test-examples", "0",
        "--max_epochs", str(MAX_EPOCHS),
        "--resume_from_checkpoint", checkpoint,
        "--checkpoint-epochs", str(CHECKPOINT_EPOCHS),
        "--quality", "high",
        # bf16 breaks training: piper_train's mel-spectrogram step calls torch.stft
        # (cuFFT) outside any autocast(enabled=False) block, and cuFFT has no bf16
        # kernel at all -- fails on the very first batch with
        # "RuntimeError: cuFFT doesn't support tensor of type: BFloat16"
        "--precision", "32",
    )


def stage_export(character: str, checkpoint: str | None = None) -> None:
    # checkpoint lets you export a specific retained checkpoint (e.g. one picked by
    # ear from stage_sample's output) instead of always the most recent by mtime --
    # a later epoch isn't guaranteed to sound better.
    run_docker("bash", "docker/export.sh", character, checkpoint or "")
    print_handoff(character)


def stage_sample(character: str, num_sentences: int) -> None:
    # Exports every currently-retained checkpoint (see Dockerfile's save_top_k=10)
    # to its own ONNX model and synthesizes the same fixed held-out sentences
    # (corpus.load_validation_sentences) from each, so they can be compared by ear
    # under work/<character>/checkpoint_samples/<checkpoint-name>/*.wav before
    # deciding which checkpoint to hand to stage_export.
    if not find_all_checkpoints(character):
        raise SystemExit(f"No checkpoints found for {character} under work/{character}/training/")
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
    parser = argparse.ArgumentParser(description="Generate a fine-tuned Piper voice model for a character.")
    parser.add_argument("character", help="Character name, matching samples/<character>/")
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
    args = parser.parse_args()

    if args.stage in ("all", "dataset"):
        stage_dataset(args.character, args.corpus_size, args.retry_failed)
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
