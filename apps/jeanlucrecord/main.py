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

from corpus import load_corpus  # noqa: E402
from generate_dataset import generate_dataset  # noqa: E402
from resample import normalize_ref_wav, resample_dir  # noqa: E402

APP_DIR = Path(__file__).resolve().parent
DOCKER_IMAGE = "jeanlucrecord-trainer"

CORPUS_SIZE_DEFAULT = 1300
BASE_CHECKPOINT = "checkpoints/ljspeech-2000.ckpt"
MAX_EPOCHS = 3000
BATCH_SIZE = 12

STAGES = ["all", "dataset", "resample", "preprocess", "smoketest", "train", "export"]


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


def find_latest_checkpoint(character: str) -> str | None:
    training_dir = APP_DIR / "work" / character / "training"
    checkpoints = list(training_dir.glob("**/*.ckpt"))
    if not checkpoints:
        return None
    latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
    return latest.relative_to(APP_DIR).as_posix()


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
        "--checkpoint-epochs", "1",
        "--quality", "high",
        # bf16 breaks training: piper_train's mel-spectrogram step calls torch.stft
        # (cuFFT) outside any autocast(enabled=False) block, and cuFFT has no bf16
        # kernel at all -- fails on the very first batch with
        # "RuntimeError: cuFFT doesn't support tensor of type: BFloat16"
        "--precision", "32",
    )


def stage_export(character: str) -> None:
    run_docker("bash", "docker/export.sh", character)
    print_handoff(character)


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
    if args.stage in ("all", "export"):
        stage_export(args.character)


if __name__ == "__main__":
    main()
