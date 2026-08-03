"""
Follow-up to Part 1 of docs/dataloader-perf-spec.md. That diagnostic already
ruled out the Windows bind mount as the bottleneck (an A/B burst against a
native Docker volume showed identical throughput and GPU utilization), so
Part 1 step 4 points at the main-process side instead: single-threaded
collate overhead, or the --precision 32 cost. Guessing between those from
source alone isn't conclusive -- this runs one short burst with Lightning's
built-in `--profiler simple`, which times every named hook Lightning calls
and prints a mean/total breakdown, to get an actual answer.

The line to look at is `get_train_batch` vs. `[LightningModule]VitsModel.
training_step`: if `get_train_batch` (time spent waiting on the DataLoader
for the next batch) dominates, that points back at DataLoader-side work
(collate, worker throughput); if `training_step` dominates, the cost is in
the GAN forward/backward pass itself -- which includes a full mel-spectrogram
STFT computed on the generated audio every step in mandatory fp32 (see the
Dockerfile's bf16 comment on why it can't be autocast).

Standalone like diagnose_io_bottleneck.py, and for the same reasons: resumes
from the base LJSpeech checkpoint with --max_epochs -1 (dodges Lightning's
already-past-max_epochs resume failure -- see diagnose_io_bottleneck.py's
start_burst() for the full explanation), and redirects --default_root_dir
into the throwaway container so nothing lands in the real
work/<character>/training/lightning_logs.

Bounded via --max_steps, not --limit_train_batches: the latter only caps
batches *per epoch*, and with --max_epochs -1 (required -- see above) there
is otherwise no total bound at all, so the run just keeps going epoch after
epoch until something kills it. Lightning's SimpleProfiler only prints its
report at trainer.fit()'s normal teardown, which a forced `docker stop`
never reaches -- so an unbounded run that has to be killed produces no
report, not a hidden one. --max_steps is computed as the resumed
checkpoint's own global_step plus --steps, since an absolute value hits the
exact same "checkpoint already past this limit" problem --max_epochs did.

Usage:
    uv run python profile_training_step.py <character> [--steps 200]
"""
import argparse
import subprocess
import time
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
DOCKER_IMAGE = "jeanlucrecord-trainer"
BASE_CHECKPOINT = "checkpoints/ljspeech-2000.ckpt"
BATCH_SIZE = 14
DIAGNOSTIC_ROOT_DIR = "/tmp/jlr-diagnostic-logs"


def docker(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", *args], check=check, capture_output=True,
        encoding="utf-8", errors="replace",
    )


def get_checkpoint_global_step(checkpoint: str) -> int:
    result = subprocess.run(
        [
            "docker", "run", "--rm",
            "-v", f"{APP_DIR.as_posix()}:/app", "-w", "/app",
            DOCKER_IMAGE,
            "python3", "-c",
            f"import torch; print(torch.load('{checkpoint}', map_location='cpu', weights_only=False)['global_step'])",
        ],
        check=True, capture_output=True, encoding="utf-8", errors="replace",
    )
    return int(result.stdout.strip().splitlines()[-1])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("character", help="Character name, matching work/<character>/training/")
    parser.add_argument("--steps", type=int, default=200, help="training batches to profile (default: 200)")
    parser.add_argument(
        "--duration-cap", type=float, default=300,
        help="max seconds to wait before force-stopping it (default: 300)",
    )
    args = parser.parse_args()

    training_dir = APP_DIR / "work" / args.character / "training"
    if not (training_dir / "config.json").exists():
        raise SystemExit(
            f"{training_dir} has no preprocessed data yet -- run "
            f"`uv run python main.py {args.character} --stage preprocess` first."
        )

    base_step = get_checkpoint_global_step(BASE_CHECKPOINT)
    target_step = base_step + args.steps
    print(
        f"Base checkpoint is at global_step={base_step}; profiling up to "
        f"step={target_step} ({args.steps} more) for '{args.character}' (--profiler simple) ..."
    )
    result = subprocess.run(
        [
            "docker", "run", "-d", "--gpus", "all", "--shm-size", "8g",
            "-v", f"{APP_DIR.as_posix()}:/app", "-w", "/app",
            DOCKER_IMAGE,
            "python3", "-m", "piper_train",
            "--dataset-dir", f"work/{args.character}/training",
            "--default_root_dir", DIAGNOSTIC_ROOT_DIR,
            "--accelerator", "gpu",
            "--devices", "1",
            "--batch-size", str(BATCH_SIZE),
            "--validation-split", "0.0",
            "--num-test-examples", "0",
            "--max_epochs", "-1",
            "--max_steps", str(target_step),
            "--resume_from_checkpoint", BASE_CHECKPOINT,
            "--quality", "high",
            "--precision", "32",
            "--profiler", "simple",
        ],
        check=True, capture_output=True, encoding="utf-8", errors="replace",
    )
    container_id = result.stdout.strip()

    deadline = time.monotonic() + args.duration_cap
    stopped_naturally = False
    while time.monotonic() < deadline:
        state = docker("inspect", "-f", "{{.State.Running}}", container_id, check=False)
        if state.stdout.strip() != "true":
            stopped_naturally = True
            break
        time.sleep(2)
    if not stopped_naturally:
        print(f"Hit the {args.duration_cap:.0f}s cap before {args.steps} steps finished -- stopping.")
        docker("stop", container_id, check=False)

    logs = docker("logs", container_id, check=False)
    docker("rm", "-f", container_id, check=False)
    full_log = logs.stdout + logs.stderr

    lines = full_log.splitlines()
    report_start = next((i for i, line in enumerate(lines) if "Profiler Report" in line), None)
    if report_start is None:
        tail = "\n".join(lines[-30:])
        print(f"No profiler report found in the log -- last lines:\n{tail}")
        return
    print("\n".join(lines[report_start:]))


if __name__ == "__main__":
    main()
