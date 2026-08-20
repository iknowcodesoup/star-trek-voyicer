"""
Part 1 diagnostic test from docs/dataloader-perf-spec.md.

Compares GPU/CPU behavior when piper_train's DataLoader reads its per-sample
.pt tensors from the Windows bind mount vs. a native Docker volume, to
confirm (or rule out) the bind mount as the I/O bottleneck before committing
to the Part 2 permanent fix (a named volume for work/<character>/training).

Runs two short, bounded training bursts against the same preprocessed data --
Run A against the existing bind-mounted path exactly as main.py trains today,
Run B against a scratch volume seeded with a copy of it -- while sampling
`docker stats` and `nvidia-smi` in parallel, then prints an A/B comparison of
average CPU%, average GPU util%, and steps/sec.

Deliberately does not import main.py: main.py's import chain pulls in
torchaudio/whisper/chatterbox for dataset generation, which is slow to import
and unrelated to this diagnostic. The handful of constants below are
mirrored from main.py -- keep them in sync if those change.

Both runs pass --default_root_dir pointed at a path inside the throwaway
container instead of the real work/<character>/training directory. This
matters: find_latest_checkpoint() in main.py picks the checkpoint with the
newest mtime, so a checkpoint saved by a several-hundred-step diagnostic
burst landing in the real lightning_logs would otherwise get picked up as
"latest" ahead of a real checkpoint that may be thousands of epochs deep,
silently regressing training progress on the next real run.

Both runs also always resume from the base LJSpeech checkpoint rather than
the character's own real checkpoint, for the same class of reason -- see the
comment in start_burst().

Usage:
    uv run python diagnose_io_bottleneck.py <character> [--steps 300]

Cleans up the scratch volume on exit (pass --keep-volume to leave it for a
repeat run without reseeding).
"""
import argparse
import re
import subprocess
import threading
import time
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
DOCKER_IMAGE = "jeanlucrecord-trainer"
BASE_CHECKPOINT = "checkpoints/ljspeech-2000.ckpt"
BATCH_SIZE = 14

SCRATCH_VOLUME = "jlr-scratch-training"
DIAGNOSTIC_ROOT_DIR = "/tmp/jlr-diagnostic-logs"


def docker(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    # Training/log output can contain bytes outside Windows' default cp1252
    # console codepage (same reason main.py itself reconfigures stdout/stderr to
    # utf-8/replace) -- text=True alone decodes with cp1252 here and crashes the
    # subprocess reader thread on the first non-cp1252 byte in `docker logs`.
    return subprocess.run(
        ["docker", *args], check=check, capture_output=True,
        encoding="utf-8", errors="replace",
    )


def seed_scratch_volume(character: str) -> None:
    training_dir = APP_DIR / "work" / character / "training"
    if not (training_dir / "config.json").exists():
        raise SystemExit(
            f"{training_dir} has no preprocessed data yet -- run "
            f"`uv run python main.py {character} --stage preprocess` first."
        )
    print(f"Seeding scratch volume '{SCRATCH_VOLUME}' from {training_dir} ...")
    docker("volume", "create", SCRATCH_VOLUME)
    subprocess.run(
        [
            "docker", "run", "--rm",
            "-v", f"{APP_DIR.as_posix()}:/src",
            "-v", f"{SCRATCH_VOLUME}:/dst",
            "alpine",
            "cp", "-a", f"/src/work/{character}/training/.", "/dst/",
        ],
        check=True,
    )


def cleanup_scratch_volume() -> None:
    print(f"Removing scratch volume '{SCRATCH_VOLUME}' ...")
    docker("volume", "rm", SCRATCH_VOLUME, check=False)


def start_burst(character: str, extra_mount: list[str], steps: int) -> str:
    # Always resume from the base LJSpeech checkpoint rather than the character's
    # own checkpoint -- same model/architecture/data reads either way, so it
    # doesn't affect what this diagnostic measures, and it's one less moving part
    # to reason about. It does NOT dodge the epoch problem below: the base
    # checkpoint is itself already at epoch 1999 (it's Piper's fully-trained
    # LJSpeech release, not a fresh model).
    checkpoint = BASE_CHECKPOINT
    result = subprocess.run(
        [
            "docker", "run", "-d", "--gpus", "all", "--shm-size", "8g",
            "-v", f"{APP_DIR.as_posix()}:/app", *extra_mount, "-w", "/app",
            DOCKER_IMAGE,
            "python3", "-m", "piper_train",
            "--dataset-dir", f"work/{character}/training",
            "--default_root_dir", DIAGNOSTIC_ROOT_DIR,
            "--accelerator", "gpu",
            "--devices", "1",
            "--batch-size", str(BATCH_SIZE),
            "--validation-split", "0.0",
            "--num-test-examples", "0",
            # --max_epochs must be exactly -1 here, not omitted and not some small
            # constant. Whatever checkpoint we resume from already has a
            # current_epoch baked in (1999 for the base checkpoint, thousands more
            # for a character's real one). Any finite --max_epochs <= that value
            # makes Lightning raise MisconfigurationException("You restored a
            # checkpoint with current_epoch=N, but you have set
            # Trainer(max_epochs=M)"). Omitting the flag doesn't dodge this either:
            # pytorch_lightning's own _parse_loop_limits() silently resolves an
            # unset max_epochs to 1000 whenever max_steps is also unset, which hits
            # the exact same wall with no error, just a silent zero-batch run.
            # -1 is the one value _is_max_limit_reached() special-cases as "no
            # limit" (and that restore_loops()'s crash guard excludes), so epoch
            # count is genuinely unbounded; --limit_train_batches below still caps
            # each epoch's batches, and run_burst()'s wall-clock duration_cap is
            # the actual stop mechanism, force-stopping the container once enough
            # real batches have been sampled.
            "--max_epochs", "-1",
            "--limit_train_batches", str(steps),
            "--resume_from_checkpoint", checkpoint,
            "--quality", "high",
            "--precision", "32",
        ],
        check=True, capture_output=True, encoding="utf-8", errors="replace",
    )
    return result.stdout.strip()


def is_running(container_id: str) -> bool:
    result = docker("inspect", "-f", "{{.State.Running}}", container_id, check=False)
    return result.stdout.strip() == "true"


def sample_cpu(container_id: str, samples: list[float], stop: threading.Event) -> None:
    while not stop.is_set():
        result = docker("stats", container_id, "--no-stream", "--format", "{{.CPUPerc}}", check=False)
        text = result.stdout.strip().rstrip("%")
        if text:
            try:
                samples.append(float(text))
            except ValueError:
                pass
        time.sleep(1)


def sample_gpu(samples: list[float], stop: threading.Event) -> None:
    while not stop.is_set():
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, encoding="utf-8", errors="replace",
        )
        lines = result.stdout.strip().splitlines()
        if lines:
            try:
                samples.append(float(lines[0]))
            except ValueError:
                pass
        time.sleep(1)


def parse_steps_per_sec(logs: str) -> float | None:
    # tqdm (Lightning's progress bar) switches from "N.NNit/s" to "N.NNs/it" once
    # the rate drops below 1/s -- exactly the regime an I/O-bound Run A may be in --
    # so both forms need to be recognized and normalized to it/s for a fair A/B
    # comparison. Take the last match overall (by position) so whichever form was
    # in use at the end of the run wins, not whichever form happens to match first.
    matches = re.findall(r"([\d.]+)(it/s|s/it)", logs)
    if not matches:
        return None
    value, unit = matches[-1]
    value = float(value)
    return value if unit == "it/s" else 1.0 / value


def run_burst(label: str, character: str, extra_mount: list[str], steps: int, duration_cap: float) -> dict:
    print(f"\n--- Run {label}: starting ({steps} steps, {duration_cap:.0f}s cap) ---")
    container_id = start_burst(character, extra_mount, steps)
    cpu_samples: list[float] = []
    gpu_samples: list[float] = []
    stop = threading.Event()
    cpu_thread = threading.Thread(target=sample_cpu, args=(container_id, cpu_samples, stop))
    gpu_thread = threading.Thread(target=sample_gpu, args=(gpu_samples, stop))
    cpu_thread.start()
    gpu_thread.start()
    try:
        deadline = time.monotonic() + duration_cap
        while is_running(container_id) and time.monotonic() < deadline:
            time.sleep(1)
        if is_running(container_id):
            print(f"Run {label}: hit the {duration_cap:.0f}s cap before {steps} steps finished -- stopping.")
            docker("stop", container_id, check=False)
    finally:
        stop.set()
        cpu_thread.join()
        gpu_thread.join()

    logs = docker("logs", container_id, check=False)
    docker("rm", "-f", container_id, check=False)

    avg_cpu = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0.0
    avg_gpu = sum(gpu_samples) / len(gpu_samples) if gpu_samples else 0.0
    full_log = logs.stdout + logs.stderr
    steps_per_sec = parse_steps_per_sec(full_log)
    print(f"Run {label}: avg CPU%={avg_cpu:.1f}  avg GPU util%={avg_gpu:.1f}  it/s(last)={steps_per_sec}")
    if steps_per_sec is None:
        # No progress-bar output at all usually means the burst never actually
        # trained (e.g. a resumed checkpoint whose epoch already exceeds
        # --max_epochs, so trainer.fit() returns instantly) rather than a
        # genuinely fast/slow run -- surface the tail of the log so that's
        # visible instead of silently reporting "n/a".
        tail = "\n".join(full_log.strip().splitlines()[-15:])
        print(f"Run {label}: no it/s in the log -- last lines:\n{tail}\n")
    return {"cpu": avg_cpu, "gpu": avg_gpu, "its": steps_per_sec}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("character", help="Character name, matching work/<character>/training/")
    parser.add_argument("--steps", type=int, default=300, help="training batches per run (default: 300)")
    parser.add_argument(
        "--duration-cap", type=float, default=300,
        help="max seconds to wait per run before force-stopping it (default: 300)",
    )
    parser.add_argument("--keep-volume", action="store_true", help="don't remove the scratch volume afterwards")
    args = parser.parse_args()

    seed_scratch_volume(args.character)
    try:
        result_a = run_burst("A (bind mount)", args.character, [], args.steps, args.duration_cap)
        result_b = run_burst(
            "B (named volume)", args.character,
            ["-v", f"{SCRATCH_VOLUME}:/app/work/{args.character}/training"],
            args.steps, args.duration_cap,
        )
    finally:
        if not args.keep_volume:
            cleanup_scratch_volume()

    print("\n=== Summary ===")
    print(f"{'run':16}{'avg CPU%':>10}{'avg GPU%':>10}{'it/s':>10}")
    for label, r in (("A (bind mount)", result_a), ("B (volume)", result_b)):
        its = f"{r['its']:.2f}" if r["its"] is not None else "n/a"
        print(f"{label:16}{r['cpu']:10.1f}{r['gpu']:10.1f}{its:>10}")

    if result_a["gpu"] == 0.0 and result_b["gpu"] == 0.0 and result_a["its"] is None and result_b["its"] is None:
        print(
            "\n-> Both runs recorded zero GPU utilization and no training throughput at "
            "all -- neither burst actually trained (see the log tails above), so this is "
            "not evidence the two storage backends are equivalent. Fix whatever kept "
            "training from starting and rerun before drawing a conclusion."
        )
    elif result_b["gpu"] > result_a["gpu"] * 1.15:
        print(
            "\n-> B shows materially higher GPU utilization than A: bind mount looks "
            "like the bottleneck. Proceed to Part 2 (named volume) of the spec."
        )
    else:
        print(
            "\n-> A and B look similar: the bind mount likely isn't the bottleneck. "
            "Skip Part 2 and look at batch size, main-process collate overhead, or "
            "fp32 precision cost instead (Part 1 step 4 of the spec)."
        )


if __name__ == "__main__":
    main()
