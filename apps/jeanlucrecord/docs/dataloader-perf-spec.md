# GPU Utilization & Training Throughput Spec

> **Note:** written against the old flat `main.py`, before the package
> migration. Every `main.py:N` reference below is now `src/voice_factory/cli.py`,
> and the line numbers are approximate -- the file moved and its imports changed.

## Background / diagnosis

Symptoms observed during the `train` stage:
- `docker stats`: CPU ~136% of 1600% available (host has 16 logical cores; only ~1.4 in use).
- `nvidia-smi`: GPU utilization fluctuating 0-90%+, averaging ~60-75%.

Current state:
- Dockerfile patches `piper_train`'s DataLoader construction to `num_workers=8`,
  `pin_memory=True`, `persistent_workers=True` ([Dockerfile:54-55](../Dockerfile#L54-L55)).
- `BATCH_SIZE = 12` ([main.py:24](../main.py#L24)).
- The training container is launched with `-v {APP_DIR}:/app` — a Windows-host bind mount
  ([main.py:44](../main.py#L44)).

Interpretation: low CPU utilization *concurrent with* spiky, sub-saturated GPU utilization is
the signature of an I/O-bound DataLoader, not a CPU-bound one — workers are blocked waiting on
file reads rather than spending CPU decoding/collating, so adding worker CPU-parallelism alone
won't fix it. The likely I/O bottleneck given this setup is the Windows-to-WSL2 bind mount
Docker Desktop uses to expose `C:\Projects\...` inside the Linux container: that path crosses a
9P/gRPC-FUSE translation layer known to be slow for many small random reads — exactly the
per-sample `.pt` tensor read pattern the DataLoader does.

## Goals

- Confirm (not assume) the bind mount is the dominant bottleneck before making permanent changes.
- Apply the fix with minimal disruption to the existing `main.py` pipeline and its
  resume-from-checkpoint logic.
- Layer in the cheap DataLoader tuning knobs (`prefetch_factor`, `num_workers`, `batch_size`)
  regardless, since they're low-risk complements independent of the mount question.

## Part 1 — Diagnostic test (confirm the bind mount is the bottleneck)

Goal: compare GPU/CPU behavior when the DataLoader reads from the Windows bind mount vs. a
native Docker volume, using the same preprocessed data, without touching the real pipeline.

1. After the `preprocess` stage has produced `work/<character>/training/`, create a scratch
   named volume and seed it with a copy of that directory:
   ```
   docker volume create jlr-scratch-training
   docker run --rm -v "<APP_DIR>:/src" -v jlr-scratch-training:/dst alpine \
     cp -a /src/work/<character>/training/. /dst/
   ```
2. Run a short training burst (a few hundred steps — no need to converge) twice, changing only
   the storage backing `--dataset-dir`:
   - **Run A (current)**: existing bind-mounted path, exactly as `main.py` runs it today.
   - **Run B (test)**: same invocation, but with
     `-v jlr-scratch-training:/app/work/<character>/training` layered on top of (overriding) the
     bind mount for that one subdirectory.
3. While each run executes, capture in parallel for ~2 minutes:
   - `docker stats <container>` (CPU%), sampled every 1s.
   - `nvidia-smi --query-gpu=utilization.gpu --format=csv -l 1`.
   - Piper/Lightning's own it/s (or TensorBoard step rate if running).
4. Compare average GPU utilization and steps/sec between A and B.
   - If B shows materially higher/steadier GPU utilization and higher steps/sec for similar CPU
     usage → bind mount confirmed as the bottleneck; proceed to Part 2.
   - If A and B look the same → bottleneck is elsewhere (e.g. batch size, single-threaded
     collate in the main process, `--precision 32` cost); skip Part 2 and investigate those
     instead.
5. Clean up: `docker volume rm jlr-scratch-training`.

## Part 2 — Permanent fix: named volume for the hot-read directory

Only implement if Part 1 confirms the bottleneck.

Scope: only `work/<character>/training/` (the directory the DataLoader randomly reads
per-sample during training) moves off the Windows bind mount. Code (`/app` root), `samples/`,
`checkpoints/`, and other lightly-read directories stay on the existing bind mount — they're
never read in a hot per-batch loop, so the mount penalty there doesn't matter.

Changes to `main.py`:
1. Add a `training_volume_name(character)` helper → `f"jlr-training-{character}"`; ensure the
   volume exists (`docker volume create`, idempotent) before any stage that touches
   `work/<character>/training`.
2. Extend `run_docker()` to accept an extra `-v` mount for the training-data volume, layered
   inside the existing `-v {APP_DIR}:/app` bind mount at `/app/work/<character>/training`
   (Docker allows a more specific mount nested inside a broader bind mount — the volume wins for
   that subpath).
3. `stage_preprocess`: if the volume is empty and a host copy already exists (e.g. from a prior
   bind-mount-only run), seed the volume from the host directory once via a helper container;
   write preprocessing output directly to the volume-backed path from then on.
4. `stage_train`: runs against the volume-backed path automatically once the mount is layered
   into `run_docker` — no other change needed.
5. After each `stage_train` invocation returns (training may be interrupted/resumed across
   separate runs), sync checkpoints back out to the host-visible
   `work/<character>/training/lightning_logs/**/*.ckpt` path via a helper container copy, so
   `find_latest_checkpoint()` ([main.py:101-107](../main.py#L101-L107)), which globs the host
   filesystem directly, keeps working unmodified.
6. `stage_export`: mount the same volume (read-only) so `docker/export.sh` sees the final
   checkpoints without depending on the host copy being perfectly in sync.

Risk/rollback: purely additive — the bind mount stays in place for everything else, and the
sync-back step guarantees a host copy of checkpoints always exists, so reverting to
all-bind-mount behavior is just removing the volume-mount lines.

## Part 3 — Cheap DataLoader tuning (do regardless of Part 1/2 outcome)

1. **`prefetch_factor`** — [Dockerfile:55](../Dockerfile#L55) currently patches in
   `pin_memory=True, persistent_workers=True` but never sets `prefetch_factor` (PyTorch default:
   2 batches/worker). Extend the same `sed` to add `prefetch_factor=4` — gives each of the 8
   workers more batches staged ahead, absorbing I/O jitter regardless of its source.
2. **`num_workers`** — [Dockerfile:54](../Dockerfile#L54), currently 8. The host has 16 logical
   cores and is using only ~1.4 of them, so there's free headroom. Try 12 (leaving a couple cores
   for the main process/OS); confirm via `docker stats` that CPU% actually rises before assuming
   it helped.
3. **`BATCH_SIZE`** — [main.py:24](../main.py#L24), currently 12. Check
   `nvidia-smi --query-gpu=memory.used,memory.total --format=csv` headroom during a run; if
   there's slack, raise it in increments (16, then 24) — larger batches mean more GPU compute per
   DataLoader fetch, smoothing utilization dips even when fetches themselves are still somewhat
   slow.

Change and test one of these at a time (not all three simultaneously), so each one's effect on
GPU utilization / steps-per-second is individually attributable.

## Suggested order of operations

1. Part 1 diagnostic test — confirms whether Part 2 is worth doing (~30-60 min).
2. Part 3.1 (`prefetch_factor`) — cheapest, no risk, apply regardless of the test outcome.
3. Based on the Part 1 result: Part 2 (named volume) if confirmed; otherwise investigate other
   causes (main-process collate overhead, fp32 precision cost) instead.
4. Part 3.2 (`num_workers`) and Part 3.3 (`batch_size`) — tune individually once the I/O question
   is settled, since their effect is cleanest to read once any mount bottleneck is out of the
   picture.
