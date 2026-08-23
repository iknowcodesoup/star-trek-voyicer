import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
from pathlib import Path

# corpus text and Whisper transcripts can contain characters outside the
# Windows console's default cp1252 codepage
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from voice_factory import config
from voice_factory.core.corpus import load_corpus, load_validation_sentences
from voice_factory.core.diarization import (
    MIN_SPEAKER_COVERAGE,
    assign_speakers,
    count_by_speaker,
    diarize,
)
from voice_factory.core.generate_dataset import generate_dataset
from voice_factory.core.import_dataset import import_dataset
from voice_factory.core.quality import (
    FLAG_THRESHOLD_DB,
    clip_quality_score,
    is_flagged,
)
from voice_factory.core.resample import normalize_ref_wav, resample_dir
from voice_factory.core.review_workflow import commit_reviewed_clips
from voice_factory.core.youtube_ingest import (
    CLIPS_DIR_NAME,
    CLIPS_NAME,
    DIARIZATION_NAME,
    FULL_WAV_NAME,
    TRANSCRIPT_NAME,
    chunk_clips,
    download_audio,
    ensure_video_meta,
    read_json,
    resolve_video_id,
    transcribe,
    write_json,
)
from voice_factory.core.youtube_search import (
    SEARCH_LIMIT_DEFAULT,
    search_videos,
)
from voice_factory.repositories.review_csv_repository import (
    REVIEW_CSV_NAME,
    write_review_csv,
)
from voice_factory.repositories.speaker_map_repository import (
    SPEAKER_MAP_FILENAME,
)
from voice_factory.schemas import (
    YOUTUBE_INGEST_STEPS,
    YOUTUBE_STAGES_NEEDING_URL,
)

# APP_DIR is the app root (apps/jeanlucrecord/), where work/, samples/,
# checkpoints/, and docker/ all live -- not this module's own package
# directory. See config.APP_ROOT for the arithmetic.
APP_DIR = config.APP_ROOT
DOCKER_IMAGE = "jeanlucrecord-trainer"

HF_TOKEN_ENV_VAR = config.HF_TOKEN_ENV_VAR

# Diarization needs a HuggingFace token. Read apps/jeanlucrecord/.env so it does
# not have to be exported by hand before every run. An already-exported variable
# wins, which is what override=False gives.
config.load_app_dotenv()

CORPUS_SIZE_DEFAULT = 1300
BASE_CHECKPOINT = "checkpoints/ljspeech-2000.ckpt"
MAX_EPOCHS = 3000
# 12 -> 14: docs/dataloader-perf-spec.md profiling showed training is GPU-compute-bound
# (DataLoader wait was <1% of total time), so a bigger batch amortizes per-step
# forward/backward overhead over more samples. Kept to a smaller increment than the
# usual 16 since headroom was thin (~1GB free of 8151MiB) at batch size 12 --
# watch nvidia-smi memory.used on the next run regardless.
BATCH_SIZE = 12
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
    "youtube-search",
    "youtube-ingest",
    "youtube-download",
    "youtube-transcribe",
    "youtube-chunk",
    "youtube-diarize",
    "youtube-review",
    "youtube-commit",
]


def dataset_dir_for(character: str) -> Path:
    return APP_DIR / "work" / character / "dataset"


_video_dirs: dict[str, Path] = {}


def video_dir_for(url: str) -> Path:
    """Where one video's ingest artifacts live.

    Shared across every character: claiming an already-ingested video for a
    second character reads the same directory, so download/transcribe/diarize
    never repeat.

    Memoized because the composed youtube-ingest stage asks five times in a
    row, and resolve_video_id costs a yt-dlp metadata request each time. When
    the orchestrator drives the five steps, each is its own process and the
    memo simply never gets a second hit.
    """
    if url not in _video_dirs:
        _video_dirs[url] = APP_DIR / "work" / "youtube" / resolve_video_id(url)
    return _video_dirs[url]


def resolve_hf_token(hf_token: str | None) -> str:
    token = hf_token or os.environ.get(HF_TOKEN_ENV_VAR)
    if not token:
        raise SystemExit(
            f"Diarization needs a HuggingFace read token. Pass --hf-token or set "
            f"{HF_TOKEN_ENV_VAR}. Accept the terms for pyannote/speaker-diarization-3.1 "
            "and pyannote/segmentation-3.0 on huggingface.co first."
        )
    return token


def container_name_for(character: str, stage: str) -> str:
    return f"{DOCKER_IMAGE}-{character}-{stage}"


def run_docker(*args: str, container_name: str | None = None) -> None:
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
    # naming the container lets an outside caller stop it. Terminating this
    # process alone leaves the container running, because docker run only
    # forwards the signal when it owns the terminal -- see api.py's cancel path.
    name_args = ["--name", container_name] if container_name else []
    subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            *name_args,
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


def stage_youtube_search(query: str, limit: int) -> None:
    videos = search_videos(query, limit)
    if not videos:
        print(f"No results for {query!r}.")
        return
    print(f"{len(videos)} result(s) for {query!r}:\n")
    for video in videos:
        duration = video["duration_sec"]
        minutes = f"{int(duration) // 60}:{int(duration) % 60:02d}" if duration else "?"
        print(f"  [{minutes:>7}] {video['title']}")
        print(f"            {video['channel'] or 'unknown channel'} -- {video['url']}")


def stage_youtube_download(url: str) -> None:
    """Download the audio and transcode it to 22050 Hz mono.

    Does nothing when full.wav is already there, so this is also the step to
    rerun after ffmpeg or yt-dlp was missing. Delete full.wav to force a fresh
    download -- every later step notices the audio changed under it.
    """
    video_dir = video_dir_for(url)
    # Before the download, not after: the download is the longest step here and
    # the one most likely to fail. Writing the name first means a video that is
    # still downloading, or whose download died, still lists under its title
    # rather than under the raw id.
    ensure_video_meta(url, video_dir)
    full_wav = download_audio(url, video_dir / FULL_WAV_NAME)
    print(f"Audio ready at {full_wav}")


def _require_audio(url: str) -> Path:
    full_wav = video_dir_for(url) / FULL_WAV_NAME
    if not full_wav.exists():
        raise SystemExit(f"No audio at {full_wav}. Run --stage youtube-download first.")
    return full_wav


def stage_youtube_transcribe(url: str, whisper_model: str) -> None:
    """Transcribe the audio to transcript.json. The slowest CPU step here."""
    video_dir = video_dir_for(url)
    segments = transcribe(
        _require_audio(url), whisper_model, video_dir / TRANSCRIPT_NAME
    )
    if not segments:
        raise SystemExit(f"No speech segments found in {url}")
    print(f"{len(segments)} segment(s) at {video_dir / TRANSCRIPT_NAME}")


def stage_youtube_chunk(url: str, min_duration: float, max_duration: float) -> None:
    """Cut the transcript's segments into clips, and record what survived."""
    video_dir = video_dir_for(url)
    transcript_path = video_dir / TRANSCRIPT_NAME
    if not transcript_path.exists():
        raise SystemExit(
            f"No transcript at {transcript_path}. Run --stage youtube-transcribe first."
        )

    clips = chunk_clips(
        _require_audio(url),
        read_json(transcript_path),
        video_dir / CLIPS_DIR_NAME,
        min_duration,
        max_duration,
    )
    if not clips:
        raise SystemExit(
            f"No clips survived duration filtering for {url} -- adjust "
            "--min/--max-clip-duration and retry."
        )
    write_json(video_dir / CLIPS_NAME, clips)
    print(f"{len(clips)} clip(s) at {video_dir / CLIPS_DIR_NAME}")


def _require_clips(video_dir: Path) -> list[dict]:
    clips_path = video_dir / CLIPS_NAME
    if not clips_path.exists():
        raise SystemExit(f"No clips at {clips_path}. Run --stage youtube-chunk first.")
    return read_json(clips_path)


def stage_youtube_diarize(
    url: str,
    hf_token: str | None = None,
    num_speakers: int | None = None,
    min_speaker_coverage: float = MIN_SPEAKER_COVERAGE,
) -> None:
    """Label each clip with the speaker who covers it.

    Writes the speaker labels back into clips.json, so the review step reads
    one file whether this ran or not.
    """
    video_dir = video_dir_for(url)
    turns = diarize(
        _require_audio(url),
        resolve_hf_token(hf_token),
        video_dir / DIARIZATION_NAME,
        num_speakers=num_speakers,
    )
    clips = assign_speakers(_require_clips(video_dir), turns, min_speaker_coverage)
    write_json(video_dir / CLIPS_NAME, clips)
    print("\nClips per speaker:")
    for speaker_label, count in count_by_speaker(clips).items():
        print(f"  {speaker_label:<12} {count}")


def stage_youtube_review(url: str, quality_flag_threshold: float) -> None:
    """Score every clip and write the review.csv an operator edits.

    Never overwrites a review.csv that is already there. That file is the one
    record of which clips a person accepted, and a retry that reached this step
    must not throw those decisions away.
    """
    video_dir = video_dir_for(url)
    review_path = video_dir / REVIEW_CSV_NAME
    if review_path.exists():
        print(f"Review already written at {review_path}, keeping its decisions.")
        return

    clips = _require_clips(video_dir)
    # The artifact is the record: diarization ran if and only if it left its
    # cache behind, so no caller has to pass the flag through a second time.
    enable_diarization = (video_dir / DIARIZATION_NAME).exists()

    rows = []
    for clip in clips:
        score = clip_quality_score(
            video_dir / CLIPS_DIR_NAME / f"{clip['clip_id']}.wav"
        )
        flagged = is_flagged(score, quality_flag_threshold)
        speaker_label = clip.get("speaker_label")
        # a clip no single speaker owns is cross-talk or noise -- default it to
        # keep=0 for the same reason a low quality score does
        rejected_by_diarization = enable_diarization and speaker_label is None
        rows.append(
            {
                "clip_id": clip["clip_id"],
                "keep": "0" if flagged or rejected_by_diarization else "1",
                "quality_score": round(score, 2),
                "flagged": int(flagged),
                "speaker_label": speaker_label or "",
                "speaker_coverage": round(clip.get("speaker_coverage", 0.0), 3),
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
    if enable_diarization:
        print(
            f"\nWrite {video_dir / SPEAKER_MAP_FILENAME} to route each speaker to a "
            'character, e.g. {"SPEAKER_00": "janeway", "SPEAKER_01": null}.'
        )
    print(
        f"Listen to clips under {video_dir / CLIPS_DIR_NAME}, edit the 'keep' column, "
        "then run, for each character this video's clips should reach:"
    )
    print("  uv run jeanlucrecord <character> --stage youtube-commit")


def stage_youtube_ingest(
    url: str,
    whisper_model: str,
    min_duration: float,
    max_duration: float,
    quality_flag_threshold: float,
    enable_diarization: bool = False,
    hf_token: str | None = None,
    num_speakers: int | None = None,
    min_speaker_coverage: float = MIN_SPEAKER_COVERAGE,
) -> None:
    """Run every ingest step in order, for one command on the command line.

    The steps stay separate stages underneath. The orchestrator starts them one
    job at a time so a failure only costs the step that failed, and this is the
    same sequence for anyone who would rather type it once. No character is
    needed here: the artifacts land under a video id shared by every
    character, so a second character claiming this video skips straight to
    stage_youtube_commit instead of repeating any of this. Nothing here
    touches work/<character>/dataset/ -- see stage_youtube_commit.
    """
    video_dir = video_dir_for(url)
    review_path = video_dir / REVIEW_CSV_NAME
    if review_path.exists():
        print(f"{url} already ingested, review at {review_path}")
        return

    # fail before downloading anything, not after the slowest step
    if enable_diarization:
        resolve_hf_token(hf_token)

    stage_youtube_download(url)
    stage_youtube_transcribe(url, whisper_model)
    stage_youtube_chunk(url, min_duration, max_duration)
    if enable_diarization:
        stage_youtube_diarize(url, hf_token, num_speakers, min_speaker_coverage)
    stage_youtube_review(url, quality_flag_threshold)


def stage_youtube_commit(character: str) -> None:
    # shared across every character now, so this scans every ingested video --
    # not just ones this character happened to ingest -- and speaker_map.json
    # decides which of them actually route clips here (see review.py)
    youtube_dir = APP_DIR / "work" / "youtube"
    if not youtube_dir.exists():
        raise SystemExit(f"No ingested YouTube videos found under {youtube_dir}")
    out_dir = dataset_dir_for(character)
    result = commit_reviewed_clips(youtube_dir, out_dir, dataset_dir_for)

    # a diarized video routes clips to several characters, so mark every dataset
    # that actually gained clips -- not just the character named on the command
    # line. Missing one lets a later --stage all overlay Chatterbox synthesis on
    # top of that character's real audio.
    for target, count in sorted(result.committed_by_target.items()):
        target_character = target.parent.name
        mark_external_source(
            target_character, f"{count} clip(s) committed from {youtube_dir}"
        )
        if target != out_dir:
            print(f"  {count} clip(s) -> {target_character}")

    print(
        f"Committed {result.newly_committed} new clip(s), "
        f"{result.already_committed} already committed."
    )


def stage_resample(character: str) -> None:
    # cheap CPU-only step (no TTS/Whisper) -- always rerun so it reflects whatever
    # the dataset stage currently has, rather than risk mirroring a stale dataset
    dataset_dir = APP_DIR / "work" / character / "dataset"
    resampled_dir = APP_DIR / "work" / character / "resampled"
    resampled_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(dataset_dir / "metadata.csv", resampled_dir / "metadata.csv")
    resample_dir(dataset_dir / "wavs", resampled_dir / "wavs")


DATASET_FINGERPRINT_NAME = "dataset-fingerprint.json"


def _metadata_fingerprint(directory: Path) -> dict:
    """Clip count plus a content hash of directory/metadata.csv.

    This is the comparison basis stage_preprocess uses to decide whether
    piper_train.preprocess needs to rerun: cheap to compute, and it changes
    exactly when the rows or text in metadata.csv change.
    """
    metadata_path = directory / "metadata.csv"
    if not metadata_path.exists():
        raise SystemExit(
            f"No metadata.csv found at {metadata_path}. Run --stage resample first."
        )
    try:
        content = metadata_path.read_bytes()
        clip_count = len(content.decode("utf-8").splitlines())
    except (FileNotFoundError, IsADirectoryError, UnicodeDecodeError) as error:
        # FileNotFoundError here means a TOCTOU race (deleted between the
        # exists() check above and this read); UnicodeDecodeError means the
        # file isn't the text file this stage expects -- both are "bad input
        # file" conditions, same clean SystemExit as the missing-file case
        # above rather than a raw traceback
        raise SystemExit(
            f"Can't read {metadata_path} ({error}). Run --stage resample first."
        ) from error
    return {
        "clip_count": clip_count,
        "metadata_hash": hashlib.sha256(content).hexdigest(),
    }


def _load_sidecar(training_dir: Path) -> dict | None:
    """The fingerprint stage_preprocess recorded after its last successful run.

    Missing entirely, or corrupt (an interrupted write from a crash, a full
    disk, a kill mid-write) -- both read back as "no sidecar", so a damaged
    file makes stage_preprocess regenerate instead of raising.
    """
    sidecar_path = training_dir / DATASET_FINGERPRINT_NAME
    if not sidecar_path.exists():
        return None
    try:
        return json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        # JSONDecodeError: valid text, broken JSON. UnicodeDecodeError: a
        # truncated write can leave invalid UTF-8 bytes. OSError: the file
        # vanished (or turned unreadable) between the exists() check above
        # and this read. All three are "corrupt sidecar", same as missing.
        return None


def _write_sidecar(training_dir: Path, fingerprint: dict) -> None:
    # write-then-replace instead of writing sidecar_path directly: Path.replace
    # is an atomic rename on both POSIX and Windows, so a crash, full disk, or
    # kill mid-write leaves either the old sidecar or the new one -- never a
    # truncated file in between for _load_sidecar to have to guard against.
    sidecar_path = training_dir / DATASET_FINGERPRINT_NAME
    tmp_path = sidecar_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(fingerprint), encoding="utf-8")
    tmp_path.replace(sidecar_path)


def stage_preprocess(character: str) -> None:
    # fingerprints work/<character>/resampled/, not dataset/ -- resample and
    # preprocess are independently runnable stages (STAGES above), and
    # resampled/metadata.csv is what piper_train.preprocess actually reads
    # (--input-dir below). Fingerprinting dataset/ would regenerate against
    # stale resampled audio whenever dataset/ changed but --stage resample
    # hadn't caught it up yet.
    training_dir = APP_DIR / "work" / character / "training"
    resampled_dir = APP_DIR / "work" / character / "resampled"
    fingerprint = _metadata_fingerprint(resampled_dir)

    config_exists = (training_dir / "config.json").exists()
    sidecar = _load_sidecar(training_dir)
    if config_exists and sidecar == fingerprint:
        print(f"Preprocessed training data already exists at {training_dir}, skipping.")
        return

    if not config_exists:
        reason = f"{training_dir / 'config.json'} is missing"
    elif sidecar is None:
        reason = "no fingerprint sidecar was recorded yet"
    else:
        reason = "the dataset fingerprint changed since the last preprocess run"
    print(f"Regenerating preprocessed training data at {training_dir} ({reason}).")
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
    # written only after run_docker succeeds -- an interrupted or failed
    # preprocess run must be retried next time, not mistaken for up to date
    _write_sidecar(training_dir, fingerprint)


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
            container_name=container_name_for(character, "train"),
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


def stage_sample(
    character: str, num_sentences: int, checkpoint: str | None = None
) -> None:
    # Exports the requested checkpoint(s) to their own ONNX model and synthesizes
    # the same fixed held-out sentences (corpus.load_validation_sentences) from
    # each, so they can be compared by ear under
    # work/<character>/checkpoint_samples/<checkpoint-name>/*.wav before deciding
    # which checkpoint to hand to stage_export. Defaults to just the latest
    # checkpoint; pass checkpoint="all" to sample every currently-retained one
    # (see CHECKPOINT_KEEP above) like this used to do unconditionally.
    all_checkpoints = find_all_checkpoints(character)
    if not all_checkpoints:
        raise SystemExit(
            f"No checkpoints found for {character} under work/{character}/training/"
        )
    if checkpoint == "all":
        targets = [p.relative_to(APP_DIR).as_posix() for p in all_checkpoints]
    elif checkpoint:
        targets = [checkpoint]
    else:
        targets = [find_latest_checkpoint(character)]

    sentences_path = APP_DIR / "work" / character / "validation_sentences.txt"
    sentences_path.write_text(
        "\n".join(load_validation_sentences(num_sentences)) + "\n", encoding="utf-8"
    )
    run_docker("bash", "docker/sample_checkpoints.sh", character, *targets)


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
        "character",
        nargs="?",
        help="Character name, matching samples/<character>/. Optional for "
        "--stage youtube-search, which searches YouTube and writes nothing, "
        "and for --stage youtube-ingest and its five sub-stages "
        "(youtube-download/transcribe/chunk/diarize/review), which act on a "
        "video shared across every character.",
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
        "recent by mtime). For --stage sample: path to a specific .ckpt to sample, "
        "or 'all' to sample every retained checkpoint (default: most recent only). "
        "Use after --stage sample to export whichever checkpoint sounded best rather "
        "than assuming the latest one is.",
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
    parser.add_argument(
        "--diarize",
        action="store_true",
        help="For --stage youtube-ingest: split the audio by speaker with pyannote and "
        "record a speaker_label per clip. Clips no single speaker owns (cross-talk, "
        "music) default to keep=0. Needs a HuggingFace token, see --hf-token.",
    )
    parser.add_argument(
        "--num-speakers",
        type=int,
        help="For --diarize: exact speaker count, when you know it. Omit to let "
        "pyannote decide.",
    )
    parser.add_argument(
        "--min-speaker-coverage",
        type=float,
        default=MIN_SPEAKER_COVERAGE,
        help="For --diarize: a clip must be this fraction single-speaker to earn a "
        f"label (default: {MIN_SPEAKER_COVERAGE}).",
    )
    parser.add_argument(
        "--hf-token",
        help=f"HuggingFace read token for the pyannote models. Falls back to the "
        f"{HF_TOKEN_ENV_VAR} environment variable.",
    )
    parser.add_argument(
        "--search-query",
        help="For --stage youtube-search: phrase to search YouTube for.",
    )
    parser.add_argument(
        "--search-limit",
        type=int,
        default=SEARCH_LIMIT_DEFAULT,
        help=f"For --stage youtube-search: how many results to return (default: "
        f"{SEARCH_LIMIT_DEFAULT}).",
    )
    args = parser.parse_args()

    if args.stage == "import" and not args.import_dir:
        parser.error("--stage import requires --import-dir")
    if args.stage in YOUTUBE_STAGES_NEEDING_URL and not args.youtube_url:
        parser.error(f"--stage {args.stage} requires --youtube-url")
    if args.stage == "youtube-search" and not args.search_query:
        parser.error("--stage youtube-search requires --search-query")
    # youtube-ingest and its five steps act on a video shared across every
    # character, so none of them need one -- only youtube-commit and the
    # character-only stages below do.
    if (
        args.stage not in ("youtube-search", *YOUTUBE_STAGES_NEEDING_URL)
        and not args.character
    ):
        parser.error(f"--stage {args.stage} requires a character")

    if args.stage in ("all", "dataset"):
        stage_dataset(args.character, args.corpus_size, args.retry_failed)
    if args.stage == "import":
        stage_import(args.character, Path(args.import_dir))
    if args.stage == "youtube-search":
        stage_youtube_search(args.search_query, args.search_limit)
    if args.stage == "youtube-ingest":
        stage_youtube_ingest(
            args.youtube_url,
            args.whisper_model,
            args.min_clip_duration,
            args.max_clip_duration,
            args.quality_flag_threshold,
            args.diarize,
            args.hf_token,
            args.num_speakers,
            args.min_speaker_coverage,
        )
    if args.stage == "youtube-download":
        stage_youtube_download(args.youtube_url)
    if args.stage == "youtube-transcribe":
        stage_youtube_transcribe(args.youtube_url, args.whisper_model)
    if args.stage == "youtube-chunk":
        stage_youtube_chunk(
            args.youtube_url,
            args.min_clip_duration,
            args.max_clip_duration,
        )
    if args.stage == "youtube-diarize":
        stage_youtube_diarize(
            args.youtube_url,
            args.hf_token,
            args.num_speakers,
            args.min_speaker_coverage,
        )
    if args.stage == "youtube-review":
        stage_youtube_review(args.youtube_url, args.quality_flag_threshold)
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
        stage_sample(args.character, args.num_validation_sentences, args.checkpoint)
    if args.stage in ("all", "export"):
        stage_export(args.character, args.checkpoint)


if __name__ == "__main__":
    main()
