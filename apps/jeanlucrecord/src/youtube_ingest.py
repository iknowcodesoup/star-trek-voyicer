import json
import subprocess
import sys
import tempfile
from pathlib import Path

TARGET_RATE = 22050

# What each ingest step leaves behind. A step that finds its own artifact
# already present does nothing, which is what lets a retry resume at the step
# that really failed instead of at the first one.
FULL_WAV_NAME = "full.wav"
TRANSCRIPT_NAME = "transcript.json"
CLIPS_NAME = "clips.json"
CLIPS_DIR_NAME = "clips"
DIARIZATION_NAME = "diarization.json"


def read_json(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: list[dict]) -> None:
    # ensure_ascii=False keeps transcript text readable in the file, and UTF-8
    # for the same reason main.py reconfigures stdout
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def cache_is_current(cache_path: Path, source_path: Path) -> bool:
    """True when the cache exists and is no older than what it derives from.

    Existence alone is not enough. Deleting full.wav is how you force a fresh
    download, and a transcript of the audio that used to be there must not
    survive that.
    """
    if not cache_path.exists() or not source_path.exists():
        return False
    return cache_path.stat().st_mtime >= source_path.stat().st_mtime


def resolve_video_id(url: str) -> str:
    """No bytes downloaded -- lets callers check "already ingested" before
    spending any bandwidth."""
    result = subprocess.run(
        [sys.executable, "-m", "yt_dlp", "--skip-download", "--print", "%(id)s", url],
        capture_output=True, text=True, check=True,
    )
    video_id = result.stdout.strip().splitlines()[-1]
    if not video_id:
        raise RuntimeError(f"yt-dlp returned no video id for {url}")
    return video_id


def download_audio(url: str, out_wav: Path) -> Path:
    """Download best audio and convert to 22050 Hz mono WAV with loudness
    normalization. No-op if out_wav already exists."""
    if out_wav.exists():
        print(f"  Using cached audio: {out_wav}")
        return out_wav

    out_wav.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="yt-dl-") as dl_dir:
        raw_template = str(Path(dl_dir) / "raw.%(ext)s")
        subprocess.run(
            [sys.executable, "-m", "yt_dlp", "-f", "bestaudio", "-o", raw_template, url],
            check=True,
        )

        candidates = sorted(Path(dl_dir).glob("raw.*"))
        if not candidates:
            raise RuntimeError(f"yt-dlp downloaded no audio file for {url}")
        raw_file = candidates[0]

        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(raw_file),
                "-ar", str(TARGET_RATE), "-ac", "1", "-af", "loudnorm",
                str(out_wav),
            ],
            check=True,
        )

    return out_wav


def transcribe(
    wav_path: Path, model_size: str = "medium", cache_path: Path | None = None
) -> list[dict]:
    """Transcribe with faster-whisper, returning sentence-level segments.

    Caches to cache_path and reuses it, the same way diarize does. Whisper is
    the slowest CPU step in the whole ingest, so a failure after it must not
    make the next attempt pay for it a second time.
    """
    if cache_path is not None and cache_is_current(cache_path, wav_path):
        print(f"  Using cached transcript: {cache_path}")
        return read_json(cache_path)

    from faster_whisper import WhisperModel

    print(f"Transcribing with faster-whisper ({model_size})...")
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, _info = model.transcribe(str(wav_path), beam_size=5, word_timestamps=False)

    results = []
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        results.append({"start": segment.start, "end": segment.end, "text": text})
        print(f"  [{segment.start:7.2f} - {segment.end:7.2f}] {text}")

    # An empty result is not cached. It means this video has no speech, which
    # the caller turns into a failure, and caching it would hide a later fix to
    # the audio behind a transcript that says there is nothing to hear.
    if cache_path is not None and results:
        write_json(cache_path, results)
    return results


def chunk_clips(
    wav_path: Path,
    segments: list[dict],
    clips_dir: Path,
    min_duration: float = 1.0,
    max_duration: float = 30.0,
) -> list[dict]:
    """Cut clips at segment boundaries into clips_dir/clip_NNNN.wav, dropping
    segments outside [min_duration, max_duration] before ever cutting them.
    Skips the ffmpeg cut if the clip already exists (resumable). Returns
    metadata for surviving clips only."""
    clips_dir.mkdir(parents=True, exist_ok=True)

    clips = []
    for i, seg in enumerate(segments):
        duration = seg["end"] - seg["start"]
        if duration < min_duration or duration > max_duration:
            print(f"  Skipping segment {i + 1}: duration {duration:.1f}s out of bounds")
            continue

        clip_id = f"clip_{i + 1:04d}"
        clip_path = clips_dir / f"{clip_id}.wav"
        if not clip_path.exists():
            subprocess.run(
                [
                    "ffmpeg", "-y", "-i", str(wav_path),
                    "-ss", f"{seg['start']:.3f}", "-to", f"{seg['end']:.3f}",
                    "-ar", str(TARGET_RATE), "-ac", "1", "-c:a", "pcm_s16le",
                    str(clip_path),
                ],
                check=True,
            )

        clips.append({
            "clip_id": clip_id,
            "start": seg["start"],
            "end": seg["end"],
            "duration": duration,
            "text": seg["text"],
        })

    return clips
