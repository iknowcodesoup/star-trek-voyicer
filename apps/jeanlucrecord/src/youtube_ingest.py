import subprocess
import sys
import tempfile
from pathlib import Path

TARGET_RATE = 22050


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


def transcribe(wav_path: Path, model_size: str = "medium") -> list[dict]:
    """Transcribe with faster-whisper, returning sentence-level segments."""
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
