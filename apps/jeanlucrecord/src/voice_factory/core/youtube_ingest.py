import json
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from voice_factory.repositories.video_meta_repository import (
    META_FILENAME,
    read_video_meta,
    write_video_meta_file,
)

TARGET_RATE = 22050

# What each ingest step leaves behind. A step that finds its own artifact
# already present does nothing, which is what lets a retry resume at the step
# that really failed instead of at the first one.
FULL_WAV_NAME = "full.wav"
TRANSCRIPT_NAME = "transcript.json"
CLIPS_NAME = "clips.json"
CLIPS_DIR_NAME = "clips"
DIARIZATION_NAME = "diarization.json"
META_NAME = META_FILENAME


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


def resolve_video_meta(url: str) -> dict:
    """What yt-dlp already knows about a video, in the shape meta.json stores.

    resolve_video_id pays for a metadata request and keeps one field of the
    answer. The title, duration, and channel come back in the same response,
    and they are the fields a dashboard needs to name the video. The field
    names match youtube_search.search_videos, so a searched video and an
    ingested one describe themselves the same way.
    """
    result = subprocess.run(
        [
            sys.executable, "-m", "yt_dlp", "--skip-download",
            "--print",
            "%(.{id,title,duration,channel,uploader,webpage_url,thumbnail})j",
            url,
        ],
        capture_output=True, text=True, check=True,
    )
    line = result.stdout.strip().splitlines()[-1] if result.stdout.strip() else ""
    try:
        entry = json.loads(line) if line else {}
    except json.JSONDecodeError as error:
        raise RuntimeError(f"yt-dlp returned no video metadata for {url}") from error

    video_id = entry.get("id")
    if not video_id:
        raise RuntimeError(f"yt-dlp returned no video id for {url}")
    return {
        "video_id": video_id,
        "title": entry.get("title") or video_id,
        "duration_sec": entry.get("duration"),
        "channel": entry.get("channel") or entry.get("uploader"),
        "thumbnail_url": entry.get("thumbnail"),
        "url": entry.get("webpage_url")
        or f"https://www.youtube.com/watch?v={video_id}",
    }


def resolve_video_id(url: str) -> str:
    """No bytes downloaded -- lets callers check "already ingested" before
    spending any bandwidth."""
    return resolve_video_meta(url)["video_id"]


def write_video_meta(video_dir: Path, meta: dict) -> Path:
    """Record who this video is, once, beside its clips.

    ingested_at is added here rather than by the caller so every meta.json
    carries the same field, whichever stage wrote it.
    """
    return write_video_meta_file(
        video_dir,
        {**meta, "ingested_at": datetime.now(UTC).isoformat()},
    )


def ensure_video_meta(url: str, video_dir: Path) -> Path | None:
    """Write meta.json unless the video already has one.

    Returns None when nothing was written, so a resumed ingest neither pays
    for a second yt-dlp request nor overwrites a title a person corrected by
    hand.
    """
    if read_video_meta(video_dir):
        return None
    return write_video_meta(video_dir, resolve_video_meta(url))


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
            [
                sys.executable, "-m", "yt_dlp", "-f", "bestaudio",
                "-o", raw_template, url,
            ],
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


PAD_BEFORE_SEC = 0.15
PAD_AFTER_SEC = 0.35

# A clip padded down to nothing, or a music/silence passage padded up to
# nothing, both still need a floor and a ceiling -- otherwise a 300s segment
# that fails the duration filter would be retained anyway.
RETAIN_FLOOR_SEC = 0.3
RETAIN_CEILING_SEC = 60.0

ExcludedReason = Literal["", "too_short", "too_long", "low_quality", "no_single_speaker"]


def plan_clips(
    media_duration_sec: float,
    segments: list[dict],
    min_duration: float = 1.0,
    max_duration: float = 30.0,
) -> list[dict]:
    """Plan clip boundaries from transcript segments -- pure metadata, no
    ffmpeg, no filesystem. Cutting is deferred to commit time (see
    core/review_workflow.py), so this only has to get the numbers right.

    Padding is applied after the duration filter runs on each segment's
    *unpadded* span, so tuning PAD_BEFORE_SEC/PAD_AFTER_SEC never shifts
    which segments pass the filter or which clip_id a surviving segment gets
    -- clip_id is the transcript index, not a count of survivors.

    Segments outside [min_duration, max_duration] are retained as keep=0
    rows with excluded_reason set, instead of being dropped outright: a short
    interjection is often the cleanest audio in the video and was previously
    unrecoverable once cut. RETAIN_FLOOR_SEC/RETAIN_CEILING_SEC bound how far
    that retention goes, so a multi-minute music passage still gets dropped.
    """
    clips = []
    for i, seg in enumerate(segments):
        unpadded_duration = seg["end"] - seg["start"]
        clip_id = f"clip_{i + 1:04d}"

        excluded_reason: ExcludedReason = ""
        if unpadded_duration < min_duration:
            excluded_reason = "too_short"
        elif unpadded_duration > max_duration:
            excluded_reason = "too_long"

        if excluded_reason and not (
            RETAIN_FLOOR_SEC <= unpadded_duration <= RETAIN_CEILING_SEC
        ):
            print(
                f"  Dropping segment {i + 1}: duration {unpadded_duration:.1f}s "
                "outside the retention bounds"
            )
            continue

        start = max(0.0, seg["start"] - PAD_BEFORE_SEC)
        end = min(media_duration_sec, seg["end"] + PAD_AFTER_SEC)

        # Clamp into the gap to each neighbour, never past its midpoint --
        # otherwise an unclamped pad on a diarized video reaches into the
        # next speaker's first phoneme and that clip becomes cross-talk.
        if i > 0:
            gap_start_midpoint = (segments[i - 1]["end"] + seg["start"]) / 2
            start = max(start, gap_start_midpoint)
        if i + 1 < len(segments):
            gap_end_midpoint = (seg["end"] + segments[i + 1]["start"]) / 2
            end = min(end, gap_end_midpoint)

        clips.append({
            "clip_id": clip_id,
            "start": start,
            "end": end,
            "duration": end - start,
            "text": seg["text"],
            "excluded_reason": excluded_reason,
        })

    return clips
