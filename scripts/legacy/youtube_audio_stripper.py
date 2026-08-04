#!/usr/bin/env python3
"""
youtube_audio_stripper.py — Download YouTube audio, transcribe with Whisper,
and split into sentence-aligned clips with metadata.csv for Piper training.

USAGE
-----
  python youtube_audio_stripper.py <youtube_url> [--model medium] [--output ./piper-tts/recordings]

DEPENDENCIES
------------
  pip install faster-whisper yt-dlp
  winget install Gyan.FFmpeg OpenJS.NodeJS
  ffmpeg must be on PATH

OUTPUT
------
  <output>/
    metadata.csv          id|text (Piper LJSpeech format)
    wav/
      clip_001.wav
      clip_002.wav
      ...
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


def run(cmd):
    print(f"\n>>> {cmd}")
    subprocess.run(cmd, shell=True, check=True)


def download_audio(url, output_dir):
    """Download best audio from YouTube and convert to 22050 Hz mono WAV.

    Caches full.wav in output_dir so re-runs skip the download + convert step.
    """
    cached_wav = os.path.join(output_dir, "full.wav")
    if os.path.exists(cached_wav):
        print(f"  Using cached audio: {cached_wav}")
        return cached_wav

    with tempfile.TemporaryDirectory(prefix="yt-dl-") as dl_dir:
        raw_template = os.path.join(dl_dir, "raw.%(ext)s")
        run(f'{sys.executable} -m yt_dlp -f bestaudio -o "{raw_template}" "{url}"')

        import glob
        candidates = glob.glob(os.path.join(dl_dir, "raw.*"))
        if not candidates:
            print("ERROR: No audio file downloaded.", file=sys.stderr)
            sys.exit(1)

        raw_file = candidates[0]

        # Convert to 22050 Hz mono (Piper's expected sample rate) with loudnorm
        run(f'ffmpeg -y -i "{raw_file}" -ar 22050 -ac 1 -af loudnorm "{cached_wav}"')

    print(f"  Cached audio to: {cached_wav}")
    return cached_wav


def transcribe(wav_path, model_size):
    """Transcribe audio using faster-whisper, returning sentence segments."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        print("ERROR: faster-whisper not installed — run: pip install faster-whisper",
              file=sys.stderr)
        sys.exit(1)

    print(f"\nTranscribing with faster-whisper ({model_size}) ...")
    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, info = model.transcribe(wav_path, beam_size=5, word_timestamps=False)

    results = []
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        results.append({
            "start": segment.start,
            "end": segment.end,
            "text": text,
        })
        print(f"  [{segment.start:7.2f} - {segment.end:7.2f}] {text}")

    print(f"\n  {len(results)} segment(s) found.")
    return results


def sanitize_text(text):
    """Clean transcript text for Piper training metadata."""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def split_and_export(wav_path, segments, output_dir):
    """Split audio at segment boundaries and write clips + metadata.csv."""
    wav_dir = output_dir / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = output_dir / "metadata.csv"
    written = 0

    with open(metadata_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="|", quoting=csv.QUOTE_NONE, escapechar="\\")

        for i, seg in enumerate(segments):
            clip_id = f"clip_{i + 1:04d}"
            clip_path = wav_dir / f"{clip_id}.wav"
            duration = seg["end"] - seg["start"]

            # Skip very short or very long segments
            if duration < 1.0:
                print(f"  Skipping {clip_id}: too short ({duration:.1f}s)")
                continue
            if duration > 30.0:
                print(f"  Skipping {clip_id}: too long ({duration:.1f}s)")
                continue

            text = sanitize_text(seg["text"])
            if not text:
                continue

            run(
                f'ffmpeg -y -i "{wav_path}" '
                f'-ss {seg["start"]:.3f} -to {seg["end"]:.3f} '
                f'-ar 22050 -ac 1 -c:a pcm_s16le "{clip_path}"'
            )

            writer.writerow([clip_id, text])
            written += 1

    print(f"\n  Exported {written} clip(s) to {wav_dir}")
    print(f"  Metadata: {metadata_path}")
    return written


def main():
    parser = argparse.ArgumentParser(
        description="Download YouTube audio, transcribe, and split into Piper training clips.",
    )
    parser.add_argument("url", help="YouTube video URL")
    parser.add_argument(
        "--model", default="medium",
        help="Whisper model size: tiny, base, small, medium, large-v3 (default: medium)",
    )
    parser.add_argument(
        "--output", default="./piper-tts/recordings",
        help="Output directory for clips + metadata.csv (default: ./piper-tts/recordings)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    wav_path = download_audio(args.url, str(output_dir))
    segments = transcribe(wav_path, args.model)

    if not segments:
        print("ERROR: No segments found in transcription.", file=sys.stderr)
        sys.exit(1)

    count = split_and_export(wav_path, segments, output_dir)

    if count == 0:
        print("\nWARNING: No clips exported (all segments were too short or too long).")
        sys.exit(1)

    print(f"\nDone! {count} clips ready for Piper training in {output_dir}")
    print(f"\nNext — train with Podman:")
    print(f"  python scripts/create-voice-model.py train \\")
    print(f"    --audio {output_dir} \\")
    print(f"    --voice my-voice \\")
    print(f"    --output ./models \\")
    print(f"    --checkpoint ./checkpoints/en_US-lessac-medium.ckpt")
    print(f"\n  Add --gpu if you have CUDA available.")


if __name__ == "__main__":
    main()
