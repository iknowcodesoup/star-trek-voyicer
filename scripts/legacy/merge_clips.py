#!/usr/bin/env python3
"""
merge_clips.py — Merge short Piper training clips into 5-10 second segments.

Reads existing metadata.csv + wav/ clips, groups adjacent clips to reach
the target duration, concatenates audio with ffmpeg, and writes merged output.

USAGE
-----
  python scripts/merge_clips.py [--input ./piper-tts/recordings] [--output ./piper-tts/merged]
  python scripts/merge_clips.py --min 5 --max 10

No re-transcription needed — uses existing clips and text.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


def get_duration(wav_path):
    """Get duration in seconds of a WAV file using ffprobe."""
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "quiet", "-print_format", "json",
                "-show_format", str(wav_path),
            ],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            print(f"  WARNING: ffprobe failed for {wav_path}", file=sys.stderr)
            return -1.0
        info = json.loads(result.stdout)
        duration = float(info.get("format", {}).get("duration", 0))
        if duration <= 0:
            print(f"  WARNING: zero/negative duration for {wav_path}", file=sys.stderr)
            return -1.0
        return duration
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"  WARNING: could not read duration for {wav_path}: {e}", file=sys.stderr)
        return -1.0


def read_metadata(metadata_path):
    """Read metadata.csv and return list of (clip_id, text) tuples."""
    entries = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="|", quoting=csv.QUOTE_NONE, escapechar="\\")
        for row in reader:
            if len(row) >= 2:
                entries.append((row[0].strip(), row[1].strip()))
    return entries


def concat_wavs(wav_paths, output_path):
    """Concatenate multiple WAV files into one using ffmpeg concat demuxer."""
    if len(wav_paths) == 1:
        # Just copy
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(wav_paths[0]),
             "-c", "copy", str(output_path)],
            capture_output=True, check=True,
        )
        return

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        for p in wav_paths:
            f.write(f"file '{p}'\n")
        concat_list = f.name

    try:
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
             "-i", concat_list, "-c", "copy", str(output_path)],
            capture_output=True, check=True,
        )
    finally:
        os.unlink(concat_list)


def main():
    parser = argparse.ArgumentParser(
        description="Merge short clips into 5-10 second segments for Piper training.",
    )
    parser.add_argument(
        "--input", default="./piper-tts/recordings",
        help="Input directory with metadata.csv and wav/ (default: ./piper-tts/recordings)",
    )
    parser.add_argument(
        "--output", default="./piper-tts/merged",
        help="Output directory for merged clips (default: ./piper-tts/merged)",
    )
    parser.add_argument("--min", type=float, default=5.0, help="Minimum clip duration in seconds (default: 5)")
    parser.add_argument("--max", type=float, default=10.0, help="Maximum clip duration in seconds (default: 10)")
    parser.add_argument("--from-clip", type=int, default=0, help="Only process clips with number >= this (e.g. 650)")
    args = parser.parse_args()

    input_dir = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()
    min_duration = args.min
    max_duration = args.max

    metadata_path = input_dir / "metadata.full.csv"
    wav_dir = input_dir

    if not metadata_path.exists():
        print(f"ERROR: {metadata_path} not found", file=sys.stderr)
        sys.exit(1)

    entries = read_metadata(metadata_path)
    print(f"Read {len(entries)} clips from metadata.csv")

    # Filter by --from-clip if specified
    if args.from_clip > 0:
        filtered = []
        for clip_id, text in entries:
            # Extract number from clip_NNNN
            num = int(clip_id.split("_")[1])
            if num >= args.from_clip:
                filtered.append((clip_id, text))
        print(f"Filtered to {len(filtered)} clips (>= clip_{args.from_clip:04d})")
        entries = filtered

    # Get durations for all clips
    clips = []
    for clip_id, text in entries:
        wav_path = wav_dir / f"{clip_id}.wav"
        if not wav_path.exists():
            print(f"  WARNING: {wav_path} not found, skipping")
            continue
        duration = get_duration(wav_path)
        if duration < 0:
            print(f"  Skipping {clip_id}: bad duration")
            continue
        clips.append({"id": clip_id, "text": text, "path": wav_path, "duration": duration})

    # Group adjacent clips to reach target duration
    groups = []
    current_group = []
    current_duration = 0.0

    for clip in clips:
        # If adding this clip would exceed max, finalize current group first
        if current_group and current_duration + clip["duration"] > max_duration:
            groups.append(current_group)
            current_group = []
            current_duration = 0.0

        current_group.append(clip)
        current_duration += clip["duration"]

        # If we've reached the minimum, finalize
        if current_duration >= min_duration:
            groups.append(current_group)
            current_group = []
            current_duration = 0.0

    # Don't discard leftover — add as final group
    if current_group:
        groups.append(current_group)

    print(f"\nGrouped into {len(groups)} merged clips")

    # Create output
    out_wav_dir = output_dir / "wav"
    out_wav_dir.mkdir(parents=True, exist_ok=True)

    out_metadata = output_dir / "metadata.csv"
    written = 0

    with open(out_metadata, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="|", quoting=csv.QUOTE_NONE, escapechar="\\")

        for i, group in enumerate(groups):
            merged_id = f"clip_{i + 1:04d}"
            merged_path = out_wav_dir / f"{merged_id}.wav"
            merged_text = " ".join(c["text"] for c in group)
            merged_duration = sum(c["duration"] for c in group)

            source_ids = ", ".join(c["id"] for c in group)
            print(f"  {merged_id} ({merged_duration:.1f}s, {len(group)} clips): {source_ids}")

            wav_paths = [c["path"] for c in group]
            concat_wavs(wav_paths, merged_path)

            writer.writerow([merged_id, merged_text])
            written += 1

    print(f"\nExported {written} merged clip(s) to {out_wav_dir}")
    print(f"Metadata: {out_metadata}")

    # Summary stats
    durations = []
    for group in groups:
        durations.append(sum(c["duration"] for c in group))
    if durations:
        print(f"\nDuration stats:")
        print(f"  Min: {min(durations):.1f}s")
        print(f"  Max: {max(durations):.1f}s")
        print(f"  Avg: {sum(durations) / len(durations):.1f}s")
        under = sum(1 for d in durations if d < min_duration)
        if under:
            print(f"  {under} clip(s) still under {min_duration}s (leftover tail)")


if __name__ == "__main__":
    main()
