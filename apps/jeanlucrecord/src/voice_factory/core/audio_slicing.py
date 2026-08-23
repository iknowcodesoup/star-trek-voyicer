"""Slice a wav by time range without cutting a new file for every clip.

full.wav is the one file every clip is judged and cut against once this
story lands -- see core/review_workflow.py's precedence rule. This module is
the one place that turns (start_sec, end_sec) into frames, so the rounding
rule can never drift between the audio route and the commit path.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import soundfile as sf

TARGET_RATE = 22050


def frames_for(seconds: float) -> int:
    return round(seconds * TARGET_RATE)


@contextmanager
def open_reader(path: Path) -> Iterator[sf.SoundFile]:
    with sf.SoundFile(path) as reader:
        yield reader


def read_slice(wav_path: Path, start_sec: float, end_sec: float) -> np.ndarray:
    with open_reader(wav_path) as reader:
        return read_slice_from(reader, start_sec, end_sec)


def read_slice_from(reader: sf.SoundFile, start_sec: float, end_sec: float) -> np.ndarray:
    """Read frames in [start_sec, end_sec) from an already-open reader, for a
    caller slicing many ranges out of one file (stage_youtube_review scoring
    every clip) without reopening it each time.

    The start is clamped to [0, duration] -- the client derives its request
    window from that same clamp, so this and the route's window math must
    agree. The end is left to run past EOF; soundfile just returns fewer
    frames, which is what lets a trailing pad silently shrink at the tail of
    the file instead of raising.
    """
    duration_frames = len(reader)
    start_frame = max(0, min(frames_for(start_sec), duration_frames))
    end_frame = max(start_frame, frames_for(end_sec))
    reader.seek(start_frame)
    frames_to_read = end_frame - start_frame
    return reader.read(frames_to_read, dtype="float32", always_2d=False)


def write_slice(source: Path, out: Path, start_sec: float, end_sec: float) -> int:
    """Write frames in [start_sec, end_sec) from source to out. Returns the
    frame count written."""
    samples = read_slice(source, start_sec, end_sec)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out, samples, TARGET_RATE, subtype="PCM_16")
    return len(samples)


def write_slice_from(
    reader: sf.SoundFile, out: Path, start_sec: float, end_sec: float
) -> int:
    """Write frames in [start_sec, end_sec) from an already-open reader to
    out. Returns the frame count written.

    For a caller committing many clips out of one video's full.wav
    (commit_reviewed_clips) without reopening the file per clip.
    """
    samples = read_slice_from(reader, start_sec, end_sec)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out, samples, TARGET_RATE, subtype="PCM_16")
    return len(samples)
