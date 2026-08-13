import json
import os
import subprocess
from collections import defaultdict
from pathlib import Path

DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"

# pyannote.audio runs in its own environment: it needs torch>=2.8 and
# chatterbox-tts pins torch==2.6. See diarizer/pyproject.toml.
DIARIZER_DIR = Path(__file__).resolve().parent.parent / "diarizer"

# A clip must be this fraction single-speaker to earn a label. Below it the clip
# is either cross-talk (two speakers split the coverage) or mostly music and
# silence (no turn covers it at all). Both are unusable for training, and both
# fall out of the same number -- see assign_speakers.
MIN_SPEAKER_COVERAGE = 0.9


def diarize(
    wav_path: Path,
    hf_token: str,
    cache_path: Path,
    num_speakers: int | None = None,
) -> list[dict]:
    """Return speaker turns as [{"start", "end", "speaker"}], sorted by start.

    Caches to cache_path and reuses it, so re-running ingest after a crash does
    not repeat the most expensive step in the pipeline. Matches how
    download_audio and chunk_clips skip work they already did.
    """
    if cache_path.exists():
        print(f"  Using cached diarization: {cache_path}")
        return json.loads(cache_path.read_text(encoding="utf-8"))

    command = [
        str(diarizer_python()),
        "diarize_worker.py",
        "--wav",
        str(wav_path.resolve()),
        "--out",
        str(cache_path.resolve()),
        "--model",
        DIARIZATION_MODEL,
    ]
    if num_speakers is not None:
        command += ["--num-speakers", str(num_speakers)]

    # the worker reads the token from the environment, so it never lands in a
    # process listing or a job log
    subprocess.run(
        command,
        cwd=str(DIARIZER_DIR),
        check=True,
        env={**os.environ, "HF_TOKEN": hf_token},
    )

    if not cache_path.exists():
        raise RuntimeError(f"Diarization wrote no output to {cache_path}")
    return json.loads(cache_path.read_text(encoding="utf-8"))


def diarizer_python() -> Path:
    """Path to the interpreter in the diarizer environment."""
    for candidate in (
        DIARIZER_DIR / ".venv" / "Scripts" / "python.exe",
        DIARIZER_DIR / ".venv" / "bin" / "python",
    ):
        if candidate.exists():
            return candidate
    raise SystemExit(
        f"No diarizer environment at {DIARIZER_DIR / '.venv'}. Run: just sync-diarizer"
    )


def assign_speakers(
    clips: list[dict],
    turns: list[dict],
    min_coverage: float = MIN_SPEAKER_COVERAGE,
) -> list[dict]:
    """Label each clip with the speaker who covers most of it.

    Returns new dicts with "speaker_label" and "speaker_coverage" added. A label
    of None means the clip is rejected: no single speaker holds min_coverage of
    it. Pure -- no I/O, so the rejection rule is unit-testable on its own.
    """
    labelled = []
    for clip in clips:
        start = clip["start"]
        end = clip["end"]
        duration = end - start

        overlap_by_speaker: dict[str, float] = defaultdict(float)
        if duration > 0:
            for turn in turns:
                overlap = min(end, turn["end"]) - max(start, turn["start"])
                if overlap > 0:
                    overlap_by_speaker[turn["speaker"]] += overlap

        if overlap_by_speaker:
            speaker, overlap = max(overlap_by_speaker.items(), key=lambda item: item[1])
            # a speaker's own turns can abut or overlap slightly, so the sum can
            # exceed the clip length -- coverage is a fraction, keep it one
            coverage = min(overlap / duration, 1.0)
        else:
            speaker = None
            coverage = 0.0

        labelled.append(
            {
                **clip,
                "speaker_label": speaker if coverage >= min_coverage else None,
                "speaker_coverage": coverage,
            }
        )
    return labelled


def count_by_speaker(clips: list[dict]) -> dict[str, int]:
    """Clip counts per speaker label, for the ingest summary. Rejected clips
    (speaker_label None) are counted under the key "rejected"."""
    counts: dict[str, int] = defaultdict(int)
    for clip in clips:
        counts[clip.get("speaker_label") or "rejected"] += 1
    return dict(sorted(counts.items()))
