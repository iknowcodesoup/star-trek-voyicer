"""Runs pyannote speaker diarization and writes the speaker turns as JSON.

Lives in its own environment because pyannote.audio needs a newer torch than
chatterbox-tts allows. src/diarize.py runs this as a subprocess, so nothing in
the main jeanlucrecord environment ever imports pyannote.

Usage:
    python diarize_worker.py --wav full.wav --out diarization.json [--num-speakers N]
"""

import argparse
import json
import os
import sys
from pathlib import Path

DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"
HF_TOKEN_ENV_VAR = "HF_TOKEN"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wav", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--num-speakers", type=int)
    parser.add_argument("--model", default=DIARIZATION_MODEL)
    arguments = parser.parse_args()

    token = os.environ.get(HF_TOKEN_ENV_VAR)
    if not token:
        raise SystemExit(f"{HF_TOKEN_ENV_VAR} is not set")

    # Windows' safe DLL search (Python 3.8+) ignores PATH for a DLL's own
    # dependencies, so torch.ops.load_library() can't find FFmpeg's shared
    # libraries here even with ffmpeg.exe on PATH. torchaudio/pyannote import
    # torchcodec below, which needs this registered explicitly to resolve
    # avcodec/avformat/avutil.
    if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
        ffmpeg_bin = Path.home() / ".local" / "bin"
        if ffmpeg_bin.is_dir():
            os.add_dll_directory(str(ffmpeg_bin))

    import soundfile
    import torch
    from pyannote.audio import Pipeline

    try:
        pipeline = Pipeline.from_pretrained(arguments.model, token=token)
    except Exception as error:
        raise SystemExit(
            f"Could not load {arguments.model}: {error}\n\n"
            "Both models are gated. Accept the terms for each, then set HF_TOKEN to a "
            "read token:\n"
            f"  https://huggingface.co/{arguments.model}\n"
            "  https://huggingface.co/pyannote/segmentation-3.0\n"
            "  https://huggingface.co/settings/tokens"
        ) from error
    if pipeline is None:
        raise SystemExit(
            f"Could not load {arguments.model}. Accept the terms for it and for "
            "pyannote/segmentation-3.0 on huggingface.co, then retry."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Diarizing {arguments.wav.name} with {arguments.model} on {device}...")
    if device.type == "cpu":
        print("  No CUDA device found. Expect this to take about as long as the audio.")
    pipeline.to(device)

    # Read the audio here and hand pyannote a tensor rather than a path. Passing
    # a path makes it decode through torchcodec, which needs FFmpeg's shared
    # libraries on PATH and fails to load them on Windows. soundfile has no such
    # dependency, and download_audio already wrote a plain 22050 Hz mono wav.
    samples, sample_rate = soundfile.read(str(arguments.wav), dtype="float32")
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    waveform = torch.from_numpy(samples).unsqueeze(0)

    annotation = pipeline(
        {"waveform": waveform, "sample_rate": sample_rate},
        num_speakers=arguments.num_speakers,
    )

    turns = [
        {"start": float(segment.start), "end": float(segment.end), "speaker": speaker}
        for segment, _track, speaker in annotation.speaker_diarization.itertracks(
            yield_label=True
        )
    ]
    turns.sort(key=lambda turn: turn["start"])

    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(turns, indent=2), encoding="utf-8")

    speakers = sorted({turn["speaker"] for turn in turns})
    print(
        f"  {len(turns)} turn(s) across {len(speakers)} speaker(s): {', '.join(speakers)}"
    )


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
