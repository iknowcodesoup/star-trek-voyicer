"""There is no pre-existing coverage of clip_quality_score -- this file is
the only proof the refactor from a path argument to samples preserves
scoring behaviour. The load-bearing test compares a slice read out of
full.wav against a file written with exactly that slice's content: if the
two scores diverge, read_slice's framing disagrees with what used to be cut
by ffmpeg."""

import numpy as np
import pytest
import soundfile as sf

from voice_factory.core.audio_slicing import TARGET_RATE, read_slice
from voice_factory.core.quality import FRAME_LENGTH, clip_quality_score


def write_speech_like_wav(path, num_seconds: float, seed: int = 0) -> None:
    """Alternating loud/quiet stretches -- clip_quality_score needs a real
    spread between p90 and p10 to return anything but a degenerate score."""
    rng = np.random.default_rng(seed)
    num_frames = round(num_seconds * TARGET_RATE)
    samples = np.zeros(num_frames, dtype=np.float32)
    chunk = TARGET_RATE // 4
    for start in range(0, num_frames, chunk):
        end = min(start + chunk, num_frames)
        amplitude = 0.8 if (start // chunk) % 2 == 0 else 0.01
        samples[start:end] = rng.uniform(-amplitude, amplitude, end - start)
    sf.write(path, samples, TARGET_RATE, subtype="PCM_16")


def test_scoring_a_slice_of_full_wav_matches_scoring_the_same_range_written_alone(
    tmp_path,
):
    full_wav = tmp_path / "full.wav"
    write_speech_like_wav(full_wav, num_seconds=6.0)

    sliced_samples = read_slice(full_wav, 1.0, 3.0)
    score_from_slice = clip_quality_score(sliced_samples)

    separate_wav = tmp_path / "separate.wav"
    separate_samples, _ = sf.read(full_wav, dtype="float32", always_2d=False)
    start_frame = round(1.0 * TARGET_RATE)
    end_frame = round(3.0 * TARGET_RATE)
    sf.write(
        separate_wav,
        separate_samples[start_frame:end_frame],
        TARGET_RATE,
        subtype="PCM_16",
    )
    score_from_separate_file, _ = sf.read(
        separate_wav, dtype="float32", always_2d=False
    )

    assert score_from_slice == pytest.approx(
        clip_quality_score(score_from_separate_file)
    )


def test_fewer_samples_than_one_frame_returns_zero():
    samples = np.zeros(FRAME_LENGTH - 1, dtype=np.float32)

    assert clip_quality_score(samples) == 0.0
