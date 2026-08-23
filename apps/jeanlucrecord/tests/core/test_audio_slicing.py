"""Wavs are a linear ramp: sample i has value i, so a slice's first and last
sample tells you exactly which frames it covered. read_slice reads back as
float32, and soundfile normalizes PCM_16 to that by dividing by 32768 -- the
expected values below account for that scaling."""

import numpy as np
import soundfile as sf

from voice_factory.core.audio_slicing import TARGET_RATE, read_slice, write_slice

PCM16_SCALE = 32768.0


def write_ramp(path, num_seconds: float) -> None:
    num_frames = round(num_seconds * TARGET_RATE)
    ramp = np.arange(num_frames, dtype=np.int16)
    sf.write(path, ramp, TARGET_RATE, subtype="PCM_16")


def test_read_slice_returns_the_right_frame_count_and_start_sample(tmp_path):
    wav_path = tmp_path / "full.wav"
    write_ramp(wav_path, num_seconds=5.0)

    samples = read_slice(wav_path, 1.0, 2.0)
    start_frame = round(1.0 * TARGET_RATE)

    assert len(samples) == TARGET_RATE
    assert samples[0] == np.float32(start_frame / PCM16_SCALE)


def test_read_slice_clamps_an_end_past_eof(tmp_path):
    wav_path = tmp_path / "full.wav"
    write_ramp(wav_path, num_seconds=2.0)

    samples = read_slice(wav_path, 1.5, 10.0)

    assert len(samples) == round(0.5 * TARGET_RATE)


def test_read_slice_returns_no_frames_for_an_empty_range(tmp_path):
    wav_path = tmp_path / "full.wav"
    write_ramp(wav_path, num_seconds=2.0)

    samples = read_slice(wav_path, 1.0, 1.0)

    assert len(samples) == 0


def test_write_slice_writes_22050_mono_pcm16(tmp_path):
    source = tmp_path / "full.wav"
    write_ramp(source, num_seconds=3.0)
    out = tmp_path / "clip.wav"

    frame_count = write_slice(source, out, 1.0, 2.0)

    assert frame_count == TARGET_RATE
    info = sf.info(out)
    assert info.samplerate == TARGET_RATE
    assert info.channels == 1
    assert info.subtype == "PCM_16"
