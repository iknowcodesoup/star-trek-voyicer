import numpy as np

FLAG_THRESHOLD_DB = 18.0

FRAME_LENGTH = 1024
HOP_LENGTH = 512


def clip_quality_score(
    samples: np.ndarray, frame_length: int = FRAME_LENGTH, hop_length: int = HOP_LENGTH
) -> float:
    """p90(frame RMS dB) - p10(frame RMS dB).

    Clean narration has a big gap between loud speech frames and quiet gaps
    between words/sentences. Continuous background noise, music, or cross-talk
    fills in the quiet frames too, compressing that range. Higher score = cleaner.

    Takes samples, not a path -- both call sites (stage_youtube_review,
    patch_clips's rescoring) already hold the slice in memory via
    audio_slicing.read_slice, and a path-taking overload would let a caller
    reread a file it already read.
    """
    data = samples
    if data.ndim > 1:
        data = data.mean(axis=1)
    if len(data) < frame_length:
        return 0.0

    num_frames = 1 + (len(data) - frame_length) // hop_length
    frames = np.lib.stride_tricks.as_strided(
        data,
        shape=(num_frames, frame_length),
        strides=(data.strides[0] * hop_length, data.strides[0]),
    )
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
    db = 20 * np.log10(rms + 1e-10)
    return float(np.percentile(db, 90) - np.percentile(db, 10))


def is_flagged(score: float, threshold: float = FLAG_THRESHOLD_DB) -> bool:
    return score < threshold
