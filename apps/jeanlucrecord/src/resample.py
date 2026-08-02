from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def resample_dir(in_dir: Path, out_dir: Path, target_rate: int = 22050, gain: float = 0.95) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for wav_path in sorted(in_dir.glob("*.wav")):
        data, rate = sf.read(wav_path)
        if rate != target_rate:
            up = target_rate
            down = rate
            data = resample_poly(data, up, down)
        sf.write(out_dir / wav_path.name, data * gain, target_rate)


def normalize_ref_wav(in_path: Path, out_path: Path) -> Path:
    """Re-encode as PCM16 so librosa/soundfile can seek it.

    Voice clips pulled from old games sometimes use compressed WAV
    codecs (e.g. GSM 6.10) that libsndfile can only read forward,
    which breaks Chatterbox's internal `librosa.load()` call.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with sf.SoundFile(in_path) as f:
        samplerate = f.samplerate
        blocks = []
        while True:
            block = f.read(65536, dtype="float32", always_2d=False)
            if len(block) == 0:
                break
            blocks.append(block)
    data = np.concatenate(blocks) if blocks else np.zeros(0, dtype="float32")
    sf.write(out_path, data, samplerate, subtype="PCM_16")
    return out_path
