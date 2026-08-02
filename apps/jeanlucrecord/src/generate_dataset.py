import re
from pathlib import Path

import torchaudio as ta
import whisper
from chatterbox.tts import ChatterboxTTS
from piper_phonemize import phonemize_espeak


def normalise_text(text: str, gaps: bool = True) -> str:
    return re.sub(r"\W+", " " if gaps else "", text).strip()


def phonemes_match(text: str, transcript: str) -> bool:
    for gaps in (True, False):
        a = phonemize_espeak(normalise_text(text, gaps), "en-us")
        b = phonemize_espeak(normalise_text(transcript, gaps), "en-us")
        if a == b:
            return True
    return False


def generate_dataset(
    character: str,
    ref_wavs: list[Path],
    phrases: list[str],
    out_dir: Path,
    max_attempts: int = 3,
) -> None:
    wav_dir = out_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = out_dir / "metadata.csv"

    print(f"Loading Chatterbox TTS for {character}...")
    model = ChatterboxTTS.from_pretrained(device="cpu")
    print("Loading Whisper (turbo)...")
    whisper_model = whisper.load_model("turbo")

    accepted = []
    for idx, text in enumerate(phrases):
        ref_wav = ref_wavs[idx % len(ref_wavs)]
        fp = wav_dir / f"{idx}.wav"
        print(f"{idx + 1}/{len(phrases)}: {text}")

        verified = False
        for attempt in range(max_attempts):
            wav = model.generate(text, audio_prompt_path=str(ref_wav), exaggeration=0.2)
            ta.save(str(fp), wav, model.sr)

            transcript = whisper_model.transcribe(str(fp))["text"].strip()
            verified = phonemes_match(text, transcript)
            if verified:
                break
            print(f"  attempt {attempt + 1} failed: expected '{text}', got '{transcript}'")

        if not verified:
            print(f"  skipping after {max_attempts} attempts")
            fp.unlink(missing_ok=True)
            continue

        accepted.append((idx, text))

    with open(metadata_path, "w", encoding="utf-8") as f:
        for idx, text in accepted:
            f.write(f"{idx}|{text}\n")

    print(f"Score: {len(accepted)}/{len(phrases)} ({len(accepted) / len(phrases) * 100:.0f}%)")
