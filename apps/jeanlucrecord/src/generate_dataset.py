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


def load_indexed(path: Path) -> dict[int, str]:
    if not path.exists():
        return {}
    indexed = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        idx_str, text = line.split("|", 1)
        # entries from an imported/YouTube-committed dataset use non-numeric ids
        # (e.g. "clip_0016", "yt_<video_id>_clip_0007") -- not part of this
        # corpus-index resume ledger, so ignore them rather than crash.
        try:
            idx = int(idx_str)
        except ValueError:
            continue
        indexed[idx] = text
    return indexed


def generate_dataset(
    character: str,
    ref_wavs: list[Path],
    phrases: list[str],
    out_dir: Path,
    max_attempts: int = 3,
    retry_failed: bool = False,
) -> None:
    wav_dir = out_dir / "wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = out_dir / "metadata.csv"
    failed_path = out_dir / "failed.csv"

    # resume support: a prior run (interrupted by a crash, or an earlier smaller
    # --corpus-size smoke test) may have already verified some phrases. Only
    # entries actually recorded in metadata.csv are trusted as done without
    # re-checking; a wav left on disk from an interrupted attempt gets re-verified
    # against Whisper below before being trusted, since we can't tell just from
    # its presence whether that attempt had actually passed.
    #
    # phrases that exhausted max_attempts get recorded in failed.csv so restarts
    # don't burn another max_attempts on a phrase that's already shown it can't
    # pass verification. Pass retry_failed=True (--retry-failed) to give them
    # another shot, e.g. after tuning exaggeration or swapping the reference wav.
    accepted = load_indexed(metadata_path)
    failed = {} if retry_failed else load_indexed(failed_path)
    remaining = [(idx, text) for idx, text in enumerate(phrases) if idx not in accepted and idx not in failed]
    if not remaining:
        print(f"Dataset for {character} already complete: {len(accepted)}/{len(phrases)} verified, {len(failed)} failed.")
        return
    if accepted or failed:
        print(
            f"Resuming dataset for {character}: {len(accepted)}/{len(phrases)} already verified, "
            f"{len(failed)} previously failed (skipped)."
        )

    print(f"Loading Chatterbox TTS for {character}...")
    model = ChatterboxTTS.from_pretrained(device="cpu")
    print("Loading Whisper (turbo)...")
    whisper_model = whisper.load_model("turbo")

    failed_mode = "w" if retry_failed else "a"
    with open(metadata_path, "a", encoding="utf-8") as metadata_file, open(
        failed_path, failed_mode, encoding="utf-8"
    ) as failed_file:
        for idx, text in remaining:
            ref_wav = ref_wavs[idx % len(ref_wavs)]
            fp = wav_dir / f"{idx}.wav"
            print(f"{idx + 1}/{len(phrases)}: {text}")

            verified = False
            if fp.exists():
                transcript = whisper_model.transcribe(str(fp))["text"].strip()
                verified = phonemes_match(text, transcript)
                if verified:
                    print("  reused wav left over from an interrupted run")

            for attempt in range(max_attempts):
                if verified:
                    break
                # exaggeration=0.5 and cfg_weight=0.5 are Chatterbox's own defaults; this
                # pipeline previously generated the whole training corpus at exaggeration=0.2
                # (below default), producing uniformly flat, deadpan reference audio that
                # Piper (VITS) then faithfully memorized. Resemble's documented "expressive
                # speech" recipe is exaggeration=0.7+ paired with a lower cfg_weight (~0.3),
                # since higher exaggeration otherwise speeds up delivery.
                wav = model.generate(text, audio_prompt_path=str(ref_wav), exaggeration=0.7, cfg_weight=0.3)
                ta.save(str(fp), wav, model.sr)

                transcript = whisper_model.transcribe(str(fp))["text"].strip()
                verified = phonemes_match(text, transcript)
                if verified:
                    break
                print(f"  attempt {attempt + 1} failed: expected '{text}', got '{transcript}'")

            if not verified:
                print(f"  skipping after {max_attempts} attempts")
                fp.unlink(missing_ok=True)
                failed[idx] = text
                failed_file.write(f"{idx}|{text}\n")
                failed_file.flush()
                continue

            accepted[idx] = text
            metadata_file.write(f"{idx}|{text}\n")
            metadata_file.flush()

    print(f"Score: {len(accepted)}/{len(phrases)} ({len(accepted) / len(phrases) * 100:.0f}%), {len(failed)} failed.")
