from pathlib import Path

CORPUS_PATH = Path(__file__).resolve().parent / "data" / "corpus.txt"


def load_corpus(n: int) -> list[str]:
    lines = CORPUS_PATH.read_text(encoding="utf-8").splitlines()
    lines = [line.strip() for line in lines if line.strip()]
    return lines[:n]


def load_validation_sentences(n: int = 8) -> list[str]:
    # Fixed, held-out sentences for comparing checkpoint quality by ear: taken from
    # the tail of corpus.txt, which sits past CORPUS_SIZE_DEFAULT (1300 of 1500
    # lines), so they're never part of the training dataset. Same slice every call
    # (no randomness) so samples from different checkpoints stay directly comparable.
    lines = CORPUS_PATH.read_text(encoding="utf-8").splitlines()
    lines = [line.strip() for line in lines if line.strip()]
    return lines[-n:]
