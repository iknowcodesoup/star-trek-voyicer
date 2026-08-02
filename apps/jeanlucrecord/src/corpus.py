from pathlib import Path

CORPUS_PATH = Path(__file__).resolve().parent / "data" / "corpus.txt"


def load_corpus(n: int) -> list[str]:
    lines = CORPUS_PATH.read_text(encoding="utf-8").splitlines()
    lines = [line.strip() for line in lines if line.strip()]
    return lines[:n]
