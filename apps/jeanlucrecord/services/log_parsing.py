"""Reads job output the same way from two different callers.

jobs.py tails a running job's raw log. training.py reads the newest epoch and
loss out of that same log, and parses checkpoint filenames. Both need the same
three small parsers, so they live here once instead of twice.
"""

import re
from pathlib import Path

# Lightning writes its progress bar to stderr, e.g.
#   Epoch 42:  73%|###   | 45/62 [00:12<00:04, 3.71it/s, loss=32.1, v_num=3]
EPOCH_PATTERN = re.compile(r"Epoch (\d+):")
LOSS_PATTERN = re.compile(r"loss=([0-9.]+)")
CHECKPOINT_PATTERN = re.compile(r"epoch=(\d+)-step=(\d+)")


def _read_from(path: Path, offset: int) -> bytes:
    with open(path, "rb") as source:
        source.seek(offset)
        return source.read()


def _as_float(value: str | None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_checkpoint(name: str) -> tuple[int | None, int | None]:
    match = CHECKPOINT_PATTERN.search(name)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def _parse_training_log(log_path: Path) -> tuple[int | None, float | None]:
    """Read epoch and loss out of Lightning's progress bar.

    Reads the last 64KB only: a long run writes a large log, and the newest
    progress line is always at the end.
    """
    if not log_path.exists():
        return None, None
    tail_start = max(0, log_path.stat().st_size - 65536)
    tail = _read_from(log_path, tail_start).decode("utf-8", errors="replace")
    # the progress bar redraws with \r, so split on both
    lines = re.split(r"[\r\n]", tail)
    epoch = None
    loss = None
    for line in reversed(lines):
        if epoch is None:
            epoch_match = EPOCH_PATTERN.search(line)
            if epoch_match:
                epoch = int(epoch_match.group(1))
        if loss is None:
            loss_match = LOSS_PATTERN.search(line)
            if loss_match:
                loss = float(loss_match.group(1))
        if epoch is not None and loss is not None:
            break
    return epoch, loss
