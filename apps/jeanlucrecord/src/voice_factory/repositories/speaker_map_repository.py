"""Persistence for one video's speaker_map.json.

Pure I/O, no conflict rules -- see core/review_workflow.py for the merge and
conflict-detection logic that reads and writes through these functions.
"""

import json
from pathlib import Path

SPEAKER_MAP_FILENAME = "speaker_map.json"


def read_speaker_map(video_dir: Path) -> dict[str, str | None]:
    map_path = video_dir / SPEAKER_MAP_FILENAME
    if not map_path.exists():
        return {}
    return json.loads(map_path.read_text(encoding="utf-8"))


def write_speaker_map_file(video_dir: Path, speaker_map: dict[str, str | None]) -> Path:
    map_path = video_dir / SPEAKER_MAP_FILENAME
    map_path.parent.mkdir(parents=True, exist_ok=True)
    map_path.write_text(json.dumps(speaker_map, indent=2), encoding="utf-8")
    return map_path
