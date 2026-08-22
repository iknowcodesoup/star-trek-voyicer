"""Persistence for one video's meta.json.

Pure I/O, no merge rules -- see core/clip_review.py's video_summary for how an
absent file turns into null fields instead of a raise.

The title belongs to the video, not to any run that claims it. A video is
ingested once and shared, so storing the title beside the clips is what lets
the second character to claim it see the same name as the first.
"""

import json
from pathlib import Path

META_FILENAME = "meta.json"


def read_video_meta(video_dir: Path) -> dict:
    meta_path = video_dir / META_FILENAME
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))


def write_video_meta_file(video_dir: Path, meta: dict) -> Path:
    meta_path = video_dir / META_FILENAME
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return meta_path
