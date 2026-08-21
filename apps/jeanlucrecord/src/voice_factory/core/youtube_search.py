import json
import subprocess
import sys

SEARCH_LIMIT_DEFAULT = 10


def search_videos(query: str, limit: int = SEARCH_LIMIT_DEFAULT) -> list[dict]:
    """Search YouTube and return candidate videos, newest match first.

    --flat-playlist keeps this to a single search request: yt-dlp reports what
    the results page already lists instead of resolving each video's full
    format list, which would be one extra request per result.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "yt_dlp",
            f"ytsearch{limit}:{query}",
            "--skip-download",
            "--flat-playlist",
            "--dump-json",
            "--ignore-errors",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )

    videos = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            # --ignore-errors lets yt-dlp keep going past a blocked or deleted
            # video, but it still prints a non-JSON warning line to stdout
            continue

        video_id = entry.get("id")
        if not video_id:
            continue

        videos.append(
            {
                "video_id": video_id,
                "title": entry.get("title") or video_id,
                "duration_sec": entry.get("duration"),
                "channel": entry.get("channel") or entry.get("uploader"),
                "thumbnail_url": _best_thumbnail(entry),
                "url": entry.get("url")
                or f"https://www.youtube.com/watch?v={video_id}",
            }
        )
    return videos


def _best_thumbnail(entry: dict) -> str | None:
    # flat-playlist entries carry a thumbnails list ordered smallest first;
    # the plain "thumbnail" key is not always present in that mode
    thumbnail = entry.get("thumbnail")
    if thumbnail:
        return thumbnail
    thumbnails = entry.get("thumbnails") or []
    if not thumbnails:
        return None
    return thumbnails[-1].get("url")
