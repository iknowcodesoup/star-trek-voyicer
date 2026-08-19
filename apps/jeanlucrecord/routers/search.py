"""Video lookup that needs no ingested state: search YouTube, or resolve a
URL to the id every other route keys its filesystem layout on."""

import asyncio

from fastapi import APIRouter, HTTPException, Query, status

from search import SEARCH_LIMIT_DEFAULT, search_videos
from youtube_ingest import resolve_video_id

router = APIRouter(tags=["Search"])


@router.get("/search")
async def get_search(
    query: str = Query(min_length=1),
    limit: int = Query(default=SEARCH_LIMIT_DEFAULT, ge=1, le=50),
) -> dict:
    videos = await asyncio.to_thread(search_videos, query, limit)
    return {"query": query, "videos": videos}


@router.get("/resolve")
async def get_resolve(url: str = Query(min_length=1)) -> dict:
    """Resolve a video URL to its id without downloading anything.

    An orchestrator needs the id up front: it names the directory every later
    call reads from, and only yt-dlp can derive it from an arbitrary URL.
    """
    try:
        video_id = await asyncio.to_thread(resolve_video_id, url)
    except Exception as error:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, f"Could not resolve {url}: {error}"
        ) from error
    return {"url": url, "video_id": video_id}
