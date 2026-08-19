"""Liveness probe."""

from fastapi import APIRouter

import fs_paths

router = APIRouter(tags=["Health"])


@router.get("/health")
async def get_health() -> dict:
    return {"status": "ok", "work_dir": str(fs_paths.WORK_DIR)}
