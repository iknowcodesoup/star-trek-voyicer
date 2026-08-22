"""Liveness probe."""

from fastapi import APIRouter

from voice_factory.infrastructure import filesystem_layout

router = APIRouter(tags=["Health"])


@router.get("/health")
async def get_health() -> dict:
    return {"status": "ok", "work_dir": str(filesystem_layout.WORK_DIR)}
