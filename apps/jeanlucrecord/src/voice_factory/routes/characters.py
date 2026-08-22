"""Which characters have a dataset started, independent of any run."""

from fastapi import APIRouter

from voice_factory.infrastructure.filesystem_layout import require_work_dir

router = APIRouter(tags=["Characters"])


@router.get("/characters")
async def get_characters() -> dict:
    work_dir = require_work_dir()
    characters = sorted(
        entry.name
        for entry in work_dir.iterdir()
        if entry.is_dir() and not entry.name.startswith("_")
    )
    return {"characters": characters}
