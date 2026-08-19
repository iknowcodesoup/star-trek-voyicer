"""Which characters have a dataset started, independent of any run."""

from fastapi import APIRouter

import fs_paths

router = APIRouter(tags=["Characters"])


@router.get("/characters")
async def get_characters() -> dict:
    if not fs_paths.WORK_DIR.exists():
        return {"characters": []}
    characters = sorted(
        entry.name
        for entry in fs_paths.WORK_DIR.iterdir()
        if entry.is_dir() and not entry.name.startswith("_")
    )
    return {"characters": characters}
