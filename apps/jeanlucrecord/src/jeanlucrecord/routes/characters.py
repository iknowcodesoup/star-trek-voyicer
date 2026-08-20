"""Which characters have a dataset started, independent of any run."""

from fastapi import APIRouter

from jeanlucrecord.infrastructure import filesystem_layout

router = APIRouter(tags=["Characters"])


@router.get("/characters")
async def get_characters() -> dict:
    if not filesystem_layout.WORK_DIR.exists():
        return {"characters": []}
    characters = sorted(
        entry.name
        for entry in filesystem_layout.WORK_DIR.iterdir()
        if entry.is_dir() and not entry.name.startswith("_")
    )
    return {"characters": characters}
