"""Filesystem layout shared by every route.

Owns WORK_DIR as its source of truth. Every other module reads
filesystem_layout.WORK_DIR at call time instead of importing the value
directly, so tests/conftest.py's work_dir fixture -- which monkeypatches this
module's attribute -- reaches every code path that touches the filesystem,
including ones several routes away from app.py itself.
"""

import re
from pathlib import Path

from fastapi import HTTPException, status

# Four parents up from src/voice_factory/infrastructure/ is the app root
# (apps/jeanlucrecord/). The nesting depth is load-bearing: move this file and
# WORK_DIR silently points at a directory that does not exist, which every
# route here reports as an empty result rather than an error. GET /videos
# answering {"videos": []} on a machine with ingested videos means this
# resolved wrong -- check /health, which returns WORK_DIR for that reason.
APP_DIR = Path(__file__).resolve().parent.parent.parent.parent
WORK_DIR = APP_DIR / "work"
JOB_STATE_PATH = WORK_DIR / "_jobs.json"

# character and video ids reach the filesystem, so keep them to characters that
# cannot escape work/ -- no dots, no separators
SAFE_NAME = re.compile(r"^[A-Za-z0-9_-]+$")


def check_name(value: str | None, label: str) -> str:
    if not value or not SAFE_NAME.match(value):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            f"{label} must match {SAFE_NAME.pattern}",
        )
    return value
