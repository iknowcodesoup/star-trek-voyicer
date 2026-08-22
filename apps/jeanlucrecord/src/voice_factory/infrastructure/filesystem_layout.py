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
# WORK_DIR points at a directory that does not exist. Every route that reads
# the filesystem calls require_work_dir() first, so that mistake now answers
# 500 naming the path instead of an empty list a dashboard cannot tell from a
# fresh install. /health still returns WORK_DIR, and still answers either way.
APP_DIR = Path(__file__).resolve().parent.parent.parent.parent
WORK_DIR = APP_DIR / "work"
JOB_STATE_PATH = WORK_DIR / "_jobs.json"

# character and video ids reach the filesystem, so keep them to characters that
# cannot escape work/ -- no dots, no separators
SAFE_NAME = re.compile(r"^[A-Za-z0-9_-]+$")


def require_work_dir() -> Path:
    """The work directory, or a 500 that names the path it looked for.

    An absent WORK_DIR is a deployment fault, never a fresh install. A fresh
    install has the directory and nothing under it. Reporting the fault as an
    empty collection is what let a bad path stay invisible behind a
    dashboard that kept rendering its own stale copy of the same facts.
    """
    if not WORK_DIR.exists():
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            f"WORK_DIR does not exist: {WORK_DIR}",
        )
    return WORK_DIR


def check_name(value: str | None, label: str) -> str:
    if not value or not SAFE_NAME.match(value):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            f"{label} must match {SAFE_NAME.pattern}",
        )
    return value
