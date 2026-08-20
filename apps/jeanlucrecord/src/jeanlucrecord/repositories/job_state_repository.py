"""Persistence for work/_jobs.json, the on-disk record of every job JobRunner
has ever tracked. Pure I/O -- JobRunner in core/job_runner.py owns the
in-memory state and the rules for what gets kept.
"""

import json
from pathlib import Path

from jeanlucrecord.schemas import Job

# how many of the most recent jobs to keep on disk -- JobRunner passes this
# through so the retention rule stays visible at the call site
JOB_HISTORY_LIMIT = 200


def load_jobs(state_path: Path) -> list[Job]:
    if not state_path.exists():
        return []
    stored = json.loads(state_path.read_text(encoding="utf-8"))
    return [Job(**record) for record in stored]


def save_jobs(state_path: Path, jobs: list[Job]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    recent = sorted(jobs, key=lambda job: job.started_at)[-JOB_HISTORY_LIMIT:]
    state_path.write_text(
        json.dumps([job.model_dump() for job in recent], indent=2),
        encoding="utf-8",
    )
