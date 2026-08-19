"""Start, list, and inspect main.py stage runs.

Every route delegates to the injected JobRunner, which owns the child
processes and their on-disk state -- see services/job_runner.py.
"""

import asyncio

from fastapi import APIRouter, Depends, Query, status

from models import Job, JobRequest
from services.job_runner import JobRunner, get_job_runner
from services.log_parsing import _read_from

router = APIRouter(tags=["Jobs"])


@router.post("/jobs", status_code=status.HTTP_202_ACCEPTED)
async def post_job(
    job_request: JobRequest, runner: JobRunner = Depends(get_job_runner)  # noqa: B008
) -> Job:
    return await runner.start(job_request)


@router.get("/jobs")
async def get_jobs(
    character: str | None = None,
    runner: JobRunner = Depends(get_job_runner),  # noqa: B008
) -> dict:
    return {"jobs": runner.list_jobs(character)}


@router.get("/jobs/{job_id}")
async def get_job(
    job_id: str, runner: JobRunner = Depends(get_job_runner)  # noqa: B008
) -> Job:
    return runner.get(job_id)


@router.get("/jobs/{job_id}/logs")
async def get_job_logs(
    job_id: str,
    offset: int = Query(default=0, ge=0),
    runner: JobRunner = Depends(get_job_runner),  # noqa: B008
) -> dict:
    job = runner.get(job_id)
    log_path = runner.log_path(job_id)
    if not log_path.exists():
        return {"offset": 0, "content": "", "state": job.state}

    # a long training run writes a large log, so keep the read off the event loop
    chunk = await asyncio.to_thread(_read_from, log_path, offset)
    return {
        "offset": offset + len(chunk),
        "content": chunk.decode("utf-8", errors="replace"),
        "state": job.state,
    }


@router.delete("/jobs/{job_id}")
async def delete_job(
    job_id: str, runner: JobRunner = Depends(get_job_runner)  # noqa: B008
) -> Job:
    return await runner.cancel(job_id)
