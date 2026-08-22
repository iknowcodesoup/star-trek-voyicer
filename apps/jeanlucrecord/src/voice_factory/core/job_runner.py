"""Spawns cli.py stages and tracks them.

One asyncio task per job waits on the child and records the outcome. Job
records also land in work/_jobs.json (via repositories/job_state_repository.py)
so a restart can still answer for jobs that finished before it.
"""

import asyncio
import os
import subprocess
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

from fastapi import HTTPException, status

from voice_factory.infrastructure import filesystem_layout
from voice_factory.infrastructure.filesystem_layout import check_name
from voice_factory.infrastructure.webhook_gateway import WebhookNotifier
from voice_factory.repositories.job_state_repository import (
    load_jobs,
    save_jobs,
)
from voice_factory.schemas import YOUTUBE_STAGES_NEEDING_URL, Job, JobRequest


def _now() -> str:
    return datetime.now(UTC).isoformat()


async def _stop_container(container_name: str) -> None:
    process = await asyncio.create_subprocess_exec(
        "docker",
        "stop",
        container_name,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    await process.wait()


def _build_command(request: JobRequest) -> list[str]:
    # youtube-ingest and its five steps act on a video shared across every
    # character, so they need no character at all. Every other stage --
    # including youtube-commit, which fans clips out to characters -- still
    # does. A character is validated whenever one is given, though: it still
    # reaches the filesystem, on the commit side or in the log line here.
    if request.character:
        check_name(request.character, "character")
    elif request.stage not in ("smoketest", *YOUTUBE_STAGES_NEEDING_URL):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            f"stage {request.stage} needs a character",
        )
    if request.stage in YOUTUBE_STAGES_NEEDING_URL and not request.youtube_url:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            f"stage {request.stage} needs a youtube_url",
        )
    if request.stage == "import" and not request.import_dir:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY, "stage import needs an import_dir"
        )

    command = [sys.executable, "-m", "voice_factory.cli"]
    if request.character:
        command.append(request.character)
    command += ["--stage", request.stage]

    optional = {
        "--youtube-url": request.youtube_url,
        "--whisper-model": request.whisper_model,
        "--num-speakers": request.num_speakers,
        "--min-speaker-coverage": request.min_speaker_coverage,
        "--min-clip-duration": request.min_clip_duration,
        "--max-clip-duration": request.max_clip_duration,
        "--quality-flag-threshold": request.quality_flag_threshold,
        "--corpus-size": request.corpus_size,
        "--checkpoint": request.checkpoint,
        "--num-validation-sentences": request.num_validation_sentences,
        "--import-dir": request.import_dir,
    }
    for flag, value in optional.items():
        if value is not None:
            command += [flag, str(value)]
    if request.diarize:
        command.append("--diarize")
    return command


class JobRunner:
    """Spawns cli.py stages and tracks them.

    One asyncio task per job waits on the child and records the outcome. Job
    records also land in work/_jobs.json so a restart can still answer for jobs
    that finished before it.
    """

    def __init__(self, state_path: Path, notifier: WebhookNotifier) -> None:
        self._state_path = state_path
        self._notifier = notifier
        self._jobs: dict[str, Job] = {}
        self._processes: dict[str, asyncio.subprocess.Process] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._progress_tasks: dict[str, asyncio.Task] = {}
        self._load()

    def _load(self) -> None:
        for job in load_jobs(self._state_path):
            # nothing survives a restart, so a job left "running" in the file
            # never completed and never will
            if job.state == "running":
                job.state = "failed"
                job.finished_at = _now()
            self._jobs[job.job_id] = job

    def _save(self) -> None:
        save_jobs(self._state_path, list(self._jobs.values()))

    def log_path(self, job_id: str) -> Path:
        return filesystem_layout.WORK_DIR / "_logs" / f"{job_id}.log"

    def get(self, job_id: str) -> Job:
        job = self._jobs.get(job_id)
        if job is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, f"No job {job_id}")
        return job

    def list_jobs(self, character: str | None = None) -> list[Job]:
        jobs = self._jobs.values()
        if character:
            jobs = [job for job in jobs if job.character == character]
        return sorted(jobs, key=lambda job: job.started_at, reverse=True)

    async def start(self, request: JobRequest) -> Job:
        command = _build_command(request)
        job_id = uuid.uuid4().hex
        log_path = self.log_path(job_id)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # deliberately not a context manager: the file has to stay open until
        # the child process exits, which happens in _wait, long after this
        # returns. _wait closes it in a finally.
        log_file = open(log_path, "wb")  # noqa: SIM115, ASYNC230
        # stderr into the same file: Lightning's progress bar goes to stderr and
        # the training monitor reads epoch and loss out of it
        process = await asyncio.create_subprocess_exec(
            *command,
            cwd=str(filesystem_layout.APP_DIR),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )

        job = Job(
            job_id=job_id,
            character=request.character,
            stage=request.stage,
            state="running",
            started_at=_now(),
            command=command,
        )
        self._jobs[job_id] = job
        self._processes[job_id] = process
        self._tasks[job_id] = asyncio.create_task(self._wait(job_id, process, log_file))
        self._save()

        await self._notifier.send(job_id, type="started", state=job.state)
        # only training writes a progress bar worth reporting
        if self._notifier.enabled and request.stage == "train":
            self._progress_tasks[job_id] = asyncio.create_task(
                self._notifier.watch_progress(job_id, log_path)
            )
        return job

    async def _wait(self, job_id: str, process, log_file) -> None:
        try:
            exit_code = await process.wait()
        finally:
            log_file.close()
        job = self._jobs[job_id]
        if job.state != "cancelled":
            job.state = "succeeded" if exit_code == 0 else "failed"
        job.exit_code = exit_code
        job.finished_at = _now()
        self._processes.pop(job_id, None)
        self._tasks.pop(job_id, None)
        self._stop_progress(job_id)
        self._save()
        await self._notifier.send(job_id, type="finished", state=job.state)

    def _stop_progress(self, job_id: str) -> None:
        progress_task = self._progress_tasks.pop(job_id, None)
        if progress_task is not None:
            progress_task.cancel()

    async def cancel(self, job_id: str) -> Job:
        job = self.get(job_id)
        if job.state != "running":
            return job
        job.state = "cancelled"

        # the GPU stages run inside a named container. Killing cli.py does not
        # stop it, because docker run only forwards signals when it owns the
        # terminal, so stop the container by name as well.
        if job.character:
            await _stop_container(f"jeanlucrecord-trainer-{job.character}-{job.stage}")

        process = self._processes.get(job_id)
        if process is not None:
            process.terminate()
        self._save()
        return job

    async def shutdown(self) -> None:
        for job_id in list(self._processes):
            await self.cancel(job_id)
        for job_id in list(self._progress_tasks):
            self._stop_progress(job_id)
        for task in list(self._tasks.values()):
            await asyncio.gather(task, return_exceptions=True)
