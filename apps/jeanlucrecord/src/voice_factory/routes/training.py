"""Training progress and sample audio for a character's checkpoints."""

import asyncio
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse

from voice_factory.core.job_runner import JobRunner
from voice_factory.core.training_log_reader import (
    parse_checkpoint_name,
    parse_training_log,
)
from voice_factory.dependencies import get_job_runner
from voice_factory.infrastructure import filesystem_layout

router = APIRouter(tags=["Training"])


@router.get("/characters/{character}/training")
async def get_training(
    character: str, runner: JobRunner = Depends(get_job_runner)  # noqa: B008
) -> dict:
    filesystem_layout.check_name(character, "character")
    training_dir = filesystem_layout.WORK_DIR / character / "training"
    checkpoints = [
        {
            "path": path.relative_to(filesystem_layout.APP_DIR).as_posix(),
            "name": path.name,
            "epoch": parse_checkpoint_name(path.name)[0],
            "step": parse_checkpoint_name(path.name)[1],
            "modified_at": datetime.fromtimestamp(
                path.stat().st_mtime, tz=UTC
            ).isoformat(),
        }
        for path in sorted(
            training_dir.glob("**/*.ckpt"), key=lambda path: path.stat().st_mtime
        )
    ]

    train_jobs = [
        job
        for job in runner.list_jobs(character)
        if job.stage == "train" and job.state == "running"
    ]
    epoch, loss = (None, None)
    if train_jobs:
        epoch, loss = await asyncio.to_thread(
            parse_training_log, runner.log_path(train_jobs[0].job_id)
        )

    return {
        "character": character,
        "preprocessed": (training_dir / "config.json").exists(),
        "running_job_id": train_jobs[0].job_id if train_jobs else None,
        "current_epoch": epoch,
        "current_loss": loss,
        "checkpoints": checkpoints,
    }


@router.get("/characters/{character}/samples")
async def get_samples(character: str) -> dict:
    filesystem_layout.check_name(character, "character")
    samples_dir = filesystem_layout.WORK_DIR / character / "checkpoint_samples"
    if not samples_dir.exists():
        return {"character": character, "samples": {}}
    samples = {
        checkpoint_dir.name: sorted(path.name for path in checkpoint_dir.glob("*.wav"))
        for checkpoint_dir in sorted(samples_dir.iterdir())
        if checkpoint_dir.is_dir()
    }
    return {"character": character, "samples": samples}


@router.get("/characters/{character}/samples/{checkpoint_name}/{sample_name}")
async def get_sample_audio(
    character: str, checkpoint_name: str, sample_name: str
) -> FileResponse:
    filesystem_layout.check_name(character, "character")
    filesystem_layout.check_name(checkpoint_name, "checkpoint_name")
    sample_path = (
        filesystem_layout.WORK_DIR
        / character
        / "checkpoint_samples"
        / checkpoint_name
        / sample_name
    )
    # resolve() then compare: sample_name carries a .wav suffix, so SAFE_NAME
    # cannot vet it, and a crafted name must not escape the samples directory
    samples_root = (
        filesystem_layout.WORK_DIR / character / "checkpoint_samples"
    ).resolve()
    if not sample_path.resolve().is_relative_to(samples_root):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid sample name")
    if not sample_path.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No sample {sample_name}")
    return FileResponse(sample_path, media_type="audio/wav")
