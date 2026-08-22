"""Tests for the job-building logic routes/jobs.py depends on.

_build_command lives in core/job_runner.py -- these are pure-function tests,
no HTTP client and no real subprocess needed.
"""

import pytest
from fastapi import HTTPException

from voice_factory.core.job_runner import _build_command
from voice_factory.schemas import JobRequest


def test_build_command_needs_no_character_for_youtube_ingest_stages():
    """FR12/decoupling: starting an ingest job for a video needs no
    character, since the artifacts it writes are not scoped to one.

    Calls _build_command directly rather than POST /jobs, which would spawn a
    real cli.py subprocess -- this is a pure function, no process needed.
    """
    command = _build_command(
        JobRequest(stage="youtube-download", youtube_url="https://example.com/v")
    )

    assert "--stage" in command
    assert command[command.index("--stage") + 1] == "youtube-download"


def test_build_command_still_needs_a_character_for_youtube_commit():
    with pytest.raises(HTTPException) as exc_info:
        _build_command(JobRequest(stage="youtube-commit"))

    assert exc_info.value.status_code == 422
