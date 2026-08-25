"""Asks the orchestrator which clips a voice is made of.

The reviewer's decisions - which clips to keep, what each one says, where it
starts and ends, and which voice it trains - live in the orchestrator's
Postgres. They used to live in review.csv beside the audio here, which meant
the two facts a dataset is built from sat in a file no query could reach and
no transaction protected.

This host still owns the audio. It gets told which slices to cut, and cuts
them from its own full.wav. Nothing about a clip is stored on this side any
more except the sound itself.

Synchronous on purpose: the only caller is cli.py, which runs as a
subprocess stage and has no event loop to await into.
"""

import os

import httpx

from voice_factory.config import (
    ORCHESTRATOR_URL_ENV_VAR,
    WEBHOOK_TOKEN_ENV_VAR,
)

ORCHESTRATOR_TOKEN_HEADER = "X-Voice-Factory-Token"
ORCHESTRATOR_TIMEOUT_SECONDS = 30.0


class OrchestratorUnavailable(Exception):
    """The orchestrator is unset or did not answer.

    Fatal for a compile, unlike a failed webhook. A webhook only reports; this
    call *is* the dataset, and guessing at it would train a voice on the wrong
    audio - or on none.
    """


def fetch_dataset_clips(character: str) -> list[dict]:
    """Every kept clip assigned to this voice, newest decisions included.

    Answers a list of {video_id, clip_id, start_sec, end_sec, text}. An empty
    list is a real answer, not an error: it means nobody has assigned this
    voice any clips yet, and the caller says so in those words.
    """
    base_url = os.environ.get(ORCHESTRATOR_URL_ENV_VAR, "").rstrip("/")
    if not base_url:
        raise OrchestratorUnavailable(
            f"{ORCHESTRATOR_URL_ENV_VAR} is not set, so there is nowhere to read "
            f"{character!r}'s clip assignments from."
        )

    token = os.environ.get(WEBHOOK_TOKEN_ENV_VAR)
    headers = {ORCHESTRATOR_TOKEN_HEADER: token} if token else {}
    try:
        response = httpx.get(
            f"{base_url}/api/voices/by-name/{character}/dataset",
            headers=headers,
            timeout=ORCHESTRATOR_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except httpx.HTTPError as error:
        raise OrchestratorUnavailable(
            f"Could not read {character!r}'s clips from {base_url}: {error}"
        ) from error
    return response.json()
