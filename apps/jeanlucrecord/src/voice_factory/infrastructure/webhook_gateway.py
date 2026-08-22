"""Tells the orchestrator when a job changes, so it need not keep asking.

Every send is best effort. A webhook that fails is logged and forgotten:
training runs for days and a job must never die because the orchestrator was
restarting. The orchestrator reconciles on a timer as well, so a lost webhook
costs latency and nothing else.
"""

import asyncio
import sys
from pathlib import Path

import httpx

from voice_factory.core.training_log_reader import parse_training_log

WEBHOOK_TOKEN_HEADER = "X-Voice-Factory-Token"
WEBHOOK_TIMEOUT_SECONDS = 10.0


class WebhookNotifier:
    def __init__(self, url: str | None, token: str | None, progress_interval: float):
        self._url = url.rstrip("/") if url else None
        self._token = token
        self.progress_interval = progress_interval
        self._client: httpx.AsyncClient | None = None

    @property
    def enabled(self) -> bool:
        return bool(self._url)

    async def start(self) -> None:
        if self.enabled:
            self._client = httpx.AsyncClient(timeout=WEBHOOK_TIMEOUT_SECONDS)

    async def shutdown(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def send(self, job_id: str, **fields) -> None:
        if self._client is None or self._url is None:
            return
        headers = {WEBHOOK_TOKEN_HEADER: self._token} if self._token else {}
        try:
            response = await self._client.post(
                f"{self._url}/api/voice/jobs/{job_id}/events",
                json={"job_id": job_id, **fields},
                headers=headers,
            )
            response.raise_for_status()
        except httpx.HTTPError as error:
            print(f"webhook for job {job_id} failed: {error}", file=sys.stderr)

    async def watch_progress(self, job_id: str, log_path: Path) -> None:
        """Report epoch and loss while a training job runs.

        Reads the same progress bar the /training endpoint reads, so there is
        one definition of what progress means. Stops when the job does.
        """
        while True:
            await asyncio.sleep(self.progress_interval)
            epoch, loss = await asyncio.to_thread(parse_training_log, log_path)
            if epoch is None and loss is None:
                continue
            await self.send(job_id, type="progress", epoch=epoch, loss=loss)
