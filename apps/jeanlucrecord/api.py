"""HTTP control surface for the voice factory -- composition root only.

Exists so an outside orchestrator can drive the pipeline that main.py already
implements. It does not reimplement any stage: every job spawns
`python main.py <character> --stage <stage>` as a child process and tails its
output. main.py stays the single definition of what each stage does, and the
command line keeps working unchanged.

This module only builds `app`, wires `lifespan`, adds CORS middleware, and
registers each domain router. The routes themselves live under routers/, the
long-lived job/webhook machinery lives under services/, and shared filesystem
layout lives in fs_paths.py.

Run it with:  just serve-jeanlucrecord
"""

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from dotenv import load_dotenv

import fs_paths
from routers import (
    characters,
    health,
    jobs,
    search,
    speaker_map,
    training,
    videos,
)
from services.job_runner import JobRunner
from services.webhook_notifier import WebhookNotifier

# Every job inherits this process's environment, so loading .env here is what
# gets HF_TOKEN to the diarization subprocess. The token is never passed as a
# command line argument: job commands are stored and served by /jobs.
load_dotenv(fs_paths.APP_DIR / ".env", override=False)

CORS_ALLOW_ORIGINS_ENV_VAR = "VOICE_FACTORY_CORS_ALLOW_ORIGINS"

# Where to report job changes, so the orchestrator does not have to poll for
# them. Unset, nothing here changes: jobs run exactly as before and the
# orchestrator falls back to asking.
WEBHOOK_URL_ENV_VAR = "VOICE_ORCHESTRATOR_WEBHOOK_URL"
WEBHOOK_TOKEN_ENV_VAR = "VOICE_WEBHOOK_TOKEN"
PROGRESS_INTERVAL_ENV_VAR = "VOICE_PROGRESS_INTERVAL_SECONDS"
DEFAULT_PROGRESS_INTERVAL_SECONDS = 30.0


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.webhook_notifier = WebhookNotifier(
        url=os.environ.get(WEBHOOK_URL_ENV_VAR),
        token=os.environ.get(WEBHOOK_TOKEN_ENV_VAR),
        progress_interval=float(
            os.environ.get(
                PROGRESS_INTERVAL_ENV_VAR, DEFAULT_PROGRESS_INTERVAL_SECONDS
            )
        ),
    )
    await app.state.webhook_notifier.start()
    app.state.job_runner = JobRunner(fs_paths.JOB_STATE_PATH, app.state.webhook_notifier)
    try:
        yield
    finally:
        await app.state.job_runner.shutdown()
        await app.state.webhook_notifier.shutdown()


app = FastAPI(title="jeanlucrecord control api", lifespan=lifespan)

# empty by default: pythonapi calls this server to server and proxies clip audio
# on to the browser, so no browser origin needs direct access
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in os.environ.get(CORS_ALLOW_ORIGINS_ENV_VAR, "").split(",")
        if origin.strip()
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(search.router)
app.include_router(characters.router)
app.include_router(jobs.router)
app.include_router(videos.router)
app.include_router(speaker_map.router)
app.include_router(training.router)
