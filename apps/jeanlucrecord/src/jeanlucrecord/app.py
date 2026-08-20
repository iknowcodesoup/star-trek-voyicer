"""HTTP control surface for the voice factory -- composition root only.

Exists so an outside orchestrator can drive the pipeline that cli.py already
implements. It does not reimplement any stage: every job spawns
`python -m jeanlucrecord.cli <character> --stage <stage>` as a child process
and tails its output. cli.py stays the single definition of what each stage
does, and the command line keeps working unchanged.

This module only builds `app`, wires `lifespan`, adds CORS middleware, and
registers each domain router. The routes themselves live under routes/, the
long-lived job/webhook machinery lives under core/ and infrastructure/, and
shared filesystem layout lives in infrastructure/filesystem_layout.py.

Run it with:  just serve-jeanlucrecord
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from jeanlucrecord import config
from jeanlucrecord.core.job_runner import JobRunner
from jeanlucrecord.infrastructure import filesystem_layout
from jeanlucrecord.infrastructure.webhook_gateway import WebhookNotifier
from jeanlucrecord.routes import (
    characters,
    health,
    jobs,
    search,
    speaker_map,
    training,
    videos,
)

# Every job inherits this process's environment, so loading .env here is what
# gets HF_TOKEN to the diarization subprocess. The token is never passed as a
# command line argument: job commands are stored and served by /jobs.
config.load_app_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.webhook_notifier = WebhookNotifier(
        url=os.environ.get(config.WEBHOOK_URL_ENV_VAR),
        token=os.environ.get(config.WEBHOOK_TOKEN_ENV_VAR),
        progress_interval=float(
            os.environ.get(
                config.PROGRESS_INTERVAL_ENV_VAR,
                config.DEFAULT_PROGRESS_INTERVAL_SECONDS,
            )
        ),
    )
    await app.state.webhook_notifier.start()
    app.state.job_runner = JobRunner(
        filesystem_layout.JOB_STATE_PATH, app.state.webhook_notifier
    )
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
        for origin in os.environ.get(config.CORS_ALLOW_ORIGINS_ENV_VAR, "").split(",")
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
