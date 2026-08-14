"""HTTP control surface for the voice factory.

Exists so an outside orchestrator can drive the pipeline that main.py already
implements. It does not reimplement any stage: every job spawns
`python main.py <character> --stage <stage>` as a child process and tails its
output. main.py stays the single definition of what each stage does, and the
command line keeps working unchanged.

Run it with:  just serve-jeanlucrecord
"""

import asyncio
import json
import os
import re
import subprocess
import sys
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import httpx
from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from dotenv import load_dotenv  # noqa: E402

from review import (  # noqa: E402
    REVIEW_CSV_NAME,
    REVIEW_FIELDS,
    SPEAKER_MAP_FILENAME,
    SpeakerMapConflict,
    read_review_csv,
    write_review_csv,
    write_speaker_map,
)
from search import SEARCH_LIMIT_DEFAULT, search_videos  # noqa: E402
from youtube_ingest import DIARIZATION_NAME, resolve_video_id  # noqa: E402

APP_DIR = Path(__file__).resolve().parent
WORK_DIR = APP_DIR / "work"
JOB_STATE_PATH = WORK_DIR / "_jobs.json"

# Every job inherits this process's environment, so loading .env here is what
# gets HF_TOKEN to the diarization subprocess. The token is never passed as a
# command line argument: job commands are stored and served by /jobs.
load_dotenv(APP_DIR / ".env", override=False)

# character and video ids reach the filesystem, so keep them to characters that
# cannot escape work/ -- no dots, no separators
SAFE_NAME = re.compile(r"^[A-Za-z0-9_-]+$")

JobState = Literal["running", "succeeded", "failed", "cancelled"]

Stage = Literal[
    "dataset",
    "import",
    "youtube-ingest",
    # youtube-ingest's five steps, each startable on its own so a retry only
    # repeats the step that failed. See YOUTUBE_INGEST_STEPS in main.py.
    "youtube-download",
    "youtube-transcribe",
    "youtube-chunk",
    "youtube-diarize",
    "youtube-review",
    "youtube-commit",
    "resample",
    "preprocess",
    "smoketest",
    "train",
    "export",
    "sample",
]

# Stages that act on one video and so cannot run without a URL. Mirrors
# YOUTUBE_STAGES_NEEDING_URL in main.py.
YOUTUBE_STAGES_NEEDING_URL = (
    "youtube-ingest",
    "youtube-download",
    "youtube-transcribe",
    "youtube-chunk",
    "youtube-diarize",
    "youtube-review",
)

# Lightning writes its progress bar to stderr, e.g.
#   Epoch 42:  73%|###   | 45/62 [00:12<00:04, 3.71it/s, loss=32.1, v_num=3]
EPOCH_PATTERN = re.compile(r"Epoch (\d+):")
LOSS_PATTERN = re.compile(r"loss=([0-9.]+)")
CHECKPOINT_PATTERN = re.compile(r"epoch=(\d+)-step=(\d+)")

CORS_ALLOW_ORIGINS_ENV_VAR = "VOICE_FACTORY_CORS_ALLOW_ORIGINS"

# Where to report job changes, so the orchestrator does not have to poll for
# them. Unset, nothing here changes: jobs run exactly as before and the
# orchestrator falls back to asking.
WEBHOOK_URL_ENV_VAR = "VOICE_ORCHESTRATOR_WEBHOOK_URL"
WEBHOOK_TOKEN_ENV_VAR = "VOICE_WEBHOOK_TOKEN"
PROGRESS_INTERVAL_ENV_VAR = "VOICE_PROGRESS_INTERVAL_SECONDS"
WEBHOOK_TOKEN_HEADER = "X-Voice-Factory-Token"
DEFAULT_PROGRESS_INTERVAL_SECONDS = 30.0
WEBHOOK_TIMEOUT_SECONDS = 10.0


class JobRequest(BaseModel):
    character: str | None = None
    stage: Stage
    youtube_url: str | None = None
    # every optional field stays None so it is only forwarded when set, which
    # keeps main.py's own defaults as the single source of truth
    whisper_model: str | None = None
    diarize: bool = False
    num_speakers: int | None = None
    min_speaker_coverage: float | None = None
    min_clip_duration: float | None = None
    max_clip_duration: float | None = None
    quality_flag_threshold: float | None = None
    corpus_size: int | None = None
    checkpoint: str | None = None
    num_validation_sentences: int | None = None
    import_dir: str | None = None


class Job(BaseModel):
    job_id: str
    character: str | None
    stage: Stage
    state: JobState
    exit_code: int | None = None
    started_at: str
    finished_at: str | None = None
    command: list[str]


class ClipDecision(BaseModel):
    clip_id: str
    keep: bool | None = None
    speaker_label: str | None = None


class ClipDecisionRequest(BaseModel):
    decisions: list[ClipDecision] = Field(min_length=1)


class SpeakerMapRequest(BaseModel):
    # speaker label -> character name. null discards that speaker's clips.
    speaker_map: dict[str, str | None]


class WebhookNotifier:
    """Tells the orchestrator when a job changes, so it need not keep asking.

    Every send is best effort. A webhook that fails is logged and forgotten:
    training runs for days and a job must never die because the orchestrator
    was restarting. The orchestrator reconciles on a timer as well, so a lost
    webhook costs latency and nothing else.
    """

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
            epoch, loss = await asyncio.to_thread(_parse_training_log, log_path)
            if epoch is None and loss is None:
                continue
            await self.send(job_id, type="progress", epoch=epoch, loss=loss)


class JobRunner:
    """Spawns main.py stages and tracks them.

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
        if not self._state_path.exists():
            return
        stored = json.loads(self._state_path.read_text(encoding="utf-8"))
        for record in stored:
            job = Job(**record)
            # nothing survives a restart, so a job left "running" in the file
            # never completed and never will
            if job.state == "running":
                job.state = "failed"
                job.finished_at = _now()
            self._jobs[job.job_id] = job

    def _save(self) -> None:
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        recent = sorted(self._jobs.values(), key=lambda job: job.started_at)[-200:]
        self._state_path.write_text(
            json.dumps([job.model_dump() for job in recent], indent=2),
            encoding="utf-8",
        )

    def log_path(self, job_id: str) -> Path:
        return WORK_DIR / "_logs" / f"{job_id}.log"

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
            cwd=str(APP_DIR),
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

        # the GPU stages run inside a named container. Killing main.py does not
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
        _check_name(request.character, "character")
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

    command = [sys.executable, "main.py"]
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


def _check_name(value: str | None, label: str) -> str:
    if not value or not SAFE_NAME.match(value):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            f"{label} must match {SAFE_NAME.pattern}",
        )
    return value


def _video_dir(video_id: str) -> Path:
    # video artifacts are shared across every character, so one video id names
    # one directory regardless of who ingested it or who claims it next
    _check_name(video_id, "video_id")
    video_dir = WORK_DIR / "youtube" / video_id
    if not video_dir.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No ingested video {video_id}")
    return video_dir


def _review_path(video_id: str) -> Path:
    review_path = _video_dir(video_id) / REVIEW_CSV_NAME
    if not review_path.exists():
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, f"No review.csv for {video_id}, ingest first"
        )
    return review_path


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
    app.state.job_runner = JobRunner(JOB_STATE_PATH, app.state.webhook_notifier)
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


@app.get("/health")
async def get_health() -> dict:
    return {"status": "ok", "work_dir": str(WORK_DIR)}


@app.get("/search")
async def get_search(
    query: str = Query(min_length=1),
    limit: int = Query(default=SEARCH_LIMIT_DEFAULT, ge=1, le=50),
) -> dict:
    videos = await asyncio.to_thread(search_videos, query, limit)
    return {"query": query, "videos": videos}


@app.get("/resolve")
async def get_resolve(url: str = Query(min_length=1)) -> dict:
    """Resolve a video URL to its id without downloading anything.

    An orchestrator needs the id up front: it names the directory every later
    call reads from, and only yt-dlp can derive it from an arbitrary URL.
    """
    try:
        video_id = await asyncio.to_thread(resolve_video_id, url)
    except Exception as error:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, f"Could not resolve {url}: {error}"
        ) from error
    return {"url": url, "video_id": video_id}


@app.get("/characters")
async def get_characters() -> dict:
    if not WORK_DIR.exists():
        return {"characters": []}
    characters = sorted(
        entry.name
        for entry in WORK_DIR.iterdir()
        if entry.is_dir() and not entry.name.startswith("_")
    )
    return {"characters": characters}


@app.post("/jobs", status_code=status.HTTP_202_ACCEPTED)
async def post_job(job_request: JobRequest, request: Request) -> Job:
    return await request.app.state.job_runner.start(job_request)


@app.get("/jobs")
async def get_jobs(request: Request, character: str | None = None) -> dict:
    return {"jobs": request.app.state.job_runner.list_jobs(character)}


@app.get("/jobs/{job_id}")
async def get_job(job_id: str, request: Request) -> Job:
    return request.app.state.job_runner.get(job_id)


@app.get("/jobs/{job_id}/logs")
async def get_job_logs(
    job_id: str, request: Request, offset: int = Query(default=0, ge=0)
) -> dict:
    runner = request.app.state.job_runner
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


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str, request: Request) -> Job:
    return await request.app.state.job_runner.cancel(job_id)


@app.get("/videos")
async def get_videos() -> dict:
    """Every ingested video, independent of any character.

    Lets the dashboard offer a video for a second character without asking
    the factory to ingest it again -- see /videos/{video_id}/speakers and the
    four clip routes below, none of which take a character either.
    """
    youtube_dir = WORK_DIR / "youtube"
    if not youtube_dir.exists():
        return {"videos": []}
    videos = [
        _video_summary(video_dir)
        for video_dir in sorted(youtube_dir.iterdir())
        if video_dir.is_dir()
    ]
    return {"videos": videos}


@app.get("/videos/{video_id}/speakers")
async def get_video_speakers(video_id: str) -> dict:
    rows = read_review_csv(_review_path(video_id))
    grouped: dict[str | None, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row.get("speaker_label") or None, []).append(row)

    speakers = [
        {
            "speaker_label": speaker_label,
            "clip_count": len(group),
            "kept_count": sum(1 for row in group if row["keep"] == "1"),
        }
        # None (undiarized/rejected) sorts last, same order the dashboard
        # already expects from the run-scoped speaker board
        for speaker_label, group in sorted(
            grouped.items(), key=lambda item: (item[0] is None, item[0] or "")
        )
    ]
    return {"video_id": video_id, "speakers": speakers}


@app.get("/videos/{video_id}/clips")
async def get_clips(video_id: str) -> dict:
    rows = read_review_csv(_review_path(video_id))
    map_path = _video_dir(video_id) / SPEAKER_MAP_FILENAME
    speaker_map = (
        json.loads(map_path.read_text(encoding="utf-8")) if map_path.exists() else {}
    )
    return {
        "video_id": video_id,
        "speaker_map": speaker_map,
        "clips": [_clip_from_row(row) for row in rows],
    }


@app.patch("/videos/{video_id}/clips")
async def patch_clips(video_id: str, decisions_request: ClipDecisionRequest) -> dict:
    """Apply keep/speaker-label decisions to review.csv.

    review.csv is shared now: once a video has more than one claimant, two
    characters' runs can both reach the same clip. Re-keeping or rejecting a
    clip is always safe -- it never changes which character the clip belongs
    to -- so `keep` is applied unconditionally, exactly as before this story.
    Reassigning `speaker_label` is different: that is how a clip moves from
    one character's dataset to another's, so silently overwriting an
    already-recorded label with a different one is exactly the
    cross-character corruption this story guards against. A conflicting
    reassignment is rejected with 409 instead of applied. Full multi-claimant
    routing is Story 2.2's job -- this is the narrow stopgap.
    """
    review_path = _review_path(video_id)
    rows = read_review_csv(review_path)
    by_clip_id = {row["clip_id"]: row for row in rows}

    unknown = [
        decision.clip_id
        for decision in decisions_request.decisions
        if decision.clip_id not in by_clip_id
    ]
    if unknown:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND, f"Unknown clip ids: {', '.join(unknown)}"
        )

    against_recorded = {
        decision.clip_id
        for decision in decisions_request.decisions
        if decision.speaker_label is not None
        and _reassigns_a_recorded_label(
            by_clip_id[decision.clip_id], decision.speaker_label
        )
    }
    within_request = _conflicting_labels_within_request(decisions_request.decisions)
    conflicts = sorted(against_recorded | within_request)
    if conflicts:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "Clip(s) already carry a different recorded speaker_label, and a "
            "shared video must not silently move them to another character's "
            f"dataset: {', '.join(conflicts)}. Update the video's speaker map "
            "instead, or resubmit without reassigning these clips.",
        )

    for decision in decisions_request.decisions:
        row = by_clip_id[decision.clip_id]
        if decision.keep is not None:
            row["keep"] = "1" if decision.keep else "0"
        if decision.speaker_label is not None:
            row["speaker_label"] = decision.speaker_label

    write_review_csv(review_path, [_fill_missing_fields(row) for row in rows])
    return {"updated": len(decisions_request.decisions)}


def _reassigns_a_recorded_label(row: dict, new_speaker_label: str) -> bool:
    # empty means never labelled (undiarized, or diarization left it blank),
    # so the first label a decision gives it is never a reassignment
    current = row.get("speaker_label") or ""
    return bool(current) and current != new_speaker_label


def _conflicting_labels_within_request(decisions: list[ClipDecision]) -> set[str]:
    """Clip ids that two decisions in the same request disagree about.

    _reassigns_a_recorded_label only ever sees one decision against the row's
    persisted state, so two decisions for the same clip_id in one payload --
    e.g. an unlabelled clip -- both pass that check independently and the
    last one applied would silently win. This catches that within the
    request itself, before anything is written.
    """
    labels_by_clip: dict[str, set[str]] = {}
    for decision in decisions:
        if decision.speaker_label is not None:
            labels_by_clip.setdefault(decision.clip_id, set()).add(
                decision.speaker_label
            )
    return {clip_id for clip_id, labels in labels_by_clip.items() if len(labels) > 1}


@app.put("/videos/{video_id}/speaker-map")
async def put_speaker_map(video_id: str, map_request: SpeakerMapRequest) -> dict:
    for target in map_request.speaker_map.values():
        if target is not None:
            _check_name(target, "character")
    try:
        written = write_speaker_map(_video_dir(video_id), map_request.speaker_map)
    except SpeakerMapConflict as error:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "Speaker(s) already carry a different recorded assignment, and a "
            "shared video must not silently move them to another character's "
            f"dataset: {', '.join(error.conflicting_labels)}. Confirm the "
            "existing assignment instead, or resubmit without these speakers.",
        ) from error
    return {"speaker_map": map_request.speaker_map, "path": str(written)}


@app.get("/videos/{video_id}/clips/{clip_id}/audio")
async def get_clip_audio(video_id: str, clip_id: str) -> FileResponse:
    _check_name(clip_id, "clip_id")
    clip_path = _video_dir(video_id) / "clips" / f"{clip_id}.wav"
    if not clip_path.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No clip {clip_id}")
    return FileResponse(clip_path, media_type="audio/wav")


def _video_summary(video_dir: Path) -> dict:
    review_path = video_dir / REVIEW_CSV_NAME
    clip_count = len(read_review_csv(review_path)) if review_path.exists() else 0
    return {
        "video_id": video_dir.name,
        "diarized": (video_dir / DIARIZATION_NAME).exists(),
        "reviewed": review_path.exists(),
        "clip_count": clip_count,
    }


@app.get("/characters/{character}/training")
async def get_training(character: str, request: Request) -> dict:
    _check_name(character, "character")
    training_dir = WORK_DIR / character / "training"
    checkpoints = [
        {
            "path": path.relative_to(APP_DIR).as_posix(),
            "name": path.name,
            "epoch": _parse_checkpoint(path.name)[0],
            "step": _parse_checkpoint(path.name)[1],
            "modified_at": datetime.fromtimestamp(
                path.stat().st_mtime, tz=UTC
            ).isoformat(),
        }
        for path in sorted(
            training_dir.glob("**/*.ckpt"), key=lambda path: path.stat().st_mtime
        )
    ]

    runner = request.app.state.job_runner
    train_jobs = [
        job
        for job in runner.list_jobs(character)
        if job.stage == "train" and job.state == "running"
    ]
    epoch, loss = (None, None)
    if train_jobs:
        epoch, loss = await asyncio.to_thread(
            _parse_training_log, runner.log_path(train_jobs[0].job_id)
        )

    return {
        "character": character,
        "preprocessed": (training_dir / "config.json").exists(),
        "running_job_id": train_jobs[0].job_id if train_jobs else None,
        "current_epoch": epoch,
        "current_loss": loss,
        "checkpoints": checkpoints,
    }


@app.get("/characters/{character}/samples")
async def get_samples(character: str) -> dict:
    _check_name(character, "character")
    samples_dir = WORK_DIR / character / "checkpoint_samples"
    if not samples_dir.exists():
        return {"character": character, "samples": {}}
    samples = {
        checkpoint_dir.name: sorted(path.name for path in checkpoint_dir.glob("*.wav"))
        for checkpoint_dir in sorted(samples_dir.iterdir())
        if checkpoint_dir.is_dir()
    }
    return {"character": character, "samples": samples}


@app.get("/characters/{character}/samples/{checkpoint_name}/{sample_name}")
async def get_sample_audio(
    character: str, checkpoint_name: str, sample_name: str
) -> FileResponse:
    _check_name(character, "character")
    _check_name(checkpoint_name, "checkpoint_name")
    sample_path = (
        WORK_DIR / character / "checkpoint_samples" / checkpoint_name / sample_name
    )
    # resolve() then compare: sample_name carries a .wav suffix, so SAFE_NAME
    # cannot vet it, and a crafted name must not escape the samples directory
    samples_root = (WORK_DIR / character / "checkpoint_samples").resolve()
    if not sample_path.resolve().is_relative_to(samples_root):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Invalid sample name")
    if not sample_path.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No sample {sample_name}")
    return FileResponse(sample_path, media_type="audio/wav")


def _read_from(path: Path, offset: int) -> bytes:
    with open(path, "rb") as source:
        source.seek(offset)
        return source.read()


def _clip_from_row(row: dict) -> dict:
    return {
        "clip_id": row["clip_id"],
        "keep": row["keep"] == "1",
        "quality_score": _as_float(row.get("quality_score")),
        "flagged": row.get("flagged") == "1",
        "speaker_label": row.get("speaker_label") or None,
        "speaker_coverage": _as_float(row.get("speaker_coverage")),
        "duration_sec": _as_float(row.get("duration_sec")),
        "start_sec": _as_float(row.get("start_sec")),
        "end_sec": _as_float(row.get("end_sec")),
        "text": row.get("text", ""),
    }


def _fill_missing_fields(row: dict) -> dict:
    # a review.csv written before diarization has no speaker columns; DictWriter
    # needs every field present
    return {field: row.get(field, "") for field in REVIEW_FIELDS}


def _as_float(value: str | None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_checkpoint(name: str) -> tuple[int | None, int | None]:
    match = CHECKPOINT_PATTERN.search(name)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def _parse_training_log(log_path: Path) -> tuple[int | None, float | None]:
    """Read epoch and loss out of Lightning's progress bar.

    Reads the last 64KB only: a long run writes a large log, and the newest
    progress line is always at the end.
    """
    if not log_path.exists():
        return None, None
    tail_start = max(0, log_path.stat().st_size - 65536)
    tail = _read_from(log_path, tail_start).decode("utf-8", errors="replace")
    # the progress bar redraws with \r, so split on both
    lines = re.split(r"[\r\n]", tail)
    epoch = None
    loss = None
    for line in reversed(lines):
        if epoch is None:
            epoch_match = EPOCH_PATTERN.search(line)
            if epoch_match:
                epoch = int(epoch_match.group(1))
        if loss is None:
            loss_match = LOSS_PATTERN.search(line)
            if loss_match:
                loss = float(loss_match.group(1))
        if epoch is not None and loss is not None:
            break
    return epoch, loss
