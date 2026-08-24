"""Request/response schemas and the stage/job-state vocabulary.

Shared by every route, so it stays dependency-free -- nothing here reaches
into infrastructure, core, or a specific route module.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, Field, StringConstraints

JobState = Literal["running", "succeeded", "failed", "cancelled"]

# youtube-ingest's five steps, each startable on its own so a retry only
# repeats the step that failed. Running them by hand in this order does the
# same thing as one youtube-ingest call.
YOUTUBE_INGEST_STEPS = (
    "youtube-download",
    "youtube-transcribe",
    "youtube-chunk",
    "youtube-diarize",
    "youtube-review",
)

# Stages that act on one video and so cannot run without a URL. The single
# source of truth for this list -- cli.py imports it from here instead of
# keeping its own copy.
YOUTUBE_STAGES_NEEDING_URL = ("youtube-ingest", *YOUTUBE_INGEST_STEPS)

Stage = Literal[
    "dataset",
    "import",
    "youtube-ingest",
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


class JobRequest(BaseModel):
    character: str | None = None
    stage: Stage
    youtube_url: str | None = None
    # every optional field stays None so it is only forwarded when set, which
    # keeps cli.py's own defaults as the single source of truth
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
    # None means "don't touch". "kept"/"excluded" is the reviewer's call;
    # "none" clears a clip back to unreviewed -- the third state a plain
    # bool cannot reach, for a clip nobody has decided on yet.
    keep: Literal["kept", "excluded", "none"] | None = None
    speaker_label: str | None = None
    # Who this one clip is for. Reassigning it is the reviewer's whole job, so
    # unlike speaker_label it is never a conflict. An empty string clears it.
    assigned_voice: str | None = None
    text: str | None = None
    # A trim from the review UI. Orthogonal to keep -- trimming a clip does
    # not mark it kept. Both must be given together; patch_clips 422s on
    # end_sec <= start_sec or either being negative.
    start_sec: float | None = None
    end_sec: float | None = None


class ClipDecisionRequest(BaseModel):
    decisions: list[ClipDecision] = Field(min_length=1)


class VideoRenameRequest(BaseModel):
    # Stripped before it is measured, so a title of spaces is rejected rather
    # than stored. A blank name would hide the video in every list, and
    # video_summary already falls back to the id when there is no title.
    title: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=300),
    ]


class SpeakerMapRequest(BaseModel):
    # speaker label -> character name. null discards that speaker's clips.
    speaker_map: dict[str, str | None]


class CommitRequest(BaseModel):
    # video_id -> {speaker_label: character}. A character of null discards
    # that speaker's clips, same meaning as SpeakerMapRequest.
    assignments: dict[str, dict[str, str | None]] = Field(min_length=1)
