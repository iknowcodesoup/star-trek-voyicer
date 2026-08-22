# jeanlucrecord

Generates a small, fast, fine-tuned [Piper](https://github.com/rhasspy/piper) voice model for a
character from just a couple of short reference clips, for use by `apps/janewav`.

Technique: [Training a new AI voice for Piper TTS with only 4 words](https://calbryant.uk/blog/training-a-new-ai-voice-for-piper-tts-with-only-4-words/).
Chatterbox TTS clones the reference voice across a ~1300-phrase public-domain corpus,
each sample is verified by comparing a Whisper transcription against the original text
as phonemes (so spelling/punctuation/number differences don't cause false failures), then
Piper is fine-tuned on the resulting dataset from its LJSpeech checkpoint.

## Setup

1. Put one or more short `.wav` reference clips of the character's voice in `samples/<character>/`.
2. Download the Piper LJSpeech "high" quality checkpoint to `checkpoints/ljspeech-2000.ckpt`
   (see [Piper's training docs](https://github.com/rhasspy/piper/blob/master/TRAINING.md)).
3. `uv sync`
4. Requires Docker Desktop (with GPU support) for the fine-tuning stage only — dataset
   generation and verification run natively on CPU.
5. `ffmpeg` and `yt-dlp` must be on PATH — only required for the `youtube-ingest` stage.
6. `just sync-diarizer` — only required for `--stage youtube-ingest --diarize`. Sets up a
   separate environment for `pyannote.audio`, which needs a newer torch than
   `chatterbox-tts` allows. See [Splitting a video by speaker](#splitting-a-video-by-speaker).

## Usage

```
uv run jeanlucrecord <character>
```

Runs the full pipeline: dataset generation, resampling, Piper preprocessing, fine-tuning,
and ONNX export. Takes hours (dataset generation) plus a multi-day training run.

Run one stage at a time with
`--stage {dataset,resample,preprocess,smoketest,train,export,sample,import,youtube-search,youtube-ingest,youtube-commit}`
— useful for resuming after the multi-day training stage without regenerating the dataset,
and for the `smoketest` stage, which should be run once after building the Docker image and
before a real training run, to confirm the GPU is usable inside the container:

```
uv run jeanlucrecord <character> --stage smoketest
```

`--corpus-size N` overrides the default corpus size (1300) — useful for a quick smoke run:

```
uv run jeanlucrecord doctor --corpus-size 10 --stage dataset
```

## Alternative dataset sources

`dataset`/`import`/`youtube-ingest`+`youtube-commit` are three different ways to populate
`work/<character>/dataset/wavs/` + `metadata.csv`. Whichever one you use, `resample`,
`preprocess`, `train`, etc. proceed exactly the same afterward.

### Bring your own dataset

If you already have clipped, transcribed audio (an `id|text` `metadata.csv` plus matching
`<id>.wav` files, flat or under `wavs/`/`wav/` — e.g. `samples/cena/`), import it directly
instead of generating a dataset with Chatterbox:

```
uv run jeanlucrecord cena --stage import --import-dir samples/cena
uv run jeanlucrecord cena --stage resample
uv run jeanlucrecord cena --stage preprocess
uv run jeanlucrecord cena --stage train
```

Metadata rows with no matching wav are logged and dropped, not treated as an error.
Re-running `--stage import` after adding more rows to the source folder only imports
what's new.

### YouTube ingestion

Pull real audio from a YouTube video instead: downloads the audio, converts it to 22050 Hz
mono, transcribes it, and cuts it into candidate clips at the transcript's sentence
boundaries:

```
uv run jeanlucrecord picard --stage youtube-ingest --youtube-url https://www.youtube.com/watch?v=XXXXXXXXXXX
```

Since a source video may have more than one speaker or background noise, clips are **not**
committed to the dataset automatically. Instead this writes
`work/picard/youtube/<video_id>/review.csv` (clip id, a rough noise/quality score, and a
`keep` column — sorted worst-quality-first, with clips scoring below
`--quality-flag-threshold` defaulting to `keep=0`) alongside the actual clips under
`work/picard/youtube/<video_id>/clips/`. Listen to any clip you're unsure about, edit
`keep` to `0`/`1` in the CSV (any spreadsheet app works, it's a plain comma-delimited
file), then commit the ones you kept:

```
uv run jeanlucrecord picard --stage youtube-commit
```

Ingesting the same URL again is a no-op (existing `review.csv` is left alone). Ingest more
videos and re-run `youtube-commit` at any time — only newly-kept clips are merged in.
Note commit is additive-only: flipping `keep` back to `0` after a clip has already been
committed doesn't remove it from `dataset/`; delete its row + wav there by hand instead.

#### Finding source videos

`youtube-search` queries YouTube and prints candidates. It needs no character and writes
nothing:

```
uv run jeanlucrecord --stage youtube-search --search-query "star trek voyager janeway"
```

#### Splitting a video by speaker

Most real footage has more than one person talking. Add `--diarize` to split the audio by
speaker before the clips reach `review.csv`:

```
uv run jeanlucrecord janeway --stage youtube-ingest --diarize \
  --youtube-url https://www.youtube.com/watch?v=XXXXXXXXXXX
```

This adds two columns to `review.csv`:

- `speaker_label` — `SPEAKER_00`, `SPEAKER_01`, and so on.
- `speaker_coverage` — how much of the clip that speaker holds, from 0 to 1.

A clip that no single speaker holds for `--min-speaker-coverage` (default 0.9) gets an
empty `speaker_label` and defaults to `keep=0`. Two cases produce that, and both are
unusable for training: two people talk at once, or the clip is mostly music and silence.
Pass `--num-speakers N` when you already know how many people speak.

**Setup.** Diarization runs in its own environment under `diarizer/`, because
`pyannote.audio` needs `torch>=2.8` and `chatterbox-tts` pins `torch==2.6`. The two never
run together — one synthesizes a dataset from a text corpus, the other splits real audio —
so isolating them costs nothing. `core/diarization.py` runs `diarizer/diarize_worker.py` as a
subprocess.

```
just sync-diarizer
```

**Token.** Both pyannote models are gated. Accept the terms for
[pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
**and** [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0),
create a [read token](https://huggingface.co/settings/tokens), then set `HF_TOKEN` or pass
`--hf-token`. Without both approvals the download fails with a 401.

The result is cached to `work/<character>/youtube/<video_id>/diarization.json`, so a re-run
does not repeat it.

Diarization uses the GPU when the diarizer environment's torch build has CUDA. The default
build is CPU-only, which takes about as long as the audio. Start with short clips.

#### Sending one video to several characters

One episode can seed several voices. Write a `speaker_map.json` next to `review.csv`:

```json
{ "SPEAKER_00": "janeway", "SPEAKER_01": "chakotay", "SPEAKER_02": null }
```

`youtube-commit` then routes each speaker's kept clips into that character's own
`work/<character>/dataset/`. A speaker mapped to `null` is discarded. A speaker missing
from the map stays uncommitted, so you can correct the map and re-run. Without a
`speaker_map.json`, every kept clip goes to the character named on the command line, which
is the behaviour from before diarization existed.

## HTTP control surface

`app.py` lets an outside orchestrator drive this pipeline. It runs no stage itself: every
job spawns `jeanlucrecord <character> --stage <stage>` and tails its output, so the
command line stays the single definition of what each stage does.

```
just serve-jeanlucrecord        # http://127.0.0.1:8100
```

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | reachability |
| `GET` | `/search?query=&limit=` | search YouTube |
| `GET` | `/resolve?url=` | resolve a video URL to its id |
| `GET` | `/characters` | characters with a `work/` directory |
| `POST` | `/jobs` | start a stage, returns a `job_id` |
| `GET` | `/jobs` · `/jobs/{id}` | job state |
| `GET` | `/jobs/{id}/logs?offset=` | tail a job's output from a byte offset |
| `DELETE` | `/jobs/{id}` | cancel, and stop the container |
| `GET` | `/videos` | every ingested video, independent of any character |
| `GET` | `/videos/{video_id}/speakers` | speaker labels and clip counts for one video |
| `GET` | `/videos/{video_id}/clips` | `review.csv` as JSON |
| `PATCH` | `/videos/{video_id}/clips` | set `keep` and `speaker_label` |
| `GET` | `/videos/{video_id}/clips/{id}/audio` | play one clip |
| `PUT` | `/videos/{video_id}/speaker-map` | write `speaker_map.json` for one video |
| `POST` | `/videos/commit` | write several videos' speaker maps, then commit every kept clip |
| `GET` | `/characters/{c}/training` | epoch, loss, and checkpoints |
| `GET` | `/characters/{c}/samples` | checkpoint sample wavs |

None of the `/videos/...` routes take a character: a video is ingested once and shared
across every character that later claims it (see `speaker_map.json` below).

`GET /videos` names each video from `meta.json`, written beside the clips at ingest. The
video owns its title, so every character that claims it reads the same name. A video
ingested before `meta.json` existed reports its id as the title and null for the rest.

`GET /videos` and `GET /characters` answer 500 when `WORK_DIR` does not exist. That is a
broken deployment, not an empty install, and an empty list would hide it. A `WORK_DIR`
that is there with no `youtube/` under it is a fresh install, and answers 200 with an
empty list.

`POST /jobs` returns at once. It never waits for a stage to finish, because training takes
days. Poll `/jobs/{id}` and read `/jobs/{id}/logs` for progress.

The server binds to localhost. It sets no CORS origins by default, because the intended
caller is another server, not a browser. Set `VOICE_FACTORY_CORS_ALLOW_ORIGINS` to a
comma-separated list to change that.

## Tests

```
just test-jeanlucrecord
```

Covers the speaker rejection rule in `core/diarization.py` and the commit routing in
`core/review_workflow.py`. Both are pure functions, so the tests need no audio and no GPU.

## Monitoring

While the `train` stage is running, from another terminal:

- GPU utilization: `nvidia-smi -l 1` (or `watch -n 1 nvidia-smi` on Linux/macOS)
- Loss curves: `uv run tensorboard --logdir work/<character>/training/lightning_logs`, then open
  <http://localhost:6006>. Requires `tensorboard` to be installed in the Docker training image
  (see Dockerfile) — without it, Lightning silently falls back to a CSV-only logger
  (`metrics.csv`, no `.tfevents`).

## Checkpoint quality

The Dockerfile patches piper_train to keep the 10 most recent checkpoints instead of just
the latest (Lightning's un-patched default silently deletes each older checkpoint as soon
as the next one saves), spaced 20 epochs apart via `cli.py`'s `CHECKPOINT_EPOCHS`. Loss
alone isn't a reliable way to pick a winner among them — compare by ear instead:

```
uv run jeanlucrecord <character> --stage sample
```

Exports every retained checkpoint under `work/<character>/training/**/*.ckpt` to its own
ONNX model and synthesizes the same fixed, held-out sentences (never seen during training)
from each, into `work/<character>/checkpoint_samples/<checkpoint-name>/*.wav`. Safe to
re-run mid-training — checkpoints already sampled are skipped. `--num-validation-sentences N`
overrides how many sentences to synthesize (default 8).

Listen to the results, then export the checkpoint that actually sounds best rather than
assuming the latest epoch is (`--stage export` alone always picks the most recent by mtime):

```
uv run jeanlucrecord <character> --stage export --checkpoint work/<character>/training/lightning_logs/version_N/checkpoints/epoch=X-step=Y.ckpt
```

## Output

`output/<character>.onnx` + `.onnx.json`. The `export` stage prints the exact commands to
copy these into `apps/janewav/src/models/` and the `.env` `MODELS` entry to add.
