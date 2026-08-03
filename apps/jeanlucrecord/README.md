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

## Usage

```
uv run python main.py <character>
```

Runs the full pipeline: dataset generation, resampling, Piper preprocessing, fine-tuning,
and ONNX export. Takes hours (dataset generation) plus a multi-day training run.

Run one stage at a time with `--stage {dataset,resample,preprocess,smoketest,train,export,sample}`
— useful for resuming after the multi-day training stage without regenerating the dataset,
and for the `smoketest` stage, which should be run once after building the Docker image and
before a real training run, to confirm the GPU is usable inside the container:

```
uv run python main.py <character> --stage smoketest
```

`--corpus-size N` overrides the default corpus size (1300) — useful for a quick smoke run:

```
uv run python main.py doctor --corpus-size 10 --stage dataset
```

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
as the next one saves), spaced 20 epochs apart via `main.py`'s `CHECKPOINT_EPOCHS`. Loss
alone isn't a reliable way to pick a winner among them — compare by ear instead:

```
uv run python main.py <character> --stage sample
```

Exports every retained checkpoint under `work/<character>/training/**/*.ckpt` to its own
ONNX model and synthesizes the same fixed, held-out sentences (never seen during training)
from each, into `work/<character>/checkpoint_samples/<checkpoint-name>/*.wav`. Safe to
re-run mid-training — checkpoints already sampled are skipped. `--num-validation-sentences N`
overrides how many sentences to synthesize (default 8).

Listen to the results, then export the checkpoint that actually sounds best rather than
assuming the latest epoch is (`--stage export` alone always picks the most recent by mtime):

```
uv run python main.py <character> --stage export --checkpoint work/<character>/training/lightning_logs/version_N/checkpoints/epoch=X-step=Y.ckpt
```

## Output

`output/<character>.onnx` + `.onnx.json`. The `export` stage prints the exact commands to
copy these into `apps/janewav/src/models/` and the `.env` `MODELS` entry to add.
