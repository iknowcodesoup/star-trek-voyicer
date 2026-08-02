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

Run one stage at a time with `--stage {dataset,resample,preprocess,smoketest,train,export}`
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

## Output

`output/<character>.onnx` + `.onnx.json`. The `export` stage prints the exact commands to
copy these into `apps/janewav/src/models/` and the `.env` `MODELS` entry to add.
