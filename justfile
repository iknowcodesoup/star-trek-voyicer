# justfile
set shell := ["bash", "-c"]

# Sync all environments
sync-all: sync-janewav sync-jeanlucrecord sync-diarizer
    @echo "All apps synced successfully!"

# Sync individual apps
sync-janewav:
    unset VIRTUAL_ENV && cd apps/janewav && uv sync

sync-jeanlucrecord:
    unset VIRTUAL_ENV && cd apps/jeanlucrecord && uv sync

# Speaker diarization runs in its own environment: pyannote.audio needs
# torch>=2.8 and chatterbox-tts pins torch==2.6, so they cannot share one.
# Only needed for `--stage youtube-ingest --diarize`.
sync-diarizer:
    unset VIRTUAL_ENV && cd apps/jeanlucrecord/diarizer && uv sync

# Run the pipeline apps
run-janewav:
    unset VIRTUAL_ENV && cd apps/janewav && uv run python main.py

run-jeanlucrecord:
    unset VIRTUAL_ENV && cd apps/jeanlucrecord && uv run jeanlucrecord

# Serve the HTTP control surface so an orchestrator can drive the pipeline.
# Binds to localhost only. cli.py stays the definition of every stage.
serve-jeanlucrecord port="8100":
    unset VIRTUAL_ENV && uv run --directory apps/jeanlucrecord python -m uvicorn jeanlucrecord.app:app --host 127.0.0.1 --port {{port}}

# Run the jeanlucrecord unit tests
test-jeanlucrecord:
    unset VIRTUAL_ENV && cd apps/jeanlucrecord && uv run pytest tests/

# Search YouTube for candidate source videos. No character needed, writes nothing.
search-youtube query limit="10":
    unset VIRTUAL_ENV && cd apps/jeanlucrecord && uv run jeanlucrecord --stage youtube-search --search-query "{{query}}" --search-limit {{limit}}

# Generate a fine-tuned Piper voice model for a character.
# stage defaults to the full pipeline; pass one to resume a single step,
# e.g. `just generate-voice doctor` or `just generate-voice doctor train`
# stages: dataset, resample, preprocess, smoketest, train, export
# checkpoint (export stage only): path to a specific .ckpt to export, e.g.
# `just generate-voice doctor export work/doctor/training/epoch=100.ckpt`
# omit it to export the most recently modified checkpoint.
generate-voice character stage="all" checkpoint="":
    unset VIRTUAL_ENV && cd apps/jeanlucrecord && uv run jeanlucrecord {{character}} --stage {{stage}} $(if [ -n "{{checkpoint}}" ]; then echo "--checkpoint {{checkpoint}}"; fi)
