# justfile
set shell := ["bash", "-c"]

# Sync all environments
sync-all: sync-janewav sync-jeanlucrecord
    @echo "All apps synced successfully!"

# Sync individual apps
sync-janewav:
    unset VIRTUAL_ENV && cd apps/janewav && uv sync

sync-jeanlucrecord:
    unset VIRTUAL_ENV && cd apps/jeanlucrecord && uv sync

# Run the pipeline apps
run-janewav:
    unset VIRTUAL_ENV && cd apps/janewav && uv run python main.py

run-jeanlucrecord:
    unset VIRTUAL_ENV && cd apps/jeanlucrecord && uv run python main.py

# Generate a fine-tuned Piper voice model for a character.
# stage defaults to the full pipeline; pass one to resume a single step,
# e.g. `just generate-voice doctor` or `just generate-voice doctor train`
# stages: dataset, resample, preprocess, smoketest, train, export
generate-voice character stage="all":
    unset VIRTUAL_ENV && cd apps/jeanlucrecord && uv run python main.py {{character}} --stage {{stage}}
