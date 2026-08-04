#!/usr/bin/env bash
set -eo pipefail
character="$1"
shift
training_dir="work/$character/training"
samples_dir="work/$character/checkpoint_samples"
sentences_file="work/$character/validation_sentences.txt"
mkdir -p "$samples_dir"

if [[ ! -f "$sentences_file" ]]; then
    echo "Missing $sentences_file -- run via 'python main.py <character> --stage sample', not this script directly." >&2
    exit 1
fi

if [[ "$#" -eq 0 ]]; then
    echo "No checkpoints given" >&2
    exit 1
fi

for checkpoint in "$@"; do
    name="$(basename "$checkpoint" .ckpt)"
    out_dir="$samples_dir/$name"

    # skip checkpoints already sampled in a prior run of this stage -- lets stage
    # sample be re-run cheaply as new checkpoints appear during a long training run,
    # without re-exporting/re-synthesizing ones already compared
    if [[ -f "$out_dir/00.wav" ]]; then
        echo "Skipping $name (already sampled)"
        continue
    fi
    mkdir -p "$out_dir"

    onnx="$out_dir/model.onnx"
    python3 -m piper_train.export_onnx "$checkpoint" "$onnx.unoptimized"
    onnxsim "$onnx.unoptimized" "$onnx"
    rm -f "$onnx.unoptimized"
    cp "$training_dir/config.json" "$onnx.json"

    i=0
    while IFS= read -r sentence; do
        [[ -z "$sentence" ]] && continue
        wav_path="$out_dir/$(printf '%02d' "$i").wav"
        printf '%s' "$sentence" | python3 -m piper -m "$onnx" -c "$onnx.json" --cuda -f "$wav_path"
        i=$((i + 1))
    done < "$sentences_file"

    echo "Sampled $name -> $out_dir ($i sentences)"
done

echo "Done. Listen under $samples_dir/<checkpoint-name>/*.wav, then export the winner with:"
echo "  uv run python main.py $character --stage export --checkpoint <the .ckpt path>"
