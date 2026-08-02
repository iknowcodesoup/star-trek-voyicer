#!/usr/bin/env bash
set -eo pipefail
character="$1"
training_dir="work/$character/training"
output_dir="output"
mkdir -p "$output_dir"

checkpoint="$(find "$training_dir" -name '*.ckpt' -type f -printf '%T+ %p\n' | sort -r | head -n1 | cut -d' ' -f2-)"
echo "Using checkpoint: $checkpoint"

onnx="$output_dir/$character.onnx"
python3 -m piper_train.export_onnx "$checkpoint" "$onnx.unoptimized"
onnxsim "$onnx.unoptimized" "$onnx"
rm -f "$onnx.unoptimized"
cp "$training_dir/config.json" "$onnx.json"
echo "Exported: $onnx"
