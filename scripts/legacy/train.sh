# cd /mnt/c/Projects/free-friend
# wsl / linux required -> uv venv --python 3.10 --seed .venv   

#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

pip install "pip<24.1"

if ! pip install piper-phonemize; then
  echo "WARNING: could not install piper-phonemize from PyPI."
  echo "If your platform does not have a wheel, try building from source:
  pip install git+https://github.com/rhasspy/piper-phonemize.git"
  echo "Or use a matching version with piper-tts docs."
fi

# Config
AUDIO_DIR="./piper-tts/recordings"
VOICE_NAME="cena"
OUTPUT_DIR="./model_host/models"
CHECKPOINT=""   # set to path if using pretrained
QUALITY="high"  # or low
METADATA_FILE="$AUDIO_DIR/metadata.csv"

if [ ! -f "$METADATA_FILE" ]; then
  cat <<EOF
ERROR: metadata.csv not found in $AUDIO_DIR

Create a file named metadata.csv with one line per recording:
  recording_001|Hello, how are you today?
  recording_002|The quick brown fox jumps over the lazy dog.

Filename is without extension; path is relative to the audio folder.
EOF
  exit 1
fi

echo "Starting Piper training..."
echo "Audio: $AUDIO_DIR"
echo "Voice: $VOICE_NAME"
echo "Output: $OUTPUT_DIR"

CMD=(
  python "$SCRIPT_DIR/create-voice-model.py" train
  --audio "$AUDIO_DIR"
  --voice "$VOICE_NAME"
  --output "$OUTPUT_DIR"
  --quality "$QUALITY"
)

# Optional checkpoint
if [ -n "$CHECKPOINT" ]; then
  CMD+=(--checkpoint "$CHECKPOINT")
fi

"${CMD[@]}"

echo "Training complete."
