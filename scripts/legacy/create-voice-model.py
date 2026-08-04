#!/usr/bin/env python3
"""
create-voice-model.py — Build a Sherpa-ONNX compatible Piper TTS voice model.

MODES
-----
convert  Convert existing Piper model files (.onnx + .onnx.json) to the ZIP
         format expected by SherpaTextToSpeechGateway.

train    Prepare a dataset from a folder of audio files and train a custom
         Piper voice inside a Podman container, then convert it.

USAGE
-----
  # Convert pre-trained Piper model files to SherpaGateway ZIP:
  python create-voice-model.py convert --input ./my-piper-model/ --output ./output/

  # Train a new voice from audio + transcripts, then convert:
  python create-voice-model.py train --audio ./recordings/ --voice my-voice --output ./output/ --checkpoint ./en_US-lessac-medium.ckpt

REQUIRED FILES (convert mode)
------------------------------
  <input>/
    *.onnx          Piper VITS model
    *.onnx.json     Piper model config

REQUIRED FILES (train mode)
----------------------------
  <audio>/
    metadata.csv    Transcripts: file|text  (no header, no extension in file id)
                    e.g.  recording_001|Hello, how are you today?
    wav/            Subdirectory containing all WAV recordings (or directly
                    in <audio>/ — both layouts are accepted)
      recording_001.wav
      recording_002.wav
      ...
    Audio will be resampled to 22050 Hz mono automatically.

  A pretrained Piper checkpoint (.ckpt) from HuggingFace is strongly recommended
  for fine-tuning. Obtain one from:
    https://huggingface.co/datasets/rhasspy/piper-checkpoints/tree/main

OUTPUT ZIP STRUCTURE (matches SherpaTextToSpeechGateway expectations)
----------------------------------------------------------------------
  <voice-name>.onnx
  <voice-name>.onnx.data   (optional, external weights for large models)
  tokens.txt
  espeak-ng-data/
  <voice-name>.onnx.json   (optional, included if present)

DEPENDENCIES
------------
  pip install onnx==1.17.0 onnxruntime==1.17.1 setuptools

  Train mode requires Podman (https://podman.io) and Git.
  The training container is built from https://github.com/veralvx/piper-train
  the first time you run train — subsequent runs reuse the cached image.

  Audio resampling requires:
  pip install soundfile librosa
"""

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path
from typing import Any, Dict, Optional


# ─── Constants ──────────────────────────────────────────────────────────────

ESPEAK_NG_DATA_URL = (
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/espeak-ng-data.tar.bz2"
)

PIPER_TRAIN_REPO_URL = "https://github.com/veralvx/piper-train"
PIPER_TRAIN_IMAGE = "piper-train"


# ─── ONNX metadata helpers ──────────────────────────────────────────────────

def add_meta_data(filename: str, meta_data: Dict[str, Any]) -> None:
    """Inject Sherpa-ONNX metadata into a Piper ONNX model (in-place).

    Loads with load_external_data=False so the companion .onnx.data file
    is not touched here — embed_external_onnx_weights() handles that later.
    """
    import onnx
    model = onnx.load(filename, load_external_data=False)
    for key, value in meta_data.items():
        entry = model.metadata_props.add()
        entry.key = key
        entry.value = str(value)
    onnx.save(model, filename)


def load_piper_config(onnx_path: str) -> Dict[str, Any]:
    """Load the .onnx.json config that sits alongside a Piper model."""
    config_path = onnx_path + ".json"
    if not os.path.exists(config_path):
        _die(f"Config file not found: {config_path}\n"
             "Expected a file named <model>.onnx.json in the same directory.")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def generate_tokens(config: Dict[str, Any], output_dir: str) -> str:
    """Write tokens.txt from the phoneme_id_map in the Piper config."""
    tokens_path = os.path.join(output_dir, "tokens.txt")
    phoneme_id_map = config.get("phoneme_id_map", {})
    if not phoneme_id_map:
        _die("phoneme_id_map not found in config — is this a valid Piper .onnx.json?")
    with open(tokens_path, "w", encoding="utf-8") as f:
        for symbol, ids in phoneme_id_map.items():
            f.write(f"{symbol} {ids[0]}\n")
    print(f"  Generated: {tokens_path}")
    return tokens_path


def build_sherpa_metadata(config: Dict[str, Any]) -> Dict[str, Any]:
    """Build the metadata dict required by sherpa-onnx for a Piper VITS model.

    Includes every key that sherpa-onnx's VitsModel loader reads so the native
    code never falls back to uninitialized defaults.
    See: https://github.com/k2-fsa/sherpa-onnx/blob/master/sherpa-onnx/csrc/vits-model-metadata.cc
    """
    espeak_voice = config["espeak"]["voice"]
    if "language" in config:
        language = config["language"]["name_english"]
    else:
        language = espeak_voice.split("-")[0] if "-" in espeak_voice else espeak_voice

    inference = config.get("inference", {})
    num_speakers = config.get("num_speakers", 1)
    num_symbols = config.get("num_symbols", 0)
    add_blank = config.get("add_blank", True)
    phoneme_type = config.get("phoneme_type", "espeak")

    return {
        # Core type / comment — sherpa checks these to select the right loader path
        "model_type": "vits",
        "comment": "piper",
        # Language / voice
        "language": language,
        "voice": espeak_voice,
        # Phonemizer flags
        "has_espeak": 1,
        "phoneme_type": phoneme_type,
        "add_blank": 1 if add_blank else 0,
        # Vocabulary
        "num_symbols": num_symbols,
        # Speaker count
        "n_speakers": num_speakers,
        # Audio
        "sample_rate": config["audio"]["sample_rate"],
        # Inference hyperparameters (consumed by C++ noise/length scaling)
        "noise_scale": inference.get("noise_scale", 0.667),
        "length_scale": inference.get("length_scale", 1.0),
        "noise_w": inference.get("noise_w", 0.8),
    }


# ─── espeak-ng-data download ─────────────────────────────────────────────────

def download_espeak_ng_data(dest_dir: str) -> str:
    """Download and extract espeak-ng-data into dest_dir/espeak-ng-data/."""
    espeak_target = os.path.join(dest_dir, "espeak-ng-data")
    if os.path.isdir(espeak_target):
        print("  espeak-ng-data already present, skipping download.")
        return espeak_target

    print(f"  Downloading espeak-ng-data (~7 MB) …")
    archive_path = os.path.join(dest_dir, "espeak-ng-data.tar.bz2")
    _download(ESPEAK_NG_DATA_URL, archive_path)

    print("  Extracting espeak-ng-data ...")
    import tarfile
    with tarfile.open(archive_path, "r:bz2") as tf:
        tf.extractall(path=dest_dir)

    os.remove(archive_path)
    print(f"  Extracted: {espeak_target}")
    return espeak_target


# ─── Embed external ONNX weights ─────────────────────────────────────────────

def _enforce_opset_15(onnx_path: str) -> None:
    """Validate that the ONNX model is opset 15 for Sherpa-ONNX compatibility.

    Fails loudly if the model was exported at the wrong opset — the fix is to
    re-export with dynamo=False, not to convert after the fact (onnx
    version_converter cannot downgrade ops like ReduceL2 that changed schema).
    """
    import onnx
    model = onnx.load(onnx_path, load_external_data=False)
    current_opset = next(
        (op.version for op in model.opset_import if not op.domain), None
    )
    if current_opset is not None and current_opset > 15:
        # Inspect which ops require >15 so we know what to fix
        from collections import Counter
        op_versions: dict[str, int] = {}
        for node in model.graph.node:
            key = f"{node.domain or 'ai.onnx'}::{node.op_type}"
            if key not in op_versions:
                op_versions[key] = 0
        # Check ONNX op schema to find which ops need >15
        from onnx import defs as onnx_defs
        high_opset_ops = []
        for key in sorted(op_versions):
            domain, op_type = key.split("::", 1)
            if domain == "ai.onnx":
                domain = ""
            try:
                schema = onnx_defs.get_schema(op_type, current_opset, domain)
                since = schema.since_version
                if since > 15:
                    high_opset_ops.append((key, since))
            except Exception:
                pass
        op_counter = Counter(node.op_type for node in model.graph.node)
        print(f"\n{'='*60}")
        print(f"OPSET INSPECTION: {os.path.basename(onnx_path)} is opset {current_opset}")
        print(f"{'='*60}")
        print(f"Total nodes: {len(model.graph.node)}")
        print(f"\nAll op types (count):")
        for op, count in op_counter.most_common():
            print(f"  {op}: {count}")
        if high_opset_ops:
            print(f"\nOps requiring opset > 15:")
            for key, since in high_opset_ops:
                print(f"  {key} (since opset {since}, used {op_counter[key.split('::')[1]]}x)")
        else:
            print(f"\nNo individual ops require opset > 15 — opset may be inflated by exporter metadata.")
        print(f"{'='*60}\n")
        raise RuntimeError(
            f"{os.path.basename(onnx_path)} is opset {current_opset}, expected 15.\n"
            "The export must use opset_version=15 and dynamo=False in torch.onnx.export.\n"
            "Post-hoc opset conversion is not supported for Piper VITS models."
        )
    print(f"  Opset: {current_opset} (OK)")


def embed_external_onnx_weights(assembly_dir: str) -> None:
    """If any .onnx file uses external data (.onnx.data), embed the weights
    directly into the .onnx file so Sherpa-ONNX can load it on Android.

    Steps:
      1. Load the model WITH external data (pulls bytes into memory).
      2. Clear data_location=EXTERNAL and external_data entries on every tensor
         so ONNX Runtime won't look for a .data sidecar file at load time.
      3. Save as a single self-contained .onnx file.
    """
    import glob
    import onnx
    onnx_files = glob.glob(os.path.join(assembly_dir, "*.onnx"))
    for onnx_path in onnx_files:
        data_path = onnx_path + ".data"
        if not os.path.exists(data_path):
            continue
        print(f"  Embedding external weights into {os.path.basename(onnx_path)} ...")
        model = onnx.load(onnx_path, load_external_data=True)

        current_opset = next(
            (op.version for op in model.opset_import if not op.domain), None
        )
        if current_opset is not None and current_opset > 15:
            raise RuntimeError(
                f"{os.path.basename(onnx_path)} is opset {current_opset}, expected 15. "
                "Re-export with opset_version=15 and dynamo=False."
            )

        # Clear the EXTERNAL flag on every tensor — without this, ONNX Runtime
        # still tries to open <model>.onnx.data even though data is inline.
        for tensor in model.graph.initializer:
            if tensor.data_location == onnx.TensorProto.EXTERNAL:
                tensor.data_location = onnx.TensorProto.DEFAULT
                del tensor.external_data[:]

        os.remove(onnx_path)
        os.remove(data_path)
        onnx.save(model, onnx_path)
        size_mb = os.path.getsize(onnx_path) / (1024 * 1024)
        print(f"  Embedded: {os.path.basename(onnx_path)} ({size_mb:.1f} MB)")


# ─── ZIP packaging ───────────────────────────────────────────────────────────

def package_to_zip(assembly_dir: str, output_zip: str) -> None:
    """Create a flat ZIP from all files/dirs in assembly_dir."""
    if os.path.exists(output_zip):
        os.remove(output_zip)

    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(assembly_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, assembly_dir)
                zf.write(file_path, arcname)

    size_mb = os.path.getsize(output_zip) / (1024 * 1024)
    print(f"\n  Packaged: {output_zip} ({size_mb:.1f} MB)")

    with zipfile.ZipFile(output_zip, "r") as zf:
        for name in sorted(zf.namelist()):
            info = zf.getinfo(name)
            size_kb = info.file_size / 1024
            print(f"    {name}  ({size_kb:.0f} KB)")


# ─── CONVERT mode ────────────────────────────────────────────────────────────

def cmd_convert(args: argparse.Namespace) -> None:
    """Convert a folder of Piper model files to a Sherpa-ONNX ZIP."""
    input_dir = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find .onnx file
    onnx_files = list(input_dir.glob("*.onnx"))
    onnx_files = [f for f in onnx_files if not f.name.endswith(".onnx.json")]
    if not onnx_files:
        _die(f"No .onnx file found in {input_dir}")
    if len(onnx_files) > 1:
        _die(f"Multiple .onnx files found — specify which one:\n  " +
             "\n  ".join(str(f) for f in onnx_files))
    onnx_file = onnx_files[0]

    voice_name = args.voice_name or onnx_file.stem  # e.g. "en_US-amy-low"
    output_zip = str(output_dir / f"{voice_name}.zip")

    print(f"\n[convert] {onnx_file.name} -> {output_zip}")

    with tempfile.TemporaryDirectory(prefix="sherpa-voice-") as tmp:
        assembly = os.path.join(tmp, "assembly")
        os.makedirs(assembly)

        # Copy and patch the ONNX model
        onnx_dest = os.path.join(assembly, onnx_file.name)
        shutil.copy2(str(onnx_file), onnx_dest)
        print(f"  Copied:  {onnx_file.name}")

        # Copy .onnx.data if present (external weights for large models)
        data_src = str(onnx_file) + ".data"
        if os.path.exists(data_src):
            shutil.copy2(data_src, os.path.join(assembly, onnx_file.name + ".data"))
            print(f"  Copied:  {onnx_file.name}.data")

        # Copy .onnx.json if present
        json_src = str(onnx_file) + ".json"
        if os.path.exists(json_src):
            shutil.copy2(json_src, os.path.join(assembly, onnx_file.name + ".json"))
            print(f"  Copied:  {onnx_file.name}.json")

        # Validate and enforce opset 15 for Sherpa-ONNX compatibility
        _enforce_opset_15(onnx_dest)

        # Load config, add metadata, generate tokens
        print("  Loading Piper config …")
        config = load_piper_config(str(onnx_file))

        print("  Adding Sherpa-ONNX metadata to ONNX model …")
        meta = build_sherpa_metadata(config)
        print(f"    {json.dumps(meta, indent=6)}")
        add_meta_data(onnx_dest, meta)

        print("  Generating tokens.txt …")
        generate_tokens(config, assembly)

        # Download espeak-ng-data
        print("  Fetching espeak-ng-data …")
        download_espeak_ng_data(assembly)

        # Embed external weights (Sherpa-ONNX can't resolve .onnx.data on Android)
        embed_external_onnx_weights(assembly)

        # Preflight: verify OnnxRuntime can open the model on this machine
        # before shipping it to the device.  Catches corrupt graphs immediately.
        print("  Validating ONNX model with onnxruntime …")
        try:
            import onnxruntime as ort
            sess_options = ort.SessionOptions()
            sess_options.log_severity_level = 3  # errors only
            session = ort.InferenceSession(
                onnx_dest, sess_options=sess_options, providers=["CPUExecutionProvider"]
            )
            inputs = {i.name: i for i in session.get_inputs()}
            outputs = [o.name for o in session.get_outputs()]
            print(f"  ORT OK — inputs: {list(inputs.keys())}, outputs: {outputs}")
            del session
        except Exception as ort_error:
            _die(f"OnnxRuntime preflight FAILED: {ort_error}\n"
                 "The model cannot be loaded by ORT — it will crash on device too.\n"
                 "Fix the ONNX graph before packaging.")

        # Package
        print("  Packaging ZIP …")
        package_to_zip(assembly, output_zip)

    print("\nDone! Test with:")
    print(f"  pip install sherpa-onnx")
    print(f"  sherpa-onnx-offline-tts \\")
    print(f"    --vits-model=./{onnx_file.name} \\")
    print(f"    --vits-tokens=./tokens.txt \\")
    print(f"    --vits-data-dir=./espeak-ng-data \\")
    print(f'    "Hello, this is a test."')


# ─── REEXPORT mode ──────────────────────────────────────────────────────────

def cmd_reexport(args: argparse.Namespace) -> None:
    """Re-run only the ONNX export step inside the Podman container.

    Use this after changing export_onnx.py (e.g. opset version) to regenerate
    the .onnx from a specific checkpoint without re-training.
    """
    _require_podman()

    model_dir = Path(args.model_dir).resolve()
    checkpoints_dir = model_dir / "checkpoints"
    voice_name = args.voice_name or model_dir.name
    onnx_filename = f"{voice_name}.onnx"
    onnx_output = checkpoints_dir / onnx_filename

    ckpt_host = Path(args.checkpoint).resolve()
    if not ckpt_host.exists():
        _die(f"Checkpoint not found: {ckpt_host}")

    # The checkpoint must be inside checkpoints_dir so the container can reach
    # it via the /piper/checkpoints mount.
    try:
        ckpt_rel = ckpt_host.relative_to(checkpoints_dir)
    except ValueError:
        _die(
            f"Checkpoint {ckpt_host} is not inside {checkpoints_dir}.\n"
            "Copy it there first, or adjust --model-dir."
        )
    ckpt_container_path = f"./checkpoints/{ckpt_rel.as_posix()}"

    print(f"\n[reexport] {ckpt_host.name} → {onnx_filename} (opset={OPSET_VERSION_DISPLAY})")

    lightning_logs_dir = model_dir / "lightning_logs"
    lightning_logs_dir.mkdir(exist_ok=True)
    wavs_placeholder = model_dir / "cache"
    wavs_placeholder.mkdir(exist_ok=True)

    export_cmd = (
        "if [ ! -f .venv/bin/python3 ]; then "
        "  python3 -m venv --system-site-packages .venv; "
        "fi && "
        "if ! .venv/bin/python3 -c 'import piper_train' 2>/dev/null; then "
        "  echo '[reexport] Installing piper_train …' && "
        "  .venv/bin/pip install -e /piper/src/python 2>&1; "
        "fi && "
        # Downgrade PyTorch to 2.6.x for ONNX export — >=2.7 ignores opset_version=15.
        "echo '[reexport] Downgrading PyTorch to 2.6.x for opset-15 export …' && "
        ".venv/bin/pip install --cache-dir /piper/pip-cache "
        "'torch>=2.6,<2.7' --index-url https://download.pytorch.org/whl/cu124 2>&1 && "
        ".venv/bin/python3 -c \"import torch; print(f'[reexport] PyTorch {torch.__version__}')\" && "
        # Inline export — same logic as piper's export_onnx.py.
        f".venv/bin/python3 -c \""
        "import sys, torch; "
        "sys.path.insert(0, '/piper/src'); "
        "from piper.train.vits.lightning import VitsModel; "
        "model = VitsModel.load_from_checkpoint(sys.argv[1], dataset=None); "
        "g = model.model_g; g.eval(); "
        "torch.no_grad().__enter__(); "
        "g.dec.remove_weight_norm(); "
        "ns = g.n_speakers; "
        "seq = torch.randint(0, g.n_vocab, (1, 50), dtype=torch.long); "
        "sl = torch.LongTensor([50]); "
        "sc = torch.FloatTensor([0.667, 1.0, 0.8]); "
        "sid = torch.LongTensor([0]) if ns > 1 else None; "
        "dummy = (seq, sl, sc, sid); "
        "def fwd(text, text_lengths, scales, sid=None): "
        "    return g.infer(text, text_lengths, noise_scale=scales[0], "
        "        length_scale=scales[1], noise_scale_w=scales[2], sid=sid)[0].unsqueeze(1); "
        "g.forward = fwd; "
        "names_in = ['input','input_lengths','scales'] + (['sid'] if ns > 1 else []); "
        "torch.onnx.export("
        "    model=g, args=dummy, f=sys.argv[2], verbose=False, "
        "    opset_version=15, "
        "    do_constant_folding=True, "
        "    input_names=names_in, output_names=['output'], "
        "    dynamic_axes={" + "'input':{0:'batch_size',1:'phonemes'}, "
        "'input_lengths':{0:'batch_size'}, 'output':{0:'batch_size',1:'time'}}); "
        f"print(f'Exported {{sys.argv[2]}} (opset 15)')\" "
        f"{ckpt_container_path} ./checkpoints/{onnx_filename}"
    )

    _run_podman(
        wavs_dir=wavs_placeholder,
        metadata_dir=wavs_placeholder,
        checkpoints_dir=checkpoints_dir,
        lightning_logs_dir=lightning_logs_dir,
        shell_cmd=export_cmd,
        use_gpu=False,
    )

    if not onnx_output.exists():
        _die(f"Export failed — {onnx_output} was not produced.")

    # Enforce opset 15 on the host side as a safety net
    _enforce_opset_15(str(onnx_output))

    opset = None
    try:
        import onnx
        m = onnx.load(str(onnx_output), load_external_data=False)
        opset = next((op.version for op in m.opset_import if not op.domain), '?')
    except Exception:
        pass

    size_mb = onnx_output.stat().st_size / 1024 / 1024
    print(f"\n  Exported: {onnx_output} ({size_mb:.1f} MB, opset={opset})")
    print(f"\nNext step: copy to source model folder and run convert:")
    print(f"  copy \"{onnx_output}\" \"{model_dir / onnx_filename}\"")
    print(f"  python create-voice-model.py convert --input \"{model_dir}\" --output <out> --voice-name {voice_name}")


OPSET_VERSION_DISPLAY = 15  # mirrors export_onnx.OPSET_VERSION for display only




def cmd_train(args: argparse.Namespace) -> None:
    """
    Train a custom Piper voice inside a Podman container, then convert to
    a Sherpa-ONNX ZIP.

    Container image: veralvx/piper-train (https://github.com/veralvx/piper-train)
    Built locally on first run; subsequent runs reuse the cached image.

    Dataset layout (either accepted):
      <audio>/metadata.csv + <audio>/wav/*.wav
      <audio>/metadata.csv + <audio>/*.wav

    metadata.csv format (no header, pipe-separated):
      recording_001|Hello, how are you today?
      recording_002|The quick brown fox jumps over the lazy dog.
    """
    audio_dir = Path(args.audio).resolve()
    output_dir = Path(args.output).resolve()
    voice_name = args.voice
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_csv = audio_dir / "metadata.csv"
    if not metadata_csv.exists():
        _die(
            f"metadata.csv not found in {audio_dir}\n\n"
            "Create a file named metadata.csv with one line per recording:\n"
            "  recording_001|Hello, how are you today?\n"
            "  recording_002|The quick brown fox jumps over the lazy dog.\n\n"
            "Filename is without extension; path is relative to the audio folder."
        )

    _validate_and_clean_metadata(metadata_csv, audio_dir / "wav")

    _require_podman()

    repo_dir = _ensure_piper_train_repo(args.piper_train_repo)
    _ensure_piper_train_image(repo_dir)

    checkpoint_path: Optional[Path] = None
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint).resolve()
        if not checkpoint_path.exists():
            _die(f"Checkpoint not found: {checkpoint_path}")

    # Persistent directories — stay in the user's own folders
    wavs_dir = audio_dir / "wav"
    wavs_dir.mkdir(exist_ok=True)
    voice_dir = output_dir / voice_name
    checkpoints_dir = voice_dir / "checkpoints"
    lightning_logs_dir = voice_dir / "lightning_logs"
    for directory in (checkpoints_dir, lightning_logs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    print(f"\n[train] voice={voice_name}  audio={audio_dir}")

    # ── Step 1: Normalize audio to 22050 Hz mono 16-bit WAV ─────────────────
    existing_wavs = list(wavs_dir.glob("*.wav"))
    if existing_wavs:
        print(f"\n[1/4] Skipping normalization — {len(existing_wavs)} WAV(s) already in {wavs_dir}")
    else:
        print("\n[1/4] Normalizing audio to 22050 Hz mono WAV …")
        src_wav_dir = audio_dir / "wav" if (audio_dir / "raw").is_dir() else audio_dir
        _normalize_audio(src_wav_dir, wavs_dir)

    # ── Step 2: Copy checkpoint into checkpoints_dir ─────────────────────────
    checkpoint_container_path: Optional[str] = None
    if checkpoint_path:
        dest = checkpoints_dir / checkpoint_path.name
        if not dest.exists():
            shutil.copy2(str(checkpoint_path), str(dest))
            print(f"[2/4] Copied checkpoint: {checkpoint_path.name}")
        else:
            print(f"[2/4] Checkpoint already present: {checkpoint_path.name}")
        checkpoint_container_path = f"./checkpoints/{checkpoint_path.name}"
    else:
        print("[2/4] No checkpoint specified (training from scratch).")

    # ── Step 3: Run training + export inside the container ───────────────────
    print("[3/4] Training inside Podman container …")
    onnx_filename = f"{voice_name}.onnx"
    shell_cmd = _build_container_shell_command(
        voice_name=voice_name,
        language=args.language,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        onnx_filename=onnx_filename,
        checkpoint_container_path=checkpoint_container_path,
        use_gpu=args.gpu,
    )
    _run_podman(
        wavs_dir=wavs_dir,
        metadata_dir=audio_dir,
        checkpoints_dir=checkpoints_dir,
        lightning_logs_dir=lightning_logs_dir,
        shell_cmd=shell_cmd,
        use_gpu=args.gpu,
    )

    # ── Step 4: Collect ONNX for convert step ────────────────────────────────
    print("[4/4] Collecting outputs …")
    onnx_out = checkpoints_dir / onnx_filename
    if not onnx_out.exists():
        candidates = list(checkpoints_dir.glob("*.onnx"))
        if not candidates:
            _die("No .onnx file found after training/export.")
        onnx_out = max(candidates, key=lambda f: f.stat().st_mtime)
        print(f"       Using: {onnx_out.name}")

    # Stage ONNX + config in voice_dir root for the convert step
    staged_onnx = voice_dir / onnx_out.name
    shutil.copy2(str(onnx_out), str(staged_onnx))
    # Copy external data file if present (large models split weights)
    data_out = onnx_out.parent / (onnx_out.name + ".data")
    if data_out.exists():
        shutil.copy2(str(data_out), str(voice_dir / (onnx_out.name + ".data")))
    config_src = checkpoints_dir / "config.json"
    if config_src.exists():
        shutil.copy2(str(config_src), str(voice_dir / (onnx_out.stem + ".onnx.json")))

    print(f"\nTraining complete. Outputs in: {voice_dir}")

    # ── Convert the freshly trained model ────────────────────────────────────
    print("\n[convert] Packaging for SherpaTextToSpeechGateway …")
    convert_args = argparse.Namespace(
        input=str(voice_dir),
        output=str(output_dir),
        voice_name=voice_name,
    )
    cmd_convert(convert_args)


def _build_container_shell_command(
    voice_name: str,
    language: str,
    batch_size: int,
    max_epochs: int,
    onnx_filename: str,
    checkpoint_container_path: Optional[str],
    use_gpu: bool = False,
) -> str:
    """Build the shell command executed inside the piper-train container."""
    
    # 1. Define the Python export logic as a clean multi-line string
    # We use a heredoc (cat << 'EOF') to write this to a file inside the container
    python_export_logic = f"""
import sys
import torch
sys.path.insert(0, '/piper/src')
from piper.train.vits.lightning import VitsModel

print(f"Loading checkpoint: {{sys.argv[1]}}")
model = VitsModel.load_from_checkpoint(sys.argv[1], dataset=None)
g = model.model_g
g.eval()

with torch.no_grad():
    g.dec.remove_weight_norm()
    ns = g.n_speakers
    seq = torch.randint(0, g.n_vocab, (1, 50), dtype=torch.long)
    sl = torch.LongTensor([50])
    sc = torch.FloatTensor([0.667, 1.0, 0.8])
    sid = torch.LongTensor([0]) if ns > 1 else None
    dummy = (seq, sl, sc, sid)

    def fwd(text, text_lengths, scales, sid=None):
        return g.infer(text, text_lengths, noise_scale=scales[0],
                      length_scale=scales[1], noise_scale_w=scales[2], sid=sid)[0].unsqueeze(1)
    
    g.forward = fwd
    names_in = ['input', 'input_lengths', 'scales'] + (['sid'] if ns > 1 else [])
    
    print(f"Exporting to {{sys.argv[2]}} (opset 15)...")
    torch.onnx.export(
        model=g,
        args=dummy,
        f=sys.argv[2],
        verbose=False,
        opset_version=15,
        do_constant_folding=True,
        input_names=names_in,
        output_names=['output'],
        dynamic_axes={{
            'input': {{0: 'batch_size', 1: 'phonemes'}},
            'input_lengths': {{0: 'batch_size'}},
            'output': {{0: 'batch_size', 1: 'time'}}
        }}
    )
print("Export Successful!")
"""

    train_args = [
        ".venv/bin/python3", "-m", "piper.train", "fit",
        "--data.voice_name", voice_name,
        "--data.csv_path", "./metadata/metadata.csv",
        "--data.audio_dir", "./wavs",
        "--model.sample_rate", "22050",
        "--data.espeak_voice", language,
        "--data.cache_dir", "./cache",
        "--data.config_path", "./checkpoints/config.json",
        "--data.batch_size", str(batch_size),
        "--data.num_workers", "4",
        "--trainer.max_epochs", str(max_epochs),
        "--trainer.log_every_n_steps", "1",
        "--trainer.callbacks", "lightning.pytorch.callbacks.ModelCheckpoint",
        "--trainer.callbacks.every_n_epochs", "50",
        "--trainer.callbacks.save_top_k", "-1",
        "--trainer.callbacks.monitor", "null",
        "--trainer.callbacks.save_last", "true",
        "--trainer.callbacks.dirpath", "./checkpoints",
        "--trainer.callbacks.filename", "epoch={epoch}-step={step}",
        "--trainer.num_sanity_val_steps", "0",
    ]

    if use_gpu:
        train_args += [
            "--trainer.accelerator", "gpu",
            "--trainer.precision", "16-mixed",
            "--trainer.benchmark", "true",
        ]
    if checkpoint_container_path:
        train_args += ["--ckpt_path", checkpoint_container_path]

    # 2. Build the final shell command
    # We write the logic to /piper/checkpoints/export_script.py then run it.
    train_cmd = " ".join(shlex.quote(str(a)) for a in train_args)
    
    # Escape the python logic for a heredoc
    write_script_cmd = f"cat << 'EOF' > /piper/checkpoints/export_script.py\n{python_export_logic}\nEOF"
    
    run_export_cmd = f".venv/bin/python3 /piper/checkpoints/export_script.py ./checkpoints/last.ckpt ./checkpoints/{onnx_filename}"


    # Python 3.13 dropped pkg_resources from stdlib.  Bootstrap pip
    # into the venv (ensurepip still bundles it) then use pip to
    # install setuptools<78 (last version that ships pkg_resources).
    # We avoid `uv pip install` because uv may target a different
    # site-packages than .venv/bin/python3 actually reads from.
    bootstrap_cmd = (
        # cuBLASLt backend fix (applied later) handles Blackwell sm_120 bugs.
        # TF32 and non-deterministic cuBLAS are safe with cublaslt and boost GPU throughput.

        # Seed the venv volume if it was just created (empty mount)
        "if [ ! -f .venv/bin/python3 ]; then "
        "  echo '[bootstrap] Initialising .venv from base image …' && "
        "  python3 -m venv --system-site-packages .venv; "
        "fi && "
        # Print CUDA diagnostic so GPU issues are visible in logs
        "echo '[bootstrap] CUDA check …' && "
        "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null "
        "  || echo '[bootstrap] WARNING: nvidia-smi not available in container' && "
        # Ensure pip + setuptools (pkg_resources) are available
        "if ! .venv/bin/python3 -c 'import pkg_resources' 2>/dev/null; then "
        "  echo '[bootstrap] Installing setuptools into .venv …' && "
        "  .venv/bin/python3 -m ensurepip --default-pip 2>&1 && "
        "  .venv/bin/pip install 'setuptools<78' 2>&1 && "
        "  .venv/bin/python3 -c 'import pkg_resources; print(\"[bootstrap] setuptools OK\")'; "
        "else "
        "  echo '[bootstrap] setuptools already present'; "
        "fi && "
        # Upgrade PyTorch for Blackwell (sm_120) — need >=2.7 for native kernel support.
        # Older cu128 builds have incomplete sm_120 kernels (F.pad etc. trigger illegal instruction).
        "if .venv/bin/python3 -c 'import torch; v=torch.__version__; assert \"cu128\" in v; "
        "parts=v.split(\".\"); assert int(parts[0])>=2 and int(parts[1])>=7' 2>/dev/null; then "
        "  echo '[bootstrap] PyTorch >=2.7 cu128 already installed — skipping'; "
        "else "
        "  echo '[bootstrap] Upgrading PyTorch >=2.7 for Blackwell GPU (sm_120) …' && "
        "  .venv/bin/pip install --cache-dir /piper/pip-cache "
        "'torch>=2.7' 'torchvision>=0.22' 'torchaudio>=2.7' --index-url https://download.pytorch.org/whl/cu128 2>&1 && "
        "  echo '[bootstrap] PyTorch upgrade complete'; "
        "fi && "
        # Make piper importable in .venv (no setup.py, so use PYTHONPATH)
        "export PYTHONPATH=/piper/src:${PYTHONPATH:-} && "
        "echo '[bootstrap] PYTHONPATH set to /piper/src' && "
        # onnxruntime needed by piper/__init__.py and export_onnx
        "if ! .venv/bin/python3 -c 'import onnxruntime' 2>/dev/null; then "
        "  echo '[bootstrap] Installing onnxruntime …' && "
        "  .venv/bin/pip install --cache-dir /piper/pip-cache onnxruntime 2>&1 && "
        "  echo '[bootstrap] onnxruntime OK'; "
        "else "
        "  echo '[bootstrap] onnxruntime already present'; "
        "fi && "
        # lightning (pytorch-lightning) is required by piper.train
        "if ! .venv/bin/python3 -c 'import lightning' 2>/dev/null; then "
        "  echo '[bootstrap] Installing lightning …' && "
        "  .venv/bin/pip install --cache-dir /piper/pip-cache 'lightning>=2.4.0' 2>&1 && "
        "  echo '[bootstrap] lightning OK'; "
        "else "
        "  echo '[bootstrap] lightning already present'; "
        "fi && "
        # librosa needed by piper.train.vits.dataset
        "if ! .venv/bin/python3 -c 'import librosa' 2>/dev/null; then "
        "  echo '[bootstrap] Installing librosa …' && "
        "  .venv/bin/pip install --cache-dir /piper/pip-cache librosa 2>&1 && "
        "  echo '[bootstrap] librosa OK'; "
        "else "
        "  echo '[bootstrap] librosa already present'; "
        "fi && "
        # pysilero-vad needed by piper.train.vits.dataset
        # Must be exactly 2.1.1 — newer versions dropped the process_array API
        "if ! .venv/bin/python3 -c "
        "\"import importlib.metadata; assert importlib.metadata.version('pysilero-vad') == '2.1.1'\" "
        "2>/dev/null; then "
        "  echo '[bootstrap] Installing pysilero-vad==2.1.1 …' && "
        "  .venv/bin/pip install --cache-dir /piper/pip-cache 'pysilero-vad==2.1.1' 2>&1 && "
        "  echo '[bootstrap] pysilero-vad OK'; "
        "else "
        "  echo '[bootstrap] pysilero-vad already present'; "
        "fi && "
        # pathvalidate is required by the piper training pipeline
        "if ! .venv/bin/python3 -c 'import pathvalidate' 2>/dev/null; then "
        "  echo '[bootstrap] Installing pathvalidate …' && "
        "  .venv/bin/pip install --cache-dir /piper/pip-cache pathvalidate 2>&1 && "
        "  echo '[bootstrap] pathvalidate OK'; "
        "else "
        "  echo '[bootstrap] pathvalidate already present'; "
        "fi && "
        # jsonargparse[signatures] required by lightning.pytorch.cli
        "if ! .venv/bin/python3 -c 'import jsonargparse' 2>/dev/null; then "
        "  echo '[bootstrap] Installing jsonargparse[signatures] …' && "
        "  .venv/bin/pip install --cache-dir /piper/pip-cache 'jsonargparse[signatures]>=4.27.7' 2>&1 && "
        "  echo '[bootstrap] jsonargparse OK'; "
        "else "
        "  echo '[bootstrap] jsonargparse already present'; "
        "fi && "
        # Cython is required to compile monotonic_align extension
        "if ! .venv/bin/python3 -c 'import Cython' 2>/dev/null; then "
        "  echo '[bootstrap] Installing Cython …' && "
        "  .venv/bin/pip install --cache-dir /piper/pip-cache Cython 2>&1 && "
        "  echo '[bootstrap] Cython OK'; "
        "else "
        "  echo '[bootstrap] Cython already present'; "
        "fi && "
        # monotonic_align is a Cython extension that must be compiled before training
        "if ! .venv/bin/python3 -c "
        "'from piper.train.vits.monotonic_align.monotonic_align.core import maximum_path_c' "
        "2>/dev/null; then "
        "  echo '[bootstrap] Building monotonic_align Cython extension …' && "
        "  ( cd /piper/src && "
        "    /piper/.venv/bin/python3 piper/train/vits/monotonic_align/setup.py build_ext --inplace 2>&1 ) && "
        "  mkdir -p /piper/src/piper/train/vits/monotonic_align/monotonic_align && "
        "  if ls /piper/src/piper/train/vits/monotonic_align/core*.so >/dev/null 2>&1; then "
        "    mv /piper/src/piper/train/vits/monotonic_align/core*.so /piper/src/piper/train/vits/monotonic_align/monotonic_align/; "
        "  fi && "
        "  /piper/.venv/bin/python3 -c 'from piper.train.vits.monotonic_align.monotonic_align.core import maximum_path_c' && "
        "  echo '[bootstrap] monotonic_align OK'; "
        "else "
        "  echo '[bootstrap] monotonic_align already built'; "
        "fi && "
        # Clean up stale dataloader_gpu_fix import if present from prior runs
        "MAIN=/piper/src/piper/train/__main__.py && "
        "if grep -qF dataloader_gpu_fix \"$MAIN\" 2>/dev/null; then "
        "  echo '[bootstrap] Removing stale dataloader_gpu_fix …' && "
        "  /piper/.venv/bin/python3 -c "
        "'import pathlib; m=pathlib.Path(\"/piper/src/piper/train/__main__.py\"); "
        "t=m.read_text(); "
        "t=t.replace(\"from piper.train.vits import dataloader_gpu_fix  # noqa\\n\",\"\"); "
        "m.write_text(t)'; "
        "fi && "
        # Blackwell (sm_120) cuBLAS bug: cublasSgemmStridedBatched fails for certain
        # stride/shape combos regardless of precision (FP16 and FP32 both fail).
        # Fix: switch to cuBLASLt backend which uses a different GEMM code path.
        # Writes cublas_lt_fix.py and prepends its import to __main__.py.
        # Guard key: 'cublas_lt_fix' in __main__.py.
        "MAIN=/piper/src/piper/train/__main__.py && "
        "FIX=/piper/src/piper/train/vits/cublas_lt_fix.py && "
        "if ! grep -qF cublas_lt_fix \"$MAIN\" 2>/dev/null; then "
        "  echo '[bootstrap] Applying cuBLASLt backend fix for Blackwell …' && "
        "  printf 'import torch\\ntorch.backends.cuda.preferred_blas_library(\"cublaslt\")\\n' > \"$FIX\" && "
        "  /piper/.venv/bin/python3 -c "
        "'import pathlib; m=pathlib.Path(\"/piper/src/piper/train/__main__.py\"); "
        "t=m.read_text(); "
        "t=t.replace(\"from piper.train.vits import cublas_cpu_fix  # noqa\\n\",\"\"); "
        "t=t.replace(\"from piper.train.vits import cublas_contiguous_fix  # noqa\\n\",\"\"); "
        "t=t.replace(\"from piper.train.vits import cublas_fp32_fix  # noqa\\n\",\"\"); "
        "m.write_text(\"from piper.train.vits import cublas_lt_fix  # noqa\\n\"+t)' && "
        "  echo '[bootstrap] cuBLASLt fix applied'; "
        "else "
        "  echo '[bootstrap] cuBLASLt fix already present'; "
        "fi && "
        # ONNX export fix: torch.export cannot guard on data-dependent tensor booleans.
        # rational_quadratic_spline uses `assert (discriminant >= 0).all()` which triggers
        # GuardOnDataDependentSymNode during torch.onnx.export.
        # Fix: replace the assert with torch.clamp so the graph stays purely tensor ops.
        # Guard key: 'onnx-export-safe' in transforms.py.
        "TRANSFORMS=/piper/src/piper/train/vits/transforms.py && "
        "if ! grep -qF onnx-export-safe \"$TRANSFORMS\" 2>/dev/null; then "
        "  echo '[bootstrap] Patching transforms.py for ONNX export (discriminant clamp) …' && "
        "  /piper/.venv/bin/python3 -c "
        "'import pathlib; t=pathlib.Path(\"/piper/src/piper/train/vits/transforms.py\"); "
        "s=t.read_text(); "
        "s=s.replace("
        "\"    assert (discriminant >= 0).all(), discriminant\","
        "\"    discriminant = torch.clamp(discriminant, min=0.0)  # onnx-export-safe\"); "
        "t.write_text(s)' && "
        "  echo '[bootstrap] transforms.py patched'; "
        "else "
        "  echo '[bootstrap] transforms.py already patched'; "
        "fi && "
        # ONNX export fix: modules.py WaveNet.forward does a conv where torch.export
        # cannot prove x.shape[2] >= 1 statically, triggering GuardOnDataDependentSymNode.
        # Fix: insert torch._check before the offending conv call.
        # Guard key: 'onnx-export-safe-modules' in modules.py.
        "MODULES=/piper/src/piper/train/vits/modules.py && "
        "if ! grep -qF onnx-export-safe-modules \"$MODULES\" 2>/dev/null; then "
        "  echo '[bootstrap] Patching modules.py for ONNX export (torch._check shape guard) …' && "
        "  /piper/.venv/bin/python3 -c "
        "'import pathlib; t=pathlib.Path(\"/piper/src/piper/train/vits/modules.py\"); "
        "s=t.read_text(); "
        "s=s.replace("
        "\"        h = self.pre(x0) * x_mask\","
        "\"        torch._check(x.shape[2] >= 256)  # onnx-export-safe-modules\\n        h = self.pre(x0) * x_mask\"); "
        "t.write_text(s)' && "
        "  echo '[bootstrap] modules.py patched'; "
        "elif grep -qF 'x.shape[2] >= 1)' \"$MODULES\" 2>/dev/null; then "
        "  echo '[bootstrap] Upgrading modules.py shape guard (>= 1 → >= 256) …' && "
        "  sed -i 's/x\\.shape\\[2\\] >= 1)/x.shape[2] >= 256)/' \"$MODULES\" && "
        "  echo '[bootstrap] modules.py guard upgraded'; "
        "else "
        "  echo '[bootstrap] modules.py already patched'; "
        "fi"

    )

    # Downgrade PyTorch to 2.6.x before ONNX export.
    # PyTorch >=2.7 ignores opset_version=15 and always emits opset 18.
    # The legacy exporter in 2.6.x respects opset_version correctly.
    # Training needs >=2.7 for Blackwell (sm_120), so we downgrade only for export.
    downgrade_torch_cmd = (
        "echo '[export] Downgrading PyTorch to 2.6.x for opset-15 ONNX export …' && "
        ".venv/bin/pip install --cache-dir /piper/pip-cache "
        "'torch>=2.6,<2.7' --index-url https://download.pytorch.org/whl/cu124 2>&1 && "
        ".venv/bin/python3 -c \"import torch; print(f'[export] PyTorch {torch.__version__}')\" "
    )

    return f"{bootstrap_cmd} && {train_cmd} && {downgrade_torch_cmd} && {write_script_cmd} && {run_export_cmd}"

def _validate_and_clean_metadata(metadata_csv: Path, wavs_dir: Path) -> None:
    """Remove invalid rows from metadata.csv in-place and abort if none remain."""
    import csv
    rows = []
    bad_rows = []
    with open(metadata_csv, encoding="utf-8", newline="") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.rstrip("\n\r")
            if not line.strip():
                continue
            parts = line.split("|", 1)
            if len(parts) != 2:
                bad_rows.append((lineno, line, "no pipe separator"))
                continue
            file_id, text = parts[0].strip(), parts[1].strip()
            if not file_id:
                bad_rows.append((lineno, line, "empty file id"))
                continue
            if len(text.replace(" ", "")) < 2:
                bad_rows.append((lineno, line, f"transcript too short: {text!r}"))
                continue
            wav_path = wavs_dir / f"{file_id}.wav"
            if not wav_path.exists():
                bad_rows.append((lineno, line, f"wav not found: {wav_path.name}"))
                continue
            rows.append(f"{file_id}|{text}")

    if bad_rows:
        print(f"\n[preflight] Removed {len(bad_rows)} invalid row(s) from metadata.csv:")
        for lineno, line, reason in bad_rows:
            print(f"  line {lineno}: {reason}  →  {line[:80]}")
        if not rows:
            _die(
                "All rows in metadata.csv are invalid — nothing to train on.\n"
                "Fix the rows listed above and re-run."
            )
        with open(metadata_csv, "w", encoding="utf-8", newline="") as f:
            f.write("\n".join(rows) + "\n")
        print(f"[preflight] metadata.csv cleaned — {len(rows)} valid row(s) remain.")
    else:
        print(f"[preflight] metadata.csv OK — {len(rows)} row(s).")


def _run_podman(
    wavs_dir: Path,
    metadata_dir: Path,
    checkpoints_dir: Path,
    lightning_logs_dir: Path,
    shell_cmd: str,
    use_gpu: bool,
) -> None:
    """Run the piper-train container, mounting the caller's actual directories."""
    # Persist preprocessing cache (mel spectrograms, phoneme alignments)
    # so they survive container restarts instead of being recomputed on CPU every run
    cache_dir = checkpoints_dir.parent / "cache"
    cache_dir.mkdir(exist_ok=True)
    cmd = [
        "podman", "run", "--rm",
        "--shm-size=4g" if use_gpu else "--shm-size=1g",
        "-v", f"{wavs_dir}:/piper/wavs",
        "-v", f"{metadata_dir}:/piper/metadata",
        "-v", f"{checkpoints_dir}:/piper/checkpoints",
        "-v", f"{lightning_logs_dir}:/piper/lightning_logs",
        "-v", f"{cache_dir}:/piper/cache",
    ]
    if use_gpu:
        cmd.append("--device=nvidia.com/gpu=all")
        # Persist .venv so PyTorch cu128 upgrade survives container restarts
        venv_dir = checkpoints_dir.parent / "venv-cache"
        venv_dir.mkdir(exist_ok=True)
        cmd += ["-v", f"{venv_dir}:/piper/.venv"]
        # Persist pip cache so wheel downloads are not re-fetched
        pip_cache = checkpoints_dir.parent / "pip-cache"
        pip_cache.mkdir(exist_ok=True)
        cmd += ["-v", f"{pip_cache}:/piper/pip-cache"]
    cmd += [PIPER_TRAIN_IMAGE, "/bin/sh", "-c", shell_cmd]
    _run(cmd)


def _require_podman() -> None:
    """Verify podman is on PATH."""
    if shutil.which("podman") is None:
        _die("podman not found on PATH. Install from https://podman.io")


def _ensure_piper_train_repo(repo_arg: Optional[str]) -> Path:
    """Return the path to the veralvx/piper-train clone, cloning if needed."""
    if repo_arg:
        path = Path(repo_arg).resolve()
        if not (path / "Dockerfile").exists():
            _die(f"No Dockerfile found in {path} — not a valid piper-train clone.")
        return path

    default = Path.home() / "piper-train"
    if (default / "Dockerfile").exists():
        return default

    print(f"  Cloning {PIPER_TRAIN_REPO_URL} → {default} …")
    _run(["git", "clone", PIPER_TRAIN_REPO_URL, str(default)])
    return default


def _ensure_piper_train_image(repo_dir: Path) -> None:
    """Build the piper-train container image if it does not already exist."""
    result = subprocess.run(
        ["podman", "image", "exists", PIPER_TRAIN_IMAGE],
        capture_output=True,
    )
    if result.returncode == 0:
        print(f"  Container image '{PIPER_TRAIN_IMAGE}' already exists.")
        return
    print(f"  Building '{PIPER_TRAIN_IMAGE}' image (this may take several minutes) …")
    _run(["podman", "build", "-f", str(repo_dir / "Dockerfile"), "-t", PIPER_TRAIN_IMAGE, str(repo_dir)])


def _normalize_audio(src_dir: Path, dest_dir: Path) -> None:
    """Resample all WAV files to 22050 Hz mono 16-bit PCM."""
    try:
        import soundfile as sf
        import librosa
    except ImportError:
        _die("Audio normalization requires: pip install soundfile librosa")

    wav_files = list(src_dir.glob("*.wav")) + list(src_dir.glob("*.mp3")) + \
                list(src_dir.glob("*.flac")) + list(src_dir.glob("*.m4a"))
    if not wav_files:
        _die(f"No audio files found in {src_dir}")

    for audio_file in wav_files:
        audio, sr = librosa.load(str(audio_file), sr=22050, mono=True)
        out_path = dest_dir / (audio_file.stem + ".wav")
        sf.write(str(out_path), audio, 22050, subtype="PCM_16")

    print(f"  Normalized {len(wav_files)} audio file(s) to {dest_dir}")


def _download(url: str, dest: str) -> None:
    def _progress(count, block_size, total_size):
        if total_size > 0:
            pct = count * block_size * 100 // total_size
            print(f"\r    {pct}%", end="", flush=True)
    urllib.request.urlretrieve(url, dest, reporthook=_progress)
    print()


def _run(cmd: list) -> None:
    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        _die(f"Command failed (exit {result.returncode}):\n  {' '.join(str(c) for c in cmd)}")


def _check_train_deps() -> None:
    missing = []
    for pkg in ("soundfile", "librosa"):
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        _die(
            "Missing Python packages for audio normalization:\n"
            f"  pip install {' '.join(missing)}"
        )


def _die(message: str) -> None:
    print(f"\nERROR: {message}", file=sys.stderr)
    sys.exit(1)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create Sherpa-ONNX Piper TTS voice model ZIPs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── reexport ──
    p_reexport = subparsers.add_parser(
        "reexport",
        help="Re-run ONNX export from a checkpoint in the Podman container (no re-training).",
    )
    p_reexport.add_argument(
        "--model-dir", required=True,
        help="Folder containing the checkpoints/ subdirectory (e.g. models/john_cena)",
    )
    p_reexport.add_argument(
        "--checkpoint", required=True,
        help="Path to the .ckpt file to export (must be inside <model-dir>/checkpoints/)",
    )
    p_reexport.add_argument(
        "--voice-name",
        help="Override output ONNX filename stem (defaults to model-dir name)",
    )

    # ── convert ──
    p_convert = subparsers.add_parser(
        "convert",
        help="Convert existing Piper model files to Sherpa-ONNX ZIP.",
    )
    p_convert.add_argument(
        "--input", required=True,
        help="Folder containing <model>.onnx and <model>.onnx.json",
    )
    p_convert.add_argument(
        "--output", required=True,
        help="Output folder for the generated ZIP",
    )
    p_convert.add_argument(
        "--voice-name",
        help="Override the voice/ZIP name (defaults to ONNX filename stem)",
    )

    # ── train ──
    p_train = subparsers.add_parser(
        "train",
        help="Train a custom Piper voice from audio files using Podman.",
    )
    p_train.add_argument(
        "--audio", required=True,
        help="Folder with WAV recordings + metadata.csv",
    )
    p_train.add_argument(
        "--voice", required=True,
        help="Name for the voice (used for output filenames)",
    )
    p_train.add_argument(
        "--output", required=True,
        help="Output folder for the model ZIP",
    )
    p_train.add_argument(
        "--language", default="en-us",
        help="espeak-ng voice code for phonemization (default: en-us)",
    )
    p_train.add_argument(
        "--max-epochs", type=int, default=400,
        help="Maximum training epochs (default: 400)",
    )
    p_train.add_argument(
        "--batch-size", type=int, default=16,
        help="Training batch size (default: 16, increase to 32 for large VRAM)",
    )
    p_train.add_argument(
        "--checkpoint",
        help="Path to a pretrained .ckpt file for fine-tuning (strongly recommended). "
             "Download from https://huggingface.co/datasets/rhasspy/piper-checkpoints/tree/main",
    )
    p_train.add_argument(
        "--piper-train-repo",
        help="Path to a local clone of veralvx/piper-train. "
             "Defaults to ~/piper-train; cloned automatically if absent.",
    )
    p_train.add_argument(
        "--gpu", action="store_true",
        help="Pass --gpus=all to podman run (requires CUDA in the container image)",
    )

    args = parser.parse_args()

    if args.command == "reexport":
        cmd_reexport(args)
    elif args.command == "convert":
        cmd_convert(args)
    elif args.command == "train":
        cmd_train(args)


if __name__ == "__main__":
    main()
