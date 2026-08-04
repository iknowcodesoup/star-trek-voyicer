ARG CUDA_IMAGE="12.9.0-devel-ubuntu24.04"
FROM nvidia/cuda:${CUDA_IMAGE}

ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y git build-essential \
       python3 python3-pip python3.12-venv gcc wget \
       ocl-icd-opencl-dev opencl-headers clinfo \
       libclblast-dev libopenblas-dev \
       rsync sox ffmpeg cmake \
    && mkdir -p /etc/OpenCL/vendors \
    && echo "libnvidia-opencl.so.1" > /etc/OpenCL/vendors/nvidia.icd

ENV CUDA_DOCKER_ARCH=all

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /usr/local/src
RUN git clone https://github.com/rhasspy/piper.git
WORKDIR /usr/local/src/piper/src/python
RUN git checkout 2023.11.14-2
RUN sed -i 's/piper-phonemize~=1.1.0/piper-phonemize-fix/' requirements.txt
# setuptools<82: pkg_resources was removed in setuptools 82.0.0 (Feb 2026), but
# lightning_fabric (pulled in by pytorch-lightning==1.9.3 below) still imports it.
RUN pip install --upgrade wheel "setuptools<82"
# torch==2.7.1+cu128 -- the article's original pin (torch<2.6.0) targeted a 2016
# Tesla P4 and predates Blackwell: it ships no sm_120 kernels, so RTX 50-series
# GPUs (e.g. RTX 5060) fail at runtime with "no kernel image is available for
# execution on the device". 2.7.0 was the first stable release with prebuilt
# sm_120 kernels, via the cu128 wheel index (the default PyPI index resolves to
# a non-CUDA-matching build, hence the explicit --index-url). Deliberately NOT
# bumping pytorch-lightning past 1.9.3 to go with it: piper_train's own code
# constructs Trainer(resume_from_checkpoint=...), a kwarg removed in lightning
# 2.0, so upgrading lightning would break piper_train outright. Installed before
# pytorch-lightning/-e . so: (1) pytorch-lightning's own torch dependency is already
# satisfied and doesn't trigger a second, different torch download; (2) edits to the
# fragile pip lines below don't invalidate this expensive layer's build cache.
RUN pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu128
RUN pip install piper-phonemize-fix onnxruntime-gpu onnxsim onnx rich librosa \
    pytorch-lightning==1.9.3 cython numpy tensorboard
# pytorch-lightning's default Trainer logger is TensorBoardLogger, but it
# silently falls back to CSVLogger (metrics.csv, no .tfevents) when the
# tensorboard package isn't importable -- installing it here is what makes
# piper_train emit real tfevents instead.
# piper_train hardcodes num_workers=1 (and never exposes it as a CLI flag, and
# never passes pin_memory/persistent_workers), so the training DataLoader runs
# single-process no matter what -- the GPU stalls waiting on one CPU thread to
# deserialize/collate each batch of preprocessed .pt tensors. Bump the default
# and turn on pin_memory/persistent_workers so multiple workers actually get used.
# (Patched here, right before the editable install, so it doesn't invalidate the
# expensive torch/pip layers above on rebuild.)
RUN sed -i 's/num_workers: int = 1,/num_workers: int = 8,/' piper_train/vits/lightning.py
# prefetch_factor defaults to 2 batches/worker; docs/dataloader-perf-spec.md Part 1
# confirmed the Windows bind mount isn't the bottleneck (an A/B test against a
# native volume showed identical throughput), but giving each of the 8 workers
# more batches staged ahead is still a cheap, no-risk way to absorb whatever
# per-batch jitter remains, independent of where that jitter comes from.
RUN sed -i 's/num_workers=self.hparams.num_workers,/num_workers=self.hparams.num_workers,\n            prefetch_factor=4,\n            pin_memory=True,\n            persistent_workers=True,/g' piper_train/vits/lightning.py
# piper_train builds ModelCheckpoint(every_n_epochs=args.checkpoint_epochs) with no
# save_top_k, so Lightning's default (save_top_k=1) deletes every prior checkpoint
# as soon as the next one is saved -- there is no way to compare an old checkpoint
# against a new one after the fact, or recover if a later epoch trains worse.
# save_top_k=-1 keeps every matching checkpoint instead of just the latest one.
# NOTE: save_top_k=10 was tried first to bound this directly, but Lightning 1.9.3
# rejects any save_top_k outside {-1, 0, 1} when monitor=None ("No quantity for
# top_k to track") -- ranking "top k" needs a tracked metric, and this training
# run has no validation split (--validation-split 0.0) and no single loss that
# meaningfully ranks a GAN checkpoint as "better" anyway. So retention is bounded
# on the host instead: main.py prunes work/<character>/training/**/*.ckpt down to
# the most recent N while training runs (see CHECKPOINT_KEEP in main.py).
RUN sed -i 's/trainer.callbacks = \[ModelCheckpoint(every_n_epochs=args.checkpoint_epochs)\]/trainer.callbacks = [ModelCheckpoint(every_n_epochs=args.checkpoint_epochs, save_top_k=-1)]/' piper_train/__main__.py
# --no-deps here is required: piper-train's own metadata pins torch<2 and
# pytorch-lightning~=1.7, which would otherwise downgrade the versions installed above.
# Its real runtime deps (librosa etc.) are already satisfied by the line above.
RUN pip install --no-deps -e .
RUN ./build_monotonic_align.sh
# src/python_run is Piper's separate inference-only package (published as piper-tts
# on PyPI, console script `piper`) -- used by docker/sample_checkpoints.sh to
# synthesize validation audio from each retained checkpoint's exported ONNX model,
# for comparing checkpoint quality by ear. --no-deps for the same reason as above:
# its requirements.txt (piper-phonemize~=1.1.0, onnxruntime<2) are already satisfied
# by piper-phonemize-fix and onnxruntime-gpu installed earlier.
RUN pip install --no-deps -e ../python_run

WORKDIR /app
# CUDA_LAUNCH_BLOCKING=1 was previously set here -- it forces every CUDA kernel
# launch to block synchronously, which serializes CPU/GPU work and was stalling
# the GPU between batches. Re-add it temporarily (docker run -e CUDA_LAUNCH_BLOCKING=1)
# only if you need an exact stack trace for a CUDA error.
ENV HF_HOME=/app/.cache/huggingface
ENV NUMBA_CACHE_DIR=/app/.cache/numba_cache
ENV XDG_CACHE_HOME=/app/.cache
ENV TORCH_CPP_LOG_LEVEL=ERROR
ENV TQDM_DISABLE=1
# torch>=2.6 defaults torch.load to weights_only=True, which rejects the
# pathlib.PosixPath global inside Piper's official ljspeech-2000.ckpt (and any
# checkpoint saved the same way) with an UnpicklingError. pytorch_lightning
# calls torch.load(f, map_location=...) internally without passing
# weights_only itself, so there's no call site here to pass it at -- this env
# var is torch's own supported override for exactly that case, applied only
# when the caller left weights_only unset. Safe here since the checkpoints
# loaded are Piper's official release or this pipeline's own prior output.
ENV TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
