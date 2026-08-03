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
    pytorch-lightning==1.9.3 cython numpy
# --no-deps here is required: piper-train's own metadata pins torch<2 and
# pytorch-lightning~=1.7, which would otherwise downgrade the versions installed above.
# Its real runtime deps (librosa etc.) are already satisfied by the line above.
RUN pip install --no-deps -e .
RUN ./build_monotonic_align.sh

WORKDIR /app
ENV HF_HOME=/app/.cache/huggingface
ENV NUMBA_CACHE_DIR=/app/.cache/numba_cache
ENV XDG_CACHE_HOME=/app/.cache
ENV CUDA_LAUNCH_BLOCKING=1
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
