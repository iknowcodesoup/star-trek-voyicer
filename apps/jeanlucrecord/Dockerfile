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
RUN pip install --upgrade wheel setuptools
RUN pip install piper-phonemize-fix onnxruntime-gpu onnxsim onnx rich pytorch-lightning==1.9.3 \
    cython --no-deps numpy -e .
# torch<2.6.0 is the article's pin for a 2016 Tesla P4 — likely too old for newer
# GPU generations' prebuilt CUDA kernels. Verify with the CUDA smoke test before
# training; bump this (and pytorch-lightning above) together if it fails.
RUN pip install "torch<2.6.0"
RUN ./build_monotonic_align.sh

WORKDIR /app
ENV HF_HOME=/app/.cache/huggingface
ENV NUMBA_CACHE_DIR=/app/.cache/numba_cache
ENV XDG_CACHE_HOME=/app/.cache
ENV CUDA_LAUNCH_BLOCKING=1
ENV TORCH_CPP_LOG_LEVEL=ERROR
ENV TQDM_DISABLE=1
