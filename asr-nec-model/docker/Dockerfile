ARG CUDA_BASE_IMAGE=nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
FROM ${CUDA_BASE_IMAGE}

ARG PYTHON_VERSION=3.11
ARG PYTORCH_VERSION=2.6.0
ARG TARGETARCH

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PATH=/opt/conda/bin:${PATH} \
    HF_HOME=/root/.cache/huggingface \
    TRANSFORMERS_CACHE=/root/.cache/huggingface/transformers

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        bash bzip2 ca-certificates curl ffmpeg git libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN case "${TARGETARCH:-amd64}" in \
        amd64) CONDA_ARCH=x86_64 ;; \
        arm64) CONDA_ARCH=aarch64 ;; \
        *) echo "Unsupported architecture: ${TARGETARCH}" >&2; exit 1 ;; \
    esac \
    && curl -fsSL -o /tmp/miniforge.sh \
        "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-${CONDA_ARCH}.sh" \
    && bash /tmp/miniforge.sh -b -p /opt/conda \
    && rm /tmp/miniforge.sh \
    && conda install -y "python=${PYTHON_VERSION}" pip \
    && conda clean -afy

RUN python -m pip install --upgrade pip \
    && python -m pip install \
        "torch==${PYTORCH_VERSION}" \
        --index-url https://download.pytorch.org/whl/cu124

WORKDIR /workspace/asr-nec-model

COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install -e ".[dev]"

COPY scripts ./scripts
COPY tests ./tests
COPY examples ./examples
COPY SERVER_TRAINING.md DOCKER_TRAINING.md ./

RUN chmod +x scripts/run_ss_training.sh

CMD ["bash", "scripts/run_ss_training.sh"]
