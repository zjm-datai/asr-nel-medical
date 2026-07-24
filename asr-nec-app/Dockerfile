# syntax=docker/dockerfile:1

FROM node:24-alpine AS frontend-builder

WORKDIR /build/ui
COPY asr-nec-app/ui/package.json asr-nec-app/ui/package-lock.json ./
RUN npm ci
COPY asr-nec-app/ui/ ./
RUN npm run build


ARG CUDA_BASE_IMAGE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
FROM ${CUDA_BASE_IMAGE} AS production

ARG PYTHON_VERSION=3.11
ARG PYTORCH_VERSION=2.6.0
ARG MINICONDA_VERSION=py311_25.5.1-1
ARG TARGETARCH
ARG PYPI_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PATH=/opt/conda/bin:${PATH} \
    PIP_INDEX_URL=${PYPI_INDEX_URL} \
    PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10

RUN apt-get update \
    && apt-get install -y --no-install-recommends bash bzip2 ca-certificates curl ffmpeg git libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN case "${TARGETARCH:-amd64}" in \
        amd64) CONDA_ARCH=x86_64 ;; \
        arm64) CONDA_ARCH=aarch64 ;; \
        *) echo "Unsupported architecture: ${TARGETARCH}" >&2; exit 1 ;; \
    esac \
    && curl -fsSL -o /tmp/miniconda.sh \
        "https://repo.anaconda.com/miniconda/Miniconda3-${MINICONDA_VERSION}-Linux-${CONDA_ARCH}.sh" \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh \
    && python -c "import sys; assert sys.version_info[:2] == tuple(map(int, '${PYTHON_VERSION}'.split('.')))" \
    && conda clean -afy

RUN python -m pip install --upgrade pip \
    && python -m pip install "torch==${PYTORCH_VERSION}" --index-url https://download.pytorch.org/whl/cu124

WORKDIR /app
COPY asr-nec-model/ ./asr-nec-model/
COPY asr-nec-app/ ./asr-nec-app/
COPY --from=frontend-builder /build/ui/dist ./asr-nec-app/ui/dist/

RUN python -m pip install -e ./asr-nec-model \
    && python -m pip install ./asr-nec-app \
    && chmod +x ./asr-nec-app/entrypoint.sh

WORKDIR /app/asr-nec-app
EXPOSE 8016
ENTRYPOINT ["/bin/bash", "./entrypoint.sh"]
