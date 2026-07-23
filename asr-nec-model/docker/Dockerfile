ARG PYTORCH_IMAGE=pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime
FROM ${PYTORCH_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/asr-nec-model

COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --upgrade pip \
    && python -m pip install -e .

COPY scripts ./scripts
COPY tests ./tests
COPY examples ./examples
COPY SERVER_TRAINING.md DOCKER_TRAINING.md ./

RUN chmod +x scripts/run_ss_training.sh

CMD ["bash", "scripts/run_ss_training.sh"]
