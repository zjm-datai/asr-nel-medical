#!/usr/bin/env bash
set -Eeuo pipefail

WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO_ROOT="${WORKSPACE_ROOT}/asr-nec-model"
MODEL_PATH="${WHISPER_MODEL:-${WORKSPACE_ROOT}/weights/base.pt}"
DATASET_MODE="${DATASET_MODE:-pilot}"

case "${DATASET_MODE}" in
  pilot)
    AUDIO_DIR="${WORKSPACE_ROOT}/data/speech_searcher/audio_pilot"
    MANIFEST_NAME="pilot_manifest.jsonl"
    FEATURE_DIR="${WORKSPACE_ROOT}/data/speech_searcher/ss_features"
    RUN_DIR="${WORKSPACE_ROOT}/runs/ss_pilot"
    ;;
  full)
    AUDIO_DIR="${WORKSPACE_ROOT}/data/speech_searcher/audio_full"
    MANIFEST_NAME="full_manifest.jsonl"
    FEATURE_DIR="${WORKSPACE_ROOT}/data/speech_searcher/ss_features_full"
    RUN_DIR="${WORKSPACE_ROOT}/runs/ss_full_seed_${SEED:-20260724}"
    ;;
  *)
    echo "Unsupported DATASET_MODE: ${DATASET_MODE}" >&2
    exit 2
    ;;
esac

mkdir -p "${RUN_DIR}"
cd "${REPO_ROOT}"

exec > >(tee -a "${RUN_DIR}/container.log") 2>&1

echo "[$(date --iso-8601=seconds)] SpeechSearcher container job started mode=${DATASET_MODE}"
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable inside the container")
print("gpu:", torch.cuda.get_device_name(0))
print("cuda_runtime:", torch.version.cuda)
PY

if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "Whisper checkpoint not found: ${MODEL_PATH}" >&2
  exit 2
fi
if [[ ! -f "${AUDIO_DIR}/${MANIFEST_NAME}" ]]; then
  echo "Audio manifest is missing: ${AUDIO_DIR}/${MANIFEST_NAME}" >&2
  exit 2
fi

if [[ "${DATASET_MODE}" == "full" ]]; then
  echo "[$(date --iso-8601=seconds)] binding full-corpus SS pairs"
  python scripts/build_ss_full_pairs.py
fi

echo "[$(date --iso-8601=seconds)] extracting/resuming frozen Whisper features"
python scripts/extract_ss_features.py \
  --audio-dir "${AUDIO_DIR}" \
  --manifest-name "${MANIFEST_NAME}" \
  --model "${MODEL_PATH}" \
  --device cuda \
  --batch-size "${FEATURE_BATCH_SIZE:-16}" \
  --output-dir "${FEATURE_DIR}"

TRAIN_ARGS=(
  --device cuda
  --epochs "${EPOCHS:-20}"
  --batch-size "${TRAIN_BATCH_SIZE:-32}"
  --eval-batch-size "${EVAL_BATCH_SIZE:-256}"
  --learning-rate "${LEARNING_RATE:-1e-4}"
  --num-workers "${NUM_WORKERS:-4}"
  --patience "${PATIENCE:-5}"
  --seed "${SEED:-20260724}"
  --data-dir "${AUDIO_DIR}"
  --feature-dir "${FEATURE_DIR}"
  --output-dir "${RUN_DIR}"
)

if [[ "${AUTO_RESUME:-1}" == "1" && -f "${RUN_DIR}/last.pt" ]]; then
  echo "[$(date --iso-8601=seconds)] resuming from ${RUN_DIR}/last.pt"
  TRAIN_ARGS+=(--resume "${RUN_DIR}/last.pt")
else
  echo "[$(date --iso-8601=seconds)] starting a new training run"
fi

python scripts/train_speech_searcher_pilot.py "${TRAIN_ARGS[@]}"
echo "[$(date --iso-8601=seconds)] SpeechSearcher container job completed"
