#!/usr/bin/env bash
set -Eeuo pipefail

WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO_ROOT="${WORKSPACE_ROOT}/asr-nec-model"
RUN_DIR="${WORKSPACE_ROOT}/runs/ss_pilot"
FEATURE_DIR="${WORKSPACE_ROOT}/data/speech_searcher/ss_features"
MODEL_PATH="${WHISPER_MODEL:-${WORKSPACE_ROOT}/weights/base.pt}"

mkdir -p "${RUN_DIR}"
cd "${REPO_ROOT}"

exec > >(tee -a "${RUN_DIR}/container.log") 2>&1

echo "[$(date --iso-8601=seconds)] SpeechSearcher container job started"
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
if [[ ! -f "${WORKSPACE_ROOT}/data/speech_searcher/audio_pilot/pilot_manifest.jsonl" ]]; then
  echo "Pilot data is missing under ${WORKSPACE_ROOT}/data/speech_searcher/audio_pilot" >&2
  exit 2
fi

echo "[$(date --iso-8601=seconds)] extracting/resuming frozen Whisper features"
python scripts/extract_ss_features.py \
  --model "${MODEL_PATH}" \
  --device cuda \
  --batch-size "${FEATURE_BATCH_SIZE:-16}" \
  --output-dir "${FEATURE_DIR}"

TRAIN_ARGS=(
  --device cuda
  --epochs "${EPOCHS:-20}"
  --batch-size "${TRAIN_BATCH_SIZE:-32}"
  --eval-batch-size "${EVAL_BATCH_SIZE:-64}"
  --learning-rate "${LEARNING_RATE:-1e-4}"
  --num-workers "${NUM_WORKERS:-4}"
  --patience "${PATIENCE:-5}"
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
