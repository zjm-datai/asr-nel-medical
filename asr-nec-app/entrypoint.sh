#!/bin/bash

set -Eeuo pipefail

if [[ "${RUN_MIGRATIONS:-true}" == "true" ]]; then
    echo "Applying database migrations..."
    alembic upgrade head
fi

if (( $# > 0 )); then
    exec "$@"
fi

LOG_LEVEL=${LOG_LEVEL:-info}
echo "Starting ASR NEC demo on ${API_HOST:-0.0.0.0}:${API_PORT:-8016}"
exec uvicorn "${APP_MODULE:-app:app}" \
    --host "${API_HOST:-0.0.0.0}" \
    --port "${API_PORT:-8016}" \
    --workers 1 \
    --log-level "${LOG_LEVEL,,}"
