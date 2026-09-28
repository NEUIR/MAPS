#!/usr/bin/env bash
set -euo pipefail

# Start the retrieval microservice (FastAPI wrapper around a FlashRAG retriever).
#
# Settings can be provided in two ways (CLI flags override environment vars,
# which override the defaults below):
#   RETRIEVER_CONFIG_FILE : config YAML for the retriever   (default: retrieval_server/retriever_config.yaml)
#   RETRIEVER_NUM         : number of retriever instances   (default: 1)
#   RETRIEVER_PORT        : HTTP port to serve on           (default: 9100)
#   RETRIEVER_GPU_ID      : GPU ids used by the retriever   (default: 0,1)
#   PYTHON_BIN            : python interpreter              (default: python)
#
# Usage:
#   bash scripts/retrieval_server.sh
#   RETRIEVER_PORT=9110 bash scripts/retrieval_server.sh
#   bash scripts/retrieval_server.sh --port 9110 --gpu-id "2,3"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

RETRIEVER_CONFIG_FILE="${RETRIEVER_CONFIG_FILE:-retrieval_server/retriever_config.yaml}"
RETRIEVER_NUM="${RETRIEVER_NUM:-1}"
RETRIEVER_PORT="${RETRIEVER_PORT:-9100}"
RETRIEVER_GPU_ID="${RETRIEVER_GPU_ID:-0,1}"
PYTHON_BIN="${PYTHON_BIN:-python}"

usage() {
  cat <<EOF
Usage: bash scripts/retrieval_server.sh [OPTIONS]

Options:
  --config-file PATH    Retriever config YAML  (env: RETRIEVER_CONFIG_FILE, default: retrieval_server/retriever_config.yaml)
  --num-retriever N     Number of retriever instances (env: RETRIEVER_NUM, default: 1)
  --port PORT           HTTP port to serve on  (env: RETRIEVER_PORT, default: 9100)
  --gpu-id IDS          GPU ids used by the retriever (env: RETRIEVER_GPU_ID, default: 0,1)
  -h, --help            Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config-file) shift; RETRIEVER_CONFIG_FILE="${1:?--config-file requires a value}"; shift ;;
    --num-retriever) shift; RETRIEVER_NUM="${1:?--num-retriever requires a value}"; shift ;;
    --port) shift; RETRIEVER_PORT="${1:?--port requires a value}"; shift ;;
    --gpu-id) shift; RETRIEVER_GPU_ID="${1:?--gpu-id requires a value}"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[RetrievalServer] Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

echo "[RetrievalServer] config=${RETRIEVER_CONFIG_FILE} num_retriever=${RETRIEVER_NUM} port=${RETRIEVER_PORT} gpu_id=${RETRIEVER_GPU_ID}"

exec "${PYTHON_BIN}" retrieval_server/app.py \
  --config_file_path "${RETRIEVER_CONFIG_FILE}" \
  --num_retriever "${RETRIEVER_NUM}" \
  --port "${RETRIEVER_PORT}" \
  --gpu_id "${RETRIEVER_GPU_ID}"
