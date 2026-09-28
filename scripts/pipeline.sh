#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# --- vLLM server ---
VLLM_MODEL="${VLLM_MODEL:-Qwen3-32B}"
VLLM_PORT="${VLLM_PORT:-9101}"
VLLM_API_KEY="${VLLM_API_KEY:-TEST}"
VLLM_GPU_IDS="${VLLM_GPU_IDS:-2,3,4,5}"
VLLM_TP="${VLLM_TP:-4}"
VLLM_DP="${VLLM_DP:-1}"
VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.7}"
VLLM_EXTRA_ARGS="${VLLM_EXTRA_ARGS:-}"

# --- retrieval server (consumed by scripts/retrieval_server.sh) ---
RETRIEVER_PORT="${RETRIEVER_PORT:-9100}"
RETRIEVER_GPU_ID="${RETRIEVER_GPU_ID:-0,1}"
RETRIEVER_NUM="${RETRIEVER_NUM:-1}"
RETRIEVER_CONFIG_FILE="${RETRIEVER_CONFIG_FILE:-retrieval_server/retriever_config.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# --- pipeline behaviour ---
START_VLLM="${START_VLLM:-1}"
START_RETRIEVER="${START_RETRIEVER:-1}"
RUN_SCRIPT="${RUN_SCRIPT:-scripts/run_maps_all.sh}"
WAIT_TIMEOUT_SEC="${WAIT_TIMEOUT_SEC:-900}"
LOG_DIR="${LOG_DIR:-logs/pipeline}"

usage() {
  cat <<EOF
Usage: bash scripts/pipeline.sh [OPTIONS]

Boot the vLLM and retrieval servers, wait until they are ready, then run
RUN_SCRIPT. CLI flags override the environment variables, which override the
defaults.

vLLM server:
  --model NAME           Served model                (env: VLLM_MODEL, default: Qwen3-32B)
  --port PORT            vLLM port                  (env: VLLM_PORT, default: 9101)
  --api-key KEY          vLLM API key               (env: VLLM_API_KEY, default: TEST)
  --gpu-ids IDS          CUDA_VISIBLE_DEVICES        (env: VLLM_GPU_IDS, default: 2,3,4,5)
  --tp N                 Tensor parallel size        (env: VLLM_TP, default: 4)
  --dp N                 Data parallel size          (env: VLLM_DP, default: 1)
  --gpu-mem-util F       GPU memory utilization      (env: VLLM_GPU_MEM_UTIL, default: 0.7)
  --vllm-extra-args S    Extra flags appended to the vllm serve command
                                                     (env: VLLM_EXTRA_ARGS, default: empty)
  --no-vllm              Do not start the vLLM server

Retrieval server:
  --retriever-port PORT  Retrieval server port       (env: RETRIEVER_PORT, default: 9100)
  --retriever-gpu-ids IDS GPU ids for the retriever  (env: RETRIEVER_GPU_ID, default: 0,1)
  --retriever-num N      Number of retriever workers (env: RETRIEVER_NUM, default: 1)
  --retriever-config P   Retriever config YAML       (env: RETRIEVER_CONFIG_FILE,
                                                      default: retrieval_server/retriever_config.yaml)
  --no-retriever         Do not start the retrieval server

Pipeline:
  --run-script PATH      Experiment script to run after the servers are up
                                                     (env: RUN_SCRIPT, default: scripts/run_maps_all.sh)
  --wait-timeout SEC     Max seconds to wait per server (env: WAIT_TIMEOUT_SEC, default: 900)
  --log-dir DIR          Directory for server logs   (env: LOG_DIR, default: logs/pipeline)
  -h, --help             Show this help

Examples:
  bash scripts/pipeline.sh
  bash scripts/pipeline.sh --model Qwen3.6-35B-A3B --port 9101 --gpu-ids "4,5,6,7"
  RUN_SCRIPT=scripts/run_maps_all.sh bash scripts/pipeline.sh --retriever-port 9110
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    # --- vLLM ---
    --model) shift; VLLM_MODEL="${1:?--model requires a value}"; shift ;;
    --port) shift; VLLM_PORT="${1:?--port requires a value}"; shift ;;
    --api-key) shift; VLLM_API_KEY="${1:?--api-key requires a value}"; shift ;;
    --gpu-ids) shift; VLLM_GPU_IDS="${1:?--gpu-ids requires a value}"; shift ;;
    --tp) shift; VLLM_TP="${1:?--tp requires a value}"; shift ;;
    --dp) shift; VLLM_DP="${1:?--dp requires a value}"; shift ;;
    --gpu-mem-util) shift; VLLM_GPU_MEM_UTIL="${1:?--gpu-mem-util requires a value}"; shift ;;
    --vllm-extra-args) shift; VLLM_EXTRA_ARGS="${1:?--vllm-extra-args requires a value}"; shift ;;
    --no-vllm) START_VLLM=0; shift ;;
    # --- retrieval ---
    --retriever-port) shift; RETRIEVER_PORT="${1:?--retriever-port requires a value}"; shift ;;
    --retriever-gpu-ids) shift; RETRIEVER_GPU_ID="${1:?--retriever-gpu-ids requires a value}"; shift ;;
    --retriever-num) shift; RETRIEVER_NUM="${1:?--retriever-num requires a value}"; shift ;;
    --retriever-config) shift; RETRIEVER_CONFIG_FILE="${1:?--retriever-config requires a value}"; shift ;;
    --no-retriever) START_RETRIEVER=0; shift ;;
    # --- pipeline ---
    --run-script) shift; RUN_SCRIPT="${1:?--run-script requires a value}"; shift ;;
    --wait-timeout) shift; WAIT_TIMEOUT_SEC="${1:?--wait-timeout requires a value}"; shift ;;
    --log-dir) shift; LOG_DIR="${1:?--log-dir requires a value}"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[Pipeline] Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

mkdir -p "${LOG_DIR}"
TS="$(date +%Y%m%d_%H%M%S)"

PIDS=()

start_bg() {
  local name="$1"
  local cmd="$2"
  local log_file="${LOG_DIR}/${TS}_${name}.log"

  echo "[Pipeline] Starting ${name} ..."
  echo "[Pipeline] Command: ${cmd}"
  bash -lc "${cmd}" >"${log_file}" 2>&1 &
  local pid=$!
  PIDS+=("${pid}")
  echo "[Pipeline] ${name} pid=${pid}, log=${log_file}"
}

port_open() {
  local host="$1"
  local port="$2"
  (echo >"/dev/tcp/${host}/${port}") >/dev/null 2>&1
}

wait_port() {
  local name="$1"
  local host="$2"
  local port="$3"
  local timeout_sec="$4"
  local start_ts now elapsed

  start_ts=$(date +%s)
  while true; do
    if port_open "${host}" "${port}"; then
      echo "[Pipeline] ${name} is ready at ${host}:${port}"
      return 0
    fi

    now=$(date +%s)
    elapsed=$((now - start_ts))
    if [[ ${elapsed} -ge ${timeout_sec} ]]; then
      echo "[Pipeline] ERROR: ${name} not ready after ${timeout_sec}s (${host}:${port})" >&2
      return 1
    fi

    sleep 2
  done
}

cleanup() {
  local code=$?
  if [[ ${#PIDS[@]} -gt 0 ]]; then
    echo "[Pipeline] Cleaning up server processes ..."
    for pid in "${PIDS[@]}"; do
      if kill -0 "${pid}" >/dev/null 2>&1; then
        kill "${pid}" >/dev/null 2>&1 || true
      fi
    done
    # Give processes a moment to exit gracefully.
    sleep 2
    for pid in "${PIDS[@]}"; do
      if kill -0 "${pid}" >/dev/null 2>&1; then
        kill -9 "${pid}" >/dev/null 2>&1 || true
      fi
    done
  fi
  if [[ ${code} -eq 0 ]]; then
    echo "[Pipeline] Finished successfully."
  else
    echo "[Pipeline] Exited with error code ${code}." >&2
  fi
}
trap cleanup EXIT INT TERM

echo "[Pipeline] Repo root: ${REPO_ROOT}"
echo "[Pipeline] Run script: ${RUN_SCRIPT}"

# --- Retrieval server: reuse scripts/retrieval_server.sh -------------------
if [[ "${START_RETRIEVER}" == "1" ]]; then
  # Export so the reused launcher picks up the resolved settings.
  export RETRIEVER_PORT RETRIEVER_GPU_ID RETRIEVER_NUM RETRIEVER_CONFIG_FILE PYTHON_BIN
  start_bg "retrieval_server" "bash scripts/retrieval_server.sh"
fi

# --- vLLM server ----------------------------------------------------------
if [[ "${START_VLLM}" == "1" ]]; then
  vllm_cmd="CUDA_VISIBLE_DEVICES='${VLLM_GPU_IDS}' vllm serve '${VLLM_MODEL}'"
  vllm_cmd="${vllm_cmd} --port ${VLLM_PORT} -tp ${VLLM_TP} -dp ${VLLM_DP}"
  vllm_cmd="${vllm_cmd} --gpu-memory-utilization ${VLLM_GPU_MEM_UTIL}"
  vllm_cmd="${vllm_cmd} --api-key '${VLLM_API_KEY}'"
  if [[ -n "${VLLM_EXTRA_ARGS}" ]]; then
    vllm_cmd="${vllm_cmd} ${VLLM_EXTRA_ARGS}"
  fi
  start_bg "vllm_server" "${vllm_cmd}"
fi

echo "[Pipeline] Waiting for services to be ready ..."
if [[ "${START_RETRIEVER}" == "1" ]]; then
  wait_port "retrieval_server" "127.0.0.1" "${RETRIEVER_PORT}" "${WAIT_TIMEOUT_SEC}"
fi
if [[ "${START_VLLM}" == "1" ]]; then
  wait_port "vllm_server" "127.0.0.1" "${VLLM_PORT}" "${WAIT_TIMEOUT_SEC}"
fi

# Propagate the server endpoints to the experiment script (unless the caller
# already set them explicitly).
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:${VLLM_PORT}/}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-${VLLM_API_KEY}}"
export RETRIEVER_BASE_URL="${RETRIEVER_BASE_URL:-http://127.0.0.1:${RETRIEVER_PORT}}"

echo "[Pipeline] All services are up. Running ${RUN_SCRIPT} ..."
bash "${RUN_SCRIPT}"

echo "[Pipeline] Run script completed."
