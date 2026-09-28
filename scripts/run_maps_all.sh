#!/usr/bin/env bash
set -euo pipefail

DATA_ROOT="${DATA_ROOT:-$HOME/llm/datasets}"
RETRIEVER_BASE_URL="${RETRIEVER_BASE_URL:-http://127.0.0.1:9100}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OPENAI_BASE_URL="${OPENAI_BASE_URL:-http://127.0.0.1:9101/}"
OPENAI_API_KEY="${OPENAI_API_KEY:-TEST}"
MODEL="${MODEL:-Qwen3-32B}"
MODEL_PATH="${MODEL_PATH:-$HOME/llm/models/Qwen3-32B}"

TEMPERATURE="${TEMPERATURE:-0.8}"
TOP_P="${TOP_P:-0.9}"

NUM_SAMPLES="${NUM_SAMPLES:-1024}"
START=0
DATASETS=(
  2wikimultihopqa
  hotpotqa
  musique
  gpqa
  popqa
  nq
)

for dataset in "${DATASETS[@]}"; do
  input_file="${DATA_ROOT}/${dataset}/test.jsonl"

  if [[ ! -f "$input_file" ]]; then
    echo " Skip ${dataset}: input file not found -> $input_file"
    continue
  fi

  dataset_output_root="./outputs/maps/${MODEL}/${dataset}"
  mkdir -p "$dataset_output_root"

  max_index="$(find "$dataset_output_root" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | sed 's#.*/##' | grep -E '^[0-9]+$' | sort -n | tail -1 || true)"
  if [[ -z "$max_index" ]]; then
    next_index=0
  else
    next_index=$((max_index + 1))
  fi

  echo "=================================================="
  echo "Result Index for ${dataset}: ${next_index}"

  result_dir="${dataset_output_root}/${next_index}"
  result_file="${result_dir}/${dataset}.jsonl"

  echo "Processing Dataset: ${dataset}"
  echo "Output Dir: ${result_dir}"

  "$PYTHON_BIN" scripts/run_maps.py \
    --input_file "$input_file" \
    --batch_size 1024 \
    --retriever_base_url "$RETRIEVER_BASE_URL" \
    --retriever_top_k 5 \
    --intent_agent_openai_base_url "$OPENAI_BASE_URL" \
    --intent_agent_openai_api_key "$OPENAI_API_KEY" \
    --intent_agent_model "$MODEL" \
    --intent_agent_model_path "$MODEL_PATH" \
    --intent_agent_max_tokens 1024 \
    --intent_agent_temperature ${TEMPERATURE} \
    --intent_agent_top_p ${TOP_P} \
    --intent_agent_use_chat_template \
    --refine_agent_openai_base_url "$OPENAI_BASE_URL" \
    --refine_agent_openai_api_key "$OPENAI_API_KEY" \
    --refine_agent_model "$MODEL" \
    --refine_agent_model_path "$MODEL_PATH" \
    --refine_agent_max_tokens 1024 \
    --refine_agent_temperature ${TEMPERATURE} \
    --refine_agent_top_p ${TOP_P} \
    --refine_agent_use_chat_template \
    --query_agent_openai_base_url "$OPENAI_BASE_URL" \
    --query_agent_openai_api_key "$OPENAI_API_KEY" \
    --query_agent_model "$MODEL" \
    --query_agent_model_path "$MODEL_PATH" \
    --query_agent_max_tokens 1024 \
    --query_agent_temperature ${TEMPERATURE} \
    --query_agent_top_p ${TOP_P} \
    --query_agent_use_chat_template \
    --repetition_penalty 1.0 \
    --max_iterations 5 \
    --num_samples "$NUM_SAMPLES" \
    --start "$START"
    # --intent_agent_enable_thinking \
    # --refine_agent_enable_thinking \
    # --query_agent_enable_thinking \

  echo "Finished Dataset: ${dataset}"
  echo "Result File: ${result_file}"
done

echo "All datasets done."