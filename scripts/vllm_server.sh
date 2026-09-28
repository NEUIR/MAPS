# Qwen3-32B
CUDA_VISIBLE_DEVICES="2,3,4,5" vllm serve Qwen3-32B --port 9101 -tp 4 -dp 1 --gpu-memory-utilization 0.7 --api-key "TEST"

# Qwen 3.6
CUDA_VISIBLE_DEVICES="4,5,6,7" vllm serve  Qwen3.6-35B-A3B --port 9101 -tp 4 -dp 1 --gpu-memory-utilization 0.85 --api-key "TEST"
