#!/usr/bin/env bash
# Launch the vLLM OpenAI-compatible server (Qwen3-8B) inside WSL.
set -e
cd "$(dirname "$0")"
exec ./vllm-venv/bin/vllm serve Qwen/Qwen3-8B \
  --host 0.0.0.0 --port 8000 \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.90 \
  --served-model-name Qwen3-8B
