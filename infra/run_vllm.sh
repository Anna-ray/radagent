#!/usr/bin/env bash
# infra/run_vllm.sh
# -----------------
# Boot a vLLM server on the MI300X for Llama 3.2 Vision.
#
# Defaults to 11B. MODEL_SIZE=90b switches to 90B (slower, 50 GB).
#
# Usage:
#   export HF_TOKEN=hf_...
#   source /workspace/env.sh
#   bash run_vllm.sh                    # 11B
#   MODEL_SIZE=90b bash run_vllm.sh     # 90B
#
# The server listens on :8000 (OpenAI-compatible API).

set -euo pipefail

MODEL_SIZE="${MODEL_SIZE:-11b}"
PORT="${PORT:-8000}"
GPU_MEM_FRAC="${GPU_MEM_FRAC:-0.85}"   # leave room for the specialist + RAG embedder

case "$MODEL_SIZE" in
    11b)
        MODEL_ID="meta-llama/Llama-3.2-11B-Vision-Instruct"
        MAX_MODEL_LEN=8192
        ;;
    90b)
        MODEL_ID="meta-llama/Llama-3.2-90B-Vision-Instruct"
        MAX_MODEL_LEN=4096
        ;;
    *)
        echo "ERROR: MODEL_SIZE must be 11b or 90b, got '$MODEL_SIZE'"
        exit 1
        ;;
esac

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "ERROR: HF_TOKEN not set. export HF_TOKEN=hf_... first."
    exit 1
fi

echo "==[vllm] launching =="
echo "  model: $MODEL_ID"
echo "  port:  $PORT"
echo "  max_model_len: $MAX_MODEL_LEN"
echo "  gpu_memory_utilization: $GPU_MEM_FRAC"
echo

# shellcheck disable=SC1091
source /workspace/venv/bin/activate

# --gpu-memory-utilization keeps headroom for our specialist + RAG embedder
# --enforce-eager skips graph compilation (faster startup, ~5% slower per step)
# --limit-mm-per-prompt image=1 since we send one CXR per request
exec vllm serve "$MODEL_ID" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --dtype auto \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEM_FRAC" \
    --enforce-eager \
    --limit-mm-per-prompt image=1 \
    --trust-remote-code