#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
for argument in "$@"; do
  if [[ "$argument" == --help || "$argument" == -h ]]; then
    exec uv run --no-sync python -m src.benchmark.native_experiment --help
  fi
done
export PATH="$PROJECT_ROOT/.venv-vllm/bin:$PATH"
VLLM_PORT="${HRP_VLLM_PORT:-8010}"
MODEL_DIR="$(realpath -m "${HRP_MODEL_DIR:-$PROJECT_ROOT/models/Qwen3.5-9B}")"
GPU_IDS="${HRP_GPU_IDS:-0,1,2,3}"
IFS=',' read -r -a GPU_ID_LIST <<< "$GPU_IDS"
DATA_PARALLEL_SIZE="${#GPU_ID_LIST[@]}"
[[ "$DATA_PARALLEL_SIZE" -gt 0 ]] || { echo "HRP_GPU_IDS must not be empty" >&2; exit 2; }
OUTPUT="${1:?Usage: eval_native_harnesses.sh OUTPUT [native_experiment arguments]}"
shift
mkdir -p "$OUTPUT"
uv run --no-sync python -m src.benchmark.native_experiment --output "$OUTPUT" "$@" --check-images
for argument in "$@"; do
  [[ "$argument" != --check-images ]] || exit 0
done
SERVER_PID=""
cleanup() {
  if [[ -n "$SERVER_PID" && "${HRP_KEEP_SERVER:-0}" != 1 ]]; then
    kill -- "-$SERVER_PID" 2>/dev/null || true
    for _ in {1..20}; do
      kill -0 -- "-$SERVER_PID" 2>/dev/null || break
      sleep 0.5
    done
    kill -KILL -- "-$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
if [[ "${HRP_START_SERVER:-1}" == 1 ]]; then
  if curl --noproxy '*' -fsS "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
    echo "Reusing model service on port ${VLLM_PORT}; launch flags and GPUs are checked against the manifest."
  else
    CUDA_VISIBLE_DEVICES="$GPU_IDS" setsid .venv-vllm/bin/python -m vllm.entrypoints.openai.api_server \
      --model "$MODEL_DIR" --served-model-name qwen3.5-9b \
      --host 127.0.0.1 --port "$VLLM_PORT" --tensor-parallel-size 1 --data-parallel-size "$DATA_PARALLEL_SIZE" \
      --dtype bfloat16 --max-model-len 65536 --gpu-memory-utilization 0.94 \
      --max-num-seqs 12 --max-num-batched-tokens 16384 --enable-prefix-caching \
      --enable-chunked-prefill --watermark 0.05 --numa-bind --language-model-only \
      --override-generation-config '{"top_p":0.95,"top_k":20,"min_p":0.0,"repetition_penalty":1.0}' \
      --default-chat-template-kwargs '{"enable_thinking":true}' \
      --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
      >"$OUTPUT/vllm.log" 2>&1 &
    SERVER_PID=$!
    echo "$SERVER_PID" >"$OUTPUT/vllm.pid"
    for _ in $(seq 1 180); do
      if curl --noproxy '*' -fsS "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then break; fi
      if ! kill -0 "$SERVER_PID" 2>/dev/null; then tail -n 60 "$OUTPUT/vllm.log"; exit 1; fi
      sleep 5
    done
    curl --noproxy '*' -fsS "http://127.0.0.1:${VLLM_PORT}/v1/models" >"$OUTPUT/served_models.json"
  fi
fi
uv run --no-sync python -u -m src.benchmark.native_experiment --output "$OUTPUT" \
  --target-api-base "http://127.0.0.1:${VLLM_PORT}/v1" \
  --gpu-ids "${GPU_ID_LIST[@]}" "$@"
