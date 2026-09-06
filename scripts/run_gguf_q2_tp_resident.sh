#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_ROOT="${POCKETLLM_TMP_DIR:-$REPO_ROOT/.tmp}"
mkdir -p "$TMP_ROOT"

LOCK_FILE="${LOCK_FILE:-$TMP_ROOT/pocketllm_gguf_q2_tp_resident.lock}"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "another GGUF Q2 TP resident benchmark is already running: $LOCK_FILE" >&2
  exit 1
fi

TORCHRUN="${TORCHRUN:-torchrun}"
MASTER_PORT="${MASTER_PORT:-29975}"
PORT="${PORT:-8071}"
HOST="${HOST:-127.0.0.1}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
if [[ -z "${PARTITION_POLICY+x}" ]]; then
  if [[ "$NPROC_PER_NODE" == "1" ]]; then
    PARTITION_POLICY="legacy"
  else
    PARTITION_POLICY="baseline_4gpu"
  fi
fi
CKPT_PATH="${CKPT_PATH:-$REPO_ROOT/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/mnt/data1/modelscope/deepseek-ai/DeepSeek-V4-Flash}"
CONFIG="${CONFIG:-$REPO_ROOT/configs/config.json}"
MODEL_ID="${MODEL_ID:-deepseek-v4-flash-gguf-q2-tp}"
ROUTED_EXPERTS_DEVICE="${ROUTED_EXPERTS_DEVICE:-cpu}"
PD_MODE="${PD_MODE:-scheduler}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
CASE="${CASE:-long_long}"
REPEAT="${REPEAT:-1}"
SHORT_INPUT_FILE="${SHORT_INPUT_FILE:-$REPO_ROOT/tests/fixtures/smoke_input.txt}"
LONG_INPUT_FILE="${LONG_INPUT_FILE:-$TMP_ROOT/gguf_q2_long_input_single.txt}"
SHORT_MAX_NEW_TOKENS="${SHORT_MAX_NEW_TOKENS:-8}"
LONG_MAX_NEW_TOKENS="${LONG_MAX_NEW_TOKENS:-64}"
LOG="${LOG:-$TMP_ROOT/gguf_q2_tp_resident_server.log}"
OUT="${OUT:-$TMP_ROOT/gguf_q2_tp_resident_results.json}"

if [[ ! -e "$CKPT_PATH" ]]; then
  echo "checkpoint not found: $CKPT_PATH" >&2
  exit 1
fi

export SERVING_PROFILE="${SERVING_PROFILE:-safe1}"
export DEEPSEEK_SERVING_PREFILL_CHUNK_TOKENS="${DEEPSEEK_SERVING_PREFILL_CHUNK_TOKENS:-4096}"
export DEEPSEEK_GGUF_ROUTES_NATIVE="${DEEPSEEK_GGUF_ROUTES_NATIVE:-1}"
export DEEPSEEK_GGUF_ROUTES_NATIVE_MAX_BATCH="${DEEPSEEK_GGUF_ROUTES_NATIVE_MAX_BATCH:-$DEEPSEEK_SERVING_PREFILL_CHUNK_TOKENS}"
if [[ "$NPROC_PER_NODE" == "1" ]]; then
  export DEEPSEEK_GGUF_GPU_PREFILL_MOE="${DEEPSEEK_GGUF_GPU_PREFILL_MOE:-0}"
  export DEEPSEEK_GGUF_GPU_GROUPED_MOE="${DEEPSEEK_GGUF_GPU_GROUPED_MOE:-0}"
else
  export DEEPSEEK_GGUF_GPU_PREFILL_MOE="${DEEPSEEK_GGUF_GPU_PREFILL_MOE:-1}"
  export DEEPSEEK_GGUF_GPU_GROUPED_MOE="${DEEPSEEK_GGUF_GPU_GROUPED_MOE:-1}"
fi
export DEEPSEEK_GGUF_FUSED_EXPERT="${DEEPSEEK_GGUF_FUSED_EXPERT:-1}"
export DEEPSEEK_GGUF_GPU_FUSED_EXPERT="${DEEPSEEK_GGUF_GPU_FUSED_EXPERT:-1}"
export DEEPSEEK_GGUF_GPU_PREFILL_MOE_KEEP_STAGED="${DEEPSEEK_GGUF_GPU_PREFILL_MOE_KEEP_STAGED:-0}"
export DEEPSEEK_GGUF_GPU_DECODE_ACTIVE_EXPERT="${DEEPSEEK_GGUF_GPU_DECODE_ACTIVE_EXPERT:-1}"
export DEEPSEEK_GGUF_GPU_DECODE_GROUPED="${DEEPSEEK_GGUF_GPU_DECODE_GROUPED:-1}"
export DEEPSEEK_GGUF_GPU_DECODE_SINGLE_TOKEN="${DEEPSEEK_GGUF_GPU_DECODE_SINGLE_TOKEN:-1}"
export DEEPSEEK_GGUF_GPU_DECODE_SLOT_CACHE="${DEEPSEEK_GGUF_GPU_DECODE_SLOT_CACHE:-1}"
export DEEPSEEK_GGUF_GPU_DECODE_SLOT_CACHE_SIZE="${DEEPSEEK_GGUF_GPU_DECODE_SLOT_CACHE_SIZE:-16}"
export DEEPSEEK_MOE_ASYNC_ALLREDUCE="${DEEPSEEK_MOE_ASYNC_ALLREDUCE:-1}"
export DEEPSEEK_MOE_FUSED_FINALIZE="${DEEPSEEK_MOE_FUSED_FINALIZE:-1}"
export DEEPSEEK_GGUF_IQ2_XXS_W13_DP4A="${DEEPSEEK_GGUF_IQ2_XXS_W13_DP4A:-1}"
export DEEPSEEK_GGUF_IQ2_XXS_W13_EXPERT_TILE="${DEEPSEEK_GGUF_IQ2_XXS_W13_EXPERT_TILE:-1}"
export DEEPSEEK_GGUF_Q2K_W2_DP4A="${DEEPSEEK_GGUF_Q2K_W2_DP4A:-1}"
export DEEPSEEK_GGUF_Q2K_W2_SUBWARP="${DEEPSEEK_GGUF_Q2K_W2_SUBWARP:-1}"
export DEEPSEEK_GGUF_Q2K_W2_EXPERT_TILE="${DEEPSEEK_GGUF_Q2K_W2_EXPERT_TILE:-1}"
export DEEPSEEK_HC_PRE_CUDA="${DEEPSEEK_HC_PRE_CUDA:-1}"
export DEEPSEEK_HC_POST_CUDA="${DEEPSEEK_HC_POST_CUDA:-1}"
export NCCL_PROTO="${NCCL_PROTO:-LL128}"

cleanup() {
  if [[ -n "${server_pid:-}" ]]; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

rm -f "$LOG" "$OUT"

PYTHONPATH="$REPO_ROOT" "$TORCHRUN" \
  --master-port "$MASTER_PORT" \
  --nproc-per-node "$NPROC_PER_NODE" \
  --module src.server.openai \
  --host "$HOST" \
  --port "$PORT" \
  --ckpt-path "$CKPT_PATH" \
  --ckpt-format gguf \
  --tokenizer-path "$TOKENIZER_PATH" \
  --config "$CONFIG" \
  --model "$MODEL_ID" \
  --routed-experts-device "$ROUTED_EXPERTS_DEVICE" \
  --pd-mode "$PD_MODE" \
  --partition-policy "$PARTITION_POLICY" \
  --max-model-len "$MAX_MODEL_LEN" \
  > "$LOG" 2>&1 &
server_pid=$!

python "$REPO_ROOT/scripts/bench_openai_resident.py" \
  --base-url "http://$HOST:$PORT" \
  --model "$MODEL_ID" \
  --short-input "$SHORT_INPUT_FILE" \
  --long-input "$LONG_INPUT_FILE" \
  --short-max-tokens "$SHORT_MAX_NEW_TOKENS" \
  --long-max-tokens "$LONG_MAX_NEW_TOKENS" \
  --case "$CASE" \
  --repeat "$REPEAT" \
  --warmup "${BENCH_WARMUP:-none}" \
  --output "$OUT"
