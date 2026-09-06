#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_ROOT="${POCKETLLM_TMP_DIR:-$REPO_ROOT/.tmp}"
mkdir -p "$TMP_ROOT"

LOCK_FILE="${LOCK_FILE:-$TMP_ROOT/pocketllm_gguf_q2_layer_pp.lock}"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "another GGUF Q2 layer-PP benchmark is already running: $LOCK_FILE" >&2
  exit 1
fi

TORCHRUN="${TORCHRUN:-torchrun}"
MASTER_PORT="${MASTER_PORT:-29961}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
CKPT_PATH="${CKPT_PATH:-$REPO_ROOT/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/mnt/data1/modelscope/deepseek-ai/DeepSeek-V4-Flash}"
CONFIG="${CONFIG:-$REPO_ROOT/configs/config.json}"
INPUT_FILE="${INPUT_FILE:-$REPO_ROOT/tests/fixtures/smoke_input.txt}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8}"
PREFILL_CHUNK_TOKENS="${PREFILL_CHUNK_TOKENS:-256}"
LOG="${LOG:-$TMP_ROOT/gguf_q2_layer_pp.log}"

export DEEPSEEK_GGUF_ROUTES_NATIVE="${DEEPSEEK_GGUF_ROUTES_NATIVE:-1}"
export DEEPSEEK_GGUF_ROUTES_NATIVE_MAX_BATCH="${DEEPSEEK_GGUF_ROUTES_NATIVE_MAX_BATCH:-$PREFILL_CHUNK_TOKENS}"
export DEEPSEEK_GGUF_FUSED_EXPERT="${DEEPSEEK_GGUF_FUSED_EXPERT:-1}"
export DEEPSEEK_HC_PRE_CUDA="${DEEPSEEK_HC_PRE_CUDA:-1}"
export DEEPSEEK_HC_POST_CUDA="${DEEPSEEK_HC_POST_CUDA:-1}"

PYTHONPATH="$REPO_ROOT" "$TORCHRUN" \
  --master-port "$MASTER_PORT" \
  --nproc-per-node "$NPROC_PER_NODE" \
  --module src.cli.generate \
  --ckpt-path "$CKPT_PATH" \
  --ckpt-format gguf \
  --tokenizer-path "$TOKENIZER_PATH" \
  --config "$CONFIG" \
  --input-file "$INPUT_FILE" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --temperature 0 \
  --routed-experts-device cpu \
  --pd-mode scheduler \
  --pd-prefill-chunk-tokens "$PREFILL_CHUNK_TOKENS" \
  --partition-policy layer_pp_4gpu \
  2>&1 | tee "$LOG"
