#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_ROOT="${POCKETLLM_TMP_DIR:-$REPO_ROOT/.tmp}"
mkdir -p "$TMP_ROOT"

LOCK_FILE="${LOCK_FILE:-$TMP_ROOT/pocketllm_fp4_resident_best.lock}"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "another FP4 resident benchmark is already running: $LOCK_FILE" >&2
  exit 1
fi

TORCHRUN="${TORCHRUN:-torchrun}"
MASTER_PORT="${MASTER_PORT:-29943}"
PORT="${PORT:-8013}"
HOST="${HOST:-127.0.0.1}"
CKPT_PATH="${CKPT_PATH:-/mnt/data1/modelscope/deepseek-ai/DeepSeek-V4-Flash}"
CONFIG="${CONFIG:-$REPO_ROOT/configs/config_fp4_active.json}"
MODEL_ID="${MODEL_ID:-deepseek-v4-flash-fp4-resident-best}"
ROUTED_EXPERTS_DEVICE="${ROUTED_EXPERTS_DEVICE:-cpu}"
PD_MODE="${PD_MODE:-scheduler}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
FP4_CASE="${FP4_CASE:-long_long}"
FP4_REPEAT="${FP4_REPEAT:-3}"
SHORT_INPUT_FILE="${SHORT_INPUT_FILE:-$REPO_ROOT/tests/fixtures/smoke_input.txt}"
LONG_INPUT_FILE="${LONG_INPUT_FILE:-$TMP_ROOT/pocketllm_long_input_single.txt}"
SHORT_MAX_NEW_TOKENS="${SHORT_MAX_NEW_TOKENS:-8}"
LONG_MAX_NEW_TOKENS="${LONG_MAX_NEW_TOKENS:-64}"
LOG="${LOG:-$TMP_ROOT/fp4_resident_best_server.log}"
OUT="${OUT:-$TMP_ROOT/fp4_resident_best_results.json}"

if [[ ! -e "$CKPT_PATH" ]]; then
  echo "checkpoint not found: $CKPT_PATH" >&2
  exit 1
fi

if [[ ! -s "$LONG_INPUT_FILE" ]]; then
  TOK_CKPT="$CKPT_PATH" LONG_INPUT_FILE="$LONG_INPUT_FILE" python - <<'PY'
import os
from transformers import AutoTokenizer
ckpt = os.environ["TOK_CKPT"]
tok = AutoTokenizer.from_pretrained(ckpt)
base = (
    "请阅读下面这段重复的技术背景，并在最后用中文总结它的核心观点。\n"
    "DeepSeek V4 Flash standalone inference is being optimized on four RTX 2080Ti GPUs. "
    "Routed MoE experts remain on CPU, exact top-6 routing must be preserved, and large native operators are preferred over Python hot paths. "
    "The validation must include both short and long sequence correctness and performance, with full-network benchmarks run serially.\n"
)
parts = []
token_count = 0
while token_count < 2048:
    parts.append(base.rstrip("\n"))
    token_count = len(tok.encode("\n".join(parts)))
text = "\n".join(parts) + "\n请用三句话总结以上内容。"
out_path = os.environ["LONG_INPUT_FILE"]
with open(out_path, "w") as f:
    f.write(text)
print(f"wrote {out_path} tokens={len(tok.encode(text))}", flush=True)
PY
fi

export MASTER_PORT PORT HOST CKPT_PATH CONFIG MODEL_ID ROUTED_EXPERTS_DEVICE PD_MODE MAX_MODEL_LEN
export DEEPSEEK_PD_PHASE_AUTO_SELECT="${DEEPSEEK_PD_PHASE_AUTO_SELECT:-1}"
export DEEPSEEK_GPU_PREFILL_MOE="${DEEPSEEK_GPU_PREFILL_MOE:-1}"
export DEEPSEEK_GPU_PREFILL_MOE_GROUPED_GEMM="${DEEPSEEK_GPU_PREFILL_MOE_GROUPED_GEMM:-1}"
export DEEPSEEK_GPU_PREFILL_MOE_PREFETCH_BEFORE_FFN="${DEEPSEEK_GPU_PREFILL_MOE_PREFETCH_BEFORE_FFN:-1}"
export DEEPSEEK_GPU_PREFILL_MOE_MAX_CACHED_LAYERS="${DEEPSEEK_GPU_PREFILL_MOE_MAX_CACHED_LAYERS:-3}"
export DEEPSEEK_GPU_PREFILL_MOE_ARENA="${DEEPSEEK_GPU_PREFILL_MOE_ARENA:-1}"
export DEEPSEEK_GPU_PREFILL_MOE_BUCKETED_GEMM="${DEEPSEEK_GPU_PREFILL_MOE_BUCKETED_GEMM:-1}"
export DEEPSEEK_GPU_PREFILL_MOE_BUCKET_EXPERTS="${DEEPSEEK_GPU_PREFILL_MOE_BUCKET_EXPERTS:-16}"
export DEEPSEEK_GPU_PREFILL_MOE_CHUNK_TOKENS="${DEEPSEEK_GPU_PREFILL_MOE_CHUNK_TOKENS:-2048}"
export DEEPSEEK_GPU_MOE_DECODE_ACTIVE="${DEEPSEEK_GPU_MOE_DECODE_ACTIVE:-1}"
export DEEPSEEK_GPU_MOE_CROSS_LAYER_PREFETCH="${DEEPSEEK_GPU_MOE_CROSS_LAYER_PREFETCH:-0}"
export DEEPSEEK_GPU_MOE_CROSS_LAYER_PREFETCH_K="${DEEPSEEK_GPU_MOE_CROSS_LAYER_PREFETCH_K:-10}"
export DEEPSEEK_GPU_MOE_CROSS_LAYER_PREFETCH_LOCAL_LIMIT="${DEEPSEEK_GPU_MOE_CROSS_LAYER_PREFETCH_LOCAL_LIMIT:-2}"
export DEEPSEEK_INT8_IMPL="${DEEPSEEK_INT8_IMPL:-cuda_ext}"
export DEEPSEEK_MOE_ASYNC_ALLREDUCE="${DEEPSEEK_MOE_ASYNC_ALLREDUCE:-1}"
export DEEPSEEK_MOE_FUSED_FINALIZE="${DEEPSEEK_MOE_FUSED_FINALIZE:-1}"
export NCCL_PROTO="${NCCL_PROTO:-LL128}"
export DEEPSEEK_SHARED_EXPERT_INT8="${DEEPSEEK_SHARED_EXPERT_INT8:-1}"
export DEEPSEEK_SHARED_EXPERT_PAIR_INT8_CUDA="${DEEPSEEK_SHARED_EXPERT_PAIR_INT8_CUDA:-1}"
export DEEPSEEK_PD_SHARED_EXPERT_FP16="${DEEPSEEK_PD_SHARED_EXPERT_FP16:-1}"
export DEEPSEEK_FLASHINFER_STYLE_ATTN_CUDA="${DEEPSEEK_FLASHINFER_STYLE_ATTN_CUDA:-1}"
export DEEPSEEK_PREFILL_SPARSE_ATTN_HEADPAIR_CUDA="${DEEPSEEK_PREFILL_SPARSE_ATTN_HEADPAIR_CUDA:-1}"
export DEEPSEEK_FUSED_C4_INDEXER_CUDA="${DEEPSEEK_FUSED_C4_INDEXER_CUDA:-1}"
export DEEPSEEK_C4_TOPK_TILE_MERGE_CUDA="${DEEPSEEK_C4_TOPK_TILE_MERGE_CUDA:-1}"
export DEEPSEEK_HC_PRE_CUDA="${DEEPSEEK_HC_PRE_CUDA:-1}"
export DEEPSEEK_HC_POST_CUDA="${DEEPSEEK_HC_POST_CUDA:-1}"
export DEEPSEEK_CPU_DECODE_INLINE_THRESHOLD="${DEEPSEEK_CPU_DECODE_INLINE_THRESHOLD:-0}"
export DEEPSEEK_CPU_TOPK_PERSISTENT="${DEEPSEEK_CPU_TOPK_PERSISTENT:-1}"
export DEEPSEEK_PD_DECODE_OMP_THREADS="${DEEPSEEK_PD_DECODE_OMP_THREADS:-12}"
export DEEPSEEK_FUSED_ATTN_PREFUSE="${DEEPSEEK_FUSED_ATTN_PREFUSE:-1}"
export DEEPSEEK_CPU_MOE_EXTERNAL_SERVER="${DEEPSEEK_CPU_MOE_EXTERNAL_SERVER:-0}"
export DEEPSEEK_CPU_MOE_INPROC_SERVER="${DEEPSEEK_CPU_MOE_INPROC_SERVER:-0}"
export DEEPSEEK_CPU_MOE_SHARED_WEIGHTS="${DEEPSEEK_CPU_MOE_SHARED_WEIGHTS:-0}"
for module in WQ_A WQ_B WKV WO_B INDEXER_WQ_B; do
  prefill_var="DEEPSEEK_PD_PREFILL_${module}_INT8"
  decode_var="DEEPSEEK_PD_DECODE_${module}_INT8"
  export "${prefill_var}=${!prefill_var:-1}"
  export "${decode_var}=${!decode_var:-1}"
done
export DEEPSEEK_PD_PREFILL_WO_A_INT8="${DEEPSEEK_PD_PREFILL_WO_A_INT8:-0}"
export DEEPSEEK_PD_DECODE_WO_A_INT8="${DEEPSEEK_PD_DECODE_WO_A_INT8:-1}"
export DEEPSEEK_WO_A_FP16="${DEEPSEEK_WO_A_FP16:-1}"

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
  --nproc-per-node 4 \
  --module src.server.openai \
  --host "$HOST" \
  --port "$PORT" \
  --ckpt-path "$CKPT_PATH" \
  --ckpt-format safetensors \
  --config "$CONFIG" \
  --model "$MODEL_ID" \
  --routed-experts-device "$ROUTED_EXPERTS_DEVICE" \
  --pd-mode "$PD_MODE" \
  --partition-policy legacy \
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
  --case "$FP4_CASE" \
  --repeat "$FP4_REPEAT" \
  --warmup "${BENCH_WARMUP:-none}" \
  --output "$OUT"
