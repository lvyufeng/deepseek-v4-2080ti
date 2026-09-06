#!/usr/bin/env bash
# A/B between current best int8 baseline (run_best_scheduler.sh) and FP4-active
# decode path (configs/config_fp4_active.json). Mirrors run_best_scheduler.sh
# launch flow but lets the caller switch CKPT_PATH + CONFIG between the two
# arms; expert_dtype=fp4 forces:
#   * cpu_backend: pinned FP4 arena for decode plus prepare-stage GPU FP4->int8
#     conversion back into CPU pinned int8 arena for prefill.
#   * cpu_backend native: routed_fp4_moe_forward_raw (AVX2 nibble + fmadd, no
#     tiled W2 layout).
#   * gpu_prefill_backend: prefill stages the prepared CPU int8 arena; decode
#     stages FP4 arena and dispatches moe_single_token_fp4_forward (PRMT unpack,
#     dp4a, e8m0 bitcast).
# All other DEEPSEEK_* flags match the int8 baseline so the A/B isolates the
# FP4 path.
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TMP_ROOT="${POCKETLLM_TMP_DIR:-$REPO_ROOT/.tmp}"
mkdir -p "$TMP_ROOT"
LOCK_FILE="${LOCK_FILE:-$TMP_ROOT/pocketllm_fp4_active_ab.lock}"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "another fp4 A/B benchmark is already running: $LOCK_FILE" >&2
  exit 1
fi

PYTHON="${PYTHON:-python}"
TORCHRUN="${TORCHRUN:-torchrun}"
INT8_CKPT_PATH="${INT8_CKPT_PATH:-/mnt/data1/modelscope/deepseek-ai/DeepSeek-V4-Flash-w8a8}"
INT8_CONFIG="${INT8_CONFIG:-$REPO_ROOT/configs/config_w8a8.json}"
FP4_CKPT_PATH="${FP4_CKPT_PATH:-/mnt/data1/modelscope/deepseek-ai/DeepSeek-V4-Flash}"
FP4_CONFIG="${FP4_CONFIG:-$REPO_ROOT/configs/config_fp4_active.json}"
SHORT_INPUT_FILE="${SHORT_INPUT_FILE:-$REPO_ROOT/tests/fixtures/smoke_input.txt}"
LONG_INPUT_FILE="${LONG_INPUT_FILE:-$TMP_ROOT/pocketllm_long_input_single.txt}"
SHORT_MAX_NEW_TOKENS="${SHORT_MAX_NEW_TOKENS:-8}"
LONG_MAX_NEW_TOKENS="${LONG_MAX_NEW_TOKENS:-64}"
LOG_DIR="${LOG_DIR:-$TMP_ROOT/pocketllm_fp4_active_ab}"
mkdir -p "$LOG_DIR"

if [[ ! -s "$LONG_INPUT_FILE" ]]; then
  TOK_CKPT="$INT8_CKPT_PATH" "$PYTHON" - <<'PY'
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
out_path = os.environ.get("LONG_INPUT_FILE", "/tmp/long_input_single.txt")
with open(out_path, "w") as f:
    f.write(text)
print(f"wrote {out_path} tokens={len(tok.encode(text))}", flush=True)
PY
fi

export_best_env() {
  export DEEPSEEK_PD_PHASE_AUTO_SELECT=1
  export DEEPSEEK_GPU_PREFILL_MOE=1
  export DEEPSEEK_GPU_PREFILL_MOE_GROUPED_GEMM=1
  export DEEPSEEK_GPU_PREFILL_MOE_PREFETCH_BEFORE_FFN=1
  export DEEPSEEK_GPU_PREFILL_MOE_MAX_CACHED_LAYERS="${DEEPSEEK_GPU_PREFILL_MOE_MAX_CACHED_LAYERS:-3}"
  export DEEPSEEK_GPU_PREFILL_MOE_ARENA="${DEEPSEEK_GPU_PREFILL_MOE_ARENA:-1}"
  export DEEPSEEK_GPU_MOE_DECODE_ACTIVE="${DEEPSEEK_GPU_MOE_DECODE_ACTIVE:-1}"
  export DEEPSEEK_GPU_MOE_CROSS_LAYER_PREFETCH="${DEEPSEEK_GPU_MOE_CROSS_LAYER_PREFETCH:-0}"
  export DEEPSEEK_GPU_MOE_CROSS_LAYER_PREFETCH_K="${DEEPSEEK_GPU_MOE_CROSS_LAYER_PREFETCH_K:-10}"
  export DEEPSEEK_GPU_MOE_CROSS_LAYER_PREFETCH_LOCAL_LIMIT="${DEEPSEEK_GPU_MOE_CROSS_LAYER_PREFETCH_LOCAL_LIMIT:-2}"
  export DEEPSEEK_INT8_IMPL=cuda_ext
  export DEEPSEEK_MOE_ASYNC_ALLREDUCE=1
  export DEEPSEEK_SHARED_EXPERT_INT8=1
  export DEEPSEEK_SHARED_EXPERT_PAIR_INT8_CUDA="${DEEPSEEK_SHARED_EXPERT_PAIR_INT8_CUDA:-1}"
  export DEEPSEEK_FLASHINFER_STYLE_ATTN_CUDA=1
  export DEEPSEEK_FUSED_C4_INDEXER_CUDA=1
  export DEEPSEEK_HC_PRE_CUDA=1
  export DEEPSEEK_CPU_DECODE_INLINE_THRESHOLD="${DEEPSEEK_CPU_DECODE_INLINE_THRESHOLD:-0}"
  export DEEPSEEK_CPU_TOPK_PERSISTENT="${DEEPSEEK_CPU_TOPK_PERSISTENT:-1}"
  export DEEPSEEK_PD_DECODE_OMP_THREADS="${DEEPSEEK_PD_DECODE_OMP_THREADS:-12}"
  export DEEPSEEK_FUSED_ATTN_PREFUSE="${DEEPSEEK_FUSED_ATTN_PREFUSE:-1}"
  for module in WQ_A WQ_B WKV WO_A WO_B INDEXER_WQ_B; do
    export "DEEPSEEK_PD_PREFILL_${module}_INT8=1"
    export "DEEPSEEK_PD_DECODE_${module}_INT8=1"
  done
  export DEEPSEEK_CPU_MOE_EXTERNAL_SERVER=0
  export DEEPSEEK_CPU_MOE_INPROC_SERVER=0
  export DEEPSEEK_CPU_MOE_SHARED_WEIGHTS=0
}

run_case() {
  local arm="$1"     # int8 | fp4
  local name="$2"    # short_short | short_long | long_short | long_long
  local input="$3"
  local max_new="$4"
  local port="$5"
  local log="$LOG_DIR/${arm}_${name}.log"
  local ckpt config
  if [[ "$arm" == "fp4" ]]; then
    ckpt="$FP4_CKPT_PATH"
    config="$FP4_CONFIG"
  else
    ckpt="$INT8_CKPT_PATH"
    config="$INT8_CONFIG"
  fi
  echo "=== arm=$arm $name (input=$input, max_new=$max_new) ==="
  echo "settings: config=$config ckpt=$ckpt cross=${DEEPSEEK_GPU_MOE_CROSS_LAYER_PREFETCH:-0} K=${DEEPSEEK_GPU_MOE_CROSS_LAYER_PREFETCH_K:-10} cache=${DEEPSEEK_GPU_PREFILL_MOE_MAX_CACHED_LAYERS:-3} local_limit=${DEEPSEEK_GPU_MOE_CROSS_LAYER_PREFETCH_LOCAL_LIMIT:-2} decode_active=${DEEPSEEK_GPU_MOE_DECODE_ACTIVE:-1} log=$log"
  rm -f "$log"
  (
    export_best_env
    PYTHONPATH="$REPO_ROOT" "$TORCHRUN" \
      --master-port "$port" \
      --nproc-per-node 4 \
      --module src.cli.generate \
      --ckpt-path "$ckpt" \
      --config "$config" \
      --input-file "$input" \
      --max-new-tokens "$max_new" \
      --temperature 0 \
      --routed-experts-device cpu \
      --pd-mode scheduler \
      > "$log" 2>&1
  )
  grep -E 'generate time:|prefill time:|^Completion:' "$log" || true
}

if [[ -n "${ARMS:-}" ]]; then
  echo "ARMS is not supported; use ARM=both|int8|fp4" >&2
  exit 2
fi
if [[ -n "${CASES:-}" ]]; then
  echo "CASES is not supported; use PD_CASE=all|short_short|short_long|long_short|long_long" >&2
  exit 2
fi

ARM="${ARM:-both}"
PD_CASE="${PD_CASE:-long_long}"

run_arm() {
  local arm="$1"
  case "$PD_CASE" in
    all)
      run_case "$arm" short_short "$SHORT_INPUT_FILE" "$SHORT_MAX_NEW_TOKENS" "${PORT_SHORT_SHORT:-30001}"
      run_case "$arm" short_long  "$SHORT_INPUT_FILE" "$LONG_MAX_NEW_TOKENS"  "${PORT_SHORT_LONG:-30002}"
      run_case "$arm" long_short  "$LONG_INPUT_FILE"  "$SHORT_MAX_NEW_TOKENS" "${PORT_LONG_SHORT:-30003}"
      run_case "$arm" long_long   "$LONG_INPUT_FILE"  "$LONG_MAX_NEW_TOKENS"  "${PORT_LONG_LONG:-30004}"
      ;;
    short_short) run_case "$arm" short_short "$SHORT_INPUT_FILE" "$SHORT_MAX_NEW_TOKENS" "${PORT_SHORT_SHORT:-30001}" ;;
    short_long)  run_case "$arm" short_long  "$SHORT_INPUT_FILE" "$LONG_MAX_NEW_TOKENS"  "${PORT_SHORT_LONG:-30002}" ;;
    long_short)  run_case "$arm" long_short  "$LONG_INPUT_FILE"  "$SHORT_MAX_NEW_TOKENS" "${PORT_LONG_SHORT:-30003}" ;;
    long_long)   run_case "$arm" long_long   "$LONG_INPUT_FILE"  "$LONG_MAX_NEW_TOKENS"  "${PORT_LONG_LONG:-30004}" ;;
    *)
      echo "unknown PD_CASE=$PD_CASE" >&2; exit 2 ;;
  esac
}

case "$ARM" in
  both) run_arm int8; run_arm fp4 ;;
  int8) run_arm int8 ;;
  fp4)  run_arm fp4 ;;
  *) echo "unknown ARM=$ARM (expected: both|int8|fp4)" >&2; exit 2 ;;
esac
