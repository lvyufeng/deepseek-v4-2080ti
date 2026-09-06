#!/usr/bin/env bash
# A/B: inline_threshold=1 vs 0; persistent worker vs none.
set -eo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="$REPO_ROOT"
TMP_ROOT="${POCKETLLM_TMP_DIR:-$REPO_ROOT/.tmp}"
mkdir -p "$TMP_ROOT"
TORCHRUN="${TORCHRUN:-torchrun}"
DEFAULT_CKPT_PATH="$REPO_ROOT/checkpoints/DeepSeek-V4-Flash-w8a8"
CKPT="${CKPT_PATH:-$DEFAULT_CKPT_PATH}"
CFG="$REPO_ROOT/configs/config_w8a8.json"

if [[ ! -e "$CKPT" ]]; then
  echo "checkpoint not found: $CKPT" >&2
  echo "Set CKPT_PATH or place the checkpoint under $DEFAULT_CKPT_PATH" >&2
  exit 1
fi
LONG="$TMP_ROOT/pocketllm_long_input_single.txt"
LOGDIR="${LOGDIR:-$TMP_ROOT/pocketllm_inline_ab}"
rm -rf "$LOGDIR"
mkdir -p "$LOGDIR"

common_env=(
  DEEPSEEK_PD_PHASE_AUTO_SELECT=1
  DEEPSEEK_GPU_PREFILL_MOE=1
  DEEPSEEK_GPU_PREFILL_MOE_GROUPED_GEMM=1
  DEEPSEEK_GPU_PREFILL_MOE_PREFETCH_BEFORE_FFN=1
  DEEPSEEK_GPU_PREFILL_MOE_MAX_CACHED_LAYERS=3
  DEEPSEEK_GPU_PREFILL_MOE_ARENA=1
  DEEPSEEK_INT8_IMPL=cuda_ext
  DEEPSEEK_MOE_ASYNC_ALLREDUCE=1
  DEEPSEEK_SHARED_EXPERT_INT8=1
  DEEPSEEK_FLASHINFER_STYLE_ATTN_CUDA=1
  DEEPSEEK_FUSED_C4_INDEXER_CUDA=1
  DEEPSEEK_HC_PRE_CUDA=1
  DEEPSEEK_PD_DECODE_OMP_THREADS=8
  DEEPSEEK_CPU_TOPK_PERSISTENT=1
  DEEPSEEK_PD_PREFILL_WQ_A_INT8=1
  DEEPSEEK_PD_PREFILL_WQ_B_INT8=1
  DEEPSEEK_PD_PREFILL_WKV_INT8=1
  DEEPSEEK_PD_PREFILL_WO_A_INT8=1
  DEEPSEEK_PD_PREFILL_WO_B_INT8=1
  DEEPSEEK_PD_PREFILL_INDEXER_WQ_B_INT8=1
  DEEPSEEK_PD_DECODE_WQ_A_INT8=1
  DEEPSEEK_PD_DECODE_WQ_B_INT8=1
  DEEPSEEK_PD_DECODE_WKV_INT8=1
  DEEPSEEK_PD_DECODE_WO_A_INT8=1
  DEEPSEEK_PD_DECODE_WO_B_INT8=1
  DEEPSEEK_PD_DECODE_INDEXER_WQ_B_INT8=1
)

run_case() {
  local name="$1"
  local port="$2"
  shift 2
  echo "=== $name ==="
  env "${common_env[@]}" "$@" PYTHONPATH="$REPO_ROOT" "$TORCHRUN" \
      --master-port "$port" --nproc-per-node 4 \
      --module src.cli.generate \
      --ckpt-path "$CKPT" --config "$CFG" \
      --input-file "$LONG" --max-new-tokens 32 --temperature 0 \
      --routed-experts-device cpu --pd-mode scheduler \
      >"$LOGDIR/$name.log" 2>&1 || echo "(exit=$?)"
  grep -E 'generate time:|prefill time:|^Completion:' "$LOGDIR/$name.log" || true
}

run_case inline1                  29920 DEEPSEEK_CPU_DECODE_INLINE_THRESHOLD=1
run_case inline0                  29921 DEEPSEEK_CPU_DECODE_INLINE_THRESHOLD=0
run_case inline0_persistent       29922 DEEPSEEK_CPU_DECODE_INLINE_THRESHOLD=0 DEEPSEEK_CPU_PERSISTENT_WORKER=1

echo
echo '=== summary ==='
for n in inline1 inline0 inline0_persistent; do
  m=$(grep -oE 'prefill time: [0-9.]+s, prefill tokens: [0-9]+, decode time: [0-9.]+s, decode tokens: [0-9]+, decode tokens/s: [0-9.]+' "$LOGDIR/$n.log" | tail -1)
  c=$(grep -E '^Completion:' "$LOGDIR/$n.log" | head -1)
  echo "[$n] $m"
  echo "  $c"
done
