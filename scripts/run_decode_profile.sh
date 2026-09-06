#!/usr/bin/env bash
# Capture a single decode step torch.profiler trace from rank 0.
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
OUTDIR="${OUTDIR:-$TMP_ROOT/pocketllm_decode_profile}"
rm -rf "$OUTDIR"
mkdir -p "$OUTDIR"

env \
  DEEPSEEK_DECODE_PROFILE_DIR="$OUTDIR" \
  DEEPSEEK_DECODE_PROFILE_STEP=2 \
  DEEPSEEK_DECODE_PROFILE_RANK=0 \
  DEEPSEEK_PD_PHASE_AUTO_SELECT=1 \
  DEEPSEEK_GPU_PREFILL_MOE=1 \
  DEEPSEEK_GPU_PREFILL_MOE_GROUPED_GEMM=1 \
  DEEPSEEK_GPU_PREFILL_MOE_PREFETCH_BEFORE_FFN=1 \
  DEEPSEEK_GPU_PREFILL_MOE_MAX_CACHED_LAYERS=3 \
  DEEPSEEK_GPU_PREFILL_MOE_ARENA=1 \
  DEEPSEEK_INT8_IMPL=cuda_ext \
  DEEPSEEK_MOE_ASYNC_ALLREDUCE=1 \
  DEEPSEEK_SHARED_EXPERT_INT8=1 \
  DEEPSEEK_FLASHINFER_STYLE_ATTN_CUDA=1 \
  DEEPSEEK_FUSED_C4_INDEXER_CUDA=1 \
  DEEPSEEK_HC_PRE_CUDA=1 \
  DEEPSEEK_PD_DECODE_OMP_THREADS=8 \
  DEEPSEEK_CPU_DECODE_INLINE_THRESHOLD=1 \
  DEEPSEEK_CPU_TOPK_PERSISTENT=1 \
  DEEPSEEK_PD_PREFILL_WQ_A_INT8=1 \
  DEEPSEEK_PD_PREFILL_WQ_B_INT8=1 \
  DEEPSEEK_PD_PREFILL_WKV_INT8=1 \
  DEEPSEEK_PD_PREFILL_WO_A_INT8=1 \
  DEEPSEEK_PD_PREFILL_WO_B_INT8=1 \
  DEEPSEEK_PD_PREFILL_INDEXER_WQ_B_INT8=1 \
  DEEPSEEK_PD_DECODE_WQ_A_INT8=1 \
  DEEPSEEK_PD_DECODE_WQ_B_INT8=1 \
  DEEPSEEK_PD_DECODE_WKV_INT8=1 \
  DEEPSEEK_PD_DECODE_WO_A_INT8=1 \
  DEEPSEEK_PD_DECODE_WO_B_INT8=1 \
  DEEPSEEK_PD_DECODE_INDEXER_WQ_B_INT8=1 \
  PYTHONPATH="$REPO_ROOT" \
  "$TORCHRUN" \
    --master-port 29915 \
    --nproc-per-node 4 \
    --module src.cli.generate \
    --ckpt-path "$CKPT" \
    --config "$CFG" \
    --input-file "$LONG" \
    --max-new-tokens 6 \
    --temperature 0 \
    --routed-experts-device cpu \
    --pd-mode scheduler \
    >"$OUTDIR/run.log" 2>&1

echo "=== timing ==="
grep -E "generate time:|prefill time:|^Completion:" "$OUTDIR/run.log" || true
echo "=== summary (top 25 by CUDA time) ==="
ls "$OUTDIR"/*.summary.txt 2>/dev/null | while read f; do
  echo "-- $f --"
  cat "$f"
done
