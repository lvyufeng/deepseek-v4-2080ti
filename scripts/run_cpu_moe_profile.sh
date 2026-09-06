#!/usr/bin/env bash
# Capture per-layer cpu_moe timings during decode.
set -eo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="$REPO_ROOT"
TMP_ROOT="${POCKETLLM_TMP_DIR:-$REPO_ROOT/.tmp}"
mkdir -p "$TMP_ROOT"
PYTHON="${PYTHON:-python}"
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
OUTDIR="${OUTDIR:-$TMP_ROOT/pocketllm_cpu_moe_profile}"
rm -rf "$OUTDIR"
mkdir -p "$OUTDIR"

env \
  DEEPSEEK_CPU_MOE_PROFILE=1 \
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
    --master-port 29917 \
    --nproc-per-node 4 \
    --module src.cli.generate \
    --ckpt-path "$CKPT" \
    --config "$CFG" \
    --input-file "$LONG" \
    --max-new-tokens 4 \
    --temperature 0 \
    --routed-experts-device cpu \
    --pd-mode scheduler \
    >"$OUTDIR/run.log" 2>&1

echo "=== timing ==="
grep -E "generate time:|prefill time:|^Completion:" "$OUTDIR/run.log" || true
echo
echo "=== cpu_moe per-layer summary (decode steps only, items=1) ==="
"$PYTHON" - <<PY
import re
from collections import defaultdict
times = defaultdict(list)  # layer_idx -> list of step timings (s)
runners = {}
text = open("$OUTDIR/run.log").read()
for m in re.finditer(r"cpu_moe layer=(\d+) items=(\d+) runner=(\S+) time=([0-9.]+)s", text):
    layer, items, runner, t = int(m.group(1)), int(m.group(2)), m.group(3), float(m.group(4))
    if items == 1:  # decode step
        times[layer].append(t)
        runners[layer] = runner
total_steps = max(len(v) for v in times.values()) if times else 0
print(f"layers seen={len(times)} max_steps_per_layer={total_steps}")
print()
print(f"{'layer':>6} {'runner':<60} {'n':>3} {'min_ms':>8} {'med_ms':>8} {'max_ms':>8}")
sums = []
for layer in sorted(times.keys()):
    ts = sorted(times[layer])
    n = len(ts)
    mn = ts[0]*1000
    md = ts[n//2]*1000
    mx = ts[-1]*1000
    sums.append((mn, md, mx))
    print(f"{layer:>6} {runners[layer]:<60} {n:>3} {mn:>8.2f} {md:>8.2f} {mx:>8.2f}")
print()
if sums:
    n_layers = len(sums)
    print(f"per-step total cpu_moe (sum across layers): min={sum(s[0] for s in sums):.1f}ms  med={sum(s[1] for s in sums):.1f}ms  max={sum(s[2] for s in sums):.1f}ms")
PY
