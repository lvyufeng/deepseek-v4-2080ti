#!/usr/bin/env bash
# Best-known logical PD scheduler config for DeepSeek-V4-Flash-w8a8.
#
# Architecture invariants:
# - Single model, single request goes through prefill -> decode naturally.
# - Phase-aware attention INT8 (prefill + decode each enable WQ_A/WQ_B/WKV/WO_A/WO_B/INDEXER_WQ_B).
# - GPU prefill MoE staging with grouped GEMM and a 3-layer LRU cache.
# - Routed CPU MoE stays in-process; no external shared-memory server, no shared-weight files.
#
# Frozen optimizations (each entry: what changed, expected effect, escape hatch).
# 1. GPU prefill MoE arena: single-copy pinned int8 expert weights replace the
#    duplicate per-expert tensors. nn.Parameter storages are released after copy
#    so the arena is the unique physical store. Saves ~10.5 GB host RSS, kills
#    ~29 s/long-prefill of cudaHostRegister, and feeds GPU prefill via one
#    non-blocking H2D copy per layer.
#    Code: cpu_routed_backend.py uses _env_enabled_default_on() so it ships ON.
#    Disable: DEEPSEEK_GPU_PREFILL_MOE_ARENA=0
# 2. CPU MoE INT8 thread_local scratch buffers: replaces 4 std::vector mallocs
#    per call inside int8_single_token_topk_persistent_impl with persistent
#    thread_local vectors. Decode TPS 1.78 -> 1.89 (+6.2%) on long_long under
#    schedutil. Effect partly diluted under performance governor (see #4).
#    Code: deepseek_cpu_moe_ext.cpp; rebuilt in-place. No env knob.
# 3. OMP_THREADS=12 for CPU MoE decode: sweep 8/12/16/22 picked 12 as the
#    sweet spot. Decode TPS 1.52 -> 1.82 (+19.9%).
#    Disable: DEEPSEEK_PD_DECODE_OMP_THREADS=<n>
# 4. Pin CPU governor to "performance" while the benchmark runs (opt-in).
#    schedutil keeps cores at 1.2-2.1 GHz (max 3.7) during decode because the
#    OMP workers idle most of each step; switching to performance unblocks
#    turbo and lifts long_long decode TPS from ~1.63 to ~1.89 (+15%). The
#    governor is restored on exit so this is not a permanent change to the
#    machine.
#    Enable: DEEPSEEK_BENCH_GOVERNOR_PIN=1 (requires passwordless sudo for
#    `tee /sys/.../scaling_governor` and the intel_pstate knobs).
# 5. Fused attention decode prefuse kernels (Plan B-小-v2). Two CUDA opaque
#    ops collapse the per-layer attention front-end (model.py:798-812):
#      - fused_q_rmsnorm_rope_inplace: per-head rsqrt(mean(square)+eps) +
#        complex rotary on the trailing rope dims of q in one launch.
#      - fused_kv_rope_actquant_inplace: kv_norm + complex rotary +
#        block-FP8 (e4m3fn, ue8m0 scale) inplace simulation in one launch.
#    Replaces ~16 PyTorch dispatches per layer with 2 kernel launches; with
#    43 layers that is ~600 dispatches/step trimmed from the eager loop.
#    short_short decode TPS 1.962 -> 2.099 (+7.0%); long_long decode TPS
#    1.750 -> 1.828 (+4.5%, governor pinned). short_short greedy output is
#    bit-identical to baseline. long_long greedy diverges in wording around
#    token ~10 because the q rsqrt path runs in fp32 here vs bf16 in the
#    reference (max abs diff 1.56e-2, mean 9.9e-4 -- inside bf16 ULP
#    envelope). Treated as accepted bf16 numerical variance.
#    Code: cuda_kernel_impl.cu (kernels), cuda_kernel.cpp (pybind),
#    model.py:Attention.forward (call sites + freqs cache).
#    Enable: DEEPSEEK_FUSED_ATTN_PREFUSE=1 (default off until shipped).
#    Validate kernels alone: PYTHONPATH=$PWD python tests/test_fused_attn_prefuse.py
# 6. Async CPU MoE / shared-expert overlap (Plan A-overlap).
#    DEEPSEEK_CPU_DECODE_INLINE_THRESHOLD default flipped 1 -> 0 so that
#    submit_forward dispatches OMP work to the worker thread (default
#    ThreadPoolExecutor) instead of running it inline on the calling thread.
#    The Python flow then immediately calls self.shared_experts(x), which
#    launches its INT8 GEMMs on the GPU while OMP grinds the routed experts
#    on CPU. The async_allreduce path already issues sync_forward on the
#    side-stream, so this just unblocks the submit -> shared sequence.
#    DEEPSEEK_BENCH_GOVERNOR_PIN=1, all four cases:
#      short_short  2.099 -> 2.424 TPS  (+15.5%)
#      short_long   1.800 -> 2.466 TPS  (+37.0%)
#      long_short   1.720 -> 2.366 TPS  (+37.6%)
#      long_long    2.057 -> 2.378 TPS  (+15.4%)
#    Sustained over 3 long_long runs ({2.417, 2.370, 2.392}, mean 2.393,
#    sigma <1%). Greedy outputs are well-formed Chinese summaries (slightly
#    different wording from baseline because submit_forward now starts on
#    a different stream timing, but content is consistent).
#    Disable: DEEPSEEK_CPU_DECODE_INLINE_THRESHOLD=1 (revert to inline).
#    Persistent worker (DEEPSEEK_CPU_PERSISTENT_WORKER=1) was tested
#    too — 2.276 TPS, slightly slower than the executor path because of the
#    extra condition-variable handshake; left at default off.
#
# Tried but NOT adopted (rationale documented so we do not retry blindly):
# - pmaddubsw sign-trick rewrite of dot_i8_avx2 / dot_i8_avx2_pair / dot_i8_avx2_4.
#   dot_microbench.cpp shows ~2.0x throughput on the inner kernel
#   (30 -> 60 GiB/s @ dim=2048/4096) and is exact-match in [-127,127] domain.
#   First A/B (pre-overlap, inline CPU MoE): mean delta +0.6%, dismissed as
#   inside +/-10% jitter because OMP was not on the critical path then.
#   Re-A/B'd under the async-overlap config (this build, optimization #6) on
#   the hypothesis that with submit_forward dispatching off-thread the OMP
#   wall is now the bottleneck. DEEPSEEK_BENCH_GOVERNOR_PIN=1 long_long over
#   3 runs: pmaddubsw {2.473, 2.393, 2.323} TPS (mean 2.396) vs
#   baseline {2.417, 2.370, 2.392} (mean 2.393), delta +0.1% -- still
#   indistinguishable from jitter. Why: the overlap path means sync_forward
#   only blocks for `max(0, OMP_done_after - GPU_done_after)`. With shared
#   experts + NCCL allreduce on the side stream covering ~1.7 ms of GPU
#   time per layer and OMP taking only ~3 ms, the residual blocking wait
#   is small enough that compressing OMP further mostly converts it into
#   slack the side stream cannot use. Re-evaluate after CPU MoE itself
#   becomes the dominant unhidden critical path again (e.g. with much
#   larger batches, or after CUDA Graph capture cuts GPU time so OMP
#   no longer fits in the side-stream window).
#   Source preserved as research artefact: dot_microbench.cpp.
# - cuBLAS IMMA replacement of int8_gemm_rows_kernel for the per-token decode
#   GEMMs (wq_a / wq_b / wkv / wo_b / indexer.wq_b / shared_expert pair).
#   Kernel is bit-identical to the dp4a baseline on every actual decode
#   shape (tests/test_int8_gemm_imma.py: max_abs=0 across 8 cases). End-to-end
#   long_long under DEEPSEEK_BENCH_GOVERNOR_PIN=1: baseline TPS {2.057,
#   2.097} vs IMMA {2.030, 1.847}, mean delta -6.6%. The dp4a kernel is
#   36% of GPU time but GPU is ~92% idle waiting for the dispatcher and
#   CPU MoE; saving GPU time does not help wallclock and the extra dequant
#   kernel + int32 buffer allocations cost more dispatch overhead than
#   they save. Code path remains in the tree behind DEEPSEEK_INT8_GEMM_IMMA=1
#   so the analysis can be re-run after dispatcher overhead drops (e.g.
#   after CUDA Graph capture). Source: cuda_kernel_impl.cu
#   int8_gemm_imma_cuda / int8_gemm_pair_imma_cuda.
# - Fused inverse-rope kernel for the attention back-end (collapses
#   apply_rotary_emb(o[..., -rd:], freqs_cis, inverse=True) into one CUDA
#   launch). Kernel is bit-exact (tests/test_fused_attn_prefuse.py [o_inv]
#   max_abs=0). End-to-end long_long under DEEPSEEK_BENCH_GOVERNOR_PIN=1:
#   baseline {1.761, 2.074, 2.002} TPS (mean 1.946) vs inv-rope {2.089,
#   2.007, 1.841} TPS (mean 1.979), delta +1.7% -- inside the per-run
#   +/-10% jitter (baseline alone spans 17%). Same root cause as IMMA: the
#   inverse rope is ~5 PyTorch dispatches/layer but GPU is dispatcher-bound,
#   so trimming a handful of launches does not move wallclock. Code path
#   stays in the tree behind DEEPSEEK_FUSED_ATTN_INV_ROPE=1 so the analysis
#   can be re-run after CUDA Graph capture. Source: cuda_kernel_impl.cu
#   fused_o_inverse_rope_inplace_cuda.
# - Fold kv_cache write into fused_kv_rope_actquant_inplace (Plan B-小-v3).
#   Kernel writes the produced row directly into kv_cache[:, start_pos %
#   win, :], eliminating the Python-side `self.kv_cache[:bsz, slot] =
#   kv.squeeze(1)` and its 80+ select/copy_/as_strided dispatcher ops per
#   layer per step. Kernel bit-exact (tests/test_fused_attn_prefuse.py
#   [kv_cache] max_abs=0, no spillage to other slots). short_short greedy
#   output bit-identical. End-to-end long_long under
#   DEEPSEEK_BENCH_GOVERNOR_PIN=1: baseline {2.049, 2.054, 2.014} TPS (mean
#   2.039) vs fold {2.036, 2.103, 1.797, 2.061} TPS (mean 1.999), delta
#   -2.0% -- inside the +/-10% jitter. THIRD per-op fusion to hit the same
#   dispatcher-bound ceiling (after IMMA and inv-rope). decode profile shows
#   GPU is only ~12% of wallclock so trimming dispatches at the per-op level
#   does not move TPS. Code stays in the tree behind
#   DEEPSEEK_FUSED_KV_CACHE_FOLD=1 so it can be re-bench'd after CUDA Graph
#   capture or async double-buffered CPU MoE lands. Source:
#   cuda_kernel_impl.cu fused_kv_rope_actquant_kernel (kv_cache_out arm).
#
#
# Reference numbers (4x RTX 2080 Ti, $TMP_ROOT/pocketllm_long_input_single.txt ~2k tokens,
# arena ON + thread_local kernel + OMP=12 + fused attn prefuse + async overlap,
# src.cli.generate max_new=64 unless noted):
#   schedutil governor (default):
#     long/long  decode 1.63-1.85 TPS (high run-to-run jitter, pre-overlap)
#   performance governor (DEEPSEEK_BENCH_GOVERNOR_PIN=1) WITH async overlap:
#     short/short  prefill 2.07s   decode 2.42 TPS  (7 decoded tokens)
#     short/long   prefill 2.18s   decode 2.47 TPS  (9 decoded tokens)
#     long/short   prefill 11.33s  decode 2.37 TPS  (7 decoded tokens)
#     long/long    prefill 11.36s  decode 2.38 TPS  (63 decoded tokens, sustained)
#   performance governor pre-overlap (kept for delta tracking):
#     long/long  decode 2.03-2.10 TPS  (steady)
#     short/short  prefill 5.62s   decode 2.10 TPS  (7 decoded tokens)
#     short/long   prefill 3.63s   decode 1.80 TPS  (9 decoded tokens)
#     long/short   prefill 13.02s  decode 1.72 TPS  (7 decoded tokens)
#     long/long    prefill 14.5s   decode 2.06 TPS  (63 decoded tokens)
set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="$REPO_ROOT"
TMP_ROOT="${POCKETLLM_TMP_DIR:-$REPO_ROOT/.tmp}"
mkdir -p "$TMP_ROOT"
LOCK_FILE="${LOCK_FILE:-$TMP_ROOT/pocketllm_best_scheduler.lock}"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "another best-scheduler benchmark is already running: $LOCK_FILE" >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# Optional CPU governor pinning (frozen optimization #4).
#
# When DEEPSEEK_BENCH_GOVERNOR_PIN=1 we pin every CPU's governor to
# "performance" plus intel_pstate min_perf_pct=100 / no_turbo=0 for the
# duration of the script, then restore the previous values on exit. This
# requires passwordless sudo for `tee` against the relevant sysfs files.
#
# We probe `sudo -n true` first; if that fails (no passwordless sudo) we log
# a warning and continue without pinning rather than blocking interactively.
# Restoration runs from a `trap EXIT` so a Ctrl-C still hands the machine
# back in its original state.
# ---------------------------------------------------------------------------
GOV_DIR="/sys/devices/system/cpu"
GOV_BACKUP_FILE=""
GOV_MIN_PERF_BACKUP=""
GOV_NO_TURBO_BACKUP=""
GOV_PINNED=0

restore_governor() {
  if [[ "$GOV_PINNED" != "1" ]]; then
    return
  fi
  echo "restoring CPU governor and intel_pstate knobs..." >&2
  if [[ -s "$GOV_BACKUP_FILE" ]]; then
    while IFS=$'\t' read -r cpu_path prev_gov; do
      [[ -z "$cpu_path" ]] && continue
      echo "$prev_gov" | sudo -n tee "$cpu_path" >/dev/null 2>&1 || true
    done < "$GOV_BACKUP_FILE"
    rm -f "$GOV_BACKUP_FILE"
  fi
  if [[ -n "$GOV_MIN_PERF_BACKUP" && -e /sys/devices/system/cpu/intel_pstate/min_perf_pct ]]; then
    echo "$GOV_MIN_PERF_BACKUP" | sudo -n tee /sys/devices/system/cpu/intel_pstate/min_perf_pct >/dev/null 2>&1 || true
  fi
  if [[ -n "$GOV_NO_TURBO_BACKUP" && -e /sys/devices/system/cpu/intel_pstate/no_turbo ]]; then
    echo "$GOV_NO_TURBO_BACKUP" | sudo -n tee /sys/devices/system/cpu/intel_pstate/no_turbo >/dev/null 2>&1 || true
  fi
  GOV_PINNED=0
}
trap restore_governor EXIT INT TERM

if [[ "${DEEPSEEK_BENCH_GOVERNOR_PIN:-0}" == "1" ]]; then
  if ! sudo -n true 2>/dev/null; then
    echo "DEEPSEEK_BENCH_GOVERNOR_PIN=1 set but passwordless sudo unavailable; skipping governor pinning" >&2
  else
    GOV_BACKUP_FILE="$(mktemp "$TMP_ROOT/pocketllm_gov_backup.XXXXXX")"
    for f in "$GOV_DIR"/cpu*/cpufreq/scaling_governor; do
      [[ -e "$f" ]] || continue
      printf '%s\t%s\n' "$f" "$(cat "$f")" >> "$GOV_BACKUP_FILE"
      echo performance | sudo -n tee "$f" >/dev/null
    done
    if [[ -e /sys/devices/system/cpu/intel_pstate/min_perf_pct ]]; then
      GOV_MIN_PERF_BACKUP="$(cat /sys/devices/system/cpu/intel_pstate/min_perf_pct)"
      echo 100 | sudo -n tee /sys/devices/system/cpu/intel_pstate/min_perf_pct >/dev/null
    fi
    if [[ -e /sys/devices/system/cpu/intel_pstate/no_turbo ]]; then
      GOV_NO_TURBO_BACKUP="$(cat /sys/devices/system/cpu/intel_pstate/no_turbo)"
      echo 0 | sudo -n tee /sys/devices/system/cpu/intel_pstate/no_turbo >/dev/null
    fi
    GOV_PINNED=1
    sample_gov="$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo unknown)"
    echo "CPU governor pinned to performance (will restore on exit). cpu0=$sample_gov" >&2
  fi
fi

PYTHON="${PYTHON:-python}"
TORCHRUN="${TORCHRUN:-torchrun}"
DEFAULT_CKPT_PATH="$REPO_ROOT/checkpoints/DeepSeek-V4-Flash-w8a8"
CKPT_PATH="${CKPT_PATH:-$DEFAULT_CKPT_PATH}"
CONFIG="${CONFIG:-$REPO_ROOT/configs/config_w8a8.json}"

SHORT_INPUT_FILE="${SHORT_INPUT_FILE:-$REPO_ROOT/tests/fixtures/smoke_input.txt}"
LONG_INPUT_FILE="${LONG_INPUT_FILE:-$TMP_ROOT/pocketllm_long_input_single.txt}"
SHORT_MAX_NEW_TOKENS="${SHORT_MAX_NEW_TOKENS:-8}"
LONG_MAX_NEW_TOKENS="${LONG_MAX_NEW_TOKENS:-64}"

PD_CASE="${PD_CASE:-all}"   # all | short_short | short_long | long_short | long_long
LOG_DIR="${LOG_DIR:-$TMP_ROOT/pocketllm_best_scheduler}"
mkdir -p "$LOG_DIR"

# Generate the long single-prompt input file if it does not exist yet.
if [[ ! -s "$LONG_INPUT_FILE" ]]; then
  "$PYTHON" - <<PY
from transformers import AutoTokenizer
ckpt = "$CKPT_PATH"
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
with open("$LONG_INPUT_FILE", "w") as f:
    f.write(text)
print(f"wrote $LONG_INPUT_FILE tokens={len(tok.encode(text))}", flush=True)
PY
fi

# Best-known environment for logical PD scheduler.
# Note: keep DEEPSEEK_CPU_MOE_*EXTERNAL*=0 — routed CPU MoE stays in-process
# so we only load expert weights once and skip shared-memory plumbing.
export_best_env() {
  export DEEPSEEK_PD_PHASE_AUTO_SELECT=1
  export DEEPSEEK_GPU_PREFILL_MOE=1
  export DEEPSEEK_GPU_PREFILL_MOE_GROUPED_GEMM=1
  export DEEPSEEK_GPU_PREFILL_MOE_PREFETCH_BEFORE_FFN=1
  export DEEPSEEK_GPU_PREFILL_MOE_MAX_CACHED_LAYERS="${DEEPSEEK_GPU_PREFILL_MOE_MAX_CACHED_LAYERS:-3}"
  export DEEPSEEK_GPU_PREFILL_MOE_ARENA="${DEEPSEEK_GPU_PREFILL_MOE_ARENA:-1}"  # default-on; explicit for clarity
  export DEEPSEEK_GPU_MOE_DECODE_ACTIVE="${DEEPSEEK_GPU_MOE_DECODE_ACTIVE:-1}"
  export DEEPSEEK_GPU_MOE_CROSS_LAYER_PREFETCH="${DEEPSEEK_GPU_MOE_CROSS_LAYER_PREFETCH:-1}"
  export DEEPSEEK_GPU_MOE_CROSS_LAYER_PREFETCH_K="${DEEPSEEK_GPU_MOE_CROSS_LAYER_PREFETCH_K:-10}"
  export DEEPSEEK_GPU_MOE_CROSS_LAYER_PREFETCH_LOCAL_LIMIT="${DEEPSEEK_GPU_MOE_CROSS_LAYER_PREFETCH_LOCAL_LIMIT:-2}"
  export DEEPSEEK_INT8_IMPL=cuda_ext
  export DEEPSEEK_MOE_ASYNC_ALLREDUCE=1
  export DEEPSEEK_SHARED_EXPERT_INT8=1
  export DEEPSEEK_FLASHINFER_STYLE_ATTN_CUDA=1
  export DEEPSEEK_FUSED_C4_INDEXER_CUDA=1
  export DEEPSEEK_HC_PRE_CUDA=1
  export DEEPSEEK_CPU_DECODE_INLINE_THRESHOLD="${DEEPSEEK_CPU_DECODE_INLINE_THRESHOLD:-0}"
  export DEEPSEEK_CPU_TOPK_PERSISTENT="${DEEPSEEK_CPU_TOPK_PERSISTENT:-1}"
  export DEEPSEEK_PD_DECODE_OMP_THREADS="${DEEPSEEK_PD_DECODE_OMP_THREADS:-12}"
  export DEEPSEEK_FUSED_ATTN_PREFUSE="${DEEPSEEK_FUSED_ATTN_PREFUSE:-1}"

  # Phase-aware attention INT8 (matches the verified best path).
  for module in WQ_A WQ_B WKV WO_A WO_B INDEXER_WQ_B; do
    export "DEEPSEEK_PD_PREFILL_${module}_INT8=1"
    export "DEEPSEEK_PD_DECODE_${module}_INT8=1"
  done

  # Make absolutely sure no stray external CPU MoE plumbing is on.
  export DEEPSEEK_CPU_MOE_EXTERNAL_SERVER=0
  export DEEPSEEK_CPU_MOE_INPROC_SERVER=0
  export DEEPSEEK_CPU_MOE_SHARED_WEIGHTS=0
}

run_case() {
  local name="$1"
  local input="$2"
  local max_new="$3"
  local port="$4"
  local log="$LOG_DIR/${name}.log"

  echo "=== $name (input=$input, max_new=$max_new) ==="
  rm -f "$log"
  (
    export_best_env
    PYTHONPATH="$REPO_ROOT" "$TORCHRUN" \
      --master-port "$port" \
      --nproc-per-node 4 \
      --module src.cli.generate \
      --ckpt-path "$CKPT_PATH" \
      --config "$CONFIG" \
      --input-file "$input" \
      --max-new-tokens "$max_new" \
      --temperature 0 \
      --routed-experts-device cpu \
      --pd-mode scheduler \
      > "$log" 2>&1
  )
  grep -E 'generate time:|prefill time:|^Completion:' "$log" || true
}

case "$PD_CASE" in
  all)
    run_case short_short "$SHORT_INPUT_FILE" "$SHORT_MAX_NEW_TOKENS" 29881
    run_case short_long  "$SHORT_INPUT_FILE" "$LONG_MAX_NEW_TOKENS"  29882
    run_case long_short  "$LONG_INPUT_FILE"  "$SHORT_MAX_NEW_TOKENS" 29883
    run_case long_long   "$LONG_INPUT_FILE"  "$LONG_MAX_NEW_TOKENS"  29884
    ;;
  short_short) run_case short_short "$SHORT_INPUT_FILE" "$SHORT_MAX_NEW_TOKENS" 29881 ;;
  short_long)  run_case short_long  "$SHORT_INPUT_FILE" "$LONG_MAX_NEW_TOKENS"  29882 ;;
  long_short)  run_case long_short  "$LONG_INPUT_FILE"  "$SHORT_MAX_NEW_TOKENS" 29883 ;;
  long_long)   run_case long_long   "$LONG_INPUT_FILE"  "$LONG_MAX_NEW_TOKENS"  29884 ;;
  *)
    echo "unknown PD_CASE=$PD_CASE (expected: all|short_short|short_long|long_short|long_long)" >&2
    exit 2
    ;;
esac
