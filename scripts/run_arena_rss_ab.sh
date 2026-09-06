#!/usr/bin/env bash
# arena 唯一存储 vs 双份的 RSS 对比。
# 跑 long_short 各一次，期间 1Hz 采集 4 个 rank 的 VmRSS，输出峰值/末值。
set -eo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="$REPO_ROOT"
TMP_ROOT="${POCKETLLM_TMP_DIR:-$REPO_ROOT/.tmp}"
mkdir -p "$TMP_ROOT"
TORCHRUN="${TORCHRUN:-torchrun}"
PYTHON="${PYTHON:-python}"
DEFAULT_CKPT_PATH="$REPO_ROOT/checkpoints/DeepSeek-V4-Flash-w8a8"
CKPT="${CKPT_PATH:-$DEFAULT_CKPT_PATH}"
CFG="$REPO_ROOT/configs/config_w8a8.json"

if [[ ! -e "$CKPT" ]]; then
  echo "checkpoint not found: $CKPT" >&2
  echo "Set CKPT_PATH or place the checkpoint under $DEFAULT_CKPT_PATH" >&2
  exit 1
fi
LONG="$TMP_ROOT/pocketllm_long_input_single.txt"
LOGDIR="$TMP_ROOT/pocketllm_arena_rss"
rm -rf "$LOGDIR"
mkdir -p "$LOGDIR"

common_env=(
  DEEPSEEK_PD_PHASE_AUTO_SELECT=1
  DEEPSEEK_GPU_PREFILL_MOE=1
  DEEPSEEK_GPU_PREFILL_MOE_GROUPED_GEMM=1
  DEEPSEEK_GPU_PREFILL_MOE_PREFETCH_BEFORE_FFN=1
  DEEPSEEK_GPU_PREFILL_MOE_MAX_CACHED_LAYERS=3
  DEEPSEEK_INT8_IMPL=cuda_ext
  DEEPSEEK_MOE_ASYNC_ALLREDUCE=1
  DEEPSEEK_SHARED_EXPERT_INT8=1
  DEEPSEEK_FLASHINFER_STYLE_ATTN_CUDA=1
  DEEPSEEK_FUSED_C4_INDEXER_CUDA=1
  DEEPSEEK_HC_PRE_CUDA=1
  DEEPSEEK_PD_DECODE_OMP_THREADS=8
  DEEPSEEK_CPU_DECODE_INLINE_THRESHOLD=1
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

sample_rss() {
  local label="$1"
  local interval=2
  local stop_marker="$LOGDIR/${label}.stop"
  local out="$LOGDIR/${label}.rss"
  local dump="$LOGDIR/${label}.proclist"
  : >"$out"
  : >"$dump"
  local diag_done=0
  while [[ ! -e "$stop_marker" ]]; do
    local ts pids
    ts=$(date +%s)
    # Match every running src.cli.generate worker (4 ranks). torchrun (the parent)
    # also carries "src.cli.generate" in its cmdline. We sum all matches since the
    # workers hold the actual model state.
    pids=$(pgrep -f "src.cli.generate" || true)
    if [[ -n "$pids" ]]; then
      local sum=0
      local rank_count=0
      for p in $pids; do
        if [[ -r "/proc/$p/status" ]]; then
          local rss
          rss=$(awk '/^VmRSS:/ {print $2}' "/proc/$p/status" 2>/dev/null || echo 0)
          [[ -n "$rss" ]] && sum=$((sum + rss))
          rank_count=$((rank_count + 1))
        fi
      done
      echo "$ts $sum $rank_count" >>"$out"
      if (( diag_done == 0 )) && (( rank_count >= 4 )); then
        {
          echo "=== first quad ts=$ts ==="
          for p in $pids; do
            local cmd rss
            cmd=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null | head -c 240)
            rss=$(awk '/^VmRSS:/ {print $2}' "/proc/$p/status" 2>/dev/null || echo 0)
            echo "  pid=$p rss_kb=$rss cmd=$cmd"
          done
        } >>"$dump"
        diag_done=1
      fi
    fi
    sleep "$interval"
  done
}

run_case() {
  local name="$1"
  local port="$2"
  local arena="$3"
  local log="$LOGDIR/${name}.log"
  local stop_marker="$LOGDIR/${name}.stop"
  rm -f "$stop_marker"
  echo "=== $name (arena=$arena) ==="
  sample_rss "$name" "$port" &
  local sampler_pid=$!
  env "${common_env[@]}" \
      DEEPSEEK_GPU_PREFILL_MOE_ARENA="$arena" \
      DEEPSEEK_GPU_PREFILL_MOE_PROFILE=1 \
      PYTHONPATH="$REPO_ROOT" \
      "$TORCHRUN" \
      --master-port "$port" \
      --nproc-per-node 4 \
      --module src.cli.generate \
      --ckpt-path "$CKPT" \
      --config "$CFG" \
      --input-file "$LONG" \
      --max-new-tokens 8 \
      --temperature 0 \
      --routed-experts-device cpu \
      --pd-mode scheduler \
      >"$log" 2>&1 || echo "(case exit=$?)"
  touch "$stop_marker"
  wait "$sampler_pid" 2>/dev/null || true
  grep -E 'generate time:|prefill time:|^Completion:' "$log" || true
}

run_case arena_off 29911 0
run_case arena_on  29912 1

"$PYTHON" - <<'PY'
import os
LOGDIR = "$TMP_ROOT/pocketllm_arena_rss"
for name in ["arena_off", "arena_on"]:
    rss_path = os.path.join(LOGDIR, f"{name}.rss")
    if not os.path.exists(rss_path):
        print(f"--- {name} --- (no rss samples)")
        continue
    samples = []
    with open(rss_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            ts = int(parts[0])
            kb = int(parts[1])
            ranks = int(parts[2]) if len(parts) >= 3 else 0
            samples.append((ts, kb, ranks))
    if not samples:
        print(f"--- {name} --- (empty rss)")
        continue
    peak = max(s[1] for s in samples) / 1024 / 1024  # GB
    last = samples[-1][1] / 1024 / 1024
    duration = samples[-1][0] - samples[0][0]
    max_ranks = max(s[2] for s in samples)
    print(
        f"--- {name} --- samples={len(samples)} duration={duration}s "
        f"peak_RSS_total={peak:.2f}GB last_RSS_total={last:.2f}GB ranks={max_ranks}"
    )
PY
