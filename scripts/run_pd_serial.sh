#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="$REPO_ROOT"
TMP_ROOT="${POCKETLLM_TMP_DIR:-$REPO_ROOT/.tmp}"
mkdir -p "$TMP_ROOT"
LOCK_FILE="${LOCK_FILE:-$TMP_ROOT/pocketllm_full_network_benchmark.lock}"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "another full-network benchmark is already running: $LOCK_FILE" >&2
  exit 1
fi
PYTHON="${PYTHON:-python}"
TORCHRUN="${TORCHRUN:-torchrun}"
DEFAULT_CKPT_PATH="$REPO_ROOT/checkpoints/DeepSeek-V4-Flash-w8a8"
CKPT_PATH="${CKPT_PATH:-$DEFAULT_CKPT_PATH}"
CONFIG="$REPO_ROOT/configs/config_w8a8.json"

if [[ ! -e "$CKPT_PATH" ]]; then
  echo "checkpoint not found: $CKPT_PATH" >&2
  echo "Set CKPT_PATH or place the checkpoint under $DEFAULT_CKPT_PATH" >&2
  exit 1
fi
SHORT_INPUT_FILE="${SHORT_INPUT_FILE:-$REPO_ROOT/tests/fixtures/smoke_input.txt}"
LONG_INPUT_FILE="${LONG_INPUT_FILE:-$TMP_ROOT/pocketllm_long_input_single.txt}"
SHORT_MAX_NEW_TOKENS="${SHORT_MAX_NEW_TOKENS:-8}"
LONG_MAX_NEW_TOKENS="${LONG_MAX_NEW_TOKENS:-1}"
SHORT_MASTER_PORT="${SHORT_MASTER_PORT:-29682}"
LONG_MASTER_PORT="${LONG_MASTER_PORT:-29683}"
CLIENT_CPUSET="${CLIENT_CPUSET:-}"
PD_MODE="${PD_MODE:-scheduler}"
PD_PREFILL_CHUNK_TOKENS="${PD_PREFILL_CHUNK_TOKENS:-0}"
PD_COMPARE_MODES="${PD_COMPARE_MODES:-0}"
PD_CASE="${PD_CASE:-both}"
DEEPSEEK_CPU_MOE_EXTERNAL_SERVER="${DEEPSEEK_CPU_MOE_EXTERNAL_SERVER:-0}"
DEEPSEEK_CPU_MOE_SHARED_WEIGHTS="${DEEPSEEK_CPU_MOE_SHARED_WEIGHTS:-0}"
DEEPSEEK_CPU_MOE_SHARED_WEIGHT_DIR="${DEEPSEEK_CPU_MOE_SHARED_WEIGHT_DIR:-$TMP_ROOT/pocketllm_shared_cpu_moe_${$}}"
DEEPSEEK_CPU_MOE_SERVER_SHM="${DEEPSEEK_CPU_MOE_SERVER_SHM:-pocketllm_cpu_moe_server_${$}}"
DEEPSEEK_CPU_MOE_SERVER_PID=""
DEEPSEEK_CPU_MOE_SHARED_WEIGHT_PRECREATE="${DEEPSEEK_CPU_MOE_SHARED_WEIGHT_PRECREATE:-$DEEPSEEK_CPU_MOE_SHARED_WEIGHTS}"

cleanup_server() {
  if [[ -n "$DEEPSEEK_CPU_MOE_SERVER_PID" ]]; then
    kill "$DEEPSEEK_CPU_MOE_SERVER_PID" 2>/dev/null || true
    wait "$DEEPSEEK_CPU_MOE_SERVER_PID" 2>/dev/null || true
    DEEPSEEK_CPU_MOE_SERVER_PID=""
  fi
}
trap cleanup_server EXIT

start_external_server() {
  if [[ "$DEEPSEEK_CPU_MOE_EXTERNAL_SERVER" != "1" ]]; then
    return
  fi
  rm -rf "$DEEPSEEK_CPU_MOE_SHARED_WEIGHT_DIR"
  mkdir -p "$DEEPSEEK_CPU_MOE_SHARED_WEIGHT_DIR"
  local server_log="$TMP_ROOT/pocketllm_cpu_moe_server_${$}.log"
  env \
    DEEPSEEK_CPU_MOE_CPP_LOOP=1 \
    DEEPSEEK_CPU_MOE_CPP_LOOP_V2="${DEEPSEEK_CPU_MOE_CPP_LOOP_V2:-1}" \
    DEEPSEEK_CPU_MOE_SHARED_WEIGHTS="$DEEPSEEK_CPU_MOE_SHARED_WEIGHTS" \
    DEEPSEEK_CPU_MOE_SHARED_WEIGHT_DIR="$DEEPSEEK_CPU_MOE_SHARED_WEIGHT_DIR" \
    DEEPSEEK_CPU_MOE_SHARED_WEIGHT_PRECREATE="$DEEPSEEK_CPU_MOE_SHARED_WEIGHT_PRECREATE" \
    DEEPSEEK_CPU_MOE_NUMA_INTERLEAVE="${DEEPSEEK_CPU_MOE_NUMA_INTERLEAVE:-1}" \
    OMP_PROC_BIND="${OMP_PROC_BIND:-spread}" \
    PYTHONPATH="$REPO_ROOT" \
    "$PYTHON" -m src.models.deepseek_v4.moe_server \
      --ckpt-path "$CKPT_PATH" \
      --config "$CONFIG" \
      --shm-name "$DEEPSEEK_CPU_MOE_SERVER_SHM" \
      --omp-threads "${DEEPSEEK_CPU_MOE_SERVER_OMP_THREADS:-22}" \
      --shared-weight-dir "$DEEPSEEK_CPU_MOE_SHARED_WEIGHT_DIR" \
      --shared-weight-world-size 4 \
      > "$server_log" 2>&1 &
  DEEPSEEK_CPU_MOE_SERVER_PID=$!
  if [[ "$DEEPSEEK_CPU_MOE_SHARED_WEIGHT_PRECREATE" == "1" ]]; then
    for _ in $(seq 1 "${DEEPSEEK_CPU_MOE_SERVER_WAIT_SECONDS:-1800}"); do
      if grep -q "cpu_moe_server created shared weight arenas" "$server_log" 2>/dev/null; then
        break
      fi
      if ! kill -0 "$DEEPSEEK_CPU_MOE_SERVER_PID" 2>/dev/null; then
        cat "$server_log"
        exit 1
      fi
      sleep 1
    done
    if ! grep -q "cpu_moe_server created shared weight arenas" "$server_log" 2>/dev/null; then
      echo "server did not create shared weight arenas" >&2
      cat "$server_log" >&2
      exit 1
    fi
  fi
}

if [[ ! -s "$LONG_INPUT_FILE" ]]; then
  "$PYTHON" - <<PY
from transformers import AutoTokenizer
ckpt = r"""$CKPT_PATH"""
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
with open("$TMP_ROOT/pocketllm_long_input_single.txt", "w") as f:
    f.write(text)
print(f"wrote $TMP_ROOT/pocketllm_long_input_single.txt tokens={len(tok.encode(text))}", flush=True)
PY
fi

run_single() {
  local mode="$1"
  local input_file="$2"
  local max_new_tokens="$3"
  local master_port="$4"
  local client_log="$5"

  rm -f "$client_log"

  local client_launcher=("$TORCHRUN")
  if [[ -n "$CLIENT_CPUSET" ]]; then
    client_launcher=(taskset -c "$CLIENT_CPUSET" "$TORCHRUN")
  fi

  start_external_server

  set +e
  DEEPSEEK_PD_PHASE_AUTO_SELECT=1 \
  DEEPSEEK_PD_SINGLE_CPU_MOE_WEIGHTS=1 \
  DEEPSEEK_CPU_MOE_INPROC_SERVER="${DEEPSEEK_CPU_MOE_INPROC_SERVER:-0}" \
  DEEPSEEK_CPU_MOE_EXTERNAL_SERVER="$DEEPSEEK_CPU_MOE_EXTERNAL_SERVER" \
  DEEPSEEK_CPU_MOE_EXTERNAL_PREFILL_LOCAL="${DEEPSEEK_CPU_MOE_EXTERNAL_PREFILL_LOCAL:-1}" \
  DEEPSEEK_CPU_MOE_SHARED_WEIGHTS="$DEEPSEEK_CPU_MOE_SHARED_WEIGHTS" \
  DEEPSEEK_CPU_MOE_SHARED_WEIGHT_DIR="$DEEPSEEK_CPU_MOE_SHARED_WEIGHT_DIR" \
  DEEPSEEK_CPU_MOE_SHARED_WEIGHT_PRECREATED="$DEEPSEEK_CPU_MOE_SHARED_WEIGHT_PRECREATE" \
  DEEPSEEK_CPU_MOE_SERVER_SHM="$DEEPSEEK_CPU_MOE_SERVER_SHM" \
  DEEPSEEK_CPU_MOE_PREDICT_SEQ="${DEEPSEEK_CPU_MOE_PREDICT_SEQ:-$DEEPSEEK_CPU_MOE_EXTERNAL_SERVER}" \
  DEEPSEEK_CPU_MOE_SERVER_OMP_THREADS="${DEEPSEEK_CPU_MOE_SERVER_OMP_THREADS:-22}" \
  DEEPSEEK_CPU_MOE_NONSERVER_OMP_THREADS="${DEEPSEEK_CPU_MOE_NONSERVER_OMP_THREADS:-1}" \
  DEEPSEEK_PD_DECODE_OMP_THREADS="${DEEPSEEK_PD_DECODE_OMP_THREADS:-8}" \
  DEEPSEEK_GPU_PREFILL_MOE=1 \
  DEEPSEEK_PD_PREFILL_WQ_A_INT8="${DEEPSEEK_PD_PREFILL_WQ_A_INT8:-${DEEPSEEK_WQ_A_INT8:-0}}" \
  DEEPSEEK_PD_PREFILL_WQ_B_INT8="${DEEPSEEK_PD_PREFILL_WQ_B_INT8:-${DEEPSEEK_WQ_B_INT8:-0}}" \
  DEEPSEEK_PD_PREFILL_WKV_INT8="${DEEPSEEK_PD_PREFILL_WKV_INT8:-${DEEPSEEK_WKV_INT8:-0}}" \
  DEEPSEEK_PD_PREFILL_WO_A_INT8="${DEEPSEEK_PD_PREFILL_WO_A_INT8:-${DEEPSEEK_WO_A_INT8:-0}}" \
  DEEPSEEK_PD_PREFILL_WO_B_INT8="${DEEPSEEK_PD_PREFILL_WO_B_INT8:-${DEEPSEEK_WO_B_INT8:-0}}" \
  DEEPSEEK_PD_PREFILL_INDEXER_WQ_B_INT8="${DEEPSEEK_PD_PREFILL_INDEXER_WQ_B_INT8:-${DEEPSEEK_INDEXER_WQ_B_INT8:-0}}" \
  DEEPSEEK_PD_DECODE_WQ_A_INT8="${DEEPSEEK_PD_DECODE_WQ_A_INT8:-${DEEPSEEK_WQ_A_INT8:-0}}" \
  DEEPSEEK_PD_DECODE_WQ_B_INT8="${DEEPSEEK_PD_DECODE_WQ_B_INT8:-${DEEPSEEK_WQ_B_INT8:-0}}" \
  DEEPSEEK_PD_DECODE_WKV_INT8="${DEEPSEEK_PD_DECODE_WKV_INT8:-${DEEPSEEK_WKV_INT8:-0}}" \
  DEEPSEEK_PD_DECODE_WO_A_INT8="${DEEPSEEK_PD_DECODE_WO_A_INT8:-${DEEPSEEK_WO_A_INT8:-0}}" \
  DEEPSEEK_PD_DECODE_WO_B_INT8="${DEEPSEEK_PD_DECODE_WO_B_INT8:-${DEEPSEEK_WO_B_INT8:-0}}" \
  DEEPSEEK_PD_DECODE_INDEXER_WQ_B_INT8="${DEEPSEEK_PD_DECODE_INDEXER_WQ_B_INT8:-${DEEPSEEK_INDEXER_WQ_B_INT8:-0}}" \
  DEEPSEEK_GPU_PREFILL_MOE_GROUPED_GEMM=1 \
  DEEPSEEK_GPU_PREFILL_MOE_PREFETCH_BEFORE_FFN=1 \
  DEEPSEEK_GPU_PREFILL_MOE_MAX_CACHED_LAYERS="${DEEPSEEK_GPU_PREFILL_MOE_MAX_CACHED_LAYERS:-3}" \
  DEEPSEEK_CPU_DECODE_INLINE_THRESHOLD="${DEEPSEEK_CPU_DECODE_INLINE_THRESHOLD:-1}" \
  DEEPSEEK_CPU_TOPK_PERSISTENT="${DEEPSEEK_CPU_TOPK_PERSISTENT:-1}" \
  DEEPSEEK_INT8_IMPL=cuda_ext \
  DEEPSEEK_MOE_ASYNC_ALLREDUCE=1 \
  DEEPSEEK_SHARED_EXPERT_INT8=1 \
  DEEPSEEK_FLASHINFER_STYLE_ATTN_CUDA=1 \
  DEEPSEEK_FUSED_C4_INDEXER_CUDA="${DEEPSEEK_FUSED_C4_INDEXER_CUDA:-1}" \
  DEEPSEEK_HC_PRE_CUDA="${DEEPSEEK_HC_PRE_CUDA:-1}" \
  PYTHONPATH="$REPO_ROOT" \
  "${client_launcher[@]}" --master-port "$master_port" --nproc-per-node 4 --module src.cli.generate \
    --ckpt-path "$CKPT_PATH" \
    --config "$CONFIG" \
    --input-file "$input_file" \
    --max-new-tokens "$max_new_tokens" \
    --temperature 0 \
    --routed-experts-device cpu \
    --pd-mode "$mode" \
    --pd-prefill-chunk-tokens "$PD_PREFILL_CHUNK_TOKENS" \
    > "$client_log" 2>&1
  local client_rc=$?
  set -e

  cleanup_server

  echo "--- client timing ---"
  grep -E 'generate time:|prefill time:|decode time:|^Completion:' "$client_log" || true
  return "$client_rc"
}

run_suite() {
  local mode="$1"
  local short_master_port="$2"
  local long_master_port="$3"
  local suffix="$4"

  if [[ "$PD_CASE" == "both" || "$PD_CASE" == "short" ]]; then
    echo "=== short sequence correctness/performance (pd-mode=$mode) ==="
    run_single "$mode" "$SHORT_INPUT_FILE" "$SHORT_MAX_NEW_TOKENS" "$short_master_port" "$TMP_ROOT/pocketllm_pd_short_client_${suffix}.log"
  fi

  if [[ "$PD_CASE" == "both" || "$PD_CASE" == "long" ]]; then
    echo "=== long sequence correctness/performance (pd-mode=$mode) ==="
    run_single "$mode" "$LONG_INPUT_FILE" "$LONG_MAX_NEW_TOKENS" "$long_master_port" "$TMP_ROOT/pocketllm_pd_long_client_${suffix}.log"
  fi
}

if [[ "$PD_COMPARE_MODES" == "1" ]]; then
  run_suite off "$SHORT_MASTER_PORT" "$LONG_MASTER_PORT" off
  run_suite scheduler "$((SHORT_MASTER_PORT + 10))" "$((LONG_MASTER_PORT + 10))" scheduler
else
  run_suite "$PD_MODE" "$SHORT_MASTER_PORT" "$LONG_MASTER_PORT" "$PD_MODE"
fi
