#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROOT="$REPO_ROOT"
TMP_ROOT="${POCKETLLM_TMP_DIR:-$REPO_ROOT/.tmp}"
mkdir -p "$TMP_ROOT"
if [[ "${DEEPSEEK_SKIP_BENCHMARK_LOCK:-0}" != "1" ]]; then
  LOCK_FILE="${DEEPSEEK_FULL_NETWORK_BENCHMARK_LOCK:-$TMP_ROOT/pocketllm_full_network_benchmark.lock}"
  exec 9>"$LOCK_FILE"
  if ! flock -n 9; then
    echo "another full-network benchmark is already running: $LOCK_FILE" >&2
    exit 1
  fi
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
INPUT_FILE="${INPUT_FILE:-$REPO_ROOT/tests/fixtures/smoke_input.txt}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-8}"
MASTER_PORT="${MASTER_PORT:-29661}"
SHM_NAME="${DEEPSEEK_CPU_MOE_SERVER_SHM:-pocketllm_cpu_moe_server}"
SERVER_LOG="${SERVER_LOG:-$TMP_ROOT/pocketllm_cpu_moe_server_best.log}"
CLIENT_LOG="${CLIENT_LOG:-$TMP_ROOT/pocketllm_best_external_cpu_moe.log}"
SERVER_OMP_THREADS="${SERVER_OMP_THREADS:-22}"
WAIT_SECONDS="${WAIT_SECONDS:-260}"

rm -f "/dev/shm/$SHM_NAME" "$SERVER_LOG" "$CLIENT_LOG"

DEEPSEEK_CPU_MOE_CPP_LOOP=1 \
DEEPSEEK_CPU_MOE_CPP_LOOP_V2="${DEEPSEEK_CPU_MOE_CPP_LOOP_V2:-1}" \
DEEPSEEK_CPU_MOE_NUMA_INTERLEAVE="${DEEPSEEK_CPU_MOE_NUMA_INTERLEAVE:-1}" \
OMP_PROC_BIND="${OMP_PROC_BIND:-spread}" \
PYTHONPATH="$REPO_ROOT" \
"$PYTHON" -m src.models.deepseek_v4.moe_server \
  --ckpt-path "$CKPT_PATH" \
  --config "$CONFIG" \
  --shm-name "$SHM_NAME" \
  --omp-threads "$SERVER_OMP_THREADS" \
  > "$SERVER_LOG" 2>&1 &
server_pid=$!

cleanup() {
  PYTHONPATH="$REPO_ROOT" "$PYTHON" - <<PY || true
from src.components.moe.ipc import CPUMoESharedMemory
try:
    shm = CPUMoESharedMemory('$SHM_NAME', 7168, 6, create=False)
    req, resp, layer, _stop = shm.read_header()
    shm.write_header(req, resp, layer, 1)
    shm.close()
except FileNotFoundError:
    pass
PY
  wait "$server_pid" 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 1 "$WAIT_SECONDS"); do
  if grep -q "cpu_moe_server native loop" "$SERVER_LOG" 2>/dev/null; then
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    cat "$SERVER_LOG"
    exit 1
  fi
  sleep 1
done

if ! grep -q "cpu_moe_server native loop" "$SERVER_LOG" 2>/dev/null; then
  echo "server did not enter native loop"
  cat "$SERVER_LOG"
  exit 1
fi

set +e
DEEPSEEK_CPU_MOE_EXTERNAL_SERVER=1 \
DEEPSEEK_INT8_IMPL=cuda_ext \
DEEPSEEK_MOE_ASYNC_ALLREDUCE=1 \
DEEPSEEK_CPU_MOE_EXTERNAL_PREFILL_LOCAL="${DEEPSEEK_CPU_MOE_EXTERNAL_PREFILL_LOCAL:-1}" \
DEEPSEEK_GPU_PREFILL_MOE="${DEEPSEEK_GPU_PREFILL_MOE:-0}" \
DEEPSEEK_GPU_PREFILL_MOE_GROUPED_GEMM="${DEEPSEEK_GPU_PREFILL_MOE_GROUPED_GEMM:-0}" \
DEEPSEEK_GPU_PREFILL_MOE_PREFETCH_BEFORE_FFN="${DEEPSEEK_GPU_PREFILL_MOE_PREFETCH_BEFORE_FFN:-0}" \
DEEPSEEK_SHARED_EXPERT_INT8=1 \
DEEPSEEK_FLASHINFER_STYLE_ATTN_CUDA=1 \
DEEPSEEK_FUSED_C4_INDEXER_CUDA="${DEEPSEEK_FUSED_C4_INDEXER_CUDA:-1}" \
DEEPSEEK_HC_PRE_CUDA="${DEEPSEEK_HC_PRE_CUDA:-1}" \
PYTHONPATH="$REPO_ROOT" \
"$TORCHRUN" --master-port "$MASTER_PORT" --nproc-per-node 4 --module src.cli.generate \
  --ckpt-path "$CKPT_PATH" \
  --config "$CONFIG" \
  --input-file "$INPUT_FILE" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --temperature 0 \
  --routed-experts-device cpu \
  > "$CLIENT_LOG" 2>&1

client_rc=$?
set -e
echo "--- client timing ---"
grep -E 'generate time:|prefill time:|decode time:|^Completion:' "$CLIENT_LOG" || true
echo "--- server tail ---"
tail -8 "$SERVER_LOG" || true
exit "$client_rc"
