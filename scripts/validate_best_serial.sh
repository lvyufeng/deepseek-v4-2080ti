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
DEFAULT_CKPT_PATH="$REPO_ROOT/checkpoints/DeepSeek-V4-Flash-w8a8"
CKPT_PATH="${CKPT_PATH:-$DEFAULT_CKPT_PATH}"

if [[ ! -e "$CKPT_PATH" ]]; then
  echo "checkpoint not found: $CKPT_PATH" >&2
  echo "Set CKPT_PATH or place the checkpoint under $DEFAULT_CKPT_PATH" >&2
  exit 1
fi
BEST_SCRIPT="$REPO_ROOT/scripts/run_best_external_cpu_moe.sh"
SHORT_INPUT_FILE="${SHORT_INPUT_FILE:-$REPO_ROOT/tests/fixtures/smoke_input.txt}"
LONG_INPUT_FILE="${LONG_INPUT_FILE:-$TMP_ROOT/pocketllm_long_input.txt}"
SHORT_MAX_NEW_TOKENS="${SHORT_MAX_NEW_TOKENS:-8}"
LONG_MAX_NEW_TOKENS="${LONG_MAX_NEW_TOKENS:-32}"
SHORT_MASTER_PORT="${SHORT_MASTER_PORT:-29682}"
LONG_MASTER_PORT="${LONG_MASTER_PORT:-29683}"

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
text = base
while len(tok.encode(text)) < 2048:
    text += base
text += "\n请用三句话总结以上内容。"
with open("$TMP_ROOT/pocketllm_long_input.txt", "w") as f:
    f.write(text)
print(f"wrote $TMP_ROOT/pocketllm_long_input.txt tokens={len(tok.encode(text))}", flush=True)
PY
fi

echo "=== short sequence correctness/performance ==="
DEEPSEEK_SKIP_BENCHMARK_LOCK=1 \
MASTER_PORT="$SHORT_MASTER_PORT" \
SERVER_LOG="$TMP_ROOT/pocketllm_validate_short_server.log" \
CLIENT_LOG="$TMP_ROOT/pocketllm_validate_short_client.log" \
INPUT_FILE="$SHORT_INPUT_FILE" \
MAX_NEW_TOKENS="$SHORT_MAX_NEW_TOKENS" \
"$BEST_SCRIPT"

echo "=== long sequence correctness/performance ==="
DEEPSEEK_SKIP_BENCHMARK_LOCK=1 \
MASTER_PORT="$LONG_MASTER_PORT" \
SERVER_LOG="$TMP_ROOT/pocketllm_validate_long_server.log" \
CLIENT_LOG="$TMP_ROOT/pocketllm_validate_long_client.log" \
INPUT_FILE="$LONG_INPUT_FILE" \
MAX_NEW_TOKENS="$LONG_MAX_NEW_TOKENS" \
"$BEST_SCRIPT"
