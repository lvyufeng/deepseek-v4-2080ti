#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
TMP_ROOT="${POCKETLLM_TMP_DIR:-$REPO_ROOT/.tmp}"
mkdir -p "$TMP_ROOT"

LOCK_FILE="${LOCK_FILE:-$TMP_ROOT/dspark_parity_bench.lock}"
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "another DSpark parity benchmark is running: $LOCK_FILE" >&2
  exit 1
fi

CKPT="${CKPT:-/mnt/data3/DeepSeek-V4-Flash-0731}"
CONFIG="${CONFIG:-$REPO_ROOT/configs/config_fp4_active.json}"
CPP_BIN="${CPP_BIN:-$REPO_ROOT/cpp_engine/build/tests/bench_dspark_decode}"
PYTHON="${PYTHON:-python}"
TORCHRUN="${TORCHRUN:-torchrun}"
DECODE_TOKENS="${DECODE_TOKENS:-16}"
LONG_PROMPT_TOKENS="${LONG_PROMPT_TOKENS:-256}"
REPEATS="${REPEATS:-3}"
MASTER_PORT="${MASTER_PORT:-29973}"
NCCL_ID="${NCCL_ID:-$TMP_ROOT/dspark_parity_nccl.id}"
FIXTURES="${FIXTURES:-$TMP_ROOT/dspark_parity_fixtures.tsv}"
CPP_LOG_PREFIX="${CPP_LOG_PREFIX:-$TMP_ROOT/dspark_parity_cpp_rank}"
PYTORCH_LOG="${PYTORCH_LOG:-$TMP_ROOT/dspark_parity_pytorch.log}"
SUMMARY="${SUMMARY:-$TMP_ROOT/dspark_parity_summary.json}"
# Prefill top-k per fixture, so a first-token mismatch is reported as a near-tie
# or a real split instead of a bare MISMATCH@0. Costs one extra prefill per
# fixture per runtime; set to 0 to skip.
TOPK_DIAG="${TOPK_DIAG:-5}"

if [[ ! -d "$CKPT" ]]; then
  echo "checkpoint not found: $CKPT" >&2
  exit 1
fi
if [[ ! -x "$CPP_BIN" ]]; then
  echo "C++ benchmark not found: $CPP_BIN" >&2
  exit 1
fi
if [[ "$(basename "${CONDA_PREFIX:-}")" != "deepseek" ]]; then
  echo "run this benchmark from the deepseek conda environment" >&2
  exit 1
fi

CKPT="$CKPT" FIXTURES="$FIXTURES" LONG_PROMPT_TOKENS="$LONG_PROMPT_TOKENS" "$PYTHON" - <<'PY'
import os
from pathlib import Path
from transformers import AutoTokenizer
from src.encoding.deepseek_v4 import encode_messages

ckpt = os.environ["CKPT"]
out = Path(os.environ["FIXTURES"])
tok = AutoTokenizer.from_pretrained(ckpt)
short_text = "请用一句话解释为什么推理基准必须使用真实输入。"
long_unit = (
    "DeepSeek V4 Flash runs on four RTX 2080 Ti GPUs. "
    "The benchmark compares plain decode with speculative decode, keeps exact greedy routing, "
    "and runs short and long full-network cases serially. 请记住这些约束。\n"
)
long_text = long_unit
long_target = int(os.environ["LONG_PROMPT_TOKENS"])
while len(tok.encode(encode_messages([{"role": "user", "content": long_text}], thinking_mode="chat"))) < long_target:
    long_text += long_unit
long_text += "请用三句话总结以上约束。"

def encode(text):
    wire = encode_messages([{"role": "user", "content": text}], thinking_mode="chat")
    return tok.encode(wire)

fixtures = [
    ("raw_cyclic", [16, 18] * 6),
    ("real_short", encode(short_text)),
    ("real_long", encode(long_text)),
]
out.write_text("".join(f"{name}\t{','.join(map(str, ids))}\n" for name, ids in fixtures))
for name, ids in fixtures:
    print(f"fixture {name}: {len(ids)} tokens")
PY

rm -f "$NCCL_ID"

echo "Starting C++ TP=4 benchmark..."
rm -f "${CPP_LOG_PREFIX}"{0,1,2,3}.log
pids=()
for rank in 0 1 2 3; do
  echo "  Starting rank $rank..."
  POCKETLLM_BENCH_MODE=dspark_suite \
  POCKETLLM_CPP_BATCHED_VERIFY=0 \
  POCKETLLM_CPP_TOPK_DIAG="$TOPK_DIAG" \
  POCKETLLM_CPP_NCCL_ID_WAIT_ATTEMPTS=18000 \
    "$CPP_BIN" "$CKPT" "$FIXTURES" "$DECODE_TOKENS" "$REPEATS" 43 4 "$rank" "$NCCL_ID" \
      > "${CPP_LOG_PREFIX}${rank}.log" 2>&1 &
  pids+=("$!")
done

echo "  All ranks started, waiting for completion..."
status=0
for pid in "${pids[@]}"; do
  wait "$pid" || status=$?
done
if [[ "$status" -ne 0 ]]; then
  echo "C++ benchmark failed; rank logs: ${CPP_LOG_PREFIX}{0,1,2,3}.log" >&2
  exit "$status"
fi

# PyTorch and C++ runs are deliberately serial. The Python runtime keeps routed
# experts on CPU, matching its validated 4x22GB full-model configuration.
MASTER_PORT="$MASTER_PORT" \
DEEPSEEK_SYNC_TIMINGS=1 \
DEEPSEEK_GPU_MOE_DECODE_ACTIVE=1 \
DEEPSEEK_GPU_MOE_MULTI_TOKEN_FP4=1 \
  "$TORCHRUN" --standalone --nproc-per-node=4 \
  "$REPO_ROOT/tests/bench_dspark_cpp_pytorch.py" \
  --ckpt "$CKPT" --config "$CONFIG" --fixtures "$FIXTURES" \
  --decode-tokens "$DECODE_TOKENS" --repeats "$REPEATS" \
  > "$PYTORCH_LOG" 2>&1

# Same prefill top-k as the C++ ranks, appended to the log the summarizer reads.
if [[ "$TOPK_DIAG" -gt 0 ]]; then
  MASTER_PORT="$MASTER_PORT" \
  DEEPSEEK_GPU_MOE_DECODE_ACTIVE=1 \
  DEEPSEEK_GPU_MOE_MULTI_TOKEN_FP4=1 \
    "$TORCHRUN" --standalone --nproc-per-node=4 \
    "$REPO_ROOT/tests/debug_prefill_topk.py" \
    --ckpt "$CKPT" --config "$CONFIG" --fixtures "$FIXTURES" \
    --topk "$TOPK_DIAG" \
    >> "$PYTORCH_LOG" 2>&1
fi

"$PYTHON" "$REPO_ROOT/tests/summarize_dspark_parity_bench.py" \
  --cpp-log "${CPP_LOG_PREFIX}0.log" --pytorch-log "$PYTORCH_LOG" --output "$SUMMARY"
