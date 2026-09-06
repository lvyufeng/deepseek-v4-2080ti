#!/usr/bin/env bash
# TP4 inference test for Qwen on Ascend 910A
#
# Usage: scripts/run_qwen_ascend_tp4.sh [prompt] [max_new_tokens]
set -euo pipefail

MODEL="${QWEN_MODEL:-/mnt/data1/modelscope/Qwen/Qwen3.8-27B}"
PROMPT="${1:-你好，请介绍一下你自己。}"
MAX_NEW="${2:-32}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

cd "$REPO"
source scripts/ascend_env.sh

# TP4 means 4 NPU devices
export HCCL_WHITELIST_DISABLE=1

echo "=== Qwen Ascend TP4 Inference ==="
echo "Model: $MODEL"
echo "Prompt: $PROMPT"
echo "Max tokens: $MAX_NEW"
echo ""

# Run each rank in background, rank 0 will print output
for rank in 0 1 2 3; do
    ASCEND_RT_VISIBLE_DEVICES=$rank \
    cpp_engine/build-ascend/pocketllm_engine \
        --model "$MODEL" \
        --prompt "$PROMPT" \
        --max-new-tokens "$MAX_NEW" \
        --tp-world 4 \
        --tp-rank $rank \
        --device $rank \
        --nccl-id-path /tmp/qwen_hccl_id.$$ &
done

# Wait for all ranks to complete
wait
rm -f /tmp/qwen_hccl_id.$$
