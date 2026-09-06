#!/bin/bash
# Launch test_gguf_generate on 4×GPU TP=4 to validate GGUF Q2 TP decode.
# Rank 0 prints results; ranks 1-3 are silent (no PASS line, no stats).
set -e

CKPT="${CKPT:-/mnt/data1/dsv4_inference/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf}"
NCCL_ID="${NCCL_ID:-/tmp/pocketllm_gguf_tp4_nccl.id}"
BIN="${BIN:-/mnt/data1/dsv4_inference/build/cpp_engine/tests/test_gguf_generate}"
LOG_DIR="${LOG_DIR:-/tmp}"
MAX_NEW="${MAX_NEW:-8}"
SEED="${SEED:-1234}"
SEED_FILE="${SEED_FILE:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
if [ "$#" -gt 0 ]; then
  EXTRA_ARGS="$EXTRA_ARGS $*"
fi
EXTRA_ENV="${EXTRA_ENV:-}"

rm -f "$NCCL_ID"

if [ -n "$SEED_FILE" ]; then
  SEED_ARGS="--seed-file $SEED_FILE"
else
  SEED_ARGS="$SEED"
fi

for rank in 0 1 2 3; do
  eval "$EXTRA_ENV CUDA_VISIBLE_DEVICES=$rank $BIN $CKPT $MAX_NEW $SEED_ARGS \
        --tp-world 4 --tp-rank $rank --device 0 --nccl-id-path $NCCL_ID $EXTRA_ARGS \
        > $LOG_DIR/pocketllm_gguf_tp4_rank${rank}.log 2>&1 &"
done

echo "started 4 ranks; PIDs:" >&2
jobs -p
wait
echo "==== rank 0 log ====" >&2
cat $LOG_DIR/pocketllm_gguf_tp4_rank0.log
