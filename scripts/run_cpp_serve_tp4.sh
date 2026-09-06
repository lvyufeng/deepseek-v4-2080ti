#!/bin/bash
# Launch the cpp_engine OpenAI-compatible server on 4×GPU TP=4.
# Rank 0 listens on $PORT (default 8000); ranks 1-3 are NCCL workers.
set -e

CKPT="${CKPT:-/mnt/data1/modelscope/deepseek-ai/DeepSeek-V4-Flash}"
PORT="${PORT:-8000}"
NCCL_ID="${NCCL_ID:-/tmp/pocketllm_cpp_serve_nccl.id}"
MAX_CONTEXT="${MAX_CONTEXT:-8192}"
PYTHON="${PYTHON:-python}"
SIDECAR="${SIDECAR:-/mnt/data1/dsv4_inference/src/server/cpp_sidecar.py}"
BIN="${BIN:-/mnt/data1/dsv4_inference/build/cpp_engine/pocketllm_engine}"
LOG_DIR="${LOG_DIR:-/tmp}"
EXTRA_ENV="${EXTRA_ENV:-}"

rm -f "$NCCL_ID"

COMMON="--serve --ckpt $CKPT --tp-world 4 --nccl-id-path $NCCL_ID --smoke-layers 43 --max-context $MAX_CONTEXT --python $PYTHON --sidecar $SIDECAR --port $PORT"

for rank in 0 1 2 3; do
  eval "$EXTRA_ENV CUDA_VISIBLE_DEVICES=$rank $BIN $COMMON --tp-rank $rank --device 0 > $LOG_DIR/pocketllm_cpp_serve_rank${rank}.log 2>&1 &"
done

echo "started 4 ranks; rank 0 PID list:" >&2
jobs -p
echo "log dir: $LOG_DIR/pocketllm_cpp_serve_rank{0,1,2,3}.log" >&2
echo "tail logs with: tail -F $LOG_DIR/pocketllm_cpp_serve_rank0.log" >&2
wait
