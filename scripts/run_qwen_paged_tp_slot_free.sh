#!/bin/bash
# Launch the TP paged slot-free regression test.
#
# TP is the only place the worker-side block leak is observable. Rank 0's pool
# returns to full after every free_slot (it releases its own blocks directly),
# so rank 0 alone can never prove a worker leaked. The decisive assertion lives
# on rank 1: after it receives Shutdown it checks its OWN pool and exits
# non-zero if FreeSlot was ignored. So this launcher must wait on BOTH ranks and
# fail if either does.
#
# TP2 suffices: the fixture's 4 KV heads split to 2 per rank (TP4 would give 0
# and paging breaks).
#
# The test builds its own fixture checkpoint (deterministic path under /tmp),
# so it needs no checkpoint argument.

NCCL_ID="/tmp/qwen_paged_tp_slot_free_nccl_$(date +%s)"
TEST_BIN="$(dirname "$0")/../cpp_engine/build/tests/test_qwen_paged_tp_slot_free"

echo "Launching TP2 paged slot-free regression"
echo "NCCL ID: $NCCL_ID"

# Rank 0 in the foreground (drives the loop; its own output is the test log).
CUDA_VISIBLE_DEVICES=0 "$TEST_BIN" \
    --tp-world 2 \
    --tp-rank 0 \
    --device 0 \
    --nccl-id "$NCCL_ID" &
PID0=$!

# Rank 1 in the background. Its verdict is the worker-side assertion; capture it
# to a log and read its exit code.
CUDA_VISIBLE_DEVICES=1 "$TEST_BIN" \
    --tp-world 2 \
    --tp-rank 1 \
    --device 0 \
    --nccl-id "$NCCL_ID" >"/tmp/qwen_paged_tp_slot_free_rank1.log" 2>&1 &
PID1=$!

wait "$PID0"
STATUS0=$?
wait "$PID1"
STATUS1=$?

rm -f "$NCCL_ID"

echo "----- rank 1 (worker) output -----"
cat "/tmp/qwen_paged_tp_slot_free_rank1.log" 2>/dev/null

echo "rank0=$STATUS0 rank1=$STATUS1"
if [ "$STATUS0" -ne 0 ] || [ "$STATUS1" -ne 0 ]; then
    exit 1
fi
exit 0
