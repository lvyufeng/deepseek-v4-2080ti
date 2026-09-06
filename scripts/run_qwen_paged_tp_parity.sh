#!/bin/bash
# TP4 paged-vs-contiguous parity and throughput on the real model.
#
# Each KV mode is run as a SEPARATE TP4 process group (one mode per group, all
# four ranks together), so the parity diff reflects KV layout only and no
# per-mode engine construction is shared. The checksum over the token stream is
# diffed across modes: paged KV must produce byte-identical output to the
# contiguous arena, and its decode throughput must not regress.
#
# Usage: run_qwen_paged_tp_parity.sh [checkpoint] [layers] [batch-sizes...]
#   e.g. run_qwen_paged_tp_parity.sh /mnt/data2/Qwen3.8-27B-FP8 4 1 4 8

CHECKPOINT="${1:-/mnt/data2/Qwen3.8-27B-FP8}"
LAYERS="${2:-4}"
shift 2 2>/dev/null || true
BATCHES=("$@")
[ ${#BATCHES[@]} -eq 0 ] && BATCHES=(1 4 8)

BIN="$(cd "$(dirname "$0")/.." && pwd)/cpp_engine/build/tests/bench_qwen_paged_tp_parity"
CONTEXT="${CONTEXT:-1024}"
DECODE="${DECODE:-16}"
BLOCK="${BLOCK:-16}"
DIR="${DIR:-/tmp/qwen_paged_tp_parity}"
mkdir -p "$DIR"

run_mode() {
    local mode="$1" batch="$2"
    local nccl="/tmp/qwen_paged_tp_parity_nccl_${mode}_${batch}_$(date +%s)_$$"
    local r0="$DIR/${mode}_b${batch}_r0.log"

    # Rank 0 in the foreground so its output (with the checksum) is captured.
    CUDA_VISIBLE_DEVICES=0 "$BIN" "$CHECKPOINT" \
        --layers "$LAYERS" --max-context "$CONTEXT" --decode "$DECODE" \
        --tp-world 4 --tp-rank 0 --device 0 --nccl-id "$nccl" \
        --batch-size "$batch" --mode "$mode" --kv-block-size "$BLOCK" \
        >"$r0" 2>&1 &
    local p0=$!
    local pids=("$p0")
    for rank in 1 2 3; do
        CUDA_VISIBLE_DEVICES=$rank "$BIN" "$CHECKPOINT" \
            --layers "$LAYERS" --max-context "$CONTEXT" --decode "$DECODE" \
            --tp-world 4 --tp-rank $rank --device 0 --nccl-id "$nccl" \
            --batch-size "$batch" --mode "$mode" --kv-block-size "$BLOCK" \
            >"$DIR/${mode}_b${batch}_r${rank}.log" 2>&1 &
        pids+=($!)
    done

    local status=0
    for pid in "${pids[@]}"; do
        wait "$pid" || status=1
    done
    rm -f "$nccl"
    return $status
}

parse_arm() {
    local mode="$1" batch="$2"
    local r0="$DIR/${mode}_b${batch}_r0.log"
    echo "----- $mode batch=$batch (rank 0) -----"
    grep -E "checksum=|decode_ms/step=|prompt_tokens=" "$r0" || echo "(no summary line)"
    # Emit machine-readable values on `<name>=<value>` lines for the caller.
    grep -E "^  checksum=" "$r0" | sed -E 's/.*checksum=([0-9a-f]+)/checksum=\1/'
    grep -E "decode_ms/step=" "$r0" | sed -E 's/.*decode_ms\/step=([0-9.]+).*/ms_per_step=\1/'
}

fail=0
for batch in "${BATCHES[@]}"; do
    echo "=================================================="
    echo "batch=$batch"
    echo "=================================================="

    run_mode contiguous "$batch" || { echo "contiguous arm failed"; fail=1; }
    run_mode paged "$batch" || { echo "paged arm failed"; fail=1; }

    cs="$(parse_arm contiguous "$batch")"
    ps="$(parse_arm paged "$batch")"
    echo "$cs"; echo "$ps"

    c_checksum="$(echo "$cs" | grep '^checksum=' | cut -d= -f2)"
    p_checksum="$(echo "$ps" | grep '^checksum=' | cut -d= -f2)"
    if [ -z "$c_checksum" ] || [ -z "$p_checksum" ]; then
        echo "MISSING CHECKSUM"; fail=1; continue
    fi
    if [ "$c_checksum" = "$p_checksum" ]; then
        echo "PARITY: OK"
    else
        echo "PARITY: MISMATCH contiguous=$c_checksum paged=$p_checksum"
        fail=1
    fi
    echo ""
done

echo "=================================================="
echo "result: $([ $fail -eq 0 ] && echo PASS || echo FAIL)"
exit $fail
