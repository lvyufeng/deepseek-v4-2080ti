#!/usr/bin/env bash
# Configure, build and test the Ascend backend.
#
#   scripts/build_ascend.sh            # configure + build + test
#   scripts/build_ascend.sh build      # configure + build only
#
# This exists to hide one environment quirk that fails confusingly: without CANN's
# set_env.sh an ACL binary hangs before aclInit returns and prints nothing at all.
# It looks like broken hardware; it is a missing ASCEND_OPP_PATH.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENGINE_DIR="${REPO_ROOT}/cpp_engine"
BUILD_DIR="${ENGINE_DIR}/build-ascend"
ASCEND_TOOLKIT_HOME="${ASCEND_TOOLKIT_HOME:-/usr/local/Ascend/cann-9.0.0}"

# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/ascend_env.sh"

# The nested AscendC ExternalProject builds select a GCC 12 toolchain while this
# image provides GCC 11's C++ headers. Environment inheritance is the reliable
# way to make every nested compiler invocation see the installed headers.
GCC_CXX_INCLUDE="/usr/include/c++/11:/usr/include/aarch64-linux-gnu/c++/11"
export CPLUS_INCLUDE_PATH="${GCC_CXX_INCLUDE}${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}"

cmake -S "${ENGINE_DIR}" -B "${BUILD_DIR}" \
    -DPOCKET_BACKEND=ascend \
    -DASCEND_TOOLKIT_HOME="${ASCEND_TOOLKIT_HOME}"

cmake --build "${BUILD_DIR}" -j"$(nproc)" --target \
    check_layering \
    test_device_runtime \
    test_qwen_config \
    test_qwen_bf16_checkpoint \
    test_qwen_ascend_norm_gamma \
    test_qwen_ascend_ops \
    test_qwen_ascend_group_b \
    test_qwen_gqa_mmad_qk_probe \
    test_qwen_gqa_decode_mmad_integrated \
    pocketllm_engine \
    test_tp_comm_smoke

# The MMAD probe is an isolated hardware experiment, not a production attention
# path. Build it explicitly so stale generated launcher headers cannot mask a
# source or ABI error.

if [ "${1:-test}" = "build" ]; then
    echo "build_ascend: build only, skipping tests"
    exit 0
fi

status=0

# Single-process tests. Binaries land in different directories depending on
# whether the target sets RUNTIME_OUTPUT_DIRECTORY, so search rather than assume.
for name in test_device_runtime test_qwen_config test_qwen_bf16_checkpoint \
            test_qwen_ascend_norm_gamma test_qwen_ascend_ops \
            test_qwen_ascend_group_b test_qwen_gqa_decode_mmad_integrated; do
    binary="$(find "${BUILD_DIR}" -name "${name}" -type f -perm -u+x | head -1)"
    if [ -z "${binary}" ]; then
        echo "build_ascend: ${name} not built"
        status=1
        continue
    fi
    if ! "${binary}"; then
        echo "build_ascend: ${name} FAILED"
        status=1
    fi
done

# Collectives need one process per rank: HcclCommInitAll (single process, several
# devices) is unusable on this CANN release, so there is no in-process form to
# fall back to. 4 ranks is the tensor-parallel width Qwen3.8-27B requires.
smoke="$(find "${BUILD_DIR}" -name test_tp_comm_smoke -type f -perm -u+x | head -1)"
if [ -n "${smoke}" ]; then
    id_path="$(mktemp -u /tmp/pocket_tp_id.XXXXXX)"
    # Rank 0 publishes the rendezvous id here; a stale file would make every rank
    # wait on peers from a previous run, so start from a path that cannot exist.
    rm -f "${id_path}"
    pids=()
    for rank in 0 1 2 3; do
        "${smoke}" --world 4 --rank "${rank}" --device "${rank}" \
            --id-path "${id_path}" &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do
        if ! wait "${pid}"; then
            echo "build_ascend: test_tp_comm_smoke FAILED"
            status=1
        fi
    done
    rm -f "${id_path}"
else
    echo "build_ascend: test_tp_comm_smoke not built (no libhccl?)"
    status=1
fi

if [ "${status}" -eq 0 ]; then
    echo "build_ascend: all tests passed"
fi
exit "${status}"
