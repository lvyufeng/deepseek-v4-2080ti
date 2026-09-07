# Enforces the core/engine/backends layering.
#
# The rule: shared layers must not depend on a vendor SDK. Public headers under
# include/ and the device-agnostic sources under core/ have to compile for any
# accelerator, so a <cuda_runtime.h> or <acl/acl.h> reaching them silently ties
# the shared code to one vendor. Backends are allowed, and expected, to include
# whatever their SDK needs.
#
# backends/api/ is checked too, and is the strictest case: it declares the
# contract both vendors implement, so a vendor type reaching it would defeat the
# whole seam.
#
# engine/ is checked as well, and gets a second, stricter test: its sources must
# not name a vendor runtime entry point either. An include check alone would not
# catch a stray cudaMalloc that compiles only because some other header pulled the
# SDK in transitively, and that is exactly the regression this seam exists to
# prevent. Kernel launches still live in backends/<vendor>/kernels, which is not
# checked.
#
# Run with: cmake --build <dir> --target check_layering

if(NOT DEFINED POCKET_ENGINE_DIR)
    message(FATAL_ERROR "POCKET_ENGINE_DIR must be set")
endif()

set(vendor_patterns
    "cuda_runtime"
    "cuda_fp16"
    "cuda_bf16"
    "cuda_fp8"
    "curand"
    "cublas"
    "nccl.h"
    "nvToolsExt"
    "nvtx3"
    "acl/acl"
    "hccl"
    "aclnn"
)

set(violations "")

file(GLOB_RECURSE guarded_files
    "${POCKET_ENGINE_DIR}/include/*.hpp"
    "${POCKET_ENGINE_DIR}/include/*.h"
    "${POCKET_ENGINE_DIR}/core/*.cpp"
    "${POCKET_ENGINE_DIR}/backends/api/*.hpp"
    "${POCKET_ENGINE_DIR}/engine/*.cpp"
)

foreach(file ${guarded_files})
    file(STRINGS "${file}" include_lines REGEX "^[ \t]*#[ \t]*include")
    foreach(line ${include_lines})
        foreach(pattern ${vendor_patterns})
            string(FIND "${line}" "${pattern}" found)
            if(NOT found EQUAL -1)
                file(RELATIVE_PATH rel "${POCKET_ENGINE_DIR}" "${file}")
                list(APPEND violations "${rel}: ${line}")
            endif()
        endforeach()
    endforeach()
endforeach()

if(violations)
    list(REMOVE_DUPLICATES violations)
    string(REPLACE ";" "\n  " pretty "${violations}")
    message(FATAL_ERROR
        "Vendor SDK headers leaked into a device-agnostic layer:\n  ${pretty}\n"
        "Move the vendor dependency into backends/<vendor>/ and expose an "
        "opaque handle instead (see include/sampler_ops.hpp for the pattern).")
endif()

# Vendor runtime entry points the shared layers must call through
# backends/api/device_runtime.hpp instead. Matched as whole words so an error
# string or a comment mentioning the old name is not a violation; the patterns
# below target call syntax and type names.
set(vendor_symbols
    "cudaMalloc" "cudaFree" "cudaMallocHost" "cudaFreeHost" "cudaHostAlloc"
    "cudaHostRegister" "cudaMemcpy" "cudaMemcpyAsync" "cudaMemcpy2D"
    "cudaMemset" "cudaMemsetAsync" "cudaMemGetInfo"
    "cudaStreamCreate" "cudaStreamCreateWithFlags" "cudaStreamDestroy"
    "cudaStreamSynchronize" "cudaStreamWaitEvent"
    "cudaEventCreate" "cudaEventCreateWithFlags" "cudaEventDestroy"
    "cudaEventRecord" "cudaEventSynchronize" "cudaEventElapsedTime"
    "cudaDeviceSynchronize" "cudaSetDevice" "cudaGetDevice"
    "cudaGetLastError" "cudaGetErrorString" "cudaStream_t" "cudaEvent_t"
    "cudaError_t" "cudaSuccess"
    "nvtxRangePush" "nvtxRangePushA" "nvtxRangePop"
    "aclrtMalloc" "aclrtFree" "aclrtMemcpy" "aclrtMemset"
    "aclrtCreateStream" "aclrtCreateEvent" "aclrtSynchronizeDevice"
    "aclrtStream" "aclrtEvent" "aclError"
)

set(symbol_violations "")

foreach(file ${guarded_files})
    file(READ "${file}" contents)
    foreach(symbol ${vendor_symbols})
        # Call sites and declarations: the symbol followed by '(' or whitespace
        # then an identifier. Quoted mentions are skipped by requiring that the
        # character before the symbol is not a double quote.
        if(contents MATCHES "[^\"_A-Za-z0-9]${symbol}[ \t]*[(*&]")
            file(RELATIVE_PATH rel "${POCKET_ENGINE_DIR}" "${file}")
            list(APPEND symbol_violations "${rel}: ${symbol}")
        endif()
    endforeach()
endforeach()

if(symbol_violations)
    list(REMOVE_DUPLICATES symbol_violations)
    string(REPLACE ";" "\n  " pretty "${symbol_violations}")
    message(FATAL_ERROR
        "Vendor runtime calls in a device-agnostic layer:\n  ${pretty}\n"
        "Call the equivalent in backends/api/device_runtime.hpp instead, and add "
        "the backend implementation under backends/<vendor>/runtime/.")
endif()

list(LENGTH guarded_files checked)
message(STATUS
    "check_layering: ${checked} files clean of vendor SDK includes and runtime calls")
