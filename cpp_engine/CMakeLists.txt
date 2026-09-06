cmake_minimum_required(VERSION 3.18)

# Backend selection happens at configure time. Each build produces a binary for
# exactly one accelerator vendor, so no vendor dispatch remains in the hot path
# and an unused vendor SDK is never a build dependency.
set(POCKET_BACKEND "cuda" CACHE STRING "Device backend: cuda or ascend")
set_property(CACHE POCKET_BACKEND PROPERTY STRINGS cuda ascend)

if(POCKET_BACKEND STREQUAL "cuda")
    project(dsv4_cpp_engine LANGUAGES CXX CUDA)
elseif(POCKET_BACKEND STREQUAL "ascend")
    project(dsv4_cpp_engine LANGUAGES CXX)
else()
    message(FATAL_ERROR "Unsupported POCKET_BACKEND '${POCKET_BACKEND}' (expected cuda or ascend)")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Python bindings are optional so the native CLI and inspection tools keep their
# existing dependency footprint.  When enabled, pybind11 is discovered through
# the active Python environment and the module links the same selected backend.
option(POCKET_BUILD_PYTHON "Build the optional pocketllm_cpp Python module" OFF)

# The Python module is a shared object, so every static library it links must be
# position independent.  This is scoped to the Python build so the executables
# keep their existing code generation, including GOT-free direct addressing.
set(POCKET_STATIC_PIC ${POCKET_BUILD_PYTHON})

# Sources that touch no vendor SDK. Keeping this list free of vendor headers is
# what lets a second backend reuse the loaders, tokenizer and HTTP server
# unchanged; see check_layering below, which enforces it.
set(POCKET_CORE_SOURCES
    core/json_lite.cpp
    core/gguf_reader.cpp
    core/safetensors_reader.cpp
    core/safetensors_model.cpp
    core/weight_source.cpp
    core/model_config.cpp
    core/qwen_config.cpp
    core/qwen_block_pool.cpp
    core/qwen_block_table.cpp
    core/qwen_weight_map.cpp
    core/tokenizer.cpp
    core/tensor.cpp
    core/sampler.cpp
    core/cmd_channel.cpp
    core/python_sidecar.cpp
    core/openai_server.cpp
    third_party/httplib.cpp
)

# One engine implementation shared by every backend. These reach the device only
# through backends/api/device_runtime.hpp and the operator contracts in include/,
# so they compile unchanged for either backend; check_layering enforces it. What
# is still vendor-specific is the operator implementations they call.
set(POCKET_ENGINE_SOURCES
    engine/qwen_weights.cpp
    engine/qwen_target_head.cpp
    engine/qwen_engine.cpp
    engine/qwen_batch_scheduler.cpp
)

# Engines that still call CUDA kernels by name. A runtime guard cannot help here:
# a CUDA symbol referenced from any object in the link must resolve even when its
# branch is unreachable, so these are excluded from a non-CUDA target and
# engine/backend_unimplemented_ascend.cpp supplies throwing definitions for the
# entry points their remaining callers name.
set(POCKET_CUDA_ONLY_ENGINE_SOURCES
    engine/qwen_dspark.cpp
    engine/qwen_dflash2.cpp
    engine/dsv4_engine.cpp
    engine/dspark_engine.cpp
)

set(POCKET_BACKEND_SOURCES)
# The device runtime is kept apart from the rest of the backend sources so it can
# be linked without any kernels. That is what makes it testable on hardware whose
# kernels do not exist yet.
set(POCKET_RUNTIME_SOURCES)
set(POCKET_BACKEND_LIBS)
set(POCKET_BACKEND_INCLUDES)
set(POCKET_BACKEND_DEFINES)
# AscendC's generated device library is consumed by the engine, but must not be
# linked into pocket_runtime: the latter is intentionally kernel-free so the
# runtime conformance test can exercise ACL without any device binary.
set(POCKET_ASCEND_KERNEL_LIBS)
set(POCKET_ASCEND_KERNEL_INCLUDES)

if(POCKET_BACKEND STREQUAL "cuda")
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)
    if(NOT CMAKE_CUDA_ARCHITECTURES)
        # sm_75. The kernels under backends/cuda/kernels are hand-tuned for
        # Turing's lack of native BF16/FP8 units; do not raise this default
        # without re-tuning them.
        set(CMAKE_CUDA_ARCHITECTURES 75)
    endif()

    find_package(CUDAToolkit REQUIRED)

    # DSV4_HAVE_TP_COMM means "a collective library is linked", not "NCCL is
    # linked": the Ascend branch below sets the same flag from HCCL. The engine's
    # TP paths are guarded on the neutral flag so they compile on either backend.
    set(NCCL_ROOT "" CACHE PATH "Optional NCCL root directory")
    option(DSV4_REQUIRE_NCCL "Fail configuration when NCCL is unavailable" OFF)
    set(DSV4_HAVE_TP_COMM OFF)
    if(NCCL_ROOT)
        find_path(NCCL_INCLUDE_DIR nccl.h PATHS "${NCCL_ROOT}/include" NO_DEFAULT_PATH)
        find_library(NCCL_LIBRARY NAMES nccl libnccl.so.2 PATHS "${NCCL_ROOT}/lib" NO_DEFAULT_PATH)
    else()
        find_path(NCCL_INCLUDE_DIR nccl.h)
        find_library(NCCL_LIBRARY NAMES nccl libnccl.so.2)
    endif()
    if(NCCL_INCLUDE_DIR AND NCCL_LIBRARY)
        set(DSV4_HAVE_TP_COMM ON)
        message(STATUS "NCCL found: ${NCCL_LIBRARY}")
    else()
        if(DSV4_REQUIRE_NCCL)
            message(FATAL_ERROR "NCCL is required but was not found; set NCCL_ROOT")
        endif()
        message(STATUS "NCCL not found; distributed C++ TP helpers disabled")
    endif()

    list(APPEND POCKET_ENGINE_SOURCES ${POCKET_CUDA_ONLY_ENGINE_SOURCES})
    list(APPEND POCKET_RUNTIME_SOURCES backends/cuda/runtime/device_runtime_cuda.cpp)
    list(APPEND POCKET_BACKEND_SOURCES
        backends/cuda/collective/tp_comm.cpp
        backends/cuda/kernels/cuda_ops.cu
        backends/cuda/kernels/bf16_ops.cu
        backends/cuda/kernels/quant_gemm.cu
        backends/cuda/kernels/fp4_ops.cu
        backends/cuda/kernels/fp8_ops.cu
        backends/cuda/kernels/qwen_fp8_ops.cu
        backends/cuda/kernels/qwen_attention_ops.cu
        backends/cuda/kernels/qwen_half_ops.cu
        backends/cuda/kernels/qwen_nvfp4_ops.cu
        backends/cuda/kernels/qwen_gqa_optimized.cu
        backends/cuda/kernels/qwen_gdn_flashqla.cu
        backends/cuda/kernels/qwen_turboquant_ops.cu
        backends/cuda/kernels/qwen_int8_per_token_head_ops.cu
        backends/cuda/kernels/qwen_dspark_ops.cu
        backends/cuda/kernels/qwen_dflash2_ops.cu
        backends/cuda/kernels/qwen_sampler.cu
        backends/cuda/kernels/q2_ops.cu
        backends/cuda/kernels/iq1_ops.cu
        backends/cuda/kernels/rmsnorm_rope.cu
        backends/cuda/kernels/flashmemory_ops.cu
    )
    list(APPEND POCKET_BACKEND_LIBS CUDA::cudart CUDA::cublas CUDA::curand)
    if(TARGET CUDA::nvtx3)
        list(APPEND POCKET_BACKEND_LIBS CUDA::nvtx3)
    elseif(TARGET CUDA::nvToolsExt)
        list(APPEND POCKET_BACKEND_LIBS CUDA::nvToolsExt)
    else()
        message(FATAL_ERROR "NVTX is required but was not found in the CUDA toolkit")
    endif()
    if(DSV4_HAVE_TP_COMM)
        list(APPEND POCKET_BACKEND_INCLUDES ${NCCL_INCLUDE_DIR})
        list(APPEND POCKET_BACKEND_LIBS ${NCCL_LIBRARY})
        list(APPEND POCKET_BACKEND_DEFINES DSV4_HAVE_TP_COMM=1)
    endif()
else()
    # Ascend: ACL runtime, AscendC kernels and HCCL collectives under
    # backends/ascend. The dense-FP16 Qwen path is complete through the linked
    # engine; CUDA-only engines and quantized Qwen modes stay outside this build.
    set(DSV4_HAVE_TP_COMM OFF)

    # CANN layout: $ASCEND_TOOLKIT_HOME/<arch>-linux/{include,lib64}. The arch
    # directory name is not always the uname value (arm64-linux exists alongside
    # aarch64-linux), so probe rather than compose it.
    if(NOT ASCEND_TOOLKIT_HOME)
        if(DEFINED ENV{ASCEND_TOOLKIT_HOME})
            set(ASCEND_TOOLKIT_HOME "$ENV{ASCEND_TOOLKIT_HOME}")
        else()
            set(ASCEND_TOOLKIT_HOME "/usr/local/Ascend/cann-9.0.0")
        endif()
    endif()
    set(ASCEND_TOOLKIT_HOME "${ASCEND_TOOLKIT_HOME}" CACHE PATH "CANN toolkit root")

    set(ASCEND_ARCH_DIR "")
    foreach(candidate aarch64-linux arm64-linux x86_64-linux)
        if(EXISTS "${ASCEND_TOOLKIT_HOME}/${candidate}/include/acl/acl.h")
            set(ASCEND_ARCH_DIR "${ASCEND_TOOLKIT_HOME}/${candidate}")
            break()
        endif()
    endforeach()
    if(NOT ASCEND_ARCH_DIR)
        message(FATAL_ERROR
            "No CANN include tree found under ${ASCEND_TOOLKIT_HOME}; "
            "set -DASCEND_TOOLKIT_HOME=<cann root>")
    endif()

    find_library(ASCEND_ACL_LIBRARY NAMES ascendcl
        PATHS "${ASCEND_ARCH_DIR}/lib64" NO_DEFAULT_PATH)
    if(NOT ASCEND_ACL_LIBRARY)
        message(FATAL_ERROR "libascendcl not found under ${ASCEND_ARCH_DIR}/lib64")
    endif()
    message(STATUS "CANN found: ${ASCEND_ARCH_DIR}")

    # First-generation 910 (Short_SoC_version=Ascend910). CANN's own
    # host_config.cmake groups ascend910b with ascend910a here and ascend910b1..b4
    # separately, so this value selects the first-generation codegen path. Kernel
    # tiling is tuned for 30 AI cores / 32 MB L2 / 900 MHz cube; do not change
    # this without re-tuning.
    set(ASCEND_SOC_VERSION "ascend910b" CACHE STRING "AscendC target SoC version")

    # aclnn operator libraries. libnnopbase carries aclCreateTensor and the
    # aclOpExecutor type, so linking libopapi alone leaves those undefined even
    # though the aclnn* entry points themselves resolve.
    find_library(ASCEND_OPAPI_LIBRARY NAMES opapi
        PATHS "${ASCEND_ARCH_DIR}/lib64" NO_DEFAULT_PATH)
    find_library(ASCEND_NNOPBASE_LIBRARY NAMES nnopbase
        PATHS "${ASCEND_ARCH_DIR}/lib64" NO_DEFAULT_PATH)
    if(NOT ASCEND_OPAPI_LIBRARY OR NOT ASCEND_NNOPBASE_LIBRARY)
        message(FATAL_ERROR
            "libopapi / libnnopbase not found under ${ASCEND_ARCH_DIR}/lib64; "
            "the aclnn operator implementations need both")
    endif()

    # HCCL. Every init entry point in hccl_comm.h is declared
    # __attribute__((weak)), so omitting this library is not a link error: the
    # call goes to address zero and segfaults at run time. Treat a missing
    # libhccl as fatal rather than letting DSV4_HAVE_TP_COMM silently stay off.
    find_library(ASCEND_HCCL_LIBRARY NAMES hccl
        PATHS "${ASCEND_ARCH_DIR}/lib64" NO_DEFAULT_PATH)
    if(ASCEND_HCCL_LIBRARY)
        set(DSV4_HAVE_TP_COMM ON)
        message(STATUS "HCCL found: ${ASCEND_HCCL_LIBRARY}")
    else()
        message(STATUS "libhccl not found; tensor parallelism disabled on Ascend")
    endif()

    list(APPEND POCKET_RUNTIME_SOURCES backends/ascend/runtime/device_runtime_ascend.cpp)
    list(APPEND POCKET_BACKEND_INCLUDES "${ASCEND_ARCH_DIR}/include")
    list(APPEND POCKET_BACKEND_LIBS
        "${ASCEND_ACL_LIBRARY}"
        "${ASCEND_OPAPI_LIBRARY}"
        "${ASCEND_NNOPBASE_LIBRARY}")
    list(APPEND POCKET_BACKEND_DEFINES POCKET_BACKEND_ASCEND=1)

    # Stands in for POCKET_CUDA_ONLY_ENGINE_SOURCES, which is not built here.
    list(APPEND POCKET_ENGINE_SOURCES engine/backend_unimplemented_ascend.cpp)

    # aclnn-backed operators: matrix products, normalizations, elementwise
    # activations and the reshape-shaped ops. The recurrent and attention kernels
    # have no aclnn equivalent on this install and are hand-written AscendC below.
    list(APPEND POCKET_BACKEND_SOURCES backends/ascend/kernels/aclnn_ops.cpp)

    # Hand-written AscendC device code. These are the operators with no aclnn
    # equivalent: the gated-delta recurrence, its gates and normalizations, the
    # causal depthwise convolution, partial RoPE, KV-cache append and GQA
    # attention.
    set(POCKET_ASCEND_KERNEL_SOURCES
        backends/ascend/kernels/qwen_gated_delta_f16.cpp
        backends/ascend/kernels/qwen_linear_f16.cpp
        backends/ascend/kernels/qwen_attention_f16.cpp
        backends/ascend/kernels/qwen_argmax_f32.cpp
    )
    # ascendc.cmake reads these three variables by name out of the enclosing
    # scope and fails unhelpfully if any is unset, so set them here rather than
    # making every caller pass -D flags.
    set(SOC_VERSION "${ASCEND_SOC_VERSION}")
    set(RUN_MODE "npu")
    set(ASCEND_CANN_PACKAGE_PATH "${ASCEND_TOOLKIT_HOME}")
    # The nested ExternalProject builds spawned by ascendc.cmake use GCC 12's
    # compiler driver but need to find GCC 11's C++ headers (the ones installed
    # on this system). Pass them via ASCENDC_COMPILE_OPTIONS which ascendc.cmake
    # appends to every nested compile command.
    set(ASCENDC_COMPILE_OPTIONS "-isystem;/usr/include/c++/11;-isystem;/usr/include/aarch64-linux-gnu/c++/11")
    include(${ASCEND_TOOLKIT_HOME}/compiler/tikcpp/ascendc_kernel_cmake/ascendc.cmake)
    ascendc_library(pocket_ascend_kernels STATIC ${POCKET_ASCEND_KERNEL_SOURCES})

    # ascendc_library() emits one aclrtlaunch_<kernel>.h per
    # `extern "C" __global__ __aicore__` entry point into
    # ${CMAKE_BINARY_DIR}/include/<target>. The launcher below is the only file
    # that includes them, but the directory has to be on the include path of
    # whatever compiles it, which is dsv4_cpp_core.
    list(APPEND POCKET_BACKEND_SOURCES backends/ascend/kernels/qwen_ascend_ops_launch.cpp)
    list(APPEND POCKET_ASCEND_KERNEL_INCLUDES
        "${CMAKE_BINARY_DIR}/include/pocket_ascend_kernels")
    list(APPEND POCKET_ASCEND_KERNEL_LIBS pocket_ascend_kernels)

    if(DSV4_HAVE_TP_COMM)
        list(APPEND POCKET_BACKEND_SOURCES backends/ascend/collective/tp_comm.cpp)
        list(APPEND POCKET_BACKEND_LIBS "${ASCEND_HCCL_LIBRARY}")
        list(APPEND POCKET_BACKEND_DEFINES DSV4_HAVE_TP_COMM=1)
    endif()

    message(STATUS
        "POCKET_BACKEND=ascend builds the Qwen FP16 path: device runtime, HCCL "
        "collectives, the neutral operator set and dsv4_cpp_engine. Quantized "
        "weights and KV caches, external drafters, device sampling and the DSV4 "
        "engine are rejected at runtime, not implemented.")
endif()

# Device-agnostic core as its own library. Anything that links only this target
# is proof the layering holds: checkpoint loading, config parsing, tokenization
# and the Qwen weight map build and run with no vendor SDK present at all, which
# is what lets a checkpoint be audited on a host without the accelerator toolkit.
add_library(pocket_core STATIC ${POCKET_CORE_SOURCES})
set_target_properties(pocket_core PROPERTIES POSITION_INDEPENDENT_CODE ${POCKET_STATIC_PIC})
target_include_directories(pocket_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)

# The selected backend's device runtime, on its own. One vendor implementation is
# linked, chosen at configure time, so a call through this costs the same as
# calling the vendor API directly: no vtable, no dispatch.
add_library(pocket_runtime STATIC ${POCKET_RUNTIME_SOURCES})
set_target_properties(pocket_runtime PROPERTIES POSITION_INDEPENDENT_CODE ${POCKET_STATIC_PIC})
target_link_libraries(pocket_runtime PUBLIC pocket_core)
target_include_directories(pocket_runtime PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/backends/api
)
if(POCKET_BACKEND_INCLUDES)
    target_include_directories(pocket_runtime PRIVATE ${POCKET_BACKEND_INCLUDES})
endif()
if(POCKET_BACKEND_LIBS)
    target_link_libraries(pocket_runtime PUBLIC ${POCKET_BACKEND_LIBS})
endif()
if(POCKET_BACKEND_DEFINES)
    target_compile_definitions(pocket_runtime PUBLIC ${POCKET_BACKEND_DEFINES})
endif()

add_library(dsv4_cpp_core STATIC
    ${POCKET_ENGINE_SOURCES}
    ${POCKET_BACKEND_SOURCES}
)
set_target_properties(dsv4_cpp_core PROPERTIES POSITION_INDEPENDENT_CODE ${POCKET_STATIC_PIC})

target_link_libraries(dsv4_cpp_core PUBLIC pocket_core pocket_runtime)

target_include_directories(dsv4_cpp_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party
)
if(POCKET_BACKEND_INCLUDES)
    target_include_directories(dsv4_cpp_core PUBLIC ${POCKET_BACKEND_INCLUDES})
endif()
if(POCKET_BACKEND_LIBS)
    target_link_libraries(dsv4_cpp_core PUBLIC ${POCKET_BACKEND_LIBS})
endif()
if(POCKET_ASCEND_KERNEL_INCLUDES)
    target_include_directories(dsv4_cpp_core PUBLIC ${POCKET_ASCEND_KERNEL_INCLUDES})
endif()
if(POCKET_ASCEND_KERNEL_LIBS)
    target_link_libraries(dsv4_cpp_core PUBLIC ${POCKET_ASCEND_KERNEL_LIBS})
endif()
if(POCKET_BACKEND_DEFINES)
    target_compile_definitions(dsv4_cpp_core PUBLIC ${POCKET_BACKEND_DEFINES})
endif()
if(POCKET_BACKEND STREQUAL "cuda")
    set_target_properties(dsv4_cpp_core PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
else()
    # Keep legacy CUDA-heavy methods in discardable sections while the Ascend
    # executable uses only the dense FP16 Qwen path. This prevents unused GGUF,
    # DSpark and DFlash2 methods from imposing link-time CUDA dependencies; all
    # reachable unsupported paths still reject their formats at runtime.
    target_compile_options(dsv4_cpp_core PRIVATE -ffunction-sections -fdata-sections)
    target_link_options(dsv4_cpp_core INTERFACE -Wl,--gc-sections)
endif()

# Fail the build if a vendor SDK leaks into the shared layers. This is the
# structural guarantee behind "one engine, swappable backends": without it the
# boundary is a convention that erodes silently.
add_custom_target(check_layering
    COMMAND ${CMAKE_COMMAND}
        -DPOCKET_ENGINE_DIR=${CMAKE_CURRENT_SOURCE_DIR}
        -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/CheckLayering.cmake
    COMMENT "Checking that core/ and include/ pull in no vendor SDK headers"
)

add_executable(dsv4_cpp_engine engine/main.cpp)
target_link_libraries(dsv4_cpp_engine PRIVATE dsv4_cpp_core)

if(POCKET_BUILD_PYTHON)
    find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
    find_package(pybind11 CONFIG REQUIRED)
    # Do not enable pybind11's LTO/strip extras here.  The selected backend
    # contains CUDA device-link objects whose fatbin registration data must stay
    # visible while the shared extension is linked.
    pybind11_add_module(pocketllm_cpp NO_EXTRAS python/bindings.cpp)
    target_link_libraries(pocketllm_cpp PRIVATE dsv4_cpp_core)
    target_include_directories(pocketllm_cpp PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party
        ${CMAKE_CURRENT_SOURCE_DIR}/backends/api
    )
    if(POCKET_BACKEND_INCLUDES)
        target_include_directories(pocketllm_cpp PRIVATE ${POCKET_BACKEND_INCLUDES})
    endif()
    if(POCKET_ASCEND_KERNEL_INCLUDES)
        target_include_directories(pocketllm_cpp PRIVATE ${POCKET_ASCEND_KERNEL_INCLUDES})
    endif()
    if(POCKET_BACKEND_DEFINES)
        target_compile_definitions(pocketllm_cpp PRIVATE ${POCKET_BACKEND_DEFINES})
    endif()
    set_target_properties(pocketllm_cpp PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/python
    )
    message(STATUS "POCKET_BUILD_PYTHON=ON: building pocketllm_cpp")
endif()

add_executable(inspect_gguf tools/inspect_gguf.cpp)
target_link_libraries(inspect_gguf PRIVATE dsv4_cpp_core)
set_target_properties(inspect_gguf PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tools)

add_executable(inspect_safetensors tools/inspect_safetensors.cpp)
target_link_libraries(inspect_safetensors PRIVATE dsv4_cpp_core)
set_target_properties(inspect_safetensors PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tools)

# Links pocket_core only, so it builds and runs without a vendor SDK.
add_executable(qwen_audit tools/qwen_audit.cpp)
target_link_libraries(qwen_audit PRIVATE pocket_core)
set_target_properties(qwen_audit PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tools)

# Backend-agnostic conformance test for the device runtime. Links pocket_runtime
# only, with no model kernels, so runtime behavior is tested independently of
# the AscendC archive.
add_executable(test_device_runtime tests/test_device_runtime.cpp)
target_link_libraries(test_device_runtime PRIVATE pocket_runtime)

add_executable(test_qwen_bf16_checkpoint tests/test_qwen_bf16_checkpoint.cpp)
target_link_libraries(test_qwen_bf16_checkpoint PRIVATE pocket_core)
set_target_properties(test_qwen_bf16_checkpoint PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(minimax_gguf_tokenizer tools/minimax_gguf_tokenizer.cpp)
target_link_libraries(minimax_gguf_tokenizer PRIVATE dsv4_cpp_core)
set_target_properties(minimax_gguf_tokenizer PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tools)

add_executable(test_gguf_reader tests/test_gguf_reader.cpp)
target_link_libraries(test_gguf_reader PRIVATE dsv4_cpp_core)
set_target_properties(test_gguf_reader PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_gguf_iq1m_reader tests/test_gguf_iq1m_reader.cpp)
target_link_libraries(test_gguf_iq1m_reader PRIVATE dsv4_cpp_core)
set_target_properties(test_gguf_iq1m_reader PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_safetensors_reader tests/test_safetensors_reader.cpp)
target_link_libraries(test_safetensors_reader PRIVATE dsv4_cpp_core)
set_target_properties(test_safetensors_reader PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_flashmemory_scorer tests/test_flashmemory_scorer.cpp)
target_link_libraries(test_flashmemory_scorer PRIVATE dsv4_cpp_core)
set_target_properties(test_flashmemory_scorer PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_safetensors_model tests/test_safetensors_model.cpp)
target_link_libraries(test_safetensors_model PRIVATE dsv4_cpp_core)
set_target_properties(test_safetensors_model PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_weight_source tests/test_weight_source.cpp)
target_link_libraries(test_weight_source PRIVATE dsv4_cpp_core)
set_target_properties(test_weight_source PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_dspark_weight_load tests/test_dspark_weight_load.cpp)
target_link_libraries(test_dspark_weight_load PRIVATE dsv4_cpp_core)
set_target_properties(test_dspark_weight_load PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_perfect_draft tests/test_perfect_draft.cpp)
target_link_libraries(test_perfect_draft PRIVATE dsv4_cpp_core)
set_target_properties(test_perfect_draft PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_moe_fp4_determinism tests/test_moe_fp4_determinism.cpp)
target_link_libraries(test_moe_fp4_determinism PRIVATE dsv4_cpp_core)
set_target_properties(test_moe_fp4_determinism PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_dspark_attention_parity tests/test_dspark_attention_parity.cpp)
target_link_libraries(test_dspark_attention_parity PRIVATE dsv4_cpp_core)
set_target_properties(test_dspark_attention_parity PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_dspark_moe_parity tests/test_dspark_moe_parity.cpp)
target_link_libraries(test_dspark_moe_parity PRIVATE dsv4_cpp_core)
set_target_properties(test_dspark_moe_parity PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_dspark_draft tests/test_dspark_draft.cpp)
target_link_libraries(test_dspark_draft PRIVATE dsv4_cpp_core)
set_target_properties(test_dspark_draft PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_verify_overwrite_safety tests/test_verify_overwrite_safety.cpp)
target_link_libraries(test_verify_overwrite_safety PRIVATE dsv4_cpp_core)
set_target_properties(test_verify_overwrite_safety PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_speculative_scheduler tests/test_speculative_scheduler.cpp)
target_link_libraries(test_speculative_scheduler PRIVATE dsv4_cpp_core)
set_target_properties(test_speculative_scheduler PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(bench_dspark_decode tests/bench_dspark_decode.cpp)
target_link_libraries(bench_dspark_decode PRIVATE dsv4_cpp_core)
set_target_properties(bench_dspark_decode PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_batch_verify_transaction tests/test_batch_verify_transaction.cpp)
target_link_libraries(test_batch_verify_transaction PRIVATE dsv4_cpp_core)
set_target_properties(test_batch_verify_transaction PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_dspark_tp_consistency tests/test_dspark_tp_consistency.cpp)
target_link_libraries(test_dspark_tp_consistency PRIVATE dsv4_cpp_core)
set_target_properties(test_dspark_tp_consistency PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_dspark_draft_sensitivity tests/test_dspark_draft_sensitivity.cpp)
target_link_libraries(test_dspark_draft_sensitivity PRIVATE dsv4_cpp_core)
set_target_properties(test_dspark_draft_sensitivity PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_dspark_hidden_capture tests/test_dspark_hidden_capture.cpp)
target_link_libraries(test_dspark_hidden_capture PRIVATE dsv4_cpp_core)
set_target_properties(test_dspark_hidden_capture PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_dspark_head_parity tests/test_dspark_head_parity.cpp)
target_link_libraries(test_dspark_head_parity PRIVATE dsv4_cpp_core)
set_target_properties(test_dspark_head_parity PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_q2_matvec tests/test_q2_matvec.cpp)
target_link_libraries(test_q2_matvec PRIVATE dsv4_cpp_core)
set_target_properties(test_q2_matvec PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_gguf_dense_smoke tests/test_gguf_dense_smoke.cpp)
target_link_libraries(test_gguf_dense_smoke PRIVATE dsv4_cpp_core)
set_target_properties(test_gguf_dense_smoke PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_gguf_min_layer_smoke tests/test_gguf_min_layer_smoke.cpp)
target_link_libraries(test_gguf_min_layer_smoke PRIVATE dsv4_cpp_core)
set_target_properties(test_gguf_min_layer_smoke PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_gguf_attn_norm_wqa tests/test_gguf_attn_norm_wqa.cpp)
target_link_libraries(test_gguf_attn_norm_wqa PRIVATE dsv4_cpp_core)
set_target_properties(test_gguf_attn_norm_wqa PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_gguf_attn_q_path tests/test_gguf_attn_q_path.cpp)
target_link_libraries(test_gguf_attn_q_path PRIVATE dsv4_cpp_core)
set_target_properties(test_gguf_attn_q_path PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_gguf_attn_kv_path tests/test_gguf_attn_kv_path.cpp)
target_link_libraries(test_gguf_attn_kv_path PRIVATE dsv4_cpp_core)
set_target_properties(test_gguf_attn_kv_path PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_gguf_attn_full tests/test_gguf_attn_full.cpp)
target_link_libraries(test_gguf_attn_full PRIVATE dsv4_cpp_core)
set_target_properties(test_gguf_attn_full PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_gguf_shared_expert tests/test_gguf_shared_expert.cpp)
target_link_libraries(test_gguf_shared_expert PRIVATE dsv4_cpp_core)
set_target_properties(test_gguf_shared_expert PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_gguf_routed_expert tests/test_gguf_routed_expert.cpp)
target_link_libraries(test_gguf_routed_expert PRIVATE dsv4_cpp_core)
set_target_properties(test_gguf_routed_expert PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_gguf_routed_moe tests/test_gguf_routed_moe.cpp)
target_link_libraries(test_gguf_routed_moe PRIVATE dsv4_cpp_core)
set_target_properties(test_gguf_routed_moe PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_gguf_layer0_full tests/test_gguf_layer0_full.cpp)
target_link_libraries(test_gguf_layer0_full PRIVATE dsv4_cpp_core)
set_target_properties(test_gguf_layer0_full PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_gguf_full_forward tests/test_gguf_full_forward.cpp)
target_link_libraries(test_gguf_full_forward PRIVATE dsv4_cpp_core)
set_target_properties(test_gguf_full_forward PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_gguf_generate tests/test_gguf_generate.cpp)
target_link_libraries(test_gguf_generate PRIVATE dsv4_cpp_core)
set_target_properties(test_gguf_generate PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(probe_host_register tests/probe_host_register.cpp)
target_link_libraries(probe_host_register PRIVATE dsv4_cpp_core)
set_target_properties(probe_host_register PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_quant_gemm tests/test_quant_gemm.cpp)
target_link_libraries(test_quant_gemm PRIVATE dsv4_cpp_core)
set_target_properties(test_quant_gemm PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_fp4_matvec tests/test_fp4_matvec.cpp)
target_link_libraries(test_fp4_matvec PRIVATE dsv4_cpp_core)
set_target_properties(test_fp4_matvec PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_fp8_matvec tests/test_fp8_matvec.cpp)
target_link_libraries(test_fp8_matvec PRIVATE dsv4_cpp_core)
set_target_properties(test_fp8_matvec PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_rmsnorm tests/test_rmsnorm.cpp)
target_link_libraries(test_rmsnorm PRIVATE dsv4_cpp_core)
set_target_properties(test_rmsnorm PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_attention_projection tests/test_attention_projection.cpp)
target_link_libraries(test_attention_projection PRIVATE dsv4_cpp_core)
set_target_properties(test_attention_projection PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_moe_single_expert tests/test_moe_single_expert.cpp)
target_link_libraries(test_moe_single_expert PRIVATE dsv4_cpp_core)
set_target_properties(test_moe_single_expert PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_moe_grouped_fp4 tests/test_moe_grouped_fp4.cpp)
target_link_libraries(test_moe_grouped_fp4 PRIVATE dsv4_cpp_core)
set_target_properties(test_moe_grouped_fp4 PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_moe_multi_token_fp4 tests/test_moe_multi_token_fp4.cpp)
target_link_libraries(test_moe_multi_token_fp4 PRIVATE dsv4_cpp_core)
set_target_properties(test_moe_multi_token_fp4 PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_continuation_attention tests/test_continuation_attention.cpp)
target_link_libraries(test_continuation_attention PRIVATE dsv4_cpp_core)
set_target_properties(test_continuation_attention PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_moe_grouped_iq1 tests/test_moe_grouped_iq1.cpp)
target_link_libraries(test_moe_grouped_iq1 PRIVATE dsv4_cpp_core)
set_target_properties(test_moe_grouped_iq1 PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_moe_grouped_q2 tests/test_moe_grouped_q2.cpp)
target_link_libraries(test_moe_grouped_q2 PRIVATE dsv4_cpp_core)
set_target_properties(test_moe_grouped_q2 PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_embedding_head tests/test_embedding_head.cpp)
target_link_libraries(test_embedding_head PRIVATE dsv4_cpp_core)
set_target_properties(test_embedding_head PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_tokenizer tests/test_tokenizer.cpp)
target_link_libraries(test_tokenizer PRIVATE dsv4_cpp_core)
set_target_properties(test_tokenizer PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_gguf_tokenizer tests/test_gguf_tokenizer.cpp)
target_link_libraries(test_gguf_tokenizer PRIVATE dsv4_cpp_core)
set_target_properties(test_gguf_tokenizer PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_min_layer_smoke tests/test_min_layer_smoke.cpp)
target_link_libraries(test_min_layer_smoke PRIVATE dsv4_cpp_core)
set_target_properties(test_min_layer_smoke PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_tp_comm_smoke tests/test_tp_comm_smoke.cpp)
target_link_libraries(test_tp_comm_smoke PRIVATE dsv4_cpp_core)
set_target_properties(test_tp_comm_smoke PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

# Calls the operators through their neutral names in qwen_ops.hpp and checks them
# against double-precision host references, so it is meaningful on either backend.
add_executable(test_qwen_ascend_ops tests/test_qwen_ascend_ops.cpp)
target_link_libraries(test_qwen_ascend_ops PRIVATE dsv4_cpp_core)
set_target_properties(test_qwen_ascend_ops PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

# The twelve operators with no aclnn equivalent, i.e. the hand-written AscendC
# kernels. Same neutral-name discipline as test_qwen_ascend_ops; it is a separate
# binary because these are the kernels most likely to fault the device, and a fault
# there should not take the aclnn coverage with it.
add_executable(test_qwen_ascend_group_b tests/test_qwen_ascend_group_b.cpp)
target_link_libraries(test_qwen_ascend_group_b PRIVATE dsv4_cpp_core)
set_target_properties(test_qwen_ascend_group_b PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

# The (1 + weight) gamma fold is host arithmetic on the staged bytes, so this needs
# no device. It still links dsv4_cpp_core because the policy body is backend-gated.
add_executable(test_qwen_ascend_norm_gamma tests/test_qwen_ascend_norm_gamma.cpp)
target_link_libraries(test_qwen_ascend_norm_gamma PRIVATE dsv4_cpp_core)
set_target_properties(test_qwen_ascend_norm_gamma PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

# Config parsing is device-agnostic, so this test links pocket_core and runs on a
# host with no vendor SDK.
add_executable(test_qwen_config tests/test_qwen_config.cpp)
target_link_libraries(test_qwen_config PRIVATE pocket_core)
set_target_properties(test_qwen_config PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_qwen_dspark tests/test_qwen_dspark.cpp)
target_link_libraries(test_qwen_dspark PRIVATE dsv4_cpp_core)
set_target_properties(test_qwen_dspark PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_qwen_dflash2 tests/test_qwen_dflash2.cpp)
target_link_libraries(test_qwen_dflash2 PRIVATE dsv4_cpp_core)
set_target_properties(test_qwen_dflash2 PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_qwen_dflash2_parity tests/test_qwen_dflash2_parity.cpp)
target_link_libraries(test_qwen_dflash2_parity PRIVATE dsv4_cpp_core)
set_target_properties(test_qwen_dflash2_parity PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_qwen_dspark_ops tests/test_qwen_dspark_ops.cpp)
target_link_libraries(test_qwen_dspark_ops PRIVATE dsv4_cpp_core)
set_target_properties(test_qwen_dspark_ops PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_qwen_dflash2_ops tests/test_qwen_dflash2_ops.cpp)
target_link_libraries(test_qwen_dflash2_ops PRIVATE dsv4_cpp_core)
set_target_properties(test_qwen_dflash2_ops PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_qwen_sampler tests/test_qwen_sampler.cpp)
target_link_libraries(test_qwen_sampler PRIVATE dsv4_cpp_core)
set_target_properties(test_qwen_sampler PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_qwen_fp8_online tests/test_qwen_fp8_online.cpp)
target_link_libraries(test_qwen_fp8_online PRIVATE dsv4_cpp_core)
set_target_properties(test_qwen_fp8_online PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_qwen_weights tests/test_qwen_weights.cpp)
target_link_libraries(test_qwen_weights PRIVATE dsv4_cpp_core)
set_target_properties(test_qwen_weights PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_qwen_engine tests/test_qwen_engine.cpp)
target_link_libraries(test_qwen_engine PRIVATE dsv4_cpp_core)
set_target_properties(test_qwen_engine PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_qwen_delta_rule tests/test_qwen_delta_rule.cpp)
target_link_libraries(test_qwen_delta_rule PRIVATE dsv4_cpp_core)
set_target_properties(test_qwen_delta_rule PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_qwen_gqa_attention tests/test_qwen_gqa_attention.cpp)
target_link_libraries(test_qwen_gqa_attention PRIVATE dsv4_cpp_core)
set_target_properties(test_qwen_gqa_attention PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_qwen_half_ops tests/test_qwen_half_ops.cpp)
target_link_libraries(test_qwen_half_ops PRIVATE dsv4_cpp_core)
set_target_properties(test_qwen_half_ops PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_qwen_batched_decode tests/test_qwen_batched_decode.cpp)
target_link_libraries(test_qwen_batched_decode PRIVATE dsv4_cpp_core)
set_target_properties(test_qwen_batched_decode PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_qwen_batched_engine_parity tests/test_qwen_batched_engine_parity.cpp)
target_link_libraries(test_qwen_batched_engine_parity PRIVATE dsv4_cpp_core)
set_target_properties(test_qwen_batched_engine_parity PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(bench_qwen_batched_throughput tests/bench_qwen_batched_throughput.cpp)
target_link_libraries(bench_qwen_batched_throughput PRIVATE dsv4_cpp_core)
set_target_properties(bench_qwen_batched_throughput PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(bench_qwen_prefill_saturation tests/bench_qwen_prefill_saturation.cpp)
target_link_libraries(bench_qwen_prefill_saturation PRIVATE dsv4_cpp_core)
set_target_properties(bench_qwen_prefill_saturation PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_qwen_chunked_prefill tests/test_qwen_chunked_prefill.cpp)
target_link_libraries(test_qwen_chunked_prefill PRIVATE dsv4_cpp_core)
set_target_properties(test_qwen_chunked_prefill PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_qwen_eos_stop tests/test_qwen_eos_stop.cpp)
target_link_libraries(test_qwen_eos_stop PRIVATE dsv4_cpp_core)
set_target_properties(test_qwen_eos_stop PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_qwen_block_pool tests/test_qwen_block_pool.cpp)
target_link_libraries(test_qwen_block_pool PRIVATE dsv4_cpp_core)
set_target_properties(test_qwen_block_pool PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_qwen_paged_kv_kernels tests/test_qwen_paged_kv_kernels.cpp)
target_link_libraries(test_qwen_paged_kv_kernels PRIVATE dsv4_cpp_core)
set_target_properties(test_qwen_paged_kv_kernels PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_qwen_paged_engine_parity tests/test_qwen_paged_engine_parity.cpp)
target_link_libraries(test_qwen_paged_engine_parity PRIVATE dsv4_cpp_core)
set_target_properties(test_qwen_paged_engine_parity PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_qwen_paged_tp_slot_free tests/test_qwen_paged_tp_slot_free.cpp)
target_link_libraries(test_qwen_paged_tp_slot_free PRIVATE dsv4_cpp_core)
set_target_properties(test_qwen_paged_tp_slot_free PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_qwen_scheduler_paged_admission tests/test_qwen_scheduler_paged_admission.cpp)
target_link_libraries(test_qwen_scheduler_paged_admission PRIVATE dsv4_cpp_core)
set_target_properties(test_qwen_scheduler_paged_admission PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(bench_qwen_prefill_interleave tests/bench_qwen_prefill_interleave.cpp)
target_link_libraries(bench_qwen_prefill_interleave PRIVATE dsv4_cpp_core)
set_target_properties(bench_qwen_prefill_interleave PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(bench_qwen_paged_admission tests/bench_qwen_paged_admission.cpp)
target_link_libraries(bench_qwen_paged_admission PRIVATE dsv4_cpp_core)
set_target_properties(bench_qwen_paged_admission PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(bench_qwen_paged_tp_parity tests/bench_qwen_paged_tp_parity.cpp)
target_link_libraries(bench_qwen_paged_tp_parity PRIVATE dsv4_cpp_core)
set_target_properties(bench_qwen_paged_tp_parity PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_qwen_nvfp4 tests/test_qwen_nvfp4.cpp)
target_link_libraries(test_qwen_nvfp4 PRIVATE dsv4_cpp_core)
set_target_properties(test_qwen_nvfp4 PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(bench_qwen_nvfp4 tests/bench_qwen_nvfp4.cpp)
target_link_libraries(bench_qwen_nvfp4 PRIVATE dsv4_cpp_core)
set_target_properties(bench_qwen_nvfp4 PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_qwen_conv_tail tests/test_qwen_conv_tail.cpp)
target_link_libraries(test_qwen_conv_tail PRIVATE dsv4_cpp_core)
set_target_properties(test_qwen_conv_tail PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(bench_qwen_fp8_kernels tests/bench_qwen_fp8_kernels.cpp)
target_link_libraries(bench_qwen_fp8_kernels PRIVATE dsv4_cpp_core)
set_target_properties(bench_qwen_fp8_kernels PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(bench_qwen_draft_lm_head tests/bench_qwen_draft_lm_head.cpp)
target_link_libraries(bench_qwen_draft_lm_head PRIVATE dsv4_cpp_core)
set_target_properties(bench_qwen_draft_lm_head PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(bench_qwen_verify_batch tests/bench_qwen_verify_batch.cpp)
target_link_libraries(bench_qwen_verify_batch PRIVATE dsv4_cpp_core)
set_target_properties(bench_qwen_verify_batch PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(bench_qwen_delta_f16 tests/bench_qwen_delta_f16.cpp)
target_link_libraries(bench_qwen_delta_f16 PRIVATE dsv4_cpp_core)
set_target_properties(bench_qwen_delta_f16 PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_qwen_gdn_flashqla tests/test_qwen_gdn_flashqla.cpp)
target_link_libraries(test_qwen_gdn_flashqla PRIVATE dsv4_cpp_core)
set_target_properties(test_qwen_gdn_flashqla PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_qwen_turboquant_k8v4 tests/test_qwen_turboquant_k8v4.cpp)
target_link_libraries(test_qwen_turboquant_k8v4 PRIVATE dsv4_cpp_core)
set_target_properties(test_qwen_turboquant_k8v4 PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(bench_qwen_transaction_copy tests/bench_qwen_transaction_copy.cpp)
target_link_libraries(bench_qwen_transaction_copy PRIVATE dsv4_cpp_core)
set_target_properties(bench_qwen_transaction_copy PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

if(POCKET_BACKEND STREQUAL "cuda")
    add_executable(test_qwen_gqa_mma_occ tests/test_qwen_gqa_mma_occ.cpp)
    target_link_libraries(test_qwen_gqa_mma_occ PRIVATE dsv4_cpp_core)
    set_target_properties(test_qwen_gqa_mma_occ PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
else()
    # Experimental MMAD probe test targets temporarily disabled (source files removed)
    # add_executable(test_qwen_gqa_mmad_qk_probe tests/test_qwen_gqa_mmad_qk_probe.cpp)
    # target_link_libraries(test_qwen_gqa_mmad_qk_probe PRIVATE dsv4_cpp_core)
    # set_target_properties(test_qwen_gqa_mmad_qk_probe PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
    # Experimental MMAD probe test targets temporarily disabled (source files removed)
    # add_executable(test_qwen_gqa_mmad_qk_fulltile_probe tests/test_qwen_gqa_mmad_qk_fulltile_probe.cpp)
    # target_link_libraries(test_qwen_gqa_mmad_qk_fulltile_probe PRIVATE dsv4_cpp_core)
    # set_target_properties(test_qwen_gqa_mmad_qk_fulltile_probe PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
    # add_executable(test_qwen_gqa_mmad_qk_multihead_probe tests/test_qwen_gqa_mmad_qk_multihead_probe.cpp)
    # target_link_libraries(test_qwen_gqa_mmad_qk_multihead_probe PRIVATE dsv4_cpp_core)
    # set_target_properties(test_qwen_gqa_mmad_qk_multihead_probe PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
    # add_executable(test_qwen_gqa_mmad_qk_queryrow_probe tests/test_qwen_gqa_mmad_qk_queryrow_probe.cpp)
    # target_link_libraries(test_qwen_gqa_mmad_qk_queryrow_probe PRIVATE dsv4_cpp_core)
    # set_target_properties(test_qwen_gqa_mmad_qk_queryrow_probe PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

    # add_executable(test_qwen_gqa_mmad_qk_queryrow_optimized tests/test_qwen_gqa_mmad_qk_queryrow_optimized.cpp)
    # target_link_libraries(test_qwen_gqa_mmad_qk_queryrow_optimized PRIVATE dsv4_cpp_core)
    # set_target_properties(test_qwen_gqa_mmad_qk_queryrow_optimized PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

    # add_executable(test_qwen_gqa_mmad_qk_batched tests/test_qwen_gqa_mmad_qk_batched.cpp)
    # target_link_libraries(test_qwen_gqa_mmad_qk_batched PRIVATE dsv4_cpp_core)
    # set_target_properties(test_qwen_gqa_mmad_qk_batched PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

    # Experimental MMAD QK test variants temporarily disabled (source files removed)
    # add_executable(test_qwen_gqa_mmad_qk_batched_30core tests/test_qwen_gqa_mmad_qk_batched_30core.cpp)
    # target_link_libraries(test_qwen_gqa_mmad_qk_batched_30core PRIVATE dsv4_cpp_core)
    # set_target_properties(test_qwen_gqa_mmad_qk_batched_30core PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

    # add_executable(test_qwen_gqa_mmad_qk_batched_30core_n128 tests/test_qwen_gqa_mmad_qk_batched_30core_n128.cpp)
    # target_link_libraries(test_qwen_gqa_mmad_qk_batched_30core_n128 PRIVATE dsv4_cpp_core)
    # set_target_properties(test_qwen_gqa_mmad_qk_batched_30core_n128 PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

    # add_executable(test_qwen_gqa_mmad_qk_batched_30core_n128_fused tests/test_qwen_gqa_mmad_qk_batched_30core_n128_fused.cpp)
    # target_link_libraries(test_qwen_gqa_mmad_qk_batched_30core_n128_fused PRIVATE dsv4_cpp_core)
    # set_target_properties(test_qwen_gqa_mmad_qk_batched_30core_n128_fused PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
    # add_executable(test_qwen_gqa_mmad_qk_batched_30core_n128_pingpong tests/test_qwen_gqa_mmad_qk_batched_30core_n128_pingpong.cpp)
    # target_link_libraries(test_qwen_gqa_mmad_qk_batched_30core_n128_pingpong PRIVATE dsv4_cpp_core)
    # set_target_properties(test_qwen_gqa_mmad_qk_batched_30core_n128_pingpong PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

    # add_executable(test_qwen_gqa_mmad_qk_batched_30core_n128_persistent_a2 tests/test_qwen_gqa_mmad_qk_batched_30core_n128_persistent_a2.cpp)
    # target_link_libraries(test_qwen_gqa_mmad_qk_batched_30core_n128_persistent_a2 PRIVATE dsv4_cpp_core)
    # set_target_properties(test_qwen_gqa_mmad_qk_batched_30core_n128_persistent_a2 PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

    # add_executable(test_qwen_gqa_mmad_qk_batched_30core_n128_batched_extraction tests/test_qwen_gqa_mmad_qk_batched_30core_n128_batched_extraction.cpp)
    # target_link_libraries(test_qwen_gqa_mmad_qk_batched_30core_n128_batched_extraction PRIVATE dsv4_cpp_core)
    # set_target_properties(test_qwen_gqa_mmad_qk_batched_30core_n128_batched_extraction PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

    # Experimental MMAD test targets temporarily disabled (source files removed)
    # add_executable(test_qwen_gqa_decode_mmad_integrated tests/test_qwen_gqa_decode_mmad_integrated.cpp)
    # target_link_libraries(test_qwen_gqa_decode_mmad_integrated PRIVATE dsv4_cpp_core)
    # set_target_properties(test_qwen_gqa_decode_mmad_integrated PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

    # add_executable(test_qwen_gqa_mmad_smoke tests/test_qwen_gqa_mmad_smoke.cpp)
    # target_link_libraries(test_qwen_gqa_mmad_smoke PRIVATE dsv4_cpp_core)
    # set_target_properties(test_qwen_gqa_mmad_smoke PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

    # add_executable(test_qwen_gqa_fused_online_softmax tests/test_qwen_gqa_fused_online_softmax.cpp)
    # target_link_libraries(test_qwen_gqa_fused_online_softmax PRIVATE dsv4_cpp_core)
    # set_target_properties(test_qwen_gqa_fused_online_softmax PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

    # add_executable(test_qwen_gqa_fused_online_softmax_direct tests/test_qwen_gqa_fused_online_softmax_direct.cpp)
    # target_link_libraries(test_qwen_gqa_fused_online_softmax_direct PRIVATE dsv4_cpp_core)
    # set_target_properties(test_qwen_gqa_fused_online_softmax_direct PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
endif()

if(DSV4_HAVE_TP_COMM)
    add_executable(bench_qwen_tp_allreduce tests/bench_qwen_tp_allreduce.cpp)
    target_link_libraries(bench_qwen_tp_allreduce PRIVATE dsv4_cpp_core)
    set_target_properties(bench_qwen_tp_allreduce PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
endif()

add_executable(test_persistent_engine tests/test_persistent_engine.cpp)
target_link_libraries(test_persistent_engine PRIVATE dsv4_cpp_core)
set_target_properties(test_persistent_engine PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(test_qwen_batch_scheduler tests/test_qwen_batch_scheduler.cpp)
target_link_libraries(test_qwen_batch_scheduler PRIVATE dsv4_cpp_core)
set_target_properties(test_qwen_batch_scheduler PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(bench_qwen_fp8_batch_crossover tests/bench_qwen_fp8_batch_crossover.cpp)
target_link_libraries(bench_qwen_fp8_batch_crossover PRIVATE dsv4_cpp_core)
set_target_properties(bench_qwen_fp8_batch_crossover PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(bench_qwen_fp8_dequant_overhead tests/bench_qwen_fp8_dequant_overhead.cpp)
target_link_libraries(bench_qwen_fp8_dequant_overhead PRIVATE dsv4_cpp_core)
set_target_properties(bench_qwen_fp8_dequant_overhead PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(bench_qwen_gqa_prefill tests/bench_qwen_gqa_prefill.cpp)
target_link_libraries(bench_qwen_gqa_prefill PRIVATE dsv4_cpp_core)
set_target_properties(bench_qwen_gqa_prefill PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

if(POCKET_BACKEND STREQUAL "cuda")
    add_executable(bench_qwen_gqa_decode tests/bench_qwen_gqa_decode.cpp)
    target_link_libraries(bench_qwen_gqa_decode PRIVATE dsv4_cpp_core)
    set_target_properties(bench_qwen_gqa_decode PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
else()
    # Ascend bench temporarily disabled (source file removed)
    # add_executable(bench_qwen_gqa_decode tests/bench_qwen_gqa_decode_ascend.cpp)
    # target_link_libraries(bench_qwen_gqa_decode PRIVATE dsv4_cpp_core)
    # set_target_properties(bench_qwen_gqa_decode PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

endif()

add_executable(bench_qwen_fp8_swiglu_decode tests/bench_qwen_fp8_swiglu_decode.cpp)
target_link_libraries(bench_qwen_fp8_swiglu_decode PRIVATE dsv4_cpp_core)
set_target_properties(bench_qwen_fp8_swiglu_decode PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(bench_qwen_gqa_verify tests/bench_qwen_gqa_verify.cpp)
target_link_libraries(bench_qwen_gqa_verify PRIVATE dsv4_cpp_core)
set_target_properties(bench_qwen_gqa_verify PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)

add_executable(bench_qwen_dflash2_topk tests/bench_qwen_dflash2_topk.cpp)
target_link_libraries(bench_qwen_dflash2_topk PRIVATE dsv4_cpp_core)
set_target_properties(bench_qwen_dflash2_topk PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/tests)
