// Host launchers for the hand-written AscendC kernels.
//
// These are the twelve operators with no aclnn equivalent on this install: the
// gated-delta recurrence and its gates/normalization, the causal depthwise
// convolution, partial RoPE, the KV-cache append and the three GQA attention
// forms. The device code lives beside this file; everything here is argument
// validation, block-count selection and the launch itself.
//
// ascendc_library() generates one host stub per `extern "C" __global__ __aicore__`
// entry point:
//
//     extern "C" uint32_t aclrtlaunch_<kernel>(uint32_t numBlocks,
//                                              aclrtStream stream, <args...>);
//
// It returns 0 on success and allocates/frees its own 8-byte overflow-status
// buffer per launch. The generated headers land in
// ${CMAKE_BINARY_DIR}/include/pocket_ascend_kernels, which is on this target's
// include path.
//
// Two things are worth knowing before changing anything here:
//
//   1. Validation is not defensive decoration. The kernels index GM with
//      unsigned arithmetic and no bounds checks, so a bad `max_context` or a
//      `heads % key_heads != 0` is an out-of-bounds device write, not a wrong
//      number. Every contract the device code assumes is checked here, and the
//      checks mirror the CUDA launchers so a caller cannot tell the backends
//      apart by which arguments they reject.
//   2. There is no device sin/cos reachable from a classic __aicore__ kernel, so
//      partial RoPE takes precomputed FP32 tables. The first correctness path builds
//      the requested tables per call on the host and uploads them synchronously; it
//      deliberately makes no retained per-device or per-geometry cache claim.

#include "aclnn_common.hpp"

#include "qwen_ascend_ops.hpp"

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <map>
#include <mutex>
#include <vector>

#include "aclrtlaunch_qwen_append_kv_cache_kernel.h"
#include "aclrtlaunch_qwen_argmax_f32_rows_kernel.h"
#include "aclrtlaunch_qwen_causal_depthwise_conv_silu_kernel.h"
#include "aclrtlaunch_qwen_gated_delta_sequence_kernel.h"
#include "aclrtlaunch_qwen_gated_delta_sequence_normalized_kernel.h"
#include "aclrtlaunch_qwen_gated_delta_step_kernel.h"
#include "aclrtlaunch_qwen_gqa_decode_attention_kernel.h"
#include "aclrtlaunch_qwen_hbm_read_probe_kernel.h"
#include "aclrtlaunch_qwen_gqa_decode_attention_merge_kernel.h"
#include "aclrtlaunch_qwen_gqa_decode_attention_normalize_kernel.h"
#include "aclrtlaunch_qwen_gqa_decode_attention_split_kernel.h"
#include "aclrtlaunch_qwen_gqa_decode_attention_normalize_raw_kernel.h"
#include "aclrtlaunch_qwen_gqa_decode_attention_split_kv_shared_kernel.h"
#include "aclrtlaunch_qwen_gqa_decode_attention_vectorized_kernel.h"
// MMAD experimental kernels removed
// #include "aclrtlaunch_qwen_gqa_decode_mmad_qk_score_kernel.h"
// #include "aclrtlaunch_qwen_gqa_decode_mmad_softmax_v_kernel.h"
// #include "aclrtlaunch_qwen_gqa_decode_softmax_v_from_scores_kernel.h"
#include "aclrtlaunch_qwen_gqa_prefill_attention_kernel.h"
#include "aclrtlaunch_qwen_gqa_verify_attention_kernel.h"
#include "aclrtlaunch_qwen_linear_attn_gates_kernel.h"
#include "aclrtlaunch_qwen_normalize_gated_delta_qk_kernel.h"
#include "aclrtlaunch_qwen_partial_rope_rows_kernel.h"

namespace pocket {
namespace {

using ascend::resolve;

// The generated stubs return 0 for success.
constexpr uint32_t kLaunchOk = 0;

// The recurrence kernels hard-code a [128, 128] state tile, so the host has to
// reject any other head geometry rather than launch a kernel that would read the
// wrong stride. The CUDA launchers reject the same values.
constexpr int kRecurrentKeyDim = 128;
constexpr int kRecurrentValueDim = 128;

// Convolution tail capacity, matching kMaxKernel in the device code and kMaxTail
// on the CUDA side.
constexpr int kMaxConvKernel = 8;

// First-generation 910 has 30 AI cores. Every kernel here is a grid-stride loop
// over independent work items, so more blocks than cores would only add launch
// overhead; fewer than the work available would leave cores idle.
constexpr uint32_t kMaxBlocks = 30;

// Query the AI core count once per device and fall back to the known
// first-generation value. Reading it rather than hard-coding it means a run on
// second-generation silicon (20 or 24 cores) still launches a legal grid, even
// though the kernels are not tuned for it.
uint32_t core_count() {
    static std::mutex mutex;
    static std::map<int32_t, uint32_t> cache;
    int32_t device = 0;
    if (aclrtGetDevice(&device) != ACL_SUCCESS) return kMaxBlocks;
    std::lock_guard<std::mutex> guard(mutex);
    auto found = cache.find(device);
    if (found != cache.end()) return found->second;
    int64_t cores = 0;
    uint32_t resolved = kMaxBlocks;
    if (aclrtGetDeviceInfo(static_cast<uint32_t>(device),
                           ACL_DEV_ATTR_AICORE_CORE_NUM,
                           &cores) == ACL_SUCCESS &&
        cores > 0) {
        // Clamped rather than trusted. A reported count above 30 means this is
        // not the SoC these kernels were written for, and launching that grid
        // would size the loop for hardware whose UB budget has not been checked.
        resolved = static_cast<uint32_t>(cores);
        if (resolved > kMaxBlocks) resolved = kMaxBlocks;
    }
    cache.emplace(device, resolved);
    return resolved;
}

// Blocks for `work` independent items: never more than there is work for, never
// more than there are cores, never zero.
uint32_t blocks_for(uint64_t work) {
    const uint32_t cores = core_count();
    if (work == 0) return 1;
    if (work < static_cast<uint64_t>(cores)) return static_cast<uint32_t>(work);
    return cores;
}

// Shared attention preconditions, mirroring valid_attention() in the CUDA
// launcher so both backends reject the same shapes.
bool valid_attention(int q_heads, int kv_heads, int head_dim, int context_len,
                     int max_context) {
    return q_heads > 0 && kv_heads > 0 && q_heads % kv_heads == 0 &&
           head_dim > 0 && context_len > 0 && context_len <= max_context;
}

// The softmax scale. Passed to the kernel as a float because aicore cannot cast
// an unsigned integer to floating point, and its integer sqrt overload truncates
// (head_dim 128 would give 11 instead of 11.3137).
float attention_scale(int head_dim) {
    return 1.0f / std::sqrt(static_cast<float>(head_dim));
}

bool valid_recurrent(int heads, int key_heads, int key_dim, int value_dim,
                     float q_scale) {
    return heads > 0 && key_heads > 0 && heads % key_heads == 0 &&
           key_dim == kRecurrentKeyDim && value_dim == kRecurrentValueDim &&
           q_scale > 0.0f;
}

// Per-call FP32 cos/sin tables for partial RoPE.
//
// The table is [rows, rotary_dim / 2] and contains the absolute positions starting
// at `start_position`. Angles are computed in double and rounded once to float;
// the CUDA device implementation's float powf/cosf path is therefore not copied
// into this first-generation kernel, which has no classic-aicore trig primitive.
//
// The device allocation comes from WorkspacePool's Intermediate slot rather than
// aclrtMalloc. The copy is queued on the same stream as the kernel, so the table
// remains live until the launch consumes it and can be reused by the next
// serialized operator without a stream-wide synchronization.
class RopeTables {
public:
    static bool acquire(int rotary_dim, float theta, int start_position, int rows,
                        aclrtStream stream, const float** cos_rows,
                        const float** sin_rows) {
        const size_t half = static_cast<size_t>(rotary_dim / 2);
        const size_t elements = static_cast<size_t>(rows) * half;
        const size_t bytes = elements * sizeof(float);
        std::vector<float> host_cos(elements);
        std::vector<float> host_sin(elements);
        std::vector<double> inv_freq(half);
        for (size_t i = 0; i < half; ++i) {
            inv_freq[i] = std::pow(
                static_cast<double>(theta),
                -2.0 * static_cast<double>(i) /
                    static_cast<double>(rotary_dim));
        }
        for (int row = 0; row < rows; ++row) {
            const double position = static_cast<double>(start_position + row);
            const size_t base = static_cast<size_t>(row) * half;
            for (size_t i = 0; i < half; ++i) {
                const double angle = position * inv_freq[i];
                host_cos[base + i] = static_cast<float>(std::cos(angle));
                host_sin[base + i] = static_cast<float>(std::sin(angle));
            }
        }

        bool pool_ok = true;
        void* storage = ascend::WorkspacePool::acquire(
            bytes * 2, stream, pool_ok,
            ascend::WorkspacePool::Purpose::Intermediate);
        if (!pool_ok || storage == nullptr) return false;
        auto* device = static_cast<float*>(storage);
        // The host vectors are temporary. A synchronous copy keeps their lifetime
        // explicit; enqueueing an async H2D and destroying them at return races the
        // DMA engine and produces position-dependent RoPE corruption.
        if (aclrtMemcpy(device, bytes, host_cos.data(), bytes,
                        ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS ||
            aclrtMemcpy(device + elements, bytes, host_sin.data(), bytes,
                        ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
            return false;
        }
        *cos_rows = device;
        *sin_rows = device + elements;
        return true;
    }
};

bool valid_rope(int start_position, int rows, int rotary_dim, float theta,
                int q_heads, int kv_heads, int head_dim) {
    return start_position >= 0 && rows > 0 && rotary_dim > 0 &&
           (rotary_dim & 1) == 0 && rotary_dim <= head_dim &&
           std::isfinite(theta) && theta > 0.0f && q_heads > 0 &&
           kv_heads > 0 && head_dim > 0;
}

// The kernels take GM_ADDR, which the generated stubs expose as void*. Const is
// dropped at the boundary because the launch ABI has no const form; the device
// code only reads these.
template <typename T>
void* gm(const T* pointer) {
    return const_cast<void*>(static_cast<const void*>(pointer));
}

}  // namespace

bool qwen_normalize_gated_delta_qk_f16_ascend(
    const uint16_t* d_q_fp16, const uint16_t* d_k_fp16, float* d_q_normalized,
    float* d_k_normalized, int rows, int key_heads, int key_dim, void* stream) {
    if (d_q_fp16 == nullptr || d_k_fp16 == nullptr ||
        d_q_normalized == nullptr || d_k_normalized == nullptr || rows <= 0 ||
        key_heads <= 0 || key_dim != kRecurrentKeyDim) {
        return false;
    }
    const uint64_t pairs = static_cast<uint64_t>(rows) * key_heads;
    return aclrtlaunch_qwen_normalize_gated_delta_qk_kernel(
               blocks_for(pairs), resolve(stream), gm(d_q_fp16), gm(d_k_fp16),
               gm(d_q_normalized), gm(d_k_normalized),
               static_cast<uint32_t>(rows),
               static_cast<uint32_t>(key_heads)) == kLaunchOk;
}

bool qwen_gated_delta_sequence_f16_ascend(
    float* d_state, const uint16_t* d_q_fp16, const uint16_t* d_k_fp16,
    const uint16_t* d_v_fp16, const uint16_t* d_g_fp16,
    const uint16_t* d_beta_fp16, uint16_t* d_out_fp16, int rows, int heads,
    int key_heads, int key_dim, int value_dim, float q_scale, void* stream) {
    if (d_state == nullptr || d_q_fp16 == nullptr || d_k_fp16 == nullptr ||
        d_v_fp16 == nullptr || d_g_fp16 == nullptr || d_beta_fp16 == nullptr ||
        d_out_fp16 == nullptr || rows <= 0 ||
        !valid_recurrent(heads, key_heads, key_dim, value_dim, q_scale)) {
        return false;
    }
    return aclrtlaunch_qwen_gated_delta_sequence_kernel(
               blocks_for(static_cast<uint64_t>(heads)), resolve(stream),
               gm(d_state), gm(d_q_fp16), gm(d_k_fp16), gm(d_v_fp16),
               gm(d_g_fp16), gm(d_beta_fp16), gm(d_out_fp16),
               static_cast<uint32_t>(rows), static_cast<uint32_t>(heads),
               static_cast<uint32_t>(key_heads), q_scale) == kLaunchOk;
}

bool qwen_gated_delta_sequence_normalized_f16_ascend(
    float* d_state, const float* d_q_normalized, const float* d_k_normalized,
    const uint16_t* d_v_fp16, const uint16_t* d_g_fp16,
    const uint16_t* d_beta_fp16, uint16_t* d_out_fp16, int rows, int heads,
    int key_heads, int key_dim, int value_dim, float q_scale, void* stream) {
    if (d_state == nullptr || d_q_normalized == nullptr ||
        d_k_normalized == nullptr || d_v_fp16 == nullptr ||
        d_g_fp16 == nullptr || d_beta_fp16 == nullptr ||
        d_out_fp16 == nullptr || rows <= 0 ||
        !valid_recurrent(heads, key_heads, key_dim, value_dim, q_scale)) {
        return false;
    }
    return aclrtlaunch_qwen_gated_delta_sequence_normalized_kernel(
               blocks_for(static_cast<uint64_t>(heads)), resolve(stream),
               gm(d_state), gm(d_q_normalized), gm(d_k_normalized),
               gm(d_v_fp16), gm(d_g_fp16), gm(d_beta_fp16), gm(d_out_fp16),
               static_cast<uint32_t>(rows), static_cast<uint32_t>(heads),
               static_cast<uint32_t>(key_heads), q_scale) == kLaunchOk;
}

// The CUDA `_shared` variant shards one head's state across a block so several
// value columns progress together in shared memory. That is a launch-geometry
// choice, not a different recurrence: it is bit-comparable to the normalized
// kernel by construction. This part has no equivalent to shard onto (a head's
// state already lives entirely in UB and tokens are strictly sequential), so the
// contract is satisfied by the normalized kernel itself rather than by a second
// device implementation.
bool qwen_gated_delta_sequence_normalized_shared_f16_ascend(
    float* d_state, const float* d_q_normalized, const float* d_k_normalized,
    const uint16_t* d_v_fp16, const uint16_t* d_g_fp16,
    const uint16_t* d_beta_fp16, uint16_t* d_out_fp16, int rows, int heads,
    int key_heads, int key_dim, int value_dim, float q_scale, void* stream) {
    return qwen_gated_delta_sequence_normalized_f16_ascend(
        d_state, d_q_normalized, d_k_normalized, d_v_fp16, d_g_fp16,
        d_beta_fp16, d_out_fp16, rows, heads, key_heads, key_dim, value_dim,
        q_scale, stream);
}

bool qwen_gated_delta_step_f16_ascend(
    float* d_state, const uint16_t* d_q_fp16, const uint16_t* d_k_fp16,
    const uint16_t* d_v_fp16, const uint16_t* d_g_fp16,
    const uint16_t* d_beta_fp16, uint16_t* d_out_fp16, int heads,
    int key_heads, int key_dim, int value_dim, float q_scale, void* stream) {
    if (d_state == nullptr || d_q_fp16 == nullptr || d_k_fp16 == nullptr ||
        d_v_fp16 == nullptr || d_g_fp16 == nullptr || d_beta_fp16 == nullptr ||
        d_out_fp16 == nullptr ||
        !valid_recurrent(heads, key_heads, key_dim, value_dim, q_scale)) {
        return false;
    }
    return aclrtlaunch_qwen_gated_delta_step_kernel(
               blocks_for(static_cast<uint64_t>(heads)), resolve(stream),
               gm(d_state), gm(d_q_fp16), gm(d_k_fp16), gm(d_v_fp16),
               gm(d_g_fp16), gm(d_beta_fp16), gm(d_out_fp16),
               static_cast<uint32_t>(heads), static_cast<uint32_t>(key_heads),
               q_scale) == kLaunchOk;
}

bool qwen_linear_attn_gates_f16_ascend(
    const uint16_t* d_a_fp16, const uint16_t* d_b_fp16,
    const uint16_t* d_a_log_fp16, const uint16_t* d_dt_bias_fp16,
    uint16_t* d_g_fp16, uint16_t* d_beta_fp16, int rows, int heads,
    void* stream) {
    if (d_a_fp16 == nullptr || d_b_fp16 == nullptr ||
        d_a_log_fp16 == nullptr || d_dt_bias_fp16 == nullptr ||
        d_g_fp16 == nullptr || d_beta_fp16 == nullptr || rows <= 0 ||
        heads <= 0) {
        return false;
    }
    // The kernel tiles the flat [rows*heads] plane in 512-lane chunks.
    const uint64_t total = static_cast<uint64_t>(rows) * heads;
    const uint64_t tiles = (total + 511) / 512;
    return aclrtlaunch_qwen_linear_attn_gates_kernel(
               blocks_for(tiles), resolve(stream), gm(d_a_fp16), gm(d_b_fp16),
               gm(d_a_log_fp16), gm(d_dt_bias_fp16), gm(d_g_fp16),
               gm(d_beta_fp16), static_cast<uint32_t>(rows),
               static_cast<uint32_t>(heads)) == kLaunchOk;
}

bool qwen_causal_depthwise_conv_silu_f16_ascend(
    const uint16_t* d_x_fp16, const uint16_t* d_weight_fp16,
    uint16_t* d_tail_fp16, uint16_t* d_y_fp16, int seq_len, int channels,
    int kernel, bool update_tail, void* stream) {
    // d_tail_fp16 is allowed to be null: that signals zero convolution history,
    // and the device code branches on it. Everything else is required.
    if (d_x_fp16 == nullptr || d_weight_fp16 == nullptr || d_y_fp16 == nullptr ||
        seq_len <= 0 || channels <= 0 || kernel <= 0 ||
        kernel > kMaxConvKernel) {
        return false;
    }
    // The kernel walks 512-channel tiles for the whole sequence.
    const uint64_t tiles = (static_cast<uint64_t>(channels) + 511) / 512;
    return aclrtlaunch_qwen_causal_depthwise_conv_silu_kernel(
               blocks_for(tiles), resolve(stream), gm(d_x_fp16),
               gm(d_weight_fp16), gm(d_tail_fp16), gm(d_y_fp16),
               static_cast<uint32_t>(seq_len), static_cast<uint32_t>(channels),
               static_cast<uint32_t>(kernel),
               update_tail ? 1u : 0u) == kLaunchOk;
}

bool qwen_partial_rope_rows_f16_ascend(
    uint16_t* d_q_fp16, uint16_t* d_k_fp16, int start_position, int rows,
    int rotary_dim, float theta, int q_heads, int kv_heads, int head_dim,
    void* stream) {
    if (d_q_fp16 == nullptr || d_k_fp16 == nullptr || start_position < 0 ||
        rows <= 0 || rotary_dim <= 0 || (rotary_dim & 1) != 0 ||
        rotary_dim > head_dim || theta <= 0.0f || q_heads <= 0 ||
        kv_heads <= 0 || head_dim <= 0) {
        return false;
    }
    const aclrtStream s = resolve(stream);
    const float* cos_rows = nullptr;
    const float* sin_rows = nullptr;
    if (!RopeTables::acquire(rotary_dim, theta, start_position, rows, s,
                             &cos_rows, &sin_rows)) {
        return false;
    }
    const uint64_t pairs = static_cast<uint64_t>(rows) * (rotary_dim / 2);
    return aclrtlaunch_qwen_partial_rope_rows_kernel(
               blocks_for(pairs), s, gm(d_q_fp16), gm(d_k_fp16), gm(cos_rows),
               gm(sin_rows), static_cast<uint32_t>(rows),
               static_cast<uint32_t>(rotary_dim),
               static_cast<uint32_t>(q_heads), static_cast<uint32_t>(kv_heads),
               static_cast<uint32_t>(head_dim)) == kLaunchOk;
}

bool qwen_append_kv_cache_f16_ascend(
    const uint16_t* d_k_rows_fp16, const uint16_t* d_v_rows_fp16,
    uint16_t* d_k_cache_fp16, uint16_t* d_v_cache_fp16, int seq_len,
    int kv_heads, int head_dim, int start_pos, int max_context, void* stream) {
    if (d_k_rows_fp16 == nullptr || d_v_rows_fp16 == nullptr ||
        d_k_cache_fp16 == nullptr || d_v_cache_fp16 == nullptr ||
        seq_len <= 0 || kv_heads <= 0 || head_dim <= 0 || start_pos < 0 ||
        max_context <= 0 || start_pos + seq_len > max_context) {
        return false;
    }
    const uint64_t elements = static_cast<uint64_t>(seq_len) * kv_heads * head_dim;
    const uint64_t tiles = (elements + 511) / 512;
    // Scalar handling of an unaligned tail is serialized to avoid two cores
    // updating different halfs in the same cache line. Qwen's 128-wide rows take
    // the aligned parallel path.
    const bool aligned = head_dim % 16 == 0;
    return aclrtlaunch_qwen_append_kv_cache_kernel(
               aligned ? blocks_for(tiles) : 1u, resolve(stream), gm(d_k_rows_fp16),
               gm(d_v_rows_fp16), gm(d_k_cache_fp16), gm(d_v_cache_fp16),
               static_cast<uint32_t>(seq_len), static_cast<uint32_t>(kv_heads),
               static_cast<uint32_t>(head_dim),
               static_cast<uint32_t>(start_pos),
               static_cast<uint32_t>(max_context)) == kLaunchOk;
}

bool qwen_gqa_decode_attention_f16_ascend(
    const uint16_t* d_q_fp16, const uint16_t* d_k_cache_fp16,
    const uint16_t* d_v_cache_fp16, uint16_t* d_out_fp16,
    float* d_score_scratch, int q_heads, int kv_heads, int head_dim,
    int context_len, int max_context, void* stream) {
    if (d_q_fp16 == nullptr || d_k_cache_fp16 == nullptr ||
        d_v_cache_fp16 == nullptr || d_out_fp16 == nullptr ||
        d_score_scratch == nullptr ||
        !valid_attention(q_heads, kv_heads, head_dim, context_len,
                         max_context)) {
        return false;
    }

    // Try vectorized path when geometry matches target (Qwen3.8-27B TP4):
    // 6 query heads, 1 KV head, head_dim=256. The vectorized kernel uses
    // UB-resident tiles to eliminate redundant scalar GM loads and exploits
    // vector operations for QK dot products. Falls back to scalar path for
    // non-standard configurations.
    // This opt-out is a diagnostic knob only: it enables an apples-to-apples
    // scalar-versus-vector benchmark at the tuned geometry without changing the
    // neutral API or the production default.
    const bool force_scalar = std::getenv("QWEN_ASCEND_FORCE_SCALAR_GQA") != nullptr;
    const bool force_unsplit = std::getenv("QWEN_ASCEND_FORCE_UNSPLIT_GQA") != nullptr;
    // Vectorized kernels temporarily disabled - using scalar fallback for all geometries
    const bool use_vectorized = false;
    // const bool use_vectorized = (q_heads == 6 && kv_heads == 1 && head_dim == 256 &&
    //                              !force_scalar);

    if (use_vectorized) {
        // Two split geometries share the same partial-record plane, merge and
        // normalize stages, and differ only in how the 30 first-generation AI
        // cores are mapped onto the work.
        //
        // With kv_heads == 1 every query head reads the identical K/V bytes, so
        // one block per (head, split) pair loads the cache q_heads times over.
        // Removing that redundancy is what the KV-sharing geometry is for. Note
        // it is not a bandwidth argument: a measured 30-core stream probe puts
        // achievable read bandwidth near 1050 GB/s, and this operator reaches
        // only 4-7% of it, so latency is set by Vector instruction count and
        // pipe hand-offs. Sharing the tile still wins at long contexts because
        // it also shares the per-tile widening and staging work. The preferred
        // geometry therefore gives each core its own context range and sweeps
        // every head across the K/V tile already in UB: the same total Vector
        // work spread over the same 30 cores, but the cache is read once.
        //
        // The saving is not free: 30 context ranges mean 30 partial records per
        // head to merge instead of 5, and a per-range prologue that reloads all
        // six Q vectors. Those fixed costs are constant while the traffic saving
        // grows with the context, so the geometry only pays off once each range
        // is long enough to amortize them. Measured on one 910A at this geometry
        // (ms/token, per-head vs KV-sharing):
        //
        //    4096   0.105   0.167
        //    8192   0.184   0.217
        //   16384   0.343   0.317
        //   32768   0.658   0.540
        //   65536   1.319   0.966
        //
        // Hence the 16K threshold rather than the 30-tile minimum the kernel
        // itself needs. Below it the per-head mapping is still the faster one.
        const bool shared_kv_split =
            kv_heads == 1 && q_heads == 6 && head_dim == 256 &&
            context_len >= 16384;
        const int splits = shared_kv_split ? 30 : 5;
        if ((context_len % 64) == 0 && context_len >= splits * 64) {
            bool workspace_ok = true;
            const size_t records =
                static_cast<size_t>(q_heads) * static_cast<size_t>(splits);
            const size_t record_stride = 8 + head_dim;
            void* partial = ascend::WorkspacePool::acquire(
                records * record_stride * sizeof(float), resolve(stream),
                workspace_ok, ascend::WorkspacePool::Purpose::Intermediate);
            if (workspace_ok && partial != nullptr) {
                const aclrtStream s = resolve(stream);
                const uint32_t launched = static_cast<uint32_t>(splits);
                if (shared_kv_split) {
                    if (aclrtlaunch_qwen_gqa_decode_attention_split_kv_shared_kernel(
                            launched, s, gm(d_q_fp16), gm(d_k_cache_fp16),
                            gm(d_v_cache_fp16), gm(d_score_scratch), partial,
                            static_cast<uint32_t>(q_heads),
                            static_cast<uint32_t>(kv_heads),
                            static_cast<uint32_t>(head_dim),
                            static_cast<uint32_t>(context_len),
                            static_cast<uint32_t>(max_context),
                            attention_scale(head_dim)) != kLaunchOk) {
                        return false;
                    }
                } else if (aclrtlaunch_qwen_gqa_decode_attention_split_kernel(
                               30u, s, gm(d_q_fp16), gm(d_k_cache_fp16),
                               gm(d_v_cache_fp16), gm(d_score_scratch), partial,
                               static_cast<uint32_t>(q_heads),
                               static_cast<uint32_t>(kv_heads),
                               static_cast<uint32_t>(head_dim),
                               static_cast<uint32_t>(context_len),
                               static_cast<uint32_t>(max_context),
                               attention_scale(head_dim)) != kLaunchOk) {
                    return false;
                }
                if (aclrtlaunch_qwen_gqa_decode_attention_merge_kernel(
                        static_cast<uint32_t>(q_heads), s, gm(d_out_fp16),
                        partial, static_cast<uint32_t>(q_heads),
                        static_cast<uint32_t>(head_dim), launched) !=
                    kLaunchOk) {
                    return false;
                }
                // The shared kernel leaves raw scaled scores in scratch because
                // it never needs a whole-plane Exp; the per-head kernel already
                // left local exponentials there and only needs a rescale.
                if (shared_kv_split) {
                    return aclrtlaunch_qwen_gqa_decode_attention_normalize_raw_kernel(
                               30u, s, gm(d_score_scratch), partial,
                               static_cast<uint32_t>(q_heads),
                               static_cast<uint32_t>(context_len),
                               launched) == kLaunchOk;
                }
                return aclrtlaunch_qwen_gqa_decode_attention_normalize_kernel(
                           30u, s, gm(d_score_scratch), partial,
                           static_cast<uint32_t>(q_heads),
                           static_cast<uint32_t>(context_len),
                           launched) == kLaunchOk;
            }
        }

        // The fallback keeps one complete block per query head. Rows whose
        // length is not a multiple of eight floats would share a 32-byte cache
        // line with the next head, so serialize that valid but irregular case.
        const bool aligned_scratch_rows = (context_len % 8) == 0;
        const uint32_t vector_blocks = aligned_scratch_rows
            ? blocks_for(static_cast<uint64_t>(q_heads)) : 1u;
        return aclrtlaunch_qwen_gqa_decode_attention_vectorized_kernel(
                   vector_blocks, resolve(stream), gm(d_q_fp16),
                   gm(d_k_cache_fp16), gm(d_v_cache_fp16), gm(d_out_fp16),
                   gm(d_score_scratch), static_cast<uint32_t>(q_heads),
                   static_cast<uint32_t>(kv_heads), static_cast<uint32_t>(head_dim),
                   static_cast<uint32_t>(context_len),
                   static_cast<uint32_t>(max_context), attention_scale(head_dim)) ==
               kLaunchOk;
    }

    // The device baseline writes both output and score scratch with scalar GM
    // stores. Their neutral layouts need not be 32-byte aligned per head (the
    // score row is only `context_len` floats), and scalar stores from adjacent
    // blocks can then lose one another in a shared GM cache line. Use one block
    // for this correctness-first implementation; the whole head loop remains
    // grid-stride-ready for a future block-owned, aligned fast path.
    return aclrtlaunch_qwen_gqa_decode_attention_kernel(
               1u, resolve(stream), gm(d_q_fp16), gm(d_k_cache_fp16),
               gm(d_v_cache_fp16), gm(d_out_fp16), gm(d_score_scratch),
               static_cast<uint32_t>(q_heads), static_cast<uint32_t>(kv_heads),
               static_cast<uint32_t>(head_dim),
               static_cast<uint32_t>(context_len),
               static_cast<uint32_t>(max_context), attention_scale(head_dim)) ==
           kLaunchOk;
}

bool qwen_gqa_prefill_attention_f16_ascend(
    const uint16_t* d_q_rows_fp16, const uint16_t* d_k_cache_fp16,
    const uint16_t* d_v_cache_fp16, uint16_t* d_out_rows_fp16, int seq_len,
    int q_heads, int kv_heads, int head_dim, int position_offset,
    int max_context, void* stream) {
    if (d_q_rows_fp16 == nullptr || d_k_cache_fp16 == nullptr ||
        d_v_cache_fp16 == nullptr || d_out_rows_fp16 == nullptr ||
        seq_len <= 0 || position_offset < 0 ||
        position_offset + seq_len > max_context ||
        !valid_attention(q_heads, kv_heads, head_dim, position_offset + seq_len,
                         max_context)) {
        return false;
    }
    const uint64_t work = static_cast<uint64_t>(seq_len) * q_heads;
    return aclrtlaunch_qwen_gqa_prefill_attention_kernel(
               blocks_for(work), resolve(stream), gm(d_q_rows_fp16),
               gm(d_k_cache_fp16), gm(d_v_cache_fp16), gm(d_out_rows_fp16),
               static_cast<uint32_t>(seq_len), static_cast<uint32_t>(q_heads),
               static_cast<uint32_t>(kv_heads),
               static_cast<uint32_t>(head_dim),
               static_cast<uint32_t>(position_offset),
               static_cast<uint32_t>(max_context),
               attention_scale(head_dim)) == kLaunchOk;
}

bool qwen_gqa_verify_attention_f16_ascend(
    const uint16_t* d_q_rows_fp16, const uint16_t* d_k_cache_fp16,
    const uint16_t* d_v_cache_fp16, uint16_t* d_out_rows_fp16,
    float* d_partial_scratch, int rows, int q_heads, int kv_heads,
    int head_dim, int position_offset, int max_context, int splits,
    void* stream) {
    // splits is also the scratch geometry the caller sized
    // [rows, q_heads, splits, head_dim + 2] against, so a mismatch here is an
    // out-of-bounds write rather than a wrong answer.
    if (d_q_rows_fp16 == nullptr || d_k_cache_fp16 == nullptr ||
        d_v_cache_fp16 == nullptr || d_out_rows_fp16 == nullptr ||
        d_partial_scratch == nullptr || rows <= 0 || position_offset < 0 ||
        position_offset + rows > max_context || splits <= 0 ||
        splits > position_offset + rows ||
        !valid_attention(q_heads, kv_heads, head_dim, position_offset + rows,
                         max_context)) {
        return false;
    }
    // Partial records are also scalar-written and a record is not necessarily an
    // integral number of 32-byte cache lines (`splits * (head_dim + 2)` floats).
    // Serialize ownership until the kernel has a block-owned aligned store path.
    return aclrtlaunch_qwen_gqa_verify_attention_kernel(
               1u, resolve(stream), gm(d_q_rows_fp16), gm(d_k_cache_fp16),
               gm(d_v_cache_fp16), gm(d_out_rows_fp16), gm(d_partial_scratch),
               static_cast<uint32_t>(rows), static_cast<uint32_t>(q_heads),
               static_cast<uint32_t>(kv_heads), static_cast<uint32_t>(head_dim),
               static_cast<uint32_t>(position_offset),
               static_cast<uint32_t>(max_context),
               static_cast<uint32_t>(splits), attention_scale(head_dim)) ==
           kLaunchOk;
}

// Greedy top-1 per logits row. aclnn has ArgMax, but it returns only the index
// and the caller needs the winning logit as well to merge vocabulary shards
// across TP ranks; running an aclnn Max beside it would scan the row twice and
// would not guarantee the two agree on which duplicate maximum won.
bool qwen_argmax_fp32_rows_ascend(const float* d_logits, int* d_tokens,
                                  float* d_logits_out, int rows, int count,
                                  int token_offset, void* stream) {
    // Exactly the CUDA launcher's preconditions, including requiring
    // d_logits_out and accepting a negative token_offset, so a caller cannot
    // tell the two backends apart by what they reject.
    if (d_logits == nullptr || d_tokens == nullptr || d_logits_out == nullptr ||
        rows <= 0 || count <= 0) {
        return false;
    }
    // The kernel gives each block eight consecutive rows so its two 4-byte-per-row
    // output spans are whole 32-byte cache lines; block count follows that grouping,
    // not the row count, or the extra blocks would idle.
    const uint64_t groups = (static_cast<uint64_t>(rows) + 7) / 8;
    return aclrtlaunch_qwen_argmax_f32_rows_kernel(
               blocks_for(groups), resolve(stream),
               gm(d_logits), gm(d_tokens), gm(d_logits_out),
               static_cast<uint32_t>(rows), static_cast<uint32_t>(count),
               static_cast<int32_t>(token_offset)) == kLaunchOk;
}

// Measurement-only: streams tile_count 64x256 FP16 tiles and touches every
// element once, to establish the achievable HBM read bandwidth that the
// bandwidth-bound decode attention kernels are judged against. Fixed at the 30
// cores those kernels use so the result is an upper bound they could reach.
bool qwen_hbm_read_probe_ascend(const uint16_t* d_source, uint16_t* d_sink,
                                int tile_count, void* stream) {
    if (d_source == nullptr || d_sink == nullptr || tile_count <= 0) {
        return false;
    }
    return aclrtlaunch_qwen_hbm_read_probe_kernel(
               30, resolve(stream), gm(d_source), gm(d_sink),
               static_cast<uint32_t>(tile_count)) == kLaunchOk;
}

}  // namespace pocket
