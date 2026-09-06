#include "qwen_cuda_ops.hpp"
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>

namespace pocket {
namespace {

// The full-attention head dimension is a runtime value (128 on some configs,
// 256 on Qwen3.5), so the slot layout is derived rather than hardcoded.
constexpr int kMaxHeadDim = 256;
constexpr int kStoreThreads = 128;

__device__ __forceinline__ int slot_bytes_for(int head_dim) {
    return head_dim + head_dim / 2 + 4;
}

__device__ __forceinline__ uint8_t quantize_value_nibble(float v, float scale,
                                                          float minimum) {
    float normalized = (v - minimum) / scale;
    int q = __float2int_rn(normalized);
    return static_cast<uint8_t>(max(0, min(15, q)));
}

__device__ __forceinline__ float dequantize_value_nibble(uint8_t nibble,
                                                          float scale,
                                                          float minimum) {
    return static_cast<float>(nibble) * scale + minimum;
}

// Store K8V4: FP8 E5M2 key + 4-bit uniform value quantization.
// Layout: [0,128) key bytes, [128,192) packed values, [192,194) scale,
// [194,196) minimum.
__global__ void append_kv_cache_turboquant_k8v4_kernel(
    const uint16_t* __restrict__ d_k_rows_fp16,
    const uint16_t* __restrict__ d_v_rows_fp16,
    uint8_t* __restrict__ d_combined_cache,
    int seq_len, int kv_heads, int head_dim, int start_pos, int max_context) {

    const int token = blockIdx.x;
    const int head = blockIdx.y;
    if (token >= seq_len || head >= kv_heads) return;

    const int cache_pos = start_pos + token;
    if (cache_pos >= max_context) return;

    const int value_bytes = head_dim / 2;
    const int slot_bytes = slot_bytes_for(head_dim);

    const size_t k_offset = (static_cast<size_t>(token) * kv_heads + head) * head_dim;
    const size_t v_offset = k_offset;
    const size_t slot_offset =
        (static_cast<size_t>(cache_pos) * kv_heads + head) * slot_bytes;

    uint8_t* slot = d_combined_cache + slot_offset;

    // Compute value min/max for uniform quantization. The reduction must span
    // the whole block: with 128 threads a warp-only reduction would derive the
    // range from the first 32 channels and clamp everything else.
    float v_min = INFINITY;
    float v_max = -INFINITY;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        __half v_h;
        std::memcpy(&v_h, &d_v_rows_fp16[v_offset + d], 2);
        float v = __half2float(v_h);
        if (isfinite(v)) {
            v_min = fminf(v_min, v);
            v_max = fmaxf(v_max, v);
        }
    }

    __shared__ float s_min[kStoreThreads];
    __shared__ float s_max[kStoreThreads];
    s_min[threadIdx.x] = v_min;
    s_max[threadIdx.x] = v_max;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_min[threadIdx.x] = fminf(s_min[threadIdx.x], s_min[threadIdx.x + stride]);
            s_max[threadIdx.x] = fmaxf(s_max[threadIdx.x], s_max[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    v_min = s_min[0];
    v_max = s_max[0];
    if (!isfinite(v_min) || !isfinite(v_max)) {
        v_min = 0.0f;
        v_max = 0.0f;
    }

    // Round the metadata to FP16 before quantizing so the store and the load
    // reconstruct values from exactly the same scale and minimum.
    const __half min_h = __float2half(v_min);
    v_min = __half2float(min_h);
    float scale = fmaxf((v_max - v_min) / 15.0f, 1e-8f);
    const __half scale_h = __float2half(scale);
    scale = __half2float(scale_h);
    if (scale <= 0.0f) scale = 1e-8f;

    // Convert key to FP8 E5M2.
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        __half k_h;
        std::memcpy(&k_h, &d_k_rows_fp16[k_offset + d], 2);
        float k = __half2float(k_h);
        slot[d] = __nv_cvt_float_to_fp8(k, __NV_SATFINITE, __NV_E5M2);
    }

    // Pack values two dimensions per byte: even dimension in the low nibble,
    // odd in the high nibble. One thread owns a whole byte so no atomics are
    // needed and the write is a single store.
    for (int byte = threadIdx.x; byte < value_bytes; byte += blockDim.x) {
        const int d_low = byte * 2;
        const int d_high = d_low + 1;

        __half v_low_h, v_high_h;
        std::memcpy(&v_low_h, &d_v_rows_fp16[v_offset + d_low], 2);
        std::memcpy(&v_high_h, &d_v_rows_fp16[v_offset + d_high], 2);

        uint8_t low = quantize_value_nibble(__half2float(v_low_h), scale, v_min);
        uint8_t high = quantize_value_nibble(__half2float(v_high_h), scale, v_min);

        slot[head_dim + byte] = static_cast<uint8_t>(low | (high << 4));
    }

    // Write metadata.
    if (threadIdx.x == 0) {
        uint16_t scale_bits, min_bits;
        std::memcpy(&scale_bits, &scale_h, 2);
        std::memcpy(&min_bits, &min_h, 2);
        std::memcpy(slot + head_dim + value_bytes, &scale_bits, 2);
        std::memcpy(slot + head_dim + value_bytes + 2, &min_bits, 2);
    }
}

// Decode attention with TurboQuant K8V4 cache.
__global__ void gqa_decode_attention_turboquant_k8v4_score_kernel(
    const uint16_t* __restrict__ d_q_fp16,
    const uint8_t* __restrict__ d_combined_cache,
    float* __restrict__ d_scores,
    int q_heads, int kv_heads, int head_dim, int context_len, int max_context,
    int attention_window, int attention_sink_tokens, float q_scale) {

    const int head = blockIdx.x;
    if (head >= q_heads) return;

    const int kv_head = head / (q_heads / kv_heads);
    const int pos = blockIdx.y * blockDim.x + threadIdx.x;

    // Window/sink filtering.
    bool visible = true;
    if (attention_window > 0) {
        int sink_end = attention_sink_tokens;
        int window_start = max(sink_end, context_len - attention_window);
        visible = (pos < sink_end) || (pos >= window_start && pos < context_len);
    } else {
        visible = pos < context_len;
    }

    float score = visible ? 0.0f : -INFINITY;

    if (visible) {
        const size_t q_offset = static_cast<size_t>(head) * head_dim;
        const size_t slot_offset =
            (static_cast<size_t>(pos) * kv_heads + kv_head) * slot_bytes_for(head_dim);
        const uint8_t* slot = d_combined_cache + slot_offset;

        // Dot product Q · K.
        for (int d = 0; d < head_dim; ++d) {
            __half q_h;
            std::memcpy(&q_h, &d_q_fp16[q_offset + d], 2);
            float q = __half2float(q_h);

            uint8_t k_fp8 = slot[d];
            __half k_h = __nv_cvt_fp8_to_halfraw(k_fp8, __NV_E5M2);
            float k = __half2float(k_h);

            score += q * k;
        }
        score *= q_scale;
    }

    if (pos < max_context) {
        d_scores[static_cast<size_t>(head) * max_context + pos] = score;
    }
}

// One block per (position, kv head); each thread expands one channel. Writes
// dense K/V with the FP16 cache's [max_context, kv_heads, head_dim] layout so
// the existing tensor-core prefill kernel can consume it unchanged.
__global__ void dequant_kv_turboquant_k8v4_kernel(
    const uint8_t* __restrict__ d_combined_cache,
    uint16_t* __restrict__ d_k_dense_fp16,
    uint16_t* __restrict__ d_v_dense_fp16,
    int context_len, int kv_heads, int head_dim, int max_context) {

    const int pos = blockIdx.x;
    const int kv_head = blockIdx.y;
    const int dim = threadIdx.x;
    if (pos >= context_len || kv_head >= kv_heads || dim >= head_dim) return;

    const int value_bytes = head_dim / 2;
    const int slot_bytes = slot_bytes_for(head_dim);
    const uint8_t* slot = d_combined_cache +
        (static_cast<size_t>(pos) * kv_heads + kv_head) * slot_bytes;

    uint16_t scale_bits, min_bits;
    std::memcpy(&scale_bits, slot + head_dim + value_bytes, 2);
    std::memcpy(&min_bits, slot + head_dim + value_bytes + 2, 2);
    __half scale_h, min_h;
    std::memcpy(&scale_h, &scale_bits, 2);
    std::memcpy(&min_h, &min_bits, 2);

    const __half k_h = __nv_cvt_fp8_to_halfraw(slot[dim], __NV_E5M2);

    const uint8_t packed = slot[head_dim + (dim >> 1)];
    const uint8_t nibble = (dim & 1) ? (packed >> 4) : (packed & 0x0F);
    const float v = dequantize_value_nibble(nibble, __half2float(scale_h),
                                            __half2float(min_h));
    const __half v_h = __float2half(v);

    const size_t out_offset =
        (static_cast<size_t>(pos) * kv_heads + kv_head) * head_dim + dim;
    uint16_t k_bits, v_bits;
    std::memcpy(&k_bits, &k_h, 2);
    std::memcpy(&v_bits, &v_h, 2);
    d_k_dense_fp16[out_offset] = k_bits;
    d_v_dense_fp16[out_offset] = v_bits;
}

// One block per query head. Masked-out positions already hold -inf, so they
// contribute nothing and stay zero after normalization.
__global__ void softmax_rows_f32_kernel(float* __restrict__ d_scores,
                                        int row_stride, int valid_cols) {
    const int row = blockIdx.x;
    float* scores = d_scores + static_cast<size_t>(row) * row_stride;

    __shared__ float s_reduce[256];

    float local_max = -INFINITY;
    for (int col = threadIdx.x; col < valid_cols; col += blockDim.x) {
        local_max = fmaxf(local_max, scores[col]);
    }
    s_reduce[threadIdx.x] = local_max;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_reduce[threadIdx.x] = fmaxf(s_reduce[threadIdx.x],
                                          s_reduce[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const float row_max = s_reduce[0];
    __syncthreads();

    float local_sum = 0.0f;
    for (int col = threadIdx.x; col < valid_cols; col += blockDim.x) {
        const float score = scores[col];
        const float weight = isfinite(score) ? __expf(score - row_max) : 0.0f;
        scores[col] = weight;
        local_sum += weight;
    }
    s_reduce[threadIdx.x] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            s_reduce[threadIdx.x] += s_reduce[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float row_sum = s_reduce[0];
    const float inv_sum = row_sum > 0.0f ? 1.0f / row_sum : 0.0f;

    for (int col = threadIdx.x; col < valid_cols; col += blockDim.x) {
        scores[col] *= inv_sum;
    }
}

__global__ void gqa_decode_attention_turboquant_k8v4_output_kernel(
    const uint16_t* __restrict__ d_q_fp16,
    const uint8_t* __restrict__ d_combined_cache,
    const float* __restrict__ d_probs,
    uint16_t* __restrict__ d_out_fp16,
    int q_heads, int kv_heads, int head_dim, int context_len, int max_context,
    int attention_window, int attention_sink_tokens) {

    const int head = blockIdx.x;
    const int dim = blockIdx.y * blockDim.x + threadIdx.x;
    if (head >= q_heads || dim >= head_dim) return;

    const int value_bytes = head_dim / 2;
    const int slot_bytes = slot_bytes_for(head_dim);

    const int kv_head = head / (q_heads / kv_heads);
    const float* head_probs = d_probs + static_cast<size_t>(head) * max_context;

    float accum = 0.0f;
    for (int pos = 0; pos < context_len; ++pos) {
        bool visible = true;
        if (attention_window > 0) {
            int sink_end = attention_sink_tokens;
            int window_start = max(sink_end, context_len - attention_window);
            visible = (pos < sink_end) || (pos >= window_start);
        }
        if (!visible) continue;

        float prob = head_probs[pos];
        const size_t slot_offset =
            (static_cast<size_t>(pos) * kv_heads + kv_head) * slot_bytes;
        const uint8_t* slot = d_combined_cache + slot_offset;

        // Read scale/min metadata.
        uint16_t scale_bits, min_bits;
        std::memcpy(&scale_bits, slot + head_dim + value_bytes, 2);
        std::memcpy(&min_bits, slot + head_dim + value_bytes + 2, 2);
        __half scale_h, min_h;
        std::memcpy(&scale_h, &scale_bits, 2);
        std::memcpy(&min_h, &min_bits, 2);
        float scale = __half2float(scale_h);
        float minimum = __half2float(min_h);

        // Decode 4-bit value.
        int byte_idx = head_dim + (dim >> 1);
        uint8_t packed = slot[byte_idx];
        uint8_t nibble = (dim & 1) ? (packed >> 4) : (packed & 0x0F);
        float v = dequantize_value_nibble(nibble, scale, minimum);

        accum += prob * v;
    }

    const size_t out_offset = static_cast<size_t>(head) * head_dim + dim;
    __half out_h = __float2half(accum);
    uint16_t out_bits;
    std::memcpy(&out_bits, &out_h, 2);
    d_out_fp16[out_offset] = out_bits;
}

// Prefill attention with TurboQuant K8V4 cache.// One block per (token, head). blockDim == head_dim, so each thread owns one
// output channel. Scores are computed one position per warp with a shuffle
// reduction: letting every channel-thread redo the full head_dim dot product
// instead costs head_dim times more score work and dominated prefill.
__global__ void gqa_prefill_attention_turboquant_k8v4_kernel(
    const uint16_t* __restrict__ d_q_rows_fp16,
    const uint8_t* __restrict__ d_combined_cache,
    uint16_t* __restrict__ d_out_rows_fp16,
    int seq_len, int q_heads, int kv_heads, int head_dim, int position_offset,
    int max_context, int attention_window, int attention_sink_tokens,
    float q_scale) {

    const int token = blockIdx.x;
    const int head = blockIdx.y;
    const int dim = threadIdx.x;
    if (token >= seq_len || head >= q_heads) return;

    const int value_bytes = head_dim / 2;
    const int slot_bytes = slot_bytes_for(head_dim);
    const int kv_head = head / (q_heads / kv_heads);
    const int q_pos = position_offset + token;
    const int visible_end = q_pos + 1;

    const int lane = dim & 31;
    const int warp = dim >> 5;
    const int warps = blockDim.x >> 5;

    // Causal visibility, optionally restricted to a sink prefix plus a trailing
    // window measured against this query's own position.
    const int sink_end = attention_window > 0 ? attention_sink_tokens : 0;
    const int window_start = attention_window > 0
        ? max(sink_end, visible_end - attention_window)
        : 0;

    const size_t q_offset =
        (static_cast<size_t>(token) * q_heads + head) * head_dim;

    __shared__ float s_q[kMaxHeadDim];
    __shared__ float s_score[kMaxHeadDim / 32];

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        __half q_h;
        std::memcpy(&q_h, &d_q_rows_fp16[q_offset + d], 2);
        s_q[d] = __half2float(q_h);
    }
    __syncthreads();

    float max_score = -INFINITY;
    float sum_exp = 0.0f;
    float accum = 0.0f;

    // Walk the attended positions in tiles of one position per warp.
    for (int base = 0; base < visible_end; base += warps) {
        const int pos = base + warp;
        const bool in_range = pos < visible_end;
        const bool masked = attention_window > 0 && pos >= sink_end &&
                            pos < window_start;

        if (warp < warps) {
            float partial = 0.0f;
            if (in_range && !masked) {
                const size_t slot_offset =
                    (static_cast<size_t>(pos) * kv_heads + kv_head) * slot_bytes;
                const uint8_t* slot = d_combined_cache + slot_offset;
                for (int d = lane; d < head_dim; d += 32) {
                    __half k_h = __nv_cvt_fp8_to_halfraw(slot[d], __NV_E5M2);
                    partial += s_q[d] * __half2float(k_h);
                }
            }
            for (int offset = 16; offset > 0; offset >>= 1) {
                partial += __shfl_xor_sync(0xffffffff, partial, offset);
            }
            if (lane == 0) {
                s_score[warp] = (in_range && !masked) ? partial * q_scale
                                                      : -INFINITY;
            }
        }
        __syncthreads();

        // Each thread folds this tile into its own channel accumulator. The
        // running max and denominator are channel-independent, so every thread
        // recomputes the same values rather than paying another reduction.
        for (int w = 0; w < warps; ++w) {
            const int tile_pos = base + w;
            if (tile_pos >= visible_end) break;
            const float score = s_score[w];
            if (!isfinite(score)) continue;

            const size_t slot_offset =
                (static_cast<size_t>(tile_pos) * kv_heads + kv_head) * slot_bytes;
            const uint8_t* slot = d_combined_cache + slot_offset;

            uint16_t scale_bits, min_bits;
            std::memcpy(&scale_bits, slot + head_dim + value_bytes, 2);
            std::memcpy(&min_bits, slot + head_dim + value_bytes + 2, 2);
            __half scale_h, min_h;
            std::memcpy(&scale_h, &scale_bits, 2);
            std::memcpy(&min_h, &min_bits, 2);
            const float scale = __half2float(scale_h);
            const float minimum = __half2float(min_h);

            const uint8_t packed = slot[head_dim + (dim >> 1)];
            const uint8_t nibble = (dim & 1) ? (packed >> 4) : (packed & 0x0F);
            const float v = dequantize_value_nibble(nibble, scale, minimum);

            if (score > max_score) {
                const float rescale =
                    isfinite(max_score) ? __expf(max_score - score) : 0.0f;
                sum_exp = sum_exp * rescale + 1.0f;
                accum = accum * rescale + v;
                max_score = score;
            } else {
                const float weight = __expf(score - max_score);
                sum_exp += weight;
                accum += weight * v;
            }
        }
        __syncthreads();
    }

    if (dim >= head_dim) return;
    const float output = sum_exp > 0.0f ? accum / sum_exp : 0.0f;
    const size_t out_offset =
        (static_cast<size_t>(token) * q_heads + head) * head_dim + dim;
    __half out_h = __float2half(output);
    uint16_t out_bits;
    std::memcpy(&out_bits, &out_h, 2);
    d_out_rows_fp16[out_offset] = out_bits;
}

}  // namespace

bool qwen_append_kv_cache_turboquant_k8v4_cuda(
    const uint16_t* d_k_rows_fp16, const uint16_t* d_v_rows_fp16,
    uint8_t* d_combined_cache, int seq_len, int kv_heads, int head_dim,
    int start_pos, int max_context, void* stream) {

    if (seq_len <= 0 || kv_heads <= 0 || head_dim <= 0 ||
        head_dim > kMaxHeadDim || (head_dim % 2) != 0 ||
        start_pos < 0 || max_context <= 0) {
        return false;
    }
    if (!d_k_rows_fp16 || !d_v_rows_fp16 || !d_combined_cache) {
        return false;
    }
    if (start_pos >= max_context) return true;  // Out of bounds, no-op.

    // Every byte of each touched slot is written by the kernel, so no clearing
    // pass is required.
    dim3 grid(seq_len, kv_heads);
    dim3 block(kStoreThreads);
    append_kv_cache_turboquant_k8v4_kernel<<<grid, block, 0,
                                             static_cast<cudaStream_t>(stream)>>>(
        d_k_rows_fp16, d_v_rows_fp16, d_combined_cache, seq_len, kv_heads,
        head_dim, start_pos, max_context);

    return cudaGetLastError() == cudaSuccess;
}

bool qwen_gqa_decode_attention_turboquant_k8v4_cuda(
    const uint16_t* d_q_fp16, const uint8_t* d_combined_cache,
    uint16_t* d_out_fp16, float* d_score_scratch, int q_heads, int kv_heads,
    int head_dim, int context_len, int max_context, int attention_window,
    int attention_sink_tokens, void* stream) {

    if (q_heads <= 0 || kv_heads <= 0 || head_dim <= 0 ||
        head_dim > kMaxHeadDim || (head_dim % 2) != 0 || kv_heads > q_heads ||
        (q_heads % kv_heads) != 0 || context_len <= 0 || max_context <= 0 ||
        context_len > max_context) {
        return false;
    }
    if (!d_q_fp16 || !d_combined_cache || !d_out_fp16 || !d_score_scratch) {
        return false;
    }

    const float q_scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    // Score phase.
    {
        dim3 grid(q_heads, (max_context + 127) / 128);
        dim3 block(128);
        gqa_decode_attention_turboquant_k8v4_score_kernel<<<grid, block, 0,
                                                            static_cast<cudaStream_t>(stream)>>>(
            d_q_fp16, d_combined_cache, d_score_scratch, q_heads, kv_heads,
            head_dim, context_len, max_context, attention_window,
            attention_sink_tokens, q_scale);
    }

    // Softmax phase over the masked score rows.
    {
        dim3 grid(q_heads);
        dim3 block(256);
        softmax_rows_f32_kernel<<<grid, block, 0,
                                  static_cast<cudaStream_t>(stream)>>>(
            d_score_scratch, max_context, context_len);
    }

    // Output phase.
    {
        dim3 grid(q_heads, (head_dim + 127) / 128);
        dim3 block(128);
        gqa_decode_attention_turboquant_k8v4_output_kernel<<<grid, block, 0,
                                                             static_cast<cudaStream_t>(stream)>>>(
            d_q_fp16, d_combined_cache, d_score_scratch, d_out_fp16, q_heads,
            kv_heads, head_dim, context_len, max_context, attention_window,
            attention_sink_tokens);
    }

    return cudaGetLastError() == cudaSuccess;
}

bool qwen_turboquant_k8v4_dequant_kv_cuda(
    const uint8_t* d_combined_cache, uint16_t* d_k_dense_fp16,
    uint16_t* d_v_dense_fp16, int context_len, int kv_heads, int head_dim,
    int max_context, void* stream) {

    if (context_len <= 0 || kv_heads <= 0 || head_dim <= 0 ||
        head_dim > kMaxHeadDim || (head_dim % 2) != 0 || max_context <= 0 ||
        context_len > max_context) {
        return false;
    }
    if (!d_combined_cache || !d_k_dense_fp16 || !d_v_dense_fp16) return false;

    dim3 grid(context_len, kv_heads);
    dim3 block(head_dim);
    dequant_kv_turboquant_k8v4_kernel<<<grid, block, 0,
                                       static_cast<cudaStream_t>(stream)>>>(
        d_combined_cache, d_k_dense_fp16, d_v_dense_fp16, context_len, kv_heads,
        head_dim, max_context);

    return cudaGetLastError() == cudaSuccess;
}

bool qwen_gqa_prefill_attention_turboquant_k8v4_cuda(
    const uint16_t* d_q_rows_fp16, const uint8_t* d_combined_cache,
    uint16_t* d_out_rows_fp16, int seq_len, int q_heads, int kv_heads,
    int head_dim, int position_offset, int max_context, int attention_window,
    int attention_sink_tokens, void* stream) {

    if (seq_len <= 0 || q_heads <= 0 || kv_heads <= 0 || head_dim <= 0 ||
        head_dim > kMaxHeadDim || (head_dim % 2) != 0 || kv_heads > q_heads ||
        (q_heads % kv_heads) != 0 || position_offset < 0 || max_context <= 0) {
        return false;
    }
    if (!d_q_rows_fp16 || !d_combined_cache || !d_out_rows_fp16) {
        return false;
    }

    const float q_scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    dim3 grid(seq_len, q_heads);
    dim3 block(head_dim);
    gqa_prefill_attention_turboquant_k8v4_kernel<<<grid, block, 0,
                                                   static_cast<cudaStream_t>(stream)>>>(
        d_q_rows_fp16, d_combined_cache, d_out_rows_fp16, seq_len, q_heads,
        kv_heads, head_dim, position_offset, max_context, attention_window,
        attention_sink_tokens, q_scale);

    return cudaGetLastError() == cudaSuccess;
}

}  // namespace pocket
