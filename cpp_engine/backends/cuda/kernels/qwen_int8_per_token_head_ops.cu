#include "qwen_cuda_ops.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <algorithm>

namespace pocket {
namespace {

// Append K/V rows to INT8 per-token-head cache with dynamic quantization.
// Each block handles one (token, kv_head), threads compute per-head max
// and quantize channels. Scale is max(abs(values)) / 127.0 per token/head.
__global__ void append_int8_per_token_head_kernel(
    const uint16_t* __restrict__ d_k_rows_fp16,
    const uint16_t* __restrict__ d_v_rows_fp16,
    int8_t* __restrict__ d_k_cache_int8,
    int8_t* __restrict__ d_v_cache_int8,
    uint16_t* __restrict__ d_k_scale_fp16,
    uint16_t* __restrict__ d_v_scale_fp16,
    int seq_len, int kv_heads, int head_dim, int start_pos, int max_context) {
    const int token_in_seq = blockIdx.x;
    const int kv_head = blockIdx.y;
    if (token_in_seq >= seq_len || kv_head >= kv_heads) return;

    const int position = start_pos + token_in_seq;
    if (position >= max_context) return;

    // Input row offset: [seq_len, kv_heads, head_dim]
    const size_t row_offset = (static_cast<size_t>(token_in_seq) * kv_heads + kv_head) * head_dim;
    // Cache offset: [max_context, kv_heads, head_dim]
    const size_t cache_offset = (static_cast<size_t>(position) * kv_heads + kv_head) * head_dim;
    // Scale offset: [max_context, kv_heads]
    const size_t scale_offset = static_cast<size_t>(position) * kv_heads + kv_head;

    __shared__ float shared_k_max;
    __shared__ float shared_v_max;

    // Phase 1: Find max(abs(value)) across head_dim channels per head
    float local_k_max = 0.0f;
    float local_v_max = 0.0f;
    for (int ch = threadIdx.x; ch < head_dim; ch += blockDim.x) {
        const float k_val = __half2float(__ushort_as_half(d_k_rows_fp16[row_offset + ch]));
        const float v_val = __half2float(__ushort_as_half(d_v_rows_fp16[row_offset + ch]));
        local_k_max = fmaxf(local_k_max, fabsf(k_val));
        local_v_max = fmaxf(local_v_max, fabsf(v_val));
    }

    // Block-wide reduction for max
    __shared__ float scratch_k[256];
    __shared__ float scratch_v[256];
    scratch_k[threadIdx.x] = local_k_max;
    scratch_v[threadIdx.x] = local_v_max;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            scratch_k[threadIdx.x] = fmaxf(scratch_k[threadIdx.x], scratch_k[threadIdx.x + stride]);
            scratch_v[threadIdx.x] = fmaxf(scratch_v[threadIdx.x], scratch_v[threadIdx.x + stride]);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        shared_k_max = scratch_k[0];
        shared_v_max = scratch_v[0];
    }
    __syncthreads();

    // Compute scale: max / 127.0, clamped to avoid division by zero
    const float k_scale = fmaxf(shared_k_max / 127.0f, 1e-8f);
    const float v_scale = fmaxf(shared_v_max / 127.0f, 1e-8f);

    // Phase 2: Quantize and store
    for (int ch = threadIdx.x; ch < head_dim; ch += blockDim.x) {
        const float k_val = __half2float(__ushort_as_half(d_k_rows_fp16[row_offset + ch]));
        const float v_val = __half2float(__ushort_as_half(d_v_rows_fp16[row_offset + ch]));

        const int8_t k_code = static_cast<int8_t>(__float2int_rn(
            fmaxf(-128.0f, fminf(127.0f, k_val / k_scale))));
        const int8_t v_code = static_cast<int8_t>(__float2int_rn(
            fmaxf(-128.0f, fminf(127.0f, v_val / v_scale))));

        d_k_cache_int8[cache_offset + ch] = k_code;
        d_v_cache_int8[cache_offset + ch] = v_code;
    }

    // Store scale (FP16)
    if (threadIdx.x == 0) {
        d_k_scale_fp16[scale_offset] = __float2half_rn(k_scale);
        d_v_scale_fp16[scale_offset] = __float2half_rn(v_scale);
    }
}

// Bulk dequantize INT8 per-token-head cache into dense FP16 buffers.
// One block per (position, kv_head), threads dequantize head_dim channels.
__global__ void dequant_int8_per_token_head_kernel(
    const int8_t* __restrict__ d_k_cache_int8,
    const int8_t* __restrict__ d_v_cache_int8,
    const uint16_t* __restrict__ d_k_scale_fp16,
    const uint16_t* __restrict__ d_v_scale_fp16,
    uint16_t* __restrict__ d_k_dense_fp16,
    uint16_t* __restrict__ d_v_dense_fp16,
    int context_len, int kv_heads, int head_dim, int max_context) {
    const int position = blockIdx.x;
    const int kv_head = blockIdx.y;
    if (position >= context_len || kv_head >= kv_heads) return;

    // Cache layout: [max_context, kv_heads, head_dim]
    const size_t cache_offset = (static_cast<size_t>(position) * kv_heads + kv_head) * head_dim;
    // Scale layout: [max_context, kv_heads]
    const size_t scale_offset = static_cast<size_t>(position) * kv_heads + kv_head;
    // Dense output layout: [context_len, kv_heads, head_dim]
    const size_t dense_offset = (static_cast<size_t>(position) * kv_heads + kv_head) * head_dim;

    const float k_scale = __half2float(__ushort_as_half(d_k_scale_fp16[scale_offset]));
    const float v_scale = __half2float(__ushort_as_half(d_v_scale_fp16[scale_offset]));

    for (int ch = threadIdx.x; ch < head_dim; ch += blockDim.x) {
        const int8_t k_code = d_k_cache_int8[cache_offset + ch];
        const int8_t v_code = d_v_cache_int8[cache_offset + ch];

        const float k_val = static_cast<float>(k_code) * k_scale;
        const float v_val = static_cast<float>(v_code) * v_scale;

        d_k_dense_fp16[dense_offset + ch] = __float2half_rn(k_val);
        d_v_dense_fp16[dense_offset + ch] = __float2half_rn(v_val);
    }
}

// Decode attention with INT8 per-token-head cache. Inline dequant during
// score/value accumulation. One block per query head.
__global__ void decode_attention_int8_per_token_head_kernel(
    const uint16_t* __restrict__ d_q_fp16,
    const int8_t* __restrict__ d_k_cache_int8,
    const int8_t* __restrict__ d_v_cache_int8,
    const uint16_t* __restrict__ d_k_scale_fp16,
    const uint16_t* __restrict__ d_v_scale_fp16,
    uint16_t* __restrict__ d_out_fp16,
    float* __restrict__ d_score_scratch,
    int q_heads, int kv_heads, int head_dim, int context_len, int max_context,
    int attention_window, int attention_sink_tokens) {
    const int q_head = blockIdx.x;
    if (q_head >= q_heads) return;

    const int kv_head = q_head % kv_heads; // GQA mapping
    const size_t q_offset = static_cast<size_t>(q_head) * head_dim;
    const float scale = 1.0f / sqrtf(static_cast<float>(head_dim));

    // Phase 1: Compute Q·K scores with inline INT8 dequant
    float max_score = -1e38f;
    for (int pos = threadIdx.x; pos < context_len; pos += blockDim.x) {
        // Window/sink filtering
        const int distance = context_len - 1 - pos;
        const bool in_window = (attention_window == 0) ||
                               (distance < attention_window) ||
                               (pos < attention_sink_tokens);
        if (!in_window) {
            d_score_scratch[q_head * max_context + pos] = -1e38f;
            continue;
        }

        const size_t k_offset = (static_cast<size_t>(pos) * kv_heads + kv_head) * head_dim;
        const size_t scale_offset = static_cast<size_t>(pos) * kv_heads + kv_head;
        const float k_scale = __half2float(__ushort_as_half(d_k_scale_fp16[scale_offset]));

        float score = 0.0f;
        for (int ch = 0; ch < head_dim; ++ch) {
            const float q_val = __half2float(__ushort_as_half(d_q_fp16[q_offset + ch]));
            const int8_t k_code = d_k_cache_int8[k_offset + ch];
            const float k_val = static_cast<float>(k_code) * k_scale;
            score += q_val * k_val;
        }
        score *= scale;
        d_score_scratch[q_head * max_context + pos] = score;
        max_score = fmaxf(max_score, score);
    }

    // Block-wide max reduction
    __shared__ float shared_max;
    __shared__ float scratch[128];
    scratch[threadIdx.x] = max_score;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] = fmaxf(scratch[threadIdx.x], scratch[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) shared_max = scratch[0];
    __syncthreads();

    // Phase 2: Softmax normalization
    float sum_exp = 0.0f;
    for (int pos = threadIdx.x; pos < context_len; pos += blockDim.x) {
        float score = d_score_scratch[q_head * max_context + pos];
        if (score > -1e37f) {
            score = expf(score - shared_max);
            d_score_scratch[q_head * max_context + pos] = score;
            sum_exp += score;
        } else {
            d_score_scratch[q_head * max_context + pos] = 0.0f;
        }
    }

    // Block-wide sum reduction
    __shared__ float shared_sum;
    scratch[threadIdx.x] = sum_exp;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) shared_sum = fmaxf(scratch[0], 1e-8f);
    __syncthreads();

    // Phase 3: Weighted value sum with inline INT8 dequant
    for (int ch = threadIdx.x; ch < head_dim; ch += blockDim.x) {
        float accum = 0.0f;
        for (int pos = 0; pos < context_len; ++pos) {
            const float weight = d_score_scratch[q_head * max_context + pos] / shared_sum;
            if (weight > 1e-8f) {
                const size_t v_offset = (static_cast<size_t>(pos) * kv_heads + kv_head) * head_dim;
                const size_t scale_offset = static_cast<size_t>(pos) * kv_heads + kv_head;
                const float v_scale = __half2float(__ushort_as_half(d_v_scale_fp16[scale_offset]));
                const int8_t v_code = d_v_cache_int8[v_offset + ch];
                const float v_val = static_cast<float>(v_code) * v_scale;
                accum += weight * v_val;
            }
        }
        d_out_fp16[q_offset + ch] = __float2half_rn(accum);
    }
}

} // namespace

bool qwen_append_kv_cache_int8_per_token_head_cuda(
    const uint16_t* d_k_rows_fp16, const uint16_t* d_v_rows_fp16,
    int8_t* d_k_cache_int8, int8_t* d_v_cache_int8,
    uint16_t* d_k_scale_fp16, uint16_t* d_v_scale_fp16,
    int seq_len, int kv_heads, int head_dim, int start_pos, int max_context,
    void* stream) {
    if (!d_k_rows_fp16 || !d_v_rows_fp16 || !d_k_cache_int8 || !d_v_cache_int8 ||
        !d_k_scale_fp16 || !d_v_scale_fp16 || seq_len <= 0 || kv_heads <= 0 ||
        head_dim <= 0 || start_pos < 0 || max_context <= 0 ||
        start_pos + seq_len > max_context) {
        return false;
    }
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    const dim3 grid(seq_len, kv_heads);
    const int block = 256;
    append_int8_per_token_head_kernel<<<grid, block, 0, s>>>(
        d_k_rows_fp16, d_v_rows_fp16, d_k_cache_int8, d_v_cache_int8,
        d_k_scale_fp16, d_v_scale_fp16, seq_len, kv_heads, head_dim,
        start_pos, max_context);
    return cudaGetLastError() == cudaSuccess;
}

bool qwen_int8_dequant_kv_cache_cuda(
    const int8_t* d_k_cache_int8, const int8_t* d_v_cache_int8,
    const uint16_t* d_k_scale_fp16, const uint16_t* d_v_scale_fp16,
    uint16_t* d_k_dense_fp16, uint16_t* d_v_dense_fp16,
    int context_len, int kv_heads, int head_dim, int max_context,
    void* stream) {
    if (!d_k_cache_int8 || !d_v_cache_int8 || !d_k_scale_fp16 || !d_v_scale_fp16 ||
        !d_k_dense_fp16 || !d_v_dense_fp16 || context_len <= 0 || kv_heads <= 0 ||
        head_dim <= 0 || max_context <= 0 || context_len > max_context) {
        return false;
    }
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    const dim3 grid(context_len, kv_heads);
    const int block = 256;
    dequant_int8_per_token_head_kernel<<<grid, block, 0, s>>>(
        d_k_cache_int8, d_v_cache_int8, d_k_scale_fp16, d_v_scale_fp16,
        d_k_dense_fp16, d_v_dense_fp16, context_len, kv_heads, head_dim,
        max_context);
    return cudaGetLastError() == cudaSuccess;
}

bool qwen_gqa_decode_attention_int8_per_token_head_cuda(
    const uint16_t* d_q_fp16, const int8_t* d_k_cache_int8,
    const int8_t* d_v_cache_int8, const uint16_t* d_k_scale_fp16,
    const uint16_t* d_v_scale_fp16, uint16_t* d_out_fp16, float* d_score_scratch,
    int q_heads, int kv_heads, int head_dim, int context_len, int max_context,
    int attention_window, int attention_sink_tokens, void* stream) {
    if (!d_q_fp16 || !d_k_cache_int8 || !d_v_cache_int8 || !d_k_scale_fp16 ||
        !d_v_scale_fp16 || !d_out_fp16 || !d_score_scratch || q_heads <= 0 ||
        kv_heads <= 0 || head_dim <= 0 || context_len <= 0 || max_context <= 0 ||
        context_len > max_context || q_heads % kv_heads != 0) {
        return false;
    }
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    const int block = 128;
    decode_attention_int8_per_token_head_kernel<<<q_heads, block, 0, s>>>(
        d_q_fp16, d_k_cache_int8, d_v_cache_int8, d_k_scale_fp16, d_v_scale_fp16,
        d_out_fp16, d_score_scratch, q_heads, kv_heads, head_dim, context_len,
        max_context, attention_window, attention_sink_tokens);
    return cudaGetLastError() == cudaSuccess;
}

} // namespace pocket
