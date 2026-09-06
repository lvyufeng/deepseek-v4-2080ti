#include "cuda_ops.hpp"

#include <cuda_runtime.h>
#include <sm_61_intrinsics.h>

#include <algorithm>
#include <climits>
#include <cmath>

namespace pocket {
namespace {

__device__ __forceinline__ int pocket_dp4a_i8(int a, int b, int acc) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 610)
    return __dp4a(a, b, acc);
#else
    const unsigned ua = static_cast<unsigned>(a);
    const unsigned ub = static_cast<unsigned>(b);
    acc += static_cast<int>(static_cast<int8_t>(ua & 0xffu)) * static_cast<int>(static_cast<int8_t>(ub & 0xffu));
    acc += static_cast<int>(static_cast<int8_t>((ua >> 8) & 0xffu)) * static_cast<int>(static_cast<int8_t>((ub >> 8) & 0xffu));
    acc += static_cast<int>(static_cast<int8_t>((ua >> 16) & 0xffu)) * static_cast<int>(static_cast<int8_t>((ub >> 16) & 0xffu));
    acc += static_cast<int>(static_cast<int8_t>((ua >> 24) & 0xffu)) * static_cast<int>(static_cast<int8_t>((ub >> 24) & 0xffu));
    return acc;
#endif
}

__global__ void gguf_route_slots_from_indices_kernel(
    const int64_t* __restrict__ indices,
    int64_t* __restrict__ route_slots,
    int topk,
    int expert_start,
    int experts_per_rank) {
    const int k = threadIdx.x;
    if (k >= topk) return;
    const int64_t eid = indices[k];
    const int64_t local = eid - static_cast<int64_t>(expert_start);
    route_slots[k] = (local >= 0 && local < experts_per_rank) ? local : -1;
}

__global__ void argmax_fp32_kernel(
    const float* __restrict__ logits,
    int* __restrict__ out_token,
    float* __restrict__ out_logit,
    int count,
    int token_offset) {
    const int tid = threadIdx.x;
    const int threads = blockDim.x;
    extern __shared__ char smem_raw[];
    float* s_val = reinterpret_cast<float*>(smem_raw);
    int* s_idx = reinterpret_cast<int*>(s_val + threads);

    float best = -INFINITY;
    int best_idx = 0;
    for (int i = tid; i < count; i += threads) {
        const float v = logits[i];
        if (v > best || (v == best && i < best_idx)) {
            best = v;
            best_idx = i;
        }
    }
    s_val[tid] = best;
    s_idx[tid] = best_idx;
    __syncthreads();
    for (int stride = threads >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const float ov = s_val[tid + stride];
            const int oi = s_idx[tid + stride];
            if (ov > s_val[tid] || (ov == s_val[tid] && oi < s_idx[tid])) {
                s_val[tid] = ov;
                s_idx[tid] = oi;
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        *out_token = token_offset + s_idx[0];
        *out_logit = s_val[0];
    }
}

__global__ void argmax_fp32_rows_kernel(
    const float* __restrict__ logits,
    int* __restrict__ out_tokens,
    float* __restrict__ out_logits,
    int rows,
    int count,
    int token_offset) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const int tid = threadIdx.x;
    const int threads = blockDim.x;
    extern __shared__ char smem_raw[];
    float* s_val = reinterpret_cast<float*>(smem_raw);
    int* s_idx = reinterpret_cast<int*>(s_val + threads);
    float best = -INFINITY;
    int best_idx = 0;
    const float* row_logits = logits + static_cast<size_t>(row) * count;
    for (int i = tid; i < count; i += threads) {
        const float value = row_logits[i];
        if (value > best || (value == best && i < best_idx)) {
            best = value;
            best_idx = i;
        }
    }
    s_val[tid] = best;
    s_idx[tid] = best_idx;
    __syncthreads();
    for (int stride = threads >> 1; stride > 0; stride >>= 1) {
        if (tid < stride) {
            const float other_value = s_val[tid + stride];
            const int other_idx = s_idx[tid + stride];
            if (other_value > s_val[tid] ||
                (other_value == s_val[tid] && other_idx < s_idx[tid])) {
                s_val[tid] = other_value;
                s_idx[tid] = other_idx;
            }
        }
        __syncthreads();
    }
    if (tid == 0) {
        out_tokens[row] = token_offset + s_idx[0];
        out_logits[row] = s_val[0];
    }
}

__device__ __forceinline__ float sigmoidf_fast(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__global__ void hc_pre_float_kernel(
    const float* h4,
    const float* fn,
    const float* scale,
    const float* base,
    float* x,
    float* post_out,
    float* comb_out,
    int dim) {
    constexpr float eps = 1e-6f;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    constexpr int kNumWarps = 8;
    __shared__ float mixes[24];
    __shared__ float pre[4];
    __shared__ float post[4];
    __shared__ float comb[16];
    __shared__ float partial[kNumWarps];
    const int n = 4 * dim;
    float sum_sq = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        const float v = h4[i];
        sum_sq += v * v;
    }
    for (int off = 16; off > 0; off >>= 1) sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, off);
    if (lane == 0) partial[warp_id] = sum_sq;
    __syncthreads();
    float total_sq = 0.0f;
    if (warp_id == 0) {
        total_sq = lane < kNumWarps ? partial[lane] : 0.0f;
        for (int off = kNumWarps / 2; off > 0; off >>= 1) total_sq += __shfl_xor_sync(0xffffffff, total_sq, off);
        if (lane == 0) partial[0] = total_sq;
    }
    __syncthreads();
    const float rsqrt = rsqrtf(partial[0] / static_cast<float>(n) + eps);
    for (int r_base = 0; r_base < 24; r_base += kNumWarps) {
        const int r = r_base + warp_id;
        float dot = 0.0f;
        if (r < 24) {
            const float* row = fn + static_cast<size_t>(r) * n;
            for (int i = lane; i < n; i += 32) dot += row[i] * h4[i];
            for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xffffffff, dot, off);
            if (lane == 0) mixes[r] = dot * rsqrt;
        }
    }
    __syncthreads();
    if (tid < 4) {
        pre[tid] = sigmoidf_fast(mixes[tid] * scale[0] + base[tid]) + eps;
        post[tid] = 2.0f * sigmoidf_fast(mixes[4 + tid] * scale[1] + base[4 + tid]);
    }
    if (tid < 16) {
        float vals[4];
        const int row = tid / 4;
        const int col = tid - row * 4;
        const float* cm = mixes + 8 + row * 4;
        float max_v = cm[0] * scale[2] + base[8 + row * 4];
        for (int c = 1; c < 4; ++c) max_v = fmaxf(max_v, cm[c] * scale[2] + base[8 + row * 4 + c]);
        float sum = 0.0f;
        for (int c = 0; c < 4; ++c) {
            vals[c] = expf(cm[c] * scale[2] + base[8 + row * 4 + c] - max_v);
            sum += vals[c];
        }
        comb[tid] = vals[col] / sum + eps;
    }
    __syncthreads();
    if (tid < 4) {
        float col_sum = 0.0f;
        for (int r = 0; r < 4; ++r) col_sum += comb[r * 4 + tid];
        for (int r = 0; r < 4; ++r) comb[r * 4 + tid] /= col_sum + eps;
    }
    __syncthreads();
    for (int iter = 0; iter < 19; ++iter) {
        if (tid < 4) {
            float row_sum = 0.0f;
            for (int c = 0; c < 4; ++c) row_sum += comb[tid * 4 + c];
            for (int c = 0; c < 4; ++c) comb[tid * 4 + c] /= row_sum + eps;
        }
        __syncthreads();
        if (tid < 4) {
            float col_sum = 0.0f;
            for (int r = 0; r < 4; ++r) col_sum += comb[r * 4 + tid];
            for (int r = 0; r < 4; ++r) comb[r * 4 + tid] /= col_sum + eps;
        }
        __syncthreads();
    }
    if (tid < 4) post_out[tid] = post[tid];
    if (tid < 16) comb_out[tid] = comb[tid];
    for (int d = tid; d < dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int h = 0; h < 4; ++h) acc += pre[h] * h4[static_cast<size_t>(h) * dim + d];
        x[d] = acc;
    }
}

__global__ void hc_post_float_kernel(const float* x, const float* residual, const float* post, const float* comb, float* y, int dim) {
    const int h = blockIdx.x;
    const int tid = threadIdx.x;
    for (int d = tid; d < dim; d += blockDim.x) {
        float acc = post[h] * x[d];
        for (int i = 0; i < 4; ++i) acc += comb[i * 4 + h] * residual[static_cast<size_t>(i) * dim + d];
        y[static_cast<size_t>(h) * dim + d] = acc;
    }
}

__global__ void hc_repeat_rows_kernel(const float* x, float* h4, int rows, int dim) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= rows) return;
    const float* src = x + static_cast<size_t>(row) * dim;
    float* dst = h4 + static_cast<size_t>(row) * 4 * dim;
    for (int i = tid; i < 4 * dim; i += blockDim.x) dst[i] = src[i % dim];
}

__global__ void hc_pre_float_rows_kernel(
    const float* h4_rows,
    const float* fn,
    const float* scale,
    const float* base,
    float* x_rows,
    float* post_rows,
    float* comb_rows,
    int rows,
    int dim) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const float* h4 = h4_rows + static_cast<size_t>(row) * 4 * dim;
    float* x = x_rows + static_cast<size_t>(row) * dim;
    float* post_out = post_rows + static_cast<size_t>(row) * 4;
    float* comb_out = comb_rows + static_cast<size_t>(row) * 16;
    constexpr float eps = 1e-6f;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    constexpr int kNumWarps = 8;
    __shared__ float mixes[24];
    __shared__ float pre[4];
    __shared__ float post[4];
    __shared__ float comb[16];
    __shared__ float partial[kNumWarps];
    const int n = 4 * dim;
    float sum_sq = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        const float v = h4[i];
        sum_sq += v * v;
    }
    for (int off = 16; off > 0; off >>= 1) sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, off);
    if (lane == 0) partial[warp_id] = sum_sq;
    __syncthreads();
    float total_sq = 0.0f;
    if (warp_id == 0) {
        total_sq = lane < kNumWarps ? partial[lane] : 0.0f;
        for (int off = kNumWarps / 2; off > 0; off >>= 1) total_sq += __shfl_xor_sync(0xffffffff, total_sq, off);
        if (lane == 0) partial[0] = total_sq;
    }
    __syncthreads();
    const float rsqrt = rsqrtf(partial[0] / static_cast<float>(n) + eps);
    // 24 dot products fn[r,:] · h4[:], computed with one warp per row (8 warps × 3 rows).
    for (int r_base = 0; r_base < 24; r_base += kNumWarps) {
        const int r = r_base + warp_id;
        float dot = 0.0f;
        if (r < 24) {
            const float* fn_row = fn + static_cast<size_t>(r) * n;
            for (int i = lane; i < n; i += 32) dot += fn_row[i] * h4[i];
            for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xffffffff, dot, off);
            if (lane == 0) mixes[r] = dot * rsqrt;
        }
    }
    __syncthreads();
    if (tid < 4) {
        pre[tid] = sigmoidf_fast(mixes[tid] * scale[0] + base[tid]) + eps;
        post[tid] = 2.0f * sigmoidf_fast(mixes[4 + tid] * scale[1] + base[4 + tid]);
    }
    if (tid < 16) {
        float vals[4];
        const int rr = tid / 4;
        const int col = tid - rr * 4;
        const float* cm = mixes + 8 + rr * 4;
        float max_v = cm[0] * scale[2] + base[8 + rr * 4];
        for (int c = 1; c < 4; ++c) max_v = fmaxf(max_v, cm[c] * scale[2] + base[8 + rr * 4 + c]);
        float sum = 0.0f;
        for (int c = 0; c < 4; ++c) {
            vals[c] = expf(cm[c] * scale[2] + base[8 + rr * 4 + c] - max_v);
            sum += vals[c];
        }
        comb[tid] = vals[col] / sum + eps;
    }
    __syncthreads();
    if (tid < 4) {
        float col_sum = 0.0f;
        for (int r = 0; r < 4; ++r) col_sum += comb[r * 4 + tid];
        for (int r = 0; r < 4; ++r) comb[r * 4 + tid] /= col_sum + eps;
    }
    __syncthreads();
    for (int iter = 0; iter < 19; ++iter) {
        if (tid < 4) {
            float row_sum = 0.0f;
            for (int c = 0; c < 4; ++c) row_sum += comb[tid * 4 + c];
            for (int c = 0; c < 4; ++c) comb[tid * 4 + c] /= row_sum + eps;
        }
        __syncthreads();
        if (tid < 4) {
            float col_sum = 0.0f;
            for (int r = 0; r < 4; ++r) col_sum += comb[r * 4 + tid];
            for (int r = 0; r < 4; ++r) comb[r * 4 + tid] /= col_sum + eps;
        }
        __syncthreads();
    }
    if (tid < 4) post_out[tid] = post[tid];
    if (tid < 16) comb_out[tid] = comb[tid];
    for (int d = tid; d < dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int h = 0; h < 4; ++h) acc += pre[h] * h4[static_cast<size_t>(h) * dim + d];
        x[d] = acc;
    }
}

// Collapse [rows, 4, dim] to [rows, dim] with the head's own gate, which is a
// different parameterization from hc_pre's: only 4 mixes, one shared scale, and
// no post/comb outputs. Same warp-per-fn-row layout as hc_pre_float_rows_kernel
// -- with 4 rows and 8 warps half the block idles, but the dot itself is the
// cost and splitting it across a warp is what matters.
__global__ void hc_head_float_rows_kernel(
    const float* h4_rows,
    const float* fn,
    const float* scale,
    const float* base,
    float* y_rows,
    int rows,
    int dim) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const float* h4 = h4_rows + static_cast<size_t>(row) * 4 * dim;
    float* y = y_rows + static_cast<size_t>(row) * dim;
    constexpr float eps = 1e-6f;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    constexpr int kNumWarps = 8;
    __shared__ float pre[4];
    __shared__ float partial[kNumWarps];
    const int n = 4 * dim;

    float sum_sq = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        const float v = h4[i];
        sum_sq += v * v;
    }
    for (int off = 16; off > 0; off >>= 1) sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, off);
    if (lane == 0) partial[warp_id] = sum_sq;
    __syncthreads();
    if (warp_id == 0) {
        float total_sq = lane < kNumWarps ? partial[lane] : 0.0f;
        for (int off = kNumWarps / 2; off > 0; off >>= 1) total_sq += __shfl_xor_sync(0xffffffff, total_sq, off);
        if (lane == 0) partial[0] = total_sq;
    }
    __syncthreads();
    const float rsqrt = rsqrtf(partial[0] / static_cast<float>(n) + eps);

    if (warp_id < 4) {
        const float* fn_row = fn + static_cast<size_t>(warp_id) * n;
        float dot = 0.0f;
        for (int i = lane; i < n; i += 32) dot += fn_row[i] * h4[i];
        for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xffffffff, dot, off);
        if (lane == 0) pre[warp_id] = sigmoidf_fast(dot * rsqrt * scale[0] + base[warp_id]) + eps;
    }
    __syncthreads();

    for (int d = tid; d < dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int h = 0; h < 4; ++h) acc += pre[h] * h4[static_cast<size_t>(h) * dim + d];
        y[d] = acc;
    }
}

// Mean over the hc dimension: [rows, 4, dim] -> [rows, dim], written into a
// column slice of a wider destination so the DSpark target layers land
// concatenated per position (the reference cats them on the last axis).
//   d_out_rows + row * out_stride + col_offset .. + dim
__global__ void hc_mean_pool_rows_kernel(const float* h4_rows, float* out_rows,
                                         int rows, int dim, int out_stride,
                                         int col_offset) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const float* h4 = h4_rows + static_cast<size_t>(row) * 4 * dim;
    float* out = out_rows + static_cast<size_t>(row) * out_stride + col_offset;
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int h = 0; h < 4; ++h) acc += h4[static_cast<size_t>(h) * dim + d];
        out[d] = acc * 0.25f;
    }
}

__global__ void hc_post_float_rows_kernel(const float* x_rows, const float* residual_rows, const float* post_rows, const float* comb_rows, float* y_rows, int rows, int dim) {
    const int row = blockIdx.x;
    const int h = blockIdx.y;
    const int tid = threadIdx.x;
    if (row >= rows || h >= 4) return;
    const float* x = x_rows + static_cast<size_t>(row) * dim;
    const float* residual = residual_rows + static_cast<size_t>(row) * 4 * dim;
    const float* post = post_rows + static_cast<size_t>(row) * 4;
    const float* comb = comb_rows + static_cast<size_t>(row) * 16;
    float* y = y_rows + static_cast<size_t>(row) * 4 * dim + static_cast<size_t>(h) * dim;
    for (int d = tid; d < dim; d += blockDim.x) {
        float acc = post[h] * x[d];
        for (int i = 0; i < 4; ++i) acc += comb[i * 4 + h] * residual[static_cast<size_t>(i) * dim + d];
        y[d] = acc;
    }
}

__global__ void silu_mul_kernel(const float* gate, const float* up, float* y, int cols) {
    const int row = blockIdx.x;
    const float* row_gate = gate + static_cast<size_t>(row) * cols;
    const float* row_up = up + static_cast<size_t>(row) * cols;
    float* row_y = y + static_cast<size_t>(row) * cols;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        const float g = row_gate[c];
        const float s = g / (1.0f + expf(-g));
        row_y[c] = s * row_up[c];
    }
}

__global__ void silu_mul_clamped_kernel(const float* gate, const float* up, float* y, int cols, float limit) {
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        const float g = fminf(gate[c], limit);
        const float u = fminf(fmaxf(up[c], -limit), limit);
        const float s = g / (1.0f + expf(-g));
        y[c] = s * u;
    }
}

__global__ void vector_add_kernel(const float* a, const float* b, float* y, int cols) {
    for (int c = threadIdx.x; c < cols; c += blockDim.x) y[c] = a[c] + b[c];
}

__global__ void vector_accum_kernel(const float* x, float* y, int cols, float scale) {
    const int row = blockIdx.x;
    const float* row_x = x + static_cast<size_t>(row) * cols;
    float* row_y = y + static_cast<size_t>(row) * cols;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) row_y[c] += row_x[c] * scale;
}

__global__ void repeat_vector_kernel(const float* x, float* y, int cols, int repeats) {
    const int total = cols * repeats;
    for (int i = threadIdx.x; i < total; i += blockDim.x) y[i] = x[i % cols];
}

__global__ void head_rmsnorm_rope_kernel(
    float* x,
    int head_dim,
    int rope_dim,
    int position,
    float theta,
    bool inverse,
    float eps) {
    const int head = blockIdx.x;
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        const float v = x[head * head_dim + i];
        sum_sq += v * v;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum_sq;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    const float norm = rsqrtf(partial[0] / static_cast<float>(head_dim) + eps);
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) x[head * head_dim + i] *= norm;
    __syncthreads();
    const int rope_start = head_dim - rope_dim;
    for (int pair = threadIdx.x * 2; pair < rope_dim; pair += blockDim.x * 2) {
        const int offset = rope_start + pair;
        const float angle = static_cast<float>(position) / powf(theta, static_cast<float>(pair) / static_cast<float>(rope_dim));
        const float c = cosf(angle);
        const float s = inverse ? -sinf(angle) : sinf(angle);
        const size_t base = static_cast<size_t>(head) * head_dim + offset;
        const float a = x[base];
        const float b = x[base + 1];
        x[base] = a * c - b * s;
        x[base + 1] = a * s + b * c;
    }
}

__global__ void rope_kernel(
    float* x,
    int head_dim,
    int rope_dim,
    int position,
    float theta,
    bool inverse) {
    const int head = blockIdx.x;
    const int rope_start = head_dim - rope_dim;
    for (int pair = threadIdx.x * 2; pair < rope_dim; pair += blockDim.x * 2) {
        const int offset = rope_start + pair;
        const float angle = static_cast<float>(position) / powf(theta, static_cast<float>(pair) / static_cast<float>(rope_dim));
        const float c = cosf(angle);
        const float s = inverse ? -sinf(angle) : sinf(angle);
        const size_t base = static_cast<size_t>(head) * head_dim + offset;
        const float a = x[base];
        const float b = x[base + 1];
        x[base] = a * c - b * s;
        x[base + 1] = a * s + b * c;
    }
}

__global__ void head_rmsnorm_rope_rows_kernel(
    float* x,
    int tokens,
    int heads,
    int head_dim,
    int rope_dim,
    int start_position,
    float theta,
    bool inverse,
    float eps) {
    const int head = blockIdx.x;
    const int token = blockIdx.y;
    if (token >= tokens || head >= heads) return;
    float* row = x + (static_cast<size_t>(token) * heads + head) * head_dim;
    if (eps > 0.0f) {
        float sum_sq = 0.0f;
        for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
            const float v = row[i];
            sum_sq += v * v;
        }
        __shared__ float partial[256];
        partial[threadIdx.x] = sum_sq;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
            __syncthreads();
        }
        const float norm = rsqrtf(partial[0] / static_cast<float>(head_dim) + eps);
        for (int i = threadIdx.x; i < head_dim; i += blockDim.x) row[i] *= norm;
        __syncthreads();
    }
    const int rope_start = head_dim - rope_dim;
    const int position = start_position + token;
    for (int pair = threadIdx.x * 2; pair < rope_dim; pair += blockDim.x * 2) {
        const int offset = rope_start + pair;
        const float angle = static_cast<float>(position) / powf(theta, static_cast<float>(pair) / static_cast<float>(rope_dim));
        const float c = cosf(angle);
        const float s = inverse ? -sinf(angle) : sinf(angle);
        const float a = row[offset];
        const float b = row[offset + 1];
        row[offset] = a * c - b * s;
        row[offset + 1] = a * s + b * c;
    }
}

__global__ void head_rmsnorm_rope_freqs_kernel(
    float* x,
    const float* inv_freqs,
    int head_dim,
    int rope_dim,
    int position,
    bool inverse,
    float eps) {
    const int head = blockIdx.x;
    if (eps > 0.0f) {
        float sum_sq = 0.0f;
        for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
            const float v = x[head * head_dim + i];
            sum_sq += v * v;
        }
        __shared__ float partial[256];
        partial[threadIdx.x] = sum_sq;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
            __syncthreads();
        }
        const float norm = rsqrtf(partial[0] / static_cast<float>(head_dim) + eps);
        for (int i = threadIdx.x; i < head_dim; i += blockDim.x) x[head * head_dim + i] *= norm;
        __syncthreads();
    }
    const int rope_start = head_dim - rope_dim;
    for (int pair = threadIdx.x * 2; pair < rope_dim; pair += blockDim.x * 2) {
        const int offset = rope_start + pair;
        const float freq = inv_freqs[pair >> 1];
        const float angle = static_cast<float>(position) * freq;
        const float c = cosf(angle);
        const float s = inverse ? -sinf(angle) : sinf(angle);
        const size_t base = static_cast<size_t>(head) * head_dim + offset;
        const float a = x[base];
        const float b = x[base + 1];
        x[base] = a * c - b * s;
        x[base + 1] = a * s + b * c;
    }
}

__device__ float round_pow2_device(float x);
__device__ float fp8_e4m3_quant_dequant(float v, float scale);

__global__ void head_rmsnorm_rope_freqs_rows_kernel(
    float* x,
    const float* inv_freqs,
    int tokens,
    int heads,
    int head_dim,
    int rope_dim,
    int start_position,
    bool inverse,
    float eps) {
    const int head = blockIdx.x;
    const int token = blockIdx.y;
    if (token >= tokens || head >= heads) return;
    float* row = x + (static_cast<size_t>(token) * heads + head) * head_dim;
    if (eps > 0.0f) {
        float sum_sq = 0.0f;
        for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
            const float v = row[i];
            sum_sq += v * v;
        }
        __shared__ float partial[256];
        partial[threadIdx.x] = sum_sq;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
            __syncthreads();
        }
        const float norm = rsqrtf(partial[0] / static_cast<float>(head_dim) + eps);
        for (int i = threadIdx.x; i < head_dim; i += blockDim.x) row[i] *= norm;
        __syncthreads();
    }
    const int rope_start = head_dim - rope_dim;
    const int position = start_position + token;
    for (int pair = threadIdx.x * 2; pair < rope_dim; pair += blockDim.x * 2) {
        const int offset = rope_start + pair;
        const float angle = static_cast<float>(position) * inv_freqs[pair >> 1];
        const float c = cosf(angle);
        const float s = inverse ? -sinf(angle) : sinf(angle);
        const float a = row[offset];
        const float b = row[offset + 1];
        row[offset] = a * c - b * s;
        row[offset + 1] = a * s + b * c;
    }
}

__global__ void fp8_act_quant_dequant_rows_kernel(float* x, int cols, int block_size) {
    const int row = blockIdx.y;
    const int block = blockIdx.x;
    const int start = block * block_size;
    float* row_x = x + static_cast<size_t>(row) * cols;
    float max_abs = 0.0f;
    for (int i = threadIdx.x; i < block_size; i += blockDim.x) max_abs = fmaxf(max_abs, fabsf(row_x[start + i]));
    __shared__ float partial[256];
    partial[threadIdx.x] = max_abs;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    const float scale = round_pow2_device(fmaxf(partial[0], 1e-4f) / 448.0f);
    for (int i = threadIdx.x; i < block_size; i += blockDim.x) row_x[start + i] = fp8_e4m3_quant_dequant(row_x[start + i], scale);
}

__global__ void fp8_act_quant_dequant_rows_strided_kernel(float* x, int cols, int row_stride, int block_size) {
    const int row = blockIdx.y;
    const int block = blockIdx.x;
    const int start = block * block_size;
    float* row_x = x + static_cast<size_t>(row) * row_stride;
    float max_abs = 0.0f;
    for (int i = threadIdx.x; i < block_size; i += blockDim.x) max_abs = fmaxf(max_abs, fabsf(row_x[start + i]));
    __shared__ float partial[256];
    partial[threadIdx.x] = max_abs;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    const float scale = round_pow2_device(fmaxf(partial[0], 1e-4f) / 448.0f);
    for (int i = threadIdx.x; i < block_size; i += blockDim.x) row_x[start + i] = fp8_e4m3_quant_dequant(row_x[start + i], scale);
}

__global__ void copy_rows_to_kv_cache_kernel(const float* rows, float* cache, int rows_count, int cols, int window_size, int start_position) {
    const int row = blockIdx.x;
    if (row >= rows_count) return;
    const int slot = (row + start_position) % window_size;
    const float* src = rows + static_cast<size_t>(row) * cols;
    float* dst = cache + static_cast<size_t>(slot) * cols;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) dst[c] = src[c];
}

__global__ void prefill_causal_attention_kernel(
    const float* q,
    const float* kv,
    const float* attn_sink,
    float* y,
    int tokens,
    int heads,
    int head_dim,
    int window_size,
    float scale) {
    const int head = blockIdx.x;
    const int token = blockIdx.y;
    if (head >= heads || token >= tokens) return;
    const float* qh = q + (static_cast<size_t>(token) * heads + head) * head_dim;
    const int start = max(0, token - window_size + 1);
    __shared__ float denom;
    __shared__ float max_logit;
    __shared__ float partial[256];
    float local_max = attn_sink == nullptr ? -INFINITY : attn_sink[head];
    for (int t = start + threadIdx.x; t <= token; t += blockDim.x) {
        const float* kth = kv + static_cast<size_t>(t) * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) dot += qh[d] * kth[d];
        local_max = fmaxf(local_max, dot * scale);
    }
    partial[threadIdx.x] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) max_logit = partial[0];
    __syncthreads();

    float local_denom = 0.0f;
    for (int t = start + threadIdx.x; t <= token; t += blockDim.x) {
        const float* kth = kv + static_cast<size_t>(t) * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) dot += qh[d] * kth[d];
        local_denom += expf(dot * scale - max_logit);
    }
    partial[threadIdx.x] = local_denom;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) denom = partial[0] + (attn_sink == nullptr ? 0.0f : expf(attn_sink[head] - max_logit));
    __syncthreads();

    float* yh = y + (static_cast<size_t>(token) * heads + head) * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float out = 0.0f;
        for (int t = start; t <= token; ++t) {
            const float* kth = kv + static_cast<size_t>(t) * head_dim;
            float dot = 0.0f;
            for (int i = 0; i < head_dim; ++i) dot += qh[i] * kth[i];
            const float w = expf(dot * scale - max_logit) / denom;
            out += w * kth[d];
        }
        yh[d] = out;
    }
}

__global__ void prefill_causal_attention_chunk_slow_kernel(
    const float* q,
    const float* kv,
    const float* attn_sink,
    float* y,
    int tokens,
    int heads,
    int kv_len,
    int head_dim,
    int window_size,
    int start_position,
    float scale) {
    const int head = blockIdx.x;
    const int token = blockIdx.y;
    if (head >= heads || token >= tokens) return;
    const int abs_token = start_position + token;
    const int start = max(0, abs_token - window_size + 1);
    const int end = min(abs_token, kv_len - 1);
    const float* qh = q + (static_cast<size_t>(token) * heads + head) * head_dim;
    __shared__ float denom;
    __shared__ float max_logit;
    __shared__ float partial[256];
    float local_max = attn_sink == nullptr ? -INFINITY : attn_sink[head];
    for (int t = start + threadIdx.x; t <= end; t += blockDim.x) {
        const float* kth = kv + static_cast<size_t>(t % window_size) * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) dot += qh[d] * kth[d];
        local_max = fmaxf(local_max, dot * scale);
    }
    partial[threadIdx.x] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    if (threadIdx.x == 0) max_logit = partial[0];
    __syncthreads();

    float local_denom = 0.0f;
    for (int t = start + threadIdx.x; t <= end; t += blockDim.x) {
        const float* kth = kv + static_cast<size_t>(t % window_size) * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) dot += qh[d] * kth[d];
        local_denom += expf(dot * scale - max_logit);
    }
    partial[threadIdx.x] = local_denom;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) denom = partial[0] + (attn_sink == nullptr ? 0.0f : expf(attn_sink[head] - max_logit));
    __syncthreads();

    float* yh = y + (static_cast<size_t>(token) * heads + head) * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float out = 0.0f;
        for (int t = start; t <= end; ++t) {
            const float* kth = kv + static_cast<size_t>(t % window_size) * head_dim;
            float dot = 0.0f;
            for (int i = 0; i < head_dim; ++i) dot += qh[i] * kth[i];
            const float w = expf(dot * scale - max_logit) / denom;
            out += w * kth[d];
        }
        yh[d] = out;
    }
}

__global__ void prefill_causal_attention_chunk_kernel(
    const float* q,
    const float* kv,
    const float* attn_sink,
    float* y,
    int tokens,
    int heads,
    int kv_len,
    int head_dim,
    int window_size,
    int start_position,
    float scale) {
    const int head = blockIdx.x;
    const int token = blockIdx.y;
    if (head >= heads || token >= tokens) return;
    const int abs_token = start_position + token;
    const int start = max(0, abs_token - window_size + 1);
    const int end = min(abs_token, kv_len - 1);
    const int tid = threadIdx.x;
    extern __shared__ float smem[];
    float* q_shared = smem;
    float* weights = q_shared + head_dim;
    float* partial = weights + kv_len;
    const float* qh = q + (static_cast<size_t>(token) * heads + head) * head_dim;
    for (int d = tid; d < head_dim; d += blockDim.x) q_shared[d] = qh[d];
    __syncthreads();

    const float sink_logit = attn_sink == nullptr ? -INFINITY : attn_sink[head];
    float local_max = sink_logit;
    for (int t = start + tid; t <= end; t += blockDim.x) {
        const float* kth = kv + static_cast<size_t>(t % window_size) * head_dim;
        float dot = 0.0f;
        for (int d = 0; d < head_dim; ++d) dot += q_shared[d] * kth[d];
        const float logit = dot * scale;
        weights[t] = logit;
        local_max = fmaxf(local_max, logit);
    }
    partial[tid] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] = fmaxf(partial[tid], partial[tid + stride]);
        __syncthreads();
    }
    const float max_logit = partial[0];
    // Barrier before the denom pass reuses partial[]: without it a fast
    // tid=0 overwrites partial[0] while slower threads are still reading
    // max_logit from it, making the kernel run-to-run nondeterministic.
    __syncthreads();

    float local_denom = 0.0f;
    for (int t = start + tid; t <= end; t += blockDim.x) {
        const float w = expf(weights[t] - max_logit);
        weights[t] = w;
        local_denom += w;
    }
    partial[tid] = local_denom;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    const float denom = partial[0] + (attn_sink == nullptr ? 0.0f : expf(sink_logit - max_logit));
    const float inv_denom = 1.0f / denom;

    float* yh = y + (static_cast<size_t>(token) * heads + head) * head_dim;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float out = 0.0f;
        for (int t = start; t <= end; ++t) {
            const float* kth = kv + static_cast<size_t>(t % window_size) * head_dim;
            out += weights[t] * kth[d];
        }
        yh[d] = out * inv_denom;
    }
}


__global__ void build_prefill_window_indices_kernel(int32_t* indices, int tokens, int window_size, int topk) {
    const int token = blockIdx.x;
    if (token >= tokens) return;
    int32_t* row = indices + static_cast<size_t>(token) * topk;
    const int start = max(0, token - window_size + 1);
    const int count = token - start + 1;
    for (int i = threadIdx.x; i < topk; i += blockDim.x) {
        row[i] = i < count ? (start + i) : -1;
    }
}

__global__ void build_decode_kv_indices_kernel(
    int* indices,
    int window_start,
    int window_len,
    int window_size,
    int compressed_count,
    int compressed_offset) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < window_len + compressed_count; i += blockDim.x * gridDim.x) {
        if (i < window_len) {
            indices[i] = (window_start + i) % window_size;
        } else {
            indices[i] = compressed_offset + (i - window_len);
        }
    }
}

__global__ void prefill_sparse_attention_indexed_kernel(
    const float* __restrict__ q,
    const float* __restrict__ kv,
    const float* __restrict__ attn_sink,
    const int32_t* __restrict__ topk_idxs,
    float* __restrict__ out,
    int tokens,
    int heads,
    int kv_len,
    int topk,
    int head_dim,
    float scale) {
    const int head = blockIdx.x;
    const int token = blockIdx.y;
    if (head >= heads || token >= tokens) return;
    const int tid = threadIdx.x;
    extern __shared__ float smem[];
    float* q_shared = smem;
    float* weights = q_shared + head_dim;
    float* partial = weights + topk;
    const float* qh = q + (static_cast<size_t>(token) * heads + head) * head_dim;
    const int32_t* idx_row = topk_idxs + static_cast<size_t>(token) * topk;
    for (int d = tid; d < head_dim; d += blockDim.x) q_shared[d] = qh[d];
    __syncthreads();

    const float sink_logit = attn_sink == nullptr ? -INFINITY : attn_sink[head];
    float local_max = sink_logit;
    for (int t = tid; t < topk; t += blockDim.x) {
        const int idx = idx_row[t];
        float logit = -INFINITY;
        if (idx >= 0 && idx < kv_len) {
            const float* kth = kv + static_cast<size_t>(idx) * head_dim;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) dot += q_shared[d] * kth[d];
            logit = dot * scale;
            local_max = fmaxf(local_max, logit);
        }
        weights[t] = logit;
    }
    partial[tid] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] = fmaxf(partial[tid], partial[tid + stride]);
        __syncthreads();
    }
    const float max_logit = partial[0];
    // Every thread must have read partial[0] before the denom pass overwrites
    // partial[tid] below; without this barrier a fast tid=0 clobbers the max
    // while slower threads are still reading it, which makes the whole kernel
    // run-to-run nondeterministic.
    __syncthreads();

    float local_denom = 0.0f;
    for (int t = tid; t < topk; t += blockDim.x) {
        const float w = expf(weights[t] - max_logit);
        weights[t] = w;
        local_denom += w;
    }
    partial[tid] = local_denom;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    const float denom = partial[0] + (attn_sink == nullptr ? 0.0f : expf(sink_logit - max_logit));
    const float inv_denom = 1.0f / denom;

    float* yh = out + (static_cast<size_t>(token) * heads + head) * head_dim;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t < topk; ++t) {
            const int idx = idx_row[t];
            if (idx >= 0 && idx < kv_len) {
                const float* kth = kv + static_cast<size_t>(idx) * head_dim;
                acc += weights[t] * kth[d];
            }
        }
        yh[d] = acc * inv_denom;
    }
}

__global__ void prefill_sparse_attention_headpair_kernel(
    const float* __restrict__ q,
    const float* __restrict__ kv,
    const float* __restrict__ attn_sink,
    const int32_t* __restrict__ topk_idxs,
    float* __restrict__ out,
    int seqlen,
    int heads,
    int kv_len,
    int topk,
    int dim,
    float softmax_scale) {
    const int head_pairs = (heads + 1) / 2;
    const int bsp = blockIdx.x;
    const int pair = bsp % head_pairs;
    const int s = (bsp / head_pairs) % seqlen;
    const int h0 = pair * 2;
    const int h1 = h0 + 1;
    const bool has_h1 = h1 < heads;
    const int tid = threadIdx.x;

    extern __shared__ float smem[];
    float* scores0 = smem;
    float* scores1 = scores0 + topk;
    float* q0_shared = scores1 + topk;
    float* q1_shared = q0_shared + dim;
    int32_t* idx_shared = reinterpret_cast<int32_t*>(q1_shared + dim);
    float* reduce0 = reinterpret_cast<float*>(idx_shared + topk);
    float* reduce1 = reduce0 + blockDim.x;

    const float* q0_ptr = q + (static_cast<size_t>(s) * heads + h0) * dim;
    const float* q1_ptr = has_h1 ? q + (static_cast<size_t>(s) * heads + h1) * dim : q0_ptr;
    const float* kv_base = kv;
    const int32_t* idx_base = topk_idxs + static_cast<size_t>(s) * topk;
    for (int d = tid; d < dim; d += blockDim.x) {
        q0_shared[d] = q0_ptr[d];
        if (has_h1) q1_shared[d] = q1_ptr[d];
    }
    for (int t = tid; t < topk; t += blockDim.x) idx_shared[t] = idx_base[t];
    __syncthreads();

    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int num_warps = blockDim.x >> 5;
    for (int t = warp_id; t < topk; t += num_warps) {
        const int idx = idx_shared[t];
        float score0 = -INFINITY;
        float score1 = -INFINITY;
        if (idx >= 0 && idx < kv_len) {
            const float* kv_ptr = kv_base + static_cast<size_t>(idx) * dim;
            float acc0 = 0.0f;
            float acc1 = 0.0f;
            for (int d = lane; d < dim; d += 32) {
                const float v = kv_ptr[d];
                acc0 += q0_shared[d] * v;
                if (has_h1) acc1 += q1_shared[d] * v;
            }
            for (int off = 16; off > 0; off >>= 1) {
                acc0 += __shfl_xor_sync(0xffffffff, acc0, off);
                if (has_h1) acc1 += __shfl_xor_sync(0xffffffff, acc1, off);
            }
            score0 = acc0 * softmax_scale;
            if (has_h1) score1 = acc1 * softmax_scale;
        }
        if (lane == 0) {
            scores0[t] = score0;
            if (has_h1) scores1[t] = score1;
        }
    }
    __syncthreads();

    float local_max0 = -INFINITY;
    float local_max1 = -INFINITY;
    for (int t = tid; t < topk; t += blockDim.x) {
        local_max0 = fmaxf(local_max0, scores0[t]);
        if (has_h1) local_max1 = fmaxf(local_max1, scores1[t]);
    }
    reduce0[tid] = local_max0;
    if (has_h1) reduce1[tid] = local_max1;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            reduce0[tid] = fmaxf(reduce0[tid], reduce0[tid + stride]);
            if (has_h1) reduce1[tid] = fmaxf(reduce1[tid], reduce1[tid + stride]);
        }
        __syncthreads();
    }
    const float max_score0 = fmaxf(reduce0[0], attn_sink[h0]);
    const float max_score1 = has_h1 ? fmaxf(reduce1[0], attn_sink[h1]) : -INFINITY;

    float local_denom0 = 0.0f;
    float local_denom1 = 0.0f;
    for (int t = tid; t < topk; t += blockDim.x) {
        const float w0 = expf(scores0[t] - max_score0);
        scores0[t] = w0;
        local_denom0 += w0;
        if (has_h1) {
            const float w1 = expf(scores1[t] - max_score1);
            scores1[t] = w1;
            local_denom1 += w1;
        }
    }
    if (tid == 0) {
        local_denom0 += expf(attn_sink[h0] - max_score0);
        if (has_h1) local_denom1 += expf(attn_sink[h1] - max_score1);
    }
    reduce0[tid] = local_denom0;
    if (has_h1) reduce1[tid] = local_denom1;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            reduce0[tid] += reduce0[tid + stride];
            if (has_h1) reduce1[tid] += reduce1[tid + stride];
        }
        __syncthreads();
    }
    const float denom0 = reduce0[0];
    const float denom1 = has_h1 ? reduce1[0] : 1.0f;

    float* out0_ptr = out + (static_cast<size_t>(s) * heads + h0) * dim;
    float* out1_ptr = has_h1 ? out + (static_cast<size_t>(s) * heads + h1) * dim : out0_ptr;
    for (int d = tid; d < dim; d += blockDim.x) {
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        for (int t = 0; t < topk; ++t) {
            const int idx = idx_shared[t];
            if (idx >= 0 && idx < kv_len) {
                const float v = kv_base[static_cast<size_t>(idx) * dim + d];
                acc0 += scores0[t] * v;
                if (has_h1) acc1 += scores1[t] * v;
            }
        }
        out0_ptr[d] = acc0 / denom0;
        if (has_h1) out1_ptr[d] = acc1 / denom1;
    }
}

__global__ void prefill_sparse_attention_headpair_serial_kernel(
    const float* __restrict__ q,
    const float* __restrict__ kv,
    const float* __restrict__ attn_sink,
    const int32_t* __restrict__ topk_idxs,
    float* __restrict__ out,
    int seqlen,
    int heads,
    int kv_len,
    int topk,
    int dim,
    float softmax_scale) {
    const int head_pairs = (heads + 1) / 2;
    const int bsp = blockIdx.x;
    const int pair = bsp % head_pairs;
    const int s = (bsp / head_pairs) % seqlen;
    const int h0 = pair * 2;
    const int h1 = h0 + 1;
    const bool has_h1 = h1 < heads;
    const int tid = threadIdx.x;

    extern __shared__ float smem[];
    float* scores0 = smem;
    float* scores1 = scores0 + topk;
    float* q0_shared = scores1 + topk;
    float* q1_shared = q0_shared + dim;
    int32_t* idx_shared = reinterpret_cast<int32_t*>(q1_shared + dim);
    float* reduce0 = reinterpret_cast<float*>(idx_shared + topk);
    float* reduce1 = reduce0 + blockDim.x;

    const float* q0_ptr = q + (static_cast<size_t>(s) * heads + h0) * dim;
    const float* q1_ptr = has_h1 ? q + (static_cast<size_t>(s) * heads + h1) * dim : q0_ptr;
    const int32_t* idx_base = topk_idxs + static_cast<size_t>(s) * topk;
    for (int d = tid; d < dim; d += blockDim.x) {
        q0_shared[d] = q0_ptr[d];
        if (has_h1) q1_shared[d] = q1_ptr[d];
    }
    for (int t = tid; t < topk; t += blockDim.x) idx_shared[t] = idx_base[t];
    __syncthreads();

    const float sink0 = attn_sink == nullptr ? -INFINITY : attn_sink[h0];
    const float sink1 = (has_h1 && attn_sink != nullptr) ? attn_sink[h1] : -INFINITY;
    float local_max0 = sink0;
    float local_max1 = sink1;
    for (int t = tid; t < topk; t += blockDim.x) {
        const int idx = idx_shared[t];
        float score0 = -INFINITY;
        float score1 = -INFINITY;
        if (idx >= 0 && idx < kv_len) {
            const float* kv_ptr = kv + static_cast<size_t>(idx) * dim;
            float acc0 = 0.0f;
            float acc1 = 0.0f;
            for (int d = 0; d < dim; ++d) {
                const float v = kv_ptr[d];
                acc0 += q0_shared[d] * v;
                if (has_h1) acc1 += q1_shared[d] * v;
            }
            score0 = acc0 * softmax_scale;
            if (has_h1) score1 = acc1 * softmax_scale;
            local_max0 = fmaxf(local_max0, score0);
            if (has_h1) local_max1 = fmaxf(local_max1, score1);
        }
        scores0[t] = score0;
        if (has_h1) scores1[t] = score1;
    }
    reduce0[tid] = local_max0;
    if (has_h1) reduce1[tid] = local_max1;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            reduce0[tid] = fmaxf(reduce0[tid], reduce0[tid + stride]);
            if (has_h1) reduce1[tid] = fmaxf(reduce1[tid], reduce1[tid + stride]);
        }
        __syncthreads();
    }
    const float max0 = reduce0[0];
    const float max1 = has_h1 ? reduce1[0] : -INFINITY;

    float local_denom0 = 0.0f;
    float local_denom1 = 0.0f;
    for (int t = tid; t < topk; t += blockDim.x) {
        const float w0 = expf(scores0[t] - max0);
        scores0[t] = w0;
        local_denom0 += w0;
        if (has_h1) {
            const float w1 = expf(scores1[t] - max1);
            scores1[t] = w1;
            local_denom1 += w1;
        }
    }
    reduce0[tid] = local_denom0;
    if (has_h1) reduce1[tid] = local_denom1;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            reduce0[tid] += reduce0[tid + stride];
            if (has_h1) reduce1[tid] += reduce1[tid + stride];
        }
        __syncthreads();
    }
    const float denom0 = reduce0[0] + (attn_sink == nullptr ? 0.0f : expf(sink0 - max0));
    const float denom1 = has_h1 ? reduce1[0] + (attn_sink == nullptr ? 0.0f : expf(sink1 - max1)) : 1.0f;
    const float inv0 = 1.0f / denom0;
    const float inv1 = has_h1 ? 1.0f / denom1 : 0.0f;

    float* out0_ptr = out + (static_cast<size_t>(s) * heads + h0) * dim;
    float* out1_ptr = has_h1 ? out + (static_cast<size_t>(s) * heads + h1) * dim : out0_ptr;
    for (int d = tid; d < dim; d += blockDim.x) {
        float acc0 = 0.0f;
        float acc1 = 0.0f;
        for (int t = 0; t < topk; ++t) {
            const int idx = idx_shared[t];
            if (idx >= 0 && idx < kv_len) {
                const float v = kv[static_cast<size_t>(idx) * dim + d];
                acc0 += scores0[t] * v;
                if (has_h1) acc1 += scores1[t] * v;
            }
        }
        out0_ptr[d] = acc0 * inv0;
        if (has_h1) out1_ptr[d] = acc1 * inv1;
    }
}

__global__ void single_token_sparse_attention_kernel(
    const float* q,
    const float* kv,
    const float* attn_sink,
    float* y,
    int head_dim,
    float scale) {
    const int head = blockIdx.x;
    float dot = 0.0f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) dot += q[head * head_dim + i] * kv[i];
    __shared__ float partial[256];
    partial[threadIdx.x] = dot;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    const float token_logit = partial[0] * scale;
    const float sink_logit = attn_sink == nullptr ? -INFINITY : attn_sink[head];
    const float m = fmaxf(token_logit, sink_logit);
    const float token_weight = expf(token_logit - m) / (expf(token_logit - m) + expf(sink_logit - m));
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) y[head * head_dim + i] = kv[i] * token_weight;
}

__global__ void cached_single_token_attention_kernel(
    const float* q,
    const float* kv_cache,
    const float* attn_sink,
    float* y,
    int head_dim,
    int cache_len,
    float scale) {
    const int head = blockIdx.x;
    const int tid = threadIdx.x;
    extern __shared__ float smem[];
    float* q_shared = smem;
    float* weights = q_shared + head_dim;
    float* partial = weights + cache_len;
    const float* q_head = q + head * head_dim;
    for (int i = tid; i < head_dim; i += blockDim.x) q_shared[i] = q_head[i];
    __syncthreads();
    const float sink_logit = attn_sink == nullptr ? -INFINITY : attn_sink[head];
    float local_max = sink_logit;
    for (int t = tid; t < cache_len; t += blockDim.x) {
        const float* kv = kv_cache + static_cast<size_t>(t) * head_dim;
        float dot = 0.0f;
        for (int i = 0; i < head_dim; ++i) dot += q_shared[i] * kv[i];
        const float logit = dot * scale;
        weights[t] = logit;
        local_max = fmaxf(local_max, logit);
    }
    partial[tid] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] = fmaxf(partial[tid], partial[tid + stride]);
        __syncthreads();
    }
    const float max_logit = partial[0];
    // Barrier before the denom pass reuses partial[]: without it a fast
    // tid=0 overwrites partial[0] while slower threads are still reading
    // max_logit from it, making the kernel run-to-run nondeterministic.
    __syncthreads();
    float local_denom = 0.0f;
    for (int t = tid; t < cache_len; t += blockDim.x) {
        const float w = expf(weights[t] - max_logit);
        weights[t] = w;
        local_denom += w;
    }
    partial[tid] = local_denom;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    const float denom = partial[0] + (attn_sink == nullptr ? 0.0f : expf(sink_logit - max_logit));
    const float inv_denom = 1.0f / denom;
    for (int i = tid; i < head_dim; i += blockDim.x) {
        float out = 0.0f;
        for (int t = 0; t < cache_len; ++t) {
            const float* kv = kv_cache + static_cast<size_t>(t) * head_dim;
            out += weights[t] * kv[i];
        }
        y[head * head_dim + i] = out * inv_denom;
    }
}

__global__ void cached_single_token_attention_workspace_kernel(
    const float* q,
    const float* kv_cache,
    const float* attn_sink,
    float* weight_scratch,
    float* y,
    int head_dim,
    int cache_len,
    float scale) {
    const int head = blockIdx.x;
    const int tid = threadIdx.x;
    extern __shared__ float smem[];
    float* q_shared = smem;
    float* partial = q_shared + head_dim;
    float* weights = weight_scratch + static_cast<size_t>(head) * cache_len;
    const float* q_head = q + static_cast<size_t>(head) * head_dim;
    for (int i = tid; i < head_dim; i += blockDim.x) q_shared[i] = q_head[i];
    __syncthreads();
    const float sink_logit = attn_sink == nullptr ? -INFINITY : attn_sink[head];
    float local_max = sink_logit;
    for (int t = tid; t < cache_len; t += blockDim.x) {
        const float* kv = kv_cache + static_cast<size_t>(t) * head_dim;
        float dot = 0.0f;
        for (int i = 0; i < head_dim; ++i) dot += q_shared[i] * kv[i];
        const float logit = dot * scale;
        weights[t] = logit;
        local_max = fmaxf(local_max, logit);
    }
    partial[tid] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] = fmaxf(partial[tid], partial[tid + stride]);
        __syncthreads();
    }
    const float max_logit = partial[0];
    // Barrier before the denom pass reuses partial[]: without it a fast
    // tid=0 overwrites partial[0] while slower threads are still reading
    // max_logit from it, making the kernel run-to-run nondeterministic.
    __syncthreads();
    float local_denom = 0.0f;
    for (int t = tid; t < cache_len; t += blockDim.x) {
        const float w = expf(weights[t] - max_logit);
        weights[t] = w;
        local_denom += w;
    }
    partial[tid] = local_denom;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    const float denom = partial[0] + (attn_sink == nullptr ? 0.0f : expf(sink_logit - max_logit));
    const float inv_denom = 1.0f / denom;
    for (int i = tid; i < head_dim; i += blockDim.x) {
        float out = 0.0f;
        for (int t = 0; t < cache_len; ++t) {
            const float* kv = kv_cache + static_cast<size_t>(t) * head_dim;
            out += weights[t] * kv[i];
        }
        y[static_cast<size_t>(head) * head_dim + i] = out * inv_denom;
    }
}

__device__ float bf16_to_float_device(uint16_t bits) {
    uint32_t value = static_cast<uint32_t>(bits) << 16;
    return __uint_as_float(value);
}


__device__ float round_pow2_device(float x) {
    return exp2f(roundf(log2f(x)));
}

__device__ float fp8_e4m3_quant_dequant(float v, float scale) {
    float q = fminf(448.0f, fmaxf(-448.0f, nearbyintf(v / scale)));
    return q * scale;
}

__global__ void fp8_act_quant_dequant_kernel(float* x, int cols, int block_size) {
    const int block = blockIdx.x;
    const int start = block * block_size;
    float max_abs = 0.0f;
    for (int i = threadIdx.x; i < block_size; i += blockDim.x) max_abs = fmaxf(max_abs, fabsf(x[start + i]));
    __shared__ float partial[256];
    partial[threadIdx.x] = max_abs;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] = fmaxf(partial[threadIdx.x], partial[threadIdx.x + stride]);
        __syncthreads();
    }
    const float scale = round_pow2_device(fmaxf(partial[0], 1e-4f) / 448.0f);
    for (int i = threadIdx.x; i < block_size; i += blockDim.x) x[start + i] = fp8_e4m3_quant_dequant(x[start + i], scale);
}

__global__ void wo_a_int8_quantize_decode_kernel(
    const float* __restrict__ x,
    int8_t* __restrict__ x_q,
    float* __restrict__ x_scale,
    int groups,
    int group_dim) {
    const int group = blockIdx.x;
    const int tid = threadIdx.x;
    if (group >= groups) return;
    const int base = group * group_dim;
    __shared__ float partial[256];
    float local_max = 0.0f;
    for (int i = tid; i < group_dim; i += blockDim.x) local_max = fmaxf(local_max, fabsf(x[base + i]));
    partial[tid] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] = fmaxf(partial[tid], partial[tid + stride]);
        __syncthreads();
    }
    const float scale = fmaxf(partial[0], 1.0e-6f) / 127.0f;
    if (tid == 0) x_scale[group] = scale;
    const float inv_scale = 1.0f / scale;
    for (int i = tid; i < group_dim; i += blockDim.x) {
        int q = __float2int_rn(x[base + i] * inv_scale);
        q = max(-127, min(127, q));
        x_q[base + i] = static_cast<int8_t>(q);
    }
}

__global__ void wo_a_int8_decode_gemm_kernel(
    const int8_t* __restrict__ x_q,
    const float* __restrict__ x_scale,
    const int8_t* __restrict__ weight_q,
    const float* __restrict__ weight_scale,
    float* __restrict__ y,
    int group_rank,
    int group_dim) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int group = blockIdx.y;
    const int packs = group_dim / 4;
    extern __shared__ int x_shared[];
    const int8_t* x_row = x_q + static_cast<size_t>(group) * group_dim;
    const int* x_i32 = reinterpret_cast<const int*>(x_row);
    for (int i = threadIdx.x; i < packs; i += blockDim.x) x_shared[i] = x_i32[i];
    __syncthreads();
    if (col >= group_rank) return;
    const int row = group * group_rank + col;
    const int8_t* w_row = weight_q + static_cast<size_t>(row) * group_dim;
    const int* w_i32 = reinterpret_cast<const int*>(w_row);
    int acc = 0;
    for (int i = 0; i < packs; ++i) acc = pocket_dp4a_i8(x_shared[i], w_i32[i], acc);
    y[row] = static_cast<float>(acc) * x_scale[group] * weight_scale[row];
}

__global__ void compressor_update_state_kernel(
    const float* kv,
    const float* score,
    const float* ape,
    float* kv_state,
    float* score_state,
    int write_slot,
    int state_cols) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= state_cols) return;
    kv_state[static_cast<size_t>(write_slot) * state_cols + c] = kv[c];
    score_state[static_cast<size_t>(write_slot) * state_cols + c] = score[c] + ape[c];
}

__global__ void compressor_pool_kernel(
    const float* kv_state,
    const float* score_state,
    float* out,
    int ratio,
    int head_dim,
    int state_cols,
    bool overlap) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= head_dim) return;
    const int pool_slots = overlap ? ratio * 2 : ratio;
    float max_score = -INFINITY;
    for (int t = 0; t < pool_slots; ++t) {
        const int col = (overlap && t >= ratio) ? head_dim + d : d;
        max_score = fmaxf(max_score, score_state[static_cast<size_t>(t) * state_cols + col]);
    }
    float denom = 0.0f;
    for (int t = 0; t < pool_slots; ++t) {
        const int col = (overlap && t >= ratio) ? head_dim + d : d;
        denom += expf(score_state[static_cast<size_t>(t) * state_cols + col] - max_score);
    }
    float sum = 0.0f;
    for (int t = 0; t < pool_slots; ++t) {
        const int col = (overlap && t >= ratio) ? head_dim + d : d;
        const float w = expf(score_state[static_cast<size_t>(t) * state_cols + col] - max_score) / denom;
        sum += w * kv_state[static_cast<size_t>(t) * state_cols + col];
    }
    out[d] = sum;
}

__global__ void compressor_shift_overlap_state_kernel(float* kv_state, float* score_state, int ratio, int state_cols) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = ratio * state_cols;
    if (idx >= total) return;
    kv_state[idx] = kv_state[total + idx];
    score_state[idx] = score_state[total + idx];
    kv_state[total + idx] = 0.0f;
    score_state[total + idx] = -INFINITY;
}

__global__ void indexed_cached_single_token_attention_kernel(
    const float* q,
    const float* kv_cache,
    const int* indices,
    const float* attn_sink,
    float* y,
    int head_dim,
    int index_count,
    float scale) {
    const int head = blockIdx.x;
    const int tid = threadIdx.x;
    extern __shared__ float smem[];
    float* q_shared = smem;
    float* weights = q_shared + head_dim;
    float* partial = weights + index_count;
    const float* q_head = q + head * head_dim;
    for (int i = tid; i < head_dim; i += blockDim.x) q_shared[i] = q_head[i];
    __syncthreads();
    const float sink_logit = attn_sink == nullptr ? -INFINITY : attn_sink[head];
    float local_max = sink_logit;
    for (int t = tid; t < index_count; t += blockDim.x) {
        const int idx = indices[t];
        float logit = -INFINITY;
        if (idx >= 0) {
            const float* kv = kv_cache + static_cast<size_t>(idx) * head_dim;
            float dot = 0.0f;
            for (int i = 0; i < head_dim; ++i) dot += q_shared[i] * kv[i];
            logit = dot * scale;
        }
        weights[t] = logit;
        local_max = fmaxf(local_max, logit);
    }
    partial[tid] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] = fmaxf(partial[tid], partial[tid + stride]);
        __syncthreads();
    }
    const float max_logit = partial[0];
    // Barrier before the denom pass reuses partial[]: without it a fast
    // tid=0 overwrites partial[0] while slower threads are still reading
    // max_logit from it, making the kernel run-to-run nondeterministic.
    __syncthreads();
    float local_denom = 0.0f;
    for (int t = tid; t < index_count; t += blockDim.x) {
        const float w = expf(weights[t] - max_logit);
        weights[t] = w;
        local_denom += w;
    }
    partial[tid] = local_denom;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    const float denom = partial[0] + (attn_sink == nullptr ? 0.0f : expf(sink_logit - max_logit));
    const float inv_denom = 1.0f / denom;
    for (int i = tid; i < head_dim; i += blockDim.x) {
        float out = 0.0f;
        for (int t = 0; t < index_count; ++t) {
            const int idx = indices[t];
            if (idx >= 0) {
                const float* kv = kv_cache + static_cast<size_t>(idx) * head_dim;
                out += weights[t] * kv[i];
            }
        }
        y[head * head_dim + i] = out * inv_denom;
    }
}
__global__ void indexed_cached_attention_rows_kernel(
    const float* q,
    const float* kv_cache,
    const int32_t* row_starts,
    const int32_t* indices,
    const float* attn_sink,
    float* y,
    int rows,
    int heads,
    int head_dim,
    int max_index_count,
    float scale) {
    const int head = blockIdx.x;
    const int row = blockIdx.y;
    if (head >= heads || row >= rows) return;
    const int tid = threadIdx.x;
    const int begin = row_starts[row];
    const int end = row_starts[row + 1];
    const int index_count = end - begin;
    if (index_count <= 0 || index_count > max_index_count) return;

    extern __shared__ float smem[];
    float* q_shared = smem;
    float* weights = q_shared + head_dim;
    float* partial = weights + max_index_count;
    const float* q_head = q + (static_cast<size_t>(row) * heads + head) * head_dim;
    for (int i = tid; i < head_dim; i += blockDim.x) q_shared[i] = q_head[i];
    __syncthreads();

    const float sink_logit = attn_sink == nullptr ? -INFINITY : attn_sink[head];
    float local_max = sink_logit;
    for (int t = tid; t < index_count; t += blockDim.x) {
        const int idx = indices[begin + t];
        float logit = -INFINITY;
        if (idx >= 0) {
            const float* kv = kv_cache + static_cast<size_t>(idx) * head_dim;
            float dot = 0.0f;
            for (int i = 0; i < head_dim; ++i) dot += q_shared[i] * kv[i];
            logit = dot * scale;
        }
        weights[t] = logit;
        local_max = fmaxf(local_max, logit);
    }
    partial[tid] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] = fmaxf(partial[tid], partial[tid + stride]);
        __syncthreads();
    }
    const float max_logit = partial[0];
    __syncthreads();

    float local_denom = 0.0f;
    for (int t = tid; t < index_count; t += blockDim.x) {
        const float w = expf(weights[t] - max_logit);
        weights[t] = w;
        local_denom += w;
    }
    partial[tid] = local_denom;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    const float denom = partial[0] +
        (attn_sink == nullptr ? 0.0f : expf(sink_logit - max_logit));
    const float inv_denom = 1.0f / denom;
    float* y_head = y + (static_cast<size_t>(row) * heads + head) * head_dim;
    for (int i = tid; i < head_dim; i += blockDim.x) {
        float out = 0.0f;
        for (int t = 0; t < index_count; ++t) {
            const int idx = indices[begin + t];
            if (idx >= 0) {
                const float* kv = kv_cache + static_cast<size_t>(idx) * head_dim;
                out += weights[t] * kv[i];
            }
        }
        y_head[i] = out * inv_denom;
    }
}
__global__ void indexed_cached_attention_rows_batch_kv_kernel(
    const float* q,
    const float* kv_cache,
    const float* batch_kv,
    const int32_t* row_starts,
    const int32_t* indices,
    const float* attn_sink,
    float* y,
    int rows,
    int heads,
    int head_dim,
    int max_index_count,
    float scale) {
    const int head = blockIdx.x;
    const int row = blockIdx.y;
    if (head >= heads || row >= rows) return;
    const int tid = threadIdx.x;
    const int begin = row_starts[row];
    const int end = row_starts[row + 1];
    const int index_count = end - begin;
    if (index_count <= 0 || index_count > max_index_count) return;

    extern __shared__ float smem[];
    float* q_shared = smem;
    float* weights = q_shared + head_dim;
    float* partial = weights + max_index_count;
    const float* q_head = q + (static_cast<size_t>(row) * heads + head) * head_dim;
    for (int i = tid; i < head_dim; i += blockDim.x) q_shared[i] = q_head[i];
    __syncthreads();

    const float sink_logit = attn_sink == nullptr ? -INFINITY : attn_sink[head];
    float local_max = sink_logit;
    for (int t = tid; t < index_count; t += blockDim.x) {
        const int idx = indices[begin + t];
        const float* kv = idx >= 0
            ? kv_cache + static_cast<size_t>(idx) * head_dim
            : (idx <= -2 ? batch_kv + static_cast<size_t>(-idx - 2) * head_dim : nullptr);
        float logit = -INFINITY;
        if (kv != nullptr) {
            float dot = 0.0f;
            for (int i = 0; i < head_dim; ++i) dot += q_shared[i] * kv[i];
            logit = dot * scale;
        }
        weights[t] = logit;
        local_max = fmaxf(local_max, logit);
    }
    partial[tid] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] = fmaxf(partial[tid], partial[tid + stride]);
        __syncthreads();
    }
    const float max_logit = partial[0];
    __syncthreads();

    float local_denom = 0.0f;
    for (int t = tid; t < index_count; t += blockDim.x) {
        const float weight = expf(weights[t] - max_logit);
        weights[t] = weight;
        local_denom += weight;
    }
    partial[tid] = local_denom;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    const float denom = partial[0] +
        (attn_sink == nullptr ? 0.0f : expf(sink_logit - max_logit));
    const float inv_denom = 1.0f / denom;
    float* y_head = y + (static_cast<size_t>(row) * heads + head) * head_dim;
    for (int i = tid; i < head_dim; i += blockDim.x) {
        float out = 0.0f;
        for (int t = 0; t < index_count; ++t) {
            const int idx = indices[begin + t];
            const float* kv = idx >= 0
                ? kv_cache + static_cast<size_t>(idx) * head_dim
                : (idx <= -2 ? batch_kv + static_cast<size_t>(-idx - 2) * head_dim : nullptr);
            if (kv != nullptr) out += weights[t] * kv[i];
        }
        y_head[i] = out * inv_denom;
    }
}


__device__ __forceinline__ float fp4_fake_quant_value_device(float x) {
    const float ax = fabsf(x);
    float mag;
    if (ax <= 0.25f) {
        mag = 0.0f;
    } else if (ax <= 0.75f) {
        mag = 0.5f;
    } else if (ax <= 1.25f) {
        mag = 1.0f;
    } else if (ax <= 1.75f) {
        mag = 1.5f;
    } else if (ax <= 2.5f) {
        mag = 2.0f;
    } else if (ax <= 3.5f) {
        mag = 3.0f;
    } else if (ax <= 5.0f) {
        mag = 4.0f;
    } else {
        mag = 6.0f;
    }
    return copysignf(mag, x);
}

__device__ __forceinline__ float fp4_pow2_scale_device(float amax) {
    const float min_scale = 1.1754943508222875e-38f;
    const float raw = fmaxf(amax / 6.0f, min_scale);
    return exp2f(ceilf(log2f(raw) - 1.0e-6f));
}

__device__ void hadamard128_inplace_device(float* x, int tid) {
    for (int stride = 1; stride < 128; stride <<= 1) {
        if (tid < 128 && (tid & stride) == 0) {
            const float a = x[tid];
            const float b = x[tid + stride];
            x[tid] = a + b;
            x[tid + stride] = a - b;
        }
        __syncthreads();
    }
    if (tid < 128) x[tid] *= 0.08838834764831845f;
    __syncthreads();
}

__global__ void hadamard128_rows_kernel(const float* x, float* y, int rows) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ float vec[128];
    if (row >= rows) return;
    if (tid < 128) vec[tid] = x[static_cast<size_t>(row) * 128 + tid];
    __syncthreads();
    hadamard128_inplace_device(vec, tid);
    if (tid < 128) y[static_cast<size_t>(row) * 128 + tid] = vec[tid];
}

__global__ void fp4_fake_quant128_rows_kernel(float* x, int rows) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    __shared__ float vec[128];
    __shared__ float scales[4];
    if (row >= rows) return;
    if (tid < 128) vec[tid] = x[static_cast<size_t>(row) * 128 + tid];
    __syncthreads();
    if (tid < 4) {
        float amax = 0.0f;
        const int base = tid * 32;
        #pragma unroll
        for (int i = 0; i < 32; ++i) amax = fmaxf(amax, fabsf(vec[base + i]));
        scales[tid] = fp4_pow2_scale_device(amax);
    }
    __syncthreads();
    if (tid < 128) {
        const float scale = scales[tid >> 5];
        const float yv = fminf(fmaxf(vec[tid] / scale, -6.0f), 6.0f);
        vec[tid] = fp4_fake_quant_value_device(yv) * scale;
    }
    __syncthreads();
    if (tid < 128) x[static_cast<size_t>(row) * 128 + tid] = vec[tid];
}

__global__ void indexer_head_weights_kernel(
    const uint16_t* weight_proj,
    const float* x,
    float* head_weights,
    int heads,
    int head_dim,
    int dim) {
    const int head = blockIdx.x;
    const int tid = threadIdx.x;
    if (head >= heads) return;
    __shared__ float reduce[256];
    float local = 0.0f;
    for (int d = tid; d < dim; d += blockDim.x) local += bf16_to_float_device(weight_proj[static_cast<size_t>(head) * dim + d]) * x[d];
    reduce[tid] = local;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) reduce[tid] += reduce[tid + stride];
        __syncthreads();
    }
    if (tid == 0) {
        const float scale = rsqrtf(static_cast<float>(heads * head_dim));
        head_weights[head] = reduce[0] * scale;
    }
}

__global__ void indexer_score_kernel(
    const float* index_q,
    const float* index_kv,
    const float* head_weights,
    float* scores,
    int compressed_count,
    int heads,
    int head_dim) {
    const int c = blockIdx.x;
    const int tid = threadIdx.x;
    if (c >= compressed_count) return;
    __shared__ float reduce[256];
    float total = 0.0f;
    for (int h = 0; h < heads; ++h) {
        float partial = 0.0f;
        if (tid < head_dim) {
            const size_t idx = static_cast<size_t>(h) * head_dim + tid;
            partial = index_q[idx] * index_kv[static_cast<size_t>(c) * head_dim + tid];
        }
        reduce[tid] = partial;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) reduce[tid] += reduce[tid + stride];
            __syncthreads();
        }
        if (tid == 0) {
            const float dot = reduce[0];
            if (dot > 0.0f) total += dot * head_weights[h];
        }
        __syncthreads();
    }
    if (tid == 0) scores[c] = total;
}

__global__ void indexer_topk_kernel(
    float* scores,
    int* out_indices,
    int compressed_count,
    int keep,
    int offset) {
    const int tid = threadIdx.x;
    extern __shared__ char smem_raw[];
    float* best_vals = reinterpret_cast<float*>(smem_raw);
    int* best_idxs = reinterpret_cast<int*>(best_vals + blockDim.x);
    for (int k = 0; k < keep; ++k) {
        float local_best = -INFINITY;
        int local_idx = compressed_count;
        for (int i = tid; i < compressed_count; i += blockDim.x) {
            const float v = scores[i];
            if (v > local_best || (v == local_best && i < local_idx)) {
                local_best = v;
                local_idx = i;
            }
        }
        best_vals[tid] = local_best;
        best_idxs[tid] = local_idx;
        __syncthreads();
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                const float other_v = best_vals[tid + stride];
                const int other_i = best_idxs[tid + stride];
                if (other_v > best_vals[tid] || (other_v == best_vals[tid] && other_i < best_idxs[tid])) {
                    best_vals[tid] = other_v;
                    best_idxs[tid] = other_i;
                }
            }
            __syncthreads();
        }
        if (tid == 0) {
            const int idx = best_idxs[0] < 0 ? 0 : best_idxs[0];
            out_indices[k] = offset + idx;
            if (idx >= 0 && idx < compressed_count) scores[idx] = -INFINITY;
        }
        __syncthreads();
    }
}

__global__ void fp32_to_bf16_kernel(const float* x, uint16_t* y, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const uint32_t bits = __float_as_uint(x[i]);
    const uint32_t lsb = (bits >> 16) & 1u;
    const uint32_t rounded = bits + 0x7FFFu + lsb;
    y[i] = static_cast<uint16_t>(rounded >> 16);
}

__global__ void bf16_to_fp32_kernel(const uint16_t* x, float* y, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    y[i] = __uint_as_float(static_cast<uint32_t>(x[i]) << 16);
}

}  // namespace

bool cuda_runtime_available() {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

bool hc_pre_float_cuda(const float* d_h4, const float* d_fn, const float* d_scale, const float* d_base, float* d_x, float* d_post, float* d_comb, int dim, void* stream) {
    if (d_h4 == nullptr || d_fn == nullptr || d_scale == nullptr || d_base == nullptr || d_x == nullptr || d_post == nullptr || d_comb == nullptr || dim <= 0) return false;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    hc_pre_float_kernel<<<1, 256, 0, cuda_stream>>>(d_h4, d_fn, d_scale, d_base, d_x, d_post, d_comb, dim);
    return cudaGetLastError() == cudaSuccess;
}

bool hc_repeat_rows_cuda(const float* d_x_rows, float* d_h4_rows, int rows, int dim, void* stream) {
    if (d_x_rows == nullptr || d_h4_rows == nullptr || rows <= 0 || dim <= 0) return false;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    hc_repeat_rows_kernel<<<rows, 256, 0, cuda_stream>>>(d_x_rows, d_h4_rows, rows, dim);
    return cudaGetLastError() == cudaSuccess;
}

bool hc_pre_float_rows_cuda(
    const float* d_h4_rows,
    const float* d_fn,
    const float* d_scale,
    const float* d_base,
    float* d_x_rows,
    float* d_post_rows,
    float* d_comb_rows,
    int rows,
    int dim,
    void* stream) {
    if (d_h4_rows == nullptr || d_fn == nullptr || d_scale == nullptr || d_base == nullptr || d_x_rows == nullptr || d_post_rows == nullptr || d_comb_rows == nullptr || rows <= 0 || dim <= 0) return false;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    hc_pre_float_rows_kernel<<<rows, 256, 0, cuda_stream>>>(d_h4_rows, d_fn, d_scale, d_base, d_x_rows, d_post_rows, d_comb_rows, rows, dim);
    return cudaGetLastError() == cudaSuccess;
}

bool hc_head_float_rows_cuda(
    const float* d_h4_rows,
    const float* d_fn,
    const float* d_scale,
    const float* d_base,
    float* d_y_rows,
    int rows,
    int dim,
    void* stream) {
    if (d_h4_rows == nullptr || d_fn == nullptr || d_scale == nullptr || d_base == nullptr || d_y_rows == nullptr || rows <= 0 || dim <= 0) return false;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    hc_head_float_rows_kernel<<<rows, 256, 0, cuda_stream>>>(d_h4_rows, d_fn, d_scale, d_base, d_y_rows, rows, dim);
    return cudaGetLastError() == cudaSuccess;
}

bool hc_mean_pool_rows_cuda(
    const float* d_h4_rows,
    float* d_out_rows,
    int rows,
    int dim,
    int out_stride,
    int col_offset,
    void* stream) {
    if (d_h4_rows == nullptr || d_out_rows == nullptr || rows <= 0 || dim <= 0) return false;
    if (out_stride < col_offset + dim || col_offset < 0) return false;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    hc_mean_pool_rows_kernel<<<rows, 256, 0, cuda_stream>>>(d_h4_rows, d_out_rows, rows, dim,
                                                           out_stride, col_offset);
    return cudaGetLastError() == cudaSuccess;
}

bool hc_post_float_cuda(const float* d_x, const float* d_residual_h4, const float* d_post, const float* d_comb, float* d_y_h4, int dim, void* stream) {    if (d_x == nullptr || d_residual_h4 == nullptr || d_post == nullptr || d_comb == nullptr || d_y_h4 == nullptr || dim <= 0) return false;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    hc_post_float_kernel<<<4, 256, 0, cuda_stream>>>(d_x, d_residual_h4, d_post, d_comb, d_y_h4, dim);
    return cudaGetLastError() == cudaSuccess;
}

bool hc_post_float_rows_cuda(
    const float* d_x_rows,
    const float* d_residual_h4_rows,
    const float* d_post_rows,
    const float* d_comb_rows,
    float* d_y_h4_rows,
    int rows,
    int dim,
    void* stream) {
    if (d_x_rows == nullptr || d_residual_h4_rows == nullptr || d_post_rows == nullptr || d_comb_rows == nullptr || d_y_h4_rows == nullptr || rows <= 0 || dim <= 0) return false;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    hc_post_float_rows_kernel<<<dim3(rows, 4), 256, 0, cuda_stream>>>(d_x_rows, d_residual_h4_rows, d_post_rows, d_comb_rows, d_y_h4_rows, rows, dim);
    return cudaGetLastError() == cudaSuccess;
}

bool gguf_route_slots_from_indices_cuda(
    const int64_t* d_indices,
    int64_t* d_route_slots,
    int topk,
    int expert_start,
    int experts_per_rank,
    void* stream) {
    if (d_indices == nullptr || d_route_slots == nullptr || topk <= 0 || experts_per_rank <= 0) return false;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    gguf_route_slots_from_indices_kernel<<<1, 32, 0, cuda_stream>>>(
        d_indices, d_route_slots, topk, expert_start, experts_per_rank);
    return cudaGetLastError() == cudaSuccess;
}

bool argmax_fp32_cuda(
    const float* d_logits,
    int* d_token,
    float* d_logit,
    int count,
    int token_offset,
    void* stream) {
    if (d_logits == nullptr || d_token == nullptr || d_logit == nullptr || count <= 0) return false;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    constexpr int threads = 256;
    const size_t shared_bytes = threads * (sizeof(float) + sizeof(int));
    argmax_fp32_kernel<<<1, threads, shared_bytes, cuda_stream>>>(
        d_logits, d_token, d_logit, count, token_offset);
    return cudaGetLastError() == cudaSuccess;
}

bool argmax_fp32_rows_cuda(
    const float* d_logits,
    int* d_tokens,
    float* d_logits_out,
    int rows,
    int count,
    int token_offset,
    void* stream) {
    if (d_logits == nullptr || d_tokens == nullptr || d_logits_out == nullptr ||
        rows <= 0 || count <= 0) return false;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    constexpr int threads = 256;
    const size_t shared_bytes = threads * (sizeof(float) + sizeof(int));
    argmax_fp32_rows_kernel<<<rows, threads, shared_bytes, cuda_stream>>>(
        d_logits, d_tokens, d_logits_out, rows, count, token_offset);
    return cudaGetLastError() == cudaSuccess;
}

bool silu_mul_cuda(const float* d_gate, const float* d_up, float* d_y, int cols, void* stream) {
    if (d_gate == nullptr || d_up == nullptr || d_y == nullptr || cols <= 0) return false;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    silu_mul_kernel<<<1, 256, 0, cuda_stream>>>(d_gate, d_up, d_y, cols);
    return cudaGetLastError() == cudaSuccess;
}

bool silu_mul_rows_cuda(const float* d_gate, const float* d_up, float* d_y, int rows, int cols, void* stream) {
    if (d_gate == nullptr || d_up == nullptr || d_y == nullptr || rows <= 0 || cols <= 0) return false;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    silu_mul_kernel<<<rows, 256, 0, cuda_stream>>>(d_gate, d_up, d_y, cols);
    return cudaGetLastError() == cudaSuccess;
}

bool silu_mul_clamped_cuda(const float* d_gate, const float* d_up, float* d_y, int cols, float limit, void* stream) {
    if (d_gate == nullptr || d_up == nullptr || d_y == nullptr || cols <= 0 || limit <= 0.0f) return false;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    silu_mul_clamped_kernel<<<1, 256, 0, cuda_stream>>>(d_gate, d_up, d_y, cols, limit);
    return cudaGetLastError() == cudaSuccess;
}

bool vector_add_cuda(const float* d_a, const float* d_b, float* d_y, int cols, void* stream) {
    if (d_a == nullptr || d_b == nullptr || d_y == nullptr || cols <= 0) return false;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    vector_add_kernel<<<1, 256, 0, cuda_stream>>>(d_a, d_b, d_y, cols);
    return cudaGetLastError() == cudaSuccess;
}

bool vector_accum_cuda(const float* d_x, float* d_y, int cols, float scale, void* stream) {
    if (d_x == nullptr || d_y == nullptr || cols <= 0) return false;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    vector_accum_kernel<<<1, 256, 0, cuda_stream>>>(d_x, d_y, cols, scale);
    return cudaGetLastError() == cudaSuccess;
}

bool vector_accum_rows_cuda(const float* d_x, float* d_y, int rows, int cols, float scale, void* stream) {
    if (d_x == nullptr || d_y == nullptr || rows <= 0 || cols <= 0) return false;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    vector_accum_kernel<<<rows, 256, 0, cuda_stream>>>(d_x, d_y, cols, scale);
    return cudaGetLastError() == cudaSuccess;
}

bool repeat_vector_cuda(const float* d_x, float* d_y, int cols, int repeats, void* stream) {
    if (d_x == nullptr || d_y == nullptr || cols <= 0 || repeats <= 0) return false;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    repeat_vector_kernel<<<1, 256, 0, cuda_stream>>>(d_x, d_y, cols, repeats);
    return cudaGetLastError() == cudaSuccess;
}

bool prefill_causal_attention_cuda(
    const float* d_q,
    const float* d_kv,
    const float* d_attn_sink,
    float* d_y,
    int tokens,
    int heads,
    int head_dim,
    int window_size,
    float scale,
    void* stream) {
    if (d_q == nullptr || d_kv == nullptr || d_y == nullptr || tokens <= 0 || heads <= 0 || head_dim <= 0 || window_size <= 0) return false;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    prefill_causal_attention_kernel<<<dim3(heads, tokens), 256, 0, cuda_stream>>>(d_q, d_kv, d_attn_sink, d_y, tokens, heads, head_dim, window_size, scale);
    return cudaGetLastError() == cudaSuccess;
}

bool prefill_causal_attention_chunk_cuda(
    const float* d_q,
    const float* d_kv,
    const float* d_attn_sink,
    float* d_y,
    int tokens,
    int heads,
    int kv_len,
    int head_dim,
    int window_size,
    int start_position,
    float scale,
    void* stream) {
    if (d_q == nullptr || d_kv == nullptr || d_y == nullptr || tokens <= 0 || heads <= 0 || kv_len <= 0 || head_dim <= 0 || window_size <= 0 || start_position < 0) return false;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    constexpr int threads = 256;
    // The fast parity path mirrors cached_single_token_attention_workspace_cuda by
    // storing logits/weights once in shared memory instead of recomputing Q·K in
    // the output pass. Keep a conservative 48 KiB cap for 2080Ti; longer exact
    // contexts fall back to the original recompute kernel rather than failing the
    // launch. (Long-context performance should move to a tiled/global-scratch
    // kernel later.)
    const size_t shared_bytes = (static_cast<size_t>(head_dim) + static_cast<size_t>(kv_len) + threads) * sizeof(float);
    if (shared_bytes <= 48 * 1024) {
        prefill_causal_attention_chunk_kernel<<<dim3(heads, tokens), threads, shared_bytes, cuda_stream>>>(
            d_q, d_kv, d_attn_sink, d_y, tokens, heads, kv_len, head_dim, window_size, start_position, scale);
    } else {
        prefill_causal_attention_chunk_slow_kernel<<<dim3(heads, tokens), threads, 0, cuda_stream>>>(
            d_q, d_kv, d_attn_sink, d_y, tokens, heads, kv_len, head_dim, window_size, start_position, scale);
    }
    return cudaGetLastError() == cudaSuccess;
}

bool build_prefill_window_indices_cuda(int32_t* d_indices, int tokens, int window_size, int topk, void* stream) {
    if (d_indices == nullptr || tokens <= 0 || window_size <= 0 || topk <= 0) return false;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    build_prefill_window_indices_kernel<<<tokens, 256, 0, cuda_stream>>>(d_indices, tokens, window_size, topk);
    return cudaGetLastError() == cudaSuccess;
}

bool build_decode_kv_indices_cuda(
    int* d_indices,
    int window_start,
    int window_len,
    int window_size,
    int compressed_count,
    int compressed_offset,
    void* stream) {
    if (d_indices == nullptr || window_start < 0 || window_len < 0 || window_size <= 0 || compressed_count < 0 || compressed_offset < 0) return false;
    const int total = window_len + compressed_count;
    if (total <= 0) return false;
    const int threads = 256;
    const int blocks = std::min((total + threads - 1) / threads, 65535);
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    build_decode_kv_indices_kernel<<<blocks, threads, 0, cuda_stream>>>(d_indices, window_start, window_len, window_size, compressed_count, compressed_offset);
    return cudaGetLastError() == cudaSuccess;
}

bool prefill_sparse_attention_indexed_cuda(
    const float* d_q,
    const float* d_kv,
    const float* d_attn_sink,
    const int32_t* d_topk_indices,
    float* d_y,
    int tokens,
    int heads,
    int kv_len,
    int topk,
    int head_dim,
    float scale,
    void* stream) {
    if (d_q == nullptr || d_kv == nullptr || d_attn_sink == nullptr || d_topk_indices == nullptr || d_y == nullptr) return false;
    if (tokens <= 0 || heads <= 0 || kv_len <= 0 || topk <= 0 || head_dim <= 0) return false;
    const int threads = 256;
    const size_t shared_bytes = (static_cast<size_t>(head_dim) + static_cast<size_t>(topk) + static_cast<size_t>(threads)) * sizeof(float);
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    prefill_sparse_attention_indexed_kernel<<<dim3(heads, tokens), threads, shared_bytes, cuda_stream>>>(d_q, d_kv, d_attn_sink, d_topk_indices, d_y, tokens, heads, kv_len, topk, head_dim, scale);
    return cudaGetLastError() == cudaSuccess;
}

bool prefill_sparse_attention_headpair_cuda(
    const float* d_q,
    const float* d_kv,
    const float* d_attn_sink,
    const int32_t* d_topk_indices,
    float* d_y,
    int tokens,
    int heads,
    int kv_len,
    int topk,
    int head_dim,
    float scale,
    void* stream) {
    if (d_q == nullptr || d_kv == nullptr || d_attn_sink == nullptr || d_topk_indices == nullptr || d_y == nullptr) return false;
    if (tokens <= 0 || heads <= 0 || kv_len <= 0 || topk <= 0 || head_dim <= 0) return false;
    const int head_pairs = (heads + 1) / 2;
    const int threads = 256;
    const size_t shared_bytes = (static_cast<size_t>(2) * topk + static_cast<size_t>(2) * head_dim + static_cast<size_t>(topk) * sizeof(int32_t) / sizeof(float) + static_cast<size_t>(2) * threads + 8) * sizeof(float);
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    prefill_sparse_attention_headpair_kernel<<<tokens * head_pairs, threads, shared_bytes, cuda_stream>>>(d_q, d_kv, d_attn_sink, d_topk_indices, d_y, tokens, heads, kv_len, topk, head_dim, scale);
    return cudaGetLastError() == cudaSuccess;
}

bool prefill_sparse_attention_headpair_serial_cuda(
    const float* d_q,
    const float* d_kv,
    const float* d_attn_sink,
    const int32_t* d_topk_indices,
    float* d_y,
    int tokens,
    int heads,
    int kv_len,
    int topk,
    int head_dim,
    float scale,
    void* stream) {
    if (d_q == nullptr || d_kv == nullptr || d_attn_sink == nullptr || d_topk_indices == nullptr || d_y == nullptr) return false;
    if (tokens <= 0 || heads <= 0 || kv_len <= 0 || topk <= 0 || head_dim <= 0) return false;
    const int head_pairs = (heads + 1) / 2;
    const int threads = 256;
    const size_t shared_bytes = (static_cast<size_t>(2) * topk + static_cast<size_t>(2) * head_dim + static_cast<size_t>(topk) * sizeof(int32_t) / sizeof(float) + static_cast<size_t>(2) * threads + 8) * sizeof(float);
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    prefill_sparse_attention_headpair_serial_kernel<<<tokens * head_pairs, threads, shared_bytes, cuda_stream>>>(d_q, d_kv, d_attn_sink, d_topk_indices, d_y, tokens, heads, kv_len, topk, head_dim, scale);
    return cudaGetLastError() == cudaSuccess;
}

bool single_token_sparse_attention_cuda(
    const float* d_q,
    const float* d_kv,
    const float* d_attn_sink,
    float* d_y,
    int heads,
    int head_dim,
    float scale,
    void* stream) {
    if (d_q == nullptr || d_kv == nullptr || d_y == nullptr || heads <= 0 || head_dim <= 0) return false;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    single_token_sparse_attention_kernel<<<heads, 256, 0, cuda_stream>>>(d_q, d_kv, d_attn_sink, d_y, head_dim, scale);
    return cudaGetLastError() == cudaSuccess;
}

bool cached_single_token_attention_cuda(
    const float* d_q,
    const float* d_kv_cache,
    const float* d_attn_sink,
    float* d_y,
    int heads,
    int head_dim,
    int cache_len,
    float scale,
    void* stream) {
    if (d_q == nullptr || d_kv_cache == nullptr || d_y == nullptr || heads <= 0 || head_dim <= 0 || cache_len <= 0) return false;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    const int threads = 256;
    const size_t shared_bytes = (static_cast<size_t>(head_dim) + cache_len + threads) * sizeof(float);
    cached_single_token_attention_kernel<<<heads, threads, shared_bytes, cuda_stream>>>(d_q, d_kv_cache, d_attn_sink, d_y, head_dim, cache_len, scale);
    return cudaGetLastError() == cudaSuccess;
}

bool cached_single_token_attention_workspace_cuda(
    const float* d_q,
    const float* d_kv_cache,
    const float* d_attn_sink,
    float* d_weight_scratch,
    float* d_y,
    int heads,
    int head_dim,
    int cache_len,
    float scale,
    void* stream) {
    if (d_q == nullptr || d_kv_cache == nullptr || d_weight_scratch == nullptr || d_y == nullptr || heads <= 0 || head_dim <= 0 || cache_len <= 0) return false;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    const int threads = 256;
    const size_t shared_bytes = (static_cast<size_t>(head_dim) + threads) * sizeof(float);
    cached_single_token_attention_workspace_kernel<<<heads, threads, shared_bytes, cuda_stream>>>(
        d_q, d_kv_cache, d_attn_sink, d_weight_scratch, d_y, head_dim, cache_len, scale);
    return cudaGetLastError() == cudaSuccess;
}

bool indexed_cached_single_token_attention_cuda(
    const float* d_q,
    const float* d_kv_cache,
    const int* d_indices,
    const float* d_attn_sink,
    float* d_y,
    int heads,
    int head_dim,
    int index_count,
    float scale,
    void* stream) {
    if (d_q == nullptr || d_kv_cache == nullptr || d_indices == nullptr || d_y == nullptr || heads <= 0 || head_dim <= 0 || index_count <= 0) return false;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    const int threads = 256;
    const size_t shared_bytes = (static_cast<size_t>(head_dim) + index_count + threads) * sizeof(float);
    indexed_cached_single_token_attention_kernel<<<heads, threads, shared_bytes, cuda_stream>>>(d_q, d_kv_cache, d_indices, d_attn_sink, d_y, head_dim, index_count, scale);
    return cudaGetLastError() == cudaSuccess;
}

bool indexed_cached_attention_rows_cuda(
    const float* d_q,
    const float* d_kv_cache,
    const int32_t* d_row_starts,
    const int32_t* d_indices,
    const float* d_attn_sink,
    float* d_y,
    int rows,
    int heads,
    int head_dim,
    int max_index_count,
    float scale,
    void* stream) {
    if (d_q == nullptr || d_kv_cache == nullptr || d_row_starts == nullptr ||
        d_indices == nullptr || d_y == nullptr || rows <= 0 || heads <= 0 ||
        head_dim <= 0 || max_index_count <= 0) return false;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    constexpr int threads = 256;
    const size_t shared_bytes =
        (static_cast<size_t>(head_dim) + max_index_count + threads) * sizeof(float);
    // Turing exposes at least 48 KiB dynamic shared memory per block. Return an
    // explicit unsupported result instead of letting an oversized launch fail
    // asynchronously; callers can fall back to per-row attention.
    if (shared_bytes > 48 * 1024) return false;
    indexed_cached_attention_rows_kernel<<<dim3(heads, rows), threads, shared_bytes, cuda_stream>>>(
        d_q, d_kv_cache, d_row_starts, d_indices, d_attn_sink, d_y, rows,
        heads, head_dim, max_index_count, scale);
    return cudaGetLastError() == cudaSuccess;
}

bool indexed_cached_attention_rows_batch_kv_cuda(
    const float* d_q,
    const float* d_kv_cache,
    const float* d_batch_kv,
    const int32_t* d_row_starts,
    const int32_t* d_indices,
    const float* d_attn_sink,
    float* d_y,
    int rows,
    int heads,
    int head_dim,
    int max_index_count,
    float scale,
    void* stream) {
    if (d_q == nullptr || d_kv_cache == nullptr || d_batch_kv == nullptr ||
        d_row_starts == nullptr || d_indices == nullptr || d_y == nullptr ||
        rows <= 0 || heads <= 0 || head_dim <= 0 || max_index_count <= 0) {
        return false;
    }
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    constexpr int threads = 256;
    const size_t shared_bytes =
        (static_cast<size_t>(head_dim) + max_index_count + threads) * sizeof(float);
    if (shared_bytes > 48 * 1024) return false;
    indexed_cached_attention_rows_batch_kv_kernel<<<
        dim3(heads, rows), threads, shared_bytes, cuda_stream>>>(
        d_q, d_kv_cache, d_batch_kv, d_row_starts, d_indices, d_attn_sink,
        d_y, rows, heads, head_dim, max_index_count, scale);
    return cudaGetLastError() == cudaSuccess;
}

bool indexer_select_topk_cuda(
    const float* d_index_q,
    const float* d_index_kv,
    const uint16_t* d_weight_proj_bf16,
    const float* d_x,
    float* d_scores_scratch,
    int* d_out_indices,
    int compressed_count,
    int keep,
    int heads,
    int head_dim,
    int dim,
    int offset,
    void* stream) {
    if (d_index_q == nullptr || d_index_kv == nullptr || d_weight_proj_bf16 == nullptr || d_x == nullptr || d_scores_scratch == nullptr || d_out_indices == nullptr) return false;
    if (compressed_count <= 0 || keep <= 0 || keep > compressed_count || heads <= 0 || head_dim <= 0 || dim <= 0) return false;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    float* d_head_weights = d_scores_scratch;
    float* d_scores = d_scores_scratch + heads;
    indexer_head_weights_kernel<<<heads, 256, 0, cuda_stream>>>(d_weight_proj_bf16, d_x, d_head_weights, heads, head_dim, dim);
    indexer_score_kernel<<<compressed_count, 256, 0, cuda_stream>>>(d_index_q, d_index_kv, d_head_weights, d_scores, compressed_count, heads, head_dim);
    indexer_topk_kernel<<<1, 256, 256 * (sizeof(float) + sizeof(int)), cuda_stream>>>(d_scores, d_out_indices, compressed_count, keep, offset);
    return cudaGetLastError() == cudaSuccess;
}

bool hadamard128_rows_cuda(const float* d_x, float* d_y, int rows, void* stream) {
    if (d_x == nullptr || d_y == nullptr || rows <= 0) return false;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    hadamard128_rows_kernel<<<rows, 128, 0, cuda_stream>>>(d_x, d_y, rows);
    return cudaGetLastError() == cudaSuccess;
}

bool fp4_fake_quant128_rows_cuda(float* d_x, int rows, void* stream) {
    if (d_x == nullptr || rows <= 0) return false;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    fp4_fake_quant128_rows_kernel<<<rows, 128, 0, cuda_stream>>>(d_x, rows);
    return cudaGetLastError() == cudaSuccess;
}

bool wo_a_int8_decode_cuda(
    const float* d_x,
    const int8_t* d_weight_q,
    const float* d_weight_scale,
    float* d_y,
    int groups,
    int group_rank,
    int group_dim,
    int8_t* d_x_q,
    float* d_x_scale,
    void* stream) {
    if (d_x == nullptr || d_weight_q == nullptr || d_weight_scale == nullptr || d_y == nullptr || d_x_q == nullptr || d_x_scale == nullptr) return false;
    if (groups <= 0 || group_rank <= 0 || group_dim <= 0 || (group_dim % 4) != 0) return false;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    wo_a_int8_quantize_decode_kernel<<<groups, 256, 0, cuda_stream>>>(d_x, d_x_q, d_x_scale, groups, group_dim);
    if (cudaGetLastError() != cudaSuccess) return false;
    const int threads = 128;
    const dim3 grid((group_rank + threads - 1) / threads, groups);
    const size_t shared_bytes = static_cast<size_t>(group_dim / 4) * sizeof(int);
    wo_a_int8_decode_gemm_kernel<<<grid, threads, shared_bytes, cuda_stream>>>(d_x_q, d_x_scale, d_weight_q, d_weight_scale, d_y, group_rank, group_dim);
    return cudaGetLastError() == cudaSuccess;
}

bool fp8_act_quant_dequant_cuda(float* d_x, int cols, int block_size, void* stream) {
    if (d_x == nullptr || cols <= 0 || block_size <= 0 || (cols % block_size) != 0) return false;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    fp8_act_quant_dequant_kernel<<<cols / block_size, 256, 0, cuda_stream>>>(d_x, cols, block_size);
    return cudaGetLastError() == cudaSuccess;
}

bool fp8_act_quant_dequant_rows_cuda(float* d_x, int rows, int cols, int block_size, void* stream) {
    if (d_x == nullptr || rows <= 0 || cols <= 0 || block_size <= 0 || (cols % block_size) != 0) return false;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    fp8_act_quant_dequant_rows_kernel<<<dim3(cols / block_size, rows), 256, 0, cuda_stream>>>(d_x, cols, block_size);
    return cudaGetLastError() == cudaSuccess;
}

bool fp8_act_quant_dequant_rows_strided_cuda(float* d_x, int rows, int cols, int row_stride, int block_size, void* stream) {
    if (d_x == nullptr || rows <= 0 || cols <= 0 || row_stride < cols || block_size <= 0 || (cols % block_size) != 0) return false;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    fp8_act_quant_dequant_rows_strided_kernel<<<dim3(cols / block_size, rows), 256, 0, cuda_stream>>>(d_x, cols, row_stride, block_size);
    return cudaGetLastError() == cudaSuccess;
}

bool copy_rows_to_kv_cache_cuda(const float* d_rows, float* d_cache, int rows, int cols, int window_size, int start_position, void* stream) {
    if (d_rows == nullptr || d_cache == nullptr || rows <= 0 || cols <= 0 || window_size <= 0) return false;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    copy_rows_to_kv_cache_kernel<<<rows, 256, 0, cuda_stream>>>(d_rows, d_cache, rows, cols, window_size, start_position);
    return cudaGetLastError() == cudaSuccess;
}

bool compressor_update_state_cuda(const float* d_kv, const float* d_score, const float* d_ape, float* d_kv_state, float* d_score_state, int offset, int write_slot, int state_cols, void* stream) {
    (void)offset;
    if (d_kv == nullptr || d_score == nullptr || d_ape == nullptr || d_kv_state == nullptr || d_score_state == nullptr || write_slot < 0 || state_cols <= 0) return false;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    const int threads = 256;
    const int blocks = (state_cols + threads - 1) / threads;
    compressor_update_state_kernel<<<blocks, threads, 0, cuda_stream>>>(d_kv, d_score, d_ape, d_kv_state, d_score_state, write_slot, state_cols);
    return cudaGetLastError() == cudaSuccess;
}

bool compressor_pool_cuda(const float* d_kv_state, const float* d_score_state, float* d_out, int ratio, int head_dim, int state_cols, bool overlap, void* stream) {
    if (d_kv_state == nullptr || d_score_state == nullptr || d_out == nullptr || ratio <= 0 || head_dim <= 0 || state_cols < head_dim) return false;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    const int threads = 256;
    const int blocks = (head_dim + threads - 1) / threads;
    compressor_pool_kernel<<<blocks, threads, 0, cuda_stream>>>(d_kv_state, d_score_state, d_out, ratio, head_dim, state_cols, overlap);
    return cudaGetLastError() == cudaSuccess;
}

bool compressor_shift_overlap_state_cuda(float* d_kv_state, float* d_score_state, int ratio, int state_cols, void* stream) {
    if (d_kv_state == nullptr || d_score_state == nullptr || ratio <= 0 || state_cols <= 0) return false;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    const int total = ratio * state_cols;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    compressor_shift_overlap_state_kernel<<<blocks, threads, 0, cuda_stream>>>(d_kv_state, d_score_state, ratio, state_cols);
    return cudaGetLastError() == cudaSuccess;
}

bool head_rmsnorm_rope_cuda(
    float* d_x,
    int heads,
    int head_dim,
    int rope_dim,
    int position,
    float theta,
    bool inverse,
    float eps,
    void* stream) {
    if (d_x == nullptr || heads <= 0 || head_dim <= 0 || rope_dim < 0 || rope_dim > head_dim || (rope_dim % 2) != 0 || theta <= 0.0f) return false;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    if (eps > 0.0f) {
        head_rmsnorm_rope_kernel<<<heads, 256, 0, cuda_stream>>>(d_x, head_dim, rope_dim, position, theta, inverse, eps);
    } else {
        rope_kernel<<<heads, 256, 0, cuda_stream>>>(d_x, head_dim, rope_dim, position, theta, inverse);
    }
    return cudaGetLastError() == cudaSuccess;
}

bool head_rmsnorm_rope_rows_cuda(
    float* d_x,
    int tokens,
    int heads,
    int head_dim,
    int rope_dim,
    int start_position,
    float theta,
    bool inverse,
    float eps,
    void* stream) {
    if (d_x == nullptr || tokens <= 0 || heads <= 0 || head_dim <= 0 || rope_dim < 0 || rope_dim > head_dim || (rope_dim % 2) != 0 || start_position < 0 || theta <= 0.0f) return false;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    head_rmsnorm_rope_rows_kernel<<<dim3(heads, tokens), 256, 0, cuda_stream>>>(
        d_x, tokens, heads, head_dim, rope_dim, start_position, theta, inverse, eps);
    return cudaGetLastError() == cudaSuccess;
}

bool head_rmsnorm_rope_freqs_cuda(
    float* d_x,
    const float* d_inv_freqs,
    int heads,
    int head_dim,
    int rope_dim,
    int position,
    bool inverse,
    float eps,
    void* stream) {
    if (d_x == nullptr || d_inv_freqs == nullptr || heads <= 0 || head_dim <= 0 || rope_dim <= 0 || rope_dim > head_dim || (rope_dim % 2) != 0) return false;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    head_rmsnorm_rope_freqs_kernel<<<heads, 256, 0, cuda_stream>>>(d_x, d_inv_freqs, head_dim, rope_dim, position, inverse, eps);
    return cudaGetLastError() == cudaSuccess;
}

bool head_rmsnorm_rope_freqs_rows_cuda(
    float* d_x,
    const float* d_inv_freqs,
    int tokens,
    int heads,
    int head_dim,
    int rope_dim,
    int start_position,
    bool inverse,
    float eps,
    void* stream) {
    if (d_x == nullptr || d_inv_freqs == nullptr || tokens <= 0 || heads <= 0 || head_dim <= 0 || rope_dim <= 0 || rope_dim > head_dim || (rope_dim % 2) != 0) return false;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    head_rmsnorm_rope_freqs_rows_kernel<<<dim3(heads, tokens), 256, 0, cuda_stream>>>(d_x, d_inv_freqs, tokens, heads, head_dim, rope_dim, start_position, inverse, eps);
    return cudaGetLastError() == cudaSuccess;
}

bool fp32_to_bf16_cuda(const float* d_x, uint16_t* d_y, int count, void* stream) {
    if (d_x == nullptr || d_y == nullptr || count <= 0) return false;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    const int threads = 256;
    const int blocks = (count + threads - 1) / threads;
    fp32_to_bf16_kernel<<<blocks, threads, 0, cuda_stream>>>(d_x, d_y, count);
    return cudaGetLastError() == cudaSuccess;
}

bool bf16_to_fp32_cuda(const uint16_t* d_x, float* d_y, int count, void* stream) {
    if (d_x == nullptr || d_y == nullptr || count <= 0) return false;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    const int threads = 256;
    const int blocks = (count + threads - 1) / threads;
    bf16_to_fp32_kernel<<<blocks, threads, 0, cuda_stream>>>(d_x, d_y, count);
    return cudaGetLastError() == cudaSuccess;
}

}  // namespace pocket
