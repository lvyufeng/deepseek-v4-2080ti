#include "qwen_cuda_ops.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace pocket {
namespace {

__device__ __forceinline__ float fp16_to_float(uint16_t bits) {
    const uint32_t sign = static_cast<uint32_t>(bits & 0x8000u) << 16;
    const uint32_t exponent = (bits >> 10) & 0x1fu;
    const uint32_t mantissa = bits & 0x03ffu;
    uint32_t value;
    if (exponent == 0) {
        if (mantissa == 0) {
            value = sign;
        } else {
            uint32_t normalized = mantissa;
            int exp = -14;
            while ((normalized & 0x400u) == 0) {
                normalized <<= 1;
                --exp;
            }
            value = sign | (static_cast<uint32_t>(exp + 127) << 23) |
                    ((normalized & 0x3ffu) << 13);
        }
    } else if (exponent == 0x1fu) {
        value = sign | 0x7f800000u | (mantissa << 13);
    } else {
        value = sign | ((exponent - 15 + 127) << 23) | (mantissa << 13);
    }
    return __uint_as_float(value);
}

__device__ __forceinline__ float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float silu(float x) {
    return x * sigmoid(x);
}

__global__ void fp16_matmul_rows_kernel(const float* __restrict__ x,
                                        const uint16_t* __restrict__ w,
                                        float* __restrict__ y,
                                        int batch, int rows, int cols,
                                        int x_stride, int y_stride,
                                        int weight_stride) {
    const int row = static_cast<int>(blockIdx.x);
    const int sample = static_cast<int>(blockIdx.y);
    if (row >= rows || sample >= batch) return;
    const float* x_row = x + static_cast<size_t>(sample) * x_stride;
    const uint16_t* w_row = w + static_cast<size_t>(row) * weight_stride;
    float sum = 0.0f;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        sum += x_row[col] * fp16_to_float(w_row[col]);
    }
    extern __shared__ float scratch[];
    scratch[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) y[static_cast<size_t>(sample) * y_stride + row] = scratch[0];
}

__global__ void embedding_fp16_gather_kernel(const uint16_t* __restrict__ table,
                                             const int* __restrict__ tokens,
                                             float* __restrict__ out, int count,
                                             int cols, int row_start, int row_count) {
    const int token_index = static_cast<int>(blockIdx.x);
    if (token_index >= count) return;
    const int token = tokens[token_index];
    float* dst = out + static_cast<size_t>(token_index) * cols;
    if (token < row_start || token >= row_start + row_count) {
        for (int col = threadIdx.x; col < cols; col += blockDim.x) dst[col] = 0.0f;
        return;
    }
    const uint16_t* src = table + static_cast<size_t>(token - row_start) * cols;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) dst[col] = fp16_to_float(src[col]);
}

__global__ void rmsnorm_fp16_gamma_rows_kernel(const float* __restrict__ x,
                                               const uint16_t* __restrict__ gamma,
                                               float* __restrict__ y, int rows,
                                               int cols, float eps) {
    const int row = static_cast<int>(blockIdx.x);
    if (row >= rows) return;
    const float* src = x + static_cast<size_t>(row) * cols;
    float* dst = y + static_cast<size_t>(row) * cols;
    float sum = 0.0f;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) sum += src[col] * src[col];
    extern __shared__ float scratch[];
    scratch[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        __syncthreads();
    }
    const float inv = rsqrtf(scratch[0] / static_cast<float>(cols) + eps);
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        dst[col] = src[col] * inv * (1.0f + fp16_to_float(gamma[col]));
    }
}

__global__ void gated_rmsnorm_fp16_gamma_rows_kernel(const float* __restrict__ x,
                                                     const uint16_t* __restrict__ gamma,
                                                     const float* __restrict__ gate,
                                                     float* __restrict__ y, int rows,
                                                     int cols, float eps) {
    const int row = static_cast<int>(blockIdx.x);
    if (row >= rows) return;
    const float* src = x + static_cast<size_t>(row) * cols;
    const float* gate_row = gate + static_cast<size_t>(row) * cols;
    float* dst = y + static_cast<size_t>(row) * cols;
    float sum = 0.0f;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) sum += src[col] * src[col];
    extern __shared__ float scratch[];
    scratch[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        __syncthreads();
    }
    const float inv = rsqrtf(scratch[0] / static_cast<float>(cols) + eps);
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        dst[col] = fp16_to_float(gamma[col]) * src[col] * inv * silu(gate_row[col]);
    }
}

__global__ void split_packed_qkv_kernel(const float* __restrict__ packed,
                                        float* __restrict__ q, float* __restrict__ k,
                                        float* __restrict__ v, int rows,
                                        int key_dim, int value_dim) {
    const int i = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int total = rows * (2 * key_dim + value_dim);
    if (i >= total) return;
    const int col = i % (2 * key_dim + value_dim);
    const int row = i / (2 * key_dim + value_dim);
    if (col < key_dim) q[static_cast<size_t>(row) * key_dim + col] = packed[i];
    else if (col < 2 * key_dim) k[static_cast<size_t>(row) * key_dim + col - key_dim] = packed[i];
    else v[static_cast<size_t>(row) * value_dim + col - 2 * key_dim] = packed[i];
}

// One thread per channel, 256 channels per block. The time axis stays serial
// because the convolution is causal and the tail must be updated in order, but
// the original <<<channels, 1>>> launch wasted 31 of every 32 lanes.
__global__ void causal_depthwise_conv_silu_kernel(const float* __restrict__ x,
                                                  const uint16_t* __restrict__ weight,
                                                  float* __restrict__ tail,
                                                  float* __restrict__ y, int seq_len,
                                                  int channels, int kernel, bool update_tail) {
    const int channel = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (channel >= channels) return;
    const uint16_t* w = weight + static_cast<size_t>(channel) * kernel;
    float* dst = y + channel;
    for (int t = 0; t < seq_len; ++t) {
        float value = 0.0f;
        for (int j = 0; j < kernel; ++j) {
            const int source_t = t - (kernel - 1) + j;
            float input = 0.0f;
            if (source_t >= 0) {
                input = x[static_cast<size_t>(source_t) * channels + channel];
            } else if (source_t >= -(kernel - 1) && tail != nullptr) {
                input = tail[static_cast<size_t>(source_t + kernel - 1) * channels + channel];
            }
            value += input * fp16_to_float(w[j]);
        }
        dst[static_cast<size_t>(t) * channels] = silu(value);
    }
    if (update_tail && tail != nullptr && kernel > 1) {
        // The new tail is the last (kernel-1) entries of [old_tail ++ x]. When
        // seq_len < kernel-1 (decode feeds one token at a time) the window still
        // straddles the old tail, so those entries must be shifted down rather
        // than zero-filled. Read everything into registers first: the loop below
        // overwrites the same cells it sources from.
        constexpr int kMaxTail = 8;
        const int tail_len = kernel - 1;
        float next[kMaxTail];
        for (int j = 0; j < tail_len && j < kMaxTail; ++j) {
            const int source_t = seq_len - tail_len + j;
            if (source_t >= 0) {
                next[j] = x[static_cast<size_t>(source_t) * channels + channel];
            } else {
                next[j] = tail[static_cast<size_t>(source_t + tail_len) * channels + channel];
            }
        }
        for (int j = 0; j < tail_len && j < kMaxTail; ++j) {
            tail[static_cast<size_t>(j) * channels + channel] = next[j];
        }
    }
}

__global__ void linear_attn_gates_kernel(const float* __restrict__ a,
                                         const float* __restrict__ b,
                                         const uint16_t* __restrict__ a_log,
                                         const uint16_t* __restrict__ dt_bias,
                                         float* __restrict__ g, float* __restrict__ beta,
                                         int rows, int heads) {
    const int head = static_cast<int>(blockIdx.x);
    if (head >= heads) return;
    const float al = fp16_to_float(a_log[head]);
    const float dt = fp16_to_float(dt_bias[head]);
    for (int row = threadIdx.x; row < rows; row += blockDim.x) {
        const float av = a[static_cast<size_t>(row) * heads + head];
        const float bv = b[static_cast<size_t>(row) * heads + head];
        g[static_cast<size_t>(row) * heads + head] = -expf(al) * log1pf(expf(av + dt));
        beta[static_cast<size_t>(row) * heads + head] = sigmoid(bv);
    }
}

__global__ void gated_delta_step_kernel(float* __restrict__ state,
                                        const float* __restrict__ q,
                                        const float* __restrict__ k,
                                        const float* __restrict__ v,
                                        const float* __restrict__ g,
                                        const float* __restrict__ beta,
                                        float* __restrict__ out, int heads,
                                        int key_heads, int key_dim, int value_dim,
                                        float q_scale) {
    const int head = static_cast<int>(blockIdx.x);
    const int value = static_cast<int>(threadIdx.x);
    const int repeat = heads / key_heads;
    const int key_head = head / repeat;
    const float* q_head = q + static_cast<size_t>(key_head) * key_dim;
    const float* k_head = k + static_cast<size_t>(key_head) * key_dim;

    // L2-normalize q/k once per block and stage them in shared memory. The
    // original version recomputed both 128-term norms in every one of the
    // value_dim threads and re-read k/q from global on each state pass.
    extern __shared__ float smem[];
    float* k_shared = smem;                       // key_dim
    float* q_shared = smem + key_dim;             // key_dim
    float* reduce = smem + 2 * key_dim;           // blockDim.x
    const int tid = static_cast<int>(threadIdx.x);
    const int threads = static_cast<int>(blockDim.x);

    float q_partial = 0.0f;
    float k_partial = 0.0f;
    for (int i = tid; i < key_dim; i += threads) {
        q_partial += q_head[i] * q_head[i];
        k_partial += k_head[i] * k_head[i];
    }
    reduce[tid] = q_partial;
    __syncthreads();
    for (int stride = threads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) reduce[tid] += reduce[tid + stride];
        __syncthreads();
    }
    const float q_norm = rsqrtf(reduce[0] + 1.0e-6f);
    __syncthreads();
    reduce[tid] = k_partial;
    __syncthreads();
    for (int stride = threads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) reduce[tid] += reduce[tid + stride];
        __syncthreads();
    }
    const float k_norm = rsqrtf(reduce[0] + 1.0e-6f);
    __syncthreads();

    for (int i = tid; i < key_dim; i += threads) {
        k_shared[i] = k_head[i] * k_norm;
        q_shared[i] = q_head[i] * q_norm;
    }
    __syncthreads();
    if (head >= heads || value >= value_dim) return;

    float* state_head = state + static_cast<size_t>(head) * key_dim * value_dim;
    const float decay = expf(g[head]);
    const float b = beta[head];

    // Single pass over the state column: apply decay, accumulate k·state, then
    // finish with the delta update and the q·state readout.
    float kv_mem = 0.0f;
    for (int i = 0; i < key_dim; ++i) {
        const float cell = state_head[static_cast<size_t>(i) * value_dim + value] * decay;
        state_head[static_cast<size_t>(i) * value_dim + value] = cell;
        kv_mem += cell * k_shared[i];
    }
    const float delta = (v[static_cast<size_t>(head) * value_dim + value] - kv_mem) * b;
    float result = 0.0f;
    for (int i = 0; i < key_dim; ++i) {
        const size_t idx = static_cast<size_t>(i) * value_dim + value;
        const float cell = state_head[idx] + k_shared[i] * delta;
        state_head[idx] = cell;
        result += cell * q_shared[i];
    }
    out[static_cast<size_t>(head) * value_dim + value] = result * q_scale;
}

// Prefill-only variant: one block per value head walks the whole sequence and
// keeps its state column in registers. Per token the arithmetic and its order
// match gated_delta_step_kernel exactly; the difference is that the 64 KiB
// per-head state is no longer streamed to global memory twice per token, and
// 512 tokens cost one launch instead of 512.
template <int kKeyDim>
__global__ void gated_delta_sequence_kernel(float* __restrict__ state,
                                            const float* __restrict__ q,
                                            const float* __restrict__ k,
                                            const float* __restrict__ v,
                                            const float* __restrict__ g,
                                            const float* __restrict__ beta,
                                            float* __restrict__ out, int rows,
                                            int heads, int key_heads, int value_dim,
                                            float q_scale) {
    const int head = static_cast<int>(blockIdx.x);
    const int value = static_cast<int>(threadIdx.x);
    const int threads = static_cast<int>(blockDim.x);
    const int repeat = heads / key_heads;
    const int key_head = head / repeat;
    const int key_stride = key_heads * kKeyDim;

    __shared__ float k_shared[kKeyDim];
    __shared__ float q_shared[kKeyDim];
    extern __shared__ float reduce[];

    float* state_head = state + static_cast<size_t>(head) * kKeyDim * value_dim;
    float st[kKeyDim];
#pragma unroll
    for (int i = 0; i < kKeyDim; ++i) {
        st[i] = state_head[static_cast<size_t>(i) * value_dim + value];
    }

    for (int t = 0; t < rows; ++t) {
        const float* q_head = q + static_cast<size_t>(t) * key_stride +
                              static_cast<size_t>(key_head) * kKeyDim;
        const float* k_head = k + static_cast<size_t>(t) * key_stride +
                              static_cast<size_t>(key_head) * kKeyDim;
        float q_partial = 0.0f;
        float k_partial = 0.0f;
        for (int i = value; i < kKeyDim; i += threads) {
            q_partial += q_head[i] * q_head[i];
            k_partial += k_head[i] * k_head[i];
        }
        reduce[value] = q_partial;
        __syncthreads();
        for (int stride = threads / 2; stride > 0; stride >>= 1) {
            if (value < stride) reduce[value] += reduce[value + stride];
            __syncthreads();
        }
        const float q_norm = rsqrtf(reduce[0] + 1.0e-6f);
        __syncthreads();
        reduce[value] = k_partial;
        __syncthreads();
        for (int stride = threads / 2; stride > 0; stride >>= 1) {
            if (value < stride) reduce[value] += reduce[value + stride];
            __syncthreads();
        }
        const float k_norm = rsqrtf(reduce[0] + 1.0e-6f);
        __syncthreads();
        for (int i = value; i < kKeyDim; i += threads) {
            k_shared[i] = k_head[i] * k_norm;
            q_shared[i] = q_head[i] * q_norm;
        }
        __syncthreads();

        const float decay = expf(g[static_cast<size_t>(t) * heads + head]);
        const float b = beta[static_cast<size_t>(t) * heads + head];
        float kv_mem = 0.0f;
#pragma unroll
        for (int i = 0; i < kKeyDim; ++i) {
            st[i] *= decay;
            kv_mem += st[i] * k_shared[i];
        }
        const float delta = (v[static_cast<size_t>(t) * heads * value_dim +
                               static_cast<size_t>(head) * value_dim + value] -
                             kv_mem) *
                            b;
        float result = 0.0f;
#pragma unroll
        for (int i = 0; i < kKeyDim; ++i) {
            st[i] += k_shared[i] * delta;
            result += st[i] * q_shared[i];
        }
        out[static_cast<size_t>(t) * heads * value_dim +
            static_cast<size_t>(head) * value_dim + value] = result * q_scale;
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < kKeyDim; ++i) {
        state_head[static_cast<size_t>(i) * value_dim + value] = st[i];
    }
}

__device__ __forceinline__ float rope_inv_freq(int index, int rotary_dim, float theta) {
    return powf(theta, -2.0f * static_cast<float>(index) / static_cast<float>(rotary_dim));
}

__global__ void partial_rope_kernel(float* __restrict__ q, float* __restrict__ k,
                                    int position, int rotary_dim, float theta,
                                    int q_heads, int kv_heads, int head_dim) {
    const int idx = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int half = rotary_dim / 2;
    if (idx >= half) return;
    const float angle = static_cast<float>(position) * rope_inv_freq(idx, rotary_dim, theta);
    const float c = cosf(angle);
    const float s = sinf(angle);
    for (int head = 0; head < q_heads; ++head) {
        float* row = q + static_cast<size_t>(head) * head_dim;
        const float a = row[idx];
        const float b = row[idx + half];
        row[idx] = a * c - b * s;
        row[idx + half] = b * c + a * s;
    }
    for (int head = 0; head < kv_heads; ++head) {
        float* row = k + static_cast<size_t>(head) * head_dim;
        const float a = row[idx];
        const float b = row[idx + half];
        row[idx] = a * c - b * s;
        row[idx + half] = b * c + a * s;
    }
}

// Prefill form: `rows` positions in one launch. Angles and rotations are
// computed exactly as in partial_rope_kernel, one block per (row, channel pair)
// group, so per-token results are unchanged.
__global__ void partial_rope_rows_kernel(float* __restrict__ q, float* __restrict__ k,
                                         int start_position, int rows, int rotary_dim,
                                         float theta, int q_heads, int kv_heads,
                                         int head_dim) {
    const int half = rotary_dim / 2;
    const int idx = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int row = static_cast<int>(blockIdx.y);
    if (idx >= half || row >= rows) return;
    const float angle =
        static_cast<float>(start_position + row) * rope_inv_freq(idx, rotary_dim, theta);
    const float c = cosf(angle);
    const float s = sinf(angle);
    float* q_row = q + static_cast<size_t>(row) * q_heads * head_dim;
    float* k_row = k + static_cast<size_t>(row) * kv_heads * head_dim;
    for (int head = 0; head < q_heads; ++head) {
        float* line = q_row + static_cast<size_t>(head) * head_dim;
        const float a = line[idx];
        const float b = line[idx + half];
        line[idx] = a * c - b * s;
        line[idx + half] = b * c + a * s;
    }
    for (int head = 0; head < kv_heads; ++head) {
        float* line = k_row + static_cast<size_t>(head) * head_dim;
        const float a = line[idx];
        const float b = line[idx + half];
        line[idx] = a * c - b * s;
        line[idx + half] = b * c + a * s;
    }
}

__global__ void split_q_gate_kernel(const float* __restrict__ src,
                                    float* __restrict__ q, float* __restrict__ gate,
                                    int rows, int q_heads, int head_dim) {
    const int i = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int total = rows * q_heads * head_dim;
    if (i >= total) return;
    const int d = i % head_dim;
    const int head = (i / head_dim) % q_heads;
    const int row = i / (q_heads * head_dim);
    const size_t source = (static_cast<size_t>(row) * q_heads + head) * head_dim * 2;
    q[i] = src[source + d];
    gate[i] = src[source + head_dim + d];
}

__global__ void select_kv_heads_kernel(const float* __restrict__ src, float* __restrict__ dst,
                                       int rows, int total_heads, int local_heads,
                                       int head_dim, int head_offset) {
    const int i = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int total = rows * local_heads * head_dim;
    if (i >= total) return;
    const int d = i % head_dim;
    const int head = (i / head_dim) % local_heads;
    const int row = i / (local_heads * head_dim);
    const size_t source = (static_cast<size_t>(row) * total_heads + head_offset + head) * head_dim + d;
    dst[i] = src[source];
}

__global__ void append_kv_cache_kernel(const float* __restrict__ k_rows,
                                       const float* __restrict__ v_rows,
                                       float* __restrict__ k_cache,
                                       float* __restrict__ v_cache, int seq_len,
                                       int kv_heads, int head_dim, int start_pos,
                                       int max_context) {
    const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int total = seq_len * kv_heads * head_dim;
    if (index >= total) return;
    const int token = index / (kv_heads * head_dim);
    const int within = index % (kv_heads * head_dim);
    const int dst_token = start_pos + token;
    if (dst_token >= max_context) return;
    const size_t dst = static_cast<size_t>(dst_token) * kv_heads * head_dim + within;
    k_cache[dst] = k_rows[index];
    v_cache[dst] = v_rows[index];
}

// One block per (query head, token). Threads cooperate along head_dim, and the
// KV cache is streamed exactly once using an online softmax: the running max and
// denominator are rescaled as larger scores appear, so no second pass over the
// cache is needed. head_dim is 256 for Qwen3.8, so a 128-thread block keeps two
// accumulators per thread in registers.
template <int kThreads>
__device__ __forceinline__ void gqa_attention_body(const float* __restrict__ q_row,
                                                   const float* __restrict__ k_cache,
                                                   const float* __restrict__ v_cache,
                                                   float* __restrict__ dst,
                                                   int kv_head, int kv_heads,
                                                   int head_dim, int context_len) {
    extern __shared__ float smem[];
    float* q_shared = smem;              // head_dim
    float* reduce = smem + head_dim;     // kThreads
    const int tid = static_cast<int>(threadIdx.x);
    for (int d = tid; d < head_dim; d += kThreads) q_shared[d] = q_row[d];
    __syncthreads();

    const float scale = rsqrtf(static_cast<float>(head_dim));
    // Per-thread slice of the value accumulator.
    constexpr int kMaxSlice = 8;
    float acc[kMaxSlice];
    const int slice = (head_dim + kThreads - 1) / kThreads;
    for (int i = 0; i < slice && i < kMaxSlice; ++i) acc[i] = 0.0f;

    float running_max = -INFINITY;
    float running_sum = 0.0f;
    const size_t kv_stride = static_cast<size_t>(kv_heads) * head_dim;
    for (int pos = 0; pos < context_len; ++pos) {
        const float* key = k_cache + static_cast<size_t>(pos) * kv_stride +
                           static_cast<size_t>(kv_head) * head_dim;
        float partial = 0.0f;
        for (int d = tid; d < head_dim; d += kThreads) partial += q_shared[d] * key[d];
        // Block reduce the dot product.
        reduce[tid] = partial;
        __syncthreads();
        for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
            if (tid < stride) reduce[tid] += reduce[tid + stride];
            __syncthreads();
        }
        const float score = reduce[0] * scale;
        __syncthreads();

        const float new_max = fmaxf(running_max, score);
        const float rescale = running_max == -INFINITY ? 0.0f : expf(running_max - new_max);
        const float p = expf(score - new_max);
        running_sum = running_sum * rescale + p;
        running_max = new_max;

        const float* value = v_cache + static_cast<size_t>(pos) * kv_stride +
                             static_cast<size_t>(kv_head) * head_dim;
        for (int i = 0, d = tid; i < slice && d < head_dim; ++i, d += kThreads) {
            acc[i] = acc[i] * rescale + p * value[d];
        }
    }

    const float inv = running_sum > 0.0f ? 1.0f / running_sum : 0.0f;
    for (int i = 0, d = tid; i < slice && d < head_dim; ++i, d += kThreads) {
        dst[d] = acc[i] * inv;
    }
}

template <int kThreads>
__global__ void gqa_decode_scores_kernel(const float* __restrict__ q,
                                         const float* __restrict__ k_cache,
                                         float* __restrict__ scores, int q_heads,
                                         int kv_heads, int head_dim, int context_len) {
    const int head = static_cast<int>(blockIdx.x);
    const int pos = static_cast<int>(blockIdx.y);
    if (head >= q_heads || pos >= context_len) return;
    const int kv_head = head / (q_heads / kv_heads);
    const float* q_row = q + static_cast<size_t>(head) * head_dim;
    const float* key = k_cache +
                       (static_cast<size_t>(pos) * kv_heads + kv_head) * head_dim;
    float partial = 0.0f;
    for (int d = static_cast<int>(threadIdx.x); d < head_dim; d += kThreads) {
        partial += q_row[d] * key[d];
    }
    extern __shared__ float reduce[];
    reduce[threadIdx.x] = partial;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) reduce[threadIdx.x] += reduce[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        scores[static_cast<size_t>(head) * context_len + pos] =
            reduce[0] * rsqrtf(static_cast<float>(head_dim));
    }
}

template <int kThreads>
__global__ void gqa_decode_softmax_kernel(float* __restrict__ scores,
                                          int q_heads, int context_len) {
    const int head = static_cast<int>(blockIdx.x);
    if (head >= q_heads) return;
    float* head_scores = scores + static_cast<size_t>(head) * context_len;
    const int tid = static_cast<int>(threadIdx.x);
    __shared__ float reduce[kThreads];

    float local_max = -INFINITY;
    for (int pos = tid; pos < context_len; pos += kThreads) {
        local_max = fmaxf(local_max, head_scores[pos]);
    }
    reduce[tid] = local_max;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) reduce[tid] = fmaxf(reduce[tid], reduce[tid + stride]);
        __syncthreads();
    }
    const float max_score = reduce[0];

    float local_sum = 0.0f;
    for (int pos = tid; pos < context_len; pos += kThreads) {
        const float probability = expf(head_scores[pos] - max_score);
        head_scores[pos] = probability;
        local_sum += probability;
    }
    reduce[tid] = local_sum;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (tid < stride) reduce[tid] += reduce[tid + stride];
        __syncthreads();
    }
    const float inv_sum = reduce[0] > 0.0f ? 1.0f / reduce[0] : 0.0f;
    for (int pos = tid; pos < context_len; pos += kThreads) {
        head_scores[pos] *= inv_sum;
    }
}

// Q heads sharing one KV head consume the same V row. Accumulate a small Q-head
// group together so each V element is loaded once per group instead of per head.
template <int kThreads, int kHeadsPerGroup>
__global__ void gqa_decode_grouped_values_kernel(
    const float* __restrict__ probabilities,
    const float* __restrict__ v_cache,
    float* __restrict__ out,
    int q_heads,
    int kv_heads,
    int head_dim,
    int context_len) {
    const int q_per_kv = q_heads / kv_heads;
    const int groups_per_kv = (q_per_kv + kHeadsPerGroup - 1) / kHeadsPerGroup;
    const int kv_head = static_cast<int>(blockIdx.x) / groups_per_kv;
    const int group = static_cast<int>(blockIdx.x) % groups_per_kv;
    const int first_head = kv_head * q_per_kv + group * kHeadsPerGroup;
    const int d = static_cast<int>(blockIdx.y) * kThreads + static_cast<int>(threadIdx.x);
    if (kv_head >= kv_heads || first_head >= (kv_head + 1) * q_per_kv || d >= head_dim) return;

    float acc[kHeadsPerGroup] = {};
    const size_t kv_stride = static_cast<size_t>(kv_heads) * head_dim;
    for (int pos = 0; pos < context_len; ++pos) {
        const float value = v_cache[static_cast<size_t>(pos) * kv_stride +
                                    static_cast<size_t>(kv_head) * head_dim + d];
#pragma unroll
        for (int i = 0; i < kHeadsPerGroup; ++i) {
            const int head = first_head + i;
            if (head < (kv_head + 1) * q_per_kv) {
                acc[i] += probabilities[static_cast<size_t>(head) * context_len + pos] * value;
            }
        }
    }
#pragma unroll
    for (int i = 0; i < kHeadsPerGroup; ++i) {
        const int head = first_head + i;
        if (head < (kv_head + 1) * q_per_kv) {
            out[static_cast<size_t>(head) * head_dim + d] = acc[i];
        }
    }
}

template <int kThreads>
__global__ void gqa_prefill_attention_kernel(const float* __restrict__ q_rows,
                                             const float* __restrict__ k_cache,
                                             const float* __restrict__ v_cache,
                                             float* __restrict__ out_rows, int seq_len,
                                             int q_heads, int kv_heads, int head_dim,
                                             int position_offset, int max_context) {
    const int head = static_cast<int>(blockIdx.x);
    const int token = static_cast<int>(blockIdx.y);
    if (head >= q_heads || token >= seq_len) return;
    const int kv_head = head / (q_heads / kv_heads);
    const size_t offset = (static_cast<size_t>(token) * q_heads + head) * head_dim;
    gqa_attention_body<kThreads>(q_rows + offset, k_cache, v_cache, out_rows + offset,
                                 kv_head, kv_heads, head_dim,
                                 position_offset + token + 1);
}

__global__ void sigmoid_mul_kernel(const float* __restrict__ x,
                                   const float* __restrict__ gate,
                                   float* __restrict__ y, int count) {
    const int i = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) y[i] = x[i] * sigmoid(gate[i]);
}

__global__ void add_inplace_kernel(float* y, const float* x, int count) {
    const int i = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < count) y[i] += x[i];
}

__global__ void silu_mul_rows_kernel(const float* gate, const float* up, float* y,
                                     int rows, int cols) {
    const int i = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= rows * cols) return;
    y[i] = silu(gate[i]) * up[i];
}

}  // namespace

bool qwen_fp16_matmul_rows_cuda(const float* d_x, const uint16_t* d_w_fp16,
                                float* d_y, int batch, int rows, int cols,
                                int x_stride, int y_stride, int weight_stride,
                                void* stream) {
    if (!d_x || !d_w_fp16 || !d_y || batch <= 0 || rows <= 0 || cols <= 0 ||
        x_stride < cols || y_stride < rows || weight_stride < cols) return false;
    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    dim3 grid(static_cast<unsigned>(rows), static_cast<unsigned>(batch), 1);
    fp16_matmul_rows_kernel<<<grid, 256, 256 * sizeof(float), s>>>(
        d_x, d_w_fp16, d_y, batch, rows, cols, x_stride, y_stride, weight_stride);
    return cudaGetLastError() == cudaSuccess;
}

bool qwen_embedding_fp16_gather_cuda(const uint16_t* d_table_fp16,
                                     const int* d_tokens, float* d_out,
                                     int count, int cols, int row_start,
                                     int row_count, void* stream) {
    if (!d_table_fp16 || !d_tokens || !d_out || count <= 0 || cols <= 0 ||
        row_start < 0 || row_count <= 0) return false;
    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    embedding_fp16_gather_kernel<<<count, 256, 0, s>>>(
        d_table_fp16, d_tokens, d_out, count, cols, row_start, row_count);
    return cudaGetLastError() == cudaSuccess;
}

bool qwen_rmsnorm_fp16_gamma_rows_cuda(const float* d_x, const uint16_t* d_gamma_fp16,
                                       float* d_y, int rows, int cols, float eps,
                                       void* stream) {
    if (!d_x || !d_gamma_fp16 || !d_y || rows <= 0 || cols <= 0 || eps < 0.0f) return false;
    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    rmsnorm_fp16_gamma_rows_kernel<<<rows, 256, 256 * sizeof(float), s>>>(
        d_x, d_gamma_fp16, d_y, rows, cols, eps);
    return cudaGetLastError() == cudaSuccess;
}

bool qwen_gated_rmsnorm_fp16_gamma_rows_cuda(const float* d_x, const uint16_t* d_gamma_fp16,
                                             const float* d_gate, float* d_y,
                                             int rows, int cols, float eps,
                                             void* stream) {
    if (!d_x || !d_gamma_fp16 || !d_gate || !d_y || rows <= 0 || cols <= 0 || eps < 0.0f) return false;
    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    gated_rmsnorm_fp16_gamma_rows_kernel<<<rows, 256, 256 * sizeof(float), s>>>(
        d_x, d_gamma_fp16, d_gate, d_y, rows, cols, eps);
    return cudaGetLastError() == cudaSuccess;
}

bool qwen_split_packed_qkv_cuda(const float* d_packed, float* d_q, float* d_k,
                                float* d_v, int rows, int key_dim, int value_dim,
                                void* stream) {
    if (!d_packed || !d_q || !d_k || !d_v || rows <= 0 || key_dim <= 0 || value_dim <= 0) return false;
    const int total = rows * (2 * key_dim + value_dim);
    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    split_packed_qkv_kernel<<<(total + 255) / 256, 256, 0, s>>>(
        d_packed, d_q, d_k, d_v, rows, key_dim, value_dim);
    return cudaGetLastError() == cudaSuccess;
}

bool qwen_causal_depthwise_conv_silu_cuda(const float* d_x_rows,
                                          const uint16_t* d_weight_fp16,
                                          float* d_tail, float* d_y_rows,
                                          int seq_len, int channels, int kernel,
                                          bool update_tail, void* stream) {
    if (!d_x_rows || !d_weight_fp16 || !d_y_rows || seq_len <= 0 || channels <= 0 ||
        kernel <= 0) return false;
    // The kernel stages the rotated tail in a fixed 8-slot register array.
    if (kernel - 1 > 8) return false;
    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    constexpr int kThreads = 256;
    const int blocks = (channels + kThreads - 1) / kThreads;
    causal_depthwise_conv_silu_kernel<<<blocks, kThreads, 0, s>>>(
        d_x_rows, d_weight_fp16, d_tail, d_y_rows, seq_len, channels, kernel, update_tail);
    return cudaGetLastError() == cudaSuccess;
}

bool qwen_linear_attn_gates_cuda(const float* d_a_rows, const float* d_b_rows,
                                 const uint16_t* d_a_log_fp16,
                                 const uint16_t* d_dt_bias_fp16,
                                 float* d_g_rows, float* d_beta_rows,
                                 int rows, int heads, void* stream) {
    if (!d_a_rows || !d_b_rows || !d_a_log_fp16 || !d_dt_bias_fp16 || !d_g_rows ||
        !d_beta_rows || rows <= 0 || heads <= 0) return false;
    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    linear_attn_gates_kernel<<<heads, 256, 0, s>>>(
        d_a_rows, d_b_rows, d_a_log_fp16, d_dt_bias_fp16,
        d_g_rows, d_beta_rows, rows, heads);
    return cudaGetLastError() == cudaSuccess;
}

bool qwen_gated_delta_step_cuda(float* d_state, const float* d_q, const float* d_k,
                                const float* d_v, const float* d_g, const float* d_beta,
                                float* d_out, int heads, int key_heads, int key_dim,
                                int value_dim, float q_scale, void* stream) {
    if (!d_state || !d_q || !d_k || !d_v || !d_g || !d_beta || !d_out || heads <= 0 ||
        key_heads <= 0 || key_dim <= 0 || value_dim <= 0 || heads % key_heads != 0 ||
        q_scale <= 0.0f) return false;
    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    // The reduction is a power-of-two tree, so round the block up to one and
    // keep it at least value_dim wide (one thread per state column).
    int threads = 32;
    while (threads < value_dim) threads <<= 1;
    if (threads > 1024) return false;
    const size_t shmem = (2u * static_cast<size_t>(key_dim) + threads) * sizeof(float);
    gated_delta_step_kernel<<<heads, threads, shmem, s>>>(
        d_state, d_q, d_k, d_v, d_g, d_beta, d_out,
        heads, key_heads, key_dim, value_dim, q_scale);
    return cudaGetLastError() == cudaSuccess;
}

bool qwen_gated_delta_sequence_cuda(float* d_state, const float* d_q, const float* d_k,
                                    const float* d_v, const float* d_g, const float* d_beta,
                                    float* d_out, int rows, int heads, int key_heads,
                                    int key_dim, int value_dim, float q_scale,
                                    void* stream) {
    if (!d_state || !d_q || !d_k || !d_v || !d_g || !d_beta || !d_out || rows <= 0 ||
        heads <= 0 || key_heads <= 0 || key_dim <= 0 || value_dim <= 0 ||
        heads % key_heads != 0 || q_scale <= 0.0f) return false;
    // The state column lives in registers, so key_dim is a compile-time bound
    // and the block must be exactly one thread per state column.
    if (key_dim != 128 || value_dim != 128) return false;
    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    const int threads = value_dim;
    const size_t shmem = static_cast<size_t>(threads) * sizeof(float);
    gated_delta_sequence_kernel<128><<<heads, threads, shmem, s>>>(
        d_state, d_q, d_k, d_v, d_g, d_beta, d_out, rows, heads, key_heads,
        value_dim, q_scale);
    return cudaGetLastError() == cudaSuccess;
}

bool qwen_partial_rope_cuda(float* d_q, float* d_k, int position, int rotary_dim,
                            float theta, int q_heads, int kv_heads, int head_dim,
                            void* stream) {
    if (!d_q || !d_k || position < 0 || rotary_dim <= 0 || (rotary_dim & 1) != 0 ||
        rotary_dim > head_dim || theta <= 0.0f || q_heads <= 0 || kv_heads <= 0 ||
        head_dim <= 0) return false;
    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    partial_rope_kernel<<<(rotary_dim / 2 + 255) / 256, 256, 0, s>>>(
        d_q, d_k, position, rotary_dim, theta, q_heads, kv_heads, head_dim);
    return cudaGetLastError() == cudaSuccess;
}

bool qwen_partial_rope_rows_cuda(float* d_q, float* d_k, int start_position, int rows,
                                 int rotary_dim, float theta, int q_heads, int kv_heads,
                                 int head_dim, void* stream) {
    if (!d_q || !d_k || start_position < 0 || rows <= 0 || rotary_dim <= 0 ||
        (rotary_dim & 1) != 0 || rotary_dim > head_dim || theta <= 0.0f || q_heads <= 0 ||
        kv_heads <= 0 || head_dim <= 0) return false;
    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    dim3 grid(static_cast<unsigned>((rotary_dim / 2 + 255) / 256),
              static_cast<unsigned>(rows), 1);
    partial_rope_rows_kernel<<<grid, 256, 0, s>>>(d_q, d_k, start_position, rows, rotary_dim,
                                                  theta, q_heads, kv_heads, head_dim);
    return cudaGetLastError() == cudaSuccess;
}

bool qwen_split_q_gate_cuda(const float* d_q_proj, float* d_q, float* d_gate,
                            int rows, int q_heads, int head_dim,
                            void* stream) {
    if (!d_q_proj || !d_q || !d_gate || rows <= 0 || q_heads <= 0 || head_dim <= 0) return false;
    const int total = rows * q_heads * head_dim;
    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    split_q_gate_kernel<<<(total + 255) / 256, 256, 0, s>>>(
        d_q_proj, d_q, d_gate, rows, q_heads, head_dim);
    return cudaGetLastError() == cudaSuccess;
}

bool qwen_select_kv_heads_cuda(const float* d_src, float* d_dst, int rows,
                               int total_kv_heads, int local_kv_heads,
                               int head_dim, int head_offset, void* stream) {
    if (!d_src || !d_dst || rows <= 0 || total_kv_heads <= 0 || local_kv_heads <= 0 ||
        head_dim <= 0 || head_offset < 0 || head_offset + local_kv_heads > total_kv_heads) return false;
    const int total = rows * local_kv_heads * head_dim;
    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    select_kv_heads_kernel<<<(total + 255) / 256, 256, 0, s>>>(
        d_src, d_dst, rows, total_kv_heads, local_kv_heads, head_dim, head_offset);
    return cudaGetLastError() == cudaSuccess;
}

bool qwen_append_kv_cache_cuda(const float* d_k_rows, const float* d_v_rows,
                               float* d_k_cache, float* d_v_cache, int seq_len,
                               int kv_heads, int head_dim, int start_pos,
                               int max_context, void* stream) {
    if (!d_k_rows || !d_v_rows || !d_k_cache || !d_v_cache || seq_len <= 0 ||
        kv_heads <= 0 || head_dim <= 0 || start_pos < 0 || max_context <= 0 ||
        start_pos + seq_len > max_context) return false;
    const int total = seq_len * kv_heads * head_dim;
    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    append_kv_cache_kernel<<<(total + 255) / 256, 256, 0, s>>>(
        d_k_rows, d_v_rows, d_k_cache, d_v_cache, seq_len, kv_heads, head_dim,
        start_pos, max_context);
    return cudaGetLastError() == cudaSuccess;
}

bool qwen_gqa_decode_attention_cuda(const float* d_q, const float* d_k_cache,
                                    const float* d_v_cache, float* d_out,
                                    float* d_score_scratch,
                                    int q_heads, int kv_heads, int head_dim,
                                    int context_len, int max_context,
                                    void* stream) {
    if (!d_q || !d_k_cache || !d_v_cache || !d_out || !d_score_scratch ||
        q_heads <= 0 || kv_heads <= 0 || head_dim <= 0 || context_len <= 0 ||
        context_len > max_context || q_heads % kv_heads != 0) return false;
    constexpr int kThreads = 128;
    if ((head_dim + kThreads - 1) / kThreads > 8) return false;
    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    dim3 score_grid(static_cast<unsigned>(q_heads), static_cast<unsigned>(context_len), 1);
    gqa_decode_scores_kernel<kThreads><<<score_grid, kThreads, kThreads * sizeof(float), s>>>(
        d_q, d_k_cache, d_score_scratch, q_heads, kv_heads, head_dim, context_len);
    if (cudaGetLastError() != cudaSuccess) return false;
    gqa_decode_softmax_kernel<kThreads><<<q_heads, kThreads, 0, s>>>(
        d_score_scratch, q_heads, context_len);
    if (cudaGetLastError() != cudaSuccess) return false;
    constexpr int kHeadsPerGroup = 3;
    const int groups_per_kv = ((q_heads / kv_heads) + kHeadsPerGroup - 1) / kHeadsPerGroup;
    const int dim_blocks = (head_dim + kThreads - 1) / kThreads;
    dim3 value_grid(static_cast<unsigned>(kv_heads * groups_per_kv),
                    static_cast<unsigned>(dim_blocks), 1);
    gqa_decode_grouped_values_kernel<kThreads, kHeadsPerGroup><<<value_grid, kThreads, 0, s>>>(
        d_score_scratch, d_v_cache, d_out, q_heads, kv_heads, head_dim, context_len);
    return cudaGetLastError() == cudaSuccess;
}

bool qwen_gqa_prefill_attention_cuda(const float* d_q_rows, const float* d_k_cache,
                                     const float* d_v_cache, float* d_out_rows,
                                     int seq_len, int q_heads, int kv_heads,
                                     int head_dim, int position_offset,
                                     int max_context, void* stream) {
    if (!d_q_rows || !d_k_cache || !d_v_cache || !d_out_rows || seq_len <= 0 ||
        q_heads <= 0 || kv_heads <= 0 || head_dim <= 0 || position_offset < 0 ||
        position_offset + seq_len > max_context || q_heads % kv_heads != 0) return false;
    constexpr int kThreads = 128;
    if ((head_dim + kThreads - 1) / kThreads > 8) return false;
    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    dim3 grid(static_cast<unsigned>(q_heads), static_cast<unsigned>(seq_len), 1);
    const size_t shmem = (static_cast<size_t>(head_dim) + kThreads) * sizeof(float);
    gqa_prefill_attention_kernel<kThreads><<<grid, kThreads, shmem, s>>>(
        d_q_rows, d_k_cache, d_v_cache, d_out_rows, seq_len, q_heads, kv_heads,
        head_dim, position_offset, max_context);
    return cudaGetLastError() == cudaSuccess;
}

bool qwen_sigmoid_mul_cuda(const float* d_x, const float* d_gate, float* d_y,
                           int count, void* stream) {
    if (!d_x || !d_gate || !d_y || count <= 0) return false;
    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    sigmoid_mul_kernel<<<(count + 255) / 256, 256, 0, s>>>(d_x, d_gate, d_y, count);
    return cudaGetLastError() == cudaSuccess;
}

bool qwen_add_inplace_cuda(float* d_y, const float* d_x, int count, void* stream) {
    if (!d_y || !d_x || count <= 0) return false;
    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    add_inplace_kernel<<<(count + 255) / 256, 256, 0, s>>>(d_y, d_x, count);
    return cudaGetLastError() == cudaSuccess;
}

bool qwen_silu_mul_rows_cuda(const float* d_gate, const float* d_up, float* d_y,
                             int rows, int cols, void* stream) {
    if (!d_gate || !d_up || !d_y || rows <= 0 || cols <= 0) return false;
    const int count = rows * cols;
    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    silu_mul_rows_kernel<<<(count + 255) / 256, 256, 0, s>>>(d_gate, d_up, d_y, rows, cols);
    return cudaGetLastError() == cudaSuccess;
}

}  // namespace pocket
