#include "qwen_cuda_ops.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <climits>
#include <cmath>
#include <cstdint>

namespace pocket {
namespace {

constexpr int kThreads = 256;
constexpr int kAttentionThreads = 128;
constexpr int kMaxHeadDim = 128;
constexpr int kMaxSelectorK = 32;

__device__ __forceinline__ float half_to_float(uint16_t bits) {
    return __half2float(__ushort_as_half(bits));
}

__device__ __forceinline__ uint16_t float_to_half(float value) {
    return __half_as_ushort(__float2half_rn(value));
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

__global__ void f16_to_f32_kernel(
    const uint16_t* __restrict__ input, float* __restrict__ output, int count) {
    const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) output[index] = half_to_float(input[index]);
}

template <bool kFloatOutput>
__global__ void rmsnorm_f32_kernel(
    const float* __restrict__ input,
    const uint16_t* __restrict__ gamma,
    void* __restrict__ output,
    int columns,
    float eps) {
    const int row = static_cast<int>(blockIdx.x);
    float sum = 0.0f;
    for (int column = static_cast<int>(threadIdx.x); column < columns;
         column += blockDim.x) {
        const float value = input[static_cast<size_t>(row) * columns + column];
        sum += value * value;
    }
    __shared__ float scratch[kThreads];
    scratch[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        __syncthreads();
    }
    const float inverse = rsqrtf(scratch[0] / static_cast<float>(columns) + eps);
    for (int column = static_cast<int>(threadIdx.x); column < columns;
         column += blockDim.x) {
        const size_t index = static_cast<size_t>(row) * columns + column;
        const float value =
            input[index] * inverse * half_to_float(gamma[column]);
        if constexpr (kFloatOutput) {
            static_cast<float*>(output)[index] = value;
        } else {
            static_cast<uint16_t*>(output)[index] = float_to_half(value);
        }
    }
}

__global__ void add_f32_kernel(
    float* __restrict__ output, const float* __restrict__ input, int count) {
    const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) output[index] += input[index];
}

template <bool kFloatInput, bool kFloatOutput>
__global__ void grouped_dynamic_conv_kernel(
    const void* __restrict__ hidden,
    const uint16_t* __restrict__ dynamic,
    const uint16_t* __restrict__ base,
    void* __restrict__ output,
    int rows, int hidden_size, int groups, int group_size, int kernel_size,
    int dynamic_row_stride, int dynamic_offset) {
    const int index = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int total = rows * hidden_size;
    if (index >= total) return;
    const int row = index / hidden_size;
    const int channel = index - row * hidden_size;
    const int group = channel / group_size;
    float result = 0.0f;
    for (int tap = 0; tap < kernel_size; ++tap) {
        const int source_row = row - tap;
        float value = 0.0f;
        if (source_row >= 0) {
            const size_t source =
                static_cast<size_t>(source_row) * hidden_size + channel;
            value = kFloatInput
                ? static_cast<const float*>(hidden)[source]
                : half_to_float(static_cast<const uint16_t*>(hidden)[source]);
        }
        const float base_value = half_to_float(base[tap * hidden_size + channel]);
        const float dynamic_value = half_to_float(
            dynamic[static_cast<size_t>(row) * dynamic_row_stride +
                    dynamic_offset + tap * groups + group]);
        result += value * (base_value + dynamic_value);
    }
    if constexpr (kFloatOutput) {
        static_cast<float*>(output)[index] = result;
    } else {
        static_cast<uint16_t*>(output)[index] = float_to_half(result);
    }
}

__global__ void head_rmsnorm_f16_kernel(
    const uint16_t* __restrict__ input,
    const uint16_t* __restrict__ gamma,
    uint16_t* __restrict__ output,
    int rows, int heads, int head_dim, float eps) {
    const int head_row = static_cast<int>(blockIdx.x);
    if (head_row >= rows * heads) return;
    const int base = head_row * head_dim;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        const float value = half_to_float(input[base + i]);
        sum += value * value;
    }
    __shared__ float scratch[kThreads];
    scratch[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float inverse = rsqrtf(scratch[0] / static_cast<float>(head_dim) + eps);
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        output[base + i] = float_to_half(
            half_to_float(input[base + i]) * inverse * half_to_float(gamma[i]));
    }
}

__global__ void qwen3_rope_rows_kernel(
    uint16_t* __restrict__ q, uint16_t* __restrict__ k,
    int rows, int q_heads, int kv_heads, int head_dim,
    int start_position, float theta) {
    const int pair = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int row = static_cast<int>(blockIdx.y);
    if (row >= rows || pair >= head_dim / 2) return;
    const float exponent = 2.0f * static_cast<float>(pair) /
                           static_cast<float>(head_dim);
    const float inverse = powf(theta, -exponent);
    const float angle = static_cast<float>(start_position + row) * inverse;
    float sine = 0.0f;
    float cosine = 0.0f;
    sincosf(angle, &sine, &cosine);
    for (int head = 0; head < q_heads; ++head) {
        const size_t base = (static_cast<size_t>(row) * q_heads + head) * head_dim;
        const float left = half_to_float(q[base + pair]);
        const float right = half_to_float(q[base + head_dim / 2 + pair]);
        q[base + pair] = float_to_half(left * cosine - right * sine);
        q[base + head_dim / 2 + pair] = float_to_half(right * cosine + left * sine);
    }
    for (int head = 0; head < kv_heads; ++head) {
        const size_t base = (static_cast<size_t>(row) * kv_heads + head) * head_dim;
        const float left = half_to_float(k[base + pair]);
        const float right = half_to_float(k[base + head_dim / 2 + pair]);
        k[base + pair] = float_to_half(left * cosine - right * sine);
        k[base + head_dim / 2 + pair] = float_to_half(right * cosine + left * sine);
    }
}

__global__ void qwen3_rope_k_rows_kernel(
    uint16_t* __restrict__ k, int rows, int kv_heads, int head_dim,
    int start_position, float theta) {
    const int pair = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int row = static_cast<int>(blockIdx.y);
    if (row >= rows || pair >= head_dim / 2) return;
    const float exponent = 2.0f * static_cast<float>(pair) /
                           static_cast<float>(head_dim);
    const float inverse = powf(theta, -exponent);
    const float angle = static_cast<float>(start_position + row) * inverse;
    float sine = 0.0f;
    float cosine = 0.0f;
    sincosf(angle, &sine, &cosine);
    for (int head = 0; head < kv_heads; ++head) {
        const size_t base = (static_cast<size_t>(row) * kv_heads + head) * head_dim;
        const float left = half_to_float(k[base + pair]);
        const float right = half_to_float(k[base + head_dim / 2 + pair]);
        k[base + pair] = float_to_half(left * cosine - right * sine);
        k[base + head_dim / 2 + pair] = float_to_half(right * cosine + left * sine);
    }
}

__global__ void dflash2_attention_kernel(
    const uint16_t* __restrict__ q,
    const uint16_t* __restrict__ context_k,
    const uint16_t* __restrict__ context_v,
    const uint16_t* __restrict__ block_k,
    const uint16_t* __restrict__ block_v,
    uint16_t* __restrict__ output,
    int block_rows, int q_heads, int kv_heads, int head_dim,
    int context_len, int max_context, int sliding_window) {
    const int q_head = static_cast<int>(blockIdx.x);
    const int query_row = static_cast<int>(blockIdx.y);
    if (q_head >= q_heads || query_row >= block_rows) return;
    const int kv_head = q_head / (q_heads / kv_heads);
    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int query_position = context_len + query_row;
    const size_t q_base = (static_cast<size_t>(query_row) * q_heads + q_head) * head_dim;
    const float query = tid < head_dim ? half_to_float(q[q_base + tid]) : 0.0f;
    float accumulator = 0.0f;
    __shared__ float warp_partials[kAttentionThreads / 32];
    __shared__ float score;
    __shared__ float running_max;
    __shared__ float running_sum;
    __shared__ float previous_scale;
    __shared__ float probability;
    if (tid == 0) {
        running_max = -INFINITY;
        running_sum = 0.0f;
    }
    __syncthreads();

    const int context_begin = max(0, query_position - sliding_window + 1);
    const int context_count = context_len - context_begin;
    const int total_keys = context_count + block_rows;
    for (int index = 0; index < total_keys; ++index) {
        const bool from_context = index < context_count;
        const int position = from_context
            ? context_begin + index : index - context_count;
        const int key_position = from_context ? position : context_len + position;
        if (abs(query_position - key_position) >= sliding_window) continue;
        const uint16_t* key = from_context ? context_k : block_k;
        const uint16_t* value = from_context ? context_v : block_v;
        const size_t base = (static_cast<size_t>(position) * kv_heads + kv_head) * head_dim;
        float partial = tid < head_dim ? query * half_to_float(key[base + tid]) : 0.0f;
        partial = warp_sum(partial);
        if (lane == 0) warp_partials[warp] = partial;
        __syncthreads();
        if (tid == 0) {
            float dot = 0.0f;
#pragma unroll
            for (int i = 0; i < kAttentionThreads / 32; ++i) dot += warp_partials[i];
            score = dot * rsqrtf(static_cast<float>(head_dim));
            const float next_max = fmaxf(running_max, score);
            previous_scale = running_max == -INFINITY ? 0.0f : expf(running_max - next_max);
            probability = expf(score - next_max);
            running_sum = running_sum * previous_scale + probability;
            running_max = next_max;
        }
        __syncthreads();
        if (tid < head_dim) {
            accumulator = accumulator * previous_scale + probability * half_to_float(value[base + tid]);
        }
        __syncthreads();
    }
    if (tid < head_dim) {
        output[q_base + tid] = float_to_half(
            running_sum > 0.0f ? accumulator / running_sum : 0.0f);
    }
    (void)max_context;
}

__device__ __forceinline__ bool better_candidate(
    float left_value, int left_token, float right_value, int right_token) {
    return left_value > right_value ||
        (left_value == right_value && left_token < right_token);
}

__global__ void local_topk_f32_kernel(
    const float* __restrict__ logits, int* __restrict__ tokens,
    float* __restrict__ values, int rows, int vocab, int vocab_start,
    int top_k) {
    const int row = static_cast<int>(blockIdx.x);
    if (row >= rows || top_k > kMaxSelectorK) return;
    float best_values[kMaxSelectorK];
    int best_tokens[kMaxSelectorK];
    for (int i = 0; i < top_k; ++i) {
        best_values[i] = -INFINITY;
        best_tokens[i] = INT_MAX;
    }
    const float* row_logits = logits + static_cast<size_t>(row) * vocab;
    for (int local_token = static_cast<int>(threadIdx.x); local_token < vocab;
         local_token += blockDim.x) {
        const float value = row_logits[local_token];
        const int token = vocab_start + local_token;
        if (isnan(value)) continue;
        int insert = top_k;
        for (int i = 0; i < top_k; ++i) {
            if (better_candidate(value, token, best_values[i], best_tokens[i])) {
                insert = i;
                break;
            }
        }
        if (insert < top_k) {
            for (int i = top_k - 1; i > insert; --i) {
                best_values[i] = best_values[i - 1];
                best_tokens[i] = best_tokens[i - 1];
            }
            best_values[insert] = value;
            best_tokens[insert] = token;
        }
    }

    extern __shared__ unsigned char shared_bytes[];
    float* shared_values = reinterpret_cast<float*>(shared_bytes);
    int* shared_tokens = reinterpret_cast<int*>(
        shared_values + static_cast<size_t>(blockDim.x) * top_k);
    for (int i = 0; i < top_k; ++i) {
        const int index = static_cast<int>(threadIdx.x) * top_k + i;
        shared_values[index] = best_values[i];
        shared_tokens[index] = best_tokens[i];
    }
    __syncthreads();

    // Reduce each warp's 32 sorted lists independently before the final merge.
    // The old implementation made lane 0 scan all blockDim.x * top_k entries;
    // for the seven DFlash2 rows this serialized most of the top-k work in one
    // thread. Each warp writes one exact top-k list back into the same shared
    // storage, so the comparator and tie-breaking remain unchanged.
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    if (lane == 0) {
        for (int i = 0; i < top_k; ++i) {
            best_values[i] = -INFINITY;
            best_tokens[i] = INT_MAX;
        }
        const int candidate_base = warp * 32 * top_k;
        for (int candidate = 0; candidate < 32 * top_k; ++candidate) {
            const float value = shared_values[candidate_base + candidate];
            const int token = shared_tokens[candidate_base + candidate];
            int insert = top_k;
            for (int i = 0; i < top_k; ++i) {
                if (better_candidate(value, token,
                                     best_values[i], best_tokens[i])) {
                    insert = i;
                    break;
                }
            }
            if (insert < top_k) {
                for (int i = top_k - 1; i > insert; --i) {
                    best_values[i] = best_values[i - 1];
                    best_tokens[i] = best_tokens[i - 1];
                }
                best_values[insert] = value;
                best_tokens[insert] = token;
            }
        }
        for (int i = 0; i < top_k; ++i) {
            shared_values[warp * top_k + i] = best_values[i];
            shared_tokens[warp * top_k + i] = best_tokens[i];
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        for (int i = 0; i < top_k; ++i) {
            best_values[i] = -INFINITY;
            best_tokens[i] = INT_MAX;
        }
        const int candidate_count = (static_cast<int>(blockDim.x) / 32) * top_k;
        for (int candidate = 0; candidate < candidate_count; ++candidate) {
            const float value = shared_values[candidate];
            const int token = shared_tokens[candidate];
            int insert = top_k;
            for (int i = 0; i < top_k; ++i) {
                if (better_candidate(value, token,
                                     best_values[i], best_tokens[i])) {
                    insert = i;
                    break;
                }
            }
            if (insert < top_k) {
                for (int i = top_k - 1; i > insert; --i) {
                    best_values[i] = best_values[i - 1];
                    best_tokens[i] = best_tokens[i - 1];
                }
                best_values[insert] = value;
                best_tokens[insert] = token;
            }
        }
        for (int i = 0; i < top_k; ++i) {
            tokens[row * top_k + i] = best_tokens[i];
            values[row * top_k + i] = best_values[i];
        }
    }
}

__device__ __forceinline__ uint32_t ordered_float_bits(float value) {
    // Canonicalize signed zero because the host comparator treats -0 and +0 as
    // equal and then breaks the tie by token id.
    if (value == 0.0f) value = 0.0f;
    const uint32_t bits = __float_as_uint(value);
    return (bits & 0x80000000u) != 0u
        ? ~bits
        : (bits ^ 0x80000000u);
}

__device__ __forceinline__ float unordered_float_bits(uint32_t ordered) {
    const uint32_t bits = (ordered & 0x80000000u) != 0u
        ? (ordered ^ 0x80000000u)
        : ~ordered;
    return __uint_as_float(bits);
}

// Encode the same total order as better_candidate() into one unsigned integer.
// The high word is monotonic in the FP32 logit; the low word is inverted token
// id so ncclMax selects the smaller token when logits compare equal. NaNs are
// represented by zero and therefore lose to every valid logit, including -inf.
__global__ void pack_top1_key_f32_kernel(
    const int* __restrict__ tokens, const float* __restrict__ values,
    uint64_t* __restrict__ keys, int rows) {
    const int row = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const float value = values[row];
    if (isnan(value)) {
        keys[row] = 0ull;
        return;
    }
    const uint32_t ordered = ordered_float_bits(value);
    const uint32_t inverted_token =
        0xffffffffu - static_cast<uint32_t>(tokens[row]);
    keys[row] = (static_cast<uint64_t>(ordered) << 32) |
                static_cast<uint64_t>(inverted_token);
}

__global__ void unpack_top1_key_f32_kernel(
    const uint64_t* __restrict__ keys, int* __restrict__ tokens,
    float* __restrict__ values, int rows) {
    const int row = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const uint64_t key = keys[row];
    if (key == 0ull) {
        tokens[row] = INT_MAX;
        values[row] = -INFINITY;
        return;
    }
    tokens[row] = static_cast<int>(0xffffffffu -
                                   static_cast<uint32_t>(key));
    values[row] = unordered_float_bits(static_cast<uint32_t>(key >> 32));
}

__global__ void merge_topk_f32_kernel(
    const int* __restrict__ gathered_tokens,
    const float* __restrict__ gathered_values,
    int* __restrict__ output_tokens,
    float* __restrict__ output_values,
    int world, int rows, int top_k) {
    const int row = static_cast<int>(blockIdx.x);
    if (row >= rows || threadIdx.x != 0 || top_k > kMaxSelectorK) return;

    float best_values[kMaxSelectorK];
    int best_tokens[kMaxSelectorK];
#pragma unroll
    for (int i = 0; i < kMaxSelectorK; ++i) {
        if (i < top_k) {
            best_values[i] = -INFINITY;
            best_tokens[i] = INT_MAX;
        }
    }

    const int local_count = rows * top_k;
    const int candidate_count = world * top_k;
    for (int candidate = 0; candidate < candidate_count; ++candidate) {
        const int source_rank = candidate / top_k;
        const int source_k = candidate - source_rank * top_k;
        const size_t source = static_cast<size_t>(source_rank) * local_count +
                              static_cast<size_t>(row) * top_k + source_k;
        const float value = gathered_values[source];
        const int token = gathered_tokens[source];
        if (isnan(value)) continue;
        int insert = top_k;
        for (int i = 0; i < top_k; ++i) {
            if (better_candidate(value, token, best_values[i], best_tokens[i])) {
                insert = i;
                break;
            }
        }
        if (insert < top_k) {
            for (int i = top_k - 1; i > insert; --i) {
                best_values[i] = best_values[i - 1];
                best_tokens[i] = best_tokens[i - 1];
            }
            best_values[insert] = value;
            best_tokens[insert] = token;
        }
    }

    for (int i = 0; i < top_k; ++i) {
        const size_t output = static_cast<size_t>(row) * top_k + i;
        output_tokens[output] = best_tokens[i];
        output_values[output] = best_values[i];
    }
}

__global__ void dflash2_attention_grouped_kernel(
    const uint16_t* __restrict__ q,
    const uint16_t* __restrict__ context_k,
    const uint16_t* __restrict__ context_v,
    const uint16_t* __restrict__ block_k,
    const uint16_t* __restrict__ block_v,
    uint16_t* __restrict__ output,
    int block_rows, int q_heads, int kv_heads, int head_dim,
    int context_len, int max_context, int sliding_window) {
    // One four-warp CTA handles all four query heads sharing one KV head.
    const int kv_head = static_cast<int>(blockIdx.x);
    const int query_row = static_cast<int>(blockIdx.y);
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    constexpr int kHeadsPerBlock = 4;
    if (kv_head >= kv_heads || query_row >= block_rows ||
        warp >= kHeadsPerBlock || q_heads / kv_heads != kHeadsPerBlock ||
        head_dim != 128) {
        return;
    }

    const int q_head = kv_head * kHeadsPerBlock + warp;
    const int query_position = context_len + query_row;
    const int context_begin = max(0, query_position - sliding_window + 1);
    const int context_count = context_len - context_begin;
    const int total_keys = context_count + block_rows;
    const float scale = rsqrtf(static_cast<float>(head_dim));

    float accumulator[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    __shared__ float running_max[kHeadsPerBlock];
    __shared__ float running_sum[kHeadsPerBlock];
    __shared__ float previous_scale[kHeadsPerBlock];
    __shared__ float probability[kHeadsPerBlock];
    __shared__ float key_shared[kMaxHeadDim];
    __shared__ float value_shared[kMaxHeadDim];
    if (lane == 0) {
        running_max[warp] = -INFINITY;
        running_sum[warp] = 0.0f;
    }
    __syncthreads();

    for (int index = 0; index < total_keys; ++index) {
        const bool from_context = index < context_count;
        const int position = from_context ? context_begin + index
                                          : index - context_count;
        const int key_position = from_context ? position : context_len + position;
        if (abs(query_position - key_position) >= sliding_window) continue;

        const uint16_t* key = from_context ? context_k : block_k;
        const uint16_t* value = from_context ? context_v : block_v;
        const size_t base =
            (static_cast<size_t>(position) * kv_heads + kv_head) * head_dim;
        if (threadIdx.x < head_dim) {
            key_shared[threadIdx.x] = half_to_float(key[base + threadIdx.x]);
            value_shared[threadIdx.x] = half_to_float(value[base + threadIdx.x]);
        }
        __syncthreads();
        float dot = 0.0f;
        for (int vector = 0; vector < 4; ++vector) {
            const int dimension = vector * 32 + lane;
            const float query_value =
                half_to_float(q[(static_cast<size_t>(query_row) * q_heads +
                                 q_head) * head_dim + dimension]);
            dot += query_value * key_shared[dimension];
        }
        dot = warp_sum(dot) * scale;
        if (lane == 0) {
            const float next_max = fmaxf(running_max[warp], dot);
            previous_scale[warp] = running_max[warp] == -INFINITY
                ? 0.0f : expf(running_max[warp] - next_max);
            probability[warp] = expf(dot - next_max);
            running_sum[warp] = running_sum[warp] * previous_scale[warp] +
                                probability[warp];
            running_max[warp] = next_max;
        }
        __syncthreads();
        for (int vector = 0; vector < 4; ++vector) {
            const int dimension = vector * 32 + lane;
            accumulator[vector] = accumulator[vector] * previous_scale[warp] +
                probability[warp] * value_shared[dimension];
        }
        __syncthreads();
    }

    const float inverse = running_sum[warp] > 0.0f
        ? 1.0f / running_sum[warp] : 0.0f;
    for (int vector = 0; vector < 4; ++vector) {
        const int dimension = vector * 32 + lane;
        output[(static_cast<size_t>(query_row) * q_heads + q_head) * head_dim +
               dimension] = float_to_half(accumulator[vector] * inverse);
    }
    (void)max_context;
}

__global__ void selector_path_kernel(
    const uint16_t* __restrict__ hidden,
    const int* __restrict__ candidates,
    const float* __restrict__ unary,
    const uint16_t* __restrict__ predecessor,
    const uint16_t* __restrict__ successor,
    const uint16_t* __restrict__ projection,
    int* __restrict__ output,
    int rows, int vocab, int hidden_size, int rank, int top_k,
    int anchor_token) {
    const int tid = static_cast<int>(threadIdx.x);
    __shared__ float projected[256];
    __shared__ float scores[kMaxSelectorK];
    __shared__ int previous;
    if (tid == 0) previous = anchor_token;
    __syncthreads();
    for (int position = 0; position < rows; ++position) {
        if (tid < rank) {
            float value = 0.0f;
            const uint16_t* hidden_row =
                hidden + static_cast<size_t>(position) * hidden_size;
            const uint16_t* projection_row =
                projection + static_cast<size_t>(tid) * hidden_size;
            for (int col = 0; col < hidden_size; ++col) {
                value += half_to_float(hidden_row[col]) *
                         half_to_float(projection_row[col]);
            }
            projected[tid] = value;
        }
        __syncthreads();
        if (tid < top_k) {
            const int token = candidates[position * top_k + tid];
            float score = unary[position * top_k + tid];
            const uint16_t* predecessor_row =
                predecessor + static_cast<size_t>(previous) * rank;
            const uint16_t* successor_row =
                successor + static_cast<size_t>(token) * rank;
            for (int r = 0; r < rank; ++r) {
                score += half_to_float(predecessor_row[r]) * projected[r] *
                         half_to_float(successor_row[r]);
            }
            scores[tid] = score;
        }
        __syncthreads();
        if (tid == 0) {
            int best_index = 0;
            for (int candidate_index = 1; candidate_index < top_k;
                 ++candidate_index) {
                const int token =
                    candidates[position * top_k + candidate_index];
                const int best_token =
                    candidates[position * top_k + best_index];
                if (scores[candidate_index] > scores[best_index] ||
                    (scores[candidate_index] == scores[best_index] &&
                     token < best_token)) {
                    best_index = candidate_index;
                }
            }
            previous = candidates[position * top_k + best_index];
            output[position] = previous;
        }
        __syncthreads();
    }
    (void)vocab;
}

__global__ void selector_projection_warp_kernel(
    const uint16_t* __restrict__ hidden,
    const uint16_t* __restrict__ projection,
    float* __restrict__ output,
    int rows, int hidden_size, int rank) {
    constexpr int kWarpsPerBlock = 8;
    const int warp_id = static_cast<int>(blockIdx.x) * kWarpsPerBlock +
                        (static_cast<int>(threadIdx.x) >> 5);
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int total = rows * rank;
    if (warp_id >= total) return;
    const int row = warp_id / rank;
    const int component = warp_id - row * rank;
    const uint16_t* hidden_row =
        hidden + static_cast<size_t>(row) * hidden_size;
    const uint16_t* projection_row =
        projection + static_cast<size_t>(component) * hidden_size;
    float value = 0.0f;
    for (int col = lane; col < hidden_size; col += 32) {
        value += half_to_float(hidden_row[col]) *
                 half_to_float(projection_row[col]);
    }
    value = warp_sum(value);
    if (lane == 0) output[warp_id] = value;
}

__global__ void selector_path_projected_kernel(
    const float* __restrict__ projected,
    const int* __restrict__ candidates,
    const float* __restrict__ unary,
    const uint16_t* __restrict__ predecessor,
    const uint16_t* __restrict__ successor,
    int* __restrict__ output,
    int rows, int vocab, int rank, int top_k,
    int anchor_token) {
    const int tid = static_cast<int>(threadIdx.x);
    __shared__ float scores[kMaxSelectorK];
    __shared__ int previous;
    if (tid == 0) previous = anchor_token;
    __syncthreads();
    for (int position = 0; position < rows; ++position) {
        if (tid < top_k) {
            const int token = candidates[position * top_k + tid];
            float score = unary[position * top_k + tid];
            const uint16_t* predecessor_row =
                predecessor + static_cast<size_t>(previous) * rank;
            const uint16_t* successor_row =
                successor + static_cast<size_t>(token) * rank;
            const float* projected_row = projected + static_cast<size_t>(position) * rank;
            for (int r = 0; r < rank; ++r) {
                score += half_to_float(predecessor_row[r]) * projected_row[r] *
                         half_to_float(successor_row[r]);
            }
            scores[tid] = score;
        }
        __syncthreads();
        if (tid == 0) {
            int best_index = 0;
            for (int candidate_index = 1; candidate_index < top_k;
                 ++candidate_index) {
                const int token = candidates[position * top_k + candidate_index];
                const int best_token = candidates[position * top_k + best_index];
                if (scores[candidate_index] > scores[best_index] ||
                    (scores[candidate_index] == scores[best_index] && token < best_token)) {
                    best_index = candidate_index;
                }
            }
            previous = candidates[position * top_k + best_index];
            output[position] = previous;
        }
        __syncthreads();
    }
    (void)vocab;
}


// Exact MAP path over the first-order selector chain. Greedy picks the argmax at
// each position given its own previous pick, which can spend the rest of the block
// recovering from one locally-attractive token. Viterbi keeps the best score for
// every candidate at every position and traces back the true maximum, at the same
// asymptotic cost (rows * top_k^2 rank-length dots, ~0.5M mults for 8x16x256).
// The per-edge score and its FP32 accumulation order match the greedy kernel
// exactly, so top_k=1 reproduces greedy bit-for-bit.
constexpr int kMaxDraftRows = 16;

__global__ void selector_path_viterbi_kernel(
    const float* __restrict__ projected,
    const int* __restrict__ candidates,
    const float* __restrict__ unary,
    const uint16_t* __restrict__ predecessor,
    const uint16_t* __restrict__ successor,
    int* __restrict__ output,
    int rows, int vocab, int rank, int top_k,
    int anchor_token) {
    const int tid = static_cast<int>(threadIdx.x);
    __shared__ float dp[kMaxSelectorK];
    __shared__ float next_dp[kMaxSelectorK];
    __shared__ int back[kMaxDraftRows][kMaxSelectorK];
    __shared__ float edge[kMaxSelectorK * kMaxSelectorK];

    // Position 0's predecessor is the committed anchor, so there is one edge per
    // candidate and the node score is its own best score.
    if (tid < top_k) {
        const int token = candidates[tid];
        float score = unary[tid];
        const uint16_t* predecessor_row =
            predecessor + static_cast<size_t>(anchor_token) * rank;
        const uint16_t* successor_row =
            successor + static_cast<size_t>(token) * rank;
        for (int r = 0; r < rank; ++r) {
            score += half_to_float(predecessor_row[r]) * projected[r] *
                     half_to_float(successor_row[r]);
        }
        dp[tid] = score;
        back[0][tid] = -1;
    }
    __syncthreads();

    for (int position = 1; position < rows; ++position) {
        const float* projected_row =
            projected + static_cast<size_t>(position) * rank;
        for (int index = tid; index < top_k * top_k;
             index += static_cast<int>(blockDim.x)) {
            const int from = index / top_k;
            const int to = index % top_k;
            const int previous_token = candidates[(position - 1) * top_k + from];
            const int token = candidates[position * top_k + to];
            float score = unary[position * top_k + to];
            const uint16_t* predecessor_row =
                predecessor + static_cast<size_t>(previous_token) * rank;
            const uint16_t* successor_row =
                successor + static_cast<size_t>(token) * rank;
            for (int r = 0; r < rank; ++r) {
                score += half_to_float(predecessor_row[r]) * projected_row[r] *
                         half_to_float(successor_row[r]);
            }
            edge[index] = score;
        }
        __syncthreads();
        if (tid < top_k) {
            float best = -INFINITY;
            int best_from = 0;
            for (int from = 0; from < top_k; ++from) {
                const float candidate_score = dp[from] + edge[from * top_k + tid];
                const int candidate_token =
                    candidates[(position - 1) * top_k + from];
                const int best_token =
                    candidates[(position - 1) * top_k + best_from];
                if (from == 0 || candidate_score > best ||
                    (candidate_score == best && candidate_token < best_token)) {
                    best = candidate_score;
                    best_from = from;
                }
            }
            next_dp[tid] = best;
            back[position][tid] = best_from;
        }
        __syncthreads();
        if (tid < top_k) dp[tid] = next_dp[tid];
        __syncthreads();
    }

    if (tid == 0) {
        int best = 0;
        for (int candidate = 1; candidate < top_k; ++candidate) {
            const int token = candidates[(rows - 1) * top_k + candidate];
            const int best_token = candidates[(rows - 1) * top_k + best];
            if (dp[candidate] > dp[best] ||
                (dp[candidate] == dp[best] && token < best_token)) {
                best = candidate;
            }
        }
        for (int position = rows - 1; position >= 0; --position) {
            output[position] = candidates[position * top_k + best];
            best = back[position][best];
        }
    }
    (void)vocab;
}


// Partition one row's vocabulary across blocks. Block b of a row scans the
// half-open range [b*chunk, min(vocab, (b+1)*chunk)) and emits its own exact
// top_k list. Because the ranges are disjoint and the comparator is a total
// order over (value, token), merging the per-split lists reproduces the
// single-block result exactly.
__global__ void local_topk_split_stage_kernel(
    const float* __restrict__ logits, int* __restrict__ partial_tokens,
    float* __restrict__ partial_values, int rows, int vocab, int vocab_start,
    int top_k, int splits) {
    const int row = static_cast<int>(blockIdx.y);
    const int split = static_cast<int>(blockIdx.x);
    if (row >= rows || split >= splits || top_k > kMaxSelectorK) return;
    const int chunk = (vocab + splits - 1) / splits;
    const int begin = split * chunk;
    const int end = min(vocab, begin + chunk);

    float best_values[kMaxSelectorK];
    int best_tokens[kMaxSelectorK];
    for (int i = 0; i < top_k; ++i) {
        best_values[i] = -INFINITY;
        best_tokens[i] = INT_MAX;
    }
    const float* row_logits = logits + static_cast<size_t>(row) * vocab;
    for (int local_token = begin + static_cast<int>(threadIdx.x);
         local_token < end; local_token += blockDim.x) {
        const float value = row_logits[local_token];
        const int token = vocab_start + local_token;
        if (isnan(value)) continue;
        int insert = top_k;
        for (int i = 0; i < top_k; ++i) {
            if (better_candidate(value, token, best_values[i], best_tokens[i])) {
                insert = i;
                break;
            }
        }
        if (insert < top_k) {
            for (int i = top_k - 1; i > insert; --i) {
                best_values[i] = best_values[i - 1];
                best_tokens[i] = best_tokens[i - 1];
            }
            best_values[insert] = value;
            best_tokens[insert] = token;
        }
    }

    extern __shared__ unsigned char shared_raw[];
    float* shared_values = reinterpret_cast<float*>(shared_raw);
    int* shared_tokens = reinterpret_cast<int*>(
        shared_values + static_cast<size_t>(blockDim.x) * top_k);
    for (int i = 0; i < top_k; ++i) {
        shared_values[static_cast<int>(threadIdx.x) * top_k + i] = best_values[i];
        shared_tokens[static_cast<int>(threadIdx.x) * top_k + i] = best_tokens[i];
    }
    __syncthreads();

    // Same two-level reduction as the single-block kernel: each warp's lane 0
    // merges its 32 lists, then thread 0 merges the per-warp lists.
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    if (lane == 0) {
        for (int i = 0; i < top_k; ++i) {
            best_values[i] = -INFINITY;
            best_tokens[i] = INT_MAX;
        }
        const int candidate_base = warp * 32 * top_k;
        for (int candidate = 0; candidate < 32 * top_k; ++candidate) {
            const float value = shared_values[candidate_base + candidate];
            const int token = shared_tokens[candidate_base + candidate];
            int insert = top_k;
            for (int i = 0; i < top_k; ++i) {
                if (better_candidate(value, token, best_values[i],
                                     best_tokens[i])) {
                    insert = i;
                    break;
                }
            }
            if (insert < top_k) {
                for (int i = top_k - 1; i > insert; --i) {
                    best_values[i] = best_values[i - 1];
                    best_tokens[i] = best_tokens[i - 1];
                }
                best_values[insert] = value;
                best_tokens[insert] = token;
            }
        }
        for (int i = 0; i < top_k; ++i) {
            shared_values[warp * top_k + i] = best_values[i];
            shared_tokens[warp * top_k + i] = best_tokens[i];
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        for (int i = 0; i < top_k; ++i) {
            best_values[i] = -INFINITY;
            best_tokens[i] = INT_MAX;
        }
        const int candidate_count = (static_cast<int>(blockDim.x) / 32) * top_k;
        for (int candidate = 0; candidate < candidate_count; ++candidate) {
            const float value = shared_values[candidate];
            const int token = shared_tokens[candidate];
            int insert = top_k;
            for (int i = 0; i < top_k; ++i) {
                if (better_candidate(value, token, best_values[i],
                                     best_tokens[i])) {
                    insert = i;
                    break;
                }
            }
            if (insert < top_k) {
                for (int i = top_k - 1; i > insert; --i) {
                    best_values[i] = best_values[i - 1];
                    best_tokens[i] = best_tokens[i - 1];
                }
                best_values[insert] = value;
                best_tokens[insert] = token;
            }
        }
        const size_t base =
            (static_cast<size_t>(row) * splits + split) * top_k;
        for (int i = 0; i < top_k; ++i) {
            partial_tokens[base + i] = best_tokens[i];
            partial_values[base + i] = best_values[i];
        }
    }
}

// Merge the per-split lists of one row. Reuses the same comparator, so ties break
// on the lower token exactly as the single-block path does.
__global__ void local_topk_split_merge_kernel(
    const int* __restrict__ partial_tokens,
    const float* __restrict__ partial_values, int* __restrict__ tokens,
    float* __restrict__ values, int rows, int top_k, int splits) {
    const int row = static_cast<int>(blockIdx.x);
    if (row >= rows || threadIdx.x != 0 || top_k > kMaxSelectorK) return;
    float best_values[kMaxSelectorK];
    int best_tokens[kMaxSelectorK];
    for (int i = 0; i < top_k; ++i) {
        best_values[i] = -INFINITY;
        best_tokens[i] = INT_MAX;
    }
    const size_t row_base = static_cast<size_t>(row) * splits * top_k;
    for (int candidate = 0; candidate < splits * top_k; ++candidate) {
        const float value = partial_values[row_base + candidate];
        const int token = partial_tokens[row_base + candidate];
        if (token == INT_MAX) continue;
        int insert = top_k;
        for (int i = 0; i < top_k; ++i) {
            if (better_candidate(value, token, best_values[i], best_tokens[i])) {
                insert = i;
                break;
            }
        }
        if (insert < top_k) {
            for (int i = top_k - 1; i > insert; --i) {
                best_values[i] = best_values[i - 1];
                best_tokens[i] = best_tokens[i - 1];
            }
            best_values[insert] = value;
            best_tokens[insert] = token;
        }
    }
    for (int i = 0; i < top_k; ++i) {
        tokens[static_cast<size_t>(row) * top_k + i] = best_tokens[i];
        values[static_cast<size_t>(row) * top_k + i] = best_values[i];
    }
}

}  // namespace

bool qwen_dflash2_f16_to_f32_cuda(
    const uint16_t* input, float* output, int count, void* stream) {
    if (!input || !output || count <= 0) return false;
    f16_to_f32_kernel<<<(count + kThreads - 1) / kThreads, kThreads, 0,
        static_cast<cudaStream_t>(stream)>>>(input, output, count);
    return cudaGetLastError() == cudaSuccess;
}

bool qwen_dflash2_rmsnorm_f32_f16_cuda(
    const float* input, const uint16_t* gamma, uint16_t* output,
    int rows, int columns, float eps, void* stream) {
    if (!input || !gamma || !output || rows <= 0 || columns <= 0 || eps < 0.0f) {
        return false;
    }
    rmsnorm_f32_kernel<false><<<rows, kThreads, 0,
        static_cast<cudaStream_t>(stream)>>>(input, gamma, output, columns, eps);
    return cudaGetLastError() == cudaSuccess;
}

bool qwen_dflash2_add_f32_cuda(
    float* output, const float* input, int count, void* stream) {
    if (!input || !output || count <= 0) return false;
    add_f32_kernel<<<(count + kThreads - 1) / kThreads, kThreads, 0,
        static_cast<cudaStream_t>(stream)>>>(output, input, count);
    return cudaGetLastError() == cudaSuccess;
}

bool qwen_dflash2_grouped_dynamic_conv_f16_cuda(
    const uint16_t* hidden, const uint16_t* dynamic, const uint16_t* base,
    uint16_t* output, int rows, int hidden_size, int groups, int group_size,
    int kernel_size, void* stream) {
    return qwen_dflash2_grouped_dynamic_conv_strided_f16_cuda(
        hidden, dynamic, base, output, rows, hidden_size, groups, group_size,
        kernel_size, kernel_size * groups, 0, stream);
}

bool qwen_dflash2_grouped_dynamic_conv_strided_f16_cuda(
    const uint16_t* hidden, const uint16_t* dynamic, const uint16_t* base,
    uint16_t* output, int rows, int hidden_size, int groups, int group_size,
    int kernel_size, int dynamic_row_stride, int dynamic_offset, void* stream) {
    if (!hidden || !dynamic || !base || !output || rows <= 0 || hidden_size <= 0 ||
        groups <= 0 || group_size <= 0 || kernel_size <= 0 || kernel_size > 8 ||
        dynamic_row_stride < kernel_size * groups || dynamic_offset < 0 ||
        dynamic_offset + kernel_size * groups > dynamic_row_stride ||
        groups * group_size != hidden_size) return false;
    grouped_dynamic_conv_kernel<false, false><<<
        (rows * hidden_size + kThreads - 1) / kThreads, kThreads, 0,
        static_cast<cudaStream_t>(stream)>>>(hidden, dynamic, base, output, rows,
                                               hidden_size, groups, group_size,
                                               kernel_size, dynamic_row_stride,
                                               dynamic_offset);
    return cudaGetLastError() == cudaSuccess;
}

bool qwen_dflash2_grouped_dynamic_conv_strided_f32_cuda(
    const float* hidden, const uint16_t* dynamic, const uint16_t* base,
    float* output, int rows, int hidden_size, int groups, int group_size,
    int kernel_size, int dynamic_row_stride, int dynamic_offset, void* stream) {
    if (!hidden || !dynamic || !base || !output || rows <= 0 || hidden_size <= 0 ||
        groups <= 0 || group_size <= 0 || kernel_size <= 0 || kernel_size > 8 ||
        dynamic_row_stride < kernel_size * groups || dynamic_offset < 0 ||
        dynamic_offset + kernel_size * groups > dynamic_row_stride ||
        groups * group_size != hidden_size) return false;
    grouped_dynamic_conv_kernel<true, true><<<
        (rows * hidden_size + kThreads - 1) / kThreads, kThreads, 0,
        static_cast<cudaStream_t>(stream)>>>(hidden, dynamic, base, output, rows,
                                               hidden_size, groups, group_size,
                                               kernel_size, dynamic_row_stride,
                                               dynamic_offset);
    return cudaGetLastError() == cudaSuccess;
}

bool qwen_dflash2_local_topk_f32_cuda(
    const float* logits, int* tokens, float* values, int rows, int vocab,
    int vocab_start, int top_k, void* stream) {
    if (!logits || !tokens || !values || rows <= 0 || vocab <= 0 ||
        vocab_start < 0 || top_k <= 0 || top_k > kMaxSelectorK ||
        top_k > vocab) return false;
    constexpr int kTopKThreads = 128;
    const size_t shared_bytes = static_cast<size_t>(kTopKThreads) * top_k *
                                (sizeof(float) + sizeof(int));
    local_topk_f32_kernel<<<rows, kTopKThreads, shared_bytes,
        static_cast<cudaStream_t>(stream)>>>(
            logits, tokens, values, rows, vocab, vocab_start, top_k);
    return cudaGetLastError() == cudaSuccess;
}

bool qwen_dflash2_merge_topk_f32_cuda(
    const int* gathered_tokens, const float* gathered_values,
    int* tokens, float* values, int world, int rows, int top_k, void* stream) {
    if (!gathered_tokens || !gathered_values || !tokens || !values ||
        world <= 0 || rows <= 0 || top_k <= 0 || top_k > kMaxSelectorK) {
        return false;
    }
    merge_topk_f32_kernel<<<rows, 1, 0, static_cast<cudaStream_t>(stream)>>>(
        gathered_tokens, gathered_values, tokens, values, world, rows, top_k);
    return cudaGetLastError() == cudaSuccess;
}

bool qwen_dflash2_pack_top1_key_f32_cuda(
    const int* tokens, const float* values, uint64_t* keys, int rows,
    void* stream) {
    if (!tokens || !values || !keys || rows <= 0) return false;
    pack_top1_key_f32_kernel<<<(rows + kThreads - 1) / kThreads, kThreads, 0,
        static_cast<cudaStream_t>(stream)>>>(tokens, values, keys, rows);
    return cudaGetLastError() == cudaSuccess;
}

bool qwen_dflash2_unpack_top1_key_f32_cuda(
    const uint64_t* keys, int* tokens, float* values, int rows, void* stream) {
    if (!keys || !tokens || !values || rows <= 0) return false;
    unpack_top1_key_f32_kernel<<<(rows + kThreads - 1) / kThreads, kThreads, 0,
        static_cast<cudaStream_t>(stream)>>>(keys, tokens, values, rows);
    return cudaGetLastError() == cudaSuccess;
}

bool qwen_dflash2_rmsnorm_heads_f16_cuda(
    const uint16_t* input, const uint16_t* gamma, uint16_t* output,
    int rows, int heads, int head_dim, float eps, void* stream) {
    if (!input || !gamma || !output || rows <= 0 || heads <= 0 || head_dim <= 0 ||
        head_dim > kMaxHeadDim || eps < 0.0f) return false;
    head_rmsnorm_f16_kernel<<<rows * heads, kThreads, 0,
        static_cast<cudaStream_t>(stream)>>>(input, gamma, output, rows, heads,
                                               head_dim, eps);
    return cudaGetLastError() == cudaSuccess;
}

bool qwen_dflash2_rope_rows_f16_cuda(
    uint16_t* q, uint16_t* k, int rows, int q_heads, int kv_heads,
    int head_dim, int start_position, float theta, void* stream) {
    if (!q || !k || rows <= 0 || q_heads <= 0 || kv_heads <= 0 ||
        q_heads % kv_heads != 0 || head_dim <= 0 || head_dim > kMaxHeadDim ||
        (head_dim & 1) || start_position < 0 || theta <= 0.0f) return false;
    const dim3 grid(static_cast<unsigned>((head_dim / 2 + kThreads - 1) / kThreads),
                    static_cast<unsigned>(rows));
    qwen3_rope_rows_kernel<<<grid, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
        q, k, rows, q_heads, kv_heads, head_dim, start_position, theta);
    return cudaGetLastError() == cudaSuccess;
}

bool qwen_dflash2_rope_k_rows_f16_cuda(
    uint16_t* k, int rows, int kv_heads, int head_dim,
    int start_position, float theta, void* stream) {
    if (!k || rows <= 0 || kv_heads <= 0 || head_dim <= 0 ||
        head_dim > kMaxHeadDim || (head_dim & 1) || start_position < 0 ||
        theta <= 0.0f) return false;
    const dim3 grid(static_cast<unsigned>((head_dim / 2 + kThreads - 1) / kThreads),
                    static_cast<unsigned>(rows));
    qwen3_rope_k_rows_kernel<<<grid, kThreads, 0,
        static_cast<cudaStream_t>(stream)>>>(
            k, rows, kv_heads, head_dim, start_position, theta);
    return cudaGetLastError() == cudaSuccess;
}

bool qwen_dflash2_attention_f16_cuda(
    const uint16_t* q, const uint16_t* context_k, const uint16_t* context_v,
    const uint16_t* block_k, const uint16_t* block_v, uint16_t* output,
    int block_rows, int q_heads, int kv_heads, int head_dim, int context_len,
    int max_context, int sliding_window, void* stream) {
    if (!q || !block_k || !block_v || !output || block_rows <= 0 || q_heads <= 0 ||
        kv_heads <= 0 || q_heads % kv_heads != 0 || head_dim <= 0 ||
        head_dim > kMaxHeadDim || context_len < 0 || max_context < context_len ||
        sliding_window <= 0 || (context_len > 0 && (!context_k || !context_v))) return false;
    dflash2_attention_kernel<<<dim3(q_heads, block_rows), kAttentionThreads, 0,
        static_cast<cudaStream_t>(stream)>>>(q, context_k, context_v, block_k,
        block_v, output, block_rows, q_heads, kv_heads, head_dim, context_len,
        max_context, sliding_window);
    return cudaGetLastError() == cudaSuccess;
}

bool qwen_dflash2_attention_grouped_f16_cuda(
    const uint16_t* q, const uint16_t* context_k, const uint16_t* context_v,
    const uint16_t* block_k, const uint16_t* block_v, uint16_t* output,
    int block_rows, int q_heads, int kv_heads, int head_dim, int context_len,
    int max_context, int sliding_window, void* stream) {
    if (!q || !block_k || !block_v || !output || block_rows <= 0 || q_heads <= 0 ||
        kv_heads <= 0 || q_heads % kv_heads != 0 || head_dim <= 0 ||
        head_dim > kMaxHeadDim || head_dim != 128 || context_len < 0 ||
        max_context < context_len || sliding_window <= 0 ||
        (context_len > 0 && (!context_k || !context_v))) return false;
    const int q_per_kv = q_heads / kv_heads;
    if (q_per_kv != 4) return false;
    dflash2_attention_grouped_kernel<<<dim3(kv_heads, block_rows),
        kAttentionThreads, 0, static_cast<cudaStream_t>(stream)>>>(
        q, context_k, context_v, block_k, block_v, output, block_rows,
        q_heads, kv_heads, head_dim, context_len, max_context, sliding_window);
    return cudaGetLastError() == cudaSuccess;
}

bool qwen_dflash2_selector_path_f16_cuda(
    const uint16_t* hidden, const int* candidates, const float* unary,
    const uint16_t* predecessor, const uint16_t* successor,
    const uint16_t* projection, int* output, int rows, int vocab,
    int hidden_size, int rank, int top_k, int anchor_token, void* stream) {
    if (!hidden || !candidates || !unary || !predecessor || !successor ||
        !projection || !output || rows <= 0 || vocab <= 0 || hidden_size <= 0 ||
        rank <= 0 || rank > 256 || top_k <= 0 || top_k > kMaxSelectorK ||
        anchor_token < 0 || anchor_token >= vocab) return false;
    selector_path_kernel<<<1, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
        hidden, candidates, unary, predecessor, successor, projection, output,
        rows, vocab, hidden_size, rank, top_k, anchor_token);
    return cudaGetLastError() == cudaSuccess;
}

bool qwen_dflash2_selector_path_projected_f16_cuda(
    const float* projected, const int* candidates, const float* unary,
    const uint16_t* predecessor, const uint16_t* successor, int* output,
    int rows, int vocab, int rank, int top_k, int anchor_token, void* stream) {
    if (!projected || !candidates || !unary || !predecessor || !successor ||
        !output || rows <= 0 || vocab <= 0 || rank <= 0 || rank > 256 ||
        top_k <= 0 || top_k > kMaxSelectorK || anchor_token < 0 ||
        anchor_token >= vocab) return false;
    selector_path_projected_kernel<<<1, kThreads, 0,
        static_cast<cudaStream_t>(stream)>>>(
        projected, candidates, unary, predecessor, successor, output,
        rows, vocab, rank, top_k, anchor_token);
    return cudaGetLastError() == cudaSuccess;
}

bool qwen_dflash2_selector_project_f16_cuda(
    const uint16_t* hidden, const uint16_t* projection, float* output,
    int rows, int hidden_size, int rank, void* stream) {
    if (!hidden || !projection || !output || rows <= 0 || hidden_size <= 0 ||
        rank <= 0 || rank > 256) return false;
    constexpr int kWarpsPerBlock = 8;
    selector_projection_warp_kernel<<<
        (rows * rank + kWarpsPerBlock - 1) / kWarpsPerBlock,
        kWarpsPerBlock * 32, 0, static_cast<cudaStream_t>(stream)>>>(
            hidden, projection, output, rows, hidden_size, rank);
    return cudaGetLastError() == cudaSuccess;
}

bool qwen_dflash2_local_topk_split_f32_cuda(
    const float* logits, int* partial_tokens, float* partial_values,
    int* tokens, float* values, int rows, int vocab, int vocab_start,
    int top_k, int splits, void* stream) {
    if (!logits || !partial_tokens || !partial_values || !tokens || !values ||
        rows <= 0 || vocab <= 0 || vocab_start < 0 || top_k <= 0 ||
        top_k > kMaxSelectorK || top_k > vocab || splits <= 0) return false;
    constexpr int kTopKThreads = 128;
    const size_t shared_bytes = static_cast<size_t>(kTopKThreads) * top_k *
                                (sizeof(float) + sizeof(int));
    const dim3 stage_grid(static_cast<unsigned>(splits),
                          static_cast<unsigned>(rows));
    local_topk_split_stage_kernel<<<stage_grid, kTopKThreads, shared_bytes,
        static_cast<cudaStream_t>(stream)>>>(
            logits, partial_tokens, partial_values, rows, vocab, vocab_start,
            top_k, splits);
    if (cudaGetLastError() != cudaSuccess) return false;
    local_topk_split_merge_kernel<<<rows, 32, 0,
        static_cast<cudaStream_t>(stream)>>>(
            partial_tokens, partial_values, tokens, values, rows, top_k, splits);
    return cudaGetLastError() == cudaSuccess;
}

bool qwen_dflash2_selector_path_viterbi_f16_cuda(
    const float* projected, const int* candidates, const float* unary,
    const uint16_t* predecessor, const uint16_t* successor, int* output,
    int rows, int vocab, int rank, int top_k, int anchor_token, void* stream) {
    if (!projected || !candidates || !unary || !predecessor || !successor ||
        !output || rows <= 0 || rows > kMaxDraftRows || vocab <= 0 ||
        rank <= 0 || top_k <= 0 || top_k > kMaxSelectorK ||
        anchor_token < 0 || anchor_token >= vocab) return false;
    constexpr int kThreads = 256;
    selector_path_viterbi_kernel<<<1, kThreads, 0,
        static_cast<cudaStream_t>(stream)>>>(
            projected, candidates, unary, predecessor, successor, output, rows,
            vocab, rank, top_k, anchor_token);
    return cudaGetLastError() == cudaSuccess;
}


}  // namespace pocket
