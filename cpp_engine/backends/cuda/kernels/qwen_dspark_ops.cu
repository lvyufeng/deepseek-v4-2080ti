#include "qwen_cuda_ops.hpp"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace pocket {
namespace {

constexpr int kThreads = 256;
constexpr int kAttentionThreads = 128;
constexpr int kMaxHeadDim = 128;
constexpr int kAttentionWarps = kAttentionThreads / 32;

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

__global__ void standard_rmsnorm_f16_kernel(
    const uint16_t* __restrict__ x,
    const uint16_t* __restrict__ gamma,
    uint16_t* __restrict__ y,
    int cols,
    float eps) {
    const int row = static_cast<int>(blockIdx.x);
    float sum = 0.0f;
    for (int col = static_cast<int>(threadIdx.x); col < cols; col += blockDim.x) {
        const float value = half_to_float(x[static_cast<size_t>(row) * cols + col]);
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
    const float inverse = rsqrtf(scratch[0] / static_cast<float>(cols) + eps);
    for (int col = static_cast<int>(threadIdx.x); col < cols; col += blockDim.x) {
        const size_t index = static_cast<size_t>(row) * cols + col;
        y[index] = float_to_half(
            half_to_float(x[index]) * inverse * half_to_float(gamma[col]));
    }
}

__global__ void yarn_rope_f16_kernel(
    uint16_t* __restrict__ x,
    const float* __restrict__ inverse_frequencies,
    int heads,
    int head_dim,
    int start_position,
    float attention_factor) {
    const int row = static_cast<int>(blockIdx.y);
    const int head = static_cast<int>(blockIdx.x);
    const int half_dim = head_dim / 2;
    const size_t base =
        (static_cast<size_t>(row) * heads + head) * head_dim;
    for (int pair = static_cast<int>(threadIdx.x); pair < half_dim;
         pair += blockDim.x) {
        const float angle = static_cast<float>(start_position + row) *
            inverse_frequencies[pair];
        float sine = 0.0f;
        float cosine = 0.0f;
        sincosf(angle, &sine, &cosine);
        sine *= attention_factor;
        cosine *= attention_factor;
        const float left = half_to_float(x[base + pair]);
        const float right = half_to_float(x[base + half_dim + pair]);
        x[base + pair] = float_to_half(left * cosine - right * sine);
        x[base + half_dim + pair] =
            float_to_half(right * cosine + left * sine);
    }
}

__global__ void dual_source_gqa_f16_kernel(
    const uint16_t* __restrict__ q,
    const uint16_t* __restrict__ context_k,
    const uint16_t* __restrict__ context_v,
    const uint16_t* __restrict__ block_k,
    const uint16_t* __restrict__ block_v,
    uint16_t* __restrict__ output,
    int block_rows,
    int q_heads,
    int kv_heads,
    int head_dim,
    int context_len,
    int max_context) {
    const int q_head = static_cast<int>(blockIdx.x);
    const int query_row = static_cast<int>(blockIdx.y);
    if (q_head >= q_heads || query_row >= block_rows) return;
    const int kv_head = q_head / (q_heads / kv_heads);
    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const size_t q_base =
        (static_cast<size_t>(query_row) * q_heads + q_head) * head_dim;
    float query_value = tid < head_dim ? half_to_float(q[q_base + tid]) : 0.0f;
    float accumulator = 0.0f;

    __shared__ float warp_partials[kAttentionWarps];
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

    const int total_keys = context_len + block_rows;
    const float attention_scale = rsqrtf(static_cast<float>(head_dim));
    for (int key_index = 0; key_index < total_keys; ++key_index) {
        const bool from_context = key_index < context_len;
        const int position = from_context ? key_index : key_index - context_len;
        const size_t base = from_context
            ? (static_cast<size_t>(position) * kv_heads + kv_head) * head_dim
            : (static_cast<size_t>(position) * kv_heads + kv_head) * head_dim;
        const uint16_t* key = from_context ? context_k : block_k;
        const uint16_t* value = from_context ? context_v : block_v;
        float partial = tid < head_dim
            ? query_value * half_to_float(key[base + tid]) : 0.0f;
        partial = warp_sum(partial);
        if (lane == 0) warp_partials[warp] = partial;
        __syncthreads();
        if (tid == 0) {
            float dot = 0.0f;
#pragma unroll
            for (int index = 0; index < kAttentionWarps; ++index) {
                dot += warp_partials[index];
            }
            score = dot * attention_scale;
            const float next_max = fmaxf(running_max, score);
            previous_scale = running_max == -INFINITY
                ? 0.0f : expf(running_max - next_max);
            probability = expf(score - next_max);
            running_sum = running_sum * previous_scale + probability;
            running_max = next_max;
        }
        __syncthreads();
        if (tid < head_dim) {
            accumulator = accumulator * previous_scale +
                probability * half_to_float(value[base + tid]);
        }
        __syncthreads();
    }
    if (tid < head_dim) {
        const float normalized = running_sum > 0.0f
            ? accumulator / running_sum : 0.0f;
        output[q_base + tid] = float_to_half(normalized);
    }
    (void)max_context;
}

__global__ void add_markov_bias_f32_kernel(
    float* __restrict__ logits,
    const uint16_t* __restrict__ embedding,
    const uint16_t* __restrict__ weight,
    int local_vocab,
    int rank) {
    const int output = static_cast<int>(blockIdx.x);
    if (output >= local_vocab) return;
    float partial = 0.0f;
    const uint16_t* row = weight + static_cast<size_t>(output) * rank;
    for (int index = static_cast<int>(threadIdx.x); index < rank;
         index += blockDim.x) {
        partial += half_to_float(embedding[index]) * half_to_float(row[index]);
    }
    __shared__ float scratch[kThreads];
    scratch[threadIdx.x] = partial;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) logits[output] += scratch[0];
}

__global__ void replicated_embedding_gather_f16_kernel(
    const uint16_t* __restrict__ table,
    const int* __restrict__ tokens,
    uint16_t* __restrict__ output,
    int cols,
    int table_rows) {
    const int row = static_cast<int>(blockIdx.x);
    const int token = tokens[row];
    if (token < 0 || token >= table_rows) return;
    for (int col = static_cast<int>(threadIdx.x); col < cols;
         col += blockDim.x) {
        output[static_cast<size_t>(row) * cols + col] =
            table[static_cast<size_t>(token) * cols + col];
    }
}

__global__ void confidence_f16_kernel(
    const uint16_t* __restrict__ hidden,
    const uint16_t* __restrict__ markov,
    const uint16_t* __restrict__ weight,
    const uint16_t* __restrict__ bias,
    float* __restrict__ output,
    int hidden_size,
    int rank) {
    const int row = static_cast<int>(blockIdx.x);
    float partial = 0.0f;
    for (int index = static_cast<int>(threadIdx.x); index < hidden_size;
         index += blockDim.x) {
        partial += half_to_float(hidden[static_cast<size_t>(row) * hidden_size + index]) *
            half_to_float(weight[index]);
    }
    for (int index = static_cast<int>(threadIdx.x); index < rank;
         index += blockDim.x) {
        partial += half_to_float(markov[static_cast<size_t>(row) * rank + index]) *
            half_to_float(weight[hidden_size + index]);
    }
    __shared__ float scratch[kThreads];
    scratch[threadIdx.x] = partial;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const float logit = scratch[0] + half_to_float(bias[0]);
        output[row] = 1.0f / (1.0f + expf(-logit));
    }
}

bool valid_positive_extent(int rows, int cols) {
    return rows > 0 && cols > 0;
}

struct DSparkCublasHandle {
    int device = -1;
    cublasHandle_t handle = nullptr;

    ~DSparkCublasHandle() {
        if (handle != nullptr) cublasDestroy(handle);
    }

    bool get(cublasHandle_t* output) {
        int current_device = 0;
        if (cudaGetDevice(&current_device) != cudaSuccess) return false;
        if (handle != nullptr && current_device != device) {
            cublasDestroy(handle);
            handle = nullptr;
        }
        if (handle == nullptr) {
            if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) return false;
            if (cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH) !=
                CUBLAS_STATUS_SUCCESS) {
                return false;
            }
            device = current_device;
        }
        *output = handle;
        return true;
    }
};

DSparkCublasHandle& dspark_cublas_handle() {
    static DSparkCublasHandle holder;
    return holder;
}

}  // namespace

bool qwen_dspark_fp16_gemm_rows_f16_cuda(
    const uint16_t* d_x_fp16, const uint16_t* d_weight_fp16,
    uint16_t* d_y_fp16, int batch, int output_rows, int columns,
    void* stream) {
    if (d_x_fp16 == nullptr || d_weight_fp16 == nullptr ||
        d_y_fp16 == nullptr || batch <= 0 || output_rows <= 0 ||
        columns <= 0) {
        return false;
    }
    cublasHandle_t handle = nullptr;
    if (!dspark_cublas_handle().get(&handle)) return false;
    const cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    if (cublasSetStream(handle, cuda_stream) != CUBLAS_STATUS_SUCCESS) {
        return false;
    }
    const float alpha = 1.0f;
    const float beta = 0.0f;
    // cuBLAS column-major view: the row-major buffers are W^T [K,N],
    // X^T [K,M], and Y^T [N,M]. Compute Y^T = W^T * X.
    return cublasGemmEx(
        handle, CUBLAS_OP_T, CUBLAS_OP_N,
        output_rows, batch, columns,
        &alpha,
        reinterpret_cast<const __half*>(d_weight_fp16), CUDA_R_16F,
        columns,
        reinterpret_cast<const __half*>(d_x_fp16), CUDA_R_16F,
        columns,
        &beta,
        reinterpret_cast<__half*>(d_y_fp16), CUDA_R_16F,
        output_rows,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP) == CUBLAS_STATUS_SUCCESS;
}

bool qwen_dspark_rmsnorm_f16_cuda(
    const uint16_t* d_x_fp16, const uint16_t* d_gamma_fp16,
    uint16_t* d_y_fp16, int rows, int cols, float eps, void* stream) {
    if (d_x_fp16 == nullptr || d_gamma_fp16 == nullptr || d_y_fp16 == nullptr ||
        !valid_positive_extent(rows, cols) || eps < 0.0f) {
        return false;
    }
    standard_rmsnorm_f16_kernel<<<rows, kThreads, 0,
        static_cast<cudaStream_t>(stream)>>>(
            d_x_fp16, d_gamma_fp16, d_y_fp16, cols, eps);
    return cudaGetLastError() == cudaSuccess;
}

bool qwen_dspark_yarn_rope_f16_cuda(
    uint16_t* d_x_fp16, const float* d_inv_freqs, int rows,
    int heads, int head_dim, int start_position, float attention_factor,
    void* stream) {
    if (d_x_fp16 == nullptr || d_inv_freqs == nullptr || rows <= 0 ||
        heads <= 0 || head_dim <= 0 || head_dim % 2 != 0 ||
        start_position < 0 || attention_factor <= 0.0f) {
        return false;
    }
    const dim3 grid(static_cast<unsigned>(heads), static_cast<unsigned>(rows));
    const int threads = head_dim / 2 < kThreads ? head_dim / 2 : kThreads;
    yarn_rope_f16_kernel<<<grid, threads, 0,
        static_cast<cudaStream_t>(stream)>>>(
            d_x_fp16, d_inv_freqs, heads, head_dim, start_position,
            attention_factor);
    return cudaGetLastError() == cudaSuccess;
}

bool qwen_dspark_dual_source_gqa_f16_cuda(
    const uint16_t* d_q_fp16, const uint16_t* d_context_k_fp16,
    const uint16_t* d_context_v_fp16, const uint16_t* d_block_k_fp16,
    const uint16_t* d_block_v_fp16, uint16_t* d_output_fp16,
    int block_rows, int q_heads, int kv_heads, int head_dim,
    int context_len, int max_context, void* stream) {
    if (d_q_fp16 == nullptr || d_block_k_fp16 == nullptr ||
        d_block_v_fp16 == nullptr || d_output_fp16 == nullptr ||
        block_rows <= 0 || q_heads <= 0 || kv_heads <= 0 ||
        q_heads % kv_heads != 0 || head_dim <= 0 ||
        head_dim > kMaxHeadDim || context_len < 0 ||
        context_len > max_context ||
        (context_len > 0 &&
         (d_context_k_fp16 == nullptr || d_context_v_fp16 == nullptr))) {
        return false;
    }
    const dim3 grid(static_cast<unsigned>(q_heads),
                    static_cast<unsigned>(block_rows));
    dual_source_gqa_f16_kernel<<<grid, kAttentionThreads, 0,
        static_cast<cudaStream_t>(stream)>>>(
            d_q_fp16, d_context_k_fp16, d_context_v_fp16,
            d_block_k_fp16, d_block_v_fp16, d_output_fp16,
            block_rows, q_heads, kv_heads, head_dim, context_len,
            max_context);
    return cudaGetLastError() == cudaSuccess;
}

bool qwen_dspark_add_markov_bias_f32_cuda(
    float* d_logits, const uint16_t* d_markov_embedding_fp16,
    const uint16_t* d_markov_w2_fp16, int local_vocab, int markov_rank,
    void* stream) {
    if (d_logits == nullptr || d_markov_embedding_fp16 == nullptr ||
        d_markov_w2_fp16 == nullptr || local_vocab <= 0 || markov_rank <= 0) {
        return false;
    }
    add_markov_bias_f32_kernel<<<local_vocab, kThreads, 0,
        static_cast<cudaStream_t>(stream)>>>(
            d_logits, d_markov_embedding_fp16, d_markov_w2_fp16,
            local_vocab, markov_rank);
    return cudaGetLastError() == cudaSuccess;
}

bool qwen_dspark_embedding_gather_f16_cuda(
    const uint16_t* d_table_fp16, const int* d_tokens,
    uint16_t* d_output_fp16, int count, int cols, int table_rows,
    void* stream) {
    if (d_table_fp16 == nullptr || d_tokens == nullptr ||
        d_output_fp16 == nullptr || count <= 0 || cols <= 0 ||
        table_rows <= 0) {
        return false;
    }
    replicated_embedding_gather_f16_kernel<<<count, kThreads, 0,
        static_cast<cudaStream_t>(stream)>>>(
            d_table_fp16, d_tokens, d_output_fp16, cols, table_rows);
    return cudaGetLastError() == cudaSuccess;
}

bool qwen_dspark_confidence_f16_cuda(
    const uint16_t* d_hidden_fp16,
    const uint16_t* d_markov_embeddings_fp16,
    const uint16_t* d_weight_fp16, const uint16_t* d_bias_fp16,
    float* d_confidence, int rows, int hidden_size, int markov_rank,
    void* stream) {
    if (d_hidden_fp16 == nullptr || d_markov_embeddings_fp16 == nullptr ||
        d_weight_fp16 == nullptr || d_bias_fp16 == nullptr ||
        d_confidence == nullptr || rows <= 0 || hidden_size <= 0 ||
        markov_rank <= 0) {
        return false;
    }
    confidence_f16_kernel<<<rows, kThreads, 0,
        static_cast<cudaStream_t>(stream)>>>(
            d_hidden_fp16, d_markov_embeddings_fp16, d_weight_fp16,
            d_bias_fp16, d_confidence, hidden_size, markov_rank);
    return cudaGetLastError() == cudaSuccess;
}

}  // namespace pocket
