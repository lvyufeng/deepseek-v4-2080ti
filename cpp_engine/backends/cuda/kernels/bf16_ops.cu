#include "cuda_ops.hpp"

#include <cuda_runtime.h>

namespace pocket {
namespace {

__device__ __forceinline__ float bf16_to_float(uint16_t bits) {
    uint32_t value = static_cast<uint32_t>(bits) << 16;
    float out;
    __builtin_memcpy(&out, &value, sizeof(out));
    return out;
}

__global__ void bf16_row_to_float_kernel(const uint16_t* matrix, float* y, int row, int cols) {
    const int out_row = blockIdx.x;
    const uint16_t* src = matrix + static_cast<size_t>(row + out_row) * cols;
    float* dst = y + static_cast<size_t>(out_row) * cols;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) dst[c] = bf16_to_float(src[c]);
}

__global__ void bf16_rows_to_float_kernel(const uint16_t* matrix, const int* rows, float* y, int count, int cols) {
    const int out_row = blockIdx.x;
    if (out_row >= count) return;
    const uint16_t* src = matrix + static_cast<size_t>(rows[out_row]) * cols;
    float* dst = y + static_cast<size_t>(out_row) * cols;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) dst[c] = bf16_to_float(src[c]);
}

__global__ void bf16_matvec_kernel(const float* x, const uint16_t* w, float* y, int rows, int cols) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    float sum = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        sum += bf16_to_float(w[static_cast<size_t>(row) * cols + c]) * x[c];
    }
    extern __shared__ float scratch[];
    scratch[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) y[row] = scratch[0];
}

// y[t, r] = sum_c x[t, c] * w[r, c], for a handful of tokens against a large
// weight matrix (the draft head's vocab). One block per weight row, looping
// over tokens inside it: the weight is what dominates traffic (129280x4096 bf16
// is ~1 GB), so it must be read once for the whole block of tokens rather than
// once per token as repeated matvecs would.
__global__ void bf16_matvec_rows_kernel(const float* x, const uint16_t* w, float* y,
                                        int tokens, int rows, int cols) {
    constexpr int kMaxTokens = 16;
    const int row = blockIdx.x;
    if (row >= rows) return;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    const int num_warps = blockDim.x >> 5;

    float sums[kMaxTokens];
    for (int t = 0; t < tokens; ++t) sums[t] = 0.0f;

    const uint16_t* w_row = w + static_cast<size_t>(row) * cols;
    for (int c = tid; c < cols; c += blockDim.x) {
        const float wv = bf16_to_float(w_row[c]);
        for (int t = 0; t < tokens; ++t) {
            sums[t] += wv * x[static_cast<size_t>(t) * cols + c];
        }
    }

    extern __shared__ float scratch[];  // [num_warps * kMaxTokens]
    for (int t = 0; t < tokens; ++t) {
        float v = sums[t];
        for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffff, v, off);
        if (lane == 0) scratch[warp_id * kMaxTokens + t] = v;
    }
    __syncthreads();
    if (tid < tokens) {
        float total = 0.0f;
        for (int wi = 0; wi < num_warps; ++wi) total += scratch[wi * kMaxTokens + tid];
        y[static_cast<size_t>(tid) * rows + row] = total;
    }
}

__global__ void bf16_dual_matvec_kernel(const float* x, const uint16_t* w_a, const uint16_t* w_b, float* y_a, float* y_b, int rows, int cols) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    const int num_warps = blockDim.x >> 5;
    float sum_a = 0.0f;
    float sum_b = 0.0f;
    for (int c = tid; c < cols; c += blockDim.x) {
        const float xv = x[c];
        const size_t idx = static_cast<size_t>(row) * cols + c;
        sum_a += bf16_to_float(w_a[idx]) * xv;
        sum_b += bf16_to_float(w_b[idx]) * xv;
    }
    for (int off = 16; off > 0; off >>= 1) {
        sum_a += __shfl_down_sync(0xffffffff, sum_a, off);
        sum_b += __shfl_down_sync(0xffffffff, sum_b, off);
    }
    __shared__ float warp_partials_a[32];
    __shared__ float warp_partials_b[32];
    if (lane == 0) {
        warp_partials_a[warp_id] = sum_a;
        warp_partials_b[warp_id] = sum_b;
    }
    __syncthreads();
    if (warp_id == 0) {
        float val_a = lane < num_warps ? warp_partials_a[lane] : 0.0f;
        float val_b = lane < num_warps ? warp_partials_b[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) {
            val_a += __shfl_down_sync(0xffffffff, val_a, off);
            val_b += __shfl_down_sync(0xffffffff, val_b, off);
        }
        if (lane == 0) {
            y_a[row] = val_a;
            y_b[row] = val_b;
        }
    }
}

__global__ void bf16_dual_matvec_bf16_x_kernel(const uint16_t* x, const uint16_t* w_a, const uint16_t* w_b, float* y_a, float* y_b, int rows, int cols) {
    const int row = blockIdx.x;
    if (row >= rows) return;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    const int num_warps = blockDim.x >> 5;
    float sum_a = 0.0f;
    float sum_b = 0.0f;
    for (int c = tid; c < cols; c += blockDim.x) {
        const float xv = bf16_to_float(x[c]);
        const size_t idx = static_cast<size_t>(row) * cols + c;
        sum_a += bf16_to_float(w_a[idx]) * xv;
        sum_b += bf16_to_float(w_b[idx]) * xv;
    }
    for (int off = 16; off > 0; off >>= 1) {
        sum_a += __shfl_down_sync(0xffffffff, sum_a, off);
        sum_b += __shfl_down_sync(0xffffffff, sum_b, off);
    }
    __shared__ float warp_partials_a[32];
    __shared__ float warp_partials_b[32];
    if (lane == 0) {
        warp_partials_a[warp_id] = sum_a;
        warp_partials_b[warp_id] = sum_b;
    }
    __syncthreads();
    if (warp_id == 0) {
        float val_a = lane < num_warps ? warp_partials_a[lane] : 0.0f;
        float val_b = lane < num_warps ? warp_partials_b[lane] : 0.0f;
        for (int off = 16; off > 0; off >>= 1) {
            val_a += __shfl_down_sync(0xffffffff, val_a, off);
            val_b += __shfl_down_sync(0xffffffff, val_b, off);
        }
        if (lane == 0) {
            y_a[row] = val_a;
            y_b[row] = val_b;
        }
    }
}

__global__ void bf16_matvec_cpu_order_kernel(const float* x, const uint16_t* w, float* y, int rows, int cols) {
    const int row = blockIdx.x;
    if (row >= rows || threadIdx.x != 0) return;
    double sum = 0.0;
    for (int c = 0; c < cols; ++c) sum += static_cast<double>(bf16_to_float(w[static_cast<size_t>(row) * cols + c])) * x[c];
    y[row] = static_cast<float>(sum);
}

__device__ __forceinline__ float sqrt_softplus(float x) {
    return sqrtf(log1pf(expf(x)));
}

__global__ void gate_scores_kernel(const float* x, const uint16_t* w, const float* bias, float* original, float* scored, int experts, int cols) {
    const int row = blockIdx.x;
    const int token = blockIdx.y;
    if (row >= experts) return;
    const float* token_x = x + static_cast<size_t>(token) * cols;
    float* token_original = original + static_cast<size_t>(token) * experts;
    float* token_scored = scored + static_cast<size_t>(token) * experts;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    const int num_warps = blockDim.x >> 5;
    float sum = 0.0f;
    for (int c = tid; c < cols; c += blockDim.x) sum += bf16_to_float(w[static_cast<size_t>(row) * cols + c]) * token_x[c];
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, off);
    __shared__ float warp_partials[32];
    if (lane == 0) warp_partials[warp_id] = sum;
    __syncthreads();
    if (warp_id == 0) {
        float val = lane < num_warps ? warp_partials[lane] : 0.0f;
        for (int off = num_warps / 2; off > 0; off >>= 1) val += __shfl_xor_sync(0xffffffff, val, off);
        if (lane == 0) {
            const float orig = sqrt_softplus(val);
            token_original[row] = orig;
            token_scored[row] = orig + (bias == nullptr ? 0.0f : bias[row]);
        }
    }
}

__global__ void gate_hash_scores_kernel(
    const float* x,
    const uint16_t* w,
    const int64_t* tid2eid,
    float* original,
    int64_t* indices,
    int token_id,
    int cols,
    int table_topk,
    int topk) {
    const int k = blockIdx.x;
    if (k >= topk) return;
    const int64_t expert = tid2eid[static_cast<size_t>(token_id) * table_topk + k];
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    const int num_warps = blockDim.x >> 5;
    float sum = 0.0f;
    for (int c = tid; c < cols; c += blockDim.x) sum += bf16_to_float(w[static_cast<size_t>(expert) * cols + c]) * x[c];
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, off);
    __shared__ float warp_partials[32];
    if (lane == 0) warp_partials[warp_id] = sum;
    __syncthreads();
    if (warp_id == 0) {
        float val = lane < num_warps ? warp_partials[lane] : 0.0f;
        for (int off = num_warps / 2; off > 0; off >>= 1) val += __shfl_xor_sync(0xffffffff, val, off);
        if (lane == 0) {
            original[k] = sqrt_softplus(val);
            indices[k] = expert;
        }
    }
}

__global__ void gate_hash_finalize_kernel(const float* original, float* weights, int topk, float route_scale) {
    if (threadIdx.x != 0) return;
    float denom = 0.0f;
    for (int k = 0; k < topk; ++k) denom += original[k];
    denom = denom == 0.0f ? 1.0f : denom;
    for (int k = 0; k < topk; ++k) weights[k] = original[k] / denom * route_scale;
}

__global__ void gate_hash_rows_scores_kernel(
    const float* x,
    const uint16_t* w,
    const int64_t* tid2eid,
    const int* token_ids,
    int64_t* indices,
    float* weights,
    int cols,
    int table_topk,
    int topk) {
    const int k = blockIdx.x;
    const int token = blockIdx.y;
    if (k >= topk) return;
    const int token_id = token_ids[token];
    const int64_t expert = tid2eid[static_cast<size_t>(token_id) * table_topk + k];
    const float* token_x = x + static_cast<size_t>(token) * cols;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp_id = tid >> 5;
    const int num_warps = blockDim.x >> 5;
    float sum = 0.0f;
    for (int c = tid; c < cols; c += blockDim.x) sum += bf16_to_float(w[static_cast<size_t>(expert) * cols + c]) * token_x[c];
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffff, sum, off);
    __shared__ float warp_partials[32];
    if (lane == 0) warp_partials[warp_id] = sum;
    __syncthreads();
    if (warp_id == 0) {
        float val = lane < num_warps ? warp_partials[lane] : 0.0f;
        for (int off = num_warps / 2; off > 0; off >>= 1) val += __shfl_xor_sync(0xffffffff, val, off);
        if (lane == 0) {
            const size_t out = static_cast<size_t>(token) * topk + k;
            weights[out] = sqrt_softplus(val);
            indices[out] = expert;
        }
    }
}

__global__ void gate_hash_rows_finalize_kernel(float* weights, int topk, float route_scale) {
    const int token = blockIdx.x;
    if (threadIdx.x != 0) return;
    float* token_weights = weights + static_cast<size_t>(token) * topk;
    float denom = 0.0f;
    for (int k = 0; k < topk; ++k) denom += token_weights[k];
    denom = denom == 0.0f ? 1.0f : denom;
    for (int k = 0; k < topk; ++k) token_weights[k] = token_weights[k] / denom * route_scale;
}

__global__ void gate_topk_finalize_kernel(const float* original, const float* scored, int64_t* indices, float* weights, int experts, int topk, float route_scale) {
    const int token = blockIdx.x;
    if (threadIdx.x != 0) return;
    const float* token_original = original + static_cast<size_t>(token) * experts;
    const float* token_scored = scored + static_cast<size_t>(token) * experts;
    int64_t* token_indices = indices + static_cast<size_t>(token) * topk;
    float* token_weights = weights + static_cast<size_t>(token) * topk;
    float denom = 0.0f;
    for (int k = 0; k < topk; ++k) {
        int best = 0;
        float best_score = -INFINITY;
        for (int e = 0; e < experts; ++e) {
            bool used = false;
            for (int j = 0; j < k; ++j) used = used || (token_indices[j] == e);
            const float s = used ? -INFINITY : token_scored[e];
            if (s > best_score) {
                best_score = s;
                best = e;
            }
        }
        token_indices[k] = best;
        denom += token_original[best];
    }
    denom = denom == 0.0f ? 1.0f : denom;
    for (int k = 0; k < topk; ++k) token_weights[k] = token_original[token_indices[k]] / denom * route_scale;
}

}  // namespace

bool bf16_row_to_float_cuda(const uint16_t* d_matrix_bf16, float* d_y, int row, int cols, void* stream) {
    if (d_matrix_bf16 == nullptr || d_y == nullptr || row < 0 || cols <= 0) return false;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    bf16_row_to_float_kernel<<<1, 256, 0, cuda_stream>>>(d_matrix_bf16, d_y, row, cols);
    return cudaGetLastError() == cudaSuccess;
}

bool bf16_rows_to_float_cuda(const uint16_t* d_matrix_bf16, const int* d_rows, float* d_y, int rows, int cols, void* stream) {
    if (d_matrix_bf16 == nullptr || d_rows == nullptr || d_y == nullptr || rows <= 0 || cols <= 0) return false;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    bf16_rows_to_float_kernel<<<rows, 256, 0, cuda_stream>>>(d_matrix_bf16, d_rows, d_y, rows, cols);
    return cudaGetLastError() == cudaSuccess;
}

bool bf16_matvec_cuda(const float* d_x, const uint16_t* d_w_bf16, float* d_y, int rows, int cols, void* stream) {
    if (d_x == nullptr || d_w_bf16 == nullptr || d_y == nullptr || rows <= 0 || cols <= 0) return false;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    bf16_matvec_kernel<<<rows, 256, 256 * sizeof(float), cuda_stream>>>(d_x, d_w_bf16, d_y, rows, cols);
    return cudaGetLastError() == cudaSuccess;
}

bool bf16_matvec_rows_cuda(const float* d_x, const uint16_t* d_w_bf16, float* d_y, int tokens, int rows, int cols, void* stream) {
    constexpr int kMaxTokens = 16;
    if (d_x == nullptr || d_w_bf16 == nullptr || d_y == nullptr) return false;
    if (tokens <= 0 || tokens > kMaxTokens || rows <= 0 || cols <= 0) return false;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    constexpr int kThreads = 256;
    const size_t shmem = static_cast<size_t>(kThreads / 32) * kMaxTokens * sizeof(float);
    bf16_matvec_rows_kernel<<<rows, kThreads, shmem, cuda_stream>>>(d_x, d_w_bf16, d_y, tokens, rows, cols);
    return cudaGetLastError() == cudaSuccess;
}

bool bf16_dual_matvec_cuda(const float* d_x, const uint16_t* d_w_a_bf16, const uint16_t* d_w_b_bf16, float* d_y_a, float* d_y_b, int rows, int cols, void* stream) {    if (d_x == nullptr || d_w_a_bf16 == nullptr || d_w_b_bf16 == nullptr || d_y_a == nullptr || d_y_b == nullptr || rows <= 0 || cols <= 0) return false;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    bf16_dual_matvec_kernel<<<rows, 256, 0, cuda_stream>>>(d_x, d_w_a_bf16, d_w_b_bf16, d_y_a, d_y_b, rows, cols);
    return cudaGetLastError() == cudaSuccess;
}

bool bf16_dual_matvec_bf16_x_cuda(const uint16_t* d_x_bf16, const uint16_t* d_w_a_bf16, const uint16_t* d_w_b_bf16, float* d_y_a, float* d_y_b, int rows, int cols, void* stream) {
    if (d_x_bf16 == nullptr || d_w_a_bf16 == nullptr || d_w_b_bf16 == nullptr || d_y_a == nullptr || d_y_b == nullptr || rows <= 0 || cols <= 0) return false;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    bf16_dual_matvec_bf16_x_kernel<<<rows, 256, 0, cuda_stream>>>(d_x_bf16, d_w_a_bf16, d_w_b_bf16, d_y_a, d_y_b, rows, cols);
    return cudaGetLastError() == cudaSuccess;
}

bool bf16_matvec_cpu_order_cuda(const float* d_x, const uint16_t* d_w_bf16, float* d_y, int rows, int cols, void* stream) {
    if (d_x == nullptr || d_w_bf16 == nullptr || d_y == nullptr || rows <= 0 || cols <= 0) return false;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    bf16_matvec_cpu_order_kernel<<<rows, 1, 0, cuda_stream>>>(d_x, d_w_bf16, d_y, rows, cols);
    return cudaGetLastError() == cudaSuccess;
}

bool gate_topk_bf16_cuda_with_buffers(
    const float* d_x,
    const uint16_t* d_w_bf16,
    const float* d_bias,
    float* d_original,
    float* d_scored,
    int64_t* d_indices,
    float* d_weights,
    int experts,
    int cols,
    int topk,
    float route_scale,
    void* stream) {
    if (d_x == nullptr || d_w_bf16 == nullptr || d_original == nullptr || d_scored == nullptr || d_indices == nullptr || d_weights == nullptr || experts <= 0 || cols <= 0 || topk <= 0) return false;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    gate_scores_kernel<<<dim3(experts, 1), 256, 0, cuda_stream>>>(d_x, d_w_bf16, d_bias, d_original, d_scored, experts, cols);
    gate_topk_finalize_kernel<<<1, 1, 0, cuda_stream>>>(d_original, d_scored, d_indices, d_weights, experts, topk, route_scale);
    return cudaGetLastError() == cudaSuccess;
}

bool gate_topk_bf16_cuda(
    const float* d_x,
    const uint16_t* d_w_bf16,
    const float* d_bias,
    int64_t* d_indices,
    float* d_weights,
    int experts,
    int cols,
    int topk,
    float route_scale,
    void* stream) {
    if (d_x == nullptr || d_w_bf16 == nullptr || d_indices == nullptr || d_weights == nullptr || experts <= 0 || cols <= 0 || topk <= 0) return false;
    float* d_original = nullptr;
    float* d_scored = nullptr;
    if (cudaMalloc(&d_original, static_cast<size_t>(experts) * sizeof(float)) != cudaSuccess) return false;
    if (cudaMalloc(&d_scored, static_cast<size_t>(experts) * sizeof(float)) != cudaSuccess) {
        cudaFree(d_original);
        return false;
    }
    const bool ok = gate_topk_bf16_cuda_with_buffers(d_x, d_w_bf16, d_bias, d_original, d_scored, d_indices, d_weights, experts, cols, topk, route_scale, stream);
    cudaFree(d_original);
    cudaFree(d_scored);
    return ok;
}

bool gate_hash_bf16_cuda(
    const float* d_x,
    const uint16_t* d_w_bf16,
    const int64_t* d_tid2eid,
    float* d_original_scratch,
    int64_t* d_indices,
    float* d_weights,
    int token,
    int cols,
    int table_topk,
    int topk,
    float route_scale,
    void* stream) {
    if (d_x == nullptr || d_w_bf16 == nullptr || d_tid2eid == nullptr || d_original_scratch == nullptr || d_indices == nullptr || d_weights == nullptr || token < 0 || cols <= 0 || table_topk <= 0 || topk <= 0 || topk > table_topk) return false;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    gate_hash_scores_kernel<<<topk, 256, 0, cuda_stream>>>(d_x, d_w_bf16, d_tid2eid, d_original_scratch, d_indices, token, cols, table_topk, topk);
    gate_hash_finalize_kernel<<<1, 1, 0, cuda_stream>>>(d_original_scratch, d_weights, topk, route_scale);
    return cudaGetLastError() == cudaSuccess;
}

bool gate_hash_bf16_rows_cuda(
    const float* d_x,
    const uint16_t* d_w_bf16,
    const int64_t* d_tid2eid,
    const int* d_token_ids,
    int64_t* d_indices,
    float* d_weights,
    int tokens,
    int cols,
    int table_topk,
    int topk,
    float route_scale,
    void* stream) {
    if (d_x == nullptr || d_w_bf16 == nullptr || d_tid2eid == nullptr || d_token_ids == nullptr || d_indices == nullptr || d_weights == nullptr || tokens <= 0 || cols <= 0 || table_topk <= 0 || topk <= 0 || topk > table_topk) return false;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    gate_hash_rows_scores_kernel<<<dim3(topk, tokens), 256, 0, cuda_stream>>>(d_x, d_w_bf16, d_tid2eid, d_token_ids, d_indices, d_weights, cols, table_topk, topk);
    gate_hash_rows_finalize_kernel<<<tokens, 1, 0, cuda_stream>>>(d_weights, topk, route_scale);
    return cudaGetLastError() == cudaSuccess;
}

bool gate_topk_bf16_rows_cuda(
    const float* d_x,
    const uint16_t* d_w_bf16,
    const float* d_bias,
    int64_t* d_indices,
    float* d_weights,
    int tokens,
    int experts,
    int cols,
    int topk,
    float route_scale,
    void* stream) {
    if (d_x == nullptr || d_w_bf16 == nullptr || d_indices == nullptr || d_weights == nullptr || tokens <= 0 || experts <= 0 || cols <= 0 || topk <= 0) return false;
    float* d_original = nullptr;
    float* d_scored = nullptr;
    if (cudaMalloc(&d_original, static_cast<size_t>(tokens) * experts * sizeof(float)) != cudaSuccess) return false;
    if (cudaMalloc(&d_scored, static_cast<size_t>(tokens) * experts * sizeof(float)) != cudaSuccess) {
        cudaFree(d_original);
        return false;
    }
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    gate_scores_kernel<<<dim3(experts, tokens), 256, 0, cuda_stream>>>(d_x, d_w_bf16, d_bias, d_original, d_scored, experts, cols);
    gate_topk_finalize_kernel<<<tokens, 1, 0, cuda_stream>>>(d_original, d_scored, d_indices, d_weights, experts, topk, route_scale);
    const cudaError_t err = cudaGetLastError();
    cudaFree(d_original);
    cudaFree(d_scored);
    return err == cudaSuccess;
}

}  // namespace pocket
