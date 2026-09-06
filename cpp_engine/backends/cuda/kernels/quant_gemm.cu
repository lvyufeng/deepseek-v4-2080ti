#include "cuda_ops.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdlib>

namespace pocket {
namespace {

int local_env_int_or_default(const char* name, int fallback) {
    const char* v = std::getenv(name);
    if (v == nullptr || *v == '\0') return fallback;
    char* end = nullptr;
    const long parsed = std::strtol(v, &end, 10);
    return (end == v) ? fallback : static_cast<int>(parsed);
}

__device__ __forceinline__ float load_f16_scale(const uint8_t* p) {
    const uint16_t h = static_cast<uint16_t>(p[0]) | (static_cast<uint16_t>(p[1]) << 8);
    return __half2float(*reinterpret_cast<const __half*>(&h));
}

__global__ void q8_0_matvec_kernel(const float* __restrict__ x, const uint8_t* __restrict__ w, float* __restrict__ y, int rows, int cols, int blocks_per_row) {
    const int row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    extern __shared__ float shared[];
    float sum = 0.0f;
    const int tid = threadIdx.x;
    const uint8_t* row_ptr = w + static_cast<size_t>(row) * blocks_per_row * 34;
    for (int b = tid; b < blocks_per_row; b += blockDim.x) {
        const uint8_t* block = row_ptr + b * 34;
        const float d = load_f16_scale(block);
        const int8_t* qs = reinterpret_cast<const int8_t*>(block + 2);
        const int base = b * 32;
        float block_sum = 0.0f;
        for (int j = 0; j < 32; ++j) {
            const int col = base + j;
            if (col < cols) {
                block_sum += x[col] * static_cast<float>(qs[j]);
            }
        }
        sum += d * block_sum;
    }
    shared[tid] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared[tid] += shared[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        y[row] = shared[0];
    }
}

__global__ void q8_0_matmul_rows_kernel(
    const float* __restrict__ x,
    const uint8_t* __restrict__ w,
    float* __restrict__ y,
    int batch,
    int rows,
    int cols,
    int x_stride,
    int y_stride,
    int blocks_per_row) {
    const int out_row = blockIdx.x;
    const int batch_row = blockIdx.y;
    if (batch_row >= batch || out_row >= rows) return;
    extern __shared__ float shared[];
    float sum = 0.0f;
    const int tid = threadIdx.x;
    const float* x_row = x + static_cast<size_t>(batch_row) * x_stride;
    const uint8_t* w_row = w + static_cast<size_t>(out_row) * blocks_per_row * 34;
    for (int b = tid; b < blocks_per_row; b += blockDim.x) {
        const uint8_t* block = w_row + b * 34;
        const float d = load_f16_scale(block);
        const int8_t* qs = reinterpret_cast<const int8_t*>(block + 2);
        const int base = b * 32;
        float block_sum = 0.0f;
        #pragma unroll
        for (int j = 0; j < 32; ++j) {
            const int col = base + j;
            if (col < cols) block_sum += x_row[col] * static_cast<float>(qs[j]);
        }
        sum += d * block_sum;
    }
    shared[tid] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) shared[tid] += shared[tid + stride];
        __syncthreads();
    }
    if (tid == 0) y[static_cast<size_t>(batch_row) * y_stride + out_row] = shared[0];
}

__global__ void q8_0_matmul_rows_warp_kernel(
    const float* __restrict__ x,
    const uint8_t* __restrict__ w,
    float* __restrict__ y,
    int batch,
    int rows,
    int cols,
    int x_stride,
    int y_stride,
    int blocks_per_row) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    const int out_row = blockIdx.x * warps_per_block + warp;
    const int batch_row = blockIdx.y;
    if (batch_row >= batch || out_row >= rows) return;

    const float* x_row = x + static_cast<size_t>(batch_row) * x_stride;
    const uint8_t* w_row = w + static_cast<size_t>(out_row) * blocks_per_row * 34;
    float sum = 0.0f;
    for (int b = 0; b < blocks_per_row; ++b) {
        const uint8_t* block = w_row + static_cast<size_t>(b) * 34;
        const int col = b * 32 + lane;
        if (col < cols) {
            const float d = load_f16_scale(block);
            const int8_t* qs = reinterpret_cast<const int8_t*>(block + 2);
            sum += d * x_row[col] * static_cast<float>(qs[lane]);
        }
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    }
    if (lane == 0) y[static_cast<size_t>(batch_row) * y_stride + out_row] = sum;
}

__global__ void q8_0_matmul_rows_warp_batch4_kernel(
    const float* __restrict__ x,
    const uint8_t* __restrict__ w,
    float* __restrict__ y,
    int batch,
    int rows,
    int cols,
    int x_stride,
    int y_stride,
    int blocks_per_row) {
    constexpr int batch_tile = 4;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    const int out_row = blockIdx.x * warps_per_block + warp;
    const int batch_base = blockIdx.y * batch_tile;
    if (out_row >= rows || batch_base >= batch) return;

    const uint8_t* w_row = w + static_cast<size_t>(out_row) * blocks_per_row * 34;
    float sum0 = 0.0f;
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    float sum3 = 0.0f;
    for (int b = 0; b < blocks_per_row; ++b) {
        const uint8_t* block = w_row + static_cast<size_t>(b) * 34;
        const int col = b * 32 + lane;
        if (col < cols) {
            const float d = load_f16_scale(block);
            const int8_t* qs = reinterpret_cast<const int8_t*>(block + 2);
            const float qv = static_cast<float>(qs[lane]);
            const size_t base0 = static_cast<size_t>(batch_base) * x_stride + col;
            sum0 += d * x[base0] * qv;
            if (batch_base + 1 < batch) sum1 += d * x[base0 + static_cast<size_t>(x_stride)] * qv;
            if (batch_base + 2 < batch) sum2 += d * x[base0 + static_cast<size_t>(2) * x_stride] * qv;
            if (batch_base + 3 < batch) sum3 += d * x[base0 + static_cast<size_t>(3) * x_stride] * qv;
        }
    }
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum0 += __shfl_down_sync(0xffffffff, sum0, offset);
        sum1 += __shfl_down_sync(0xffffffff, sum1, offset);
        sum2 += __shfl_down_sync(0xffffffff, sum2, offset);
        sum3 += __shfl_down_sync(0xffffffff, sum3, offset);
    }
    if (lane == 0) {
        y[static_cast<size_t>(batch_base) * y_stride + out_row] = sum0;
        if (batch_base + 1 < batch) y[static_cast<size_t>(batch_base + 1) * y_stride + out_row] = sum1;
        if (batch_base + 2 < batch) y[static_cast<size_t>(batch_base + 2) * y_stride + out_row] = sum2;
        if (batch_base + 3 < batch) y[static_cast<size_t>(batch_base + 3) * y_stride + out_row] = sum3;
    }
}

}  // namespace

bool q8_0_matvec_cuda(const float* d_x, const uint8_t* d_w, float* d_y, int rows, int cols, void* stream) {
    if (d_x == nullptr || d_w == nullptr || d_y == nullptr || rows <= 0 || cols <= 0) {
        return false;
    }
    const int blocks_per_row = (cols + 31) / 32;
    const int threads = 128;
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    q8_0_matvec_kernel<<<rows, threads, threads * sizeof(float), cuda_stream>>>(d_x, d_w, d_y, rows, cols, blocks_per_row);
    return cudaGetLastError() == cudaSuccess;
}

bool q8_0_matmul_rows_strided_cuda(
    const float* d_x,
    const uint8_t* d_w,
    float* d_y,
    int batch,
    int rows,
    int cols,
    int x_stride,
    int y_stride,
    void* stream) {
    if (d_x == nullptr || d_w == nullptr || d_y == nullptr || batch <= 0 || rows <= 0 || cols <= 0 ||
        x_stride < cols || y_stride < rows) {
        return false;
    }
    const int blocks_per_row = (cols + 31) / 32;
    cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    if (cols >= 256 && (cols % 32) == 0) {
        constexpr int threads = 256;
        constexpr int warps_per_block = threads / 32;
        if (batch >= 4 && local_env_int_or_default("POCKETLLM_Q8_ROWS_BATCH4", 1) != 0) {
            constexpr int batch_tile = 4;
            q8_0_matmul_rows_warp_batch4_kernel<<<dim3((rows + warps_per_block - 1) / warps_per_block, (batch + batch_tile - 1) / batch_tile), threads, 0, cuda_stream>>>(
                d_x, d_w, d_y, batch, rows, cols, x_stride, y_stride, blocks_per_row);
        } else {
            q8_0_matmul_rows_warp_kernel<<<dim3((rows + warps_per_block - 1) / warps_per_block, batch), threads, 0, cuda_stream>>>(
                d_x, d_w, d_y, batch, rows, cols, x_stride, y_stride, blocks_per_row);
        }
    } else {
        const int threads = 128;
        q8_0_matmul_rows_kernel<<<dim3(rows, batch), threads, threads * sizeof(float), cuda_stream>>>(
            d_x, d_w, d_y, batch, rows, cols, x_stride, y_stride, blocks_per_row);
    }
    return cudaGetLastError() == cudaSuccess;
}

bool q8_0_matmul_rows_cuda(
    const float* d_x,
    const uint8_t* d_w,
    float* d_y,
    int batch,
    int rows,
    int cols,
    void* stream) {
    return q8_0_matmul_rows_strided_cuda(d_x, d_w, d_y, batch, rows, cols, cols, rows, stream);
}

}  // namespace pocket
