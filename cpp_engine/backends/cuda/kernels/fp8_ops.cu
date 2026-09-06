#include "cuda_ops.hpp"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace pocket {
namespace {

constexpr int kFp8BlockSize = 128;
constexpr int kCublasBatchThreshold = 16;

__constant__ float kFp8E4M3Lut[256];
__constant__ float kE8M0Lut[256];

float fp8_e4m3_value_host(uint8_t code) {
    const int sign = (code >> 7) & 0x1;
    const int exp = (code >> 3) & 0xf;
    const int mant = code & 0x7;
    float value;
    if (exp == 0) {
        value = std::ldexp(static_cast<float>(mant) * (1.0f / 8.0f), -6);
    } else {
        value = std::ldexp(1.0f + static_cast<float>(mant) * (1.0f / 8.0f), exp - 7);
    }
    return sign ? -value : value;
}

float fp8_e8m0_value_host(uint8_t code) {
    return std::exp2(static_cast<float>(static_cast<int>(code) - 127));
}

bool ensure_fp8_luts() {
    static int initialized_device = -1;
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) return false;
    if (initialized_device == device) return true;

    float fp8[256];
    float e8m0[256];
    for (int i = 0; i < 256; ++i) {
        fp8[i] = fp8_e4m3_value_host(static_cast<uint8_t>(i));
        e8m0[i] = fp8_e8m0_value_host(static_cast<uint8_t>(i));
    }
    if (cudaMemcpyToSymbol(kFp8E4M3Lut, fp8, sizeof(fp8)) != cudaSuccess) return false;
    if (cudaMemcpyToSymbol(kE8M0Lut, e8m0, sizeof(e8m0)) != cudaSuccess) return false;
    initialized_device = device;
    return true;
}

__device__ __forceinline__ float fp8_e4m3_value(uint8_t code) {
    const int sign = (code >> 7) & 0x1;
    const int exp = (code >> 3) & 0xf;
    const int mant = code & 0x7;
    float value;
    if (exp == 0) {
        value = ldexpf(static_cast<float>(mant) * (1.0f / 8.0f), -6);
    } else {
        value = ldexpf(1.0f + static_cast<float>(mant) * (1.0f / 8.0f), exp - 7);
    }
    return sign ? -value : value;
}

__device__ __forceinline__ float fp8_e8m0_value(uint8_t code) {
    return exp2f(static_cast<float>(static_cast<int>(code) - 127));
}

__global__ void fp8_e4m3_e8m0_matvec_kernel(
    const float* __restrict__ x,
    const uint8_t* __restrict__ weight,
    const uint8_t* __restrict__ scale,
    float* __restrict__ y,
    int rows,
    int cols,
    int scale_cols) {
    const int row = blockIdx.x;
    const int batch = blockIdx.y;
    if (row >= rows) return;
    const int row_block = row / kFp8BlockSize;
    const float* batch_x = x + static_cast<size_t>(batch) * cols;
    float* batch_y = y + static_cast<size_t>(batch) * rows;

    float sum = 0.0f;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        const int col_block = col / kFp8BlockSize;
        const uint8_t code = weight[static_cast<size_t>(row) * cols + col];
        const float s = fp8_e8m0_value(scale[static_cast<size_t>(row_block) * scale_cols + col_block]);
        sum += fp8_e4m3_value(code) * s * batch_x[col];
    }

    extern __shared__ float scratch[];
    scratch[threadIdx.x] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) batch_y[row] = scratch[0];
}

__global__ void fp8_e4m3_e8m0_matvec_strided_kernel(
    const float* __restrict__ x,
    const uint8_t* __restrict__ weight,
    const uint8_t* __restrict__ scale,
    float* __restrict__ y,
    int rows,
    int cols,
    int scale_cols,
    int x_stride,
    int y_stride) {
    const int row = blockIdx.x;
    const int batch = blockIdx.y;
    if (row >= rows) return;
    const int row_block = row / kFp8BlockSize;
    const float* batch_x = x + static_cast<size_t>(batch) * x_stride;
    float* batch_y = y + static_cast<size_t>(batch) * y_stride;

    float sum = 0.0f;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        const int col_block = col / kFp8BlockSize;
        const uint8_t code = weight[static_cast<size_t>(row) * cols + col];
        const float s = fp8_e8m0_value(scale[static_cast<size_t>(row_block) * scale_cols + col_block]);
        sum += fp8_e4m3_value(code) * s * batch_x[col];
    }

    extern __shared__ float scratch[];
    scratch[threadIdx.x] = sum;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) batch_y[row] = scratch[0];
}

__global__ void fp8_weight_to_half_kernel(
    const uint8_t* __restrict__ weight,
    const uint8_t* __restrict__ scale,
    __half* __restrict__ out,
    int rows,
    int cols,
    int scale_cols) {
    const int64_t total = static_cast<int64_t>(rows) * cols;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int row = static_cast<int>(idx / cols);
        const int col = static_cast<int>(idx - static_cast<int64_t>(row) * cols);
        const uint8_t code = weight[idx];
        const uint8_t scale_code = scale[static_cast<size_t>(row / kFp8BlockSize) * scale_cols + col / kFp8BlockSize];
        out[idx] = __float2half_rn(kFp8E4M3Lut[code] * kE8M0Lut[scale_code]);
    }
}

__global__ void float_rows_to_half_strided_kernel(
    const float* __restrict__ x,
    __half* __restrict__ out,
    int batch,
    int cols,
    int x_stride) {
    const int64_t total = static_cast<int64_t>(batch) * cols;
    for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         idx < total;
         idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
        const int row = static_cast<int>(idx / cols);
        const int col = static_cast<int>(idx - static_cast<int64_t>(row) * cols);
        out[static_cast<int64_t>(row) * x_stride + col] = __float2half_rn(x[static_cast<int64_t>(row) * x_stride + col]);
    }
}

struct Fp8GemmWorkspace {
    int device = -1;
    __half* d_x_half = nullptr;
    __half* d_w_half = nullptr;
    size_t x_cap = 0;
    size_t w_cap = 0;
    cublasHandle_t handle = nullptr;

    ~Fp8GemmWorkspace() { release(); }

    void release() {
        cudaFree(d_x_half);
        cudaFree(d_w_half);
        if (handle != nullptr) cublasDestroy(handle);
        d_x_half = nullptr;
        d_w_half = nullptr;
        x_cap = 0;
        w_cap = 0;
        handle = nullptr;
        device = -1;
    }

    bool ensure(size_t x_elems, size_t w_elems) {
        int current_device = 0;
        if (cudaGetDevice(&current_device) != cudaSuccess) return false;
        if (device != -1 && device != current_device) release();
        device = current_device;
        if (handle == nullptr) {
            if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) return false;
            (void)cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
        }
        if (x_cap < x_elems) {
            cudaFree(d_x_half);
            d_x_half = nullptr;
            x_cap = 0;
            if (cudaMalloc(&d_x_half, x_elems * sizeof(__half)) != cudaSuccess) return false;
            x_cap = x_elems;
        }
        if (w_cap < w_elems) {
            cudaFree(d_w_half);
            d_w_half = nullptr;
            w_cap = 0;
            if (cudaMalloc(&d_w_half, w_elems * sizeof(__half)) != cudaSuccess) return false;
            w_cap = w_elems;
        }
        return true;
    }
};

Fp8GemmWorkspace& fp8_gemm_workspace() {
    static Fp8GemmWorkspace workspace;
    return workspace;
}

bool fp8_e4m3_e8m0_matmul_old_cuda(
    const float* d_x,
    const uint8_t* d_weight,
    const uint8_t* d_scale,
    float* d_y,
    int batch,
    int rows,
    int cols,
    void* stream) {
    const int threads = 256;
    const int scale_cols = cols / kFp8BlockSize;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    fp8_e4m3_e8m0_matvec_kernel<<<dim3(rows, batch), threads, threads * sizeof(float), cuda_stream>>>(
        d_x, d_weight, d_scale, d_y, rows, cols, scale_cols);
    return cudaGetLastError() == cudaSuccess;
}

bool fp8_e4m3_e8m0_matmul_strided_old_cuda(
    const float* d_x,
    const uint8_t* d_weight,
    const uint8_t* d_scale,
    float* d_y,
    int batch,
    int rows,
    int cols,
    int x_stride,
    int y_stride,
    void* stream) {
    const int threads = 256;
    const int scale_cols = cols / kFp8BlockSize;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    fp8_e4m3_e8m0_matvec_strided_kernel<<<dim3(rows, batch), threads, threads * sizeof(float), cuda_stream>>>(
        d_x, d_weight, d_scale, d_y, rows, cols, scale_cols, x_stride, y_stride);
    return cudaGetLastError() == cudaSuccess;
}

bool fp8_e4m3_e8m0_matmul_cublas_cuda(
    const float* d_x,
    const uint8_t* d_weight,
    const uint8_t* d_scale,
    float* d_y,
    int batch,
    int rows,
    int cols,
    int x_stride,
    int y_stride,
    void* stream) {
    if (!ensure_fp8_luts()) return false;
    Fp8GemmWorkspace& workspace = fp8_gemm_workspace();
    if (!workspace.ensure(static_cast<size_t>(batch) * x_stride, static_cast<size_t>(rows) * cols)) return false;

    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    const int threads = 256;
    const int scale_cols = cols / kFp8BlockSize;
    const int64_t weight_elems = static_cast<int64_t>(rows) * cols;
    const int64_t x_elems = static_cast<int64_t>(batch) * cols;
    const int weight_blocks = static_cast<int>(std::min<int64_t>((weight_elems + threads - 1) / threads, 65535));
    const int x_blocks = static_cast<int>(std::min<int64_t>((x_elems + threads - 1) / threads, 65535));

    fp8_weight_to_half_kernel<<<weight_blocks, threads, 0, cuda_stream>>>(
        d_weight, d_scale, workspace.d_w_half, rows, cols, scale_cols);
    if (cudaGetLastError() != cudaSuccess) return false;
    float_rows_to_half_strided_kernel<<<x_blocks, threads, 0, cuda_stream>>>(
        d_x, workspace.d_x_half, batch, cols, x_stride);
    if (cudaGetLastError() != cudaSuccess) return false;

    if (cublasSetStream(workspace.handle, cuda_stream) != CUBLAS_STATUS_SUCCESS) return false;
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const cublasStatus_t status = cublasGemmEx(
        workspace.handle,
        CUBLAS_OP_T,
        CUBLAS_OP_N,
        rows,
        batch,
        cols,
        &alpha,
        workspace.d_w_half,
        CUDA_R_16F,
        cols,
        workspace.d_x_half,
        CUDA_R_16F,
        x_stride,
        &beta,
        d_y,
        CUDA_R_32F,
        y_stride,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    return status == CUBLAS_STATUS_SUCCESS;
}

}  // namespace

bool fp8_e4m3_e8m0_matvec_cuda(
    const float* d_x,
    const uint8_t* d_weight,
    const uint8_t* d_scale,
    float* d_y,
    int rows,
    int cols,
    void* stream) {
    if (d_x == nullptr || d_weight == nullptr || d_scale == nullptr || d_y == nullptr) return false;
    if (rows <= 0 || cols <= 0) return false;
    if ((rows % kFp8BlockSize) != 0 || (cols % kFp8BlockSize) != 0) return false;
    const int threads = 256;
    const int scale_cols = cols / kFp8BlockSize;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    fp8_e4m3_e8m0_matvec_kernel<<<dim3(rows, 1), threads, threads * sizeof(float), cuda_stream>>>(
        d_x, d_weight, d_scale, d_y, rows, cols, scale_cols);
    return cudaGetLastError() == cudaSuccess;
}

bool fp8_e4m3_e8m0_matmul_cuda(
    const float* d_x,
    const uint8_t* d_weight,
    const uint8_t* d_scale,
    float* d_y,
    int batch,
    int rows,
    int cols,
    void* stream) {
    if (d_x == nullptr || d_weight == nullptr || d_scale == nullptr || d_y == nullptr) return false;
    if (batch <= 0 || rows <= 0 || cols <= 0) return false;
    if ((rows % kFp8BlockSize) != 0 || (cols % kFp8BlockSize) != 0) return false;
    if (batch < kCublasBatchThreshold) {
        return fp8_e4m3_e8m0_matmul_old_cuda(d_x, d_weight, d_scale, d_y, batch, rows, cols, stream);
    }
    return fp8_e4m3_e8m0_matmul_cublas_cuda(d_x, d_weight, d_scale, d_y, batch, rows, cols, cols, rows, stream);
}

bool fp8_e4m3_e8m0_matmul_strided_cuda(
    const float* d_x,
    const uint8_t* d_weight,
    const uint8_t* d_scale,
    float* d_y,
    int batch,
    int rows,
    int cols,
    int x_stride,
    int y_stride,
    void* stream) {
    if (d_x == nullptr || d_weight == nullptr || d_scale == nullptr || d_y == nullptr) return false;
    if (batch <= 0 || rows <= 0 || cols <= 0 || x_stride < cols || y_stride < rows) return false;
    if ((rows % kFp8BlockSize) != 0 || (cols % kFp8BlockSize) != 0) return false;
    if (batch < kCublasBatchThreshold) {
        return fp8_e4m3_e8m0_matmul_strided_old_cuda(d_x, d_weight, d_scale, d_y, batch, rows, cols, x_stride, y_stride, stream);
    }
    return fp8_e4m3_e8m0_matmul_cublas_cuda(d_x, d_weight, d_scale, d_y, batch, rows, cols, x_stride, y_stride, stream);
}

}  // namespace pocket
