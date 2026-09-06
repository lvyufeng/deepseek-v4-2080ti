#include "qwen_cuda_ops.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace pocket {
namespace {

constexpr int kBlock = 128;

__device__ __forceinline__ float fp8_e4m3_to_float(uint8_t code) {
    const uint32_t sign = static_cast<uint32_t>(code & 0x80u) << 24;
    const uint32_t exponent = (code >> 3) & 0xfu;
    const uint32_t mantissa = code & 0x7u;
    uint32_t bits;
    if (exponent != 0) {
        bits = sign | ((exponent + 120u) << 23) | (mantissa << 20);
    } else if (mantissa >= 4) {
        bits = sign | (120u << 23) | ((mantissa - 4u) << 21);
    } else if (mantissa >= 2) {
        bits = sign | (119u << 23) | ((mantissa - 2u) << 22);
    } else {
        bits = sign | (mantissa == 0 ? 0u : (118u << 23));
    }
    return __uint_as_float(bits);
}

__device__ __forceinline__ float bf16_bits_to_float(uint16_t bits) {
    return __uint_as_float(static_cast<uint32_t>(bits) << 16);
}

__device__ __forceinline__ float fp16_bits_to_float(uint16_t bits) {
    return __half2float(__ushort_as_half(bits));
}

__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + expf(-x));
}

template <bool kFp16Scale>
__device__ __forceinline__ float scale_bits_to_float(uint16_t bits) {
    return kFp16Scale ? fp16_bits_to_float(bits) : bf16_bits_to_float(bits);
}

template <bool kFp16Scale>
__device__ __forceinline__ float block_scale(const uint16_t* scale,
                                              int scale_stride,
                                              int row,
                                              int col) {
    return scale_bits_to_float<kFp16Scale>(
        scale[static_cast<size_t>(row / kBlock) * scale_stride + col / kBlock]);
}

__device__ __forceinline__ float reduce_sum(float value, float* scratch) {
    scratch[threadIdx.x] = value;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) scratch[threadIdx.x] += scratch[threadIdx.x + stride];
        __syncthreads();
    }
    return scratch[0];
}

// Prefill path: one warp per output row, but each warp carries kTileBatch token
// accumulators so a decoded FP8 weight is reused across tokens instead of being
// re-read once per (row, token) pair. This turns the projection from
// bandwidth-bound-per-token into a single weight sweep per row tile.
template <bool kFp16Scale, int kRowsPerBlock, int kTileBatch>
__global__ void fp8_matmul_tiled_kernel(
    const float* __restrict__ x,
    const uint8_t* __restrict__ weight,
    const uint16_t* __restrict__ scale,
    float* __restrict__ y,
    int batch,
    int rows,
    int cols,
    int x_stride,
    int y_stride,
    int weight_stride,
    int scale_stride) {
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int row = static_cast<int>(blockIdx.x) * kRowsPerBlock + warp;
    const int batch_base = static_cast<int>(blockIdx.y) * kTileBatch;
    // The LUT fill is a block-wide barrier, so no thread may return before it.
    __shared__ float lut[256];
    for (int i = static_cast<int>(threadIdx.x); i < 256; i += static_cast<int>(blockDim.x)) {
        lut[i] = fp8_e4m3_to_float(static_cast<uint8_t>(i));
    }
    __syncthreads();
    if (row >= rows || batch_base >= batch) return;

    const int active = min(kTileBatch, batch - batch_base);
    const uint8_t* w_row = weight + static_cast<size_t>(row) * weight_stride;
    const uint16_t* scale_row = scale + static_cast<size_t>(row / kBlock) * scale_stride;
    float acc[kTileBatch];
    for (int b = 0; b < kTileBatch; ++b) acc[b] = 0.0f;

    const int vec_cols = cols & ~3;
    for (int col = lane * 4; col < vec_cols; col += 128) {
        const uchar4 codes = *reinterpret_cast<const uchar4*>(w_row + col);
        const float s = scale_bits_to_float<kFp16Scale>(scale_row[col / kBlock]);
        const float w0 = lut[codes.x] * s;
        const float w1 = lut[codes.y] * s;
        const float w2 = lut[codes.z] * s;
        const float w3 = lut[codes.w] * s;
        for (int b = 0; b < active; ++b) {
            const float* x_row = x + static_cast<size_t>(batch_base + b) * x_stride;
            acc[b] += x_row[col + 0] * w0 + x_row[col + 1] * w1 +
                      x_row[col + 2] * w2 + x_row[col + 3] * w3;
        }
    }
    for (int col = vec_cols + lane; col < cols; col += 32) {
        const float w = lut[w_row[col]] *
                        scale_bits_to_float<kFp16Scale>(scale_row[col / kBlock]);
        for (int b = 0; b < active; ++b) {
            acc[b] += x[static_cast<size_t>(batch_base + b) * x_stride + col] * w;
        }
    }
    for (int b = 0; b < active; ++b) {
        float sum = acc[b];
        for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, off);
        if (lane == 0) y[static_cast<size_t>(batch_base + b) * y_stride + row] = sum;
    }
}

// Large-prefill path for SM75: a block computes a 64-token by 64-output tile
// with 4x4 outputs per thread. FP8 weights are decoded only for the current
// K tile, then reused by all 64 tokens. Inputs, products, accumulation and
// output remain FP32; the temporary decoded weight tile is block-local shared
// memory, so the full matrix is never expanded.
//
// Accuracy: the legacy prefill kernel reduces its partial sums in a 32-wide
// tree. Accumulating all `cols` products into one register would lengthen the
// dependent chain enough to fail the existing precision gate, so each K tile
// is summed locally and only the tile sums feed the long-running accumulator.
// That keeps the worst-case chain at max(kColTile, cols / kColTile) instead of
// cols, with no extra arithmetic.
template <bool kFp16Scale, int kColTile>
__global__ void fp8_matmul_simt_kernel(
    const float* __restrict__ x,
    const uint8_t* __restrict__ weight,
    const uint16_t* __restrict__ scale,
    float* __restrict__ y,
    int batch,
    int rows,
    int cols,
    int x_stride,
    int y_stride,
    int weight_stride,
    int scale_stride) {
    constexpr int kBatchTile = 64;
    constexpr int kRowTile = 64;
    constexpr int kThreads = 256;
    constexpr int kPerThread = 4;
    constexpr int kStride = kColTile + 1;
    __shared__ float x_tile[kBatchTile][kStride];
    __shared__ float w_tile[kRowTile][kStride];
    __shared__ float lut[256];

    const int tid = static_cast<int>(threadIdx.x);
    for (int i = tid; i < 256; i += kThreads) {
        lut[i] = fp8_e4m3_to_float(static_cast<uint8_t>(i));
    }
    __syncthreads();

    const int batch_base = static_cast<int>(blockIdx.y) * kBatchTile;
    const int row_base = static_cast<int>(blockIdx.x) * kRowTile;
    const int thread_batch = tid >> 4;
    const int thread_row = tid & 15;
    float acc[kPerThread][kPerThread] = {};

    for (int col_base = 0; col_base < cols; col_base += kColTile) {
        for (int index = tid; index < kBatchTile * kColTile; index += kThreads) {
            const int local_batch = index / kColTile;
            const int local_col = index - local_batch * kColTile;
            const int global_batch = batch_base + local_batch;
            const int global_col = col_base + local_col;
            x_tile[local_batch][local_col] =
                global_batch < batch && global_col < cols
                    ? x[static_cast<size_t>(global_batch) * x_stride + global_col]
                    : 0.0f;
        }
        for (int index = tid; index < kRowTile * kColTile; index += kThreads) {
            const int local_row = index / kColTile;
            const int local_col = index - local_row * kColTile;
            const int global_row = row_base + local_row;
            const int global_col = col_base + local_col;
            float value = 0.0f;
            if (global_row < rows && global_col < cols) {
                const uint8_t code =
                    weight[static_cast<size_t>(global_row) * weight_stride + global_col];
                const uint16_t bits =
                    scale[static_cast<size_t>(global_row / kBlock) * scale_stride +
                          global_col / kBlock];
                value = lut[code] * scale_bits_to_float<kFp16Scale>(bits);
            }
            w_tile[local_row][local_col] = value;
        }
        __syncthreads();

        float tile_acc[kPerThread][kPerThread] = {};
#pragma unroll 8
        for (int k = 0; k < kColTile; ++k) {
            float a[kPerThread];
            float b[kPerThread];
#pragma unroll
            for (int i = 0; i < kPerThread; ++i) {
                a[i] = x_tile[thread_batch + i * 16][k];
                b[i] = w_tile[thread_row + i * 16][k];
            }
#pragma unroll
            for (int i = 0; i < kPerThread; ++i) {
#pragma unroll
                for (int j = 0; j < kPerThread; ++j) {
                    tile_acc[i][j] += a[i] * b[j];
                }
            }
        }
#pragma unroll
        for (int i = 0; i < kPerThread; ++i) {
#pragma unroll
            for (int j = 0; j < kPerThread; ++j) acc[i][j] += tile_acc[i][j];
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < kPerThread; ++i) {
        const int global_batch = batch_base + thread_batch + i * 16;
        if (global_batch >= batch) continue;
#pragma unroll
        for (int j = 0; j < kPerThread; ++j) {
            const int global_row = row_base + thread_row + j * 16;
            if (global_row < rows) {
                y[static_cast<size_t>(global_batch) * y_stride + global_row] =
                    acc[i][j];
            }
        }
    }
}

// Long-prompt prefill path for SM75. Same online FP8 decode, but a block owns
// a 128-token by 128-output tile and each thread holds 8x8 outputs, so the
// FMA-to-shared-load ratio is 4:1 instead of the 2:1 of the 64x64 tile above.
// On Turing 2:1 exactly saturates shared-memory bandwidth against the FP32
// pipes, which caps that kernel at half of peak.
//
// Tiles are staged transposed ([k][m] and [k][n]) so the inner loop reads
// float4 lanes, and each thread's two float4 groups are 64 apart to keep the
// shared reads bank-conflict free. Per-K-tile local sums feed the long
// accumulator, preserving the shallow reduction depth the precision gate needs.
template <bool kFp16Scale>
__global__ void fp8_matmul_simt_wide_kernel(
    const float* __restrict__ x,
    const uint8_t* __restrict__ weight,
    const uint16_t* __restrict__ scale,
    float* __restrict__ y,
    int batch,
    int rows,
    int cols,
    int x_stride,
    int y_stride,
    int weight_stride,
    int scale_stride) {
    constexpr int kM = 128;
    constexpr int kN = 128;
    constexpr int kK = 16;
    constexpr int kThreads = 256;
    // `m0`/`n0` are float4 aligned. A 128-float row stride preserves that
    // alignment for every K row; staging is transposed so it needs no padding
    // to avoid bank conflicts.
    constexpr int kPad = kM;
    __shared__ float as[kK][kPad];
    __shared__ float bs[kK][kPad];
    __shared__ float lut[256];

    const int tid = static_cast<int>(threadIdx.x);
    for (int i = tid; i < 256; i += kThreads) {
        lut[i] = fp8_e4m3_to_float(static_cast<uint8_t>(i));
    }
    __syncthreads();

    const int batch_base = static_cast<int>(blockIdx.y) * kM;
    const int row_base = static_cast<int>(blockIdx.x) * kN;
    const int tx = tid & 15;
    const int ty = tid >> 4;
    const int m0 = ty * 4;
    const int m1 = 64 + ty * 4;
    const int n0 = tx * 4;
    const int n1 = 64 + tx * 4;
    // Staging coordinates: 4 threads cover one row's 16-wide K slice.
    const int load_lane = (tid & 3) * 4;
    const int load_line = tid >> 2;
    float acc[8][8] = {};

    for (int col_base = 0; col_base < cols; col_base += kK) {
#pragma unroll
        for (int it = 0; it < 2; ++it) {
            const int local_batch = load_line + it * 64;
            const int global_batch = batch_base + local_batch;
            const int global_col = col_base + load_lane;
            float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (global_batch < batch && global_col + 3 < cols) {
                v = *reinterpret_cast<const float4*>(
                    x + static_cast<size_t>(global_batch) * x_stride + global_col);
            } else if (global_batch < batch) {
                const float* src = x + static_cast<size_t>(global_batch) * x_stride;
                float tmp[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                for (int t = 0; t < 4; ++t) {
                    if (global_col + t < cols) tmp[t] = src[global_col + t];
                }
                v = make_float4(tmp[0], tmp[1], tmp[2], tmp[3]);
            }
            as[load_lane + 0][local_batch] = v.x;
            as[load_lane + 1][local_batch] = v.y;
            as[load_lane + 2][local_batch] = v.z;
            as[load_lane + 3][local_batch] = v.w;
        }
#pragma unroll
        for (int it = 0; it < 2; ++it) {
            const int local_row = load_line + it * 64;
            const int global_row = row_base + local_row;
            const int global_col = col_base + load_lane;
            float w[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            if (global_row < rows) {
                const uint8_t* src =
                    weight + static_cast<size_t>(global_row) * weight_stride;
                const float s = scale_bits_to_float<kFp16Scale>(
                    scale[static_cast<size_t>(global_row / kBlock) * scale_stride +
                          global_col / kBlock]);
                if (global_col + 3 < cols) {
                    const uchar4 codes = *reinterpret_cast<const uchar4*>(src + global_col);
                    w[0] = lut[codes.x] * s;
                    w[1] = lut[codes.y] * s;
                    w[2] = lut[codes.z] * s;
                    w[3] = lut[codes.w] * s;
                } else {
                    for (int t = 0; t < 4; ++t) {
                        if (global_col + t < cols) w[t] = lut[src[global_col + t]] * s;
                    }
                }
            }
            bs[load_lane + 0][local_row] = w[0];
            bs[load_lane + 1][local_row] = w[1];
            bs[load_lane + 2][local_row] = w[2];
            bs[load_lane + 3][local_row] = w[3];
        }
        __syncthreads();

        float tile[8][8] = {};
#pragma unroll
        for (int k = 0; k < kK; ++k) {
            float a[8];
            float b[8];
            *reinterpret_cast<float4*>(&a[0]) = *reinterpret_cast<const float4*>(&as[k][m0]);
            *reinterpret_cast<float4*>(&a[4]) = *reinterpret_cast<const float4*>(&as[k][m1]);
            *reinterpret_cast<float4*>(&b[0]) = *reinterpret_cast<const float4*>(&bs[k][n0]);
            *reinterpret_cast<float4*>(&b[4]) = *reinterpret_cast<const float4*>(&bs[k][n1]);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
#pragma unroll
                for (int j = 0; j < 8; ++j) tile[i][j] += a[i] * b[j];
            }
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
#pragma unroll
            for (int j = 0; j < 8; ++j) acc[i][j] += tile[i][j];
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int local_batch = i < 4 ? m0 + i : m1 + (i - 4);
        const int global_batch = batch_base + local_batch;
        if (global_batch >= batch) continue;
        float* dst = y + static_cast<size_t>(global_batch) * y_stride;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int global_row = row_base + (j < 4 ? n0 + j : n1 + (j - 4));
            if (global_row < rows) dst[global_row] = acc[i][j];
        }
    }
}

// Lower-register alternative to the 128x128 tile above. Halving N reduces each
// thread from 64 to 32 outputs, which allows two blocks to reside on an SM75.
// The K=16 local accumulation is unchanged, so this variant keeps the same
// reduction depth and precision behavior while trading some shared-load reuse
// for occupancy.
template <bool kFp16Scale>
__global__ void fp8_matmul_simt_wide_n64_kernel(
    const float* __restrict__ x,
    const uint8_t* __restrict__ weight,
    const uint16_t* __restrict__ scale,
    float* __restrict__ y,
    int batch,
    int rows,
    int cols,
    int x_stride,
    int y_stride,
    int weight_stride,
    int scale_stride) {
    constexpr int kM = 128;
    constexpr int kN = 64;
    constexpr int kK = 16;
    constexpr int kThreads = 256;
    __shared__ float as[kK][kM];
    __shared__ float bs[kK][kN];
    __shared__ float lut[256];

    const int tid = static_cast<int>(threadIdx.x);
    for (int i = tid; i < 256; i += kThreads) {
        lut[i] = fp8_e4m3_to_float(static_cast<uint8_t>(i));
    }
    __syncthreads();

    const int batch_base = static_cast<int>(blockIdx.y) * kM;
    const int row_base = static_cast<int>(blockIdx.x) * kN;
    const int tx = tid & 15;
    const int ty = tid >> 4;
    const int m0 = ty * 4;
    const int m1 = 64 + ty * 4;
    const int n0 = tx * 4;
    const int load_lane = (tid & 3) * 4;
    const int load_line = tid >> 2;
    float acc[8][4] = {};

    for (int col_base = 0; col_base < cols; col_base += kK) {
#pragma unroll
        for (int it = 0; it < 2; ++it) {
            const int local_batch = load_line + it * 64;
            const int global_batch = batch_base + local_batch;
            const int global_col = col_base + load_lane;
            float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (global_batch < batch && global_col + 3 < cols) {
                v = *reinterpret_cast<const float4*>(
                    x + static_cast<size_t>(global_batch) * x_stride + global_col);
            } else if (global_batch < batch) {
                const float* src = x + static_cast<size_t>(global_batch) * x_stride;
                float tmp[4] = {0.0f, 0.0f, 0.0f, 0.0f};
                for (int t = 0; t < 4; ++t) {
                    if (global_col + t < cols) tmp[t] = src[global_col + t];
                }
                v = make_float4(tmp[0], tmp[1], tmp[2], tmp[3]);
            }
            as[load_lane + 0][local_batch] = v.x;
            as[load_lane + 1][local_batch] = v.y;
            as[load_lane + 2][local_batch] = v.z;
            as[load_lane + 3][local_batch] = v.w;
        }

        const int local_row = load_line;
        const int global_row = row_base + local_row;
        const int global_col = col_base + load_lane;
        float w[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        if (global_row < rows) {
            const uint8_t* src =
                weight + static_cast<size_t>(global_row) * weight_stride;
            const float s = scale_bits_to_float<kFp16Scale>(
                scale[static_cast<size_t>(global_row / kBlock) * scale_stride +
                      global_col / kBlock]);
            if (global_col + 3 < cols) {
                const uchar4 codes = *reinterpret_cast<const uchar4*>(src + global_col);
                w[0] = lut[codes.x] * s;
                w[1] = lut[codes.y] * s;
                w[2] = lut[codes.z] * s;
                w[3] = lut[codes.w] * s;
            } else {
                for (int t = 0; t < 4; ++t) {
                    if (global_col + t < cols) w[t] = lut[src[global_col + t]] * s;
                }
            }
        }
        bs[load_lane + 0][local_row] = w[0];
        bs[load_lane + 1][local_row] = w[1];
        bs[load_lane + 2][local_row] = w[2];
        bs[load_lane + 3][local_row] = w[3];
        __syncthreads();

        float tile[8][4] = {};
#pragma unroll
        for (int k = 0; k < kK; ++k) {
            float a[8];
            float b[4];
            *reinterpret_cast<float4*>(&a[0]) = *reinterpret_cast<const float4*>(&as[k][m0]);
            *reinterpret_cast<float4*>(&a[4]) = *reinterpret_cast<const float4*>(&as[k][m1]);
            *reinterpret_cast<float4*>(&b[0]) = *reinterpret_cast<const float4*>(&bs[k][n0]);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
#pragma unroll
                for (int j = 0; j < 4; ++j) tile[i][j] += a[i] * b[j];
            }
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
#pragma unroll
            for (int j = 0; j < 4; ++j) acc[i][j] += tile[i][j];
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int local_batch = i < 4 ? m0 + i : m1 + (i - 4);
        const int global_batch = batch_base + local_batch;
        if (global_batch >= batch) continue;
        float* dst = y + static_cast<size_t>(global_batch) * y_stride;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int global_row = row_base + n0 + j;
            if (global_row < rows) dst[global_row] = acc[i][j];
        }
    }
}

template <bool kFp16Scale, int kBatchTile = 64, int kRowTile = 32,
          int kColTile = 32>
__global__ void fp8_matmul_simt_legacy_kernel(
    const float* __restrict__ x,
    const uint8_t* __restrict__ weight,
    const uint16_t* __restrict__ scale,
    float* __restrict__ y,
    int batch,
    int rows,
    int cols,
    int x_stride,
    int y_stride,
    int weight_stride,
    int scale_stride) {
    static_assert(kBatchTile == 64 && kRowTile == 32 && kColTile == 32,
                  "thread mapping assumes a 64x32x32 tile");
    constexpr int kSharedStride = kColTile + 1;
    __shared__ float x_tile[kBatchTile][kSharedStride];
    __shared__ float w_tile[kRowTile][kSharedStride];
    __shared__ float lut[256];

    const int tid = static_cast<int>(threadIdx.x);
    for (int i = tid; i < 256; i += static_cast<int>(blockDim.x)) {
        lut[i] = fp8_e4m3_to_float(static_cast<uint8_t>(i));
    }
    __syncthreads();

    const int batch_base = static_cast<int>(blockIdx.y) * kBatchTile;
    const int row_base = static_cast<int>(blockIdx.x) * kRowTile;
    const int thread_batch = tid >> 4;
    const int thread_row = (tid & 15) * 2;
    float acc[4][2] = {};
    // The legacy prefill path reduces 32 FP32 partial sums in a tree. This
    // blocked tile changes the association, so retain the same precision gate
    // with a local compensated sum rather than weakening the test tolerance.
    float compensation[4][2] = {};

    for (int col_base = 0; col_base < cols; col_base += kColTile) {
        for (int index = tid; index < kBatchTile * kColTile;
             index += static_cast<int>(blockDim.x)) {
            const int local_batch = index / kColTile;
            const int local_col = index - local_batch * kColTile;
            const int global_batch = batch_base + local_batch;
            const int global_col = col_base + local_col;
            x_tile[local_batch][local_col] =
                global_batch < batch && global_col < cols
                    ? x[static_cast<size_t>(global_batch) * x_stride + global_col]
                    : 0.0f;
        }
        for (int index = tid; index < kRowTile * kColTile;
             index += static_cast<int>(blockDim.x)) {
            const int local_row = index / kColTile;
            const int local_col = index - local_row * kColTile;
            const int global_row = row_base + local_row;
            const int global_col = col_base + local_col;
            float value = 0.0f;
            if (global_row < rows && global_col < cols) {
                const uint8_t code =
                    weight[static_cast<size_t>(global_row) * weight_stride + global_col];
                const uint16_t bits =
                    scale[static_cast<size_t>(global_row / kBlock) * scale_stride +
                          global_col / kBlock];
                value = lut[code] * scale_bits_to_float<kFp16Scale>(bits);
            }
            w_tile[local_row][local_col] = value;
        }
        __syncthreads();

#pragma unroll
        for (int k = 0; k < kColTile; ++k) {
            float a[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                a[i] = x_tile[thread_batch + i * 16][k];
            }
            const float b0 = w_tile[thread_row][k];
            const float b1 = w_tile[thread_row + 1][k];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float p0 = a[i] * b0;
                const float s0 = acc[i][0] + p0;
                compensation[i][0] += fabsf(acc[i][0]) >= fabsf(p0)
                                          ? (acc[i][0] - s0) + p0
                                          : (p0 - s0) + acc[i][0];
                acc[i][0] = s0;
                const float p1 = a[i] * b1;
                const float s1 = acc[i][1] + p1;
                compensation[i][1] += fabsf(acc[i][1]) >= fabsf(p1)
                                          ? (acc[i][1] - s1) + p1
                                          : (p1 - s1) + acc[i][1];
                acc[i][1] = s1;
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int global_batch = batch_base + thread_batch + i * 16;
        if (global_batch >= batch) continue;
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int global_row = row_base + thread_row + j;
            if (global_row < rows) {
                y[static_cast<size_t>(global_batch) * y_stride + global_row] =
                    acc[i][j] + compensation[i][j];
            }
        }
    }
}

template <bool kFp16Scale>
__global__ void fp8_matmul_rows_kernel(
    const float* __restrict__ x,
    const uint8_t* __restrict__ weight,
    const uint16_t* __restrict__ scale,
    float* __restrict__ y,
    int batch,
    int rows,
    int cols,
    int x_stride,
    int y_stride,
    int weight_stride,
    int scale_stride) {
    const int row = static_cast<int>(blockIdx.x);
    const int sample = static_cast<int>(blockIdx.y);
    if (row >= rows || sample >= batch) return;

    const float* x_row = x + static_cast<size_t>(sample) * x_stride;
    const uint8_t* w_row = weight + static_cast<size_t>(row) * weight_stride;
    float sum = 0.0f;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        const float w = fp8_e4m3_to_float(w_row[col]) * block_scale<kFp16Scale>(scale, scale_stride, row, col);
        sum += x_row[col] * w;
    }

    extern __shared__ float scratch[];
    const float total = reduce_sum(sum, scratch);
    if (threadIdx.x == 0) y[static_cast<size_t>(sample) * y_stride + row] = total;
}

template <bool kFp16Scale>
__global__ void fp8_matvec_kernel(
    const float* __restrict__ x,
    const uint8_t* __restrict__ weight,
    const uint16_t* __restrict__ scale,
    float* __restrict__ y,
    int rows,
    int cols,
    int weight_stride,
    int scale_stride) {
    const int row = static_cast<int>(blockIdx.x);
    if (row >= rows) return;

    const uint8_t* w_row = weight + static_cast<size_t>(row) * weight_stride;
    float sum = 0.0f;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        const float w = fp8_e4m3_to_float(w_row[col]) * block_scale<kFp16Scale>(scale, scale_stride, row, col);
        sum += x[col] * w;
    }

    extern __shared__ float scratch[];
    const float total = reduce_sum(sum, scratch);
    if (threadIdx.x == 0) y[row] = total;
}

// Decode-latency path: one warp per output row, so there is no block-wide
// __syncthreads in the reduction and several rows retire per block. FP8 codes
// are pulled 4 bytes at a time; the 128-wide scale block means the scale lookup
// is loop-invariant across each aligned run of 128 columns.
template <bool kFp16Scale, int kRowsPerBlock>
__global__ void fp8_matvec_warp_kernel(
    const float* __restrict__ x,
    const uint8_t* __restrict__ weight,
    const uint16_t* __restrict__ scale,
    float* __restrict__ y,
    int rows,
    int cols,
    int weight_stride,
    int scale_stride) {
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int row = static_cast<int>(blockIdx.x) * kRowsPerBlock + warp;

    // Decode E4M3 through a 256-entry shared-memory LUT (1 KB) instead of the
    // per-byte ldexpf/branch sequence. x itself is left in global memory: every
    // warp sweeps the same vector so L1/L2 already serves that reuse, and
    // staging x measured slower because its 20 KB footprint caps occupancy.
    // An arithmetic in-register decode was also measured and came out ~10%
    // slower here: these kernels are bandwidth-bound, so trading the LUT loads
    // for extra ALU work in the inner loop does not help.
    __shared__ float lut[256];
    for (int i = static_cast<int>(threadIdx.x); i < 256; i += static_cast<int>(blockDim.x)) {
        lut[i] = fp8_e4m3_to_float(static_cast<uint8_t>(i));
    }
    __syncthreads();
    if (row >= rows) return;
    const float* x_shared = x;

    const uint8_t* w_row = weight + static_cast<size_t>(row) * weight_stride;
    const uint16_t* scale_row = scale + static_cast<size_t>(row / kBlock) * scale_stride;
    const int vec_cols = cols & ~3;
    float sum = 0.0f;
    for (int col = lane * 4; col < vec_cols; col += 128) {
        const uchar4 codes = *reinterpret_cast<const uchar4*>(w_row + col);
        const float s = scale_bits_to_float<kFp16Scale>(scale_row[col / kBlock]);
        const float4 xv = *reinterpret_cast<const float4*>(x_shared + col);
        sum += (xv.x * lut[codes.x] + xv.y * lut[codes.y] +
                xv.z * lut[codes.z] + xv.w * lut[codes.w]) * s;
    }
    for (int col = vec_cols + lane; col < cols; col += 32) {
        sum += x_shared[col] * lut[w_row[col]] *
               scale_bits_to_float<kFp16Scale>(scale_row[col / kBlock]);
    }
    for (int off = 16; off > 0; off >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, off);
    if (lane == 0) y[row] = sum;
}

// Decode matvec, x-traffic-reduced form. Each warp owns kRowsPerWarp output
// rows and loads every x element once for all of them, so the vector is pulled
// from memory rows/kRowsPerWarp times instead of once per row. Column order and
// the per-row accumulation order are identical to fp8_matvec_warp_kernel, so
// results are unchanged; only the number of x loads differs.
template <bool kFp16Scale, int kWarpsPerBlock, int kRowsPerWarp>
__global__ void fp8_matvec_multirow_kernel(
    const float* __restrict__ x,
    const uint8_t* __restrict__ weight,
    const uint16_t* __restrict__ scale,
    float* __restrict__ y,
    int rows,
    int cols,
    int weight_stride,
    int scale_stride) {
    // Shared LUT, as in fp8_matvec_warp_kernel; see the note there on why the
    // arithmetic decode is not used.
    __shared__ float lut[256];
    for (int i = static_cast<int>(threadIdx.x); i < 256; i += static_cast<int>(blockDim.x)) {
        lut[i] = fp8_e4m3_to_float(static_cast<uint8_t>(i));
    }
    __syncthreads();

    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int row_base = (static_cast<int>(blockIdx.x) * kWarpsPerBlock + warp) * kRowsPerWarp;
    if (row_base >= rows) return;
    const int active = min(kRowsPerWarp, rows - row_base);

    const uint8_t* w_rows[kRowsPerWarp];
    const uint16_t* scale_rows[kRowsPerWarp];
#pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) {
        const int row = min(row_base + r, rows - 1);
        w_rows[r] = weight + static_cast<size_t>(row) * weight_stride;
        scale_rows[r] = scale + static_cast<size_t>(row / kBlock) * scale_stride;
    }

    float sum[kRowsPerWarp] = {};
    const int vec_cols = cols & ~3;
    for (int col = lane * 4; col < vec_cols; col += 128) {
        const float4 xv = *reinterpret_cast<const float4*>(x + col);
        const int block_col = col / kBlock;
#pragma unroll
        for (int r = 0; r < kRowsPerWarp; ++r) {
            if (r >= active) break;
            const uchar4 codes = *reinterpret_cast<const uchar4*>(w_rows[r] + col);
            const float s = scale_bits_to_float<kFp16Scale>(scale_rows[r][block_col]);
            sum[r] += (xv.x * lut[codes.x] + xv.y * lut[codes.y] +
                       xv.z * lut[codes.z] + xv.w * lut[codes.w]) * s;
        }
    }
    for (int col = vec_cols + lane; col < cols; col += 32) {
        const float xv = x[col];
#pragma unroll
        for (int r = 0; r < kRowsPerWarp; ++r) {
            if (r >= active) break;
            sum[r] += xv * lut[w_rows[r][col]] *
                      scale_bits_to_float<kFp16Scale>(scale_rows[r][col / kBlock]);
        }
    }
#pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) {
        if (r >= active) break;
        float value = sum[r];
        for (int off = 16; off > 0; off >>= 1) value += __shfl_xor_sync(0xffffffffu, value, off);
        if (lane == 0) y[row_base + r] = value;
    }
}

// Each warp evaluates the same output rows from the MLP gate and up matrices.
// This halves x traffic versus two independent matvec launches and avoids both
// projection outputs, while preserving each dot product's accumulation order.
template <int kWarpsPerBlock, int kRowsPerWarp>
__global__ void fp8_swiglu_matvec_kernel(
    const float* __restrict__ x,
    const uint8_t* __restrict__ gate_weight,
    const uint16_t* __restrict__ gate_scale,
    const uint8_t* __restrict__ up_weight,
    const uint16_t* __restrict__ up_scale,
    float* __restrict__ y,
    int rows,
    int cols,
    int weight_stride,
    int scale_stride) {
    __shared__ float lut[256];
    for (int i = static_cast<int>(threadIdx.x); i < 256; i += static_cast<int>(blockDim.x)) {
        lut[i] = fp8_e4m3_to_float(static_cast<uint8_t>(i));
    }
    __syncthreads();

    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int row_base = (static_cast<int>(blockIdx.x) * kWarpsPerBlock + warp) * kRowsPerWarp;
    if (row_base >= rows) return;
    const int active = min(kRowsPerWarp, rows - row_base);

    const uint8_t* gate_rows[kRowsPerWarp];
    const uint8_t* up_rows[kRowsPerWarp];
    const uint16_t* gate_scale_rows[kRowsPerWarp];
    const uint16_t* up_scale_rows[kRowsPerWarp];
#pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) {
        const int row = min(row_base + r, rows - 1);
        gate_rows[r] = gate_weight + static_cast<size_t>(row) * weight_stride;
        up_rows[r] = up_weight + static_cast<size_t>(row) * weight_stride;
        gate_scale_rows[r] = gate_scale + static_cast<size_t>(row / kBlock) * scale_stride;
        up_scale_rows[r] = up_scale + static_cast<size_t>(row / kBlock) * scale_stride;
    }

    float gate_sum[kRowsPerWarp] = {};
    float up_sum[kRowsPerWarp] = {};
    const int vec_cols = cols & ~3;
    for (int col = lane * 4; col < vec_cols; col += 128) {
        const float4 xv = *reinterpret_cast<const float4*>(x + col);
        const int block_col = col / kBlock;
#pragma unroll
        for (int r = 0; r < kRowsPerWarp; ++r) {
            if (r >= active) break;
            const uchar4 gate_codes = *reinterpret_cast<const uchar4*>(gate_rows[r] + col);
            const uchar4 up_codes = *reinterpret_cast<const uchar4*>(up_rows[r] + col);
            const float gate_scale_value = fp16_bits_to_float(gate_scale_rows[r][block_col]);
            const float up_scale_value = fp16_bits_to_float(up_scale_rows[r][block_col]);
            gate_sum[r] += (xv.x * lut[gate_codes.x] + xv.y * lut[gate_codes.y] +
                            xv.z * lut[gate_codes.z] + xv.w * lut[gate_codes.w]) * gate_scale_value;
            up_sum[r] += (xv.x * lut[up_codes.x] + xv.y * lut[up_codes.y] +
                          xv.z * lut[up_codes.z] + xv.w * lut[up_codes.w]) * up_scale_value;
        }
    }
    for (int col = vec_cols + lane; col < cols; col += 32) {
        const float xv = x[col];
#pragma unroll
        for (int r = 0; r < kRowsPerWarp; ++r) {
            if (r >= active) break;
            gate_sum[r] += xv * lut[gate_rows[r][col]] *
                           fp16_bits_to_float(gate_scale_rows[r][col / kBlock]);
            up_sum[r] += xv * lut[up_rows[r][col]] *
                         fp16_bits_to_float(up_scale_rows[r][col / kBlock]);
        }
    }
#pragma unroll
    for (int r = 0; r < kRowsPerWarp; ++r) {
        if (r >= active) break;
        float gate_value = gate_sum[r];
        float up_value = up_sum[r];
        for (int off = 16; off > 0; off >>= 1) {
            gate_value += __shfl_xor_sync(0xffffffffu, gate_value, off);
            up_value += __shfl_xor_sync(0xffffffffu, up_value, off);
        }
        if (lane == 0) {
            const float gate_silu = gate_value * (1.0f / (1.0f + expf(-gate_value)));
            y[row_base + r] = gate_silu * up_value;
        }
    }
}

__global__ void rmsnorm_kernel(const float* __restrict__ x,
                               const float* __restrict__ weight,
                               float* __restrict__ y,
                               int rows, int cols, float eps) {
    const int row = static_cast<int>(blockIdx.x);
    if (row >= rows) return;
    const float* x_row = x + static_cast<size_t>(row) * cols;
    float* y_row = y + static_cast<size_t>(row) * cols;
    float sum = 0.0f;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) sum += x_row[col] * x_row[col];
    extern __shared__ float scratch[];
    const float variance_sum = reduce_sum(sum, scratch);
    const float inv = rsqrtf(variance_sum / static_cast<float>(cols) + eps);
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        y_row[col] = x_row[col] * inv * (1.0f + weight[col]);
    }
}

__global__ void gated_rmsnorm_kernel(const float* __restrict__ x,
                                     const float* __restrict__ weight,
                                     const float* __restrict__ gate,
                                     float* __restrict__ y,
                                     int rows, int cols, float eps) {
    const int row = static_cast<int>(blockIdx.x);
    if (row >= rows) return;
    const float* x_row = x + static_cast<size_t>(row) * cols;
    const float* gate_row = gate + static_cast<size_t>(row) * cols;
    float* y_row = y + static_cast<size_t>(row) * cols;
    float sum = 0.0f;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) sum += x_row[col] * x_row[col];
    extern __shared__ float scratch[];
    const float variance_sum = reduce_sum(sum, scratch);
    const float inv = rsqrtf(variance_sum / static_cast<float>(cols) + eps);
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        y_row[col] = weight[col] * x_row[col] * inv * silu(gate_row[col]);
    }
}

__global__ void l2_norm_kernel(const float* __restrict__ x,
                               float* __restrict__ y,
                               int rows, int cols, float eps) {
    const int row = static_cast<int>(blockIdx.x);
    if (row >= rows) return;
    const float* x_row = x + static_cast<size_t>(row) * cols;
    float* y_row = y + static_cast<size_t>(row) * cols;
    float sum = 0.0f;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) sum += x_row[col] * x_row[col];
    extern __shared__ float scratch[];
    const float norm_sq = reduce_sum(sum, scratch);
    const float inv = rsqrtf(norm_sq + eps);
    for (int col = threadIdx.x; col < cols; col += blockDim.x) y_row[col] = x_row[col] * inv;
}

bool valid_common(const void* x, const void* weight, const void* scale, void* y,
                  int rows, int cols, int scale_stride) {
    return x != nullptr && weight != nullptr && scale != nullptr && y != nullptr &&
           rows > 0 && cols > 0 && scale_stride >= (cols + kBlock - 1) / kBlock;
}

}  // namespace

bool qwen_fp8_e4m3_fp16scale_matvec_cuda(
    const float* d_x,
    const uint8_t* d_weight,
    const uint16_t* d_scale_fp16,
    float* d_y,
    int rows,
    int cols,
    int weight_stride,
    int scale_stride,
    void* stream) {
    if (!valid_common(d_x, d_weight, d_scale_fp16, d_y, rows, cols, scale_stride) ||
        weight_stride < cols) return false;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    // uchar4 loads need a 4-byte aligned row base; every Qwen3.8 projection has
    // a stride that is a multiple of 128, so this is the common path. The kernel
    // reads x straight from global (only its 1 KB static LUT is shared), so it
    // needs no dynamic shared memory and no longer has a cols ceiling. The
    // previous cols*4 request pushed wide projections such as TP1 down_proj
    // (17408 cols) past the 48 KB limit and into the slow fallback.
    if (weight_stride % 4 == 0) {
        constexpr int kWarps = 8;  // 8 warps = 256 threads
        // Batching rows per warp cuts x re-reads, but only pays off while the
        // grid still covers the GPU: 136 blocks is two waves on the 68 SMs of a
        // 2080 Ti. Measured on Qwen3.8 TP4 shapes, picking the largest factor
        // that clears that bar beats a fixed factor for both the wide MLP
        // projections and the narrower attention ones.
        constexpr int kMinBlocks = 136;
        auto blocks_for = [&](int per_warp) {
            const int rows_per_block = kWarps * per_warp;
            return (rows + rows_per_block - 1) / rows_per_block;
        };
        if (blocks_for(4) >= kMinBlocks) {
            fp8_matvec_multirow_kernel<true, kWarps, 4><<<blocks_for(4), kWarps * 32, 0, s>>>(
                d_x, d_weight, d_scale_fp16, d_y, rows, cols, weight_stride, scale_stride);
        } else if (blocks_for(2) >= kMinBlocks) {
            fp8_matvec_multirow_kernel<true, kWarps, 2><<<blocks_for(2), kWarps * 32, 0, s>>>(
                d_x, d_weight, d_scale_fp16, d_y, rows, cols, weight_stride, scale_stride);
        } else {
            fp8_matvec_warp_kernel<true, kWarps><<<blocks_for(1), kWarps * 32, 0, s>>>(
                d_x, d_weight, d_scale_fp16, d_y, rows, cols, weight_stride, scale_stride);
        }
        return cudaGetLastError() == cudaSuccess;
    }
    fp8_matvec_kernel<true><<<rows, 256, 256 * sizeof(float), s>>>(
        d_x, d_weight, d_scale_fp16, d_y, rows, cols, weight_stride, scale_stride);
    return cudaGetLastError() == cudaSuccess;
}

bool qwen_fp8_e4m3_fp16scale_swiglu_matvec_cuda(
    const float* d_x,
    const uint8_t* d_gate_weight,
    const uint16_t* d_gate_scale_fp16,
    const uint8_t* d_up_weight,
    const uint16_t* d_up_scale_fp16,
    float* d_y,
    int rows,
    int cols,
    int weight_stride,
    int scale_stride,
    void* stream) {
    if (!valid_common(d_x, d_gate_weight, d_gate_scale_fp16, d_y, rows, cols, scale_stride) ||
        !d_up_weight || !d_up_scale_fp16 || weight_stride < cols || weight_stride % 4 != 0) {
        return false;
    }
    constexpr int kWarps = 8;
    constexpr int kRowsPerWarp = 1;
    const int blocks = (rows + kWarps * kRowsPerWarp - 1) / (kWarps * kRowsPerWarp);
    const cudaStream_t s = static_cast<cudaStream_t>(stream);
    fp8_swiglu_matvec_kernel<kWarps, kRowsPerWarp><<<blocks, kWarps * 32, 0, s>>>(
        d_x, d_gate_weight, d_gate_scale_fp16, d_up_weight, d_up_scale_fp16,
        d_y, rows, cols, weight_stride, scale_stride);
    return cudaGetLastError() == cudaSuccess;
}

bool qwen_fp8_e4m3_fp16scale_matmul_simt_cuda(
    const float* d_x,
    const uint8_t* d_weight,
    const uint16_t* d_scale_fp16,
    float* d_y,
    int batch,
    int rows,
    int cols,
    int x_stride,
    int y_stride,
    int weight_stride,
    int scale_stride,
    void* stream) {
    if (!valid_common(d_x, d_weight, d_scale_fp16, d_y, rows, cols, scale_stride) ||
        batch <= 0 || x_stride < cols || y_stride < rows || weight_stride < cols) return false;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    // The 128x128 tile needs enough tokens to fill its M dimension; below that
    // the 64x64 tile wastes less work on masked-out rows. It also stages weights
    // with uchar4 and x with float4, so both row strides must be aligned; the
    // 64x64 tile loads scalars and has no such requirement.
    if (batch >= 96 && weight_stride % 4 == 0 && x_stride % 4 == 0) {
        // N64 is the default on SM75: its lower register footprint allows two
        // resident blocks and is faster on the real TP4 projection shapes.
        const char* n64 = std::getenv("QWEN_FP8_PREFILL_WIDE_N64");
        if (n64 == nullptr || std::strcmp(n64, "0") != 0) {
            dim3 grid(static_cast<unsigned>((rows + 63) / 64),
                      static_cast<unsigned>((batch + 127) / 128), 1);
            fp8_matmul_simt_wide_n64_kernel<true><<<grid, 256, 0, s>>>(
                d_x, d_weight, d_scale_fp16, d_y, batch, rows, cols,
                x_stride, y_stride, weight_stride, scale_stride);
        } else {
            dim3 grid(static_cast<unsigned>((rows + 127) / 128),
                      static_cast<unsigned>((batch + 127) / 128), 1);
            fp8_matmul_simt_wide_kernel<true><<<grid, 256, 0, s>>>(
                d_x, d_weight, d_scale_fp16, d_y, batch, rows, cols,
                x_stride, y_stride, weight_stride, scale_stride);
        }
        return cudaGetLastError() == cudaSuccess;
    }
    constexpr int kBatchTile = 64;
    constexpr int kRowTile = 64;
    constexpr int kColTile = 32;
    dim3 grid(static_cast<unsigned>((rows + kRowTile - 1) / kRowTile),
              static_cast<unsigned>((batch + kBatchTile - 1) / kBatchTile), 1);
    fp8_matmul_simt_kernel<true, kColTile><<<grid, 256, 0, s>>>(
        d_x, d_weight, d_scale_fp16, d_y, batch, rows, cols,
        x_stride, y_stride, weight_stride, scale_stride);
    return cudaGetLastError() == cudaSuccess;
}

bool qwen_fp8_e4m3_fp16scale_matmul_rows_cuda(
    const float* d_x,
    const uint8_t* d_weight,
    const uint16_t* d_scale_fp16,
    float* d_y,
    int batch,
    int rows,
    int cols,
    int x_stride,
    int y_stride,
    int weight_stride,
    int scale_stride,
    void* stream) {
    if (!valid_common(d_x, d_weight, d_scale_fp16, d_y, rows, cols, scale_stride) ||
        batch <= 0 || x_stride < cols || y_stride < rows || weight_stride < cols) return false;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    // Blocked SIMT tiles are the default prefill path: same online FP8 decode,
    // 3-9x faster than the per-row sweep below. QWEN_FP8_PREFILL_SIMT=0 restores
    // the old kernel for A/B comparison.
    const char* simt = std::getenv("QWEN_FP8_PREFILL_SIMT");
    if (batch >= 32 && (simt == nullptr || std::strcmp(simt, "0") != 0)) {
        return qwen_fp8_e4m3_fp16scale_matmul_simt_cuda(
            d_x, d_weight, d_scale_fp16, d_y, batch, rows, cols,
            x_stride, y_stride, weight_stride, scale_stride, stream);
    }
    if (weight_stride % 4 == 0) {
        constexpr int kRowsPerBlock = 4;   // 4 warps = 128 threads
        constexpr int kTileBatch = 8;
        dim3 grid(static_cast<unsigned>((rows + kRowsPerBlock - 1) / kRowsPerBlock),
                  static_cast<unsigned>((batch + kTileBatch - 1) / kTileBatch), 1);
        fp8_matmul_tiled_kernel<true, kRowsPerBlock, kTileBatch>
            <<<grid, kRowsPerBlock * 32, 0, s>>>(
                d_x, d_weight, d_scale_fp16, d_y, batch, rows, cols,
                x_stride, y_stride, weight_stride, scale_stride);
        return cudaGetLastError() == cudaSuccess;
    }
    dim3 grid(static_cast<unsigned>(rows), static_cast<unsigned>(batch), 1);
    fp8_matmul_rows_kernel<true><<<grid, 256, 256 * sizeof(float), s>>>(
        d_x, d_weight, d_scale_fp16, d_y, batch, rows, cols,
        x_stride, y_stride, weight_stride, scale_stride);
    return cudaGetLastError() == cudaSuccess;
}

bool qwen_fp8_e4m3_bf16_matvec_cuda(
    const float* d_x,
    const uint8_t* d_weight,
    const uint16_t* d_scale_bf16,
    float* d_y,
    int rows,
    int cols,
    int weight_stride,
    int scale_stride,
    void* stream) {
    if (!valid_common(d_x, d_weight, d_scale_bf16, d_y, rows, cols, scale_stride) ||
        weight_stride < cols) return false;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    fp8_matvec_kernel<false><<<rows, 256, 256 * sizeof(float), s>>>(
        d_x, d_weight, d_scale_bf16, d_y, rows, cols, weight_stride, scale_stride);
    return cudaGetLastError() == cudaSuccess;
}

bool qwen_fp8_e4m3_bf16_matmul_rows_cuda(
    const float* d_x,
    const uint8_t* d_weight,
    const uint16_t* d_scale_bf16,
    float* d_y,
    int batch,
    int rows,
    int cols,
    int x_stride,
    int y_stride,
    int weight_stride,
    int scale_stride,
    void* stream) {
    if (!valid_common(d_x, d_weight, d_scale_bf16, d_y, rows, cols, scale_stride) ||
        batch <= 0 || x_stride < cols || y_stride < rows || weight_stride < cols) return false;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    dim3 grid(static_cast<unsigned>(rows), static_cast<unsigned>(batch), 1);
    fp8_matmul_rows_kernel<false><<<grid, 256, 256 * sizeof(float), s>>>(
        d_x, d_weight, d_scale_bf16, d_y, batch, rows, cols,
        x_stride, y_stride, weight_stride, scale_stride);
    return cudaGetLastError() == cudaSuccess;
}

bool qwen_rmsnorm_f32_cuda(const float* d_x, const float* d_weight,
                           float* d_y, int rows, int cols, float eps,
                           void* stream) {
    if (!d_x || !d_weight || !d_y || rows <= 0 || cols <= 0 || eps < 0.0f) return false;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    rmsnorm_kernel<<<rows, 256, 256 * sizeof(float), s>>>(d_x, d_weight, d_y, rows, cols, eps);
    return cudaGetLastError() == cudaSuccess;
}

bool qwen_gated_rmsnorm_f32_cuda(const float* d_x, const float* d_weight,
                                 const float* d_gate, float* d_y,
                                 int rows, int cols, float eps,
                                 void* stream) {
    if (!d_x || !d_weight || !d_gate || !d_y || rows <= 0 || cols <= 0 || eps < 0.0f) return false;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    gated_rmsnorm_kernel<<<rows, 256, 256 * sizeof(float), s>>>(
        d_x, d_weight, d_gate, d_y, rows, cols, eps);
    return cudaGetLastError() == cudaSuccess;
}

bool qwen_l2_norm_f32_cuda(const float* d_x, float* d_y, int rows, int cols,
                           float eps, void* stream) {
    if (!d_x || !d_y || rows <= 0 || cols <= 0 || eps < 0.0f) return false;
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    l2_norm_kernel<<<rows, 256, 256 * sizeof(float), s>>>(d_x, d_y, rows, cols, eps);
    return cudaGetLastError() == cudaSuccess;
}

// Bulk dequantize FP8 KV cache into dense FP16 buffers. Each block handles one
// (position, kv_head) pair, and threads within the block dequantize head_dim
// channels. The scale tensor uses block quantization (scale_block channels per
// scale), so each thread reads one FP8 code, looks up the corresponding scale,
// multiplies, and writes FP16. This amortizes dequant cost before calling
// tensor-core prefill kernels on quantized KV history (O(context) dequant once,
// not O(rows×context) dequant per query row).
template <bool kFp16Scale>
__global__ void dequant_fp8_kv_cache_kernel(
    const uint8_t* __restrict__ d_k_cache_fp8,
    const uint8_t* __restrict__ d_v_cache_fp8,
    const uint16_t* __restrict__ d_k_scale,
    const uint16_t* __restrict__ d_v_scale,
    uint16_t* __restrict__ d_k_dense_fp16,
    uint16_t* __restrict__ d_v_dense_fp16,
    int context_len, int kv_heads, int head_dim, int scale_block, int max_context) {
    // One block per (position, kv_head). position = blockIdx.x, kv_head = blockIdx.y
    const int position = blockIdx.x;
    const int kv_head = blockIdx.y;
    if (position >= context_len || kv_head >= kv_heads) return;

    // FP8 cache layout: [max_context, kv_heads, head_dim]
    const size_t cache_offset = (static_cast<size_t>(position) * kv_heads + kv_head) * head_dim;
    // Scale layout: [max_context, kv_heads, head_dim / scale_block]
    const int scales_per_head = head_dim / scale_block;
    const size_t scale_offset = (static_cast<size_t>(position) * kv_heads + kv_head) * scales_per_head;
    // Dense FP16 output layout: [max_context, kv_heads, head_dim] (matching FP16 cache)
    const size_t dense_offset = (static_cast<size_t>(position) * kv_heads + kv_head) * head_dim;

    // Each thread dequantizes one channel
    for (int ch = threadIdx.x; ch < head_dim; ch += blockDim.x) {
        const int scale_idx = ch / scale_block;

        // Dequantize K
        const uint8_t k_code = d_k_cache_fp8[cache_offset + ch];
        const float k_scale = scale_bits_to_float<kFp16Scale>(d_k_scale[scale_offset + scale_idx]);
        const float k_val = fp8_e4m3_to_float(k_code) * k_scale;
        d_k_dense_fp16[dense_offset + ch] = __float2half_rn(k_val);

        // Dequantize V
        const uint8_t v_code = d_v_cache_fp8[cache_offset + ch];
        const float v_scale = scale_bits_to_float<kFp16Scale>(d_v_scale[scale_offset + scale_idx]);
        const float v_val = fp8_e4m3_to_float(v_code) * v_scale;
        d_v_dense_fp16[dense_offset + ch] = __float2half_rn(v_val);
    }
}

bool qwen_fp8_dequant_kv_cache_cuda(
    const uint8_t* d_k_cache_fp8, const uint8_t* d_v_cache_fp8,
    const uint16_t* d_k_scale_fp16, const uint16_t* d_v_scale_fp16,
    uint16_t* d_k_dense_fp16, uint16_t* d_v_dense_fp16,
    int context_len, int kv_heads, int head_dim, int scale_block,
    int max_context, void* stream) {
    if (!d_k_cache_fp8 || !d_v_cache_fp8 || !d_k_scale_fp16 || !d_v_scale_fp16 ||
        !d_k_dense_fp16 || !d_v_dense_fp16 || context_len <= 0 || kv_heads <= 0 ||
        head_dim <= 0 || scale_block <= 0 || head_dim % scale_block != 0 ||
        max_context <= 0 || context_len > max_context) {
        return false;
    }
    cudaStream_t s = static_cast<cudaStream_t>(stream);
    const dim3 grid(context_len, kv_heads);
    const int block = 256;
    dequant_fp8_kv_cache_kernel<true><<<grid, block, 0, s>>>(
        d_k_cache_fp8, d_v_cache_fp8, d_k_scale_fp16, d_v_scale_fp16,
        d_k_dense_fp16, d_v_dense_fp16, context_len, kv_heads, head_dim,
        scale_block, max_context);
    return cudaGetLastError() == cudaSuccess;
}


}  // namespace pocket
