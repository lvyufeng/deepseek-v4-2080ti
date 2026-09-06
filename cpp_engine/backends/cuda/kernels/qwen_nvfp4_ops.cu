#include "qwen_cuda_ops.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace pocket {
namespace {

constexpr int kBlockWeights = 64;
constexpr int kScaleGroup = 16;
constexpr int kQ8Group = 32;
constexpr int kRecordBytes = 36;
constexpr int kWarps = 8;
constexpr int kQuantThreads = 256;

__device__ __forceinline__ float half_to_float(uint16_t bits) {
    return __half2float(__ushort_as_half(bits));
}

__device__ __forceinline__ uint16_t float_to_half(float value) {
    return __half_as_ushort(__float2half_rn(value));
}

__device__ __forceinline__ float silu(float value) {
    return value / (1.0f + expf(-value));
}

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

__device__ __forceinline__ float fp4_e2m1_to_float(uint8_t code) {
    constexpr float values[8] = {0.0f, 0.5f, 1.0f, 1.5f,
                                 2.0f, 3.0f, 4.0f, 6.0f};
    const float magnitude = values[code & 7u];
    return (code & 8u) != 0 ? -magnitude : magnitude;
}

__device__ __forceinline__ int fp4_unpack_4codes_x2(uint16_t packed) {
    const uint32_t control_mags = static_cast<uint32_t>(packed) & 0x7777u;
    const uint32_t positive = __byte_perm(
        0x03020100u, 0x0c080604u, control_mags);
    const uint32_t negative = __byte_perm(
        0xfdfeff00u, 0xf4f8fafcu, control_mags);
    const uint32_t control_sign =
        (static_cast<uint32_t>(packed) >> 3) & 0x1111u;
    const uint32_t mask = __byte_perm(
        0x0000ff00u, 0x00000000u, control_sign);
    return static_cast<int>((positive & ~mask) | (negative & mask));
}

__device__ __forceinline__ int dot_i8x4(int activation, int weight,
                                        int accumulator) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610
    return __dp4a(activation, weight, accumulator);
#else
    const int8_t* a = reinterpret_cast<const int8_t*>(&activation);
    const int8_t* w = reinterpret_cast<const int8_t*>(&weight);
    return accumulator + a[0] * w[0] + a[1] * w[1] +
           a[2] * w[2] + a[3] * w[3];
#endif
}

__global__ void quantize_q8_group32_kernel(
    const uint16_t* __restrict__ x,
    int8_t* __restrict__ q8,
    float* __restrict__ scales,
    int batch, int cols, int x_stride) {
    const int group = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int sample = static_cast<int>(blockIdx.y);
    const int groups_per_row = cols / kQ8Group;
    if (sample >= batch || group >= groups_per_row) return;
    const int begin = group * kQ8Group;
    const uint16_t* input = x + static_cast<size_t>(sample) * x_stride + begin;
    int8_t* output = q8 + static_cast<size_t>(sample) * cols + begin;
    float maximum = 0.0f;
#pragma unroll
    for (int index = 0; index < kQ8Group; ++index) {
        maximum = fmaxf(maximum, fabsf(half_to_float(input[index])));
    }
    const float scale = fmaxf(maximum, 1.0e-8f) / 127.0f;
    scales[static_cast<size_t>(sample) * groups_per_row + group] = scale;
    const float inverse = 1.0f / scale;
#pragma unroll
    for (int index = 0; index < kQ8Group; ++index) {
        const int quantized = __float2int_rn(
            half_to_float(input[index]) * inverse);
        output[index] = static_cast<int8_t>(
            max(-127, min(127, quantized)));
    }
}

template <int kWarpsPerBlock>
__global__ void nvfp4_group16_dp4a_warp_kernel(
    const int8_t* __restrict__ q8,
    const float* __restrict__ q8_scales,
    const uint8_t* __restrict__ blocks,
    uint16_t* __restrict__ output,
    int batch, int rows, int cols,
    int q8_stride, int y_stride, int blocks_per_row,
    float weight_global_factor) {
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int row = static_cast<int>(blockIdx.x) * kWarpsPerBlock + warp;
    const int sample = static_cast<int>(blockIdx.y);
    if (row >= rows || sample >= batch) return;
    const int8_t* activation = q8 + static_cast<size_t>(sample) * q8_stride;
    const float* activation_scales = q8_scales +
        static_cast<size_t>(sample) * (cols / kQ8Group);
    const uint8_t* row_blocks = blocks +
        static_cast<size_t>(row) * blocks_per_row * kRecordBytes;
    float sum = 0.0f;
    for (int block = lane; block < blocks_per_row; block += 32) {
        const uint8_t* record = row_blocks + block * kRecordBytes;
        const int8_t* activation_block = activation + block * kBlockWeights;
        const int* activation_packs =
            reinterpret_cast<const int*>(activation_block);
#pragma unroll
        for (int subgroup = 0; subgroup < kBlockWeights / kScaleGroup;
             ++subgroup) {
            int integer_sum = 0;
            const uint8_t* packed = record + 4 + subgroup * 8;
#pragma unroll
            for (int pack = 0; pack < kScaleGroup / 4; ++pack) {
                const uint16_t weight_bytes = static_cast<uint16_t>(
                    packed[pack * 2]) |
                    (static_cast<uint16_t>(packed[pack * 2 + 1]) << 8);
                integer_sum = dot_i8x4(
                    activation_packs[subgroup * 4 + pack],
                    fp4_unpack_4codes_x2(weight_bytes), integer_sum);
            }
            sum += static_cast<float>(integer_sum) * 0.5f *
                   fp8_e4m3_to_float(record[subgroup]) *
                   activation_scales[block * 2 + subgroup / 2];
        }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffffu, sum, offset);
    }
    if (lane == 0) {
        const size_t index = static_cast<size_t>(sample) * y_stride + row;
        output[index] = float_to_half(sum * weight_global_factor);
    }
}

template <bool kFloatOutput, int kWarpsPerBlock>
__global__ void nvfp4_group16_wmma_kernel(
    const int8_t* __restrict__ q8,
    const float* __restrict__ q8_scales,
    const uint8_t* __restrict__ blocks,
    void* __restrict__ output,
    int batch, int rows, int cols,
    int q8_stride, int y_stride, int blocks_per_row,
    float weight_global_factor) {
    using namespace nvcuda;
    constexpr int kTile = 16;
    constexpr int kBlockThreads = kWarpsPerBlock * 32;
    const int tile_row = static_cast<int>(blockIdx.y) * kTile;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int tile_col =
        (static_cast<int>(blockIdx.x) * kWarpsPerBlock + warp) * kTile;

    extern __shared__ __align__(16) unsigned char smem[];
    signed char* activation_tile = reinterpret_cast<signed char*>(smem);
    signed char* weight_tiles = activation_tile + kTile * kTile;
    signed char* weight_tile = weight_tiles + warp * kTile * kTile;
    int* integer_tiles = reinterpret_cast<int*>(
        weight_tiles + kWarpsPerBlock * kTile * kTile);
    int* integer_tile = integer_tiles + warp * kTile * kTile;
    float* accumulators = reinterpret_cast<float*>(
        integer_tiles + kWarpsPerBlock * kTile * kTile);
    float* accumulator = accumulators + warp * kTile * kTile;

    for (int index = lane; index < kTile * kTile; index += 32) {
        accumulator[index] = 0.0f;
    }
    __syncthreads();

    const int groups_per_row = cols / kQ8Group;
    for (int block = 0; block < blocks_per_row; ++block) {
        const int activation_group = block * 2;
#pragma unroll
        for (int subgroup = 0; subgroup < kBlockWeights / kScaleGroup;
             ++subgroup) {
            const int group16 =
                block * (kBlockWeights / kScaleGroup) + subgroup;
            const int k0 = group16 * kScaleGroup;
            for (int index = threadIdx.x; index < kTile * kTile;
                 index += kBlockThreads) {
                const int local_row = index / kTile;
                const int k = index & (kTile - 1);
                const int sample = tile_row + local_row;
                activation_tile[index] = sample < batch
                    ? q8[static_cast<size_t>(sample) * q8_stride + k0 + k]
                    : 0;
            }
            for (int index = lane; index < kTile * kTile; index += 32) {
                const int k = index & (kTile - 1);
                const int local_col = index >> 4;
                const int row = tile_col + local_col;
                int8_t value = 0;
                if (row < rows) {
                    const uint8_t* record = blocks +
                        (static_cast<size_t>(row) * blocks_per_row + block) *
                            kRecordBytes;
                    const uint8_t packed = record[
                        4 + subgroup * (kScaleGroup / 2) + (k >> 1)];
                    const uint8_t code =
                        (k & 1) != 0 ? packed >> 4 : packed & 0x0fu;
                    value = static_cast<int8_t>(
                        fp4_e2m1_to_float(code) * 2.0f);
                }
                weight_tile[k + local_col * kTile] = value;
            }
            __syncthreads();

            wmma::fragment<wmma::matrix_a, kTile, kTile, kTile,
                           signed char, wmma::row_major> activation_fragment;
            wmma::fragment<wmma::matrix_b, kTile, kTile, kTile,
                           signed char, wmma::col_major> weight_fragment;
            wmma::fragment<wmma::accumulator, kTile, kTile, kTile,
                           int> integer_fragment;
            wmma::fill_fragment(integer_fragment, 0);
            wmma::load_matrix_sync(activation_fragment, activation_tile, kTile);
            wmma::load_matrix_sync(weight_fragment, weight_tile, kTile);
            wmma::mma_sync(integer_fragment, activation_fragment,
                           weight_fragment, integer_fragment);
            wmma::store_matrix_sync(integer_tile, integer_fragment, kTile,
                                    wmma::mem_row_major);
            __syncwarp();

            for (int index = lane; index < kTile * kTile; index += 32) {
                const int local_row = index / kTile;
                const int local_col = index & (kTile - 1);
                const int sample = tile_row + local_row;
                const int row = tile_col + local_col;
                if (sample >= batch || row >= rows) continue;
                const uint8_t* record = blocks +
                    (static_cast<size_t>(row) * blocks_per_row + block) *
                        kRecordBytes;
                const float activation_scale = q8_scales[
                    static_cast<size_t>(sample) * groups_per_row +
                    activation_group + subgroup / 2];
                accumulator[index] +=
                    static_cast<float>(integer_tile[index]) * 0.5f *
                    fp8_e4m3_to_float(record[subgroup]) * activation_scale;
            }
            __syncthreads();
        }
    }

    for (int index = lane; index < kTile * kTile; index += 32) {
        const int local_row = index / kTile;
        const int local_col = index & (kTile - 1);
        const int sample = tile_row + local_row;
        const int row = tile_col + local_col;
        if (sample >= batch || row >= rows) continue;
        const size_t output_index =
            static_cast<size_t>(sample) * y_stride + row;
        const float value = accumulator[index] * weight_global_factor;
        if constexpr (kFloatOutput) {
            static_cast<float*>(output)[output_index] = value;
        } else {
            static_cast<uint16_t*>(output)[output_index] =
                float_to_half(value);
        }
    }
}

// Wide INT8 prefill tile. The 128x64 output tile reuses each Q8 activation
// tile across 64 output rows and keeps the two K=16 scale boundaries in one
// 32-value load/compute step. This follows the reuse direction of the
// DeepSeek-V4-Flash INT8 kernels without changing Qwen's group-16 E4M3 math.
__global__ void nvfp4_group16_q8_wide_n64_kernel(
    const int8_t* __restrict__ q8,
    const float* __restrict__ q8_scales,
    const uint8_t* __restrict__ blocks,
    uint16_t* __restrict__ output,
    int batch, int rows, int cols,
    int q8_stride, int y_stride, int blocks_per_row,
    float weight_global_factor) {
    constexpr int kTileM = 128;
    constexpr int kTileN = 64;
    constexpr int kPairK = 32;
    constexpr int kHalfK = 16;
    constexpr int kThreads = 256;

    const int tid = static_cast<int>(threadIdx.x);
    const int batch_base = static_cast<int>(blockIdx.y) * kTileM;
    const int row_base = static_cast<int>(blockIdx.x) * kTileN;
    const int tx = tid & 15;
    const int ty = tid >> 4;
    const int m0 = ty * 4;
    const int m1 = 64 + ty * 4;
    const int n0 = tx * 4;
    const int groups_per_row = cols / kQ8Group;

    extern __shared__ __align__(16) unsigned char smem[];
    int8_t* activation_tile = reinterpret_cast<int8_t*>(smem);
    int8_t* weight_tile = activation_tile + kTileM * kPairK;
    uint8_t* weight_scales = reinterpret_cast<uint8_t*>(
        weight_tile + kTileN * kPairK);
    float* activation_scales = reinterpret_cast<float*>(
        weight_scales + kTileN * 2);

    float accumulator[8][4] = {};
    for (int block = 0; block < blocks_per_row; ++block) {
        for (int pair = 0; pair < 2; ++pair) {
            const int k0 = block * kBlockWeights + pair * kPairK;

            // A is loaded as int32 packs, matching the Q8 layout and avoiding
            // four separate byte loads for every DP4A operand.
            for (int index = tid; index < kTileM * (kPairK / 4);
                 index += kThreads) {
                const int local_sample = index / (kPairK / 4);
                const int pack = index % (kPairK / 4);
                const int sample = batch_base + local_sample;
                int value = 0;
                if (sample < batch) {
                    value = *reinterpret_cast<const int*>(
                        q8 + static_cast<size_t>(sample) * q8_stride +
                        k0 + pack * 4);
                }
                *reinterpret_cast<int*>(
                    activation_tile + local_sample * kPairK + pack * 4) = value;
            }

            // Decode the packed low/high nibbles to the signed INT8 values
            // used by DP4A. The x2 representation is corrected by 0.5 below.
            for (int index = tid; index < kTileN * (kPairK / 4);
                 index += kThreads) {
                const int local_row = index / (kPairK / 4);
                const int pack = index % (kPairK / 4);
                const int row = row_base + local_row;
                int value = 0;
                if (row < rows) {
                    const uint8_t* record = blocks +
                        (static_cast<size_t>(row) * blocks_per_row + block) *
                            kRecordBytes;
                    const int offset = 4 + pair * (kPairK / 2) + pack * 2;
                    const uint16_t packed = static_cast<uint16_t>(
                        record[offset]) |
                        (static_cast<uint16_t>(record[offset + 1]) << 8);
                    value = fp4_unpack_4codes_x2(packed);
                }
                *reinterpret_cast<int*>(
                    weight_tile + local_row * kPairK + pack * 4) = value;
            }

            if (tid < kTileN) {
                const int row = row_base + tid;
                const uint8_t* record = row < rows
                    ? blocks + (static_cast<size_t>(row) * blocks_per_row +
                                block) * kRecordBytes
                    : nullptr;
                weight_scales[tid * 2] =
                    record != nullptr ? record[pair * 2] : 0;
                weight_scales[tid * 2 + 1] =
                    record != nullptr ? record[pair * 2 + 1] : 0;
            }
            if (tid < kTileM) {
                const int sample = batch_base + tid;
                activation_scales[tid] = sample < batch
                    ? q8_scales[static_cast<size_t>(sample) * groups_per_row +
                                k0 / kQ8Group]
                    : 0.0f;
            }
            __syncthreads();

#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int k_half = half * kHalfK;
#pragma unroll
                for (int output_index = 0; output_index < 4; ++output_index) {
                    const int local_output = n0 + output_index;
                    const float scale = fp8_e4m3_to_float(
                        weight_scales[local_output * 2 + half]);
#pragma unroll
                    for (int input_index = 0; input_index < 8; ++input_index) {
                        const int local_input = input_index < 4
                            ? m0 + input_index : m1 + input_index - 4;
                        int sum = 0;
#pragma unroll
                        for (int pack = 0; pack < kHalfK / 4; ++pack) {
                            const int activation = *reinterpret_cast<const int*>(
                                activation_tile + local_input * kPairK +
                                k_half + pack * 4);
                            const int weight = *reinterpret_cast<const int*>(
                                weight_tile + local_output * kPairK +
                                k_half + pack * 4);
                            sum = dot_i8x4(activation, weight, sum);
                        }
                        accumulator[input_index][output_index] +=
                            static_cast<float>(sum) * 0.5f * scale *
                            activation_scales[local_input];
                    }
                }
            }
            // The next pair overwrites both shared tiles.
            __syncthreads();
        }
    }

#pragma unroll
    for (int input_index = 0; input_index < 8; ++input_index) {
        const int local_input = input_index < 4
            ? m0 + input_index : m1 + input_index - 4;
        const int sample = batch_base + local_input;
        if (sample >= batch) continue;
#pragma unroll
        for (int output_index = 0; output_index < 4; ++output_index) {
            const int row = row_base + n0 + output_index;
            if (row >= rows) continue;
            output[static_cast<size_t>(sample) * y_stride + row] =
                float_to_half(accumulator[input_index][output_index] *
                              weight_global_factor);
        }
    }
}

template <int kWarpsPerBlock>
__global__ void nvfp4_group16_swiglu_wmma_kernel(
    const int8_t* __restrict__ q8,
    const float* __restrict__ q8_scales,
    const uint8_t* __restrict__ gate_blocks,
    const uint8_t* __restrict__ up_blocks,
    uint16_t* __restrict__ output,
    int batch, int rows, int cols,
    int q8_stride, int y_stride, int blocks_per_row,
    float gate_weight_global_factor,
    float up_weight_global_factor) {
    using namespace nvcuda;
    constexpr int kTile = 16;
    constexpr int kBlockThreads = kWarpsPerBlock * 32;
    const int tile_row = static_cast<int>(blockIdx.y) * kTile;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int tile_col =
        (static_cast<int>(blockIdx.x) * kWarpsPerBlock + warp) * kTile;

    extern __shared__ __align__(16) unsigned char smem[];
    signed char* activation_tile = reinterpret_cast<signed char*>(smem);
    signed char* weight_tiles = activation_tile + kTile * kTile;
    signed char* weight_tile = weight_tiles + warp * kTile * kTile;
    int* integer_tiles = reinterpret_cast<int*>(
        weight_tiles + kWarpsPerBlock * kTile * kTile);
    int* integer_tile = integer_tiles + warp * kTile * kTile;
    float* accumulators = reinterpret_cast<float*>(
        integer_tiles + kWarpsPerBlock * kTile * kTile);
    float* gate_accumulator = accumulators + warp * 2 * kTile * kTile;
    float* up_accumulator = gate_accumulator + kTile * kTile;

    for (int index = lane; index < kTile * kTile; index += 32) {
        gate_accumulator[index] = 0.0f;
        up_accumulator[index] = 0.0f;
    }
    __syncthreads();

    const int groups_per_row = cols / kQ8Group;
    for (int block = 0; block < blocks_per_row; ++block) {
        const int activation_group = block * 2;
#pragma unroll
        for (int subgroup = 0; subgroup < kBlockWeights / kScaleGroup;
             ++subgroup) {
            const int group16 =
                block * (kBlockWeights / kScaleGroup) + subgroup;
            const int k0 = group16 * kScaleGroup;
            for (int index = threadIdx.x; index < kTile * kTile;
                 index += kBlockThreads) {
                const int local_row = index / kTile;
                const int k = index & (kTile - 1);
                const int sample = tile_row + local_row;
                activation_tile[index] = sample < batch
                    ? q8[static_cast<size_t>(sample) * q8_stride + k0 + k]
                    : 0;
            }
            __syncthreads();

#pragma unroll
            for (int projection = 0; projection < 2; ++projection) {
                const uint8_t* projection_blocks =
                    projection == 0 ? gate_blocks : up_blocks;
                for (int index = lane; index < kTile * kTile; index += 32) {
                    const int k = index & (kTile - 1);
                    const int local_col = index >> 4;
                    const int row = tile_col + local_col;
                    int8_t value = 0;
                    if (row < rows) {
                        const uint8_t* record = projection_blocks +
                            (static_cast<size_t>(row) * blocks_per_row + block) *
                                kRecordBytes;
                        const uint8_t packed = record[
                            4 + subgroup * (kScaleGroup / 2) + (k >> 1)];
                        const uint8_t code =
                            (k & 1) != 0 ? packed >> 4 : packed & 0x0fu;
                        value = static_cast<int8_t>(
                            fp4_e2m1_to_float(code) * 2.0f);
                    }
                    weight_tile[k + local_col * kTile] = value;
                }
                __syncwarp();

                wmma::fragment<wmma::matrix_a, kTile, kTile, kTile,
                               signed char, wmma::row_major> activation_fragment;
                wmma::fragment<wmma::matrix_b, kTile, kTile, kTile,
                               signed char, wmma::col_major> weight_fragment;
                wmma::fragment<wmma::accumulator, kTile, kTile, kTile,
                               int> integer_fragment;
                wmma::fill_fragment(integer_fragment, 0);
                wmma::load_matrix_sync(
                    activation_fragment, activation_tile, kTile);
                wmma::load_matrix_sync(weight_fragment, weight_tile, kTile);
                wmma::mma_sync(integer_fragment, activation_fragment,
                               weight_fragment, integer_fragment);
                wmma::store_matrix_sync(integer_tile, integer_fragment, kTile,
                                        wmma::mem_row_major);
                __syncwarp();

                float* accumulator = projection == 0
                    ? gate_accumulator : up_accumulator;
                for (int index = lane; index < kTile * kTile; index += 32) {
                    const int local_row = index / kTile;
                    const int local_col = index & (kTile - 1);
                    const int sample = tile_row + local_row;
                    const int row = tile_col + local_col;
                    if (sample >= batch || row >= rows) continue;
                    const uint8_t* record = projection_blocks +
                        (static_cast<size_t>(row) * blocks_per_row + block) *
                            kRecordBytes;
                    const float activation_scale = q8_scales[
                        static_cast<size_t>(sample) * groups_per_row +
                        activation_group + subgroup / 2];
                    accumulator[index] +=
                        static_cast<float>(integer_tile[index]) * 0.5f *
                        fp8_e4m3_to_float(record[subgroup]) * activation_scale;
                }
                __syncwarp();
            }
            __syncthreads();
        }
    }

    for (int index = lane; index < kTile * kTile; index += 32) {
        const int local_row = index / kTile;
        const int local_col = index & (kTile - 1);
        const int sample = tile_row + local_row;
        const int row = tile_col + local_col;
        if (sample >= batch || row >= rows) continue;
        const float gate = gate_accumulator[index] *
            gate_weight_global_factor;
        const float up = up_accumulator[index] * up_weight_global_factor;
        output[static_cast<size_t>(sample) * y_stride + row] =
            float_to_half(silu(gate) * up);
    }
}

template <bool kFloatOutput>
__global__ void nvfp4_group16_dp4a_kernel(
    const int8_t* __restrict__ q8,
    const float* __restrict__ q8_scales,
    const uint8_t* __restrict__ blocks,
    void* __restrict__ output,
    int batch, int rows, int cols,
    int q8_stride, int y_stride, int blocks_per_row,
    float weight_global_factor) {
    const int row = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int sample = static_cast<int>(blockIdx.y);
    if (row >= rows || sample >= batch) return;
    const int8_t* activation = q8 + static_cast<size_t>(sample) * q8_stride;
    const float* activation_scales = q8_scales +
        static_cast<size_t>(sample) * (cols / kQ8Group);
    const uint8_t* row_blocks = blocks +
        static_cast<size_t>(row) * blocks_per_row * kRecordBytes;
    float sum = 0.0f;
    for (int block = 0; block < blocks_per_row; ++block) {
        const uint8_t* record = row_blocks + block * kRecordBytes;
        const int8_t* activation_block = activation + block * kBlockWeights;
        const int* activation_packs =
            reinterpret_cast<const int*>(activation_block);
#pragma unroll
        for (int subgroup = 0; subgroup < kBlockWeights / kScaleGroup;
             ++subgroup) {
            int integer_sum = 0;
            const uint8_t* packed = record + 4 + subgroup * 8;
#pragma unroll
            for (int pack = 0; pack < kScaleGroup / 4; ++pack) {
                const uint16_t weight_bytes = static_cast<uint16_t>(
                    packed[pack * 2]) |
                    (static_cast<uint16_t>(packed[pack * 2 + 1]) << 8);
                const int weights_x2 = fp4_unpack_4codes_x2(weight_bytes);
                integer_sum = dot_i8x4(
                    activation_packs[subgroup * 4 + pack], weights_x2,
                    integer_sum);
            }
            sum += static_cast<float>(integer_sum) * 0.5f *
                   fp8_e4m3_to_float(record[subgroup]) *
                   activation_scales[block * 2 + subgroup / 2];
        }
    }
    const size_t index = static_cast<size_t>(sample) * y_stride + row;
    const float value = sum * weight_global_factor;
    if constexpr (kFloatOutput) {
        static_cast<float*>(output)[index] = value;
    } else {
        static_cast<uint16_t*>(output)[index] = float_to_half(value);
    }
}

template <bool kFloatOutput>
__global__ void nvfp4_group16_reference_kernel(
    const uint16_t* __restrict__ x,
    const uint8_t* __restrict__ blocks,
    void* __restrict__ output,
    int batch, int rows, int cols,
    int x_stride, int y_stride, int blocks_per_row,
    float weight_global_factor) {
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int row = static_cast<int>(blockIdx.x) * kWarps + warp;
    const int sample = static_cast<int>(blockIdx.y);
    if (row >= rows || sample >= batch) return;

    const uint16_t* input = x + static_cast<size_t>(sample) * x_stride;
    const uint8_t* row_blocks = blocks +
        static_cast<size_t>(row) * blocks_per_row * kRecordBytes;
    float sum = 0.0f;
    for (int col = lane; col < cols; col += 32) {
        const int block = col / kBlockWeights;
        const int local = col - block * kBlockWeights;
        const uint8_t* record = row_blocks + block * kRecordBytes;
        const uint8_t packed = record[4 + local / 2];
        const uint8_t code = (local & 1) != 0 ? packed >> 4 : packed & 0x0fu;
        const float local_scale = fp8_e4m3_to_float(
            record[local / kScaleGroup]);
        sum += half_to_float(input[col]) * fp4_e2m1_to_float(code) *
               local_scale;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(0xffffffffu, sum, offset);
    }
    if (lane == 0) {
        const size_t index = static_cast<size_t>(sample) * y_stride + row;
        const float value = sum * weight_global_factor;
        if constexpr (kFloatOutput) {
            static_cast<float*>(output)[index] = value;
        } else {
            static_cast<uint16_t*>(output)[index] = float_to_half(value);
        }
    }
}

bool launch_quantize_q8_group32(
    const uint16_t* x, int8_t* q8, float* scales,
    int batch, int cols, int x_stride, void* stream) {
    if (x == nullptr || q8 == nullptr || scales == nullptr || batch <= 0 ||
        cols <= 0 || cols % kQ8Group != 0 || x_stride < cols) {
        return false;
    }
    const int groups_per_row = cols / kQ8Group;
    const dim3 grid(static_cast<unsigned>(
                        (groups_per_row + kQuantThreads - 1) / kQuantThreads),
                    static_cast<unsigned>(batch), 1);
    quantize_q8_group32_kernel<<<grid, kQuantThreads, 0,
        static_cast<cudaStream_t>(stream)>>>(
            x, q8, scales, batch, cols, x_stride);
    return cudaGetLastError() == cudaSuccess;
}

template <bool kFloatOutput>
bool launch_nvfp4_group16_dp4a(
    const int8_t* q8, const float* q8_scales, const uint8_t* blocks,
    void* output, int batch, int rows, int cols, int q8_stride,
    int y_stride, int blocks_per_row, float weight_global_factor,
    void* stream) {
    if (q8 == nullptr || q8_scales == nullptr || blocks == nullptr ||
        output == nullptr || batch <= 0 || rows <= 0 || cols <= 0 ||
        cols % kBlockWeights != 0 || q8_stride < cols || y_stride < rows ||
        blocks_per_row != cols / kBlockWeights ||
        !isfinite(weight_global_factor)) {
        return false;
    }
    if constexpr (!kFloatOutput) {
        constexpr int warps = 8;
        const dim3 grid(static_cast<unsigned>((rows + warps - 1) / warps),
                        static_cast<unsigned>(batch), 1);
        nvfp4_group16_dp4a_warp_kernel<warps>
            <<<grid, warps * 32, 0, static_cast<cudaStream_t>(stream)>>>(
                q8, q8_scales, blocks, static_cast<uint16_t*>(output), batch,
                rows, cols, q8_stride, y_stride, blocks_per_row,
                weight_global_factor);
    } else {
        constexpr int threads = 128;
        const dim3 grid(static_cast<unsigned>((rows + threads - 1) / threads),
                        static_cast<unsigned>(batch), 1);
        nvfp4_group16_dp4a_kernel<kFloatOutput>
            <<<grid, threads, 0, static_cast<cudaStream_t>(stream)>>>(
                q8, q8_scales, blocks, output, batch, rows, cols, q8_stride,
                y_stride, blocks_per_row, weight_global_factor);
    }
    return cudaGetLastError() == cudaSuccess;
}

template <bool kFloatOutput>
bool launch_nvfp4_group16_wmma(
    const int8_t* q8, const float* q8_scales, const uint8_t* blocks,
    void* output, int batch, int rows, int cols, int q8_stride,
    int y_stride, int blocks_per_row, float weight_global_factor,
    void* stream) {
    if (q8 == nullptr || q8_scales == nullptr || blocks == nullptr ||
        output == nullptr || batch <= 0 || rows <= 0 || cols <= 0 ||
        cols % kBlockWeights != 0 || q8_stride < cols || y_stride < rows ||
        blocks_per_row != cols / kBlockWeights ||
        !isfinite(weight_global_factor)) {
        return false;
    }
    constexpr int warps = 4;
    constexpr int tile = 16;
    constexpr size_t shared_bytes =
        tile * tile * sizeof(int8_t) +
        warps * tile * tile * sizeof(int8_t) +
        warps * tile * tile * sizeof(int) +
        warps * tile * tile * sizeof(float);
    const dim3 grid(
        static_cast<unsigned>((rows + warps * tile - 1) / (warps * tile)),
        static_cast<unsigned>((batch + tile - 1) / tile), 1);
    nvfp4_group16_wmma_kernel<kFloatOutput, warps>
        <<<grid, warps * 32, shared_bytes,
           static_cast<cudaStream_t>(stream)>>>(
            q8, q8_scales, blocks, output, batch, rows, cols, q8_stride,
            y_stride, blocks_per_row, weight_global_factor);
    return cudaGetLastError() == cudaSuccess;
}

bool launch_nvfp4_group16_wide_n64(
    const int8_t* q8, const float* q8_scales, const uint8_t* blocks,
    uint16_t* output, int batch, int rows, int cols, int q8_stride,
    int y_stride, int blocks_per_row, float weight_global_factor,
    void* stream) {
    if (q8 == nullptr || q8_scales == nullptr || blocks == nullptr ||
        output == nullptr || batch <= 0 || rows <= 0 || cols <= 0 ||
        cols % kBlockWeights != 0 || q8_stride < cols || y_stride < rows ||
        blocks_per_row != cols / kBlockWeights ||
        !isfinite(weight_global_factor)) {
        return false;
    }
    constexpr int kThreads = 256;
    constexpr int kTileM = 128;
    constexpr int kTileN = 64;
    constexpr int kPairK = 32;
    constexpr size_t shared_bytes =
        static_cast<size_t>(kTileM) * kPairK * sizeof(int8_t) +
        static_cast<size_t>(kTileN) * kPairK * sizeof(int8_t) +
        static_cast<size_t>(kTileN) * 2 * sizeof(uint8_t) +
        static_cast<size_t>(kTileM) * sizeof(float);
    const dim3 grid(
        static_cast<unsigned>((rows + kTileN - 1) / kTileN),
        static_cast<unsigned>((batch + kTileM - 1) / kTileM), 1);
    nvfp4_group16_q8_wide_n64_kernel<<<
        grid, kThreads, shared_bytes, static_cast<cudaStream_t>(stream)>>>(
        q8, q8_scales, blocks, output, batch, rows, cols, q8_stride, y_stride,
        blocks_per_row, weight_global_factor);
    return cudaGetLastError() == cudaSuccess;
}

bool launch_nvfp4_group16_swiglu_wmma(
    const int8_t* q8, const float* q8_scales,
    const uint8_t* gate_blocks, float gate_weight_global_factor,
    const uint8_t* up_blocks, float up_weight_global_factor,
    uint16_t* output, int batch, int rows, int cols, int q8_stride,
    int y_stride, int blocks_per_row, void* stream) {
    if (q8 == nullptr || q8_scales == nullptr || gate_blocks == nullptr ||
        up_blocks == nullptr || output == nullptr || batch <= 0 || rows <= 0 ||
        cols <= 0 || cols % kBlockWeights != 0 || q8_stride < cols ||
        y_stride < rows || blocks_per_row != cols / kBlockWeights ||
        !isfinite(gate_weight_global_factor) ||
        !isfinite(up_weight_global_factor)) {
        return false;
    }
    constexpr int warps = 4;
    constexpr int tile = 16;
    constexpr size_t shared_bytes =
        tile * tile * sizeof(int8_t) +
        warps * tile * tile * sizeof(int8_t) +
        warps * tile * tile * sizeof(int) +
        2 * warps * tile * tile * sizeof(float);
    const dim3 grid(
        static_cast<unsigned>((rows + warps * tile - 1) / (warps * tile)),
        static_cast<unsigned>((batch + tile - 1) / tile), 1);
    nvfp4_group16_swiglu_wmma_kernel<warps>
        <<<grid, warps * 32, shared_bytes,
           static_cast<cudaStream_t>(stream)>>>(
            q8, q8_scales, gate_blocks, up_blocks, output, batch, rows, cols,
            q8_stride, y_stride, blocks_per_row, gate_weight_global_factor,
            up_weight_global_factor);
    return cudaGetLastError() == cudaSuccess;
}

template <bool kFloatOutput>
bool launch_nvfp4_group16_reference(
    const uint16_t* x, const uint8_t* blocks, void* output,
    int batch, int rows, int cols, int x_stride, int y_stride,
    int blocks_per_row, float weight_global_factor, void* stream) {
    if (x == nullptr || blocks == nullptr || output == nullptr || batch <= 0 ||
        rows <= 0 || cols <= 0 || cols % kBlockWeights != 0 ||
        x_stride < cols || y_stride < rows ||
        blocks_per_row != cols / kBlockWeights ||
        !isfinite(weight_global_factor)) {
        return false;
    }
    const dim3 grid(static_cast<unsigned>((rows + kWarps - 1) / kWarps),
                    static_cast<unsigned>(batch), 1);
    nvfp4_group16_reference_kernel<kFloatOutput>
        <<<grid, kWarps * 32, 0, static_cast<cudaStream_t>(stream)>>>(
            x, blocks, output, batch, rows, cols, x_stride, y_stride,
            blocks_per_row, weight_global_factor);
    return cudaGetLastError() == cudaSuccess;
}

}  // namespace

bool qwen_nvfp4_group16_matvec_f16_cuda(
    const uint16_t* x, const uint8_t* blocks, uint16_t* y,
    int rows, int cols, int blocks_per_row, float weight_global_factor,
    void* stream) {
    return launch_nvfp4_group16_reference<false>(
        x, blocks, y, 1, rows, cols, cols, rows, blocks_per_row,
        weight_global_factor, stream);
}

bool qwen_nvfp4_group16_matmul_rows_f16_cuda(
    const uint16_t* x, const uint8_t* blocks, uint16_t* y,
    int batch, int rows, int cols, int x_stride, int y_stride,
    int blocks_per_row, float weight_global_factor, void* stream) {
    return launch_nvfp4_group16_reference<false>(
        x, blocks, y, batch, rows, cols, x_stride, y_stride,
        blocks_per_row, weight_global_factor, stream);
}

bool qwen_nvfp4_group16_matmul_rows_f16_f32_cuda(
    const uint16_t* x, const uint8_t* blocks, float* y,
    int batch, int rows, int cols, int x_stride, int y_stride,
    int blocks_per_row, float weight_global_factor, void* stream) {
    return launch_nvfp4_group16_reference<true>(
        x, blocks, y, batch, rows, cols, x_stride, y_stride,
        blocks_per_row, weight_global_factor, stream);
}

bool qwen_nvfp4_quantize_q8_group32_f16_cuda(
    const uint16_t* x, int8_t* q8, float* q8_scale,
    int batch, int cols, int x_stride, void* stream) {
    return launch_quantize_q8_group32(
        x, q8, q8_scale, batch, cols, x_stride, stream);
}

bool qwen_nvfp4_group16_matmul_q8_f16_cuda(
    const int8_t* q8, const float* q8_scale, const uint8_t* blocks,
    uint16_t* y, int batch, int rows, int cols, int q8_stride,
    int y_stride, int blocks_per_row, float weight_global_factor,
    void* stream) {
    return launch_nvfp4_group16_dp4a<false>(
        q8, q8_scale, blocks, y, batch, rows, cols, q8_stride, y_stride,
        blocks_per_row, weight_global_factor, stream);
}

bool qwen_nvfp4_group16_matmul_q8_f32_cuda(
    const int8_t* q8, const float* q8_scale, const uint8_t* blocks,
    float* y, int batch, int rows, int cols, int q8_stride,
    int y_stride, int blocks_per_row, float weight_global_factor,
    void* stream) {
    return launch_nvfp4_group16_dp4a<true>(
        q8, q8_scale, blocks, y, batch, rows, cols, q8_stride, y_stride,
        blocks_per_row, weight_global_factor, stream);
}

bool qwen_nvfp4_group16_matmul_q8_wmma_f16_cuda(
    const int8_t* q8, const float* q8_scale, const uint8_t* blocks,
    uint16_t* y, int batch, int rows, int cols, int q8_stride,
    int y_stride, int blocks_per_row, float weight_global_factor,
    void* stream) {
    return launch_nvfp4_group16_wmma<false>(
        q8, q8_scale, blocks, y, batch, rows, cols, q8_stride, y_stride,
        blocks_per_row, weight_global_factor, stream);
}

bool qwen_nvfp4_group16_matmul_q8_wmma_f32_cuda(
    const int8_t* q8, const float* q8_scale, const uint8_t* blocks,
    float* y, int batch, int rows, int cols, int q8_stride,
    int y_stride, int blocks_per_row, float weight_global_factor,
    void* stream) {
    return launch_nvfp4_group16_wmma<true>(
        q8, q8_scale, blocks, y, batch, rows, cols, q8_stride, y_stride,
        blocks_per_row, weight_global_factor, stream);
}

bool qwen_nvfp4_group16_matmul_q8_wide_n64_f16_cuda(
    const int8_t* q8, const float* q8_scale, const uint8_t* blocks,
    uint16_t* y, int batch, int rows, int cols, int q8_stride,
    int y_stride, int blocks_per_row, float weight_global_factor,
    void* stream) {
    return launch_nvfp4_group16_wide_n64(
        q8, q8_scale, blocks, y, batch, rows, cols, q8_stride, y_stride,
        blocks_per_row, weight_global_factor, stream);
}

bool qwen_nvfp4_group16_swiglu_q8_wmma_f16_cuda(
    const int8_t* q8, const float* q8_scale,
    const uint8_t* gate_blocks, float gate_weight_global_factor,
    const uint8_t* up_blocks, float up_weight_global_factor,
    uint16_t* y, int batch, int rows, int cols, int q8_stride,
    int y_stride, int blocks_per_row, void* stream) {
    return launch_nvfp4_group16_swiglu_wmma(
        q8, q8_scale, gate_blocks, gate_weight_global_factor,
        up_blocks, up_weight_global_factor, y, batch, rows, cols,
        q8_stride, y_stride, blocks_per_row, stream);
}

}  // namespace pocket
