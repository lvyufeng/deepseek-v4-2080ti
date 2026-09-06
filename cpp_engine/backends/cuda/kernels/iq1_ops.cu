// IQ1_M single-token MoE decode kernels for cpp_engine.
// Correctness-first path: consumes raw IQ1_M blocks directly on device; does not
// route IQ1_M through the existing IQ2_XXS/Q2_K DP4A kernels.

#include "cuda_ops.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <mutex>

namespace pocket {
namespace {

#include "iq1_grid.inc"

constexpr int kIQ1TileN = 8;
constexpr int kIQ1WmmaTile = 16;

constexpr size_t IQ1_W13_SHARED_NSPLIT_BYTES(int W) {
    return kIQ1WmmaTile * kIQ1WmmaTile * 1
         + 2 * static_cast<size_t>(W) * kIQ1WmmaTile * kIQ1WmmaTile * 1
         + static_cast<size_t>(W) * kIQ1WmmaTile * kIQ1WmmaTile * sizeof(int)
         + 2 * kIQ1WmmaTile * sizeof(int)
         + 6 * static_cast<size_t>(W) * kIQ1WmmaTile * sizeof(float)
         + 2 * static_cast<size_t>(W) * kIQ1WmmaTile * kIQ1WmmaTile * sizeof(float);
}

constexpr size_t IQ1_W2_SHARED_NSPLIT_BYTES(int W) {
    return kIQ1WmmaTile * kIQ1WmmaTile * 1
         + static_cast<size_t>(W) * kIQ1WmmaTile * kIQ1WmmaTile * 1
         + static_cast<size_t>(W) * kIQ1WmmaTile * kIQ1WmmaTile * sizeof(int)
         + 2 * kIQ1WmmaTile * sizeof(int)
         + 3 * static_cast<size_t>(W) * kIQ1WmmaTile * sizeof(float)
         + static_cast<size_t>(W) * kIQ1WmmaTile * kIQ1WmmaTile * sizeof(float);
}

static int8_t* g_iq1_grid_device = nullptr;
static std::once_flag g_iq1_grid_init_flag;

const int8_t* iq1_grid_device() {
    std::call_once(g_iq1_grid_init_flag, [] {
        constexpr size_t bytes = 2048ULL * 8;
        if (cudaMalloc(&g_iq1_grid_device, bytes) != cudaSuccess) {
            g_iq1_grid_device = nullptr;
            return;
        }
        if (cudaMemcpy(g_iq1_grid_device, kIQ1MGrid, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
            cudaFree(g_iq1_grid_device);
            g_iq1_grid_device = nullptr;
        }
    });
    return g_iq1_grid_device;
}

inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

int local_env_int_or_default(const char* name, int fallback) {
    const char* v = std::getenv(name);
    if (v == nullptr || *v == '\0') return fallback;
    char* end = nullptr;
    const long parsed = std::strtol(v, &end, 10);
    return (end == v) ? fallback : static_cast<int>(parsed);
}

__device__ __forceinline__ float iq1m_super_scale(const uint16_t* sc) {
    const uint16_t d_bits = static_cast<uint16_t>(
        ((sc[0] & 0xF000u) >> 12) | ((sc[1] & 0xF000u) >> 8) |
        ((sc[2] & 0xF000u) >> 4) | (sc[3] & 0xF000u));
    return __half2float(__ushort_as_half(d_bits));
}

__device__ __forceinline__ int iq1_pack_i8x4(int a, int b, int c, int d) {
    return (a & 0xff) | ((b & 0xff) << 8) | ((c & 0xff) << 16) | ((d & 0xff) << 24);
}

__device__ __forceinline__ float iq1m_block_dot_256(
    const float* __restrict__ x_shared,
    const uint8_t* __restrict__ block,
    const int8_t* __restrict__ iq1_grid,
    int lane) {
    const uint8_t* qs = block;
    const uint8_t* qh = block + 32;
    const uint16_t* sc = reinterpret_cast<const uint16_t*>(block + 48);
    const float d = iq1m_super_scale(sc);
    float local = 0.0f;
    const int j = lane;
    const int sub = j >> 2;
    const int half = (j >> 1) & 1;
    const int pair = j & 1;
    const int s = sub * 2 + half;
    const int local_scale = (sc[s >> 2] >> ((s & 3) * 3)) & 0x07;
    const float dl = d * static_cast<float>(2 * local_scale + 1);
    const int qhv = (qh[j >> 1] >> ((j & 1) * 4)) & 0x0F;
    const int qidx = static_cast<int>(qs[j]) | ((qhv & 0x07) << 8);
    const float delta = (qhv & 0x08) ? -0.125f : 0.125f;
    const int8_t* gvals = iq1_grid + qidx * 8;
    const int k_out = sub * 32 + half * 16 + pair * 8;
    float partial = 0.0f;
    #pragma unroll
    for (int g = 0; g < 8; ++g) {
        partial += x_shared[k_out + g] * (static_cast<float>(gvals[g]) + delta);
    }
    local = dl * partial;
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local += __shfl_down_sync(0xffffffff, local, offset);
    }
    return local;
}

__device__ __forceinline__ float iq1m_block_dot_q8_256(
    const int8_t* __restrict__ xq_shared,
    const float* __restrict__ xs_shared,
    const uint8_t* __restrict__ block,
    const int8_t* __restrict__ iq1_grid,
    int lane) {
    const uint8_t* qs = block;
    const uint8_t* qh = block + 32;
    const uint16_t* sc = reinterpret_cast<const uint16_t*>(block + 48);
    const float d = iq1m_super_scale(sc);
    const int j = lane;
    const int sub = j >> 2;
    const int half = (j >> 1) & 1;
    const int pair = j & 1;
    const int s = sub * 2 + half;
    const int local_scale = (sc[s >> 2] >> ((s & 3) * 3)) & 0x07;
    const float dl = d * static_cast<float>(2 * local_scale + 1);
    const int qhv = (qh[j >> 1] >> ((j & 1) * 4)) & 0x0F;
    const int qidx = static_cast<int>(qs[j]) | ((qhv & 0x07) << 8);
    const float delta = (qhv & 0x08) ? -0.125f : 0.125f;
    const int8_t* gvals = iq1_grid + qidx * 8;
    const int k_out = sub * 32 + half * 16 + pair * 8;
    const int h0 = iq1_pack_i8x4(static_cast<int>(xq_shared[k_out + 0]), static_cast<int>(xq_shared[k_out + 1]), static_cast<int>(xq_shared[k_out + 2]), static_cast<int>(xq_shared[k_out + 3]));
    const int h1 = iq1_pack_i8x4(static_cast<int>(xq_shared[k_out + 4]), static_cast<int>(xq_shared[k_out + 5]), static_cast<int>(xq_shared[k_out + 6]), static_cast<int>(xq_shared[k_out + 7]));
    const int g0 = iq1_pack_i8x4(static_cast<int>(gvals[0]), static_cast<int>(gvals[1]), static_cast<int>(gvals[2]), static_cast<int>(gvals[3]));
    const int g1 = iq1_pack_i8x4(static_cast<int>(gvals[4]), static_cast<int>(gvals[5]), static_cast<int>(gvals[6]), static_cast<int>(gvals[7]));
    const int dot = __dp4a(h0, g0, 0) + __dp4a(h1, g1, 0);
    const int sum_h = __dp4a(h0, 0x01010101, 0) + __dp4a(h1, 0x01010101, 0);
    float local = dl * xs_shared[s] * (static_cast<float>(dot) + delta * static_cast<float>(sum_h));
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        local += __shfl_down_sync(0xffffffff, local, offset);
    }
    return local;
}

__device__ __forceinline__ int iq1_grouped_route_expert(
    const int32_t* __restrict__ seg_starts,
    int n_experts,
    int route) {
    int lo = 0;
    int hi = n_experts;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (seg_starts[mid] <= route) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    if (lo >= n_experts || route < seg_starts[lo] || route >= seg_starts[lo + 1]) return -1;
    return lo;
}

__global__ void iq1_moe_single_w13_kernel(
    const float* __restrict__ x,
    const int64_t* __restrict__ route_slots,
    const uint8_t* __restrict__ w1_blocks,
    const uint8_t* __restrict__ w3_blocks,
    const int8_t* __restrict__ iq1_grid,
    float* __restrict__ gate,
    float* __restrict__ up,
    int routes,
    int n_experts,
    int dim,
    int inter_dim,
    int blocks_per_row) {
    const int route = blockIdx.y;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int out_col = blockIdx.x * kIQ1TileN + warp;
    if (route >= routes || warp >= kIQ1TileN || out_col >= inter_dim) return;
    const int expert = static_cast<int>(route_slots[route]);
    if (expert < 0 || expert >= n_experts) return;

    __shared__ float x_shared[256];
    const uint8_t* w1_row = w1_blocks + (static_cast<int64_t>(expert) * inter_dim + out_col) * blocks_per_row * 56;
    const uint8_t* w3_row = w3_blocks + (static_cast<int64_t>(expert) * inter_dim + out_col) * blocks_per_row * 56;
    float acc1 = 0.0f;
    float acc3 = 0.0f;
    for (int block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
        const int k_base = block_idx * 256;
        for (int k = threadIdx.x; k < 256; k += blockDim.x) {
            const int idx = k_base + k;
            x_shared[k] = idx < dim ? x[idx] : 0.0f;
        }
        __syncthreads();
        const float b1 = iq1m_block_dot_256(x_shared, w1_row + static_cast<int64_t>(block_idx) * 56, iq1_grid, lane);
        const float b3 = iq1m_block_dot_256(x_shared, w3_row + static_cast<int64_t>(block_idx) * 56, iq1_grid, lane);
        if (lane == 0) {
            acc1 += b1;
            acc3 += b3;
        }
        __syncthreads();
    }
    if (lane == 0) {
        gate[static_cast<int64_t>(route) * inter_dim + out_col] = acc1;
        up[static_cast<int64_t>(route) * inter_dim + out_col] = acc3;
    }
}

__global__ void iq1_moe_single_w13_swiglu_kernel(
    const float* __restrict__ x,
    const int64_t* __restrict__ route_slots,
    const float* __restrict__ route_weights,
    const uint8_t* __restrict__ w1_blocks,
    const uint8_t* __restrict__ w3_blocks,
    const int8_t* __restrict__ iq1_grid,
    float* __restrict__ hidden,
    int routes,
    int n_experts,
    int dim,
    int inter_dim,
    int blocks_per_row,
    float swiglu_limit) {
    const int route = blockIdx.y;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int out_col = blockIdx.x * kIQ1TileN + warp;
    if (route >= routes || warp >= kIQ1TileN || out_col >= inter_dim) return;
    const int expert = static_cast<int>(route_slots[route]);
    if (expert < 0 || expert >= n_experts) return;

    __shared__ float x_shared[256];
    const uint8_t* w1_row = w1_blocks + (static_cast<int64_t>(expert) * inter_dim + out_col) * blocks_per_row * 56;
    const uint8_t* w3_row = w3_blocks + (static_cast<int64_t>(expert) * inter_dim + out_col) * blocks_per_row * 56;
    float acc1 = 0.0f;
    float acc3 = 0.0f;
    for (int block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
        const int k_base = block_idx * 256;
        for (int k = threadIdx.x; k < 256; k += blockDim.x) {
            const int idx = k_base + k;
            x_shared[k] = idx < dim ? x[idx] : 0.0f;
        }
        __syncthreads();
        const float b1 = iq1m_block_dot_256(x_shared, w1_row + static_cast<int64_t>(block_idx) * 56, iq1_grid, lane);
        const float b3 = iq1m_block_dot_256(x_shared, w3_row + static_cast<int64_t>(block_idx) * 56, iq1_grid, lane);
        if (lane == 0) {
            acc1 += b1;
            acc3 += b3;
        }
        __syncthreads();
    }
    if (lane == 0) {
        float g = acc1;
        float u = acc3;
        if (swiglu_limit > 0.0f) {
            u = fminf(fmaxf(u, -swiglu_limit), swiglu_limit);
            g = fminf(g, swiglu_limit);
        }
        hidden[static_cast<int64_t>(route) * inter_dim + out_col] =
            (g / (1.0f + expf(-g))) * u * route_weights[route];
    }
}

__global__ void iq1_route_swiglu_kernel(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    const float* __restrict__ route_weights,
    float* __restrict__ hidden,
    int routes,
    int inter_dim,
    float swiglu_limit) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = routes * inter_dim;
    if (idx >= total) return;
    const int route = idx / inter_dim;
    float g = gate[idx];
    float u = up[idx];
    if (swiglu_limit > 0.0f) {
        u = fminf(fmaxf(u, -swiglu_limit), swiglu_limit);
        g = fminf(g, swiglu_limit);
    }
    hidden[idx] = (g / (1.0f + expf(-g))) * u * route_weights[route];
}

__global__ void iq1_moe_single_w2_kernel(
    const float* __restrict__ hidden,
    const int64_t* __restrict__ route_slots,
    const uint8_t* __restrict__ w2_blocks,
    const int8_t* __restrict__ iq1_grid,
    float* __restrict__ y,
    int routes,
    int n_experts,
    int dim,
    int inter_dim,
    int blocks_per_row) {
    const int route = blockIdx.y;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int out_col = blockIdx.x * kIQ1TileN + warp;
    if (route >= routes || warp >= kIQ1TileN || out_col >= dim) return;
    const int expert = static_cast<int>(route_slots[route]);
    if (expert < 0 || expert >= n_experts) return;

    __shared__ float h_shared[256];
    const float* hidden_row = hidden + static_cast<int64_t>(route) * inter_dim;
    const uint8_t* w2_row = w2_blocks + (static_cast<int64_t>(expert) * dim + out_col) * blocks_per_row * 56;
    float acc = 0.0f;
    for (int block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
        const int k_base = block_idx * 256;
        for (int k = threadIdx.x; k < 256; k += blockDim.x) {
            const int idx = k_base + k;
            h_shared[k] = idx < inter_dim ? hidden_row[idx] : 0.0f;
        }
        __syncthreads();
        const float b = iq1m_block_dot_256(h_shared, w2_row + static_cast<int64_t>(block_idx) * 56, iq1_grid, lane);
        if (lane == 0) acc += b;
        __syncthreads();
    }
    if (lane == 0) atomicAdd(y + out_col, acc);
}

__global__ void iq1_moe_single_w2_reduce_kernel(
    const float* __restrict__ hidden,
    const int64_t* __restrict__ route_slots,
    const uint8_t* __restrict__ w2_blocks,
    const int8_t* __restrict__ iq1_grid,
    float* __restrict__ y,
    int routes,
    int n_experts,
    int dim,
    int inter_dim,
    int blocks_per_row) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int out_col = blockIdx.x * kIQ1TileN + warp;
    if (warp >= kIQ1TileN || out_col >= dim) return;

    __shared__ float h_shared[256];
    float total = 0.0f;
    for (int route = 0; route < routes; ++route) {
        const int expert = static_cast<int>(route_slots[route]);
        if (expert < 0 || expert >= n_experts) continue;
        const float* hidden_row = hidden + static_cast<int64_t>(route) * inter_dim;
        const uint8_t* w2_row = w2_blocks + (static_cast<int64_t>(expert) * dim + out_col) * blocks_per_row * 56;
        float acc = 0.0f;
        for (int block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
            const int k_base = block_idx * 256;
            for (int k = threadIdx.x; k < 256; k += blockDim.x) {
                const int idx = k_base + k;
                h_shared[k] = idx < inter_dim ? hidden_row[idx] : 0.0f;
            }
            __syncthreads();
            const float b = iq1m_block_dot_256(h_shared, w2_row + static_cast<int64_t>(block_idx) * 56, iq1_grid, lane);
            if (lane == 0) acc += b;
            __syncthreads();
        }
        if (lane == 0) total += acc;
    }
    if (lane == 0) y[out_col] = total;
}

__global__ void iq1_moe_grouped_w13_swiglu_kernel(
    const float* __restrict__ x_rows,
    const int64_t* __restrict__ route_tokens,
    const float* __restrict__ route_weights,
    const int32_t* __restrict__ seg_starts,
    const uint8_t* __restrict__ w1_blocks,
    const uint8_t* __restrict__ w3_blocks,
    const int8_t* __restrict__ iq1_grid,
    float* __restrict__ hidden,
    int routes,
    int n_experts,
    int dim,
    int inter_dim,
    int blocks_per_row,
    float swiglu_limit) {
    const int expert = blockIdx.z;
    const int route_ordinal = blockIdx.y;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int out_col = blockIdx.x * kIQ1TileN + warp;
    if (expert >= n_experts || route_ordinal >= (seg_starts[expert + 1] - seg_starts[expert]) ||
        warp >= kIQ1TileN || out_col >= inter_dim) return;
    const int route = seg_starts[expert] + route_ordinal;
    if (route < 0 || route >= routes) return;
    const int64_t token = route_tokens[route];
    if (token < 0) return;

    __shared__ float x_shared[256];
    const float* x = x_rows + token * static_cast<int64_t>(dim);
    const uint8_t* w1_row = w1_blocks + (static_cast<int64_t>(expert) * inter_dim + out_col) * blocks_per_row * 56;
    const uint8_t* w3_row = w3_blocks + (static_cast<int64_t>(expert) * inter_dim + out_col) * blocks_per_row * 56;
    float acc1 = 0.0f;
    float acc3 = 0.0f;
    for (int block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
        const int k_base = block_idx * 256;
        for (int k = threadIdx.x; k < 256; k += blockDim.x) {
            const int idx = k_base + k;
            x_shared[k] = idx < dim ? x[idx] : 0.0f;
        }
        __syncthreads();
        const float b1 = iq1m_block_dot_256(x_shared, w1_row + static_cast<int64_t>(block_idx) * 56, iq1_grid, lane);
        const float b3 = iq1m_block_dot_256(x_shared, w3_row + static_cast<int64_t>(block_idx) * 56, iq1_grid, lane);
        if (lane == 0) {
            acc1 += b1;
            acc3 += b3;
        }
        __syncthreads();
    }
    if (lane == 0) {
        float g = acc1;
        float u = acc3;
        if (swiglu_limit > 0.0f) {
            u = fminf(fmaxf(u, -swiglu_limit), swiglu_limit);
            g = fminf(g, swiglu_limit);
        }
        hidden[static_cast<int64_t>(route) * inter_dim + out_col] =
            (g / (1.0f + expf(-g))) * u * route_weights[route];
    }
}

__global__ void iq1_moe_grouped_w13_swiglu_route_kernel(
    const float* __restrict__ x_rows,
    const int64_t* __restrict__ route_tokens,
    const float* __restrict__ route_weights,
    const int32_t* __restrict__ seg_starts,
    const uint8_t* __restrict__ w1_blocks,
    const uint8_t* __restrict__ w3_blocks,
    const int8_t* __restrict__ iq1_grid,
    float* __restrict__ hidden,
    int routes,
    int n_experts,
    int dim,
    int inter_dim,
    int blocks_per_row,
    float swiglu_limit) {
    const int route = blockIdx.y;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int out_col = blockIdx.x * kIQ1TileN + warp;
    if (route >= routes || warp >= kIQ1TileN || out_col >= inter_dim) return;
    const int expert = iq1_grouped_route_expert(seg_starts, n_experts, route);
    if (expert < 0) return;
    const int64_t token = route_tokens[route];
    if (token < 0) return;

    __shared__ float x_shared[256];
    const float* x = x_rows + token * static_cast<int64_t>(dim);
    const uint8_t* w1_row = w1_blocks + (static_cast<int64_t>(expert) * inter_dim + out_col) * blocks_per_row * 56;
    const uint8_t* w3_row = w3_blocks + (static_cast<int64_t>(expert) * inter_dim + out_col) * blocks_per_row * 56;
    float acc1 = 0.0f;
    float acc3 = 0.0f;
    for (int block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
        const int k_base = block_idx * 256;
        for (int k = threadIdx.x; k < 256; k += blockDim.x) {
            const int idx = k_base + k;
            x_shared[k] = idx < dim ? x[idx] : 0.0f;
        }
        __syncthreads();
        const float b1 = iq1m_block_dot_256(x_shared, w1_row + static_cast<int64_t>(block_idx) * 56, iq1_grid, lane);
        const float b3 = iq1m_block_dot_256(x_shared, w3_row + static_cast<int64_t>(block_idx) * 56, iq1_grid, lane);
        if (lane == 0) {
            acc1 += b1;
            acc3 += b3;
        }
        __syncthreads();
    }
    if (lane == 0) {
        float g = acc1;
        float u = acc3;
        if (swiglu_limit > 0.0f) {
            u = fminf(fmaxf(u, -swiglu_limit), swiglu_limit);
            g = fminf(g, swiglu_limit);
        }
        hidden[static_cast<int64_t>(route) * inter_dim + out_col] =
            (g / (1.0f + expf(-g))) * u * route_weights[route];
    }
}

__global__ void iq1_moe_grouped_w2_kernel(
    const float* __restrict__ hidden,
    const int64_t* __restrict__ route_tokens,
    const int32_t* __restrict__ seg_starts,
    const uint8_t* __restrict__ w2_blocks,
    const int8_t* __restrict__ iq1_grid,
    float* __restrict__ y_rows,
    int routes,
    int n_experts,
    int dim,
    int inter_dim,
    int blocks_per_row) {
    const int expert = blockIdx.z;
    const int route_ordinal = blockIdx.y;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int out_col = blockIdx.x * kIQ1TileN + warp;
    if (expert >= n_experts || route_ordinal >= (seg_starts[expert + 1] - seg_starts[expert]) ||
        warp >= kIQ1TileN || out_col >= dim) return;
    const int route = seg_starts[expert] + route_ordinal;
    if (route < 0 || route >= routes) return;
    const int64_t token = route_tokens[route];
    if (token < 0) return;

    __shared__ float h_shared[256];
    const float* hidden_row = hidden + static_cast<int64_t>(route) * inter_dim;
    const uint8_t* w2_row = w2_blocks + (static_cast<int64_t>(expert) * dim + out_col) * blocks_per_row * 56;
    float acc = 0.0f;
    for (int block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
        const int k_base = block_idx * 256;
        for (int k = threadIdx.x; k < 256; k += blockDim.x) {
            const int idx = k_base + k;
            h_shared[k] = idx < inter_dim ? hidden_row[idx] : 0.0f;
        }
        __syncthreads();
        const float b = iq1m_block_dot_256(h_shared, w2_row + static_cast<int64_t>(block_idx) * 56, iq1_grid, lane);
        if (lane == 0) acc += b;
        __syncthreads();
    }
    if (lane == 0) {
        atomicAdd(y_rows + token * static_cast<int64_t>(dim) + out_col, acc);
    }
}

__global__ void iq1_moe_grouped_w2_route_kernel(
    const float* __restrict__ hidden,
    const int64_t* __restrict__ route_tokens,
    const int32_t* __restrict__ seg_starts,
    const uint8_t* __restrict__ w2_blocks,
    const int8_t* __restrict__ iq1_grid,
    float* __restrict__ y_rows,
    int routes,
    int n_experts,
    int dim,
    int inter_dim,
    int blocks_per_row) {
    const int route = blockIdx.y;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int out_col = blockIdx.x * kIQ1TileN + warp;
    if (route >= routes || warp >= kIQ1TileN || out_col >= dim) return;
    const int expert = iq1_grouped_route_expert(seg_starts, n_experts, route);
    if (expert < 0) return;
    const int64_t token = route_tokens[route];
    if (token < 0) return;

    __shared__ float h_shared[256];
    const float* hidden_row = hidden + static_cast<int64_t>(route) * inter_dim;
    const uint8_t* w2_row = w2_blocks + (static_cast<int64_t>(expert) * dim + out_col) * blocks_per_row * 56;
    float acc = 0.0f;
    for (int block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
        const int k_base = block_idx * 256;
        for (int k = threadIdx.x; k < 256; k += blockDim.x) {
            const int idx = k_base + k;
            h_shared[k] = idx < inter_dim ? hidden_row[idx] : 0.0f;
        }
        __syncthreads();
        const float b = iq1m_block_dot_256(h_shared, w2_row + static_cast<int64_t>(block_idx) * 56, iq1_grid, lane);
        if (lane == 0) acc += b;
        __syncthreads();
    }
    if (lane == 0) atomicAdd(y_rows + token * static_cast<int64_t>(dim) + out_col, acc);
}

__global__ void iq1_moe_grouped_w2_q8_kernel(
    const int8_t* __restrict__ hidden_q,
    const float* __restrict__ hidden_scale,
    const int64_t* __restrict__ route_tokens,
    const int32_t* __restrict__ seg_starts,
    const uint8_t* __restrict__ w2_blocks,
    const int8_t* __restrict__ iq1_grid,
    float* __restrict__ y_rows,
    int routes,
    int n_experts,
    int max_count,
    int dim,
    int inter_dim,
    int blocks_per_row,
    int hidden_groups) {
    const int expert = blockIdx.z;
    const int route_ordinal = blockIdx.y;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int out_col = blockIdx.x * kIQ1TileN + warp;
    if (expert >= n_experts || route_ordinal >= max_count || warp >= kIQ1TileN || out_col >= dim) return;
    const int route = seg_starts[expert] + route_ordinal;
    if (route >= seg_starts[expert + 1] || route >= routes) return;
    const int64_t token = route_tokens[route];
    if (token < 0) return;

    __shared__ int8_t hq_shared[256];
    __shared__ float hs_shared[16];
    const int8_t* hidden_row = hidden_q + static_cast<int64_t>(route) * inter_dim;
    const float* scale_row = hidden_scale + static_cast<int64_t>(route) * hidden_groups;
    const uint8_t* w2_row = w2_blocks + (static_cast<int64_t>(expert) * dim + out_col) * blocks_per_row * 56;
    float acc = 0.0f;
    for (int block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
        const int k_base = block_idx * 256;
        for (int k = threadIdx.x; k < 256; k += blockDim.x) {
            const int idx = k_base + k;
            hq_shared[k] = idx < inter_dim ? hidden_row[idx] : 0;
        }
        for (int g = threadIdx.x; g < 16; g += blockDim.x) {
            hs_shared[g] = (block_idx * 16 + g) < hidden_groups ? scale_row[block_idx * 16 + g] : 0.0f;
        }
        __syncthreads();
        const float b = iq1m_block_dot_q8_256(hq_shared, hs_shared, w2_row + static_cast<int64_t>(block_idx) * 56, iq1_grid, lane);
        if (lane == 0) acc += b;
        __syncthreads();
    }
    if (lane == 0) atomicAdd(y_rows + token * static_cast<int64_t>(dim) + out_col, acc);
}

__global__ void iq1_moe_grouped_w2_q8_route_kernel(
    const int8_t* __restrict__ hidden_q,
    const float* __restrict__ hidden_scale,
    const int64_t* __restrict__ route_tokens,
    const int32_t* __restrict__ seg_starts,
    const uint8_t* __restrict__ w2_blocks,
    const int8_t* __restrict__ iq1_grid,
    float* __restrict__ y_rows,
    int routes,
    int n_experts,
    int dim,
    int inter_dim,
    int blocks_per_row,
    int hidden_groups) {
    const int route = blockIdx.y;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int out_col = blockIdx.x * kIQ1TileN + warp;
    if (route >= routes || warp >= kIQ1TileN || out_col >= dim) return;
    const int expert = iq1_grouped_route_expert(seg_starts, n_experts, route);
    if (expert < 0) return;
    const int64_t token = route_tokens[route];
    if (token < 0) return;

    __shared__ int8_t hq_shared[256];
    __shared__ float hs_shared[16];
    const int8_t* hidden_row = hidden_q + static_cast<int64_t>(route) * inter_dim;
    const float* scale_row = hidden_scale + static_cast<int64_t>(route) * hidden_groups;
    const uint8_t* w2_row = w2_blocks + (static_cast<int64_t>(expert) * dim + out_col) * blocks_per_row * 56;
    float acc = 0.0f;
    for (int block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
        const int k_base = block_idx * 256;
        for (int k = threadIdx.x; k < 256; k += blockDim.x) {
            const int idx = k_base + k;
            hq_shared[k] = idx < inter_dim ? hidden_row[idx] : 0;
        }
        for (int g = threadIdx.x; g < 16; g += blockDim.x) {
            hs_shared[g] = (block_idx * 16 + g) < hidden_groups ? scale_row[block_idx * 16 + g] : 0.0f;
        }
        __syncthreads();
        const float b = iq1m_block_dot_q8_256(hq_shared, hs_shared, w2_row + static_cast<int64_t>(block_idx) * 56, iq1_grid, lane);
        if (lane == 0) acc += b;
        __syncthreads();
    }
    if (lane == 0) atomicAdd(y_rows + token * static_cast<int64_t>(dim) + out_col, acc);
}

__global__ void iq1_quantize_route_rows_q8_16_kernel(
    const float* __restrict__ x_rows,
    const int64_t* __restrict__ route_tokens,
    int8_t* __restrict__ x_q,
    float* __restrict__ x_scale,
    int routes,
    int dim,
    int x_groups) {
    const int route = blockIdx.x;
    const int group = blockIdx.y;
    const int lane = threadIdx.x;
    if (route >= routes || group >= x_groups || lane >= 16) return;
    const int k = group * 16 + lane;
    const int64_t token = route_tokens[route];
    const bool valid = token >= 0 && k < dim;
    const float xv = valid ? x_rows[token * static_cast<int64_t>(dim) + k] : 0.0f;
    float maxv = fabsf(xv);
    #pragma unroll
    for (int offset = 8; offset > 0; offset >>= 1) {
        maxv = fmaxf(maxv, __shfl_down_sync(0xffff, maxv, offset));
    }
    const float scale = __shfl_sync(0xffff, maxv > 0.0f ? maxv / 127.0f : 1.0e-8f, 0);
    if (lane == 0) {
        x_scale[static_cast<int64_t>(route) * x_groups + group] = scale;
    }
    if (k < dim) {
        int q = __float2int_rn(xv / scale);
        q = q < -127 ? -127 : (q > 127 ? 127 : q);
        x_q[static_cast<int64_t>(route) * dim + k] = static_cast<int8_t>(valid ? q : 0);
    }
}

__device__ __forceinline__ void iq1m_group_meta(
    const uint8_t* __restrict__ block,
    int s,
    float& dl,
    float& delta0,
    float& delta1) {
    const uint8_t* qh = block + 32;
    const uint16_t* sc = reinterpret_cast<const uint16_t*>(block + 48);
    const float d = iq1m_super_scale(sc);
    const int local_scale = (sc[s >> 2] >> ((s & 3) * 3)) & 0x07;
    dl = d * static_cast<float>(2 * local_scale + 1);
    const int sub = s >> 1;
    const int half = s & 1;
    const int j0 = sub * 4 + half * 2;
    const int qhv0 = (qh[j0 >> 1] >> ((j0 & 1) * 4)) & 0x0F;
    const int qhv1 = (qh[(j0 + 1) >> 1] >> (((j0 + 1) & 1) * 4)) & 0x0F;
    delta0 = (qhv0 & 0x08) ? -0.125f : 0.125f;
    delta1 = (qhv1 & 0x08) ? -0.125f : 0.125f;
}

__device__ __forceinline__ signed char iq1m_group_value(
    const uint8_t* __restrict__ block,
    const int8_t* __restrict__ iq1_grid,
    int s,
    int k) {
    const uint8_t* qs = block;
    const uint8_t* qh = block + 32;
    const int sub = s >> 1;
    const int half = s & 1;
    const int pair = k >> 3;
    const int g = k & 7;
    const int j = sub * 4 + half * 2 + pair;
    const int qhv = (qh[j >> 1] >> ((j & 1) * 4)) & 0x0F;
    const int qidx = static_cast<int>(qs[j]) | ((qhv & 0x07) << 8);
    return static_cast<signed char>(iq1_grid[qidx * 8 + g]);
}

template <int WARPS>
__global__ void iq1_moe_grouped_w13_q8_wmma_compact_nsplit_kernel(
    const int8_t* __restrict__ x_q,
    const float* __restrict__ x_scale,
    const float* __restrict__ route_weights,
    const int32_t* __restrict__ seg_starts,
    const int32_t* __restrict__ tile_experts,
    const int32_t* __restrict__ tile_rows,
    const uint8_t* __restrict__ w1_blocks,
    const uint8_t* __restrict__ w3_blocks,
    const int8_t* __restrict__ iq1_grid,
    float* __restrict__ hidden,
    int tile_count,
    int dim,
    int inter_dim,
    int blocks_per_row,
    int x_groups,
    float swiglu_limit) {
    using namespace nvcuda;
    constexpr int TILE = kIQ1WmmaTile;
    const int tile_id = blockIdx.y;
    if (tile_id >= tile_count) return;
    const int expert = tile_experts[tile_id];
    const int row_base = tile_rows[tile_id];
    const int col_base_block = blockIdx.x * (TILE * WARPS);
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int col_base = col_base_block + warp_id * TILE;
    const int count = seg_starts[expert + 1] - seg_starts[expert];
    extern __shared__ __align__(16) unsigned char smem[];
    signed char* a_tile = reinterpret_cast<signed char*>(smem);
    signed char* b_tile_block = a_tile + TILE * TILE;
    signed char* b_tile_w1 = b_tile_block + warp_id * TILE * TILE;
    signed char* b_tile_w3 = b_tile_block + (WARPS + warp_id) * TILE * TILE;
    int* c_tile_block = reinterpret_cast<int*>(b_tile_block + 2 * WARPS * TILE * TILE);
    int* c_tile_warp = c_tile_block + warp_id * TILE * TILE;
    int* sum0 = c_tile_block + WARPS * TILE * TILE;
    int* sum1 = sum0 + TILE;
    float* meta = reinterpret_cast<float*>(sum1 + TILE);
    float* w1_dl = meta + warp_id * TILE;
    float* w1_delta0 = meta + (WARPS + warp_id) * TILE;
    float* w1_delta1 = meta + (2 * WARPS + warp_id) * TILE;
    float* w3_dl = meta + (3 * WARPS + warp_id) * TILE;
    float* w3_delta0 = meta + (4 * WARPS + warp_id) * TILE;
    float* w3_delta1 = meta + (5 * WARPS + warp_id) * TILE;
    float* acc_base = meta + 6 * WARPS * TILE;
    float* gate_acc = acc_base + warp_id * TILE * TILE;
    float* up_acc = acc_base + (WARPS + warp_id) * TILE * TILE;

    #pragma unroll
    for (int i = lane; i < TILE * TILE; i += 32) {
        gate_acc[i] = 0.0f;
        up_acc[i] = 0.0f;
    }
    __syncthreads();

    for (int kg = 0; kg < x_groups; ++kg) {
        const int block_idx = kg >> 4;
        const int s = kg & 15;
        const int k0 = kg * TILE;
        wmma::fragment<wmma::matrix_a, TILE, TILE, TILE, signed char, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, TILE, TILE, TILE, signed char, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, TILE, TILE, TILE, int> c1_frag;
        wmma::fragment<wmma::accumulator, TILE, TILE, TILE, int> c3_frag;
        wmma::fill_fragment(c1_frag, 0);
        wmma::fill_fragment(c3_frag, 0);

        for (int i = tid; i < TILE * TILE; i += blockDim.x) {
            const int r = i / TILE;
            const int k = i & (TILE - 1);
            const int row = row_base + r;
            const int route = seg_starts[expert] + row;
            a_tile[i] = (row < count && (k0 + k) < dim) ? x_q[static_cast<int64_t>(route) * dim + k0 + k] : 0;
        }
        __syncthreads();
        if (tid < TILE) {
            int s0 = 0;
            int s1 = 0;
            #pragma unroll
            for (int k = 0; k < 8; ++k) s0 += static_cast<int>(a_tile[tid * TILE + k]);
            #pragma unroll
            for (int k = 8; k < 16; ++k) s1 += static_cast<int>(a_tile[tid * TILE + k]);
            sum0[tid] = s0;
            sum1[tid] = s1;
        }
        for (int i = lane; i < TILE * TILE; i += 32) {
            const int k = i & (TILE - 1);
            const int n = i >> 4;
            const int col = col_base + n;
            signed char v1 = 0;
            signed char v3 = 0;
            if (col < inter_dim && (k0 + k) < dim && block_idx < blocks_per_row) {
                const uint8_t* w1_block = w1_blocks + ((static_cast<int64_t>(expert) * inter_dim + col) * blocks_per_row + block_idx) * 56;
                const uint8_t* w3_block = w3_blocks + ((static_cast<int64_t>(expert) * inter_dim + col) * blocks_per_row + block_idx) * 56;
                v1 = iq1m_group_value(w1_block, iq1_grid, s, k);
                v3 = iq1m_group_value(w3_block, iq1_grid, s, k);
            }
            b_tile_w1[k + n * TILE] = v1;
            b_tile_w3[k + n * TILE] = v3;
        }
        if (lane < TILE) {
            const int n = lane;
            const int col = col_base + n;
            float dl1 = 0.0f, d10 = 0.0f, d11 = 0.0f;
            float dl3 = 0.0f, d30 = 0.0f, d31 = 0.0f;
            if (col < inter_dim && block_idx < blocks_per_row) {
                const uint8_t* w1_block = w1_blocks + ((static_cast<int64_t>(expert) * inter_dim + col) * blocks_per_row + block_idx) * 56;
                const uint8_t* w3_block = w3_blocks + ((static_cast<int64_t>(expert) * inter_dim + col) * blocks_per_row + block_idx) * 56;
                iq1m_group_meta(w1_block, s, dl1, d10, d11);
                iq1m_group_meta(w3_block, s, dl3, d30, d31);
            }
            w1_dl[n] = dl1; w1_delta0[n] = d10; w1_delta1[n] = d11;
            w3_dl[n] = dl3; w3_delta0[n] = d30; w3_delta1[n] = d31;
        }
        __syncthreads();
        wmma::load_matrix_sync(a_frag, a_tile, TILE);
        wmma::load_matrix_sync(b_frag, b_tile_w1, TILE);
        wmma::mma_sync(c1_frag, a_frag, b_frag, c1_frag);
        wmma::load_matrix_sync(b_frag, b_tile_w3, TILE);
        wmma::mma_sync(c3_frag, a_frag, b_frag, c3_frag);
        wmma::store_matrix_sync(c_tile_warp, c1_frag, TILE, wmma::mem_row_major);
        #pragma unroll
        for (int i = lane; i < TILE * TILE; i += 32) {
            const int r = i / TILE;
            const int n = i & (TILE - 1);
            const int row = row_base + r;
            const int route = seg_starts[expert] + row;
            const int col = col_base + n;
            if (row < count && col < inter_dim) {
                const float xs = x_scale[static_cast<int64_t>(route) * x_groups + kg];
                const float corrected = static_cast<float>(c_tile_warp[i]) + w1_delta0[n] * static_cast<float>(sum0[r]) + w1_delta1[n] * static_cast<float>(sum1[r]);
                gate_acc[i] += xs * w1_dl[n] * corrected;
            }
        }
        wmma::store_matrix_sync(c_tile_warp, c3_frag, TILE, wmma::mem_row_major);
        #pragma unroll
        for (int i = lane; i < TILE * TILE; i += 32) {
            const int r = i / TILE;
            const int n = i & (TILE - 1);
            const int row = row_base + r;
            const int route = seg_starts[expert] + row;
            const int col = col_base + n;
            if (row < count && col < inter_dim) {
                const float xs = x_scale[static_cast<int64_t>(route) * x_groups + kg];
                const float corrected = static_cast<float>(c_tile_warp[i]) + w3_delta0[n] * static_cast<float>(sum0[r]) + w3_delta1[n] * static_cast<float>(sum1[r]);
                up_acc[i] += xs * w3_dl[n] * corrected;
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = lane; i < TILE * TILE; i += 32) {
        const int r = i / TILE;
        const int n = i & (TILE - 1);
        const int row = row_base + r;
        const int col = col_base + n;
        if (row < count && col < inter_dim) {
            const int route = seg_starts[expert] + row;
            float g = gate_acc[i];
            float u = up_acc[i];
            if (swiglu_limit > 0.0f) {
                u = fminf(fmaxf(u, -swiglu_limit), swiglu_limit);
                g = fminf(g, swiglu_limit);
            }
            hidden[static_cast<int64_t>(route) * inter_dim + col] =
                (g / (1.0f + expf(-g))) * u * route_weights[route];
        }
    }
}

template <int WARPS>
__global__ void iq1_moe_grouped_w2_q8_wmma_compact_nsplit_kernel(
    const int8_t* __restrict__ hidden_q,
    const float* __restrict__ hidden_scale,
    const int64_t* __restrict__ route_tokens,
    const int32_t* __restrict__ seg_starts,
    const int32_t* __restrict__ tile_experts,
    const int32_t* __restrict__ tile_rows,
    const uint8_t* __restrict__ w2_blocks,
    const int8_t* __restrict__ iq1_grid,
    float* __restrict__ y_rows,
    int tile_count,
    int dim,
    int inter_dim,
    int blocks_per_row,
    int hidden_groups) {
    using namespace nvcuda;
    constexpr int TILE = kIQ1WmmaTile;
    const int tile_id = blockIdx.y;
    if (tile_id >= tile_count) return;
    const int expert = tile_experts[tile_id];
    const int row_base = tile_rows[tile_id];
    const int col_base_block = blockIdx.x * (TILE * WARPS);
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int col_base = col_base_block + warp_id * TILE;
    const int count = seg_starts[expert + 1] - seg_starts[expert];
    extern __shared__ __align__(16) unsigned char smem[];
    signed char* a_tile = reinterpret_cast<signed char*>(smem);
    signed char* b_tile_block = a_tile + TILE * TILE;
    signed char* b_tile = b_tile_block + warp_id * TILE * TILE;
    int* c_tile_block = reinterpret_cast<int*>(b_tile_block + WARPS * TILE * TILE);
    int* c_tile_warp = c_tile_block + warp_id * TILE * TILE;
    int* sum0 = c_tile_block + WARPS * TILE * TILE;
    int* sum1 = sum0 + TILE;
    float* meta = reinterpret_cast<float*>(sum1 + TILE);
    float* w2_dl = meta + warp_id * TILE;
    float* w2_delta0 = meta + (WARPS + warp_id) * TILE;
    float* w2_delta1 = meta + (2 * WARPS + warp_id) * TILE;
    float* acc_base = meta + 3 * WARPS * TILE;
    float* acc = acc_base + warp_id * TILE * TILE;

    #pragma unroll
    for (int i = lane; i < TILE * TILE; i += 32) acc[i] = 0.0f;
    __syncthreads();

    for (int kg = 0; kg < hidden_groups; ++kg) {
        const int block_idx = kg >> 4;
        const int s = kg & 15;
        const int k0 = kg * TILE;
        wmma::fragment<wmma::matrix_a, TILE, TILE, TILE, signed char, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, TILE, TILE, TILE, signed char, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, TILE, TILE, TILE, int> c_frag;
        wmma::fill_fragment(c_frag, 0);

        for (int i = tid; i < TILE * TILE; i += blockDim.x) {
            const int r = i / TILE;
            const int k = i & (TILE - 1);
            const int row = row_base + r;
            const int route = seg_starts[expert] + row;
            a_tile[i] = (row < count && (k0 + k) < inter_dim) ? hidden_q[static_cast<int64_t>(route) * inter_dim + k0 + k] : 0;
        }
        __syncthreads();
        if (tid < TILE) {
            int s0 = 0;
            int s1 = 0;
            #pragma unroll
            for (int k = 0; k < 8; ++k) s0 += static_cast<int>(a_tile[tid * TILE + k]);
            #pragma unroll
            for (int k = 8; k < 16; ++k) s1 += static_cast<int>(a_tile[tid * TILE + k]);
            sum0[tid] = s0;
            sum1[tid] = s1;
        }
        for (int i = lane; i < TILE * TILE; i += 32) {
            const int k = i & (TILE - 1);
            const int n = i >> 4;
            const int col = col_base + n;
            signed char v = 0;
            if (col < dim && (k0 + k) < inter_dim && block_idx < blocks_per_row) {
                const uint8_t* w2_block = w2_blocks + ((static_cast<int64_t>(expert) * dim + col) * blocks_per_row + block_idx) * 56;
                v = iq1m_group_value(w2_block, iq1_grid, s, k);
            }
            b_tile[k + n * TILE] = v;
        }
        if (lane < TILE) {
            const int n = lane;
            const int col = col_base + n;
            float dl = 0.0f, d0 = 0.0f, d1 = 0.0f;
            if (col < dim && block_idx < blocks_per_row) {
                const uint8_t* w2_block = w2_blocks + ((static_cast<int64_t>(expert) * dim + col) * blocks_per_row + block_idx) * 56;
                iq1m_group_meta(w2_block, s, dl, d0, d1);
            }
            w2_dl[n] = dl; w2_delta0[n] = d0; w2_delta1[n] = d1;
        }
        __syncthreads();
        wmma::load_matrix_sync(a_frag, a_tile, TILE);
        wmma::load_matrix_sync(b_frag, b_tile, TILE);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        wmma::store_matrix_sync(c_tile_warp, c_frag, TILE, wmma::mem_row_major);
        #pragma unroll
        for (int i = lane; i < TILE * TILE; i += 32) {
            const int r = i / TILE;
            const int n = i & (TILE - 1);
            const int row = row_base + r;
            const int route = seg_starts[expert] + row;
            const int col = col_base + n;
            if (row < count && col < dim) {
                const float hs = hidden_scale[static_cast<int64_t>(route) * hidden_groups + kg];
                const float corrected = static_cast<float>(c_tile_warp[i]) + w2_delta0[n] * static_cast<float>(sum0[r]) + w2_delta1[n] * static_cast<float>(sum1[r]);
                acc[i] += hs * w2_dl[n] * corrected;
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = lane; i < TILE * TILE; i += 32) {
        const int r = i / TILE;
        const int n = i & (TILE - 1);
        const int row = row_base + r;
        const int col = col_base + n;
        if (row < count && col < dim) {
            const int route = seg_starts[expert] + row;
            const int64_t token = route_tokens[route];
            if (token >= 0) {
                atomicAdd(y_rows + token * static_cast<int64_t>(dim) + col, acc[i]);
            }
        }
    }
}

__global__ void iq1_moe_grouped_w13_swiglu_tile4_kernel(
    const float* __restrict__ x_rows,
    const int64_t* __restrict__ route_tokens,
    const float* __restrict__ route_weights,
    const int32_t* __restrict__ seg_starts,
    const uint8_t* __restrict__ w1_blocks,
    const uint8_t* __restrict__ w3_blocks,
    const int8_t* __restrict__ iq1_grid,
    float* __restrict__ hidden,
    int routes,
    int n_experts,
    int max_count,
    int dim,
    int inter_dim,
    int blocks_per_row,
    float swiglu_limit) {
    constexpr int route_tile = 4;
    const int expert = blockIdx.z;
    const int route_base_ordinal = blockIdx.y * route_tile;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int out_col = blockIdx.x * kIQ1TileN + warp;
    if (expert >= n_experts || route_base_ordinal >= max_count || warp >= kIQ1TileN || out_col >= inter_dim) return;
    const int count = seg_starts[expert + 1] - seg_starts[expert];
    const int base_route = seg_starts[expert] + route_base_ordinal;
    int route[route_tile];
    int64_t token[route_tile];
    bool valid[route_tile];
    #pragma unroll
    for (int r = 0; r < route_tile; ++r) {
        route[r] = base_route + r;
        valid[r] = (route_base_ordinal + r) < count && route[r] >= 0 && route[r] < routes;
        token[r] = valid[r] ? route_tokens[route[r]] : -1;
        valid[r] = valid[r] && token[r] >= 0;
    }

    __shared__ float x_shared[route_tile][256];
    const uint8_t* w1_row = w1_blocks + (static_cast<int64_t>(expert) * inter_dim + out_col) * blocks_per_row * 56;
    const uint8_t* w3_row = w3_blocks + (static_cast<int64_t>(expert) * inter_dim + out_col) * blocks_per_row * 56;
    float acc1[route_tile] = {0.0f, 0.0f, 0.0f, 0.0f};
    float acc3[route_tile] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
        const int k_base = block_idx * 256;
        for (int k = threadIdx.x; k < 256; k += blockDim.x) {
            const int idx = k_base + k;
            #pragma unroll
            for (int r = 0; r < route_tile; ++r) {
                const float* x = valid[r] ? (x_rows + token[r] * static_cast<int64_t>(dim)) : nullptr;
                x_shared[r][k] = (valid[r] && idx < dim) ? x[idx] : 0.0f;
            }
        }
        __syncthreads();
        const uint8_t* w1_block = w1_row + static_cast<int64_t>(block_idx) * 56;
        const uint8_t* w3_block = w3_row + static_cast<int64_t>(block_idx) * 56;
        #pragma unroll
        for (int r = 0; r < route_tile; ++r) {
            const float b1 = iq1m_block_dot_256(x_shared[r], w1_block, iq1_grid, lane);
            const float b3 = iq1m_block_dot_256(x_shared[r], w3_block, iq1_grid, lane);
            if (lane == 0) {
                acc1[r] += b1;
                acc3[r] += b3;
            }
        }
        __syncthreads();
    }
    if (lane == 0) {
        #pragma unroll
        for (int r = 0; r < route_tile; ++r) {
            if (!valid[r]) continue;
            float g = acc1[r];
            float u = acc3[r];
            if (swiglu_limit > 0.0f) {
                u = fminf(fmaxf(u, -swiglu_limit), swiglu_limit);
                g = fminf(g, swiglu_limit);
            }
            hidden[static_cast<int64_t>(route[r]) * inter_dim + out_col] =
                (g / (1.0f + expf(-g))) * u * route_weights[route[r]];
        }
    }
}

__global__ void iq1_moe_grouped_w2_tile4_kernel(
    const float* __restrict__ hidden,
    const int64_t* __restrict__ route_tokens,
    const int32_t* __restrict__ seg_starts,
    const uint8_t* __restrict__ w2_blocks,
    const int8_t* __restrict__ iq1_grid,
    float* __restrict__ y_rows,
    int routes,
    int n_experts,
    int max_count,
    int dim,
    int inter_dim,
    int blocks_per_row) {
    constexpr int route_tile = 4;
    const int expert = blockIdx.z;
    const int route_base_ordinal = blockIdx.y * route_tile;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int out_col = blockIdx.x * kIQ1TileN + warp;
    if (expert >= n_experts || route_base_ordinal >= max_count || warp >= kIQ1TileN || out_col >= dim) return;
    const int count = seg_starts[expert + 1] - seg_starts[expert];
    const int base_route = seg_starts[expert] + route_base_ordinal;
    int route[route_tile];
    int64_t token[route_tile];
    bool valid[route_tile];
    #pragma unroll
    for (int r = 0; r < route_tile; ++r) {
        route[r] = base_route + r;
        valid[r] = (route_base_ordinal + r) < count && route[r] >= 0 && route[r] < routes;
        token[r] = valid[r] ? route_tokens[route[r]] : -1;
        valid[r] = valid[r] && token[r] >= 0;
    }

    __shared__ float h_shared[route_tile][256];
    const uint8_t* w2_row = w2_blocks + (static_cast<int64_t>(expert) * dim + out_col) * blocks_per_row * 56;
    float acc[route_tile] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
        const int k_base = block_idx * 256;
        for (int k = threadIdx.x; k < 256; k += blockDim.x) {
            const int idx = k_base + k;
            #pragma unroll
            for (int r = 0; r < route_tile; ++r) {
                const float* hidden_row = valid[r] ? (hidden + static_cast<int64_t>(route[r]) * inter_dim) : nullptr;
                h_shared[r][k] = (valid[r] && idx < inter_dim) ? hidden_row[idx] : 0.0f;
            }
        }
        __syncthreads();
        const uint8_t* w2_block = w2_row + static_cast<int64_t>(block_idx) * 56;
        #pragma unroll
        for (int r = 0; r < route_tile; ++r) {
            const float b = iq1m_block_dot_256(h_shared[r], w2_block, iq1_grid, lane);
            if (lane == 0) acc[r] += b;
        }
        __syncthreads();
    }
    if (lane == 0) {
        #pragma unroll
        for (int r = 0; r < route_tile; ++r) {
            if (valid[r]) atomicAdd(y_rows + token[r] * static_cast<int64_t>(dim) + out_col, acc[r]);
        }
    }
}

}  // namespace

bool iq1_moe_single_w13_cuda(
    const float* d_x,
    const int64_t* d_route_slots,
    const uint8_t* d_w1_blocks,
    const uint8_t* d_w3_blocks,
    float* d_gate,
    float* d_up,
    int routes,
    int n_experts,
    int dim,
    int inter_dim,
    void* stream) {
    if (routes <= 0 || n_experts <= 0 || dim <= 0 || inter_dim <= 0) return true;
    const int8_t* grid = iq1_grid_device();
    if (grid == nullptr) return false;
    const int blocks_per_row = (dim + 255) / 256;
    dim3 grid_dim(ceil_div(inter_dim, kIQ1TileN), routes);
    dim3 block(256);
    iq1_moe_single_w13_kernel<<<grid_dim, block, 0, static_cast<cudaStream_t>(stream)>>>(
        d_x, d_route_slots, d_w1_blocks, d_w3_blocks, grid, d_gate, d_up,
        routes, n_experts, dim, inter_dim, blocks_per_row);
    return cudaGetLastError() == cudaSuccess;
}

bool iq1_moe_single_w13_swiglu_cuda(
    const float* d_x,
    const int64_t* d_route_slots,
    const float* d_route_weights,
    const uint8_t* d_w1_blocks,
    const uint8_t* d_w3_blocks,
    float* d_hidden,
    int routes,
    int n_experts,
    int dim,
    int inter_dim,
    float swiglu_limit,
    void* stream) {
    if (routes <= 0 || n_experts <= 0 || dim <= 0 || inter_dim <= 0) return true;
    const int8_t* grid = iq1_grid_device();
    if (grid == nullptr) return false;
    const int blocks_per_row = (dim + 255) / 256;
    dim3 grid_dim(ceil_div(inter_dim, kIQ1TileN), routes);
    dim3 block(256);
    iq1_moe_single_w13_swiglu_kernel<<<grid_dim, block, 0, static_cast<cudaStream_t>(stream)>>>(
        d_x, d_route_slots, d_route_weights, d_w1_blocks, d_w3_blocks, grid, d_hidden,
        routes, n_experts, dim, inter_dim, blocks_per_row, swiglu_limit);
    return cudaGetLastError() == cudaSuccess;
}

bool iq1_route_swiglu_cuda(
    const float* d_gate,
    const float* d_up,
    const float* d_route_weights,
    float* d_hidden,
    int routes,
    int inter_dim,
    float swiglu_limit,
    void* stream) {
    const int total = routes * inter_dim;
    if (total <= 0) return true;
    const int threads = 256;
    const int blocks = ceil_div(total, threads);
    iq1_route_swiglu_kernel<<<blocks, threads, 0, static_cast<cudaStream_t>(stream)>>>(
        d_gate, d_up, d_route_weights, d_hidden, routes, inter_dim, swiglu_limit);
    return cudaGetLastError() == cudaSuccess;
}

bool iq1_moe_single_w2_cuda(
    const float* d_hidden,
    const int64_t* d_route_slots,
    const uint8_t* d_w2_blocks,
    float* d_y,
    int routes,
    int n_experts,
    int dim,
    int inter_dim,
    void* stream) {
    if (routes <= 0 || n_experts <= 0 || dim <= 0 || inter_dim <= 0) return true;
    const int8_t* grid = iq1_grid_device();
    if (grid == nullptr) return false;
    const int blocks_per_row = (inter_dim + 255) / 256;
    dim3 grid_dim(ceil_div(dim, kIQ1TileN), routes);
    dim3 block(256);
    iq1_moe_single_w2_kernel<<<grid_dim, block, 0, static_cast<cudaStream_t>(stream)>>>(
        d_hidden, d_route_slots, d_w2_blocks, grid, d_y,
        routes, n_experts, dim, inter_dim, blocks_per_row);
    return cudaGetLastError() == cudaSuccess;
}

bool iq1_moe_single_w2_reduce_cuda(
    const float* d_hidden,
    const int64_t* d_route_slots,
    const uint8_t* d_w2_blocks,
    float* d_y,
    int routes,
    int n_experts,
    int dim,
    int inter_dim,
    void* stream) {
    if (routes <= 0 || n_experts <= 0 || dim <= 0 || inter_dim <= 0) return true;
    const int8_t* grid = iq1_grid_device();
    if (grid == nullptr) return false;
    const int blocks_per_row = (inter_dim + 255) / 256;
    dim3 grid_dim(ceil_div(dim, kIQ1TileN));
    dim3 block(256);
    iq1_moe_single_w2_reduce_kernel<<<grid_dim, block, 0, static_cast<cudaStream_t>(stream)>>>(
        d_hidden, d_route_slots, d_w2_blocks, grid, d_y,
        routes, n_experts, dim, inter_dim, blocks_per_row);
    return cudaGetLastError() == cudaSuccess;
}

bool iq1_moe_grouped_w13_swiglu_cuda(
    const float* d_x_rows,
    const int64_t* d_route_tokens,
    const float* d_route_weights,
    const int32_t* d_seg_starts,
    const uint8_t* d_w1_blocks,
    const uint8_t* d_w3_blocks,
    float* d_hidden,
    int routes,
    int n_experts,
    int max_count,
    int dim,
    int inter_dim,
    float swiglu_limit,
    void* stream) {
    if (d_x_rows == nullptr || d_route_tokens == nullptr || d_route_weights == nullptr ||
        d_seg_starts == nullptr || d_w1_blocks == nullptr || d_w3_blocks == nullptr ||
        d_hidden == nullptr) return false;
    if (routes <= 0 || n_experts <= 0 || max_count <= 0 || dim <= 0 || inter_dim <= 0) return true;
    const int8_t* grid = iq1_grid_device();
    if (grid == nullptr) return false;
    const int w13_blocks_per_row = (dim + 255) / 256;
    dim3 block(256);
    const bool route_tile4 = local_env_int_or_default("POCKETLLM_IQ1_GROUPED_ROUTE_TILE4", 0) != 0 && max_count >= 4;
    const bool route_major = !route_tile4 && local_env_int_or_default("POCKETLLM_IQ1_GROUPED_ROUTE_MAJOR", 1) != 0;
    if (route_tile4) {
        constexpr int route_tile = 4;
        dim3 grid_w13(ceil_div(inter_dim, kIQ1TileN), ceil_div(max_count, route_tile), n_experts);
        iq1_moe_grouped_w13_swiglu_tile4_kernel<<<grid_w13, block, 0, static_cast<cudaStream_t>(stream)>>>(
            d_x_rows, d_route_tokens, d_route_weights, d_seg_starts,
            d_w1_blocks, d_w3_blocks, grid, d_hidden,
            routes, n_experts, max_count, dim, inter_dim, w13_blocks_per_row, swiglu_limit);
    } else if (route_major) {
        dim3 grid_w13(ceil_div(inter_dim, kIQ1TileN), routes);
        iq1_moe_grouped_w13_swiglu_route_kernel<<<grid_w13, block, 0, static_cast<cudaStream_t>(stream)>>>(
            d_x_rows, d_route_tokens, d_route_weights, d_seg_starts,
            d_w1_blocks, d_w3_blocks, grid, d_hidden,
            routes, n_experts, dim, inter_dim, w13_blocks_per_row, swiglu_limit);
    } else {
        dim3 grid_w13(ceil_div(inter_dim, kIQ1TileN), max_count, n_experts);
        iq1_moe_grouped_w13_swiglu_kernel<<<grid_w13, block, 0, static_cast<cudaStream_t>(stream)>>>(
            d_x_rows, d_route_tokens, d_route_weights, d_seg_starts,
            d_w1_blocks, d_w3_blocks, grid, d_hidden,
            routes, n_experts, dim, inter_dim, w13_blocks_per_row, swiglu_limit);
    }
    return cudaGetLastError() == cudaSuccess;
}

bool moe_prefill_iq1_grouped_cuda_with_workspace(
    const float* d_x_rows,
    const int64_t* d_route_tokens,
    const float* d_route_weights,
    const int32_t* d_seg_starts,
    const uint8_t* d_w1_blocks,
    const uint8_t* d_w2_blocks,
    const uint8_t* d_w3_blocks,
    float* d_y_rows,
    int tokens,
    int routes,
    int n_experts,
    int max_count,
    int dim,
    int inter_dim,
    float swiglu_limit,
    MoePrefillIq1GroupedWorkspace workspace,
    void* stream) {
    if (d_x_rows == nullptr || d_route_tokens == nullptr || d_route_weights == nullptr ||
        d_seg_starts == nullptr || d_w1_blocks == nullptr || d_w2_blocks == nullptr ||
        d_w3_blocks == nullptr || d_y_rows == nullptr) return false;
    if (tokens <= 0 || routes < 0 || n_experts <= 0 || max_count < 0 || dim <= 0 || inter_dim <= 0) return false;
    if (routes == 0 || max_count == 0) {
        return cudaMemsetAsync(d_y_rows, 0, static_cast<size_t>(tokens) * dim * sizeof(float),
                              static_cast<cudaStream_t>(stream)) == cudaSuccess;
    }
    if (workspace.d_hidden == nullptr || workspace.routes_cap < routes || workspace.inter_dim < inter_dim) return false;
    const int8_t* grid = iq1_grid_device();
    if (grid == nullptr) return false;
    cudaStream_t cs = static_cast<cudaStream_t>(stream);
    const bool profile_iq1 = local_env_int_or_default("POCKETLLM_IQ1_GROUPED_PROFILE", 0) != 0;
    cudaEvent_t ev0 = nullptr, ev1 = nullptr, ev2 = nullptr, ev3 = nullptr, ev4 = nullptr;
    if (profile_iq1) {
        cudaEventCreate(&ev0); cudaEventCreate(&ev1); cudaEventCreate(&ev2); cudaEventCreate(&ev3); cudaEventCreate(&ev4);
        cudaEventRecord(ev0, cs);
    }
    if (cudaMemsetAsync(d_y_rows, 0, static_cast<size_t>(tokens) * dim * sizeof(float), cs) != cudaSuccess) return false;

    const int w13_blocks_per_row = (dim + 255) / 256;
    const int w2_blocks_per_row = (inter_dim + 255) / 256;
    const int x_groups16 = ceil_div(dim, 16);
    const int hidden_groups = ceil_div(inter_dim, 16);
    dim3 block(256);
    const bool route_tile4 = local_env_int_or_default("POCKETLLM_IQ1_GROUPED_ROUTE_TILE4", 0) != 0 && max_count >= 4;
    const bool route_major = !route_tile4 && local_env_int_or_default("POCKETLLM_IQ1_GROUPED_ROUTE_MAJOR", 1) != 0;
    const bool q8_w2 = local_env_int_or_default("POCKETLLM_IQ1_GROUPED_W2_Q8", 1) != 0 &&
        workspace.d_hidden_q != nullptr && workspace.d_hidden_scale != nullptr;
    const bool tile_workspace = workspace.d_tile_experts != nullptr && workspace.d_tile_rows != nullptr &&
        workspace.tile_count > 0 && workspace.tile_cap >= workspace.tile_count && workspace.routes_cap >= routes;
    const bool gemm_enabled = local_env_int_or_default("POCKETLLM_IQ1_GROUPED_GEMM", 1) != 0 && tile_workspace;
    const bool gemm_w13 = gemm_enabled && local_env_int_or_default("POCKETLLM_IQ1_GROUPED_GEMM_W13", 1) != 0 &&
        workspace.d_x_q != nullptr && workspace.d_x_scale != nullptr && workspace.dim >= dim;
    const bool gemm_w2 = gemm_enabled && local_env_int_or_default("POCKETLLM_IQ1_GROUPED_GEMM_W2", 1) != 0 && q8_w2;
    const int nsplit = local_env_int_or_default("POCKETLLM_IQ1_GROUPED_GEMM_NSPLIT", 4);

    if (gemm_w13) {
        dim3 q_grid(routes, x_groups16);
        iq1_quantize_route_rows_q8_16_kernel<<<q_grid, 16, 0, cs>>>(
            d_x_rows, d_route_tokens, workspace.d_x_q, workspace.d_x_scale, routes, dim, x_groups16);
        if (cudaGetLastError() != cudaSuccess) return false;
        if (profile_iq1) cudaEventRecord(ev1, cs);
        if (nsplit == 2) {
            dim3 grid_w13(ceil_div(inter_dim, kIQ1WmmaTile * 2), workspace.tile_count);
            iq1_moe_grouped_w13_q8_wmma_compact_nsplit_kernel<2><<<grid_w13, 64, IQ1_W13_SHARED_NSPLIT_BYTES(2), cs>>>(
                workspace.d_x_q, workspace.d_x_scale, d_route_weights, d_seg_starts,
                workspace.d_tile_experts, workspace.d_tile_rows, d_w1_blocks, d_w3_blocks, grid,
                workspace.d_hidden, workspace.tile_count, dim, inter_dim, w13_blocks_per_row, x_groups16, swiglu_limit);
        } else {
            dim3 grid_w13(ceil_div(inter_dim, kIQ1WmmaTile * 4), workspace.tile_count);
            iq1_moe_grouped_w13_q8_wmma_compact_nsplit_kernel<4><<<grid_w13, 128, IQ1_W13_SHARED_NSPLIT_BYTES(4), cs>>>(
                workspace.d_x_q, workspace.d_x_scale, d_route_weights, d_seg_starts,
                workspace.d_tile_experts, workspace.d_tile_rows, d_w1_blocks, d_w3_blocks, grid,
                workspace.d_hidden, workspace.tile_count, dim, inter_dim, w13_blocks_per_row, x_groups16, swiglu_limit);
        }
    } else if (route_tile4) {
        constexpr int route_tile = 4;
        dim3 grid_w13(ceil_div(inter_dim, kIQ1TileN), ceil_div(max_count, route_tile), n_experts);
        iq1_moe_grouped_w13_swiglu_tile4_kernel<<<grid_w13, block, 0, cs>>>(
            d_x_rows, d_route_tokens, d_route_weights, d_seg_starts,
            d_w1_blocks, d_w3_blocks, grid, workspace.d_hidden,
            routes, n_experts, max_count, dim, inter_dim, w13_blocks_per_row, swiglu_limit);
    } else if (route_major) {
        dim3 grid_w13(ceil_div(inter_dim, kIQ1TileN), routes);
        iq1_moe_grouped_w13_swiglu_route_kernel<<<grid_w13, block, 0, cs>>>(
            d_x_rows, d_route_tokens, d_route_weights, d_seg_starts,
            d_w1_blocks, d_w3_blocks, grid, workspace.d_hidden,
            routes, n_experts, dim, inter_dim, w13_blocks_per_row, swiglu_limit);
    } else {
        dim3 grid_w13(ceil_div(inter_dim, kIQ1TileN), max_count, n_experts);
        iq1_moe_grouped_w13_swiglu_kernel<<<grid_w13, block, 0, cs>>>(
            d_x_rows, d_route_tokens, d_route_weights, d_seg_starts,
            d_w1_blocks, d_w3_blocks, grid, workspace.d_hidden,
            routes, n_experts, dim, inter_dim, w13_blocks_per_row, swiglu_limit);
    }
    if (cudaGetLastError() != cudaSuccess) return false;
    if (profile_iq1) cudaEventRecord(ev2, cs);

    if (q8_w2) {
        if (!q2_quantize_hidden_q8_1_cuda(workspace.d_hidden, workspace.d_hidden_q, workspace.d_hidden_scale, routes, inter_dim, cs)) return false;
        if (profile_iq1) cudaEventRecord(ev3, cs);
        if (gemm_w2) {
            if (nsplit == 2) {
                dim3 grid_w2(ceil_div(dim, kIQ1WmmaTile * 2), workspace.tile_count);
                iq1_moe_grouped_w2_q8_wmma_compact_nsplit_kernel<2><<<grid_w2, 64, IQ1_W2_SHARED_NSPLIT_BYTES(2), cs>>>(
                    workspace.d_hidden_q, workspace.d_hidden_scale, d_route_tokens, d_seg_starts,
                    workspace.d_tile_experts, workspace.d_tile_rows, d_w2_blocks, grid, d_y_rows,
                    workspace.tile_count, dim, inter_dim, w2_blocks_per_row, hidden_groups);
            } else {
                dim3 grid_w2(ceil_div(dim, kIQ1WmmaTile * 4), workspace.tile_count);
                iq1_moe_grouped_w2_q8_wmma_compact_nsplit_kernel<4><<<grid_w2, 128, IQ1_W2_SHARED_NSPLIT_BYTES(4), cs>>>(
                    workspace.d_hidden_q, workspace.d_hidden_scale, d_route_tokens, d_seg_starts,
                    workspace.d_tile_experts, workspace.d_tile_rows, d_w2_blocks, grid, d_y_rows,
                    workspace.tile_count, dim, inter_dim, w2_blocks_per_row, hidden_groups);
            }
        } else if (route_major) {
            dim3 grid_w2(ceil_div(dim, kIQ1TileN), routes);
            iq1_moe_grouped_w2_q8_route_kernel<<<grid_w2, block, 0, cs>>>(
                workspace.d_hidden_q, workspace.d_hidden_scale, d_route_tokens, d_seg_starts, d_w2_blocks, grid, d_y_rows,
                routes, n_experts, dim, inter_dim, w2_blocks_per_row, hidden_groups);
        } else {
            dim3 grid_w2(ceil_div(dim, kIQ1TileN), max_count, n_experts);
            iq1_moe_grouped_w2_q8_kernel<<<grid_w2, block, 0, cs>>>(
                workspace.d_hidden_q, workspace.d_hidden_scale, d_route_tokens, d_seg_starts, d_w2_blocks, grid, d_y_rows,
                routes, n_experts, max_count, dim, inter_dim, w2_blocks_per_row, hidden_groups);
        }
    } else {
        if (profile_iq1) cudaEventRecord(ev3, cs);
        if (route_tile4) {
            constexpr int route_tile = 4;
            dim3 grid_w2(ceil_div(dim, kIQ1TileN), ceil_div(max_count, route_tile), n_experts);
            iq1_moe_grouped_w2_tile4_kernel<<<grid_w2, block, 0, cs>>>(
                workspace.d_hidden, d_route_tokens, d_seg_starts, d_w2_blocks, grid, d_y_rows,
                routes, n_experts, max_count, dim, inter_dim, w2_blocks_per_row);
        } else if (route_major) {
            dim3 grid_w2(ceil_div(dim, kIQ1TileN), routes);
            iq1_moe_grouped_w2_route_kernel<<<grid_w2, block, 0, cs>>>(
                workspace.d_hidden, d_route_tokens, d_seg_starts, d_w2_blocks, grid, d_y_rows,
                routes, n_experts, dim, inter_dim, w2_blocks_per_row);
        } else {
            dim3 grid_w2(ceil_div(dim, kIQ1TileN), max_count, n_experts);
            iq1_moe_grouped_w2_kernel<<<grid_w2, block, 0, cs>>>(
                workspace.d_hidden, d_route_tokens, d_seg_starts, d_w2_blocks, grid, d_y_rows,
                routes, n_experts, dim, inter_dim, w2_blocks_per_row);
        }
    }
    const cudaError_t last = cudaGetLastError();
    if (profile_iq1) {
        cudaEventRecord(ev4, cs);
        cudaEventSynchronize(ev4);
        float total_ms = 0.0f, x_quant_ms = 0.0f, w13_ms = 0.0f, quant_ms = 0.0f, w2_ms = 0.0f;
        cudaEventElapsedTime(&total_ms, ev0, ev4);
        if (gemm_w13) {
            cudaEventElapsedTime(&x_quant_ms, ev0, ev1);
            cudaEventElapsedTime(&w13_ms, ev1, ev2);
        } else {
            cudaEventElapsedTime(&w13_ms, ev0, ev2);
        }
        cudaEventElapsedTime(&quant_ms, ev2, ev3);
        cudaEventElapsedTime(&w2_ms, ev3, ev4);
        std::cerr << "iq1_grouped_profile routes=" << routes
                  << " tokens=" << tokens
                  << " max_count=" << max_count
                  << " n_experts=" << n_experts
                  << " dim=" << dim
                  << " inter_dim=" << inter_dim
                  << " q8_w2=" << (q8_w2 ? 1 : 0)
                  << " gemm=" << ((gemm_w13 || gemm_w2) ? 1 : 0)
                  << " gemm_w13=" << (gemm_w13 ? 1 : 0)
                  << " gemm_w2=" << (gemm_w2 ? 1 : 0)
                  << " tiles=" << workspace.tile_count
                  << " nsplit=" << (nsplit == 2 ? 2 : 4)
                  << " tile4=" << (route_tile4 ? 1 : 0)
                  << " route_major=" << (route_major ? 1 : 0)
                  << " hidden_q_present=" << (workspace.d_hidden_q != nullptr ? 1 : 0)
                  << " hidden_scale_present=" << (workspace.d_hidden_scale != nullptr ? 1 : 0)
                  << " total_ms=" << total_ms
                  << " x_quant_ms=" << x_quant_ms
                  << " w13_ms=" << w13_ms
                  << " quant_ms=" << quant_ms
                  << " w2_ms=" << w2_ms << "\n";
        cudaEventDestroy(ev0); cudaEventDestroy(ev1); cudaEventDestroy(ev2); cudaEventDestroy(ev3); cudaEventDestroy(ev4);
    }
    return last == cudaSuccess;
}

}  // namespace pocket
