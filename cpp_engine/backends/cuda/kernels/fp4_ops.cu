#include "cuda_ops.hpp"

#include <chrono>
#include <cstdlib>
#include <iostream>

#include <cuda_runtime.h>
#include <mma.h>
#include <sm_61_intrinsics.h>

namespace pocket {
namespace {

constexpr int kQuantThreads = 256;
constexpr int kGemmThreads = 128;

// shared-memory bytes per block for the N-split compact kernels.
// w13: a_tile (16*16 i8) + 2*WARPS b_tile (i8) + WARPS c_tile (int) + 2*WARPS acc (float).
// w2 : a_tile (16*16 i8) +   WARPS b_tile (i8) + WARPS c_tile (int) +   WARPS acc (float).
constexpr size_t TILE_SHARED_NSPLIT_BYTES_W13(int W) {
    return 16 * 16 * 1
         + 2 * static_cast<size_t>(W) * 16 * 16 * 1
         + static_cast<size_t>(W) * 16 * 16 * sizeof(int)
         + 2 * static_cast<size_t>(W) * 16 * 16 * sizeof(float);
}
constexpr size_t TILE_SHARED_NSPLIT_BYTES_W2(int W) {
    return 16 * 16 * 1
         + static_cast<size_t>(W) * 16 * 16 * 1
         + static_cast<size_t>(W) * 16 * 16 * sizeof(int)
         + static_cast<size_t>(W) * 16 * 16 * sizeof(float);
}

__device__ __constant__ int8_t fp4_lut_x2[16] = {
    0, 1, 2, 3, 4, 6, 8, 12,
    0, -1, -2, -3, -4, -6, -8, -12
};

__host__ __device__ __forceinline__ int ceil_div_int(int a, int b) {
    return (a + b - 1) / b;
}

__device__ __forceinline__ int fp4_unpack_4codes_prmt(uint32_t two_bytes) {
    const uint32_t control_mags = two_bytes & 0x7777u;
    const uint32_t pos = __byte_perm(0x03020100u, 0x0c080604u, control_mags);
    const uint32_t neg = __byte_perm(0xfdfeff00u, 0xf4f8fafcu, control_mags);
    const uint32_t control_sign = (two_bytes >> 3) & 0x1111u;
    const uint32_t mask = __byte_perm(0x0000ff00u, 0x00000000u, control_sign);
    return static_cast<int>((pos & ~mask) | (neg & mask));
}

__device__ __forceinline__ int dot_i8x4(int a, int b, int acc) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610
    return __dp4a(a, b, acc);
#else
    const int8_t a0 = static_cast<int8_t>(a & 0xff);
    const int8_t a1 = static_cast<int8_t>((a >> 8) & 0xff);
    const int8_t a2 = static_cast<int8_t>((a >> 16) & 0xff);
    const int8_t a3 = static_cast<int8_t>((a >> 24) & 0xff);
    const int8_t b0 = static_cast<int8_t>(b & 0xff);
    const int8_t b1 = static_cast<int8_t>((b >> 8) & 0xff);
    const int8_t b2 = static_cast<int8_t>((b >> 16) & 0xff);
    const int8_t b3 = static_cast<int8_t>((b >> 24) & 0xff);
    return acc + static_cast<int>(a0) * b0 + static_cast<int>(a1) * b1 + static_cast<int>(a2) * b2 + static_cast<int>(a3) * b3;
#endif
}

__device__ __forceinline__ float fp4_block_scale(uint8_t scale_byte) {
    const int exponent = max(0, static_cast<int>(scale_byte) - 1);
    return __int_as_float(exponent << 23);
}

__global__ void quantize_float_row_kernel(
    const float* __restrict__ x,
    int8_t* __restrict__ x_q,
    float* __restrict__ x_scale,
    int k) {
    const int tid = threadIdx.x;
    __shared__ float sdata[kQuantThreads];
    float local_max = 0.0f;
    for (int idx = tid; idx < k; idx += blockDim.x) {
        local_max = fmaxf(local_max, fabsf(x[idx]));
    }
    sdata[tid] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        __syncthreads();
    }
    const float scale = fmaxf(sdata[0], 1.0e-6f) / 127.0f;
    if (tid == 0) x_scale[0] = scale;
    __syncthreads();
    const float inv_scale = 1.0f / scale;
    for (int idx = tid; idx < k; idx += blockDim.x) {
        int q = __float2int_rn(x[idx] * inv_scale);
        q = max(-127, min(127, q));
        x_q[idx] = static_cast<int8_t>(q);
    }
}

__global__ void moe_route_count_kernel(
    const int64_t* __restrict__ indices,
    int32_t* __restrict__ counts,
    int tokens,
    int topk,
    int experts_start_idx,
    int n_local_experts) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = tokens * topk;
    if (idx >= total) return;
    const int expert = static_cast<int>(indices[idx]);
    const int local = expert - experts_start_idx;
    if (local >= 0 && local < n_local_experts) atomicAdd(counts + local, 1);
}

__global__ void moe_route_prefix_kernel(
    const int32_t* __restrict__ counts,
    int32_t* __restrict__ seg_starts,
    int32_t* __restrict__ offsets,
    int32_t* __restrict__ total_routes,
    int n_local_experts) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    int32_t total = 0;
    seg_starts[0] = 0;
    for (int e = 0; e < n_local_experts; ++e) {
        offsets[e] = total;
        total += counts[e];
        seg_starts[e + 1] = total;
    }
    total_routes[0] = total;
}

// `token_slot_routes` is the inverse of route_tokens: for token t and router
// slot k, it holds the route id that landed there (or -1 when that slot's
// expert is not local to this rank). The fill order below is atomic and so
// varies run to run, but (t, k) does not -- which is what lets the deterministic
// reduction sum a token's contributions in a fixed order. Pass nullptr to skip.
__global__ void moe_route_fill_kernel(
    const int64_t* __restrict__ indices,
    const float* __restrict__ weights,
    int32_t* __restrict__ offsets,
    int64_t* __restrict__ route_tokens,
    float* __restrict__ route_weights,
    int32_t* __restrict__ token_slot_routes,
    int tokens,
    int topk,
    int experts_start_idx,
    int n_local_experts) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = tokens * topk;
    if (idx >= total) return;
    const int expert = static_cast<int>(indices[idx]);
    const int local = expert - experts_start_idx;
    if (local < 0 || local >= n_local_experts) return;
    const int out = atomicAdd(offsets + local, 1);
    route_tokens[out] = static_cast<int64_t>(idx / topk);
    route_weights[out] = weights[idx];
    if (token_slot_routes != nullptr) token_slot_routes[idx] = out;
}

__global__ void gather_routes_float_kernel(
    const float* __restrict__ x,
    const int64_t* __restrict__ route_tokens,
    float* __restrict__ x_sorted,
    int routes,
    int topk,
    int dim) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = routes * dim;
    if (idx >= total) return;
    const int route = idx / dim;
    const int d = idx - route * dim;
    const int64_t token = route_tokens[route];
    x_sorted[idx] = x[static_cast<size_t>(token) * dim + d];
}

__global__ void quantize_float_rows_kernel(
    const float* __restrict__ x,
    int8_t* __restrict__ x_q,
    float* __restrict__ x_scale,
    int rows,
    int k) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    if (row >= rows) return;
    const int base = row * k;
    __shared__ float sdata[kQuantThreads];
    float local_max = 0.0f;
    for (int idx = tid; idx < k; idx += blockDim.x) {
        local_max = fmaxf(local_max, fabsf(x[base + idx]));
    }
    sdata[tid] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        __syncthreads();
    }
    const float scale = fmaxf(sdata[0], 1.0e-6f) / 127.0f;
    if (tid == 0) x_scale[row] = scale;
    __syncthreads();
    const float inv_scale = 1.0f / scale;
    for (int idx = tid; idx < k; idx += blockDim.x) {
        int q = __float2int_rn(x[base + idx] * inv_scale);
        q = max(-127, min(127, q));
        x_q[base + idx] = static_cast<int8_t>(q);
    }
}

__global__ void pad_q_rows_kernel(
    const int8_t* __restrict__ src_q,
    const float* __restrict__ src_scale,
    const int32_t* __restrict__ seg_starts,
    int8_t* __restrict__ dst_q,
    float* __restrict__ dst_scale,
    int n_experts,
    int rows_per_expert,
    int k) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = n_experts * rows_per_expert * k;
    if (idx >= total) return;
    const int col = idx % k;
    const int row = (idx / k) % rows_per_expert;
    const int expert = idx / (rows_per_expert * k);
    const int count = seg_starts[expert + 1] - seg_starts[expert];
    const int dst = (expert * rows_per_expert + row) * k + col;
    if (row < count) {
        const int route = seg_starts[expert] + row;
        dst_q[dst] = src_q[route * k + col];
        if (col == 0) dst_scale[expert * rows_per_expert + row] = src_scale[route];
    } else {
        dst_q[dst] = 0;
        if (col == 0) dst_scale[expert * rows_per_expert + row] = 0.0f;
    }
}

__device__ __forceinline__ float fp4_e2m1_value(uint8_t code) {
    constexpr float table[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
    };
    return table[code & 0x0f];
}

__device__ __forceinline__ float e8m0_value(uint8_t code) {
    return exp2f(static_cast<float>(static_cast<int>(code) - 127));
}

__global__ void fp4_e2m1_e8m0_matvec_kernel(
    const float* __restrict__ x,
    const uint8_t* __restrict__ weight,
    const uint8_t* __restrict__ scale,
    float* __restrict__ y,
    int rows,
    int cols,
    int packed_cols,
    int scale_cols) {
    const int row = blockIdx.x;
    if (row >= rows) return;

    float sum = 0.0f;
    for (int col = threadIdx.x; col < cols; col += blockDim.x) {
        const uint8_t packed = weight[static_cast<size_t>(row) * packed_cols + col / 2];
        const uint8_t code = (col & 1) ? static_cast<uint8_t>(packed >> 4) : static_cast<uint8_t>(packed & 0x0f);
        const float s = e8m0_value(scale[static_cast<size_t>(row) * scale_cols + col / 32]);
        sum += fp4_e2m1_value(code) * s * x[col];
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

__global__ void moe_single_w1w3_fp4_kernel(
    const int8_t* __restrict__ x_q,
    const float* __restrict__ x_scale,
    const int64_t* __restrict__ indices,
    const uint8_t* __restrict__ w1q,
    const uint8_t* __restrict__ w1s,
    const uint8_t* __restrict__ w3q,
    const uint8_t* __restrict__ w3s,
    float* __restrict__ gate_f32,
    float* __restrict__ up_f32,
    int topk,
    int experts_start_idx,
    int n_local_experts,
    int dim,
    int inter_dim) {
    const int route = blockIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (route >= topk) return;
    const int local = static_cast<int>(indices[route]) - experts_start_idx;
    const int dim_packs = dim / 4;
    const int blocks_k = dim / 32;
    extern __shared__ int x_shared[];
    const int* x_i32 = reinterpret_cast<const int*>(x_q);
    for (int idx = threadIdx.x; idx < dim_packs; idx += blockDim.x) x_shared[idx] = x_i32[idx];
    __syncthreads();
    if (col >= inter_dim) return;
    if (local < 0 || local >= n_local_experts) {
        gate_f32[route * inter_dim + col] = 0.0f;
        up_f32[route * inter_dim + col] = 0.0f;
        return;
    }
    const uint8_t* w1_row_bytes = w1q + (static_cast<int64_t>(local) * inter_dim + col) * (dim / 2);
    const uint8_t* w3_row_bytes = w3q + (static_cast<int64_t>(local) * inter_dim + col) * (dim / 2);
    const uint8_t* w1_scale_row = w1s + (static_cast<int64_t>(local) * inter_dim + col) * blocks_k;
    const uint8_t* w3_scale_row = w3s + (static_cast<int64_t>(local) * inter_dim + col) * blocks_k;
    const uint16_t* w1_pack_base = reinterpret_cast<const uint16_t*>(w1_row_bytes);
    const uint16_t* w3_pack_base = reinterpret_cast<const uint16_t*>(w3_row_bytes);
    float gate_acc = 0.0f;
    float up_acc = 0.0f;
    for (int kb = 0; kb < blocks_k; ++kb) {
        const uint16_t* w1_pack = w1_pack_base + kb * 8;
        const uint16_t* w3_pack = w3_pack_base + kb * 8;
        const int* x_pack = x_shared + kb * 8;
        int gate_block = 0;
        int up_block = 0;
        #pragma unroll
        for (int ip = 0; ip < 8; ++ip) {
            const int w1_p = fp4_unpack_4codes_prmt(static_cast<uint32_t>(w1_pack[ip]));
            const int w3_p = fp4_unpack_4codes_prmt(static_cast<uint32_t>(w3_pack[ip]));
            const int x_p = x_pack[ip];
            gate_block = dot_i8x4(x_p, w1_p, gate_block);
            up_block = dot_i8x4(x_p, w3_p, up_block);
        }
        gate_acc += static_cast<float>(gate_block) * fp4_block_scale(w1_scale_row[kb]);
        up_acc += static_cast<float>(up_block) * fp4_block_scale(w3_scale_row[kb]);
    }
    const float xs = x_scale[0];
    gate_f32[route * inter_dim + col] = gate_acc * xs;
    up_f32[route * inter_dim + col] = up_acc * xs;
}

__global__ void moe_single_swiglu_quant_fp4_kernel(
    const float* __restrict__ gate_f32,
    const float* __restrict__ up_f32,
    const int64_t* __restrict__ indices,
    const float* __restrict__ weights,
    int8_t* __restrict__ hidden_q,
    float* __restrict__ hidden_scale,
    int topk,
    int experts_start_idx,
    int n_local_experts,
    int inter_dim,
    float swiglu_limit) {
    const int route = blockIdx.x;
    const int tid = threadIdx.x;
    if (route >= topk) return;
    const int local = static_cast<int>(indices[route]) - experts_start_idx;
    if (local < 0 || local >= n_local_experts) {
        if (tid == 0) hidden_scale[route] = 0.0f;
        return;
    }
    __shared__ float sdata[kQuantThreads];
    float local_max = 0.0f;
    const float route_weight = weights[route];
    for (int col = tid; col < inter_dim; col += blockDim.x) {
        float gate = gate_f32[route * inter_dim + col];
        float up = up_f32[route * inter_dim + col];
        if (swiglu_limit > 0.0f) {
            up = fminf(fmaxf(up, -swiglu_limit), swiglu_limit);
            gate = fminf(gate, swiglu_limit);
        }
        const float silu = gate / (1.0f + expf(-gate));
        const float v = silu * up * route_weight;
        local_max = fmaxf(local_max, fabsf(v));
    }
    sdata[tid] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        __syncthreads();
    }
    const float scale = fmaxf(sdata[0], 1.0e-6f) / 127.0f;
    if (tid == 0) hidden_scale[route] = scale;
    __syncthreads();
    const float inv_scale = 1.0f / scale;
    for (int col = tid; col < inter_dim; col += blockDim.x) {
        float gate = gate_f32[route * inter_dim + col];
        float up = up_f32[route * inter_dim + col];
        if (swiglu_limit > 0.0f) {
            up = fminf(fmaxf(up, -swiglu_limit), swiglu_limit);
            gate = fminf(gate, swiglu_limit);
        }
        const float silu = gate / (1.0f + expf(-gate));
        const float v = silu * up * route_weight;
        int q = __float2int_rn(v * inv_scale);
        q = max(-127, min(127, q));
        hidden_q[route * inter_dim + col] = static_cast<int8_t>(q);
    }
}

__device__ __forceinline__ int8_t fp4_decode_code_x2(uint8_t code) {
    return fp4_lut_x2[code & 0x0f];
}

__global__ void moe_prefill_fp4_grouped_w13_wmma_kernel(
    const int8_t* __restrict__ x_q_pad,
    const float* __restrict__ x_scale_pad,
    const int32_t* __restrict__ seg_starts,
    const uint8_t* __restrict__ w1q,
    const uint8_t* __restrict__ w1s,
    const uint8_t* __restrict__ w3q,
    const uint8_t* __restrict__ w3s,
    float* __restrict__ gate_f32,
    float* __restrict__ up_f32,
    int rows_per_expert,
    int dim,
    int inter_dim) {
    using namespace nvcuda;
    constexpr int TILE = 16;
    const int expert = blockIdx.z;
    const int row_base = blockIdx.y * TILE;
    const int col_base = blockIdx.x * TILE;
    const int tid = threadIdx.x;
    const int blocks_k = dim / 32;
    const int count = seg_starts[expert + 1] - seg_starts[expert];
    extern __shared__ __align__(16) unsigned char smem[];
    signed char* a_tile = reinterpret_cast<signed char*>(smem);
    signed char* b_tile = a_tile + TILE * TILE;
    int* c_tile = reinterpret_cast<int*>(b_tile + 2 * TILE * TILE);
    float* gate_acc = reinterpret_cast<float*>(c_tile + TILE * TILE);
    float* up_acc = gate_acc + TILE * TILE;

    for (int i = tid; i < TILE * TILE; i += blockDim.x) {
        gate_acc[i] = 0.0f;
        up_acc[i] = 0.0f;
    }
    __syncthreads();

    for (int kb = 0; kb < blocks_k; ++kb) {
        wmma::fragment<wmma::matrix_a, TILE, TILE, TILE, signed char, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, TILE, TILE, TILE, signed char, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, TILE, TILE, TILE, int> c1_frag;
        wmma::fragment<wmma::accumulator, TILE, TILE, TILE, int> c3_frag;
        wmma::fill_fragment(c1_frag, 0);
        wmma::fill_fragment(c3_frag, 0);
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int k0 = kb * 32 + half * TILE;
            for (int i = tid; i < TILE * TILE; i += blockDim.x) {
                const int r = i / TILE;
                const int k = i - r * TILE;
                const int row = row_base + r;
                a_tile[i] = (row < count && row < rows_per_expert)
                    ? x_q_pad[(static_cast<int64_t>(expert) * rows_per_expert + row) * dim + k0 + k]
                    : 0;
            }
            for (int i = tid; i < TILE * TILE; i += blockDim.x) {
                const int k = i & (TILE - 1);
                const int n = i >> 4;
                const int col = col_base + n;
                signed char v1 = 0;
                signed char v3 = 0;
                if (col < inter_dim) {
                    const int kk = k0 + k;
                    const uint8_t* w1_row = w1q + (static_cast<int64_t>(expert) * inter_dim + col) * (dim / 2);
                    const uint8_t* w3_row = w3q + (static_cast<int64_t>(expert) * inter_dim + col) * (dim / 2);
                    const uint8_t b1 = w1_row[kk >> 1];
                    const uint8_t b3 = w3_row[kk >> 1];
                    const uint8_t c1 = (kk & 1) ? (b1 >> 4) : (b1 & 0x0f);
                    const uint8_t c3 = (kk & 1) ? (b3 >> 4) : (b3 & 0x0f);
                    v1 = fp4_decode_code_x2(c1);
                    v3 = fp4_decode_code_x2(c3);
                }
                b_tile[k + n * TILE] = v1;
                b_tile[TILE * TILE + k + n * TILE] = v3;
            }
            __syncthreads();
            wmma::load_matrix_sync(a_frag, a_tile, TILE);
            wmma::load_matrix_sync(b_frag, b_tile, TILE);
            wmma::mma_sync(c1_frag, a_frag, b_frag, c1_frag);
            wmma::load_matrix_sync(b_frag, b_tile + TILE * TILE, TILE);
            wmma::mma_sync(c3_frag, a_frag, b_frag, c3_frag);
            __syncthreads();
        }
        wmma::store_matrix_sync(c_tile, c1_frag, TILE, wmma::mem_row_major);
        __syncthreads();
        for (int i = tid; i < TILE * TILE; i += blockDim.x) {
            const int n = i & (TILE - 1);
            const int col = col_base + n;
            if (col < inter_dim) {
                const float s = fp4_block_scale(w1s[(static_cast<int64_t>(expert) * inter_dim + col) * blocks_k + kb]);
                gate_acc[i] += static_cast<float>(c_tile[i]) * s;
            }
        }
        __syncthreads();
        wmma::store_matrix_sync(c_tile, c3_frag, TILE, wmma::mem_row_major);
        __syncthreads();
        for (int i = tid; i < TILE * TILE; i += blockDim.x) {
            const int n = i & (TILE - 1);
            const int col = col_base + n;
            if (col < inter_dim) {
                const float s = fp4_block_scale(w3s[(static_cast<int64_t>(expert) * inter_dim + col) * blocks_k + kb]);
                up_acc[i] += static_cast<float>(c_tile[i]) * s;
            }
        }
        __syncthreads();
    }

    for (int i = tid; i < TILE * TILE; i += blockDim.x) {
        const int r = i / TILE;
        const int n = i - r * TILE;
        const int row = row_base + r;
        const int col = col_base + n;
        if (row < count && row < rows_per_expert && col < inter_dim) {
            const int64_t out = (static_cast<int64_t>(expert) * rows_per_expert + row) * inter_dim + col;
            const float xs = x_scale_pad[expert * rows_per_expert + row];
            gate_f32[out] = gate_acc[i] * xs;
            up_f32[out] = up_acc[i] * xs;
        }
    }
}

__global__ void moe_prefill_fp4_grouped_w13_wmma_compact_kernel(
    const int8_t* __restrict__ x_q,
    const float* __restrict__ x_scale,
    const int32_t* __restrict__ seg_starts,
    const int32_t* __restrict__ tile_experts,
    const int32_t* __restrict__ tile_rows,
    const uint8_t* __restrict__ w1q,
    const uint8_t* __restrict__ w1s,
    const uint8_t* __restrict__ w3q,
    const uint8_t* __restrict__ w3s,
    float* __restrict__ gate_f32,
    float* __restrict__ up_f32,
    int tile_count,
    int dim,
    int inter_dim) {
    using namespace nvcuda;
    constexpr int TILE = 16;
    const int tile_id = blockIdx.y;
    if (tile_id >= tile_count) return;
    const int expert = tile_experts[tile_id];
    const int row_base = tile_rows[tile_id];
    const int col_base = blockIdx.x * TILE;
    const int tid = threadIdx.x;
    const int blocks_k = dim / 32;
    const int count = seg_starts[expert + 1] - seg_starts[expert];
    extern __shared__ __align__(16) unsigned char smem[];
    signed char* a_tile = reinterpret_cast<signed char*>(smem);
    signed char* b_tile = a_tile + TILE * TILE;
    int* c_tile = reinterpret_cast<int*>(b_tile + 2 * TILE * TILE);
    float* gate_acc = reinterpret_cast<float*>(c_tile + TILE * TILE);
    float* up_acc = gate_acc + TILE * TILE;

    for (int i = tid; i < TILE * TILE; i += blockDim.x) {
        gate_acc[i] = 0.0f;
        up_acc[i] = 0.0f;
    }
    __syncthreads();

    for (int kb = 0; kb < blocks_k; ++kb) {
        wmma::fragment<wmma::matrix_a, TILE, TILE, TILE, signed char, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, TILE, TILE, TILE, signed char, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, TILE, TILE, TILE, int> c1_frag;
        wmma::fragment<wmma::accumulator, TILE, TILE, TILE, int> c3_frag;
        wmma::fill_fragment(c1_frag, 0);
        wmma::fill_fragment(c3_frag, 0);
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int k0 = kb * 32 + half * TILE;
            for (int i = tid; i < TILE * TILE; i += blockDim.x) {
                const int r = i / TILE;
                const int k = i - r * TILE;
                const int row = row_base + r;
                const int route = seg_starts[expert] + row;
                a_tile[i] = row < count ? x_q[static_cast<int64_t>(route) * dim + k0 + k] : 0;
            }
            for (int i = tid; i < TILE * TILE; i += blockDim.x) {
                const int k = i & (TILE - 1);
                const int n = i >> 4;
                const int col = col_base + n;
                signed char v1 = 0;
                signed char v3 = 0;
                if (col < inter_dim) {
                    const int kk = k0 + k;
                    const uint8_t* w1_row = w1q + (static_cast<int64_t>(expert) * inter_dim + col) * (dim / 2);
                    const uint8_t* w3_row = w3q + (static_cast<int64_t>(expert) * inter_dim + col) * (dim / 2);
                    const uint8_t b1 = w1_row[kk >> 1];
                    const uint8_t b3 = w3_row[kk >> 1];
                    const uint8_t c1 = (kk & 1) ? (b1 >> 4) : (b1 & 0x0f);
                    const uint8_t c3 = (kk & 1) ? (b3 >> 4) : (b3 & 0x0f);
                    v1 = fp4_decode_code_x2(c1);
                    v3 = fp4_decode_code_x2(c3);
                }
                b_tile[k + n * TILE] = v1;
                b_tile[TILE * TILE + k + n * TILE] = v3;
            }
            __syncthreads();
            wmma::load_matrix_sync(a_frag, a_tile, TILE);
            wmma::load_matrix_sync(b_frag, b_tile, TILE);
            wmma::mma_sync(c1_frag, a_frag, b_frag, c1_frag);
            wmma::load_matrix_sync(b_frag, b_tile + TILE * TILE, TILE);
            wmma::mma_sync(c3_frag, a_frag, b_frag, c3_frag);
            __syncthreads();
        }
        wmma::store_matrix_sync(c_tile, c1_frag, TILE, wmma::mem_row_major);
        __syncthreads();
        for (int i = tid; i < TILE * TILE; i += blockDim.x) {
            const int n = i & (TILE - 1);
            const int col = col_base + n;
            if (col < inter_dim) {
                const float s = fp4_block_scale(w1s[(static_cast<int64_t>(expert) * inter_dim + col) * blocks_k + kb]);
                gate_acc[i] += static_cast<float>(c_tile[i]) * s;
            }
        }
        __syncthreads();
        wmma::store_matrix_sync(c_tile, c3_frag, TILE, wmma::mem_row_major);
        __syncthreads();
        for (int i = tid; i < TILE * TILE; i += blockDim.x) {
            const int n = i & (TILE - 1);
            const int col = col_base + n;
            if (col < inter_dim) {
                const float s = fp4_block_scale(w3s[(static_cast<int64_t>(expert) * inter_dim + col) * blocks_k + kb]);
                up_acc[i] += static_cast<float>(c_tile[i]) * s;
            }
        }
        __syncthreads();
    }

    for (int i = tid; i < TILE * TILE; i += blockDim.x) {
        const int r = i / TILE;
        const int n = i - r * TILE;
        const int row = row_base + r;
        const int col = col_base + n;
        if (row < count && col < inter_dim) {
            const int route = seg_starts[expert] + row;
            const float xs = x_scale[route];
            gate_f32[static_cast<int64_t>(route) * inter_dim + col] = gate_acc[i] * xs;
            up_f32[static_cast<int64_t>(route) * inter_dim + col] = up_acc[i] * xs;
        }
    }
}

template <int WARPS>
__global__ void moe_prefill_fp4_grouped_w13_wmma_compact_nsplit_kernel(
    const int8_t* __restrict__ x_q,
    const float* __restrict__ x_scale,
    const int32_t* __restrict__ seg_starts,
    const int32_t* __restrict__ tile_experts,
    const int32_t* __restrict__ tile_rows,
    const uint8_t* __restrict__ w1q,
    const uint8_t* __restrict__ w1s,
    const uint8_t* __restrict__ w3q,
    const uint8_t* __restrict__ w3s,
    float* __restrict__ gate_f32,
    float* __restrict__ up_f32,
    int tile_count,
    int dim,
    int inter_dim) {
    using namespace nvcuda;
    constexpr int TILE = 16;
    const int tile_id = blockIdx.y;
    if (tile_id >= tile_count) return;
    const int expert = tile_experts[tile_id];
    const int row_base = tile_rows[tile_id];
    const int col_base_block = blockIdx.x * (TILE * WARPS);
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int col_base = col_base_block + warp_id * TILE;
    const int blocks_k = dim / 32;
    const int count = seg_starts[expert + 1] - seg_starts[expert];
    extern __shared__ __align__(16) unsigned char smem[];
    signed char* a_tile = reinterpret_cast<signed char*>(smem);
    signed char* b_tile_block = a_tile + TILE * TILE;
    signed char* b_tile_w1 = b_tile_block + warp_id * TILE * TILE;
    signed char* b_tile_w3 = b_tile_block + (WARPS + warp_id) * TILE * TILE;
    int* c_tile_block = reinterpret_cast<int*>(b_tile_block + 2 * WARPS * TILE * TILE);
    int* c_tile_warp = c_tile_block + warp_id * TILE * TILE;
    float* gate_acc_block = reinterpret_cast<float*>(c_tile_block + WARPS * TILE * TILE);
    float* gate_acc = gate_acc_block + warp_id * TILE * TILE;
    float* up_acc = gate_acc_block + (WARPS + warp_id) * TILE * TILE;

    #pragma unroll
    for (int i = lane; i < TILE * TILE; i += 32) {
        gate_acc[i] = 0.0f;
        up_acc[i] = 0.0f;
    }
    __syncthreads();

    for (int kb = 0; kb < blocks_k; ++kb) {
        wmma::fragment<wmma::matrix_a, TILE, TILE, TILE, signed char, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, TILE, TILE, TILE, signed char, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, TILE, TILE, TILE, int> c1_frag;
        wmma::fragment<wmma::accumulator, TILE, TILE, TILE, int> c3_frag;
        wmma::fill_fragment(c1_frag, 0);
        wmma::fill_fragment(c3_frag, 0);
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int k0 = kb * 32 + half * TILE;
            // load a_tile cooperatively across all warps (32 entries per warp covers all 256)
            #pragma unroll
            for (int i = tid; i < TILE * TILE; i += blockDim.x) {
                const int r = i / TILE;
                const int k = i - r * TILE;
                const int row = row_base + r;
                const int route = seg_starts[expert] + row;
                a_tile[i] = row < count ? x_q[static_cast<int64_t>(route) * dim + k0 + k] : 0;
            }
            // each warp loads its own b_tile slice for its own col range
            #pragma unroll
            for (int i = lane; i < TILE * TILE; i += 32) {
                const int k = i & (TILE - 1);
                const int n = i >> 4;
                const int col = col_base + n;
                signed char v1 = 0;
                signed char v3 = 0;
                if (col < inter_dim) {
                    const int kk = k0 + k;
                    const uint8_t b1 = w1q[(static_cast<int64_t>(expert) * inter_dim + col) * (dim / 2) + (kk >> 1)];
                    const uint8_t b3 = w3q[(static_cast<int64_t>(expert) * inter_dim + col) * (dim / 2) + (kk >> 1)];
                    const uint8_t c1 = (kk & 1) ? (b1 >> 4) : (b1 & 0x0f);
                    const uint8_t c3 = (kk & 1) ? (b3 >> 4) : (b3 & 0x0f);
                    v1 = fp4_decode_code_x2(c1);
                    v3 = fp4_decode_code_x2(c3);
                }
                b_tile_w1[k + n * TILE] = v1;
                b_tile_w3[k + n * TILE] = v3;
            }
            __syncthreads();
            wmma::load_matrix_sync(a_frag, a_tile, TILE);
            wmma::load_matrix_sync(b_frag, b_tile_w1, TILE);
            wmma::mma_sync(c1_frag, a_frag, b_frag, c1_frag);
            wmma::load_matrix_sync(b_frag, b_tile_w3, TILE);
            wmma::mma_sync(c3_frag, a_frag, b_frag, c3_frag);
            __syncthreads();
        }
        wmma::store_matrix_sync(c_tile_warp, c1_frag, TILE, wmma::mem_row_major);
        #pragma unroll
        for (int i = lane; i < TILE * TILE; i += 32) {
            const int n = i & (TILE - 1);
            const int col = col_base + n;
            if (col < inter_dim) {
                const float s = fp4_block_scale(w1s[(static_cast<int64_t>(expert) * inter_dim + col) * blocks_k + kb]);
                gate_acc[i] += static_cast<float>(c_tile_warp[i]) * s;
            }
        }
        wmma::store_matrix_sync(c_tile_warp, c3_frag, TILE, wmma::mem_row_major);
        #pragma unroll
        for (int i = lane; i < TILE * TILE; i += 32) {
            const int n = i & (TILE - 1);
            const int col = col_base + n;
            if (col < inter_dim) {
                const float s = fp4_block_scale(w3s[(static_cast<int64_t>(expert) * inter_dim + col) * blocks_k + kb]);
                up_acc[i] += static_cast<float>(c_tile_warp[i]) * s;
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = lane; i < TILE * TILE; i += 32) {
        const int r = i / TILE;
        const int n = i - r * TILE;
        const int row = row_base + r;
        const int col = col_base + n;
        if (row < count && col < inter_dim) {
            const int route = seg_starts[expert] + row;
            const float xs = x_scale[route];
            gate_f32[static_cast<int64_t>(route) * inter_dim + col] = gate_acc[i] * xs;
            up_f32[static_cast<int64_t>(route) * inter_dim + col] = up_acc[i] * xs;
        }
    }
}

__global__ void moe_prefill_fp4_grouped_w2_wmma_kernel(
    const int8_t* __restrict__ hidden_q,
    const float* __restrict__ hidden_scale,
    const int64_t* __restrict__ route_tokens,
    const int32_t* __restrict__ seg_starts,
    const uint8_t* __restrict__ w2q,
    const uint8_t* __restrict__ w2s,
    float* __restrict__ y,
    int rows_per_expert,
    int dim,
    int inter_dim,
    bool write_partials = false) {
    using namespace nvcuda;
    constexpr int TILE = 16;
    const int expert = blockIdx.z;
    const int row_base = blockIdx.y * TILE;
    const int col_base = blockIdx.x * TILE;
    const int tid = threadIdx.x;
    const int blocks_k = inter_dim / 32;
    const int count = seg_starts[expert + 1] - seg_starts[expert];
    extern __shared__ __align__(16) unsigned char smem[];
    signed char* a_tile = reinterpret_cast<signed char*>(smem);
    signed char* b_tile = a_tile + TILE * TILE;
    int* c_tile = reinterpret_cast<int*>(b_tile + TILE * TILE);
    float* acc = reinterpret_cast<float*>(c_tile + TILE * TILE);

    for (int i = tid; i < TILE * TILE; i += blockDim.x) acc[i] = 0.0f;
    __syncthreads();

    for (int kb = 0; kb < blocks_k; ++kb) {
        wmma::fragment<wmma::matrix_a, TILE, TILE, TILE, signed char, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, TILE, TILE, TILE, signed char, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, TILE, TILE, TILE, int> c_frag;
        wmma::fill_fragment(c_frag, 0);
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int k0 = kb * 32 + half * TILE;
            for (int i = tid; i < TILE * TILE; i += blockDim.x) {
                const int r = i / TILE;
                const int k = i - r * TILE;
                const int row = row_base + r;
                a_tile[i] = (row < count && row < rows_per_expert)
                    ? hidden_q[(static_cast<int64_t>(expert) * rows_per_expert + row) * inter_dim + k0 + k]
                    : 0;
            }
            for (int i = tid; i < TILE * TILE; i += blockDim.x) {
                const int k = i & (TILE - 1);
                const int n = i >> 4;
                const int col = col_base + n;
                signed char v = 0;
                if (col < dim) {
                    const int kk = k0 + k;
                    const uint8_t* w2_row = w2q + (static_cast<int64_t>(expert) * dim + col) * (inter_dim / 2);
                    const uint8_t b = w2_row[kk >> 1];
                    const uint8_t c = (kk & 1) ? (b >> 4) : (b & 0x0f);
                    v = fp4_decode_code_x2(c);
                }
                b_tile[k + n * TILE] = v;
            }
            __syncthreads();
            wmma::load_matrix_sync(a_frag, a_tile, TILE);
            wmma::load_matrix_sync(b_frag, b_tile, TILE);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            __syncthreads();
        }
        wmma::store_matrix_sync(c_tile, c_frag, TILE, wmma::mem_row_major);
        __syncthreads();
        for (int i = tid; i < TILE * TILE; i += blockDim.x) {
            const int n = i & (TILE - 1);
            const int col = col_base + n;
            if (col < dim) {
                const float s = fp4_block_scale(w2s[(static_cast<int64_t>(expert) * dim + col) * blocks_k + kb]);
                acc[i] += static_cast<float>(c_tile[i]) * s;
            }
        }
        __syncthreads();
    }

    for (int i = tid; i < TILE * TILE; i += blockDim.x) {
        const int r = i / TILE;
        const int n = i - r * TILE;
        const int row = row_base + r;
        const int col = col_base + n;
        if (row < count && row < rows_per_expert && col < dim) {
            const int route = seg_starts[expert] + row;
            const float val = acc[i] * hidden_scale[expert * rows_per_expert + row];
            if (write_partials) {
                // Deterministic path: one row per route, summed in slot order later.
                y[static_cast<size_t>(route) * dim + col] = val;
            } else {
                const int64_t token = route_tokens[route];
                atomicAdd(y + static_cast<size_t>(token) * dim + col, val);
            }
        }
    }
}

__global__ void moe_prefill_fp4_grouped_w2_wmma_compact_kernel(
    const int8_t* __restrict__ hidden_q,
    const float* __restrict__ hidden_scale,
    const int64_t* __restrict__ route_tokens,
    const int32_t* __restrict__ seg_starts,
    const int32_t* __restrict__ tile_experts,
    const int32_t* __restrict__ tile_rows,
    const uint8_t* __restrict__ w2q,
    const uint8_t* __restrict__ w2s,
    float* __restrict__ y,
    int tile_count,
    int dim,
    int inter_dim,
    bool write_partials = false) {
    using namespace nvcuda;
    constexpr int TILE = 16;
    const int tile_id = blockIdx.y;
    if (tile_id >= tile_count) return;
    const int expert = tile_experts[tile_id];
    const int row_base = tile_rows[tile_id];
    const int col_base = blockIdx.x * TILE;
    const int tid = threadIdx.x;
    const int blocks_k = inter_dim / 32;
    const int count = seg_starts[expert + 1] - seg_starts[expert];
    extern __shared__ __align__(16) unsigned char smem[];
    signed char* a_tile = reinterpret_cast<signed char*>(smem);
    signed char* b_tile = a_tile + TILE * TILE;
    int* c_tile = reinterpret_cast<int*>(b_tile + TILE * TILE);
    float* acc = reinterpret_cast<float*>(c_tile + TILE * TILE);

    for (int i = tid; i < TILE * TILE; i += blockDim.x) acc[i] = 0.0f;
    __syncthreads();

    for (int kb = 0; kb < blocks_k; ++kb) {
        wmma::fragment<wmma::matrix_a, TILE, TILE, TILE, signed char, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, TILE, TILE, TILE, signed char, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, TILE, TILE, TILE, int> c_frag;
        wmma::fill_fragment(c_frag, 0);
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int k0 = kb * 32 + half * TILE;
            for (int i = tid; i < TILE * TILE; i += blockDim.x) {
                const int r = i / TILE;
                const int k = i - r * TILE;
                const int row = row_base + r;
                const int route = seg_starts[expert] + row;
                a_tile[i] = row < count ? hidden_q[static_cast<int64_t>(route) * inter_dim + k0 + k] : 0;
            }
            for (int i = tid; i < TILE * TILE; i += blockDim.x) {
                const int k = i & (TILE - 1);
                const int n = i >> 4;
                const int col = col_base + n;
                signed char v = 0;
                if (col < dim) {
                    const int kk = k0 + k;
                    const uint8_t* w2_row = w2q + (static_cast<int64_t>(expert) * dim + col) * (inter_dim / 2);
                    const uint8_t b = w2_row[kk >> 1];
                    const uint8_t c = (kk & 1) ? (b >> 4) : (b & 0x0f);
                    v = fp4_decode_code_x2(c);
                }
                b_tile[k + n * TILE] = v;
            }
            __syncthreads();
            wmma::load_matrix_sync(a_frag, a_tile, TILE);
            wmma::load_matrix_sync(b_frag, b_tile, TILE);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            __syncthreads();
        }
        wmma::store_matrix_sync(c_tile, c_frag, TILE, wmma::mem_row_major);
        __syncthreads();
        for (int i = tid; i < TILE * TILE; i += blockDim.x) {
            const int n = i & (TILE - 1);
            const int col = col_base + n;
            if (col < dim) {
                const float s = fp4_block_scale(w2s[(static_cast<int64_t>(expert) * dim + col) * blocks_k + kb]);
                acc[i] += static_cast<float>(c_tile[i]) * s;
            }
        }
        __syncthreads();
    }

    for (int i = tid; i < TILE * TILE; i += blockDim.x) {
        const int r = i / TILE;
        const int n = i - r * TILE;
        const int row = row_base + r;
        const int col = col_base + n;
        if (row < count && col < dim) {
            const int route = seg_starts[expert] + row;
            const int64_t token = route_tokens[route];
            const float val = acc[i] * hidden_scale[route];
            if (write_partials) {
                // Deterministic path: one row per route, summed in slot order later.
                y[static_cast<size_t>(route) * dim + col] = val;
            } else {
                atomicAdd(y + static_cast<size_t>(token) * dim + col, val);
            }
        }
    }
}

template <int WARPS>
__global__ void moe_prefill_fp4_grouped_w2_wmma_compact_nsplit_kernel(
    const int8_t* __restrict__ hidden_q,
    const float* __restrict__ hidden_scale,
    const int64_t* __restrict__ route_tokens,
    const int32_t* __restrict__ seg_starts,
    const int32_t* __restrict__ tile_experts,
    const int32_t* __restrict__ tile_rows,
    const uint8_t* __restrict__ w2q,
    const uint8_t* __restrict__ w2s,
    float* __restrict__ y,
    int tile_count,
    int dim,
    int inter_dim,
    bool write_partials = false) {
    using namespace nvcuda;
    constexpr int TILE = 16;
    const int tile_id = blockIdx.y;
    if (tile_id >= tile_count) return;
    const int expert = tile_experts[tile_id];
    const int row_base = tile_rows[tile_id];
    const int col_base_block = blockIdx.x * (TILE * WARPS);
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;
    const int col_base = col_base_block + warp_id * TILE;
    const int blocks_k = inter_dim / 32;
    const int count = seg_starts[expert + 1] - seg_starts[expert];
    extern __shared__ __align__(16) unsigned char smem[];
    signed char* a_tile = reinterpret_cast<signed char*>(smem);
    signed char* b_tile_block = a_tile + TILE * TILE;
    signed char* b_tile_warp = b_tile_block + warp_id * TILE * TILE;
    int* c_tile_block = reinterpret_cast<int*>(b_tile_block + WARPS * TILE * TILE);
    int* c_tile_warp = c_tile_block + warp_id * TILE * TILE;
    float* acc_block = reinterpret_cast<float*>(c_tile_block + WARPS * TILE * TILE);
    float* acc = acc_block + warp_id * TILE * TILE;

    #pragma unroll
    for (int i = lane; i < TILE * TILE; i += 32) acc[i] = 0.0f;
    __syncthreads();

    for (int kb = 0; kb < blocks_k; ++kb) {
        wmma::fragment<wmma::matrix_a, TILE, TILE, TILE, signed char, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, TILE, TILE, TILE, signed char, wmma::col_major> b_frag;
        wmma::fragment<wmma::accumulator, TILE, TILE, TILE, int> c_frag;
        wmma::fill_fragment(c_frag, 0);
        #pragma unroll
        for (int half = 0; half < 2; ++half) {
            const int k0 = kb * 32 + half * TILE;
            #pragma unroll
            for (int i = tid; i < TILE * TILE; i += blockDim.x) {
                const int r = i / TILE;
                const int k = i - r * TILE;
                const int row = row_base + r;
                const int route = seg_starts[expert] + row;
                a_tile[i] = row < count ? hidden_q[static_cast<int64_t>(route) * inter_dim + k0 + k] : 0;
            }
            #pragma unroll
            for (int i = lane; i < TILE * TILE; i += 32) {
                const int k = i & (TILE - 1);
                const int n = i >> 4;
                const int col = col_base + n;
                signed char v = 0;
                if (col < dim) {
                    const int kk = k0 + k;
                    const uint8_t b = w2q[(static_cast<int64_t>(expert) * dim + col) * (inter_dim / 2) + (kk >> 1)];
                    const uint8_t c = (kk & 1) ? (b >> 4) : (b & 0x0f);
                    v = fp4_decode_code_x2(c);
                }
                b_tile_warp[k + n * TILE] = v;
            }
            __syncthreads();
            wmma::load_matrix_sync(a_frag, a_tile, TILE);
            wmma::load_matrix_sync(b_frag, b_tile_warp, TILE);
            wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
            __syncthreads();
        }
        wmma::store_matrix_sync(c_tile_warp, c_frag, TILE, wmma::mem_row_major);
        #pragma unroll
        for (int i = lane; i < TILE * TILE; i += 32) {
            const int n = i & (TILE - 1);
            const int col = col_base + n;
            if (col < dim) {
                const float s = fp4_block_scale(w2s[(static_cast<int64_t>(expert) * dim + col) * blocks_k + kb]);
                acc[i] += static_cast<float>(c_tile_warp[i]) * s;
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = lane; i < TILE * TILE; i += 32) {
        const int r = i / TILE;
        const int n = i - r * TILE;
        const int row = row_base + r;
        const int col = col_base + n;
        if (row < count && col < dim) {
            const int route = seg_starts[expert] + row;
            const int64_t token = route_tokens[route];
            const float val = acc[i] * hidden_scale[route];
            if (write_partials) {
                // Deterministic path: one row per route, summed in slot order later.
                y[static_cast<size_t>(route) * dim + col] = val;
            } else {
                atomicAdd(y + static_cast<size_t>(token) * dim + col, val);
            }
        }
    }
}

// Deterministic reduction for prefill grouped MoE. The w2 kernels normally
// atomicAdd each route's contribution straight into y, which makes the sum
// order depend on block completion order -- float addition is not associative,
// so topk>=3 diverges run to run. Instead each route writes its own row of
// `partials` [routes, dim] and this kernel sums a token's routes in router-slot
// order, which is fixed.
//
// partials is [routes, dim] (one row per route, same order as d_x_sorted), not
// [routes, tokens, dim): the latter is quadratic in tokens and blows past device
// memory past a few hundred tokens.
__global__ void moe_prefill_reduce_partials_kernel(
    const float* __restrict__ partials,           // [routes, dim]
    float* __restrict__ y,                        // [tokens, dim]
    const int32_t* __restrict__ token_slot_routes, // [tokens, topk], -1 = not local
    int topk,
    int tokens,
    int dim) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int token = blockIdx.y;
    if (token >= tokens || col >= dim) return;

    float sum = 0.0f;
    for (int k = 0; k < topk; ++k) {
        const int route = token_slot_routes[static_cast<size_t>(token) * topk + k];
        if (route >= 0) sum += partials[static_cast<size_t>(route) * dim + col];
    }
    y[static_cast<size_t>(token) * dim + col] = sum;
}

__global__ void moe_prefill_swiglu_quant_fp4_routes_kernel(
    const float* __restrict__ gate_f32,
    const float* __restrict__ up_f32,
    const float* __restrict__ route_weights,
    int8_t* __restrict__ hidden_q,
    float* __restrict__ hidden_scale,
    int routes,
    int inter_dim,
    float swiglu_limit) {
    const int route = blockIdx.x;
    const int tid = threadIdx.x;
    if (route >= routes) return;
    const int base = route * inter_dim;
    __shared__ float sdata[kQuantThreads];
    const float route_weight = route_weights[route];
    float local_max = 0.0f;
    for (int col = tid; col < inter_dim; col += blockDim.x) {
        const int idx = base + col;
        float gate = gate_f32[idx];
        float up = up_f32[idx];
        if (swiglu_limit > 0.0f) {
            up = fminf(fmaxf(up, -swiglu_limit), swiglu_limit);
            gate = fminf(gate, swiglu_limit);
        }
        const float silu = gate / (1.0f + expf(-gate));
        const float value = silu * up * route_weight;
        local_max = fmaxf(local_max, fabsf(value));
    }
    sdata[tid] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        __syncthreads();
    }
    const float scale = fmaxf(sdata[0], 1.0e-6f) / 127.0f;
    if (tid == 0) hidden_scale[route] = scale;
    __syncthreads();
    const float inv_scale = 1.0f / scale;
    for (int col = tid; col < inter_dim; col += blockDim.x) {
        const int idx = base + col;
        float gate = gate_f32[idx];
        float up = up_f32[idx];
        if (swiglu_limit > 0.0f) {
            up = fminf(fmaxf(up, -swiglu_limit), swiglu_limit);
            gate = fminf(gate, swiglu_limit);
        }
        const float silu = gate / (1.0f + expf(-gate));
        const float value = silu * up * route_weight;
        int q = __float2int_rn(value * inv_scale);
        q = max(-127, min(127, q));
        hidden_q[idx] = static_cast<int8_t>(q);
    }
}

__global__ void moe_prefill_swiglu_quant_fp4_kernel(
    const float* __restrict__ gate_f32,
    const float* __restrict__ up_f32,
    const int32_t* __restrict__ seg_starts,
    const float* __restrict__ route_weights,
    int8_t* __restrict__ hidden_q,
    float* __restrict__ hidden_scale,
    int rows_per_expert,
    int inter_dim,
    float swiglu_limit) {
    const int expert = blockIdx.x;
    const int row = blockIdx.y;
    const int tid = threadIdx.x;
    const int count = seg_starts[expert + 1] - seg_starts[expert];
    const int padded_row = expert * rows_per_expert + row;
    const int base = padded_row * inter_dim;
    __shared__ float sdata[kQuantThreads];
    if (row >= count) {
        for (int col = tid; col < inter_dim; col += blockDim.x) hidden_q[base + col] = 0;
        if (tid == 0) hidden_scale[padded_row] = 0.0f;
        return;
    }
    const int route = seg_starts[expert] + row;
    const float route_weight = route_weights[route];
    float local_max = 0.0f;
    for (int col = tid; col < inter_dim; col += blockDim.x) {
        const int idx = base + col;
        float gate = gate_f32[idx];
        float up = up_f32[idx];
        if (swiglu_limit > 0.0f) {
            up = fminf(fmaxf(up, -swiglu_limit), swiglu_limit);
            gate = fminf(gate, swiglu_limit);
        }
        const float silu = gate / (1.0f + expf(-gate));
        const float value = silu * up * route_weight;
        local_max = fmaxf(local_max, fabsf(value));
    }
    sdata[tid] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        __syncthreads();
    }
    const float scale = fmaxf(sdata[0], 1.0e-6f) / 127.0f;
    if (tid == 0) hidden_scale[padded_row] = scale;
    __syncthreads();
    const float inv_scale = 1.0f / scale;
    for (int col = tid; col < inter_dim; col += blockDim.x) {
        const int idx = base + col;
        float gate = gate_f32[idx];
        float up = up_f32[idx];
        if (swiglu_limit > 0.0f) {
            up = fminf(fmaxf(up, -swiglu_limit), swiglu_limit);
            gate = fminf(gate, swiglu_limit);
        }
        const float silu = gate / (1.0f + expf(-gate));
        const float value = silu * up * route_weight;
        int q = __float2int_rn(value * inv_scale);
        q = max(-127, min(127, q));
        hidden_q[idx] = static_cast<int8_t>(q);
    }
}

__global__ void moe_single_w2_write_fp4_kernel(
    const int8_t* __restrict__ hidden_q,
    const float* __restrict__ hidden_scale,
    const int64_t* __restrict__ indices,
    const uint8_t* __restrict__ w2q,
    const uint8_t* __restrict__ w2s,
    float* __restrict__ route_y,
    int topk,
    int experts_start_idx,
    int n_local_experts,
    int dim,
    int inter_dim) {
    const int route = blockIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (route >= topk) return;
    const int local = static_cast<int>(indices[route]) - experts_start_idx;
    const bool valid = local >= 0 && local < n_local_experts;
    const int packs = inter_dim / 4;
    const int blocks_k = inter_dim / 32;
    extern __shared__ int h_shared[];
    if (valid) {
        const int* h_i32 = reinterpret_cast<const int*>(hidden_q + route * inter_dim);
        for (int idx = threadIdx.x; idx < packs; idx += blockDim.x) h_shared[idx] = h_i32[idx];
    }
    __syncthreads();
    if (col >= dim) return;
    float value = 0.0f;
    if (valid) {
        const uint8_t* w2_row_bytes = w2q + (static_cast<int64_t>(local) * dim + col) * (inter_dim / 2);
        const uint8_t* w2_scale_row = w2s + (static_cast<int64_t>(local) * dim + col) * blocks_k;
        const uint16_t* w2_pack_base = reinterpret_cast<const uint16_t*>(w2_row_bytes);
        float row_acc = 0.0f;
        for (int kb = 0; kb < blocks_k; ++kb) {
            const uint16_t* w2_pack = w2_pack_base + kb * 8;
            const int* h_pack = h_shared + kb * 8;
            int blk = 0;
            #pragma unroll
            for (int ip = 0; ip < 8; ++ip) {
                const int w_p = fp4_unpack_4codes_prmt(static_cast<uint32_t>(w2_pack[ip]));
                blk = dot_i8x4(h_pack[ip], w_p, blk);
            }
            row_acc += static_cast<float>(blk) * fp4_block_scale(w2_scale_row[kb]);
        }
        value = row_acc * hidden_scale[route];
    }
    route_y[static_cast<size_t>(route) * dim + col] = value;
}

__global__ void sum_topk_rows_kernel(
    const float* __restrict__ route_y,
    float* __restrict__ y,
    int topk,
    int dim) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= dim) return;
    float sum = 0.0f;
    for (int route = 0; route < topk; ++route) sum += route_y[static_cast<size_t>(route) * dim + idx];
    y[idx] = sum;
}

// Active-slot small-batch path. One block reuses an expert's weight row across
// up to eight routed token rows; larger slots are processed in multiple tiles.
__global__ void moe_multi_w1w3_fp4_kernel(
    const int8_t* __restrict__ x_q,
    const float* __restrict__ x_scale,
    const int32_t* __restrict__ slot_expert,
    const int32_t* __restrict__ slot_starts,
    const int32_t* __restrict__ slot_tokens,
    const uint8_t* __restrict__ w1q,
    const uint8_t* __restrict__ w1s,
    const uint8_t* __restrict__ w3q,
    const uint8_t* __restrict__ w3s,
    float* __restrict__ gate_f32,
    float* __restrict__ up_f32,
    int dim,
    int inter_dim) {
    const int slot = blockIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int start = slot_starts[slot];
    const int end = slot_starts[slot + 1];
    const int n_tok = end - start;
    if (n_tok <= 0) return;
    const int local = slot_expert[slot];
    const int dim_packs = dim / 4;
    const int blocks_k = dim / 32;
    extern __shared__ int xs_shared[];
    const uint8_t* w1_row = w1q + (static_cast<int64_t>(local) * inter_dim + col) * (dim / 2);
    const uint8_t* w3_row = w3q + (static_cast<int64_t>(local) * inter_dim + col) * (dim / 2);
    const uint8_t* w1_scale = w1s + (static_cast<int64_t>(local) * inter_dim + col) * blocks_k;
    const uint8_t* w3_scale = w3s + (static_cast<int64_t>(local) * inter_dim + col) * blocks_k;
    const uint16_t* w1_pack_base = reinterpret_cast<const uint16_t*>(w1_row);
    const uint16_t* w3_pack_base = reinterpret_cast<const uint16_t*>(w3_row);
    constexpr int kMaxTokens = 8;
    for (int base_t = 0; base_t < n_tok; base_t += kMaxTokens) {
        const int tile = min(kMaxTokens, n_tok - base_t);
        __syncthreads();
        for (int idx = threadIdx.x; idx < tile * dim_packs; idx += blockDim.x) {
            const int r = idx / dim_packs;
            const int pack = idx - r * dim_packs;
            const int token = slot_tokens[start + base_t + r];
            xs_shared[idx] = reinterpret_cast<const int*>(x_q + static_cast<int64_t>(token) * dim)[pack];
        }
        __syncthreads();
        if (col >= inter_dim) continue;
        float gate_acc[kMaxTokens];
        float up_acc[kMaxTokens];
#pragma unroll
        for (int t = 0; t < kMaxTokens; ++t) { gate_acc[t] = 0.0f; up_acc[t] = 0.0f; }
        for (int kb = 0; kb < blocks_k; ++kb) {
            const uint16_t* w1_pack = w1_pack_base + kb * 8;
            const uint16_t* w3_pack = w3_pack_base + kb * 8;
            int gate_blk[kMaxTokens];
            int up_blk[kMaxTokens];
#pragma unroll
            for (int t = 0; t < kMaxTokens; ++t) { gate_blk[t] = 0; up_blk[t] = 0; }
#pragma unroll
            for (int ip = 0; ip < 8; ++ip) {
                const int w1_p = fp4_unpack_4codes_prmt(static_cast<uint32_t>(w1_pack[ip]));
                const int w3_p = fp4_unpack_4codes_prmt(static_cast<uint32_t>(w3_pack[ip]));
                for (int t = 0; t < tile; ++t) {
                    const int x_p = xs_shared[t * dim_packs + kb * 8 + ip];
                    gate_blk[t] = dot_i8x4(x_p, w1_p, gate_blk[t]);
                    up_blk[t] = dot_i8x4(x_p, w3_p, up_blk[t]);
                }
            }
            const float s1 = fp4_block_scale(w1_scale[kb]);
            const float s3 = fp4_block_scale(w3_scale[kb]);
            for (int t = 0; t < tile; ++t) {
                gate_acc[t] += static_cast<float>(gate_blk[t]) * s1;
                up_acc[t] += static_cast<float>(up_blk[t]) * s3;
            }
        }
        for (int t = 0; t < tile; ++t) {
            const int pair = start + base_t + t;
            const float xs = x_scale[slot_tokens[pair]];
            gate_f32[static_cast<int64_t>(pair) * inter_dim + col] = gate_acc[t] * xs;
            up_f32[static_cast<int64_t>(pair) * inter_dim + col] = up_acc[t] * xs;
        }
    }
}

__global__ void moe_multi_swiglu_quant_fp4_kernel(
    const float* __restrict__ gate_f32,
    const float* __restrict__ up_f32,
    const float* __restrict__ pair_weights,
    int8_t* __restrict__ hidden_q,
    float* __restrict__ hidden_scale,
    int pairs,
    int inter_dim,
    float swiglu_limit) {
    const int pair = blockIdx.x;
    const int tid = threadIdx.x;
    if (pair >= pairs) return;
    __shared__ float sdata[kQuantThreads];
    const float route_weight = pair_weights[pair];
    const int64_t base = static_cast<int64_t>(pair) * inter_dim;
    float local_max = 0.0f;
    for (int col = tid; col < inter_dim; col += blockDim.x) {
        float gate = gate_f32[base + col];
        float up = up_f32[base + col];
        if (swiglu_limit > 0.0f) {
            up = fminf(fmaxf(up, -swiglu_limit), swiglu_limit);
            gate = fminf(gate, swiglu_limit);
        }
        const float silu = gate / (1.0f + expf(-gate));
        local_max = fmaxf(local_max, fabsf(silu * up * route_weight));
    }
    sdata[tid] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) sdata[tid] = fmaxf(sdata[tid], sdata[tid + stride]);
        __syncthreads();
    }
    const float scale = fmaxf(sdata[0], 1.0e-6f) / 127.0f;
    if (tid == 0) hidden_scale[pair] = scale;
    __syncthreads();
    const float inv_scale = 1.0f / scale;
    for (int col = tid; col < inter_dim; col += blockDim.x) {
        float gate = gate_f32[base + col];
        float up = up_f32[base + col];
        if (swiglu_limit > 0.0f) {
            up = fminf(fmaxf(up, -swiglu_limit), swiglu_limit);
            gate = fminf(gate, swiglu_limit);
        }
        const float silu = gate / (1.0f + expf(-gate));
        int q = __float2int_rn(silu * up * route_weight * inv_scale);
        q = max(-127, min(127, q));
        hidden_q[base + col] = static_cast<int8_t>(q);
    }
}

__global__ void moe_multi_w2_partial_fp4_kernel(
    const int8_t* __restrict__ hidden_q,
    const float* __restrict__ hidden_scale,
    const int32_t* __restrict__ slot_expert,
    const int32_t* __restrict__ slot_starts,
    const uint8_t* __restrict__ w2q,
    const uint8_t* __restrict__ w2s,
    float* __restrict__ partials,
    int dim,
    int inter_dim) {
    const int slot = blockIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int start = slot_starts[slot];
    const int end = slot_starts[slot + 1];
    const int n_tok = end - start;
    if (n_tok <= 0) return;
    const int local = slot_expert[slot];
    const int packs = inter_dim / 4;
    const int blocks_k = inter_dim / 32;
    extern __shared__ int hs_shared[];
    const uint8_t* w2_row = w2q + (static_cast<int64_t>(local) * dim + col) * (inter_dim / 2);
    const uint8_t* w2_scale = w2s + (static_cast<int64_t>(local) * dim + col) * blocks_k;
    const uint16_t* w2_pack_base = reinterpret_cast<const uint16_t*>(w2_row);
    constexpr int kMaxTokens = 8;
    for (int base_t = 0; base_t < n_tok; base_t += kMaxTokens) {
        const int tile = min(kMaxTokens, n_tok - base_t);
        __syncthreads();
        for (int idx = threadIdx.x; idx < tile * packs; idx += blockDim.x) {
            const int r = idx / packs;
            const int pack = idx - r * packs;
            hs_shared[idx] = reinterpret_cast<const int*>(hidden_q + static_cast<int64_t>(start + base_t + r) * inter_dim)[pack];
        }
        __syncthreads();
        if (col >= dim) continue;
        float acc[kMaxTokens];
#pragma unroll
        for (int t = 0; t < kMaxTokens; ++t) acc[t] = 0.0f;
        for (int kb = 0; kb < blocks_k; ++kb) {
            const uint16_t* w2_pack = w2_pack_base + kb * 8;
            int blk[kMaxTokens];
#pragma unroll
            for (int t = 0; t < kMaxTokens; ++t) blk[t] = 0;
#pragma unroll
            for (int ip = 0; ip < 8; ++ip) {
                const int w_p = fp4_unpack_4codes_prmt(static_cast<uint32_t>(w2_pack[ip]));
                for (int t = 0; t < tile; ++t) {
                    blk[t] = dot_i8x4(hs_shared[t * packs + kb * 8 + ip], w_p, blk[t]);
                }
            }
            const float scale = fp4_block_scale(w2_scale[kb]);
            for (int t = 0; t < tile; ++t) acc[t] += static_cast<float>(blk[t]) * scale;
        }
        for (int t = 0; t < tile; ++t) {
            const int pair = start + base_t + t;
            partials[static_cast<int64_t>(pair) * dim + col] = acc[t] * hidden_scale[pair];
        }
    }
}

__global__ void moe_multi_reduce_partials_kernel(
    const float* __restrict__ partials,
    const int32_t* __restrict__ slot_tokens,
    float* __restrict__ y,
    int pairs,
    int dim) {
    const int token = blockIdx.y;
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= dim) return;
    float acc = 0.0f;
    for (int pair = 0; pair < pairs; ++pair) {
        if (slot_tokens[pair] == token) {
            acc += partials[static_cast<int64_t>(pair) * dim + col];
        }
    }
    y[static_cast<int64_t>(token) * dim + col] = acc;
}

}  // namespace

bool fp4_e2m1_e8m0_matvec_cuda(
    const float* d_x,
    const uint8_t* d_weight,
    const uint8_t* d_scale,
    float* d_y,
    int rows,
    int cols,
    void* stream) {
    if (d_x == nullptr || d_weight == nullptr || d_scale == nullptr || d_y == nullptr) return false;
    if (rows <= 0 || cols <= 0 || (cols % 32) != 0) return false;
    const int threads = 256;
    const int packed_cols = cols / 2;
    const int scale_cols = cols / 32;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    fp4_e2m1_e8m0_matvec_kernel<<<rows, threads, threads * sizeof(float), cuda_stream>>>(
        d_x, d_weight, d_scale, d_y, rows, cols, packed_cols, scale_cols);
    return cudaGetLastError() == cudaSuccess;
}

bool moe_group_routes_cuda(
    const int64_t* d_indices,
    const float* d_weights,
    int64_t* d_route_tokens,
    float* d_route_weights,
    int32_t* d_seg_starts,
    int32_t* d_counts,
    int32_t* d_offsets,
    int32_t* d_total_routes,
    int tokens,
    int topk,
    int experts_start_idx,
    int n_local_experts,
    int32_t* d_token_slot_routes,
    void* stream) {
    if (d_indices == nullptr || d_weights == nullptr || d_route_tokens == nullptr || d_route_weights == nullptr || d_seg_starts == nullptr || d_counts == nullptr || d_offsets == nullptr || d_total_routes == nullptr) return false;
    if (tokens <= 0 || topk <= 0 || n_local_experts <= 0) return false;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    const int threads = 256;
    const int blocks = ceil_div_int(tokens * topk, threads);
    cudaMemsetAsync(d_counts, 0, static_cast<size_t>(n_local_experts) * sizeof(int32_t), cuda_stream);
    if (d_token_slot_routes != nullptr) {
        // -1 marks "no local route in this slot"; the fill below only writes
        // the slots whose expert lives on this rank.
        cudaMemsetAsync(d_token_slot_routes, 0xff,
                        static_cast<size_t>(tokens) * topk * sizeof(int32_t), cuda_stream);
    }
    moe_route_count_kernel<<<blocks, threads, 0, cuda_stream>>>(d_indices, d_counts, tokens, topk, experts_start_idx, n_local_experts);
    moe_route_prefix_kernel<<<1, 1, 0, cuda_stream>>>(d_counts, d_seg_starts, d_offsets, d_total_routes, n_local_experts);
    moe_route_fill_kernel<<<blocks, threads, 0, cuda_stream>>>(d_indices, d_weights, d_offsets, d_route_tokens, d_route_weights, d_token_slot_routes, tokens, topk, experts_start_idx, n_local_experts);
    return cudaGetLastError() == cudaSuccess;
}

bool moe_prefill_fp4_grouped_cuda_with_workspace(
    const float* d_x,
    const int64_t* d_route_tokens,
    const float* d_route_weights,
    const int32_t* d_seg_starts,
    const uint8_t* d_w1q,
    const uint8_t* d_w1s,
    const uint8_t* d_w2q,
    const uint8_t* d_w2s,
    const uint8_t* d_w3q,
    const uint8_t* d_w3s,
    float* d_y,
    int tokens,
    int topk,
    int routes,
    int n_local_experts,
    int max_count,
    int dim,
    int inter_dim,
    float swiglu_limit,
    MoePrefillFp4GroupedWorkspace workspace,
    const int32_t* d_token_slot_routes,
    void* stream) {
    if (d_x == nullptr || d_route_tokens == nullptr || d_route_weights == nullptr || d_seg_starts == nullptr || d_w1q == nullptr || d_w1s == nullptr || d_w2q == nullptr || d_w2s == nullptr || d_w3q == nullptr || d_w3s == nullptr || d_y == nullptr) return false;
    if (tokens <= 0 || topk <= 0 || routes < 0 || n_local_experts <= 0 || dim <= 0 || inter_dim <= 0 || (dim % 32) != 0 || (inter_dim % 32) != 0) return false;
    if (workspace.dim < dim || workspace.inter_dim < inter_dim) return false;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    if (cudaMemsetAsync(d_y, 0, static_cast<size_t>(tokens) * dim * sizeof(float), cuda_stream) != cudaSuccess) return false;
    if (routes == 0 || max_count <= 0) return cudaGetLastError() == cudaSuccess;
    const int threads = 256;
    const size_t routes_dim = static_cast<size_t>(routes) * dim;
    const int gather_blocks = static_cast<int>((routes_dim + threads - 1) / threads);
    const bool compact_tiles = workspace.d_tile_experts != nullptr && workspace.d_tile_rows != nullptr && workspace.tile_count > 0;
    if (compact_tiles) {
        if (workspace.routes_cap < routes || workspace.tile_cap < workspace.tile_count) return false;
        if (workspace.d_x_sorted == nullptr || workspace.d_x_q == nullptr || workspace.d_x_scale == nullptr || workspace.d_gate == nullptr || workspace.d_up == nullptr || workspace.d_hidden_q == nullptr || workspace.d_hidden_scale == nullptr) return false;
        const char* profile_env = std::getenv("POCKETLLM_CPP_PROFILE_MOE_PREFILL");
        const bool profile = profile_env != nullptr && profile_env[0] != '\0' && profile_env[0] != '0';
        std::chrono::steady_clock::time_point stage;
        if (profile) {
            cudaStreamSynchronize(cuda_stream);
            stage = std::chrono::steady_clock::now();
        }
        gather_routes_float_kernel<<<gather_blocks, threads, 0, cuda_stream>>>(d_x, d_route_tokens, workspace.d_x_sorted, routes, topk, dim);
        quantize_float_rows_kernel<<<routes, kQuantThreads, 0, cuda_stream>>>(workspace.d_x_sorted, workspace.d_x_q, workspace.d_x_scale, routes, dim);
        std::chrono::steady_clock::time_point after_quant;
        if (profile) {
            cudaStreamSynchronize(cuda_stream);
            after_quant = std::chrono::steady_clock::now();
        }
        static const int kMoeWmmaWarps = []() {
            const char* v = std::getenv("POCKETLLM_CPP_MOE_WMMA_WARPS");
            int n = (v != nullptr && v[0] != '\0') ? std::atoi(v) : 4;
            return n > 0 ? n : 4;
        }();
        static const int kMoeNSplit = []() {
            const char* v = std::getenv("POCKETLLM_CPP_MOE_NSPLIT");
            int n = (v != nullptr && v[0] != '\0') ? std::atoi(v) : 4;
            return n > 0 ? n : 4;
        }();
        static const bool kDeterministicReduce = []() {
            const char* v = std::getenv("POCKETLLM_CPP_MOE_DETERMINISTIC_REDUCE");
            if (v == nullptr || v[0] == '\0') return true;  // default enabled
            return v[0] != '0';
        }();
        const dim3 fp4_block(32 * kMoeWmmaWarps);
        const dim3 fp4_block_nsplit(32 * kMoeNSplit);
        const size_t x_shared_bytes = 4096;
        if (kMoeNSplit == 4) {
            const dim3 w13_grid(ceil_div_int(inter_dim, 16 * 4), workspace.tile_count);
            const size_t w13_nsplit_smem = TILE_SHARED_NSPLIT_BYTES_W13(4);
            moe_prefill_fp4_grouped_w13_wmma_compact_nsplit_kernel<4><<<w13_grid, fp4_block_nsplit, w13_nsplit_smem, cuda_stream>>>(
                workspace.d_x_q, workspace.d_x_scale, d_seg_starts, workspace.d_tile_experts, workspace.d_tile_rows, d_w1q, d_w1s, d_w3q, d_w3s, workspace.d_gate, workspace.d_up, workspace.tile_count, dim, inter_dim);
        } else if (kMoeNSplit == 2) {
            const dim3 w13_grid(ceil_div_int(inter_dim, 16 * 2), workspace.tile_count);
            const size_t w13_nsplit_smem = TILE_SHARED_NSPLIT_BYTES_W13(2);
            moe_prefill_fp4_grouped_w13_wmma_compact_nsplit_kernel<2><<<w13_grid, fp4_block_nsplit, w13_nsplit_smem, cuda_stream>>>(
                workspace.d_x_q, workspace.d_x_scale, d_seg_starts, workspace.d_tile_experts, workspace.d_tile_rows, d_w1q, d_w1s, d_w3q, d_w3s, workspace.d_gate, workspace.d_up, workspace.tile_count, dim, inter_dim);
        } else {
            const dim3 w13_grid(ceil_div_int(inter_dim, 16), workspace.tile_count);
            moe_prefill_fp4_grouped_w13_wmma_compact_kernel<<<w13_grid, fp4_block, x_shared_bytes, cuda_stream>>>(
                workspace.d_x_q, workspace.d_x_scale, d_seg_starts, workspace.d_tile_experts, workspace.d_tile_rows, d_w1q, d_w1s, d_w3q, d_w3s, workspace.d_gate, workspace.d_up, workspace.tile_count, dim, inter_dim);
        }
        std::chrono::steady_clock::time_point after_w13;
        if (profile) {
            cudaStreamSynchronize(cuda_stream);
            after_w13 = std::chrono::steady_clock::now();
        }
        moe_prefill_swiglu_quant_fp4_routes_kernel<<<routes, kQuantThreads, 0, cuda_stream>>>(
            workspace.d_gate, workspace.d_up, d_route_weights, workspace.d_hidden_q, workspace.d_hidden_scale, routes, inter_dim, swiglu_limit);
        std::chrono::steady_clock::time_point after_swiglu;
        if (profile) {
            cudaStreamSynchronize(cuda_stream);
            after_swiglu = std::chrono::steady_clock::now();
        }
        const size_t h_shared_bytes = 4096;

        // Deterministic reduction: each route writes its own row of `partials`
        // [routes, dim] and a second pass sums each token's routes in router-slot
        // order. Only worth it for topk>=3 -- a two-term atomic sum has a single
        // possible order and is already reproducible.
        const bool deterministic = kDeterministicReduce && topk >= 3 &&
                                   d_token_slot_routes != nullptr &&
                                   workspace.d_partials != nullptr &&
                                   workspace.routes_cap >= routes;
        float* const w2_out = deterministic ? workspace.d_partials : d_y;

        if (kMoeNSplit == 4) {
            const dim3 w2_grid(ceil_div_int(dim, 16 * 4), workspace.tile_count);
            const size_t w2_nsplit_smem = TILE_SHARED_NSPLIT_BYTES_W2(4);
            moe_prefill_fp4_grouped_w2_wmma_compact_nsplit_kernel<4><<<w2_grid, fp4_block_nsplit, w2_nsplit_smem, cuda_stream>>>(
                workspace.d_hidden_q, workspace.d_hidden_scale, d_route_tokens, d_seg_starts, workspace.d_tile_experts, workspace.d_tile_rows, d_w2q, d_w2s,
                w2_out, workspace.tile_count, dim, inter_dim, deterministic);
        } else if (kMoeNSplit == 2) {
            const dim3 w2_grid(ceil_div_int(dim, 16 * 2), workspace.tile_count);
            const size_t w2_nsplit_smem = TILE_SHARED_NSPLIT_BYTES_W2(2);
            moe_prefill_fp4_grouped_w2_wmma_compact_nsplit_kernel<2><<<w2_grid, fp4_block_nsplit, w2_nsplit_smem, cuda_stream>>>(
                workspace.d_hidden_q, workspace.d_hidden_scale, d_route_tokens, d_seg_starts, workspace.d_tile_experts, workspace.d_tile_rows, d_w2q, d_w2s,
                w2_out, workspace.tile_count, dim, inter_dim, deterministic);
        } else {
            const dim3 w2_grid(ceil_div_int(dim, 16), workspace.tile_count);
            moe_prefill_fp4_grouped_w2_wmma_compact_kernel<<<w2_grid, fp4_block, h_shared_bytes, cuda_stream>>>(
                workspace.d_hidden_q, workspace.d_hidden_scale, d_route_tokens, d_seg_starts, workspace.d_tile_experts, workspace.d_tile_rows, d_w2q, d_w2s,
                w2_out, workspace.tile_count, dim, inter_dim, deterministic);
        }

        if (deterministic) {
            const dim3 reduce_grid(ceil_div_int(dim, 256), tokens);
            moe_prefill_reduce_partials_kernel<<<reduce_grid, 256, 0, cuda_stream>>>(
                workspace.d_partials, d_y, d_token_slot_routes, topk, tokens, dim);
        }

        std::chrono::steady_clock::time_point after_w2;
        if (profile) {
            cudaStreamSynchronize(cuda_stream);
            after_w2 = std::chrono::steady_clock::now();
            using std::chrono::duration;
            std::cerr << "CPP_MOE_PREFILL_STAGE routes=" << routes
                      << " tiles=" << workspace.tile_count
                      << " quant_ms=" << duration<double, std::milli>(after_quant - stage).count()
                      << " w13_ms=" << duration<double, std::milli>(after_w13 - after_quant).count()
                      << " swiglu_ms=" << duration<double, std::milli>(after_swiglu - after_w13).count()
                      << " w2_ms=" << duration<double, std::milli>(after_w2 - after_swiglu).count()
                      << "\n";
        }
        return cudaGetLastError() == cudaSuccess;
    }
    const int padded_rows = n_local_experts * max_count;
    if (workspace.routes_cap < routes || workspace.padded_rows_cap < padded_rows) return false;
    if (workspace.d_x_sorted == nullptr || workspace.d_x_q == nullptr || workspace.d_x_scale == nullptr || workspace.d_x_pad == nullptr || workspace.d_x_scale_pad == nullptr || workspace.d_gate == nullptr || workspace.d_up == nullptr || workspace.d_hidden_q == nullptr || workspace.d_hidden_scale == nullptr) return false;
    const size_t padded_dim = static_cast<size_t>(padded_rows) * dim;
    const size_t padded_inter = static_cast<size_t>(padded_rows) * inter_dim;
    if (cudaMemsetAsync(workspace.d_x_pad, 0, padded_dim, cuda_stream) != cudaSuccess) return false;
    if (cudaMemsetAsync(workspace.d_x_scale_pad, 0, static_cast<size_t>(padded_rows) * sizeof(float), cuda_stream) != cudaSuccess) return false;
    if (cudaMemsetAsync(workspace.d_gate, 0, padded_inter * sizeof(float), cuda_stream) != cudaSuccess) return false;
    if (cudaMemsetAsync(workspace.d_up, 0, padded_inter * sizeof(float), cuda_stream) != cudaSuccess) return false;
    if (cudaMemsetAsync(workspace.d_hidden_q, 0, padded_inter, cuda_stream) != cudaSuccess) return false;
    if (cudaMemsetAsync(workspace.d_hidden_scale, 0, static_cast<size_t>(padded_rows) * sizeof(float), cuda_stream) != cudaSuccess) return false;
    gather_routes_float_kernel<<<gather_blocks, threads, 0, cuda_stream>>>(d_x, d_route_tokens, workspace.d_x_sorted, routes, topk, dim);
    quantize_float_rows_kernel<<<routes, kQuantThreads, 0, cuda_stream>>>(workspace.d_x_sorted, workspace.d_x_q, workspace.d_x_scale, routes, dim);
    const int pad_blocks = static_cast<int>((padded_dim + threads - 1) / threads);
    pad_q_rows_kernel<<<pad_blocks, threads, 0, cuda_stream>>>(workspace.d_x_q, workspace.d_x_scale, d_seg_starts, workspace.d_x_pad, workspace.d_x_scale_pad, n_local_experts, max_count, dim);

    static const bool kDeterministicReduce = []() {
        const char* v = std::getenv("POCKETLLM_CPP_MOE_DETERMINISTIC_REDUCE");
        if (v == nullptr || v[0] == '\0') return true;  // default enabled
        return v[0] != '0';
    }();

    const dim3 fp4_block(128);
    const dim3 w13_grid(ceil_div_int(inter_dim, 16), ceil_div_int(max_count, 16), n_local_experts);
    const size_t x_shared_bytes = 4096;
    moe_prefill_fp4_grouped_w13_wmma_kernel<<<w13_grid, fp4_block, x_shared_bytes, cuda_stream>>>(
        workspace.d_x_pad, workspace.d_x_scale_pad, d_seg_starts, d_w1q, d_w1s, d_w3q, d_w3s, workspace.d_gate, workspace.d_up, max_count, dim, inter_dim);
    moe_prefill_swiglu_quant_fp4_kernel<<<dim3(n_local_experts, max_count), kQuantThreads, 0, cuda_stream>>>(
        workspace.d_gate, workspace.d_up, d_seg_starts, d_route_weights, workspace.d_hidden_q, workspace.d_hidden_scale, max_count, inter_dim, swiglu_limit);

    const dim3 w2_grid(ceil_div_int(dim, 16), ceil_div_int(max_count, 16), n_local_experts);
    const size_t h_shared_bytes = 4096;

    // Same deterministic reduction as the compact path above.
    const bool deterministic = kDeterministicReduce && topk >= 3 &&
                               d_token_slot_routes != nullptr &&
                               workspace.d_partials != nullptr &&
                               workspace.routes_cap >= routes;

    moe_prefill_fp4_grouped_w2_wmma_kernel<<<w2_grid, fp4_block, h_shared_bytes, cuda_stream>>>(
        workspace.d_hidden_q, workspace.d_hidden_scale, d_route_tokens, d_seg_starts, d_w2q, d_w2s,
        deterministic ? workspace.d_partials : d_y, max_count, dim, inter_dim, deterministic);

    if (deterministic) {
        const dim3 reduce_grid(ceil_div_int(dim, 256), tokens);
        moe_prefill_reduce_partials_kernel<<<reduce_grid, 256, 0, cuda_stream>>>(
            workspace.d_partials, d_y, d_token_slot_routes, topk, tokens, dim);
    }

    return cudaGetLastError() == cudaSuccess;
}

bool moe_prefill_fp4_grouped_cuda(
    const float* d_x,
    const int64_t* d_route_tokens,
    const float* d_route_weights,
    const int32_t* d_seg_starts,
    const uint8_t* d_w1q,
    const uint8_t* d_w1s,
    const uint8_t* d_w2q,
    const uint8_t* d_w2s,
    const uint8_t* d_w3q,
    const uint8_t* d_w3s,
    float* d_y,
    int tokens,
    int topk,
    int routes,
    int n_local_experts,
    int max_count,
    int dim,
    int inter_dim,
    float swiglu_limit,
    void* stream) {
    if (d_x == nullptr || d_route_tokens == nullptr || d_route_weights == nullptr || d_seg_starts == nullptr || d_w1q == nullptr || d_w1s == nullptr || d_w2q == nullptr || d_w2s == nullptr || d_w3q == nullptr || d_w3s == nullptr || d_y == nullptr) return false;
    if (tokens <= 0 || topk <= 0 || routes < 0 || n_local_experts <= 0 || max_count <= 0 || dim <= 0 || inter_dim <= 0 || (dim % 32) != 0 || (inter_dim % 32) != 0) return false;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    MoePrefillFp4GroupedWorkspace workspace;
    workspace.dim = dim;
    workspace.inter_dim = inter_dim;
    workspace.routes_cap = routes;
    workspace.padded_rows_cap = n_local_experts * max_count;
    if (cudaMemsetAsync(d_y, 0, static_cast<size_t>(tokens) * dim * sizeof(float), cuda_stream) != cudaSuccess) return false;
    if (routes == 0) return cudaGetLastError() == cudaSuccess;
    const size_t routes_dim = static_cast<size_t>(routes) * dim;
    const size_t padded_dim = static_cast<size_t>(workspace.padded_rows_cap) * dim;
    const size_t padded_inter = static_cast<size_t>(workspace.padded_rows_cap) * inter_dim;
    if (cudaMalloc(&workspace.d_x_sorted, routes_dim * sizeof(float)) != cudaSuccess) return false;
    if (cudaMalloc(&workspace.d_partials, routes_dim * sizeof(float)) != cudaSuccess) goto fail;
    if (cudaMalloc(&workspace.d_x_q, routes_dim) != cudaSuccess) goto fail;
    if (cudaMalloc(&workspace.d_x_scale, static_cast<size_t>(routes) * sizeof(float)) != cudaSuccess) goto fail;
    if (cudaMalloc(&workspace.d_x_pad, padded_dim) != cudaSuccess) goto fail;
    if (cudaMalloc(&workspace.d_x_scale_pad, static_cast<size_t>(workspace.padded_rows_cap) * sizeof(float)) != cudaSuccess) goto fail;
    if (cudaMalloc(&workspace.d_gate, padded_inter * sizeof(float)) != cudaSuccess) goto fail;
    if (cudaMalloc(&workspace.d_up, padded_inter * sizeof(float)) != cudaSuccess) goto fail;
    if (cudaMalloc(&workspace.d_hidden_q, padded_inter) != cudaSuccess) goto fail;
    if (cudaMalloc(&workspace.d_hidden_scale, static_cast<size_t>(workspace.padded_rows_cap) * sizeof(float)) != cudaSuccess) goto fail;
    {
        const bool ok = moe_prefill_fp4_grouped_cuda_with_workspace(
            d_x, d_route_tokens, d_route_weights, d_seg_starts,
            d_w1q, d_w1s, d_w2q, d_w2s, d_w3q, d_w3s,
            d_y, tokens, topk, routes, n_local_experts, max_count, dim, inter_dim, swiglu_limit,
            workspace, /*d_token_slot_routes=*/nullptr, stream);
        cudaFree(workspace.d_x_sorted);
        cudaFree(workspace.d_partials);
        cudaFree(workspace.d_x_q);
        cudaFree(workspace.d_x_scale);
        cudaFree(workspace.d_x_pad);
        cudaFree(workspace.d_x_scale_pad);
        cudaFree(workspace.d_gate);
        cudaFree(workspace.d_up);
        cudaFree(workspace.d_hidden_q);
        cudaFree(workspace.d_hidden_scale);
        return ok;
    }
fail:
    cudaFree(workspace.d_x_sorted);
    cudaFree(workspace.d_partials);
    cudaFree(workspace.d_x_q);
    cudaFree(workspace.d_x_scale);
    cudaFree(workspace.d_x_pad);
    cudaFree(workspace.d_x_scale_pad);
    cudaFree(workspace.d_gate);
    cudaFree(workspace.d_up);
    cudaFree(workspace.d_hidden_q);
    cudaFree(workspace.d_hidden_scale);
    return false;
}

bool moe_multi_token_fp4_cuda_with_workspace(
    const float* d_x,
    const int32_t* d_slot_expert,
    const int32_t* d_slot_starts,
    const int32_t* d_slot_tokens,
    const float* d_pair_weights,
    const uint8_t* d_w1q,
    const uint8_t* d_w1s,
    const uint8_t* d_w2q,
    const uint8_t* d_w2s,
    const uint8_t* d_w3q,
    const uint8_t* d_w3s,
    float* d_y,
    int tokens,
    int slots,
    int pairs,
    int n_local_experts,
    int dim,
    int inter_dim,
    float swiglu_limit,
    MoeMultiTokenFp4Workspace workspace,
    void* stream) {
    if (d_x == nullptr || d_slot_starts == nullptr || d_w1q == nullptr ||
        d_w1s == nullptr || d_w2q == nullptr || d_w2s == nullptr ||
        d_w3q == nullptr || d_w3s == nullptr || d_y == nullptr) return false;
    if (tokens <= 0 || slots < 0 || pairs < 0 || n_local_experts <= 0 ||
        dim <= 0 || inter_dim <= 0 || (dim % 32) != 0 ||
        (inter_dim % 32) != 0) return false;
    if (workspace.tokens_cap < tokens || workspace.pairs_cap < pairs ||
        workspace.dim < dim || workspace.inter_dim < inter_dim ||
        workspace.d_x_q == nullptr || workspace.d_x_scale == nullptr) return false;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    if (cudaMemsetAsync(d_y, 0, static_cast<size_t>(tokens) * dim * sizeof(float),
                        cuda_stream) != cudaSuccess) return false;
    if (slots == 0 || pairs == 0) return cudaGetLastError() == cudaSuccess;
    if (d_slot_expert == nullptr || d_slot_tokens == nullptr ||
        d_pair_weights == nullptr || workspace.d_gate == nullptr ||
        workspace.d_up == nullptr || workspace.d_hidden_q == nullptr ||
        workspace.d_hidden_scale == nullptr || workspace.d_partials == nullptr) return false;

    quantize_float_rows_kernel<<<tokens, kQuantThreads, 0, cuda_stream>>>(
        d_x, workspace.d_x_q, workspace.d_x_scale, tokens, dim);
    constexpr int kMultiMaxTokens = 8;
    const dim3 gemm_block(kGemmThreads);
    const dim3 w1w3_grid(ceil_div_int(inter_dim, kGemmThreads), slots);
    const size_t x_shared_bytes =
        static_cast<size_t>(kMultiMaxTokens) * (dim / 4) * sizeof(int);
    moe_multi_w1w3_fp4_kernel<<<w1w3_grid, gemm_block, x_shared_bytes, cuda_stream>>>(
        workspace.d_x_q, workspace.d_x_scale, d_slot_expert, d_slot_starts,
        d_slot_tokens, d_w1q, d_w1s, d_w3q, d_w3s, workspace.d_gate,
        workspace.d_up, dim, inter_dim);
    moe_multi_swiglu_quant_fp4_kernel<<<pairs, kQuantThreads, 0, cuda_stream>>>(
        workspace.d_gate, workspace.d_up, d_pair_weights, workspace.d_hidden_q,
        workspace.d_hidden_scale, pairs, inter_dim, swiglu_limit);
    const dim3 w2_grid(ceil_div_int(dim, kGemmThreads), slots);
    const size_t h_shared_bytes =
        static_cast<size_t>(kMultiMaxTokens) * (inter_dim / 4) * sizeof(int);
    moe_multi_w2_partial_fp4_kernel<<<w2_grid, gemm_block, h_shared_bytes, cuda_stream>>>(
        workspace.d_hidden_q, workspace.d_hidden_scale, d_slot_expert,
        d_slot_starts, d_w2q, d_w2s, workspace.d_partials, dim, inter_dim);
    const dim3 reduce_grid(ceil_div_int(dim, kGemmThreads), tokens);
    moe_multi_reduce_partials_kernel<<<reduce_grid, gemm_block, 0, cuda_stream>>>(
        workspace.d_partials, d_slot_tokens, d_y, pairs, dim);
    return cudaGetLastError() == cudaSuccess;
}

bool moe_single_token_fp4_cuda_with_workspace(
    const float* d_x,
    const int64_t* d_indices,
    const float* d_weights,
    const uint8_t* d_w1q,
    const uint8_t* d_w1s,
    const uint8_t* d_w2q,
    const uint8_t* d_w2s,
    const uint8_t* d_w3q,
    const uint8_t* d_w3s,
    float* d_y,
    int topk,
    int experts_start_idx,
    int n_local_experts,
    int dim,
    int inter_dim,
    float swiglu_limit,
    MoeSingleTokenFp4Workspace workspace,
    void* stream) {
    if (d_x == nullptr || d_indices == nullptr || d_weights == nullptr || d_w1q == nullptr || d_w1s == nullptr || d_w2q == nullptr || d_w2s == nullptr || d_w3q == nullptr || d_w3s == nullptr || d_y == nullptr) return false;
    if (workspace.d_x_q == nullptr || workspace.d_x_scale == nullptr || workspace.d_gate == nullptr || workspace.d_up == nullptr || workspace.d_hidden_q == nullptr || workspace.d_hidden_scale == nullptr || workspace.d_route_y == nullptr) return false;
    if (topk <= 0 || n_local_experts <= 0 || dim <= 0 || inter_dim <= 0 || (dim % 32) != 0 || (inter_dim % 32) != 0) return false;
    if (workspace.topk < topk || workspace.dim < dim || workspace.inter_dim < inter_dim) return false;
    auto cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    cudaMemsetAsync(d_y, 0, static_cast<size_t>(dim) * sizeof(float), cuda_stream);
    quantize_float_row_kernel<<<1, kQuantThreads, 0, cuda_stream>>>(d_x, workspace.d_x_q, workspace.d_x_scale, dim);
    const dim3 w1w3_grid(ceil_div_int(inter_dim, kGemmThreads), topk);
    const dim3 gemm_block(kGemmThreads);
    moe_single_w1w3_fp4_kernel<<<w1w3_grid, gemm_block, static_cast<size_t>(dim / 4) * sizeof(int), cuda_stream>>>(
        workspace.d_x_q, workspace.d_x_scale, d_indices, d_w1q, d_w1s, d_w3q, d_w3s, workspace.d_gate, workspace.d_up, topk, experts_start_idx, n_local_experts, dim, inter_dim);
    moe_single_swiglu_quant_fp4_kernel<<<topk, kQuantThreads, 0, cuda_stream>>>(
        workspace.d_gate, workspace.d_up, d_indices, d_weights, workspace.d_hidden_q, workspace.d_hidden_scale, topk, experts_start_idx, n_local_experts, inter_dim, swiglu_limit);
    const dim3 w2_grid(ceil_div_int(dim, kGemmThreads), topk);
    moe_single_w2_write_fp4_kernel<<<w2_grid, gemm_block, static_cast<size_t>(inter_dim / 4) * sizeof(int), cuda_stream>>>(
        workspace.d_hidden_q, workspace.d_hidden_scale, d_indices, d_w2q, d_w2s, workspace.d_route_y, topk, experts_start_idx, n_local_experts, dim, inter_dim);
    sum_topk_rows_kernel<<<ceil_div_int(dim, 256), 256, 0, cuda_stream>>>(workspace.d_route_y, d_y, topk, dim);
    return cudaGetLastError() == cudaSuccess;
}

bool moe_single_token_fp4_cuda(
    const float* d_x,
    const int64_t* d_indices,
    const float* d_weights,
    const uint8_t* d_w1q,
    const uint8_t* d_w1s,
    const uint8_t* d_w2q,
    const uint8_t* d_w2s,
    const uint8_t* d_w3q,
    const uint8_t* d_w3s,
    float* d_y,
    int topk,
    int experts_start_idx,
    int n_local_experts,
    int dim,
    int inter_dim,
    float swiglu_limit,
    void* stream) {
    if (topk <= 0 || dim <= 0 || inter_dim <= 0) return false;
    MoeSingleTokenFp4Workspace workspace;
    workspace.topk = topk;
    workspace.dim = dim;
    workspace.inter_dim = inter_dim;
    if (cudaMalloc(&workspace.d_x_q, static_cast<size_t>(dim)) != cudaSuccess) return false;
    if (cudaMalloc(&workspace.d_x_scale, sizeof(float)) != cudaSuccess) return false;
    if (cudaMalloc(&workspace.d_gate, static_cast<size_t>(topk) * inter_dim * sizeof(float)) != cudaSuccess) return false;
    if (cudaMalloc(&workspace.d_up, static_cast<size_t>(topk) * inter_dim * sizeof(float)) != cudaSuccess) return false;
    if (cudaMalloc(&workspace.d_hidden_q, static_cast<size_t>(topk) * inter_dim) != cudaSuccess) return false;
    if (cudaMalloc(&workspace.d_hidden_scale, static_cast<size_t>(topk) * sizeof(float)) != cudaSuccess) return false;
    if (cudaMalloc(&workspace.d_route_y, static_cast<size_t>(topk) * dim * sizeof(float)) != cudaSuccess) return false;
    const bool ok = moe_single_token_fp4_cuda_with_workspace(d_x, d_indices, d_weights, d_w1q, d_w1s, d_w2q, d_w2s, d_w3q, d_w3s, d_y, topk, experts_start_idx, n_local_experts, dim, inter_dim, swiglu_limit, workspace, stream);
    cudaFree(workspace.d_x_q);
    cudaFree(workspace.d_x_scale);
    cudaFree(workspace.d_gate);
    cudaFree(workspace.d_up);
    cudaFree(workspace.d_hidden_q);
    cudaFree(workspace.d_hidden_scale);
    cudaFree(workspace.d_route_y);
    return ok;
}

}  // namespace pocket
