// IQ2_XXS / Q2_K single-token MoE decode kernels for cpp_engine.
//
// Ported from src/csrc/cuda_kernel_impl.cu (gguf_moe_single_w13_iq2_xxs_dp4a_kernel
// and gguf_moe_single_w2_q2k_dp4a_kernel).
//
// Layout assumptions match the GGUF on-disk blocks:
//   * IQ2_XXS block = 66 bytes for 256 weight elements:
//       [0..1] f16 d, [2..9] chunk0, [10..17] chunk1, ... [58..65] chunk7
//       each chunk: bytes 0..3 = grid_id, bytes 4..7 = packed aux uint32
//   * Q2_K block = 84 bytes for 256 weight elements:
//       [0..15] 16 byte scales (low4=q-scale, high4=min-scale),
//       [16..79] 64 bytes 2-bit qs (16 groups × 16 elems),
//       [80..81] f16 d, [82..83] f16 dmin
//
// The kernels follow the exact DP4A math used by the PyTorch path so a single
// expert's compute output matches PyTorch ext kernel within float rounding.

#include "cuda_ops.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>

namespace pocket {

namespace {

// --- IQ2_XXS lookup table (256 × 8 bytes), expanded into signed_grid lazily. ---
// Source: src/gguf/tensor_reader.py:_IQ2XXS_GRID.

static const uint64_t kIQ2XXSGrid[256] = {
    0x0808080808080808ULL, 0x080808080808082bULL, 0x0808080808081919ULL, 0x0808080808082b08ULL,
    0x0808080808082b2bULL, 0x0808080808190819ULL, 0x0808080808191908ULL, 0x08080808082b0808ULL,
    0x08080808082b082bULL, 0x08080808082b2b08ULL, 0x08080808082b2b2bULL, 0x0808080819080819ULL,
    0x0808080819081908ULL, 0x0808080819190808ULL, 0x0808080819192b08ULL, 0x08080808192b0819ULL,
    0x08080808192b1908ULL, 0x080808082b080808ULL, 0x080808082b08082bULL, 0x080808082b082b2bULL,
    0x080808082b2b082bULL, 0x0808081908080819ULL, 0x0808081908081908ULL, 0x0808081908190808ULL,
    0x0808081908191919ULL, 0x0808081919080808ULL, 0x080808192b081908ULL, 0x080808192b192b08ULL,
    0x0808082b08080808ULL, 0x0808082b0808082bULL, 0x0808082b082b082bULL, 0x0808082b2b08082bULL,
    0x0808190808080819ULL, 0x0808190808081908ULL, 0x0808190808190808ULL, 0x08081908082b0819ULL,
    0x08081908082b1908ULL, 0x0808190819080808ULL, 0x080819081908082bULL, 0x0808190819082b08ULL,
    0x08081908192b0808ULL, 0x080819082b080819ULL, 0x080819082b081908ULL, 0x080819082b190808ULL,
    0x080819082b2b1908ULL, 0x0808191908080808ULL, 0x080819190808082bULL, 0x0808191908082b08ULL,
    0x08081919082b0808ULL, 0x080819191908192bULL, 0x08081919192b2b19ULL, 0x080819192b080808ULL,
    0x080819192b190819ULL, 0x0808192b08082b19ULL, 0x0808192b08190808ULL, 0x0808192b19080808ULL,
    0x0808192b2b081908ULL, 0x0808192b2b2b1908ULL, 0x08082b0808080808ULL, 0x08082b0808081919ULL,
    0x08082b0808082b08ULL, 0x08082b0808191908ULL, 0x08082b08082b2b08ULL, 0x08082b0819080819ULL,
    0x08082b0819081908ULL, 0x08082b0819190808ULL, 0x08082b081919082bULL, 0x08082b082b082b08ULL,
    0x08082b1908081908ULL, 0x08082b1919080808ULL, 0x08082b2b0808082bULL, 0x08082b2b08191908ULL,
    0x0819080808080819ULL, 0x0819080808081908ULL, 0x0819080808190808ULL, 0x08190808082b0819ULL,
    0x0819080819080808ULL, 0x08190808192b0808ULL, 0x081908082b081908ULL, 0x081908082b190808ULL,
    0x081908082b191919ULL, 0x0819081908080808ULL, 0x0819081908082b08ULL, 0x08190819082b0808ULL,
    0x0819081919190808ULL, 0x0819081919192b2bULL, 0x081908192b080808ULL, 0x0819082b082b1908ULL,
    0x0819082b19081919ULL, 0x0819190808080808ULL, 0x0819190808082b08ULL, 0x08191908082b0808ULL,
    0x08191908082b1919ULL, 0x0819190819082b19ULL, 0x081919082b080808ULL, 0x0819191908192b08ULL,
    0x08191919192b082bULL, 0x0819192b08080808ULL, 0x0819192b0819192bULL, 0x08192b0808080819ULL,
    0x08192b0808081908ULL, 0x08192b0808190808ULL, 0x08192b0819080808ULL, 0x08192b082b080819ULL,
    0x08192b1908080808ULL, 0x08192b1908081919ULL, 0x08192b192b2b0808ULL, 0x08192b2b19190819ULL,
    0x082b080808080808ULL, 0x082b08080808082bULL, 0x082b080808082b2bULL, 0x082b080819081908ULL,
    0x082b0808192b0819ULL, 0x082b08082b080808ULL, 0x082b08082b08082bULL, 0x082b0819082b2b19ULL,
    0x082b081919082b08ULL, 0x082b082b08080808ULL, 0x082b082b0808082bULL, 0x082b190808080819ULL,
    0x082b190808081908ULL, 0x082b190808190808ULL, 0x082b190819080808ULL, 0x082b19081919192bULL,
    0x082b191908080808ULL, 0x082b191919080819ULL, 0x082b1919192b1908ULL, 0x082b192b2b190808ULL,
    0x082b2b0808082b08ULL, 0x082b2b08082b0808ULL, 0x082b2b082b191908ULL, 0x082b2b2b19081908ULL,
    0x1908080808080819ULL, 0x1908080808081908ULL, 0x1908080808190808ULL, 0x1908080808192b08ULL,
    0x19080808082b0819ULL, 0x19080808082b1908ULL, 0x1908080819080808ULL, 0x1908080819082b08ULL,
    0x190808081919192bULL, 0x19080808192b0808ULL, 0x190808082b080819ULL, 0x190808082b081908ULL,
    0x190808082b190808ULL, 0x1908081908080808ULL, 0x19080819082b0808ULL, 0x19080819192b0819ULL,
    0x190808192b080808ULL, 0x190808192b081919ULL, 0x1908082b08080819ULL, 0x1908082b08190808ULL,
    0x1908082b19082b08ULL, 0x1908082b1919192bULL, 0x1908082b192b2b08ULL, 0x1908190808080808ULL,
    0x1908190808082b08ULL, 0x19081908082b0808ULL, 0x190819082b080808ULL, 0x190819082b192b19ULL,
    0x190819190819082bULL, 0x19081919082b1908ULL, 0x1908192b08080808ULL, 0x19082b0808080819ULL,
    0x19082b0808081908ULL, 0x19082b0808190808ULL, 0x19082b0819080808ULL, 0x19082b0819081919ULL,
    0x19082b1908080808ULL, 0x19082b1919192b08ULL, 0x19082b19192b0819ULL, 0x19082b192b08082bULL,
    0x19082b2b19081919ULL, 0x19082b2b2b190808ULL, 0x1919080808080808ULL, 0x1919080808082b08ULL,
    0x1919080808190819ULL, 0x1919080808192b19ULL, 0x19190808082b0808ULL, 0x191908082b080808ULL,
    0x191908082b082b08ULL, 0x1919081908081908ULL, 0x191908191908082bULL, 0x191908192b2b1908ULL,
    0x1919082b2b190819ULL, 0x191919082b190808ULL, 0x191919082b19082bULL, 0x1919191908082b2bULL,
    0x1919192b08080819ULL, 0x1919192b19191908ULL, 0x19192b0808080808ULL, 0x19192b0808190819ULL,
    0x19192b0808192b19ULL, 0x19192b08192b1908ULL, 0x19192b1919080808ULL, 0x19192b2b08082b08ULL,
    0x192b080808081908ULL, 0x192b080808190808ULL, 0x192b080819080808ULL, 0x192b0808192b2b08ULL,
    0x192b081908080808ULL, 0x192b081919191919ULL, 0x192b082b08192b08ULL, 0x192b082b192b0808ULL,
    0x192b190808080808ULL, 0x192b190808081919ULL, 0x192b191908190808ULL, 0x192b19190819082bULL,
    0x192b19192b081908ULL, 0x192b2b081908082bULL, 0x2b08080808080808ULL, 0x2b0808080808082bULL,
    0x2b08080808082b2bULL, 0x2b08080819080819ULL, 0x2b0808082b08082bULL, 0x2b08081908081908ULL,
    0x2b08081908192b08ULL, 0x2b08081919080808ULL, 0x2b08082b08190819ULL, 0x2b08190808080819ULL,
    0x2b08190808081908ULL, 0x2b08190808190808ULL, 0x2b08190808191919ULL, 0x2b08190819080808ULL,
    0x2b081908192b0808ULL, 0x2b08191908080808ULL, 0x2b0819191908192bULL, 0x2b0819192b191908ULL,
    0x2b08192b08082b19ULL, 0x2b08192b19080808ULL, 0x2b08192b192b0808ULL, 0x2b082b080808082bULL,
    0x2b082b1908081908ULL, 0x2b082b2b08190819ULL, 0x2b19080808081908ULL, 0x2b19080808190808ULL,
    0x2b190808082b1908ULL, 0x2b19080819080808ULL, 0x2b1908082b2b0819ULL, 0x2b1908190819192bULL,
    0x2b1908192b080808ULL, 0x2b19082b19081919ULL, 0x2b19190808080808ULL, 0x2b191908082b082bULL,
    0x2b19190819081908ULL, 0x2b19191919190819ULL, 0x2b192b082b080819ULL, 0x2b192b19082b0808ULL,
    0x2b2b08080808082bULL, 0x2b2b080819190808ULL, 0x2b2b08082b081919ULL, 0x2b2b081908082b19ULL,
    0x2b2b082b08080808ULL, 0x2b2b190808192b08ULL, 0x2b2b2b0819190808ULL, 0x2b2b2b1908081908ULL,
};

// Constants borrowed from the PyTorch kernel.
constexpr int kQ2TileN = 8;  // output cols per thread block (mirror kGGUFQuantTileN)

bool env_enabled_q2(const char* name, bool default_value = false) {
    const char* v = std::getenv(name);
    if (v == nullptr) return default_value;
    return !(v[0] == '\0' || v[0] == '0' || v[0] == 'f' || v[0] == 'F' || v[0] == 'n' || v[0] == 'N');
}

// signed_grid layout: [256 grid_ids][128 sign_indices][8 int8 elems] = 262144 bytes.
// Total ~256 KiB — too large for __constant__ memory on Turing (64 KiB limit), so
// we keep it in regular device global memory and lazily initialize once per process.
static int8_t* g_signed_grid_device = nullptr;
static std::once_flag g_signed_grid_init_flag;

void build_signed_grid_host(int8_t* out) {
    // out shape: [256, 128, 8] int8.
    for (int grid_id = 0; grid_id < 256; ++grid_id) {
        uint64_t base_value = kIQ2XXSGrid[grid_id];
        int8_t base[8];
        for (int b = 0; b < 8; ++b) {
            base[b] = static_cast<int8_t>(static_cast<uint8_t>((base_value >> (8 * b)) & 0xff));
        }
        for (int sign_idx = 0; sign_idx < 128; ++sign_idx) {
            // sign_mask = idx | ((popcount(idx) & 1) << 7) — extends the 7-bit sign to 8.
            const int popcount = __builtin_popcount(sign_idx);
            const int sign_mask = sign_idx | ((popcount & 1) << 7);
            for (int b = 0; b < 8; ++b) {
                const int s = (sign_mask >> b) & 1;
                const int8_t sign = s ? -1 : 1;
                out[(grid_id * 128 + sign_idx) * 8 + b] = static_cast<int8_t>(base[b] * sign);
            }
        }
    }
}

const int8_t* signed_grid_device() {
    std::call_once(g_signed_grid_init_flag, [] {
        constexpr size_t bytes = 256ULL * 128 * 8;
        int8_t* host = static_cast<int8_t*>(std::malloc(bytes));
        if (host == nullptr) return;
        build_signed_grid_host(host);
        if (cudaMalloc(&g_signed_grid_device, bytes) != cudaSuccess) {
            std::free(host);
            g_signed_grid_device = nullptr;
            return;
        }
        if (cudaMemcpy(g_signed_grid_device, host, bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
            cudaFree(g_signed_grid_device);
            g_signed_grid_device = nullptr;
        }
        std::free(host);
    });
    return g_signed_grid_device;
}

__device__ __forceinline__ float gguf_block_scale_f16(const uint8_t* ptr) {
    const uint16_t bits = static_cast<uint16_t>(ptr[0]) | (static_cast<uint16_t>(ptr[1]) << 8);
    return __half2float(__ushort_as_half(bits));
}

__device__ __forceinline__ int pack_i8x4(int a, int b, int c, int d) {
    return (a & 0xff) | ((b & 0xff) << 8) | ((c & 0xff) << 16) | ((d & 0xff) << 24);
}

// Quantize one row of fp32 activations to Q8_1 (int8 + per-32-group fp32 scale).
__global__ void gguf_q8_1_quantize_x_32_kernel(
    const float* __restrict__ x,
    int8_t* __restrict__ x_q,
    float* __restrict__ x_scale,
    int rows,
    int row_elems) {
    const int row = blockIdx.x;
    const int group = blockIdx.y;
    const int lane = threadIdx.x;
    if (row >= rows || group * 32 >= row_elems || lane >= 32) return;
    const int k = group * 32 + lane;
    const float xv = k < row_elems ? x[static_cast<int64_t>(row) * row_elems + k] : 0.0f;
    float maxv = fabsf(xv);
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        maxv = fmaxf(maxv, __shfl_down_sync(0xffffffff, maxv, offset));
    }
    const float scale = __shfl_sync(0xffffffff, maxv > 0.0f ? maxv / 127.0f : 1.0e-8f, 0);
    if (lane == 0) {
        x_scale[row * ((row_elems + 31) / 32) + group] = scale;
    }
    int q = __float2int_rn(xv / scale);
    q = max(-127, min(127, q));
    if (k < row_elems) {
        x_q[static_cast<int64_t>(row) * row_elems + k] = static_cast<int8_t>(q);
    }
}

// Quantize one row of fp32 to Q8_1 with 16-element groups (post-SwiGLU hidden).
__global__ void gguf_q8_1_quantize_hidden_16_kernel(
    const float* __restrict__ hidden,
    int8_t* __restrict__ hidden_q,
    float* __restrict__ hidden_scale,
    int routes,
    int inter_dim) {
    const int route = blockIdx.x;
    const int group = blockIdx.y;
    const int lane = threadIdx.x;
    if (route >= routes || group * 16 >= inter_dim || lane >= 16) return;
    const int k = group * 16 + lane;
    const float xv = k < inter_dim ? hidden[static_cast<int64_t>(route) * inter_dim + k] : 0.0f;
    float maxv = fabsf(xv);
    #pragma unroll
    for (int offset = 8; offset > 0; offset >>= 1) {
        maxv = fmaxf(maxv, __shfl_down_sync(0xffff, maxv, offset));
    }
    const float scale = __shfl_sync(0xffff, maxv > 0.0f ? maxv / 127.0f : 1.0e-8f, 0);
    if (lane == 0) {
        hidden_scale[route * ((inter_dim + 15) / 16) + group] = scale;
    }
    int q = __float2int_rn(xv / scale);
    q = max(-127, min(127, q));
    hidden_q[static_cast<int64_t>(route) * inter_dim + k] = static_cast<int8_t>(q);
}

// Fused SwiGLU + per-route weight scale + Q8_1 quantize hidden (group=16).
__global__ void gguf_route_swiglu_quantize_hidden_16_kernel(
    const float* __restrict__ gate,
    const float* __restrict__ up,
    const float* __restrict__ route_weights,
    int8_t* __restrict__ hidden_q,
    float* __restrict__ hidden_scale,
    int routes,
    int inter_dim,
    float swiglu_limit) {
    const int route = blockIdx.x;
    const int group = blockIdx.y;
    const int lane = threadIdx.x;
    if (route >= routes || group * 16 >= inter_dim || lane >= 16) return;
    const int k = group * 16 + lane;
    float g = k < inter_dim ? gate[static_cast<int64_t>(route) * inter_dim + k] : 0.0f;
    float u = k < inter_dim ? up[static_cast<int64_t>(route) * inter_dim + k] : 0.0f;
    if (swiglu_limit > 0.0f) {
        u = fminf(fmaxf(u, -swiglu_limit), swiglu_limit);
        g = fminf(g, swiglu_limit);
    }
    const float v = (g / (1.0f + expf(-g))) * u * route_weights[route];
    float maxv = fabsf(v);
    #pragma unroll
    for (int offset = 8; offset > 0; offset >>= 1) {
        maxv = fmaxf(maxv, __shfl_down_sync(0xffff, maxv, offset));
    }
    const float scale = __shfl_sync(0xffff, maxv > 0.0f ? maxv / 127.0f : 1.0e-8f, 0);
    if (lane == 0) {
        hidden_scale[route * ((inter_dim + 15) / 16) + group] = scale;
    }
    int q = __float2int_rn(v / scale);
    q = max(-127, min(127, q));
    if (k < inter_dim) {
        hidden_q[static_cast<int64_t>(route) * inter_dim + k] = static_cast<int8_t>(q);
    }
}

// IQ2_XXS W1+W3 fused MoE single-token matvec.
// Layout: w1_blocks/w3_blocks per-expert [inter_dim, blocks_per_row*66] flat bytes.
__global__ void gguf_moe_single_w13_iq2_xxs_dp4a_kernel(
    const int8_t* __restrict__ x_q,
    const float* __restrict__ x_scale,
    const int64_t* __restrict__ route_slots,
    const uint8_t* __restrict__ w1_blocks,
    const uint8_t* __restrict__ w3_blocks,
    const int8_t* __restrict__ signed_grid,
    float* __restrict__ gate,
    float* __restrict__ up,
    int routes,
    int n_experts,
    int dim,
    int inter_dim,
    int blocks_per_row,
    int x_groups) {
    const int route = blockIdx.y;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int out_col = blockIdx.x * kQ2TileN + warp;
    if (route >= routes || warp >= kQ2TileN) return;
    const int expert = static_cast<int>(route_slots[route]);
    if (expert < 0 || expert >= n_experts) return;

    const uint8_t* w1_row = w1_blocks + (static_cast<int64_t>(expert) * inter_dim + out_col) * blocks_per_row * 66;
    const uint8_t* w3_row = w3_blocks + (static_cast<int64_t>(expert) * inter_dim + out_col) * blocks_per_row * 66;

    __shared__ int x_shared_q[64];
    __shared__ float x_shared_scale[8];

    float acc1 = 0.0f;
    float acc3 = 0.0f;

    for (int block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
        const int k_base = block_idx * 256;
        const int tid = threadIdx.x;
        if (tid < 64) {
            const int byte_off = tid * 4;
            int v = 0;
            if (k_base + byte_off + 4 <= dim) {
                v = *reinterpret_cast<const int*>(x_q + k_base + byte_off);
            }
            x_shared_q[tid] = v;
        }
        if (tid < 8) {
            const int sg_idx = block_idx * 8 + tid;
            x_shared_scale[tid] = sg_idx < x_groups ? x_scale[sg_idx] : 0.0f;
        }
        __syncthreads();

        if (out_col < inter_dim) {
            const int sub = lane >> 2;
            const int part = lane & 3;
            const uint8_t* w1_block = w1_row + static_cast<int64_t>(block_idx) * 66;
            const uint8_t* w3_block = w3_row + static_cast<int64_t>(block_idx) * 66;
            const uint8_t* chunk1 = w1_block + 2 + sub * 8;
            const uint8_t* chunk3 = w3_block + 2 + sub * 8;

            const float d1 = gguf_block_scale_f16(w1_block);
            const float d3 = gguf_block_scale_f16(w3_block);

            const uint32_t aux1 = static_cast<uint32_t>(chunk1[4]) |
                (static_cast<uint32_t>(chunk1[5]) << 8) |
                (static_cast<uint32_t>(chunk1[6]) << 16) |
                (static_cast<uint32_t>(chunk1[7]) << 24);
            const uint32_t aux3 = static_cast<uint32_t>(chunk3[4]) |
                (static_cast<uint32_t>(chunk3[5]) << 8) |
                (static_cast<uint32_t>(chunk3[6]) << 16) |
                (static_cast<uint32_t>(chunk3[7]) << 24);

            const int grid_id1 = chunk1[part];
            const int grid_id3 = chunk3[part];
            const int sign_idx1 = static_cast<int>((aux1 >> (7 * part)) & 127);
            const int sign_idx3 = static_cast<int>((aux3 >> (7 * part)) & 127);

            const float s1 = 0.125f * d1 * static_cast<float>(2 * (aux1 >> 28) + 1);
            const float s3 = 0.125f * d3 * static_cast<float>(2 * (aux3 >> 28) + 1);

            const int8_t* vals1 = signed_grid + (grid_id1 * 128 + sign_idx1) * 8;
            const int8_t* vals3 = signed_grid + (grid_id3 * 128 + sign_idx3) * 8;
            const int v1_p0 = *reinterpret_cast<const int*>(vals1);
            const int v1_p1 = *reinterpret_cast<const int*>(vals1 + 4);
            const int v3_p0 = *reinterpret_cast<const int*>(vals3);
            const int v3_p1 = *reinterpret_cast<const int*>(vals3 + 4);

            const int xq_base = sub * 8 + part * 2;
            const int x_p0 = x_shared_q[xq_base];
            const int x_p1 = x_shared_q[xq_base + 1];

            int sumi1 = __dp4a(v1_p0, x_p0, 0);
            sumi1 = __dp4a(v1_p1, x_p1, sumi1);
            int sumi3 = __dp4a(v3_p0, x_p0, 0);
            sumi3 = __dp4a(v3_p1, x_p1, sumi3);

            const float xs = x_shared_scale[sub];
            float local1 = s1 * xs * static_cast<float>(sumi1);
            float local3 = s3 * xs * static_cast<float>(sumi3);

            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                local1 += __shfl_down_sync(0xffffffff, local1, offset);
                local3 += __shfl_down_sync(0xffffffff, local3, offset);
            }
            if (lane == 0) {
                acc1 += local1;
                acc3 += local3;
            }
        }
        __syncthreads();
    }

    if (lane == 0 && out_col < inter_dim) {
        gate[static_cast<int64_t>(route) * inter_dim + out_col] = acc1;
        up[static_cast<int64_t>(route) * inter_dim + out_col] = acc3;
    }
}

// Q2_K W2 single-token matvec (per-route hidden -> per-token output).
// Output is accumulated via atomicAdd onto y; caller must zero y beforehand.
__global__ void gguf_moe_single_w2_q2k_dp4a_kernel(
    const int8_t* __restrict__ hidden_q,
    const float* __restrict__ hidden_scale,
    const int64_t* __restrict__ route_slots,
    const uint8_t* __restrict__ w2_blocks,
    float* __restrict__ y,
    int routes,
    int n_experts,
    int dim,
    int inter_dim,
    int w2_blocks_per_row,
    int hidden_groups) {
    const int route = blockIdx.y;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int out_col = blockIdx.x * kQ2TileN + warp;
    if (route >= routes || warp >= kQ2TileN || out_col >= dim) return;
    const int expert = static_cast<int>(route_slots[route]);
    if (expert < 0 || expert >= n_experts) return;

    const int8_t* hidden_row = hidden_q + static_cast<int64_t>(route) * inter_dim;
    const float* hs_row = hidden_scale + static_cast<int64_t>(route) * hidden_groups;
    const uint8_t* w2_row = w2_blocks + (static_cast<int64_t>(expert) * dim + out_col) * w2_blocks_per_row * 84;
    float acc = 0.0f;

    for (int block_idx = 0; block_idx < w2_blocks_per_row; ++block_idx) {
        const uint8_t* block = w2_row + static_cast<int64_t>(block_idx) * 84;
        const uint8_t* scales = block;
        const uint8_t* qs = block + 16;
        const float d = gguf_block_scale_f16(block + 80);
        const float dmin = gguf_block_scale_f16(block + 82);
        const int k_base = block_idx * 256;
        for (int half = 0; half < 2; ++half) {
            const int sub = lane >> 2;      // eight 4-lane groups per half-block.
            const int part = lane & 3;
            const int group = half * 8 + sub;
            const int shift = (sub >> 1) * 2;
            const int byte_start = half * 32 + (sub & 1) * 16;
            const int idx = part * 4;
            const int q0 = static_cast<int>((qs[byte_start + idx + 0] >> shift) & 0x03);
            const int q1 = static_cast<int>((qs[byte_start + idx + 1] >> shift) & 0x03);
            const int q2 = static_cast<int>((qs[byte_start + idx + 2] >> shift) & 0x03);
            const int q3 = static_cast<int>((qs[byte_start + idx + 3] >> shift) & 0x03);
            const int w_pack = pack_i8x4(q0, q1, q2, q3);
            const int h_idx = k_base + group * 16 + idx;
            const int h_pack = pack_i8x4(
                static_cast<int>(hidden_row[h_idx + 0]),
                static_cast<int>(hidden_row[h_idx + 1]),
                static_cast<int>(hidden_row[h_idx + 2]),
                static_cast<int>(hidden_row[h_idx + 3]));
            const int dot_q = __dp4a(w_pack, h_pack, 0);
            const int sum_h = __dp4a(0x01010101, h_pack, 0);
            const float qscale = d * static_cast<float>(scales[group] & 0x0f);
            const float base = dmin * static_cast<float>(scales[group] >> 4);
            float local = hs_row[block_idx * 16 + group] * (qscale * static_cast<float>(dot_q) - base * static_cast<float>(sum_h));
            const unsigned group_mask = 0x0fu << (sub * 4);
            local += __shfl_down_sync(group_mask, local, 2, 4);
            local += __shfl_down_sync(group_mask, local, 1, 4);
            float group_sum = (part == 0) ? local : 0.0f;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                group_sum += __shfl_down_sync(0xffffffff, group_sum, offset);
            }
            if (lane == 0) acc += group_sum;
        }
    }

    if (lane == 0) {
        atomicAdd(y + out_col, acc);
    }
}

__global__ void gather_q2_route_rows_kernel(
    const float* __restrict__ x_rows,
    const int64_t* __restrict__ route_tokens,
    float* __restrict__ x_route_rows,
    int routes,
    int dim) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t total = static_cast<int64_t>(routes) * dim;
    if (idx >= total) return;
    const int route = idx / dim;
    const int col = idx - route * dim;
    const int64_t token = route_tokens[route];
    x_route_rows[idx] = token >= 0 ? x_rows[token * static_cast<int64_t>(dim) + col] : 0.0f;
}

__global__ void gguf_moe_grouped_w13_iq2_xxs_dp4a_kernel(
    const int8_t* __restrict__ x_q,
    const float* __restrict__ x_scale,
    const int64_t* __restrict__ route_slots,
    const uint8_t* __restrict__ w1_blocks,
    const uint8_t* __restrict__ w3_blocks,
    const int8_t* __restrict__ signed_grid,
    float* __restrict__ gate,
    float* __restrict__ up,
    int routes,
    int n_experts,
    int dim,
    int inter_dim,
    int blocks_per_row,
    int x_groups) {
    const int route = blockIdx.y;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int out_col = blockIdx.x * kQ2TileN + warp;
    if (route >= routes || warp >= kQ2TileN) return;
    const int expert = static_cast<int>(route_slots[route]);
    if (expert < 0 || expert >= n_experts) return;

    const int8_t* xq_row = x_q + static_cast<int64_t>(route) * dim;
    const float* xs_row = x_scale + static_cast<int64_t>(route) * x_groups;
    const uint8_t* w1_row = w1_blocks + (static_cast<int64_t>(expert) * inter_dim + out_col) * blocks_per_row * 66;
    const uint8_t* w3_row = w3_blocks + (static_cast<int64_t>(expert) * inter_dim + out_col) * blocks_per_row * 66;

    __shared__ int x_shared_q[64];
    __shared__ float x_shared_scale[8];
    float acc1 = 0.0f;
    float acc3 = 0.0f;
    for (int block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
        const int k_base = block_idx * 256;
        const int tid = threadIdx.x;
        if (tid < 64) {
            const int byte_off = tid * 4;
            int v = 0;
            if (k_base + byte_off + 4 <= dim) {
                v = *reinterpret_cast<const int*>(xq_row + k_base + byte_off);
            }
            x_shared_q[tid] = v;
        }
        if (tid < 8) {
            const int sg_idx = block_idx * 8 + tid;
            x_shared_scale[tid] = sg_idx < x_groups ? xs_row[sg_idx] : 0.0f;
        }
        __syncthreads();
        if (out_col < inter_dim) {
            const int sub = lane >> 2;
            const int part = lane & 3;
            const uint8_t* w1_block = w1_row + static_cast<int64_t>(block_idx) * 66;
            const uint8_t* w3_block = w3_row + static_cast<int64_t>(block_idx) * 66;
            const uint8_t* chunk1 = w1_block + 2 + sub * 8;
            const uint8_t* chunk3 = w3_block + 2 + sub * 8;
            const float d1 = gguf_block_scale_f16(w1_block);
            const float d3 = gguf_block_scale_f16(w3_block);
            const uint32_t aux1 = static_cast<uint32_t>(chunk1[4]) |
                (static_cast<uint32_t>(chunk1[5]) << 8) |
                (static_cast<uint32_t>(chunk1[6]) << 16) |
                (static_cast<uint32_t>(chunk1[7]) << 24);
            const uint32_t aux3 = static_cast<uint32_t>(chunk3[4]) |
                (static_cast<uint32_t>(chunk3[5]) << 8) |
                (static_cast<uint32_t>(chunk3[6]) << 16) |
                (static_cast<uint32_t>(chunk3[7]) << 24);
            const int grid_id1 = chunk1[part];
            const int grid_id3 = chunk3[part];
            const int sign_idx1 = static_cast<int>((aux1 >> (7 * part)) & 127);
            const int sign_idx3 = static_cast<int>((aux3 >> (7 * part)) & 127);
            const float s1 = 0.125f * d1 * static_cast<float>(2 * (aux1 >> 28) + 1);
            const float s3 = 0.125f * d3 * static_cast<float>(2 * (aux3 >> 28) + 1);
            const int8_t* vals1 = signed_grid + (grid_id1 * 128 + sign_idx1) * 8;
            const int8_t* vals3 = signed_grid + (grid_id3 * 128 + sign_idx3) * 8;
            const int v1_p0 = *reinterpret_cast<const int*>(vals1);
            const int v1_p1 = *reinterpret_cast<const int*>(vals1 + 4);
            const int v3_p0 = *reinterpret_cast<const int*>(vals3);
            const int v3_p1 = *reinterpret_cast<const int*>(vals3 + 4);
            const int xq_base = sub * 8 + part * 2;
            const int x_p0 = x_shared_q[xq_base];
            const int x_p1 = x_shared_q[xq_base + 1];
            int sumi1 = __dp4a(v1_p0, x_p0, 0);
            sumi1 = __dp4a(v1_p1, x_p1, sumi1);
            int sumi3 = __dp4a(v3_p0, x_p0, 0);
            sumi3 = __dp4a(v3_p1, x_p1, sumi3);
            const float xs = x_shared_scale[sub];
            float local1 = s1 * xs * static_cast<float>(sumi1);
            float local3 = s3 * xs * static_cast<float>(sumi3);
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                local1 += __shfl_down_sync(0xffffffff, local1, offset);
                local3 += __shfl_down_sync(0xffffffff, local3, offset);
            }
            if (lane == 0) {
                acc1 += local1;
                acc3 += local3;
            }
        }
        __syncthreads();
    }
    if (lane == 0 && out_col < inter_dim) {
        gate[static_cast<int64_t>(route) * inter_dim + out_col] = acc1;
        up[static_cast<int64_t>(route) * inter_dim + out_col] = acc3;
    }
}

__global__ void gguf_moe_grouped_w13_iq2_xxs_dp4a_expert_tile_kernel(
    const int8_t* __restrict__ x_q,
    const float* __restrict__ x_scale,
    const int32_t* __restrict__ seg_starts,
    const uint8_t* __restrict__ w1_blocks,
    const uint8_t* __restrict__ w3_blocks,
    const int8_t* __restrict__ signed_grid,
    float* __restrict__ gate,
    float* __restrict__ up,
    int n_experts,
    int dim,
    int inter_dim,
    int blocks_per_row,
    int x_groups) {
    constexpr int kRouteTile = 8;
    const int expert = blockIdx.y;
    const int route_tile = blockIdx.z;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int out_col = blockIdx.x * kQ2TileN + warp;
    if (expert >= n_experts || warp >= kQ2TileN) return;

    const int route_start = seg_starts[expert] + route_tile * kRouteTile;
    const int route_end = seg_starts[expert + 1];
    const int valid_rows = min(kRouteTile, route_end - route_start);
    if (valid_rows <= 0) return;

    __shared__ int x_shared_q[kRouteTile][64];
    __shared__ float x_shared_scale[kRouteTile][8];

    float acc1[kRouteTile];
    float acc3[kRouteTile];
    #pragma unroll
    for (int r = 0; r < kRouteTile; ++r) {
        acc1[r] = 0.0f;
        acc3[r] = 0.0f;
    }

    const uint8_t* w1_row = w1_blocks + (static_cast<int64_t>(expert) * inter_dim + out_col) * blocks_per_row * 66;
    const uint8_t* w3_row = w3_blocks + (static_cast<int64_t>(expert) * inter_dim + out_col) * blocks_per_row * 66;

    for (int block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
        const int k_base = block_idx * 256;
        const int tid = threadIdx.x;
        #pragma unroll
        for (int r = 0; r < kRouteTile; ++r) {
            if (r < valid_rows) {
                const int route = route_start + r;
                const int8_t* xq_row = x_q + static_cast<int64_t>(route) * dim;
                const float* xs_row = x_scale + static_cast<int64_t>(route) * x_groups;
                if (tid < 64) {
                    const int byte_off = tid * 4;
                    int v = 0;
                    if (k_base + byte_off + 4 <= dim) {
                        v = *reinterpret_cast<const int*>(xq_row + k_base + byte_off);
                    }
                    x_shared_q[r][tid] = v;
                }
                if (tid < 8) {
                    const int sg_idx = block_idx * 8 + tid;
                    x_shared_scale[r][tid] = sg_idx < x_groups ? xs_row[sg_idx] : 0.0f;
                }
            }
        }
        __syncthreads();

        if (out_col < inter_dim) {
            const int sub = lane >> 2;
            const int part = lane & 3;
            const uint8_t* w1_block = w1_row + static_cast<int64_t>(block_idx) * 66;
            const uint8_t* w3_block = w3_row + static_cast<int64_t>(block_idx) * 66;
            const uint8_t* chunk1 = w1_block + 2 + sub * 8;
            const uint8_t* chunk3 = w3_block + 2 + sub * 8;
            const float d1 = gguf_block_scale_f16(w1_block);
            const float d3 = gguf_block_scale_f16(w3_block);
            const uint32_t aux1 = static_cast<uint32_t>(chunk1[4]) |
                (static_cast<uint32_t>(chunk1[5]) << 8) |
                (static_cast<uint32_t>(chunk1[6]) << 16) |
                (static_cast<uint32_t>(chunk1[7]) << 24);
            const uint32_t aux3 = static_cast<uint32_t>(chunk3[4]) |
                (static_cast<uint32_t>(chunk3[5]) << 8) |
                (static_cast<uint32_t>(chunk3[6]) << 16) |
                (static_cast<uint32_t>(chunk3[7]) << 24);
            const int grid_id1 = chunk1[part];
            const int grid_id3 = chunk3[part];
            const int sign_idx1 = static_cast<int>((aux1 >> (7 * part)) & 127);
            const int sign_idx3 = static_cast<int>((aux3 >> (7 * part)) & 127);
            const float s1 = 0.125f * d1 * static_cast<float>(2 * (aux1 >> 28) + 1);
            const float s3 = 0.125f * d3 * static_cast<float>(2 * (aux3 >> 28) + 1);
            const int8_t* vals1 = signed_grid + (grid_id1 * 128 + sign_idx1) * 8;
            const int8_t* vals3 = signed_grid + (grid_id3 * 128 + sign_idx3) * 8;
            const int v1_p0 = *reinterpret_cast<const int*>(vals1);
            const int v1_p1 = *reinterpret_cast<const int*>(vals1 + 4);
            const int v3_p0 = *reinterpret_cast<const int*>(vals3);
            const int v3_p1 = *reinterpret_cast<const int*>(vals3 + 4);
            const int xq_base = sub * 8 + part * 2;
            #pragma unroll
            for (int r = 0; r < kRouteTile; ++r) {
                float local1 = 0.0f;
                float local3 = 0.0f;
                if (r < valid_rows) {
                    const int x_p0 = x_shared_q[r][xq_base];
                    const int x_p1 = x_shared_q[r][xq_base + 1];
                    int sumi1 = __dp4a(v1_p0, x_p0, 0);
                    sumi1 = __dp4a(v1_p1, x_p1, sumi1);
                    int sumi3 = __dp4a(v3_p0, x_p0, 0);
                    sumi3 = __dp4a(v3_p1, x_p1, sumi3);
                    const float xs = x_shared_scale[r][sub];
                    local1 = s1 * xs * static_cast<float>(sumi1);
                    local3 = s3 * xs * static_cast<float>(sumi3);
                }
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1) {
                    local1 += __shfl_down_sync(0xffffffff, local1, offset);
                    local3 += __shfl_down_sync(0xffffffff, local3, offset);
                }
                if (lane == 0) {
                    acc1[r] += local1;
                    acc3[r] += local3;
                }
            }
        }
        __syncthreads();
    }

    if (lane == 0 && out_col < inter_dim) {
        #pragma unroll
        for (int r = 0; r < kRouteTile; ++r) {
            if (r < valid_rows) {
                const int route = route_start + r;
                gate[static_cast<int64_t>(route) * inter_dim + out_col] = acc1[r];
                up[static_cast<int64_t>(route) * inter_dim + out_col] = acc3[r];
            }
        }
    }
}

__global__ void q2_route_slots_from_segments_kernel(
    const int32_t* __restrict__ seg_starts,
    int64_t* __restrict__ route_slots,
    int routes,
    int n_experts,
    int max_count) {
    const int expert = blockIdx.x;
    const int ordinal = blockIdx.y * blockDim.x + threadIdx.x;
    if (expert >= n_experts || ordinal >= max_count) return;
    const int start = seg_starts[expert];
    const int end = seg_starts[expert + 1];
    const int route = start + ordinal;
    if (route < end && route < routes) route_slots[route] = expert;
}

__global__ void gguf_moe_grouped_w2_q2k_dp4a_kernel(
    const int8_t* __restrict__ hidden_q,
    const float* __restrict__ hidden_scale,
    const int64_t* __restrict__ route_tokens,
    const int32_t* __restrict__ seg_starts,
    const uint8_t* __restrict__ w2_blocks,
    float* __restrict__ y_rows,
    int routes,
    int n_experts,
    int max_count,
    int dim,
    int inter_dim,
    int w2_blocks_per_row,
    int hidden_groups) {
    const int expert = blockIdx.z;
    const int route_ordinal = blockIdx.y;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int out_col = blockIdx.x * kQ2TileN + warp;
    if (expert >= n_experts || route_ordinal >= max_count || warp >= kQ2TileN || out_col >= dim) return;
    const int route = seg_starts[expert] + route_ordinal;
    if (route >= seg_starts[expert + 1] || route >= routes) return;
    const int64_t token = route_tokens[route];
    if (token < 0) return;

    const int8_t* hidden_row = hidden_q + static_cast<int64_t>(route) * inter_dim;
    const float* hs_row = hidden_scale + static_cast<int64_t>(route) * hidden_groups;
    const uint8_t* w2_row = w2_blocks + (static_cast<int64_t>(expert) * dim + out_col) * w2_blocks_per_row * 84;
    float acc = 0.0f;
    for (int block_idx = 0; block_idx < w2_blocks_per_row; ++block_idx) {
        const uint8_t* block = w2_row + static_cast<int64_t>(block_idx) * 84;
        const uint8_t* scales = block;
        const uint8_t* qs = block + 16;
        const float d = gguf_block_scale_f16(block + 80);
        const float dmin = gguf_block_scale_f16(block + 82);
        const int k_base = block_idx * 256;
        for (int half = 0; half < 2; ++half) {
            const int sub = lane >> 2;      // eight 4-lane groups per half-block.
            const int part = lane & 3;
            const int group = half * 8 + sub;
            const int shift = (sub >> 1) * 2;
            const int byte_start = half * 32 + (sub & 1) * 16;
            const int idx = part * 4;
            const int q0 = static_cast<int>((qs[byte_start + idx + 0] >> shift) & 0x03);
            const int q1 = static_cast<int>((qs[byte_start + idx + 1] >> shift) & 0x03);
            const int q2 = static_cast<int>((qs[byte_start + idx + 2] >> shift) & 0x03);
            const int q3 = static_cast<int>((qs[byte_start + idx + 3] >> shift) & 0x03);
            const int w_pack = pack_i8x4(q0, q1, q2, q3);
            const int h_idx = k_base + group * 16 + idx;
            const int h_pack = pack_i8x4(
                static_cast<int>(hidden_row[h_idx + 0]),
                static_cast<int>(hidden_row[h_idx + 1]),
                static_cast<int>(hidden_row[h_idx + 2]),
                static_cast<int>(hidden_row[h_idx + 3]));
            const int dot_q = __dp4a(w_pack, h_pack, 0);
            const int sum_h = __dp4a(0x01010101, h_pack, 0);
            const float qscale = d * static_cast<float>(scales[group] & 0x0f);
            const float base = dmin * static_cast<float>(scales[group] >> 4);
            float local = hs_row[block_idx * 16 + group] * (qscale * static_cast<float>(dot_q) - base * static_cast<float>(sum_h));
            const unsigned group_mask = 0x0fu << (sub * 4);
            local += __shfl_down_sync(group_mask, local, 2, 4);
            local += __shfl_down_sync(group_mask, local, 1, 4);
            float group_sum = (part == 0) ? local : 0.0f;
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                group_sum += __shfl_down_sync(0xffffffff, group_sum, offset);
            }
            if (lane == 0) acc += group_sum;
        }
    }
    if (lane == 0) atomicAdd(y_rows + token * static_cast<int64_t>(dim) + out_col, acc);
}

__global__ void gguf_moe_grouped_w2_q2k_dp4a_route_tile_kernel(
    const int8_t* __restrict__ hidden_q,
    const float* __restrict__ hidden_scale,
    const int64_t* __restrict__ route_tokens,
    const int32_t* __restrict__ seg_starts,
    const uint8_t* __restrict__ w2_blocks,
    float* __restrict__ y_rows,
    int n_experts,
    int dim,
    int inter_dim,
    int w2_blocks_per_row,
    int hidden_groups) {
    constexpr int kRouteTile = 8;
    const int expert = blockIdx.y;
    const int route_tile = blockIdx.z;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int out_col = blockIdx.x * kQ2TileN + warp;
    if (expert >= n_experts || warp >= kQ2TileN || out_col >= dim) return;

    const int route_start = seg_starts[expert] + route_tile * kRouteTile;
    const int route_end = seg_starts[expert + 1];
    const int valid_rows = min(kRouteTile, route_end - route_start);
    if (valid_rows <= 0) return;

    const uint8_t* w2_row = w2_blocks + (static_cast<int64_t>(expert) * dim + out_col) * w2_blocks_per_row * 84;
    float acc[kRouteTile];
    #pragma unroll
    for (int r = 0; r < kRouteTile; ++r) acc[r] = 0.0f;

    for (int block_idx = 0; block_idx < w2_blocks_per_row; ++block_idx) {
        const uint8_t* block = w2_row + static_cast<int64_t>(block_idx) * 84;
        const uint8_t* scales = block;
        const uint8_t* qs = block + 16;
        const float d = gguf_block_scale_f16(block + 80);
        const float dmin = gguf_block_scale_f16(block + 82);
        const int k_base = block_idx * 256;
        for (int half = 0; half < 2; ++half) {
            const int sub = lane >> 2;
            const int part = lane & 3;
            const int group = half * 8 + sub;
            const int shift = (sub >> 1) * 2;
            const int byte_start = half * 32 + (sub & 1) * 16;
            const int idx = part * 4;
            const int q0 = static_cast<int>((qs[byte_start + idx + 0] >> shift) & 0x03);
            const int q1 = static_cast<int>((qs[byte_start + idx + 1] >> shift) & 0x03);
            const int q2 = static_cast<int>((qs[byte_start + idx + 2] >> shift) & 0x03);
            const int q3 = static_cast<int>((qs[byte_start + idx + 3] >> shift) & 0x03);
            const int w_pack = pack_i8x4(q0, q1, q2, q3);
            const int h_idx = k_base + group * 16 + idx;
            const float qscale = d * static_cast<float>(scales[group] & 0x0f);
            const float base = dmin * static_cast<float>(scales[group] >> 4);
            const unsigned group_mask = 0x0fu << (sub * 4);
            #pragma unroll
            for (int r = 0; r < kRouteTile; ++r) {
                float local = 0.0f;
                if (r < valid_rows) {
                    const int route = route_start + r;
                    const int64_t token = route_tokens[route];
                    if (token >= 0) {
                        const int8_t* hidden_row = hidden_q + static_cast<int64_t>(route) * inter_dim;
                        const float* hs_row = hidden_scale + static_cast<int64_t>(route) * hidden_groups;
                        const int h_pack = pack_i8x4(
                            static_cast<int>(hidden_row[h_idx + 0]),
                            static_cast<int>(hidden_row[h_idx + 1]),
                            static_cast<int>(hidden_row[h_idx + 2]),
                            static_cast<int>(hidden_row[h_idx + 3]));
                        const int dot_q = __dp4a(w_pack, h_pack, 0);
                        const int sum_h = __dp4a(0x01010101, h_pack, 0);
                        local = hs_row[block_idx * 16 + group] * (qscale * static_cast<float>(dot_q) - base * static_cast<float>(sum_h));
                    }
                }
                local += __shfl_down_sync(group_mask, local, 2, 4);
                local += __shfl_down_sync(group_mask, local, 1, 4);
                float group_sum = (part == 0) ? local : 0.0f;
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1) {
                    group_sum += __shfl_down_sync(0xffffffff, group_sum, offset);
                }
                if (lane == 0) acc[r] += group_sum;
            }
        }
    }

    if (lane == 0) {
        #pragma unroll
        for (int r = 0; r < kRouteTile; ++r) {
            if (r < valid_rows) {
                const int route = route_start + r;
                const int64_t token = route_tokens[route];
                if (token >= 0) atomicAdd(y_rows + token * static_cast<int64_t>(dim) + out_col, acc[r]);
            }
        }
    }
}

__global__ void gguf_moe_grouped_w2_q2k_dp4a_expert_tile_kernel(
    const int8_t* __restrict__ hidden_q,
    const float* __restrict__ hidden_scale,
    const int64_t* __restrict__ route_tokens,
    const int32_t* __restrict__ seg_starts,
    const uint8_t* __restrict__ w2_blocks,
    float* __restrict__ y_rows,
    int n_experts,
    int dim,
    int inter_dim,
    int w2_blocks_per_row,
    int hidden_groups) {
    constexpr int kRouteTile = 8;
    constexpr int kSubwarpCols = 8;
    const int expert = blockIdx.y;
    const int route_tile = blockIdx.z;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int sub = lane >> 2;
    const int part = lane & 3;
    const int out_col = blockIdx.x * (kQ2TileN * kSubwarpCols) + warp * kSubwarpCols + sub;
    if (expert >= n_experts || warp >= kQ2TileN || out_col >= dim) return;

    const int route_start = seg_starts[expert] + route_tile * kRouteTile;
    const int route_end = seg_starts[expert + 1];
    const int valid_rows = min(kRouteTile, route_end - route_start);
    if (valid_rows <= 0) return;

    const uint8_t* w2_row = w2_blocks + (static_cast<int64_t>(expert) * dim + out_col) * w2_blocks_per_row * 84;
    const unsigned mask = 0x0fu << (sub * 4);
    float acc[kRouteTile];
    #pragma unroll
    for (int r = 0; r < kRouteTile; ++r) acc[r] = 0.0f;

    for (int block_idx = 0; block_idx < w2_blocks_per_row; ++block_idx) {
        const uint8_t* block = w2_row + static_cast<int64_t>(block_idx) * 84;
        const uint8_t* scales = block;
        const uint8_t* qs = block + 16;
        const float d = gguf_block_scale_f16(block + 80);
        const float dmin = gguf_block_scale_f16(block + 82);
        const int k_base = block_idx * 256;
        for (int group = 0; group < 16; ++group) {
            const int half_block = group >> 3;
            const int group_in_half = group & 7;
            const int shift = (group_in_half >> 1) * 2;
            const int byte_start = half_block * 32 + (group_in_half & 1) * 16;
            const int idx = part * 4;
            const int q0 = static_cast<int>((qs[byte_start + idx + 0] >> shift) & 0x03);
            const int q1 = static_cast<int>((qs[byte_start + idx + 1] >> shift) & 0x03);
            const int q2 = static_cast<int>((qs[byte_start + idx + 2] >> shift) & 0x03);
            const int q3 = static_cast<int>((qs[byte_start + idx + 3] >> shift) & 0x03);
            const int w_pack = pack_i8x4(q0, q1, q2, q3);
            const float qscale = d * static_cast<float>(scales[group] & 0x0f);
            const float base = dmin * static_cast<float>(scales[group] >> 4);
            const int h_idx = k_base + group * 16 + idx;
            #pragma unroll
            for (int r = 0; r < kRouteTile; ++r) {
                float local = 0.0f;
                if (r < valid_rows) {
                    const int route = route_start + r;
                    const int64_t token = route_tokens[route];
                    if (token >= 0) {
                        const int8_t* hidden_row = hidden_q + static_cast<int64_t>(route) * inter_dim;
                        const float* hs_row = hidden_scale + static_cast<int64_t>(route) * hidden_groups;
                        const int h_pack = pack_i8x4(
                            static_cast<int>(hidden_row[h_idx + 0]),
                            static_cast<int>(hidden_row[h_idx + 1]),
                            static_cast<int>(hidden_row[h_idx + 2]),
                            static_cast<int>(hidden_row[h_idx + 3]));
                        const int dot_q = __dp4a(w_pack, h_pack, 0);
                        const int sum_h = __dp4a(0x01010101, h_pack, 0);
                        local = hs_row[block_idx * 16 + group] * (qscale * static_cast<float>(dot_q) - base * static_cast<float>(sum_h));
                    }
                }
                local += __shfl_down_sync(mask, local, 2, 4);
                local += __shfl_down_sync(mask, local, 1, 4);
                if (part == 0) acc[r] += local;
            }
        }
    }

    if (part == 0) {
        #pragma unroll
        for (int r = 0; r < kRouteTile; ++r) {
            if (r < valid_rows) {
                const int route = route_start + r;
                const int64_t token = route_tokens[route];
                if (token >= 0) atomicAdd(y_rows + token * static_cast<int64_t>(dim) + out_col, acc[r]);
            }
        }
    }
}

inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

}  // namespace

const int8_t* q2_signed_grid_device() {
    return signed_grid_device();
}

bool q2_quantize_x_q8_1_cuda(
    const float* d_x,
    int8_t* d_x_q,
    float* d_x_scale,
    int rows,
    int row_elems,
    void* stream) {
    if (rows <= 0 || row_elems <= 0) return true;
    const int groups = ceil_div(row_elems, 32);
    dim3 grid(rows, groups);
    gguf_q8_1_quantize_x_32_kernel<<<grid, 32, 0, static_cast<cudaStream_t>(stream)>>>(
        d_x, d_x_q, d_x_scale, rows, row_elems);
    return cudaGetLastError() == cudaSuccess;
}

bool q2_quantize_hidden_q8_1_cuda(
    const float* d_hidden,
    int8_t* d_hidden_q,
    float* d_hidden_scale,
    int routes,
    int inter_dim,
    void* stream) {
    if (routes <= 0 || inter_dim <= 0) return true;
    const int groups = ceil_div(inter_dim, 16);
    dim3 grid(routes, groups);
    gguf_q8_1_quantize_hidden_16_kernel<<<grid, 16, 0, static_cast<cudaStream_t>(stream)>>>(
        d_hidden, d_hidden_q, d_hidden_scale, routes, inter_dim);
    return cudaGetLastError() == cudaSuccess;
}

bool q2_route_swiglu_quantize_hidden_q8_1_cuda(
    const float* d_gate,
    const float* d_up,
    const float* d_route_weights,
    int8_t* d_hidden_q,
    float* d_hidden_scale,
    int routes,
    int inter_dim,
    float swiglu_limit,
    void* stream) {
    if (routes <= 0 || inter_dim <= 0) return true;
    const int groups = ceil_div(inter_dim, 16);
    dim3 grid(routes, groups);
    gguf_route_swiglu_quantize_hidden_16_kernel<<<grid, 16, 0, static_cast<cudaStream_t>(stream)>>>(
        d_gate, d_up, d_route_weights, d_hidden_q, d_hidden_scale, routes, inter_dim, swiglu_limit);
    return cudaGetLastError() == cudaSuccess;
}

bool q2_moe_single_w13_iq2_xxs_cuda(
    const int8_t* d_x_q,
    const float* d_x_scale,
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
    if (routes <= 0 || inter_dim <= 0 || dim <= 0) return true;
    const int blocks_per_row = dim / 256;
    const int x_groups = ceil_div(dim, 32);
    const int8_t* sg = signed_grid_device();
    if (sg == nullptr) return false;
    dim3 grid(ceil_div(inter_dim, kQ2TileN), routes);
    dim3 block(256);
    gguf_moe_single_w13_iq2_xxs_dp4a_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
        d_x_q, d_x_scale, d_route_slots, d_w1_blocks, d_w3_blocks, sg,
        d_gate, d_up,
        routes, n_experts, dim, inter_dim, blocks_per_row, x_groups);
    return cudaGetLastError() == cudaSuccess;
}

bool q2_moe_single_w2_q2k_cuda(
    const int8_t* d_hidden_q,
    const float* d_hidden_scale,
    const int64_t* d_route_slots,
    const uint8_t* d_w2_blocks,
    float* d_y,
    int routes,
    int n_experts,
    int dim,
    int inter_dim,
    void* stream) {
    if (routes <= 0 || inter_dim <= 0 || dim <= 0) return true;
    const int w2_blocks_per_row = inter_dim / 256;
    const int hidden_groups = ceil_div(inter_dim, 16);
    dim3 grid(ceil_div(dim, kQ2TileN), routes);
    dim3 block(256);
    gguf_moe_single_w2_q2k_dp4a_kernel<<<grid, block, 0, static_cast<cudaStream_t>(stream)>>>(
        d_hidden_q, d_hidden_scale, d_route_slots, d_w2_blocks, d_y,
        routes, n_experts, dim, inter_dim, w2_blocks_per_row, hidden_groups);
    return cudaGetLastError() == cudaSuccess;
}

bool q2_moe_grouped_w2_q2k_cuda(
    const int8_t* d_hidden_q,
    const float* d_hidden_scale,
    const int64_t* d_route_tokens,
    const int32_t* d_seg_starts,
    const uint8_t* d_w2_blocks,
    float* d_y_rows,
    int tokens,
    int routes,
    int n_experts,
    int max_count,
    int dim,
    int inter_dim,
    void* stream) {
    if (tokens <= 0 || routes < 0 || n_experts <= 0 || max_count < 0 || dim <= 0 || inter_dim <= 0) return false;
    cudaStream_t cs = static_cast<cudaStream_t>(stream);
    if (cudaMemsetAsync(d_y_rows, 0, static_cast<size_t>(tokens) * dim * sizeof(float), cs) != cudaSuccess) return false;
    if (routes == 0 || max_count == 0) return cudaGetLastError() == cudaSuccess;
    const int w2_blocks_per_row = inter_dim / 256;
    const int hidden_groups = ceil_div(inter_dim, 16);
    dim3 grid(ceil_div(dim, kQ2TileN), max_count, n_experts);
    dim3 block(256);
    gguf_moe_grouped_w2_q2k_dp4a_kernel<<<grid, block, 0, cs>>>(
        d_hidden_q, d_hidden_scale, d_route_tokens, d_seg_starts, d_w2_blocks,
        d_y_rows, routes, n_experts, max_count, dim, inter_dim, w2_blocks_per_row, hidden_groups);
    return cudaGetLastError() == cudaSuccess;
}

bool moe_prefill_q2_grouped_cuda_with_workspace(
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
    MoePrefillQ2GroupedWorkspace workspace,
    void* stream) {
    if (d_x_rows == nullptr || d_route_tokens == nullptr || d_route_weights == nullptr ||
        d_seg_starts == nullptr || d_w1_blocks == nullptr || d_w2_blocks == nullptr ||
        d_w3_blocks == nullptr || d_y_rows == nullptr) return false;
    if (tokens <= 0 || routes < 0 || n_experts <= 0 || max_count < 0 || dim <= 0 || inter_dim <= 0) return false;
    cudaStream_t cs = static_cast<cudaStream_t>(stream);
    if (cudaMemsetAsync(d_y_rows, 0, static_cast<size_t>(tokens) * dim * sizeof(float), cs) != cudaSuccess) return false;
    if (routes == 0 || max_count == 0) return cudaGetLastError() == cudaSuccess;
    if (workspace.routes_cap < routes || workspace.dim < dim || workspace.inter_dim < inter_dim ||
        workspace.d_x_route == nullptr || workspace.d_x_q == nullptr || workspace.d_x_scale == nullptr ||
        workspace.d_route_slots == nullptr || workspace.d_gate == nullptr || workspace.d_up == nullptr ||
        workspace.d_hidden_q == nullptr || workspace.d_hidden_scale == nullptr) return false;
    const int8_t* sg = signed_grid_device();
    if (sg == nullptr) return false;
    const int threads = 256;
    const int64_t routes_dim = static_cast<int64_t>(routes) * dim;
    gather_q2_route_rows_kernel<<<static_cast<int>((routes_dim + threads - 1) / threads), threads, 0, cs>>>(
        d_x_rows, d_route_tokens, workspace.d_x_route, routes, dim);
    if (cudaGetLastError() != cudaSuccess) return false;
    if (!q2_quantize_x_q8_1_cuda(workspace.d_x_route, workspace.d_x_q, workspace.d_x_scale, routes, dim, stream)) return false;
    dim3 slot_grid(n_experts, (max_count + threads - 1) / threads);
    q2_route_slots_from_segments_kernel<<<slot_grid, threads, 0, cs>>>(
        d_seg_starts, workspace.d_route_slots, routes, n_experts, max_count);
    if (cudaGetLastError() != cudaSuccess) return false;
    const int blocks_per_row = dim / 256;
    const int x_groups = ceil_div(dim, 32);
    dim3 block(256);
    if (env_enabled_q2("POCKETLLM_Q2_PREFILL_W13_EXPERT_TILE", true)) {
        dim3 grid_w13_tile(ceil_div(inter_dim, kQ2TileN), n_experts, ceil_div(max_count, 8));
        gguf_moe_grouped_w13_iq2_xxs_dp4a_expert_tile_kernel<<<grid_w13_tile, block, 0, cs>>>(
            workspace.d_x_q, workspace.d_x_scale, d_seg_starts, d_w1_blocks, d_w3_blocks, sg,
            workspace.d_gate, workspace.d_up, n_experts, dim, inter_dim, blocks_per_row, x_groups);
    } else {
        dim3 grid_w13(ceil_div(inter_dim, kQ2TileN), routes);
        gguf_moe_grouped_w13_iq2_xxs_dp4a_kernel<<<grid_w13, block, 0, cs>>>(
            workspace.d_x_q, workspace.d_x_scale, workspace.d_route_slots, d_w1_blocks, d_w3_blocks, sg,
            workspace.d_gate, workspace.d_up, routes, n_experts, dim, inter_dim, blocks_per_row, x_groups);
    }
    if (cudaGetLastError() != cudaSuccess) return false;
    if (!q2_route_swiglu_quantize_hidden_q8_1_cuda(workspace.d_gate, workspace.d_up, d_route_weights,
                                                   workspace.d_hidden_q, workspace.d_hidden_scale,
                                                   routes, inter_dim, swiglu_limit, stream)) {
        return false;
    }
    const int w2_blocks_per_row = inter_dim / 256;
    const int hidden_groups = ceil_div(inter_dim, 16);
    if (env_enabled_q2("POCKETLLM_Q2_PREFILL_W2_EXPERT_TILE", false)) {
        dim3 grid_w2_tile(ceil_div(dim, kQ2TileN * 8), n_experts, ceil_div(max_count, 8));
        gguf_moe_grouped_w2_q2k_dp4a_expert_tile_kernel<<<grid_w2_tile, block, 0, cs>>>(
            workspace.d_hidden_q, workspace.d_hidden_scale, d_route_tokens, d_seg_starts, d_w2_blocks,
            d_y_rows, n_experts, dim, inter_dim, w2_blocks_per_row, hidden_groups);
    } else if (env_enabled_q2("POCKETLLM_Q2_PREFILL_W2_ROUTE_TILE", true)) {
        dim3 grid_w2_route_tile(ceil_div(dim, kQ2TileN), n_experts, ceil_div(max_count, 8));
        gguf_moe_grouped_w2_q2k_dp4a_route_tile_kernel<<<grid_w2_route_tile, block, 0, cs>>>(
            workspace.d_hidden_q, workspace.d_hidden_scale, d_route_tokens, d_seg_starts, d_w2_blocks,
            d_y_rows, n_experts, dim, inter_dim, w2_blocks_per_row, hidden_groups);
    } else {
        dim3 grid_w2(ceil_div(dim, kQ2TileN), max_count, n_experts);
        gguf_moe_grouped_w2_q2k_dp4a_kernel<<<grid_w2, block, 0, cs>>>(
            workspace.d_hidden_q, workspace.d_hidden_scale, d_route_tokens, d_seg_starts, d_w2_blocks,
            d_y_rows, routes, n_experts, max_count, dim, inter_dim, w2_blocks_per_row, hidden_groups);
    }
    return cudaGetLastError() == cudaSuccess;
}

// --- Generic F16 row dequant ------------------------------------------------
// GGUF stores `token_embd.weight` and `output.weight` as F16. The layout is
// row-major [vocab, dim] (per-token row is contiguous), matching the FP4 path's
// BF16 embed/head layout. We provide the F16 counterpart of
// `bf16_row_to_float_cuda` so the GGUF dense path can lift one token's
// embedding into fp32 with a single 8 KiB H2D + a tiny kernel.

__global__ void f16_row_to_float_kernel(
    const uint16_t* matrix, float* y, int row, int cols) {
    const int out_row = blockIdx.x;
    const uint16_t* src = matrix + static_cast<size_t>(row + out_row) * cols;
    float* dst = y + static_cast<size_t>(out_row) * cols;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        dst[c] = __half2float(__ushort_as_half(src[c]));
    }
}

__global__ void f16_rows_to_float_kernel(
    const uint16_t* matrix,
    const int* rows,
    float* y,
    int count,
    int cols) {
    const int out_row = blockIdx.x;
    if (out_row >= count) return;
    const uint16_t* src = matrix + static_cast<size_t>(rows[out_row]) * cols;
    float* dst = y + static_cast<size_t>(out_row) * cols;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        dst[c] = __half2float(__ushort_as_half(src[c]));
    }
}

bool f16_row_to_float_cuda(
    const uint16_t* d_matrix_f16,
    float* d_y,
    int row,
    int cols,
    void* stream) {
    if (cols <= 0) return true;
    cudaStream_t cs = static_cast<cudaStream_t>(stream);
    f16_row_to_float_kernel<<<1, 256, 0, cs>>>(d_matrix_f16, d_y, row, cols);
    return cudaGetLastError() == cudaSuccess;
}

bool f16_rows_to_float_cuda(
    const uint16_t* d_matrix_f16,
    const int* d_rows,
    float* d_y,
    int rows,
    int cols,
    void* stream) {
    if (rows <= 0 || cols <= 0) return true;
    if (d_matrix_f16 == nullptr || d_rows == nullptr || d_y == nullptr) return false;
    cudaStream_t cs = static_cast<cudaStream_t>(stream);
    f16_rows_to_float_kernel<<<rows, 256, 0, cs>>>(d_matrix_f16, d_rows, d_y, rows, cols);
    return cudaGetLastError() == cudaSuccess;
}

bool f16_contiguous_rows_to_float_cuda(
    const uint16_t* d_matrix_f16,
    float* d_y,
    int rows,
    int cols,
    void* stream) {
    if (rows <= 0 || cols <= 0) return true;
    if (d_matrix_f16 == nullptr || d_y == nullptr) return false;
    cudaStream_t cs = static_cast<cudaStream_t>(stream);
    f16_row_to_float_kernel<<<rows, 256, 0, cs>>>(d_matrix_f16, d_y, 0, cols);
    return cudaGetLastError() == cudaSuccess;
}

}  // namespace pocket
