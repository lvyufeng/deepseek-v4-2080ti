#include <torch/extension.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/util/BFloat16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace {

constexpr int kDim = 256;
constexpr int kWarpSize = 32;
constexpr int kMaxHeadsPerKv = 12;
constexpr int kMmaRows = 16;
constexpr int kMmaKvTile = 32;
constexpr int kMmaSlices = 4;
constexpr int kMmaSteps = kDim / 8;
constexpr int kMmaWarps = kMmaSlices;
constexpr int kMmaThreads = kMmaWarps * kWarpSize;
constexpr int kMmaKsStride = kDim + 8;
constexpr int kMmaVtStride = kMmaKvTile + 2;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
#define QWEN4_EXP_QSA_MMA_AVAILABLE 1
#endif

union HalfBits {
    __half value;
    uint16_t bits;
};

__device__ __forceinline__ uint16_t float_to_half_bits(float value) {
    HalfBits converted;
    converted.value = __float2half_rn(value);
    return converted.bits;
}

__device__ __forceinline__ float half_bits_to_float(uint16_t bits) {
    HalfBits converted;
    converted.bits = bits;
    return __half2float(converted.value);
}

__device__ __forceinline__ uint32_t pack_half2(float first, float second) {
    return static_cast<uint32_t>(float_to_half_bits(first)) |
        (static_cast<uint32_t>(float_to_half_bits(second)) << 16);
}

__device__ __forceinline__ void mma_m16n8k8_f16_f32(
    float (&d)[4], const uint32_t (&a)[2], uint32_t b) {
#ifdef QWEN4_EXP_QSA_MMA_AVAILABLE
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 "
        "{%0, %1, %2, %3}, {%4, %5}, {%6}, {%0, %1, %2, %3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(b));
#else
    (void)d;
    (void)a;
    (void)b;
#endif
}

template <int HeadsPerKv>
__global__ __launch_bounds__(kMmaThreads) void qsa_mma_bf16_kernel(
    const c10::BFloat16* __restrict__ query,
    const c10::BFloat16* __restrict__ key,
    const c10::BFloat16* __restrict__ value,
    const int32_t* __restrict__ selected,
    c10::BFloat16* __restrict__ output,
    int batch,
    int query_len,
    int query_heads,
    int kv_heads,
    int kv_len,
    int selected_len,
    float softmax_scale) {
    __shared__ __align__(16) uint16_t ks[kMmaKvTile][kMmaKsStride];
    __shared__ __align__(16) uint16_t vt[kDim][kMmaVtStride];
    __shared__ float scores[kMmaRows][kMmaKvTile];
    __shared__ __align__(16) uint16_t probabilities[kMmaRows][kMmaVtStride];
    __shared__ float row_max[kMmaRows];
    __shared__ float row_sum[kMmaRows];
    __shared__ float row_alpha[kMmaRows];
    __shared__ int32_t token_ids[kMmaKvTile];

    const int query_row = static_cast<int>(blockIdx.x);
    const int kv_head = static_cast<int>(blockIdx.y);
    const int batch_index = query_row / query_len;
    const int query_index = query_row - batch_index * query_len;
    if (batch_index >= batch || kv_head >= kv_heads) return;

    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int slice = warp;
    const int lane = tid & (kWarpSize - 1);
    const int gid = lane >> 2;
    const int tig = lane & 3;
    const int row0 = gid;
    const int row1 = gid + 8;
    const int first_head = kv_head * HeadsPerKv;
    const int active_heads = min(HeadsPerKv, query_heads - first_head);
    const int64_t query_row_base =
        (static_cast<int64_t>(batch_index) * query_len + query_index) *
        query_heads * kDim;
    const int64_t kv_batch_stride =
        static_cast<int64_t>(kv_heads) * kv_len * kDim;
    const int64_t kv_head_base =
        static_cast<int64_t>(batch_index) * kv_batch_stride +
        static_cast<int64_t>(kv_head) * kv_len * kDim;
    const int32_t* selected_row =
        selected + (static_cast<int64_t>(batch_index) * query_len + query_index) *
                       selected_len;

    uint32_t query_fragment[kMmaSteps][2];
#pragma unroll
    for (int step = 0; step < kMmaSteps; ++step) {
        const int dim = step * 8 + tig * 2;
        float first0 = 0.0f;
        float second0 = 0.0f;
        float first1 = 0.0f;
        float second1 = 0.0f;
        if (row0 < active_heads) {
            const int64_t at = query_row_base +
                static_cast<int64_t>(first_head + row0) * kDim + dim;
            first0 = static_cast<float>(query[at]);
            second0 = static_cast<float>(query[at + 1]);
        }
        if (row1 < active_heads) {
            const int64_t at = query_row_base +
                static_cast<int64_t>(first_head + row1) * kDim + dim;
            first1 = static_cast<float>(query[at]);
            second1 = static_cast<float>(query[at + 1]);
        }
        query_fragment[step][0] = pack_half2(first0, second0);
        query_fragment[step][1] = pack_half2(first1, second1);
    }

    float accumulator[8][4] = {};
    if (tid < kMmaRows) {
        row_max[tid] = -INFINITY;
        row_sum[tid] = 0.0f;
    }
    __syncthreads();

    for (int selected_base = 0; selected_base < selected_len;
         selected_base += kMmaKvTile) {
        const int count = min(kMmaKvTile, selected_len - selected_base);
        if (tid < kMmaKvTile) {
            token_ids[tid] = tid < count
                ? selected_row[selected_base + tid]
                : -1;
        }
        __syncthreads();

        for (int idx = tid; idx < kMmaKvTile * kMmaSteps;
             idx += kMmaThreads) {
            const int slot = idx / kMmaSteps;
            const int chunk = idx - slot * kMmaSteps;
            const int token = token_ids[slot];
            uint16_t key_half[8] = {};
            uint16_t value_half[8] = {};
            if (token >= 0 && token < kv_len) {
                const int64_t at = kv_head_base +
                    static_cast<int64_t>(token) * kDim + chunk * 8;
#pragma unroll
                for (int item = 0; item < 8; ++item) {
                    key_half[item] =
                        float_to_half_bits(static_cast<float>(key[at + item]));
                    value_half[item] =
                        float_to_half_bits(static_cast<float>(value[at + item]));
                }
            }
            *reinterpret_cast<uint4*>(&ks[slot][chunk * 8]) =
                *reinterpret_cast<const uint4*>(key_half);
#pragma unroll
            for (int item = 0; item < 8; ++item) {
                vt[chunk * 8 + item][slot] = value_half[item];
            }
        }
        __syncthreads();

        float score_fragment[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int step = 0; step < kMmaSteps; ++step) {
            const uint32_t b = *reinterpret_cast<const uint32_t*>(
                &ks[slice * 8 + gid][step * 8 + tig * 2]);
            mma_m16n8k8_f16_f32(score_fragment, query_fragment[step], b);
        }
        const int score_col = slice * 8 + tig * 2;
        scores[row0][score_col] = score_fragment[0];
        scores[row0][score_col + 1] = score_fragment[1];
        scores[row1][score_col] = score_fragment[2];
        scores[row1][score_col + 1] = score_fragment[3];
        __syncthreads();

        if (tid < kMmaRows) {
            const int head_slot = tid;
            const float old_max = row_max[head_slot];
            const float old_sum = row_sum[head_slot];
            float tile_max = -INFINITY;
            float scaled_scores[kMmaKvTile];
#pragma unroll
            for (int col = 0; col < kMmaKvTile; ++col) {
                const float score = head_slot < active_heads &&
                        token_ids[col] >= 0 && token_ids[col] < kv_len
                    ? scores[head_slot][col] * softmax_scale
                    : -INFINITY;
                scaled_scores[col] = score;
                tile_max = fmaxf(tile_max, score);
            }
            const float next_max = fmaxf(old_max, tile_max);
            const float alpha = old_max == -INFINITY
                ? 0.0f
                : expf(old_max - next_max);
            float tile_sum = 0.0f;
#pragma unroll
            for (int col = 0; col < kMmaKvTile; ++col) {
                uint16_t bits = 0;
                if (scaled_scores[col] != -INFINITY) {
                    bits = float_to_half_bits(
                        expf(scaled_scores[col] - next_max));
                    tile_sum += half_bits_to_float(bits);
                }
                probabilities[head_slot][col] = bits;
            }
            row_alpha[head_slot] = alpha;
            row_max[head_slot] = next_max;
            row_sum[head_slot] = old_sum * alpha + tile_sum;
        }
        __syncthreads();

        const float alpha0 = row_alpha[row0];
        const float alpha1 = row_alpha[row1];
#pragma unroll
        for (int item = 0; item < 8; ++item) {
            accumulator[item][0] *= alpha0;
            accumulator[item][1] *= alpha0;
            accumulator[item][2] *= alpha1;
            accumulator[item][3] *= alpha1;
        }
#pragma unroll
        for (int kt = 0; kt < kMmaKvTile / 8; ++kt) {
            uint32_t a[2];
            a[0] = *reinterpret_cast<const uint32_t*>(
                &probabilities[row0][kt * 8 + tig * 2]);
            a[1] = *reinterpret_cast<const uint32_t*>(
                &probabilities[row1][kt * 8 + tig * 2]);
#pragma unroll
            for (int item = 0; item < 8; ++item) {
                const uint32_t b = *reinterpret_cast<const uint32_t*>(
                    &vt[slice * 64 + item * 8 + gid][kt * 8 + tig * 2]);
                mma_m16n8k8_f16_f32(accumulator[item], a, b);
            }
        }
        __syncthreads();
    }

    const float inverse0 = row0 < active_heads && row_sum[row0] > 0.0f
        ? 1.0f / row_sum[row0]
        : 0.0f;
    const float inverse1 = row1 < active_heads && row_sum[row1] > 0.0f
        ? 1.0f / row_sum[row1]
        : 0.0f;
#pragma unroll
    for (int item = 0; item < 8; ++item) {
        const int dim = slice * 64 + item * 8 + tig * 2;
        if (row0 < active_heads) {
            const int64_t at = query_row_base +
                static_cast<int64_t>(first_head + row0) * kDim + dim;
            output[at] = c10::BFloat16(accumulator[item][0] * inverse0);
            output[at + 1] = c10::BFloat16(accumulator[item][1] * inverse0);
        }
        if (row1 < active_heads) {
            const int64_t at = query_row_base +
                static_cast<int64_t>(first_head + row1) * kDim + dim;
            output[at] = c10::BFloat16(accumulator[item][2] * inverse1);
            output[at + 1] = c10::BFloat16(accumulator[item][3] * inverse1);
        }
    }
}

template <int HeadsPerKv>
__global__ void qsa_scalar_bf16_kernel(
    const c10::BFloat16* __restrict__ query,
    const c10::BFloat16* __restrict__ key,
    const c10::BFloat16* __restrict__ value,
    const int32_t* __restrict__ selected,
    c10::BFloat16* __restrict__ output,
    int batch,
    int query_len,
    int query_heads,
    int kv_heads,
    int kv_len,
    int selected_len,
    float softmax_scale) {
    const int query_row = static_cast<int>(blockIdx.x);
    const int kv_head = static_cast<int>(blockIdx.y);
    const int batch_index = query_row / query_len;
    const int query_index = query_row - batch_index * query_len;
    if (batch_index >= batch || kv_head >= kv_heads) return;

    const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
    const int head_slot = static_cast<int>(threadIdx.x) / kWarpSize;
    const int query_head = kv_head * HeadsPerKv + head_slot;
    if (head_slot >= HeadsPerKv || query_head >= query_heads) return;

    constexpr unsigned kFullMask = 0xffffffffU;
    const int64_t query_base =
        ((static_cast<int64_t>(batch_index) * query_len + query_index) *
             query_heads +
         query_head) *
        kDim;

    float query_fragment[8];
    float output_fragment[8];
#pragma unroll
    for (int item = 0; item < 8; ++item) {
        const int dim = lane + item * kWarpSize;
        query_fragment[item] = static_cast<float>(query[query_base + dim]);
        output_fragment[item] = 0.0f;
    }

    const int32_t* selected_row =
        selected + (static_cast<int64_t>(batch_index) * query_len + query_index) *
                       selected_len;
    const int64_t kv_batch_stride =
        static_cast<int64_t>(kv_heads) * kv_len * kDim;
    const int64_t kv_head_base =
        static_cast<int64_t>(batch_index) * kv_batch_stride +
        static_cast<int64_t>(kv_head) * kv_len * kDim;

    float running_max = -INFINITY;
    float running_sum = 0.0f;
    for (int selected_index = 0; selected_index < selected_len;
         ++selected_index) {
        int token = lane == 0 ? selected_row[selected_index] : 0;
        token = __shfl_sync(kFullMask, token, 0);
        if (token < 0 || token >= kv_len) continue;

        const int64_t token_base =
            kv_head_base + static_cast<int64_t>(token) * kDim;
        float sum = 0.0f;
#pragma unroll
        for (int item = 0; item < 8; ++item) {
            const int dim = lane + item * kWarpSize;
            sum = fmaf(
                query_fragment[item],
                static_cast<float>(key[token_base + dim]),
                sum);
        }
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum += __shfl_down_sync(kFullMask, sum, offset);
        }

        float alpha = 0.0f;
        float weight = 0.0f;
        if (lane == 0) {
            const float score = sum * softmax_scale;
            const float next_max = fmaxf(running_max, score);
            alpha = running_max == -INFINITY
                ? 0.0f
                : expf(running_max - next_max);
            weight = expf(score - next_max);
            running_max = next_max;
            running_sum = running_sum * alpha + weight;
        }
        alpha = __shfl_sync(kFullMask, alpha, 0);
        weight = __shfl_sync(kFullMask, weight, 0);

#pragma unroll
        for (int item = 0; item < 8; ++item) {
            const int dim = lane + item * kWarpSize;
            const float value_item = static_cast<float>(value[token_base + dim]);
            output_fragment[item] =
                output_fragment[item] * alpha + value_item * weight;
        }
    }

    float inverse = lane == 0 && running_sum > 0.0f
        ? 1.0f / running_sum
        : 0.0f;
    inverse = __shfl_sync(kFullMask, inverse, 0);
#pragma unroll
    for (int item = 0; item < 8; ++item) {
        const int dim = lane + item * kWarpSize;
        output[query_base + dim] =
            c10::BFloat16(output_fragment[item] * inverse);
    }
}

template <int HeadsPerKv>
void launch_qsa(
    const c10::BFloat16* query,
    const c10::BFloat16* key,
    const c10::BFloat16* value,
    const int32_t* selected,
    c10::BFloat16* output,
    int batch,
    int query_len,
    int query_heads,
    int kv_heads,
    int kv_len,
    int selected_len,
    float softmax_scale,
    bool use_mma,
    cudaStream_t stream) {
    static_assert(HeadsPerKv >= 1 && HeadsPerKv <= kMaxHeadsPerKv);
    const dim3 grid(batch * query_len, kv_heads);
    if (use_mma) {
        qsa_mma_bf16_kernel<HeadsPerKv>
            <<<grid, kMmaThreads, 0, stream>>>(
                query,
                key,
                value,
                selected,
                output,
                batch,
                query_len,
                query_heads,
                kv_heads,
                kv_len,
                selected_len,
                softmax_scale);
        return;
    }
    qsa_scalar_bf16_kernel<HeadsPerKv>
        <<<grid, HeadsPerKv * kWarpSize, 0, stream>>>(
            query,
            key,
            value,
            selected,
            output,
            batch,
            query_len,
            query_heads,
            kv_heads,
            kv_len,
            selected_len,
            softmax_scale);
}

}  // namespace

torch::Tensor qwen4_exp_qsa_bf16_forward_cuda(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    const torch::Tensor& selected,
    double softmax_scale) {
    c10::cuda::CUDAGuard device_guard(query.device());
    const int batch = static_cast<int>(query.size(0));
    const int query_len = static_cast<int>(query.size(1));
    const int query_heads = static_cast<int>(query.size(2));
    const int kv_heads = static_cast<int>(key.size(1));
    const int kv_len = static_cast<int>(key.size(2));
    const int selected_len = static_cast<int>(selected.size(2));
    const int heads_per_kv = query_heads / kv_heads;
    const char* mma_env = std::getenv("POCKETLLM_QWEN4_QSA_MMA");
    const bool use_mma = selected_len >= kMmaKvTile && mma_env != nullptr &&
        std::strcmp(mma_env, "0") != 0;
    auto output = torch::empty(query.sizes(), query.options());
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

#define QWEN4_EXP_LAUNCH_QSA(HEADS)                                       \
    launch_qsa<HEADS>(                                                     \
        query.data_ptr<c10::BFloat16>(),                                   \
        key.data_ptr<c10::BFloat16>(),                                     \
        value.data_ptr<c10::BFloat16>(),                                   \
        selected.data_ptr<int32_t>(),                                      \
        output.data_ptr<c10::BFloat16>(),                                  \
        batch,                                                             \
        query_len,                                                         \
        query_heads,                                                       \
        kv_heads,                                                          \
        kv_len,                                                            \
        selected_len,                                                      \
        static_cast<float>(softmax_scale),                                 \
        use_mma,                                                           \
        stream)
    switch (heads_per_kv) {
        case 1: QWEN4_EXP_LAUNCH_QSA(1); break;
        case 2: QWEN4_EXP_LAUNCH_QSA(2); break;
        case 3: QWEN4_EXP_LAUNCH_QSA(3); break;
        case 4: QWEN4_EXP_LAUNCH_QSA(4); break;
        case 5: QWEN4_EXP_LAUNCH_QSA(5); break;
        case 6: QWEN4_EXP_LAUNCH_QSA(6); break;
        case 7: QWEN4_EXP_LAUNCH_QSA(7); break;
        case 8: QWEN4_EXP_LAUNCH_QSA(8); break;
        case 9: QWEN4_EXP_LAUNCH_QSA(9); break;
        case 10: QWEN4_EXP_LAUNCH_QSA(10); break;
        case 11: QWEN4_EXP_LAUNCH_QSA(11); break;
        case 12: QWEN4_EXP_LAUNCH_QSA(12); break;
    }
#undef QWEN4_EXP_LAUNCH_QSA
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}
