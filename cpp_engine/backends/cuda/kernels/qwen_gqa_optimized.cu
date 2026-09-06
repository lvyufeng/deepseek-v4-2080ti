#include "qwen_cuda_ops.hpp"

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace pocket {
namespace {

constexpr int kThreads = 128;
constexpr int kWarps = kThreads / 32;
constexpr int kHeadsPerGroup = 3;
constexpr int kQueryRows = 2;
constexpr int kMaxHeadDim = 256;
constexpr int kValuesPerThread = kMaxHeadDim / kThreads;
constexpr int kPrefillCombos = kHeadsPerGroup * kQueryRows;
// Hard ceiling for the tunable decode split count. Sizes the merge kernel's
// shared weight array at 4 bytes per split, so 1024 costs 4 KiB of the 48 KiB
// budget. The default target sits well under it; the headroom only matters for
// A/B sweeps that ask for a finer split than the default.
constexpr int kDecodeSplitCeiling = 1024;
// Target decode split count. The split kernel walks its slice serially and the
// launch is only kv_heads*groups*splits blocks, so at the real TP4 shape (6 Q
// heads, 1 KV head, 2 groups) it is the split count, not bytes, that sets the
// runtime: the old 2048-position target measured a flat 1.41 ms from 4K to 65K
// context, about 47 GB/s of the 616 GB/s the card can sustain.
//
// What matters is the number of blocks, not the slice length. The previous
// default fixed 256 positions per split, which leaves 8K context with 32 splits
// and 64 blocks -- under one block per SM on the 68-SM card, so most of the
// device sits idle. Targeting a fixed split count instead scales the slice with
// the context and keeps the grid at 512 blocks everywhere. Fused kernel
// milliseconds from `bench_qwen_gqa_decode` at the real TP4 shape, and decode
// tok/s from the serial full-network TP4 benchmark at TG128:
//
//   context   fixed 256 pos      target 256 splits
//              kernel  tok/s      kernel  tok/s
//      8192    0.2382  37.25      0.0920  39.49   (medians of 3 repetitions)
//     16384    0.1923  36.69      0.1578  38.68
//     32768    0.2247  35.37      0.2080  36.11
//     65536    0.3873  32.05      0.3881  31.59   (same launch either way)
//
// 65536 already reached 256 splits under the old default, so its geometry is
// unchanged and the difference there is run-to-run noise. Generated tokens stay
// bit-identical to the pre-change sequential baseline for all 128 steps at every
// context above.
//
// The target is now expressed as CTAs per SM rather than a round number, because
// the optimum tracks occupancy exactly. Once the group covers the whole KV head
// (see the split kernel) the CTA needs 80 registers, which allows 6 resident per
// SM, so 68 * 6 = 408 splits is the largest grid that still fits one wave. Fused
// kernel milliseconds, medians of 3 after discarding the warm-up repetition:
//
//   context   256    272 (4/SM)  340 (5/SM)  408 (6/SM)
//      4096  0.0435    0.0436      0.0460      0.0491
//      8192  0.0661    0.0665      0.0654      0.0691
//     16384  0.1095    0.1075      0.1038      0.1035
//     32768  0.1954    0.1888      0.1766      0.1767
//     65536  0.3683    0.3487      0.3239      0.3193
//
// Long contexts want the full wave; short ones do not, because every CTA writes a
// fixed (head_dim + 2) * kHPG partial and that scratch traffic stops being
// amortised once the slice gets short. Splitting the default at 16384 takes the
// better column on both sides. Going past one wave is strictly worse at every
// context (476 splits: 0.1321 / 0.2271 / 0.4051), so the ceiling is a real edge.
constexpr int kDecodeSplitsPerSm = 6;
constexpr int kDecodeShortSplitsPerSm = 4;
constexpr int kDecodeShortContext = 16384;
// Verify attention splits far more finely than decode: it has only a handful of
// query rows to fill the device with, so a 32-position tile at 8192 context wants
// 256 splits where the decode ceiling would force 128. Sizes the merge kernel's
// shared weight array, so 256 costs 1 KiB of shared memory per merge block.
constexpr int kVerifyMaxSplits = 256;
// Below this the reference score/value kernels win; the fused launch refuses
// short contexts so the engine gate has to agree with it.
constexpr int kDecodeMinContext = 4096;
constexpr int kVerifyMaxRows = 8;

// Target split count, the split ceiling, and a positions-per-split override,
// all available for A/B sweeps. Read once: these are launch geometry, not
// per-call state.
int decode_sm_count() {
    static const int value = [] {
        int device = 0;
        if (cudaGetDevice(&device) != cudaSuccess) return 68;
        cudaDeviceProp properties{};
        if (cudaGetDeviceProperties(&properties, device) != cudaSuccess) {
            return 68;  // SM75 2080 Ti, the shape every measurement above used.
        }
        return properties.multiProcessorCount;
    }();
    return value;
}

// The tensor-core split kernel reassociates both dot products, so it is not
// bit-identical to the scalar kernel. Its gain used to be small at short context,
// where near ties are more visible, so the default kept those on the exact path.
// Raising the CTA occupancy to three per SM removed that trade: the candidate now
// wins from the minimum context the fused path accepts. Full-network TP2 decode
// at 8192 and TG128, serial fresh processes, three repetitions each:
//
//   scalar   29.5123  29.3403  29.2454
//   candidate 30.6626  30.4263  30.3284
//
// `=1` forces it at every context for A/B, while `=0` forces scalar.
constexpr int kDecodeMmaDefaultMinContext = 0;
bool decode_mma_enabled(int attended_positions) {
    static const int mode = [] {
        const char* env = std::getenv("POCKETLLM_QWEN_DECODE_MMA");
        if (env == nullptr) return -1;  // automatic context-based selection
        return std::strcmp(env, "0") == 0 ? 0 : 1;
    }();
    return mode < 0 ? attended_positions >= kDecodeMmaDefaultMinContext
                    : mode != 0;
}

int decode_mma_target_splits() {
    static const int value = [] {
        const char* env = std::getenv("POCKETLLM_QWEN_DECODE_MMA_TARGET_SPLITS");
        const int parsed = env != nullptr ? std::atoi(env) : 0;
        return parsed > 0 ? parsed : 0;
    }();
    return value;
}

bool decode_mma_device_supported() {
    static const bool value = [] {
        int device = 0;
        cudaDeviceProp properties{};
        return cudaGetDevice(&device) == cudaSuccess &&
            cudaGetDeviceProperties(&properties, device) == cudaSuccess &&
            (properties.major > 7 ||
             (properties.major == 7 && properties.minor >= 5));
    }();
    return value;
}

// `mma` is passed in rather than read from the environment so a caller can ask
// for one variant's geometry while the process default selects the other.
int decode_target_splits(int attended_positions, bool mma) {
    static const int override_value = [] {
        const char* env = std::getenv("POCKETLLM_QWEN_DECODE_TARGET_SPLITS");
        const int parsed = env != nullptr ? std::atoi(env) : 0;
        return parsed > 0 ? parsed : 0;
    }();
    if (mma) {
        const int mma_override = decode_mma_target_splits();
        if (mma_override > 0) return mma_override;
        // The tensor-core CTA is 128 threads with a 26 KiB shared footprint, so
        // only two fit per SM. As with the scalar kernel the optimum lands on a
        // whole wave, but one wave lower: the candidate does six heads' worth of
        // work per CTA, so it needs fewer CTAs to saturate. Fused milliseconds
        // from `bench_qwen_gqa_decode` at the real TP4 shape, medians of the
        // second and third repetitions:
        //
        //   context   scalar   68 (1/SM)  102   136 (2/SM)  204   272   408
        //      4096   0.0428     0.0204     -      0.0234  0.0333 0.0411 0.0598
        //      8192   0.0655     0.0362  0.0376    0.0377  0.0515 0.0490 0.0690
        //     16384   0.1020     0.0610  0.0611    0.0584  0.0729 0.0722 0.1006
        //     32768   0.1742     0.1107  0.1024    0.0986  0.1194 0.1142 0.1362
        //     65536   0.3149     0.2052  0.1832    0.1784  0.1999 0.1939 0.2239
        //
        // Cutting the shared footprint to 18632 bytes raised the CTA occupancy
        // from two to three per SM, which moved the optimum. Re-measured at the
        // TP4 shape, fused milliseconds, medians of the second and third
        // repetitions:
        //
        //   context   scalar   68 (1 wave)  136 (2)   204 (3)   272 (4)
        //      4096   0.2728      0.0598     0.0413    0.0413    0.0412
        //      8192   0.8359      0.0889     0.0633    0.0633    0.0633
        //     16384   1.6727      0.0550     0.0546    0.0629    0.0699
        //     32768   3.3562      0.0973     0.0904    0.1009    0.1088
        //     65536   6.7720      0.1788     0.1589    0.1648    0.1822
        //
        // Two waves now win at every context, so the short-context exception
        // that the one-CTA geometry needed is gone. Past two waves the per-CTA
        // partial write stops being amortised and the long contexts regress.
        // The TP2 shape agrees: with 68 splits per KV group, 65536 costs
        // 0.2844 ms against 0.3204 ms for twice as many.
        if (override_value <= 0) {
            return decode_sm_count() * 2;
        }
    }
    if (override_value > 0) return override_value;
    const int per_sm = attended_positions < kDecodeShortContext
        ? kDecodeShortSplitsPerSm : kDecodeSplitsPerSm;
    return decode_sm_count() * per_sm;
}

// Setting this pins the slice length and ignores the split target, which is how
// the fixed-positions geometry that shipped before can still be measured.
int decode_forced_positions() {
    static const int value = [] {
        const char* env = std::getenv("POCKETLLM_QWEN_DECODE_SPLIT_POSITIONS");
        const int parsed = env != nullptr ? std::atoi(env) : 0;
        return parsed > 0 ? parsed : 0;
    }();
    return value;
}

int decode_max_splits() {
    static const int value = [] {
        const char* env = std::getenv("POCKETLLM_QWEN_DECODE_MAX_SPLITS");
        const int parsed = env != nullptr ? std::atoi(env) : 0;
        return parsed > 0 ? std::min(parsed, kDecodeSplitCeiling)
                          : kDecodeSplitCeiling;
    }();
    return value;
}

// A window of 0 keeps exact full attention. Otherwise every query attends to
// the leading sink prefix plus the most recent window positions, expressed as
// two logical ranges over the unmodified KV cache.
struct SparseRanges {
    int sink_count;
    int window_start;
    int window_count;

    __host__ __device__ int total() const { return sink_count + window_count; }

    __host__ __device__ int position(int logical) const {
        return logical < sink_count
            ? logical
            : window_start + (logical - sink_count);
    }
};

__host__ __device__ SparseRanges sparse_ranges(
    int context_len, int window, int sink) {
    SparseRanges ranges;
    if (window <= 0) {
        ranges.sink_count = 0;
        ranges.window_start = 0;
        ranges.window_count = context_len;
        return ranges;
    }
    ranges.sink_count = sink < context_len ? sink : context_len;
    const int tail = context_len - window;
    ranges.window_start = tail > ranges.sink_count ? tail : ranges.sink_count;
    ranges.window_count = context_len - ranges.window_start;
    return ranges;
}

__device__ __forceinline__ float half_to_float(uint16_t bits) {
    return __half2float(__ushort_as_half(bits));
}

__device__ __forceinline__ uint16_t float_to_half(float value) {
    return __half_as_ushort(__float2half_rn(value));
}

// Butterfly reduction: every lane ends with the full warp total, so a warp that
// owns one dot product needs no lane-0 broadcast afterwards.
__device__ __forceinline__ float warp_sum_all(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_xor_sync(0xffffffffu, value, offset);
    }
    return value;
}

__device__ __forceinline__ float warp_sum_max(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, offset));
    }
    return value;
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffu, value, offset);
    }
    return value;
}

template <int kCombos>
__device__ __forceinline__ void reduce_dot_products(
    const float (&partial)[kCombos], float (&warp_sums)[kCombos][kWarps],
    float (&scores)[kCombos], float scale) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int warp = static_cast<int>(threadIdx.x) >> 5;
#pragma unroll
    for (int combo = 0; combo < kCombos; ++combo) {
        const float sum = warp_sum(partial[combo]);
        if (lane == 0) warp_sums[combo][warp] = sum;
    }
    __syncthreads();
    if (static_cast<int>(threadIdx.x) < kCombos) {
        float sum = 0.0f;
#pragma unroll
        for (int w = 0; w < kWarps; ++w) {
            sum += warp_sums[threadIdx.x][w];
        }
        scores[threadIdx.x] = sum * scale;
    }
    __syncthreads();
}

__host__ __device__ bool attends(
    int position, int context_limit, int window, int sink) {
    if (position >= context_limit) return false;
    if (window <= 0 || position < sink) return true;
    const int start = context_limit - window;
    return position >= (start > sink ? start : sink);
}

// Batched-position variant of the tiled prefill kernel below. The per-position
// kernel spends four block-wide __syncthreads() on every history element, so an
// 8192-token prefill pays roughly half a million barriers per layer and becomes
// synchronisation bound rather than bandwidth bound. This scores kPosTile
// positions per barrier round: the dot products for the whole sub-tile are
// reduced together, then the online-softmax update walks the sub-tile in
// position order exactly as before. The running max/sum recurrence and the
// accumulator update order are therefore identical to the per-position kernel,
// so the result is bit-identical.
constexpr int kPosTile = 8;

template <int kHPG, int kVPT, int kQR>
__global__ void gqa_prefill_tiled_batched_f16_kernel(
    const uint16_t* __restrict__ q_rows,
    const uint16_t* __restrict__ k_cache,
    const uint16_t* __restrict__ v_cache,
    uint16_t* __restrict__ output,
    int seq_len,
    int q_heads,
    int kv_heads,
    int head_dim,
    int position_offset,
    int window,
    int sink) {
    constexpr int kCombosT = kHPG * kQR;
    const int q_per_kv = q_heads / kv_heads;
    const int groups_per_kv =
        (q_per_kv + kHPG - 1) / kHPG;
    const int kv_head = static_cast<int>(blockIdx.x) / groups_per_kv;
    const int group = static_cast<int>(blockIdx.x) % groups_per_kv;
    const int first_head = kv_head * q_per_kv + group * kHPG;
    const int first_token = static_cast<int>(blockIdx.y) * kQR;
    if (kv_head >= kv_heads || first_token >= seq_len) return;

    __shared__ float warp_sums[kPosTile][kCombosT][kWarps];
    __shared__ float scores[kPosTile][kCombosT];
    __shared__ float rescale[kPosTile][kCombosT];
    __shared__ float probability[kPosTile][kCombosT];
    __shared__ float running_max[kCombosT];
    __shared__ float running_sum[kCombosT];

    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    float query[kCombosT][kVPT];
    float accumulator[kCombosT][kVPT];
    bool active[kCombosT];
    int context_limit[kCombosT];

#pragma unroll
    for (int combo = 0; combo < kCombosT; ++combo) {
        const int query_row = combo / kHPG;
        const int head_in_group = combo % kHPG;
        const int token = first_token + query_row;
        const int head = first_head + head_in_group;
        active[combo] = token < seq_len &&
            head < (kv_head + 1) * q_per_kv && head < q_heads;
        context_limit[combo] = position_offset + token + 1;
#pragma unroll
        for (int i = 0; i < kVPT; ++i) {
            const int d = tid + i * kThreads;
            query[combo][i] = active[combo] && d < head_dim
                ? half_to_float(q_rows[
                      (static_cast<size_t>(token) * q_heads + head) * head_dim + d])
                : 0.0f;
            accumulator[combo][i] = 0.0f;
        }
        if (tid == combo) {
            running_max[combo] = -INFINITY;
            running_sum[combo] = 0.0f;
        }
    }
    __syncthreads();

    const int last_token = min(first_token + kQR - 1, seq_len - 1);
    const int tile_context = position_offset + last_token + 1;
    const int tile_window_start = window > 0
        ? max(sink, position_offset + first_token + 1 - window) : 0;
    const float attention_scale = rsqrtf(static_cast<float>(head_dim));
    const size_t kv_stride = static_cast<size_t>(kv_heads) * head_dim;

    for (int base = 0; base < tile_context; base += kPosTile) {
        // Skip whole sub-tiles that no query row in this CTA attends. The
        // per-position kernel jumps the same gap one element at a time.
        if (window > 0) {
            const int last = min(base + kPosTile, tile_context) - 1;
            if (last >= sink && last < tile_window_start) continue;
        }
        const int count = min(kPosTile, tile_context - base);
        float value[kPosTile][kVPT];
#pragma unroll
        for (int slot = 0; slot < kPosTile; ++slot) {
            if (slot >= count) break;
            const int position = base + slot;
            const bool skipped = window > 0 && position >= sink &&
                position < tile_window_start;
            const size_t kv_base = static_cast<size_t>(position) * kv_stride +
                static_cast<size_t>(kv_head) * head_dim;
            float key[kVPT];
#pragma unroll
            for (int i = 0; i < kVPT; ++i) {
                const int d = tid + i * kThreads;
                const bool load = !skipped && d < head_dim;
                key[i] = load ? half_to_float(k_cache[kv_base + d]) : 0.0f;
                value[slot][i] = load ? half_to_float(v_cache[kv_base + d]) : 0.0f;
            }
#pragma unroll
            for (int combo = 0; combo < kCombosT; ++combo) {
                float partial = 0.0f;
                if (!skipped && active[combo] &&
                    attends(position, context_limit[combo], window, sink)) {
#pragma unroll
                    for (int i = 0; i < kVPT; ++i) {
                        partial += query[combo][i] * key[i];
                    }
                }
                const float sum = warp_sum(partial);
                if (lane == 0) warp_sums[slot][combo][warp] = sum;
            }
        }
        __syncthreads();
        // One thread per (slot, combo) folds the warp partials, then the online
        // softmax recurrence is replayed in position order by thread `combo`.
        if (tid < kCombosT) {
            const int combo = tid;
            for (int slot = 0; slot < count; ++slot) {
                float sum = 0.0f;
#pragma unroll
                for (int w = 0; w < kWarps; ++w) sum += warp_sums[slot][combo][w];
                scores[slot][combo] = sum * attention_scale;
            }
            for (int slot = 0; slot < count; ++slot) {
                const int position = base + slot;
                const bool skipped = window > 0 && position >= sink &&
                    position < tile_window_start;
                if (!skipped && active[combo] &&
                    attends(position, context_limit[combo], window, sink)) {
                    const float old_max = running_max[combo];
                    const float new_max = fmaxf(old_max, scores[slot][combo]);
                    const float old_scale = old_max == -INFINITY
                        ? 0.0f : expf(old_max - new_max);
                    const float weight = expf(scores[slot][combo] - new_max);
                    running_max[combo] = new_max;
                    running_sum[combo] = running_sum[combo] * old_scale + weight;
                    rescale[slot][combo] = old_scale;
                    probability[slot][combo] = weight;
                } else {
                    rescale[slot][combo] = 1.0f;
                    probability[slot][combo] = 0.0f;
                }
            }
        }
        __syncthreads();
        // Replay the accumulator update in the same position order.
        for (int slot = 0; slot < count; ++slot) {
#pragma unroll
            for (int combo = 0; combo < kCombosT; ++combo) {
#pragma unroll
                for (int i = 0; i < kVPT; ++i) {
                    accumulator[combo][i] =
                        accumulator[combo][i] * rescale[slot][combo] +
                        probability[slot][combo] * value[slot][i];
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int combo = 0; combo < kCombosT; ++combo) {
        if (!active[combo]) continue;
        const int query_row = combo / kHPG;
        const int head_in_group = combo % kHPG;
        const int token = first_token + query_row;
        const int head = first_head + head_in_group;
        const float inverse = running_sum[combo] > 0.0f
            ? 1.0f / running_sum[combo] : 0.0f;
#pragma unroll
        for (int i = 0; i < kVPT; ++i) {
            const int d = tid + i * kThreads;
            if (d < head_dim) {
                output[(static_cast<size_t>(token) * q_heads + head) * head_dim + d] =
                    float_to_half(accumulator[combo][i] * inverse);
            }
        }
    }
}

// Warp-per-combo variant. The kernels above spread each dot product across all
// 128 threads, so with head_dim 128 every thread contributes a single FMA and
// then pays a five-shuffle reduction plus block barriers: measured 27.6 GB/s and
// 441 GFLOP/s on SM75, roughly 3% of both roofs, so the cost is reduction
// instructions rather than data. Here each warp owns a slice of the (query row,
// head) combos and each lane holds kDPL contiguous head dimensions, so a dot
// product is one warp butterfly with no block barrier and the online-softmax
// state stays in registers. K/V for the position tile is staged once in shared
// memory so the four warps do not re-read it. The reduction tree changes, so
// results differ from the per-position kernel in the last FP32 bits.
template <int kHPG, int kQR, int kDPL>
__global__ void gqa_prefill_warp_combo_f16_kernel(
    const uint16_t* __restrict__ q_rows,
    const uint16_t* __restrict__ k_cache,
    const uint16_t* __restrict__ v_cache,
    uint16_t* __restrict__ output,
    int seq_len,
    int q_heads,
    int kv_heads,
    int head_dim,
    int position_offset,
    int window,
    int sink) {
    constexpr int kCombosT = kHPG * kQR;
    constexpr int kCPW = (kCombosT + kWarps - 1) / kWarps;
    constexpr int kDim = kDPL * 32;
    const int q_per_kv = q_heads / kv_heads;
    const int groups_per_kv = (q_per_kv + kHPG - 1) / kHPG;
    const int kv_head = static_cast<int>(blockIdx.x) / groups_per_kv;
    const int group = static_cast<int>(blockIdx.x) % groups_per_kv;
    const int first_head = kv_head * q_per_kv + group * kHPG;
    const int first_token = static_cast<int>(blockIdx.y) * kQR;
    if (kv_head >= kv_heads || first_token >= seq_len) return;

    __shared__ uint16_t ks[kPosTile][kDim];
    __shared__ uint16_t vs[kPosTile][kDim];

    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int lane_base = lane * kDPL;

    float query[kCPW][kDPL];
    float accumulator[kCPW][kDPL];
    float running_max[kCPW];
    float running_sum[kCPW];
    bool active[kCPW];
    int context_limit[kCPW];

#pragma unroll
    for (int c = 0; c < kCPW; ++c) {
        const int combo = warp * kCPW + c;
        const int query_row = combo / kHPG;
        const int head_in_group = combo % kHPG;
        const int token = first_token + query_row;
        const int head = first_head + head_in_group;
        active[c] = combo < kCombosT && token < seq_len &&
            head < (kv_head + 1) * q_per_kv && head < q_heads;
        context_limit[c] = position_offset + token + 1;
        running_max[c] = -INFINITY;
        running_sum[c] = 0.0f;
#pragma unroll
        for (int i = 0; i < kDPL; ++i) {
            query[c][i] = active[c]
                ? half_to_float(q_rows[
                      (static_cast<size_t>(token) * q_heads + head) * head_dim +
                      lane_base + i])
                : 0.0f;
            accumulator[c][i] = 0.0f;
        }
    }

    const int last_token = min(first_token + kQR - 1, seq_len - 1);
    const int tile_context = position_offset + last_token + 1;
    const int tile_window_start = window > 0
        ? max(sink, position_offset + first_token + 1 - window) : 0;
    const float attention_scale = rsqrtf(static_cast<float>(head_dim));
    const size_t kv_stride = static_cast<size_t>(kv_heads) * head_dim;

    for (int base = 0; base < tile_context; base += kPosTile) {
        if (window > 0) {
            const int last = min(base + kPosTile, tile_context) - 1;
            if (last >= sink && last < tile_window_start) continue;
        }
        const int count = min(kPosTile, tile_context - base);
        __syncthreads();
        for (int idx = tid; idx < count * kDim; idx += kThreads) {
            const int slot = idx / kDim;
            const int d = idx - slot * kDim;
            const size_t at = static_cast<size_t>(base + slot) * kv_stride +
                static_cast<size_t>(kv_head) * head_dim + d;
            ks[slot][d] = k_cache[at];
            vs[slot][d] = v_cache[at];
        }
        __syncthreads();
        for (int slot = 0; slot < count; ++slot) {
            const int position = base + slot;
            const bool skipped = window > 0 && position >= sink &&
                position < tile_window_start;
            if (skipped) continue;
            float key[kDPL];
            float value[kDPL];
#pragma unroll
            for (int i = 0; i < kDPL; ++i) {
                key[i] = half_to_float(ks[slot][lane_base + i]);
                value[i] = half_to_float(vs[slot][lane_base + i]);
            }
#pragma unroll
            for (int c = 0; c < kCPW; ++c) {
                const bool contributes = active[c] &&
                    attends(position, context_limit[c], window, sink);
                float partial = 0.0f;
                if (contributes) {
#pragma unroll
                    for (int i = 0; i < kDPL; ++i) {
                        partial += query[c][i] * key[i];
                    }
                }
                // Butterfly reduction leaves the full sum in every lane, so no
                // broadcast is needed before the softmax update.
                const float score = warp_sum_all(partial) * attention_scale;
                if (!contributes) continue;
                const float old_max = running_max[c];
                const float new_max = fmaxf(old_max, score);
                const float old_scale = old_max == -INFINITY
                    ? 0.0f : expf(old_max - new_max);
                const float weight = expf(score - new_max);
                running_max[c] = new_max;
                running_sum[c] = running_sum[c] * old_scale + weight;
#pragma unroll
                for (int i = 0; i < kDPL; ++i) {
                    accumulator[c][i] =
                        accumulator[c][i] * old_scale + weight * value[i];
                }
            }
        }
    }

#pragma unroll
    for (int c = 0; c < kCPW; ++c) {
        if (!active[c]) continue;
        const int combo = warp * kCPW + c;
        const int token = first_token + combo / kHPG;
        const int head = first_head + combo % kHPG;
        const float inverse = running_sum[c] > 0.0f
            ? 1.0f / running_sum[c] : 0.0f;
#pragma unroll
        for (int i = 0; i < kDPL; ++i) {
            output[(static_cast<size_t>(token) * q_heads + head) * head_dim +
                   lane_base + i] = float_to_half(accumulator[c][i] * inverse);
        }
    }
}


// Two-phase flash tile for long FP16-cache prefill. The established kernels
// spend a five-shuffle warp reduction on every single key position; this one
// reduces once per 32-position tile instead:
//   phase A: lane p scores position base+p over the whole head_dim, so the dot
//            product needs no cross-lane reduction at all, and only the tile
//            max/denominator are reduced.
//   phase B: lane owns head_dim/32 output dimensions and walks the 32 stored
//            weights, so the value accumulation is also shuffle-free.
// K and V share one padded staging buffer: phase A only reads K and phase B
// only reads V. The +2 row padding makes the phase A access pattern (all lanes
// on distinct positions, same dimension) bank-conflict free.
template <int kDPL, int kQR>
__global__ void gqa_prefill_flash_tile_f16_kernel(
    const uint16_t* __restrict__ q_rows,
    const uint16_t* __restrict__ k_cache,
    const uint16_t* __restrict__ v_cache,
    uint16_t* __restrict__ output,
    int seq_len,
    int q_heads,
    int kv_heads,
    int head_dim,
    int position_offset) {
    constexpr int kWarpCount = 6;
    constexpr int kDim = kDPL * 32;
    constexpr int kTile = 32;
    constexpr int kPad = kDim + 2;
    const int kv_head = static_cast<int>(blockIdx.x);
    const int first_token = static_cast<int>(blockIdx.y) * kQR;
    if (kv_head >= kv_heads || first_token >= seq_len ||
        q_heads != kWarpCount || kv_heads != 1 || head_dim != kDim) {
        return;
    }

    __shared__ uint16_t stage[kTile][kPad];
    __shared__ uint16_t qs[kWarpCount][kQR][kDim];
    __shared__ float ws[kWarpCount][kQR][kTile];
    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;

    for (int index = tid; index < kWarpCount * kQR * kDim;
         index += kWarpCount * 32) {
        const int head = index / (kQR * kDim);
        const int rest = index - head * kQR * kDim;
        const int row = rest / kDim;
        const int d = rest - row * kDim;
        const int token = first_token + row;
        qs[head][row][d] = token < seq_len
            ? q_rows[(static_cast<size_t>(token) * q_heads + head) * kDim + d]
            : 0;
    }
    // Every warp reads the query tile owned by every other warp during phase A.
    // Without this barrier, the first K tile can observe a partially filled qs.
    __syncthreads();

    float accumulator[kQR][kDPL];
    float running_max[kQR];
    float running_sum[kQR];
#pragma unroll
    for (int row = 0; row < kQR; ++row) {
#pragma unroll
        for (int i = 0; i < kDPL; ++i) accumulator[row][i] = 0.0f;
        running_max[row] = -INFINITY;
        running_sum[row] = 0.0f;
    }

    const int last_token = min(first_token + kQR - 1, seq_len - 1);
    const int tile_context = position_offset + last_token + 1;
    const float attention_scale = rsqrtf(static_cast<float>(kDim));
    const size_t kv_stride = static_cast<size_t>(kv_heads) * kDim;

    for (int base = 0; base < tile_context; base += kTile) {
        const int count = min(kTile, tile_context - base);
        __syncthreads();
        for (int index = tid; index < count * kDim; index += kWarpCount * 32) {
            const int slot = index / kDim;
            const int d = index - slot * kDim;
            stage[slot][d] = k_cache[static_cast<size_t>(base + slot) *
                kv_stride + static_cast<size_t>(kv_head) * kDim + d];
        }
        __syncthreads();

        const int position = base + lane;
        float score[kQR];
#pragma unroll
        for (int row = 0; row < kQR; ++row) {
            const int token = first_token + row;
            const bool active = lane < count && token < seq_len &&
                position < position_offset + token + 1;
            float partial = 0.0f;
            if (active) {
                for (int d = 0; d < kDim; ++d) {
                    partial += half_to_float(qs[warp][row][d]) *
                        half_to_float(stage[lane][d]);
                }
            }
            score[row] = active ? partial * attention_scale : -INFINITY;
        }

        float old_scale[kQR];
#pragma unroll
        for (int row = 0; row < kQR; ++row) {
            float tile_max = score[row];
#pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1) {
                tile_max = fmaxf(tile_max,
                    __shfl_xor_sync(0xffffffffu, tile_max, offset));
            }
            const float new_max = fmaxf(running_max[row], tile_max);
            const float weight = new_max == -INFINITY || score[row] == -INFINITY
                ? 0.0f : expf(score[row] - new_max);
            ws[warp][row][lane] = weight;
            const float tile_sum = warp_sum_all(weight);
            old_scale[row] = running_max[row] == -INFINITY
                ? 0.0f : expf(running_max[row] - new_max);
            running_sum[row] = running_sum[row] * old_scale[row] + tile_sum;
            running_max[row] = new_max;
        }

        __syncthreads();
        for (int index = tid; index < count * kDim; index += kWarpCount * 32) {
            const int slot = index / kDim;
            const int d = index - slot * kDim;
            stage[slot][d] = v_cache[static_cast<size_t>(base + slot) *
                kv_stride + static_cast<size_t>(kv_head) * kDim + d];
        }
        __syncthreads();

#pragma unroll
        for (int row = 0; row < kQR; ++row) {
#pragma unroll
            for (int i = 0; i < kDPL; ++i) {
                accumulator[row][i] *= old_scale[row];
            }
            for (int slot = 0; slot < count; ++slot) {
                const float weight = ws[warp][row][slot];
                if (weight == 0.0f) continue;
#pragma unroll
                for (int i = 0; i < kDPL; ++i) {
                    accumulator[row][i] += weight *
                        half_to_float(stage[slot][lane * kDPL + i]);
                }
            }
        }
    }

#pragma unroll
    for (int row = 0; row < kQR; ++row) {
        const int token = first_token + row;
        if (token >= seq_len) continue;
        const float inverse = running_sum[row] > 0.0f
            ? 1.0f / running_sum[row] : 0.0f;
#pragma unroll
        for (int i = 0; i < kDPL; ++i) {
            const int d = lane * kDPL + i;
            output[(static_cast<size_t>(token) * q_heads + warp) * kDim + d] =
                float_to_half(accumulator[row][i] * inverse);
        }
    }
}

// Long-prefill path for the common Qwen TP4 shape: one six-warp CTA owns all
// six query heads of a KV head and two adjacent query rows. Compared with the
// generic hpg=2 path this loads each K/V tile once instead of three times, while
// keeping one warp's dot product and online-softmax state independent. Dispatch
// enables it by default for the validated exact shape; `=0` remains an escape
// hatch for A/B and fallback debugging.
template <int kDPL, int kQR>
__global__ void gqa_prefill_hpg6_f16_kernel(
    const uint16_t* __restrict__ q_rows,
    const uint16_t* __restrict__ k_cache,
    const uint16_t* __restrict__ v_cache,
    uint16_t* __restrict__ output,
    int seq_len,
    int q_heads,
    int kv_heads,
    int head_dim,
    int position_offset,
    int window,
    int sink) {
    constexpr int kWarpCount = 6;
    constexpr int kDim = kDPL * 32;
    const int kv_head = static_cast<int>(blockIdx.x);
    const int first_token = static_cast<int>(blockIdx.y) * kQR;
    const int q_per_kv = q_heads / kv_heads;
    if (kv_head >= kv_heads || first_token >= seq_len ||
        q_heads <= 0 || kv_heads <= 0 || q_heads % kv_heads != 0 ||
        q_per_kv != kWarpCount || head_dim != kDim) {
        return;
    }
    __shared__ uint16_t ks[kPosTile][kDim];
    __shared__ uint16_t vs[kPosTile][kDim];
    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    if (warp >= kWarpCount) return;

    float query[kQR][kDPL];
    float accumulator[kQR][kDPL];
    float running_max[kQR];
    float running_sum[kQR];
#pragma unroll
    for (int row = 0; row < kQR; ++row) {
        const int token = first_token + row;
        const bool active = token < seq_len;
        const size_t query_base =
            (static_cast<size_t>(token) * q_heads + warp) * head_dim;
#pragma unroll
        for (int i = 0; i < kDPL; ++i) {
            const int d = lane * kDPL + i;
            query[row][i] = active && d < head_dim
                ? half_to_float(q_rows[query_base + d]) : 0.0f;
            accumulator[row][i] = 0.0f;
        }
        running_max[row] = -INFINITY;
        running_sum[row] = 0.0f;
    }

    const int last_token = min(first_token + kQR - 1, seq_len - 1);
    const int tile_context = position_offset + last_token + 1;
    const int tile_window_start = window > 0
        ? max(sink, position_offset + first_token + 1 - window) : 0;
    const float attention_scale = rsqrtf(static_cast<float>(head_dim));
    const size_t kv_stride = static_cast<size_t>(kv_heads) * head_dim;

    for (int base = 0; base < tile_context; base += kPosTile) {
        if (window > 0) {
            const int last = min(base + kPosTile, tile_context) - 1;
            if (last >= sink && last < tile_window_start) continue;
        }
        const int count = min(kPosTile, tile_context - base);
        for (int index = tid; index < count * kDim; index += kWarpCount * 32) {
            const int slot = index / kDim;
            const int d = index - slot * kDim;
            const size_t at = static_cast<size_t>(base + slot) * kv_stride +
                static_cast<size_t>(kv_head) * head_dim + d;
            ks[slot][d] = d < head_dim ? k_cache[at] : 0;
            vs[slot][d] = d < head_dim ? v_cache[at] : 0;
        }
        __syncthreads();

        for (int slot = 0; slot < count; ++slot) {
            const int position = base + slot;
            if (window > 0 && position >= sink && position < tile_window_start) {
                continue;
            }
            float key[kDPL];
            float value[kDPL];
#pragma unroll
            for (int i = 0; i < kDPL; ++i) {
                key[i] = half_to_float(ks[slot][lane * kDPL + i]);
                value[i] = half_to_float(vs[slot][lane * kDPL + i]);
            }
#pragma unroll
            for (int row = 0; row < kQR; ++row) {
                const int token = first_token + row;
                if (token >= seq_len || position >= position_offset + token + 1) {
                    continue;
                }
                float partial = 0.0f;
#pragma unroll
                for (int i = 0; i < kDPL; ++i) {
                    partial += query[row][i] * key[i];
                }
                const float score = warp_sum_all(partial) * attention_scale;
                const float old_max = running_max[row];
                const float new_max = fmaxf(old_max, score);
                const float old_scale = old_max == -INFINITY
                    ? 0.0f : expf(old_max - new_max);
                const float weight = expf(score - new_max);
                running_max[row] = new_max;
                running_sum[row] = running_sum[row] * old_scale + weight;
#pragma unroll
                for (int i = 0; i < kDPL; ++i) {
                    accumulator[row][i] = accumulator[row][i] * old_scale +
                        weight * value[i];
                }
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int row = 0; row < kQR; ++row) {
        const int token = first_token + row;
        if (token >= seq_len) continue;
        const float inverse = running_sum[row] > 0.0f
            ? 1.0f / running_sum[row] : 0.0f;
#pragma unroll
        for (int i = 0; i < kDPL; ++i) {
            const int d = lane * kDPL + i;
            if (d < head_dim) {
                output[(static_cast<size_t>(token) * q_heads + warp) * head_dim + d] =
                    float_to_half(accumulator[row][i] * inverse);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tensor-core exact prefill.
//
// The scalar paths above are not bandwidth bound. Measured at the real TP4 shape
// (6 q heads over 1 kv head, head_dim 256, rows=4096 offset=32768) hpg6 takes
// 770 ms for 876 GFLOP, i.e. 1.14 TFLOP/s or roughly 8.5% of the FP32 CUDA-core
// peak, while the DRAM side only moves ~47 GB/s of a 616 GB/s card. The limit is
// scalar issue: every (row, position) pair costs five warp shuffles plus an expf.
// Tile-shape tuning cannot move that ceiling, so this path puts both GEMMs on the
// tensor cores instead.
//
// SM75 has no m16n8k16, so the primitive is m16n8k8 with FP32 accumulate. The
// dot products are reassociated relative to the scalar kernels, so this is exact
// in the sense that matters here (full causal history, no window, no sink) but
// not bit-identical to hpg6; it carries its own numerical and token-parity gates.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
#define POCKET_QWEN_MMA_AVAILABLE 1
#endif

__device__ __forceinline__ void mma_m16n8k8_f16_f32(
    float (&d)[4], const uint32_t (&a)[2], uint32_t b) {
#ifdef POCKET_QWEN_MMA_AVAILABLE
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

// One CTA owns 32 query rows of one Q head as two 16-row MMA blocks. The eight
// warps are indexed as (row block, slice): for Q*K each slice takes eight KV
// positions, for P*V each slice takes 64 of the 256 head dims. Both products are
// therefore full-width tensor-core work and the only scalar phase left is the
// 32x32 softmax. Four row blocks per CTA cuts long-context KV traffic in half.
// A 512-thread CTA is intentionally limited to one resident block so the
// accumulator registers are not forced to spill by a two-block launch bound.
constexpr int kMmaQRows = 64;
constexpr int kMmaRowBlocks = kMmaQRows / 16;
constexpr int kMmaKvTile = 32;
constexpr int kMmaDim = 256;
constexpr int kMmaSlices = 4;
constexpr int kMmaWarps = kMmaRowBlocks * kMmaSlices;
constexpr int kMmaSteps = kMmaDim / 8;
// Shared row strides are chosen for bank behaviour, and both must stay even so
// the 4-byte fragment loads remain aligned.
//
// ks stride 264 halves = 132 words: a Q*K fragment load varies over gid and tig,
// giving banks (132*gid + tig) % 32 = (4*gid + tig) % 32, which is 32 distinct
// banks. Conflict free.
//
// vt stride 34 halves = 17 words: the natural padding of 40 would make the
// transpose store stride 8*40/2 = 160 words, a multiple of 32, so all 32 lanes of
// a store would hit one bank. 34 cuts that to 8*34/2 = 136 ≡ 8 (mod 32), i.e. 4
// banks, and makes the fragment loads (17*gid + tig) % 32 nearly conflict free.
constexpr int kMmaKsStride = kMmaDim + 8;
constexpr int kMmaVtStride = kMmaKvTile + 2;

__global__ __launch_bounds__(kMmaWarps * 32) void gqa_prefill_mma_f16_kernel(
    const uint16_t* __restrict__ q_rows,
    const uint16_t* __restrict__ k_cache,
    const uint16_t* __restrict__ v_cache,
    uint16_t* __restrict__ output,
    int seq_len,
    int q_heads,
    int kv_heads,
    int head_dim,
    int position_offset) {
    __shared__ __align__(16) uint16_t ks[kMmaKvTile][kMmaKsStride];
    __shared__ __align__(16) uint16_t vt[kMmaDim][kMmaVtStride];
    __shared__ float sc[kMmaQRows][kMmaKvTile];
    __shared__ __align__(16) uint16_t ph[kMmaQRows][kMmaVtStride];
    __shared__ float row_m[kMmaQRows];
    __shared__ float row_l[kMmaQRows];
    __shared__ float row_alpha[kMmaQRows];

    const int head = static_cast<int>(blockIdx.x);
    const int qbase = static_cast<int>(blockIdx.y) * kMmaQRows;
    if (head >= q_heads || qbase >= seq_len || head_dim != kMmaDim ||
        q_heads <= 0 || kv_heads <= 0 || q_heads % kv_heads != 0) {
        return;
    }
    const int kv_head = head / (q_heads / kv_heads);
    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int block_row = (warp / kMmaSlices) * 16;
    const int slice = warp % kMmaSlices;
    const int lane = tid & 31;
    // m16n8k8 operand mapping: the eight lanes of a group share a matrix row and
    // the four lanes within a group cover two adjacent columns each.
    const int gid = lane >> 2;
    const int tig = lane & 3;
    // Rows of the 16x8 A/D fragment owned by this lane, in CTA-local terms.
    const int lrow0 = block_row + gid;
    const int lrow1 = block_row + gid + 8;

    float acc[8][4];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[j][i] = 0.0f;
    }
    if (tid < kMmaQRows) {
        row_m[tid] = -INFINITY;
        row_l[tid] = 0.0f;
    }

    // Q is reused by every KV tile, so it stays resident in registers.
    uint32_t qfrag[kMmaSteps][2];
#pragma unroll
    for (int k = 0; k < kMmaSteps; ++k) {
        const int col = k * 8 + tig * 2;
#pragma unroll
        for (int part = 0; part < 2; ++part) {
            const int row = qbase + (part == 0 ? lrow0 : lrow1);
            uint32_t value = 0;
            if (row < seq_len) {
                value = *reinterpret_cast<const uint32_t*>(
                    q_rows + (static_cast<size_t>(row) * q_heads + head) *
                        head_dim + col);
            }
            qfrag[k][part] = value;
        }
    }

    const size_t kv_stride = static_cast<size_t>(kv_heads) * head_dim;
    const int q_last = position_offset + min(qbase + kMmaQRows - 1, seq_len - 1);
    const int tile_context = q_last + 1;
    const float attention_scale = rsqrtf(static_cast<float>(head_dim));
    __syncthreads();

    for (int base = 0; base < tile_context; base += kMmaKvTile) {
        const int count = min(kMmaKvTile, tile_context - base);
        for (int idx = tid; idx < kMmaKvTile * kMmaSteps;
             idx += kMmaWarps * 32) {
            const int slot = idx / kMmaSteps;
            const int chunk = idx - slot * kMmaSteps;
            uint4 kv = make_uint4(0u, 0u, 0u, 0u);
            uint4 vv = make_uint4(0u, 0u, 0u, 0u);
            if (slot < count) {
                const size_t at = static_cast<size_t>(base + slot) * kv_stride +
                    static_cast<size_t>(kv_head) * head_dim + chunk * 8;
                kv = *reinterpret_cast<const uint4*>(k_cache + at);
                vv = *reinterpret_cast<const uint4*>(v_cache + at);
            }
            *reinterpret_cast<uint4*>(&ks[slot][chunk * 8]) = kv;
            // V is transposed on the way in so the P*V B operand, which wants
            // eight consecutive positions for one dim, is a 4-byte shared load.
            const uint16_t* src = reinterpret_cast<const uint16_t*>(&vv);
#pragma unroll
            for (int i = 0; i < 8; ++i) vt[chunk * 8 + i][slot] = src[i];
        }
        __syncthreads();

        float s[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int k = 0; k < kMmaSteps; ++k) {
            const uint32_t b = *reinterpret_cast<const uint32_t*>(
                &ks[slice * 8 + gid][k * 8 + tig * 2]);
            mma_m16n8k8_f16_f32(s, qfrag[k], b);
        }
        {
            const int col = slice * 8 + tig * 2;
            sc[lrow0][col] = s[0];
            sc[lrow0][col + 1] = s[1];
            sc[lrow1][col] = s[2];
            sc[lrow1][col + 1] = s[3];
        }
        __syncthreads();

        // Online softmax over the 16x32 tile. Eight threads share a row, so the
        // row reductions are three shuffles inside a warp.
        {
            const int row = tid >> 3;
            const int col0 = (tid & 7) * 4;
            const int token = qbase + row;
            const int limit = token < seq_len ? position_offset + token : -1;
            const float m_old = row_m[row];
            const float l_old = row_l[row];
            float sv[4];
            float local_max = -INFINITY;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int col = col0 + i;
                float value = -INFINITY;
                if (col < count && base + col <= limit) {
                    value = sc[row][col] * attention_scale;
                }
                sv[i] = value;
                local_max = fmaxf(local_max, value);
            }
#pragma unroll
            for (int off = 1; off < 8; off <<= 1) {
                local_max = fmaxf(local_max,
                                  __shfl_xor_sync(0xffffffffu, local_max, off));
            }
            const float m_new = fmaxf(m_old, local_max);
            float local_sum = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                // The numerator is rounded to half for the tensor-core operand,
                // so the denominator sums the rounded values to keep the two
                // consistent.
                uint16_t bits = 0;
                if (m_new != -INFINITY && sv[i] != -INFINITY) {
                    bits = float_to_half(expf(sv[i] - m_new));
                    local_sum += half_to_float(bits);
                }
                ph[row][col0 + i] = bits;
            }
#pragma unroll
            for (int off = 1; off < 8; off <<= 1) {
                local_sum += __shfl_xor_sync(0xffffffffu, local_sum, off);
            }
            __syncwarp();
            if ((tid & 7) == 0) {
                const float alpha =
                    m_old == -INFINITY ? 0.0f : expf(m_old - m_new);
                row_alpha[row] = alpha;
                row_m[row] = m_new;
                row_l[row] = l_old * alpha + local_sum;
            }
        }
        __syncthreads();

        {
            const float alpha0 = row_alpha[lrow0];
            const float alpha1 = row_alpha[lrow1];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                acc[j][0] *= alpha0;
                acc[j][1] *= alpha0;
                acc[j][2] *= alpha1;
                acc[j][3] *= alpha1;
            }
#pragma unroll
            for (int kt = 0; kt < kMmaKvTile / 8; ++kt) {
                uint32_t a[2];
                a[0] = *reinterpret_cast<const uint32_t*>(
                    &ph[lrow0][kt * 8 + tig * 2]);
                a[1] = *reinterpret_cast<const uint32_t*>(
                    &ph[lrow1][kt * 8 + tig * 2]);
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const uint32_t b = *reinterpret_cast<const uint32_t*>(
                        &vt[slice * 64 + j * 8 + gid][kt * 8 + tig * 2]);
                    mma_m16n8k8_f16_f32(acc[j], a, b);
                }
            }
        }
        __syncthreads();
    }

    const float inv0 = row_l[lrow0] > 0.0f ? 1.0f / row_l[lrow0] : 0.0f;
    const float inv1 = row_l[lrow1] > 0.0f ? 1.0f / row_l[lrow1] : 0.0f;
    const int row0 = qbase + lrow0;
    const int row1 = qbase + lrow1;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int dim = slice * 64 + j * 8 + tig * 2;
        if (row0 < seq_len) {
            uint16_t* out = output +
                (static_cast<size_t>(row0) * q_heads + head) * head_dim + dim;
            out[0] = float_to_half(acc[j][0] * inv0);
            out[1] = float_to_half(acc[j][1] * inv0);
        }
        if (row1 < seq_len) {
            uint16_t* out = output +
                (static_cast<size_t>(row1) * q_heads + head) * head_dim + dim;
            out[0] = float_to_half(acc[j][2] * inv1);
            out[1] = float_to_half(acc[j][3] * inv1);
        }
    }
}

// At 64 query rows the CTA is 512 threads, and asking for two resident blocks
// would cap registers at 64 per thread and spill the accumulators: measured at
// the real TP4 shape that costs 3.2x on the deep-history cases (268.4 ms against
// 83.6 at offset 61440). One resident block per SM is therefore requested
// explicitly, and the wider row tile pays for the lost occupancy by halving how
// often the KV history is re-streamed.
__global__ __launch_bounds__(kMmaWarps * 32, 1) void gqa_prefill_mma_occ_f16_kernel(
    const uint16_t* __restrict__ q_rows,
    const uint16_t* __restrict__ k_cache,
    const uint16_t* __restrict__ v_cache,
    uint16_t* __restrict__ output,
    int seq_len,
    int q_heads,
    int kv_heads,
    int head_dim,
    int position_offset) {
    // ks and vt are live in disjoint phases: Q*K reads ks and is fenced by a
    // __syncthreads() before the softmax, and P*V reads vt only after that. So
    // they share one buffer, which keeps the K/V staging at 17408 B instead of
    // 40960 B and leaves room for the wider row tile's sc/ph arrays.
    constexpr int kStageHalves =
        kMmaKvTile * kMmaKsStride > kMmaDim * kMmaVtStride
            ? kMmaKvTile * kMmaKsStride
            : kMmaDim * kMmaVtStride;
    __shared__ __align__(16) uint16_t kv_stage[kStageHalves];
    uint16_t (*ks)[kMmaKsStride] =
        reinterpret_cast<uint16_t (*)[kMmaKsStride]>(kv_stage);
    uint16_t (*vt)[kMmaVtStride] =
        reinterpret_cast<uint16_t (*)[kMmaVtStride]>(kv_stage);
    __shared__ float sc[kMmaQRows][kMmaKvTile];
    __shared__ __align__(16) uint16_t ph[kMmaQRows][kMmaVtStride];
    __shared__ float row_m[kMmaQRows];
    __shared__ float row_l[kMmaQRows];
    __shared__ float row_alpha[kMmaQRows];

    const int head = static_cast<int>(blockIdx.x);
    const int qbase = static_cast<int>(blockIdx.y) * kMmaQRows;
    if (head >= q_heads || qbase >= seq_len || head_dim != kMmaDim ||
        q_heads <= 0 || kv_heads <= 0 || q_heads % kv_heads != 0) {
        return;
    }
    const int kv_head = head / (q_heads / kv_heads);
    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int block_row = (warp / kMmaSlices) * 16;
    const int slice = warp % kMmaSlices;
    const int lane = tid & 31;
    // m16n8k8 operand mapping: the eight lanes of a group share a matrix row and
    // the four lanes within a group cover two adjacent columns each.
    const int gid = lane >> 2;
    const int tig = lane & 3;
    // Rows of the 16x8 A/D fragment owned by this lane, in CTA-local terms.
    const int lrow0 = block_row + gid;
    const int lrow1 = block_row + gid + 8;

    float acc[8][4];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[j][i] = 0.0f;
    }
    if (tid < kMmaQRows) {
        row_m[tid] = -INFINITY;
        row_l[tid] = 0.0f;
    }

    // Q is reused by every KV tile, so it stays resident in registers.
    uint32_t qfrag[kMmaSteps][2];
#pragma unroll
    for (int k = 0; k < kMmaSteps; ++k) {
        const int col = k * 8 + tig * 2;
#pragma unroll
        for (int part = 0; part < 2; ++part) {
            const int row = qbase + (part == 0 ? lrow0 : lrow1);
            uint32_t value = 0;
            if (row < seq_len) {
                value = *reinterpret_cast<const uint32_t*>(
                    q_rows + (static_cast<size_t>(row) * q_heads + head) *
                        head_dim + col);
            }
            qfrag[k][part] = value;
        }
    }

    const size_t kv_stride = static_cast<size_t>(kv_heads) * head_dim;
    const int q_last = position_offset + min(qbase + kMmaQRows - 1, seq_len - 1);
    const int tile_context = q_last + 1;
    const float attention_scale = rsqrtf(static_cast<float>(head_dim));
    __syncthreads();

    for (int base = 0; base < tile_context; base += kMmaKvTile) {
        const int count = min(kMmaKvTile, tile_context - base);
        for (int idx = tid; idx < kMmaKvTile * kMmaSteps;
             idx += kMmaWarps * 32) {
            const int slot = idx / kMmaSteps;
            const int chunk = idx - slot * kMmaSteps;
            uint4 kv = make_uint4(0u, 0u, 0u, 0u);
            if (slot < count) {
                const size_t at = static_cast<size_t>(base + slot) * kv_stride +
                    static_cast<size_t>(kv_head) * head_dim + chunk * 8;
                kv = *reinterpret_cast<const uint4*>(k_cache + at);
            }
            *reinterpret_cast<uint4*>(&ks[slot][chunk * 8]) = kv;
        }
        __syncthreads();

        float s[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int k = 0; k < kMmaSteps; ++k) {
            const uint32_t b = *reinterpret_cast<const uint32_t*>(
                &ks[slice * 8 + gid][k * 8 + tig * 2]);
            mma_m16n8k8_f16_f32(s, qfrag[k], b);
        }
        {
            const int col = slice * 8 + tig * 2;
            sc[lrow0][col] = s[0];
            sc[lrow0][col + 1] = s[1];
            sc[lrow1][col] = s[2];
            sc[lrow1][col + 1] = s[3];
        }
        // Scores are in sc[][] now, so ks is dead and V can reuse its memory.
        // V is transposed on the way in so the P*V B operand, which wants eight
        // consecutive positions for one dim, is a 4-byte shared load.
        __syncthreads();
        for (int idx = tid; idx < kMmaKvTile * kMmaSteps;
             idx += kMmaWarps * 32) {
            const int slot = idx / kMmaSteps;
            const int chunk = idx - slot * kMmaSteps;
            uint4 vv = make_uint4(0u, 0u, 0u, 0u);
            if (slot < count) {
                const size_t at = static_cast<size_t>(base + slot) * kv_stride +
                    static_cast<size_t>(kv_head) * head_dim + chunk * 8;
                vv = *reinterpret_cast<const uint4*>(v_cache + at);
            }
            const uint16_t* src = reinterpret_cast<const uint16_t*>(&vv);
#pragma unroll
            for (int i = 0; i < 8; ++i) vt[chunk * 8 + i][slot] = src[i];
        }
        __syncthreads();

        // Online softmax over the 16x32 tile. Eight threads share a row, so the
        // row reductions are three shuffles inside a warp.
        {
            const int row = tid >> 3;
            const int col0 = (tid & 7) * 4;
            const int token = qbase + row;
            const int limit = token < seq_len ? position_offset + token : -1;
            const float m_old = row_m[row];
            const float l_old = row_l[row];
            float sv[4];
            float local_max = -INFINITY;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int col = col0 + i;
                float value = -INFINITY;
                if (col < count && base + col <= limit) {
                    value = sc[row][col] * attention_scale;
                }
                sv[i] = value;
                local_max = fmaxf(local_max, value);
            }
#pragma unroll
            for (int off = 1; off < 8; off <<= 1) {
                local_max = fmaxf(local_max,
                                  __shfl_xor_sync(0xffffffffu, local_max, off));
            }
            const float m_new = fmaxf(m_old, local_max);
            float local_sum = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                // The numerator is rounded to half for the tensor-core operand,
                // so the denominator sums the rounded values to keep the two
                // consistent.
                uint16_t bits = 0;
                if (m_new != -INFINITY && sv[i] != -INFINITY) {
                    bits = float_to_half(expf(sv[i] - m_new));
                    local_sum += half_to_float(bits);
                }
                ph[row][col0 + i] = bits;
            }
#pragma unroll
            for (int off = 1; off < 8; off <<= 1) {
                local_sum += __shfl_xor_sync(0xffffffffu, local_sum, off);
            }
            __syncwarp();
            if ((tid & 7) == 0) {
                const float alpha =
                    m_old == -INFINITY ? 0.0f : expf(m_old - m_new);
                row_alpha[row] = alpha;
                row_m[row] = m_new;
                row_l[row] = l_old * alpha + local_sum;
            }
        }
        __syncthreads();

        {
            const float alpha0 = row_alpha[lrow0];
            const float alpha1 = row_alpha[lrow1];
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                acc[j][0] *= alpha0;
                acc[j][1] *= alpha0;
                acc[j][2] *= alpha1;
                acc[j][3] *= alpha1;
            }
#pragma unroll
            for (int kt = 0; kt < kMmaKvTile / 8; ++kt) {
                uint32_t a[2];
                a[0] = *reinterpret_cast<const uint32_t*>(
                    &ph[lrow0][kt * 8 + tig * 2]);
                a[1] = *reinterpret_cast<const uint32_t*>(
                    &ph[lrow1][kt * 8 + tig * 2]);
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const uint32_t b = *reinterpret_cast<const uint32_t*>(
                        &vt[slice * 64 + j * 8 + gid][kt * 8 + tig * 2]);
                    mma_m16n8k8_f16_f32(acc[j], a, b);
                }
            }
        }
        __syncthreads();
    }

    const float inv0 = row_l[lrow0] > 0.0f ? 1.0f / row_l[lrow0] : 0.0f;
    const float inv1 = row_l[lrow1] > 0.0f ? 1.0f / row_l[lrow1] : 0.0f;
    const int row0 = qbase + lrow0;
    const int row1 = qbase + lrow1;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int dim = slice * 64 + j * 8 + tig * 2;
        if (row0 < seq_len) {
            uint16_t* out = output +
                (static_cast<size_t>(row0) * q_heads + head) * head_dim + dim;
            out[0] = float_to_half(acc[j][0] * inv0);
            out[1] = float_to_half(acc[j][1] * inv0);
        }
        if (row1 < seq_len) {
            uint16_t* out = output +
                (static_cast<size_t>(row1) * q_heads + head) * head_dim + dim;
            out[0] = float_to_half(acc[j][2] * inv1);
            out[1] = float_to_half(acc[j][3] * inv1);
        }
    }
}

// Two adjacent query rows and three Q heads sharing a KV head are processed by
// one CTA. Each K/V element is loaded once for six attention outputs.
__global__ void gqa_prefill_tiled_f16_kernel(
    const uint16_t* __restrict__ q_rows,
    const uint16_t* __restrict__ k_cache,
    const uint16_t* __restrict__ v_cache,
    uint16_t* __restrict__ output,
    int seq_len,
    int q_heads,
    int kv_heads,
    int head_dim,
    int position_offset,
    int window,
    int sink) {
    const int q_per_kv = q_heads / kv_heads;
    const int groups_per_kv =
        (q_per_kv + kHeadsPerGroup - 1) / kHeadsPerGroup;
    const int kv_head = static_cast<int>(blockIdx.x) / groups_per_kv;
    const int group = static_cast<int>(blockIdx.x) % groups_per_kv;
    const int first_head = kv_head * q_per_kv + group * kHeadsPerGroup;
    const int first_token = static_cast<int>(blockIdx.y) * kQueryRows;
    if (kv_head >= kv_heads || first_token >= seq_len) return;

    __shared__ float warp_sums[kPrefillCombos][kWarps];
    __shared__ float scores[kPrefillCombos];
    __shared__ float running_max[kPrefillCombos];
    __shared__ float running_sum[kPrefillCombos];
    __shared__ float rescale[kPrefillCombos];
    __shared__ float probability[kPrefillCombos];

    const int tid = static_cast<int>(threadIdx.x);
    float query[kPrefillCombos][kValuesPerThread];
    float accumulator[kPrefillCombos][kValuesPerThread];
    bool active[kPrefillCombos];
    int context_limit[kPrefillCombos];

#pragma unroll
    for (int combo = 0; combo < kPrefillCombos; ++combo) {
        const int query_row = combo / kHeadsPerGroup;
        const int head_in_group = combo % kHeadsPerGroup;
        const int token = first_token + query_row;
        const int head = first_head + head_in_group;
        active[combo] = token < seq_len &&
            head < (kv_head + 1) * q_per_kv && head < q_heads;
        context_limit[combo] = position_offset + token + 1;
#pragma unroll
        for (int i = 0; i < kValuesPerThread; ++i) {
            const int d = tid + i * kThreads;
            query[combo][i] = active[combo] && d < head_dim
                ? half_to_float(q_rows[
                      (static_cast<size_t>(token) * q_heads + head) * head_dim + d])
                : 0.0f;
            accumulator[combo][i] = 0.0f;
        }
        if (tid == combo) {
            running_max[combo] = -INFINITY;
            running_sum[combo] = 0.0f;
        }
    }
    __syncthreads();

    const int last_token = min(first_token + kQueryRows - 1, seq_len - 1);
    const int tile_context = position_offset + last_token + 1;
    // The earliest query in this tile bounds the shared window; positions
    // between the sink prefix and that bound are attended by no query row.
    const int tile_window_start = window > 0
        ? max(sink, position_offset + first_token + 1 - window) : 0;
    const float attention_scale = rsqrtf(static_cast<float>(head_dim));
    const size_t kv_stride = static_cast<size_t>(kv_heads) * head_dim;
    for (int position = 0; position < tile_context; ++position) {
        if (window > 0 && position >= sink && position < tile_window_start) {
            position = tile_window_start - 1;
            continue;
        }
        const size_t kv_base = static_cast<size_t>(position) * kv_stride +
            static_cast<size_t>(kv_head) * head_dim;
        float key[kValuesPerThread];
        float value[kValuesPerThread];
#pragma unroll
        for (int i = 0; i < kValuesPerThread; ++i) {
            const int d = tid + i * kThreads;
            key[i] = d < head_dim ? half_to_float(k_cache[kv_base + d]) : 0.0f;
            value[i] = d < head_dim ? half_to_float(v_cache[kv_base + d]) : 0.0f;
        }

        float partial[kPrefillCombos] = {};
#pragma unroll
        for (int combo = 0; combo < kPrefillCombos; ++combo) {
            if (active[combo] &&
                attends(position, context_limit[combo], window, sink)) {
#pragma unroll
                for (int i = 0; i < kValuesPerThread; ++i) {
                    partial[combo] += query[combo][i] * key[i];
                }
            }
        }
        reduce_dot_products(partial, warp_sums, scores, attention_scale);

        if (tid < kPrefillCombos) {
            if (active[tid] &&
                attends(position, context_limit[tid], window, sink)) {
                const float old_max = running_max[tid];
                const float new_max = fmaxf(old_max, scores[tid]);
                const float old_scale = old_max == -INFINITY
                    ? 0.0f : expf(old_max - new_max);
                const float weight = expf(scores[tid] - new_max);
                running_max[tid] = new_max;
                running_sum[tid] = running_sum[tid] * old_scale + weight;
                rescale[tid] = old_scale;
                probability[tid] = weight;
            } else {
                rescale[tid] = 1.0f;
                probability[tid] = 0.0f;
            }
        }
        __syncthreads();
#pragma unroll
        for (int combo = 0; combo < kPrefillCombos; ++combo) {
#pragma unroll
            for (int i = 0; i < kValuesPerThread; ++i) {
                accumulator[combo][i] = accumulator[combo][i] * rescale[combo] +
                    probability[combo] * value[i];
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int combo = 0; combo < kPrefillCombos; ++combo) {
        if (!active[combo]) continue;
        const int query_row = combo / kHeadsPerGroup;
        const int head_in_group = combo % kHeadsPerGroup;
        const int token = first_token + query_row;
        const int head = first_head + head_in_group;
        const float inverse = running_sum[combo] > 0.0f
            ? 1.0f / running_sum[combo] : 0.0f;
#pragma unroll
        for (int i = 0; i < kValuesPerThread; ++i) {
            const int d = tid + i * kThreads;
            if (d < head_dim) {
                output[(static_cast<size_t>(token) * q_heads + head) * head_dim + d] =
                    float_to_half(accumulator[combo][i] * inverse);
            }
        }
    }
}

// Each CTA computes exact online-softmax partials for kHPG Q heads over one
// context split while loading their shared K/V row only once. The width matters
// at long context: with kHPG below q_per_kv the KV head is split across several
// CTAs that each stream the whole slice, so a 65K step read the 64 MiB cache
// twice per layer (2 groups x 6 Q heads / 3). Widening the group to cover every
// Q head sharing the KV head removes that duplicate traffic outright. Per-head
// arithmetic is untouched: the dot product still reduces over the same lanes in
// the same warp_sum tree, and the softmax recurrence still walks positions in
// order, so a given split count gives bit-identical output at any width.
// Batched decode drives one CTA row per sequence through blockIdx.y. Every
// row-dependent value is resolved before the position loop, so the arithmetic
// inside it -- and therefore the result for a given split count -- is identical
// to the single-row instantiation. `kBatched=false` keeps the original codegen
// with no extra registers for the unused row indirection.
//
// `kPaged` selects the paged KV cache, where a sequence's positions live in
// fixed-size blocks scattered through one shared arena instead of in a
// contiguous per-slot reservation. It is a separate template parameter rather
// than a runtime null check so the contiguous instantiation keeps its exact
// register count and address arithmetic.
template <int kHPG, bool kBatched = false, bool kPaged = false>
__global__ void gqa_decode_split_f16_kernel(
    const uint16_t* __restrict__ q,
    const uint16_t* __restrict__ k_cache,
    const uint16_t* __restrict__ v_cache,
    float* __restrict__ partial_output,
    int q_heads,
    int kv_heads,
    int head_dim,
    int context_len,
    int splits,
    int positions_per_split,
    int window,
    int sink,
    const int* __restrict__ context_lens = nullptr,
    const int* __restrict__ slot_ids = nullptr,
    size_t kv_slot_stride = 0,
    const int* __restrict__ block_table = nullptr,
    int block_size = 0,
    int max_blocks_per_seq = 0) {
    const int q_per_kv = q_heads / kv_heads;
    const int groups_per_kv = (q_per_kv + kHPG - 1) / kHPG;
    const int grouped = static_cast<int>(blockIdx.x) / splits;
    const int split = static_cast<int>(blockIdx.x) % splits;
    const int kv_head = grouped / groups_per_kv;
    const int group = grouped % groups_per_kv;
    const int first_head = kv_head * q_per_kv + group * kHPG;
    // Each sequence owns a contiguous KV slot and its own context length. The
    // launch sizes the split geometry from the longest row, so shorter rows run
    // some splits dry and publish the neutral partial the merge already handles.
    // The slot is not the batch index: slots are allocated as requests arrive,
    // so a four-row batch can hold slots 1/3/5/6.
    const int* __restrict__ slot_blocks = nullptr;
    if constexpr (kBatched) {
        const int row = static_cast<int>(blockIdx.y);
        context_len = context_lens[row];
        const int slot_index = slot_ids != nullptr ? slot_ids[row] : row;
        if constexpr (kPaged) {
            // Paged rows share one arena, so the row's base is the table it
            // indexes through rather than an offset added to the pointer.
            slot_blocks = block_table +
                static_cast<size_t>(slot_index) * max_blocks_per_seq;
        } else {
            const size_t slot =
                static_cast<size_t>(slot_index) * kv_slot_stride;
            k_cache += slot;
            v_cache += slot;
        }
        q += static_cast<size_t>(row) * q_heads * head_dim;
        partial_output += static_cast<size_t>(row) * q_heads * splits *
            (head_dim + 2);
    }
    const SparseRanges ranges = sparse_ranges(context_len, window, sink);
    const int start = split * positions_per_split;
    const int end = min(ranges.total(), start + positions_per_split);
    // An empty split still has to publish its partial. Rounding the slice length
    // up leaves the last splits with nothing to scan whenever the split count
    // does not divide the context -- 8193 positions over 256 splits is 33 each,
    // so seven splits run dry. Returning early would leave their scratch slots
    // uninitialized and the merge would read them, which showed up as -inf
    // logits from the second decode step onward. Falling through with a zero-trip
    // loop writes the neutral (-inf max, zero sum, zero accumulator) partial that
    // the merge already handles: exp(-inf - max) contributes nothing.
    if (kv_head >= kv_heads) return;

    __shared__ float warp_sums[kHPG][kWarps];
    __shared__ float scores[kHPG];
    __shared__ float running_max[kHPG];
    __shared__ float running_sum[kHPG];
    __shared__ float rescale[kHPG];
    __shared__ float probability[kHPG];

    const int tid = static_cast<int>(threadIdx.x);
    float query[kHPG][kValuesPerThread];
    float accumulator[kHPG][kValuesPerThread] = {};
    bool active[kHPG];
#pragma unroll
    for (int h = 0; h < kHPG; ++h) {
        const int head = first_head + h;
        active[h] = head < (kv_head + 1) * q_per_kv && head < q_heads;
#pragma unroll
        for (int i = 0; i < kValuesPerThread; ++i) {
            const int d = tid + i * kThreads;
            query[h][i] = active[h] && d < head_dim
                ? half_to_float(q[static_cast<size_t>(head) * head_dim + d])
                : 0.0f;
        }
        if (tid == h) {
            running_max[h] = -INFINITY;
            running_sum[h] = 0.0f;
        }
    }
    __syncthreads();

    const float attention_scale = rsqrtf(static_cast<float>(head_dim));
    const size_t kv_stride = static_cast<size_t>(kv_heads) * head_dim;
    // Paged translation is loop-invariant within a block, so the block id is
    // cached and refetched only when the scan leaves it. The sparse ranges walk
    // positions in ascending order within a split, and the sink-to-window jump
    // is the only discontinuity, so a one-entry cache covers the whole scan with
    // one table read per `block_size` positions rather than one per position.
    int cached_block_index = -1;
    size_t cached_block_base = 0;
    for (int logical = start; logical < end; ++logical) {
        const int position = ranges.position(logical);
        size_t kv_base;
        if constexpr (kPaged) {
            const int block_index = position / block_size;
            if (block_index != cached_block_index) {
                cached_block_index = block_index;
                cached_block_base =
                    static_cast<size_t>(slot_blocks[block_index]) *
                    block_size * kv_stride;
            }
            kv_base = cached_block_base +
                static_cast<size_t>(position % block_size) * kv_stride +
                static_cast<size_t>(kv_head) * head_dim;
        } else {
            kv_base = static_cast<size_t>(position) * kv_stride +
                static_cast<size_t>(kv_head) * head_dim;
        }
        float key[kValuesPerThread];
        float value[kValuesPerThread];
#pragma unroll
        for (int i = 0; i < kValuesPerThread; ++i) {
            const int d = tid + i * kThreads;
            key[i] = d < head_dim ? half_to_float(k_cache[kv_base + d]) : 0.0f;
            value[i] = d < head_dim ? half_to_float(v_cache[kv_base + d]) : 0.0f;
        }
        float dot[kHPG] = {};
#pragma unroll
        for (int h = 0; h < kHPG; ++h) {
#pragma unroll
            for (int i = 0; i < kValuesPerThread; ++i) {
                dot[h] += query[h][i] * key[i];
            }
        }
        reduce_dot_products(dot, warp_sums, scores, attention_scale);
        if (tid < kHPG) {
            if (active[tid]) {
                const float old_max = running_max[tid];
                const float new_max = fmaxf(old_max, scores[tid]);
                const float old_scale = old_max == -INFINITY
                    ? 0.0f : expf(old_max - new_max);
                const float weight = expf(scores[tid] - new_max);
                running_max[tid] = new_max;
                running_sum[tid] = running_sum[tid] * old_scale + weight;
                rescale[tid] = old_scale;
                probability[tid] = weight;
            } else {
                rescale[tid] = 1.0f;
                probability[tid] = 0.0f;
            }
        }
        __syncthreads();
#pragma unroll
        for (int h = 0; h < kHPG; ++h) {
#pragma unroll
            for (int i = 0; i < kValuesPerThread; ++i) {
                accumulator[h][i] = accumulator[h][i] * rescale[h] +
                    probability[h] * value[i];
            }
        }
        // No trailing barrier is needed. Every thread reads only its own
        // accumulator registers here, while rescale/probability cannot be
        // overwritten by the next position until reduce_dot_products() reaches
        // its first block-wide barrier. That barrier protects the shared
        // softmax scalars before their next write.
    }

    const int partial_stride = head_dim + 2;
#pragma unroll
    for (int h = 0; h < kHPG; ++h) {
        if (!active[h]) continue;
        const int head = first_head + h;
        float* destination = partial_output +
            (static_cast<size_t>(head) * splits + split) * partial_stride;
        if (tid == 0) {
            destination[0] = running_max[h];
            destination[1] = running_sum[h];
        }
#pragma unroll
        for (int i = 0; i < kValuesPerThread; ++i) {
            const int d = tid + i * kThreads;
            if (d < head_dim) destination[2 + d] = accumulator[h][i];
        }
    }
}

// The decode tensor-core shape has six Q rows sharing each KV head. This candidate
// maps one group to the M dimension of SM75's m16n8k8 instruction, padding ten
// inactive rows, and maps four warps to four eight-position N tiles. The same
// 32-position tile is then used for P*V. Unlike the scalar split kernel, a CTA
// scans a longer slice and performs the two 256-wide products on tensor cores.
// The softmax remains an ordered FP32 recurrence and the probability operand is
// rounded to half before P*V, matching the proven prefill MMA path. Consequently
// this is numerically close rather than bit-identical and is opt-in.
constexpr int kDecodeMmaThreads = 128;
constexpr int kDecodeMmaRows = 16;
constexpr int kDecodeMmaValidRows = 6;
// The tile stays at 32 positions. Halving it to 16 cuts shared memory to 9864
// bytes but the CTA still needs 133 registers, so 4 * 128 * 133 = 68096 exceeds
// the 65536-register file and occupancy stays at three CTAs per SM either way.
// The same register wall blocks `__launch_bounds__(128, 4)`: capping registers at
// 128 fits the file but 4 * 18632 bytes of shared memory does not, and the
// microbenchmark confirmed no change (65536: 0.3068 ms against 0.3066 ms).
constexpr int kDecodeMmaKvTile = 32;
constexpr int kDecodeMmaDim = 256;
constexpr int kDecodeMmaSteps = kDecodeMmaDim / 8;
constexpr int kDecodeMmaKsStride = kDecodeMmaDim + 8;
constexpr int kDecodeMmaVtStride = kDecodeMmaKvTile + 2;

// Six valid rows all fall in the first half of the MMA M dimension, so the
// second row a lane owns (`gid + 8`) is never active at this shape. Only the
// first half of every accumulator fragment is therefore consumed, and the
// per-lane output set is small enough to live in registers.
static_assert(kDecodeMmaValidRows <= 8,
              "the decode MMA epilogue assumes the padded row half is inactive");

// Shared memory is what limits this kernel's occupancy: the register budget
// admits four CTAs per SM but every byte above 21845 costs a resident CTA. So
// the score and probability tiles are cut to the valid rows and the output
// accumulator is held in registers, where each element already had exactly one
// writer.
__global__ __launch_bounds__(kDecodeMmaThreads, 3)
void gqa_decode_split_mma_f16_kernel(
    const uint16_t* __restrict__ q,
    const uint16_t* __restrict__ k_cache,
    const uint16_t* __restrict__ v_cache,
    float* __restrict__ partial_output,
    int kv_groups,
    int context_len,
    int splits,
    int positions_per_split) {
    __shared__ __align__(16) uint16_t kv_stage[
        kDecodeMmaDim * kDecodeMmaVtStride];
    uint16_t (*ks)[kDecodeMmaKsStride] =
        reinterpret_cast<uint16_t (*)[kDecodeMmaKsStride]>(kv_stage);
    uint16_t (*vt)[kDecodeMmaVtStride] =
        reinterpret_cast<uint16_t (*)[kDecodeMmaVtStride]>(kv_stage);
    __shared__ float scores[kDecodeMmaValidRows][kDecodeMmaKvTile];
    __shared__ uint16_t probability[kDecodeMmaValidRows][kDecodeMmaKvTile];
    __shared__ float running_max[kDecodeMmaValidRows];
    __shared__ float running_sum[kDecodeMmaValidRows];
    __shared__ float rescale[kDecodeMmaValidRows];

    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int gid = lane >> 2;
    const int tig = lane & 3;
    const int lrow0 = gid;
    const int lrow1 = gid + 8;
    const int split = static_cast<int>(blockIdx.x);
    const int kv_group = static_cast<int>(blockIdx.y);
    if (kv_group >= kv_groups) return;
    const size_t q_offset =
        static_cast<size_t>(kv_group) * kDecodeMmaValidRows * kDecodeMmaDim;
    const size_t kv_offset = static_cast<size_t>(kv_group) * kDecodeMmaDim;
    const size_t partial_group_offset =
        static_cast<size_t>(kv_group) * kDecodeMmaValidRows * splits *
        (kDecodeMmaDim + 2);
    const int start = split * positions_per_split;
    const int end = min(context_len, start + positions_per_split);

    // The eight P*V fragments a lane accumulates. Lane `lane` owns row `gid` and
    // channels `slice * 64 + j * 8 + tig * 2` for j in [0, 8), which is a
    // disjoint set across the CTA.
    float accumulator[8][2];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        accumulator[j][0] = 0.0f;
        accumulator[j][1] = 0.0f;
    }
    if (tid < kDecodeMmaValidRows) {
        running_max[tid] = -INFINITY;
        running_sum[tid] = 0.0f;
        rescale[tid] = 0.0f;
    }
    __syncthreads();

    // Each lane keeps the fragment for two rows of the padded 16-row Q matrix.
    // Q is reused for every K/V tile, so keeping it in registers avoids a second
    // global read of the query vector.
    uint32_t qfrag[kDecodeMmaSteps][2];
#pragma unroll
    for (int k = 0; k < kDecodeMmaSteps; ++k) {
        const int col = k * 8 + tig * 2;
        uint32_t first = 0u;
        uint32_t second = 0u;
        if (lrow0 < kDecodeMmaValidRows) {
            first = *reinterpret_cast<const uint32_t*>(q + q_offset +
                static_cast<size_t>(lrow0) * kDecodeMmaDim + col);
        }
        if (lrow1 < kDecodeMmaValidRows) {
            second = *reinterpret_cast<const uint32_t*>(q + q_offset +
                static_cast<size_t>(lrow1) * kDecodeMmaDim + col);
        }
        qfrag[k][0] = first;
        qfrag[k][1] = second;
    }

    const float attention_scale = rsqrtf(static_cast<float>(kDecodeMmaDim));
    for (int base = start; base < end; base += kDecodeMmaKvTile) {
        const int count = min(kDecodeMmaKvTile, end - base);
        const size_t kv_stride = kDecodeMmaDim;
        for (int index = tid;
             index < kDecodeMmaKvTile * kDecodeMmaSteps;
             index += kDecodeMmaThreads) {
            const int slot = index / kDecodeMmaSteps;
            const int chunk = index - slot * kDecodeMmaSteps;
            uint4 key = make_uint4(0u, 0u, 0u, 0u);
            if (slot < count) {
                key = *reinterpret_cast<const uint4*>(k_cache +
                    static_cast<size_t>(base + slot) * kv_groups * kv_stride +
                    kv_offset + chunk * 8);
            }
            *reinterpret_cast<uint4*>(&ks[slot][chunk * 8]) = key;
        }
        __syncthreads();

        float score_fragment[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int k = 0; k < kDecodeMmaSteps; ++k) {
            const uint32_t b = *reinterpret_cast<const uint32_t*>(
                &ks[warp * 8 + gid][k * 8 + tig * 2]);
            mma_m16n8k8_f16_f32(score_fragment, qfrag[k], b);
        }
        const int score_col = warp * 8 + tig * 2;
        if (lrow0 < kDecodeMmaValidRows) {
            scores[lrow0][score_col] = score_fragment[0];
            scores[lrow0][score_col + 1] = score_fragment[1];
        }
        __syncthreads();

        // Eight threads own one score row and reduce its 32 columns. Invalid
        // positions in the final tile are represented as -inf/zero.
        const int row = tid >> 3;
        const int row_lane = tid & 7;
        // All eight-lane row groups must execute the warp reductions with the
        // full mask. Padded rows participate with -inf and are discarded after
        // the reduction; branching before __shfl_xor would leave lanes 6/7 of
        // the second warp out of a full-mask collective.
        if (row < kDecodeMmaRows) {
            const int col0 = row_lane * 4;
            const bool valid_row = row < kDecodeMmaValidRows;
            const float old_max = valid_row ? running_max[row] : -INFINITY;
            const float old_sum = valid_row ? running_sum[row] : 0.0f;
            float local_max = -INFINITY;
            float values[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int col = col0 + i;
                const float value = valid_row && col < count
                    ? scores[row < kDecodeMmaValidRows ? row : 0][col] *
                          attention_scale
                    : -INFINITY;
                values[i] = value;
                local_max = fmaxf(local_max, value);
            }
#pragma unroll
            for (int offset = 1; offset < 8; offset <<= 1) {
                local_max = fmaxf(local_max,
                    __shfl_xor_sync(0xffffffffu, local_max, offset));
            }
            const float new_max = fmaxf(old_max, local_max);
            float local_sum = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                uint16_t bits = 0;
                if (new_max != -INFINITY && values[i] != -INFINITY) {
                    bits = float_to_half(expf(values[i] - new_max));
                    local_sum += half_to_float(bits);
                }
                if (valid_row) {
                    probability[row][col0 + i] = bits;
                }
            }
#pragma unroll
            for (int offset = 1; offset < 8; offset <<= 1) {
                local_sum += __shfl_xor_sync(0xffffffffu, local_sum, offset);
            }
            if (valid_row && row_lane == 0) {
                rescale[row] = old_max == -INFINITY
                    ? 0.0f : expf(old_max - new_max);
                running_max[row] = new_max;
                running_sum[row] = old_sum * rescale[row] + local_sum;
            }
        }
        __syncthreads();

        // The V tile reuses the K staging buffer after Q*K and score reduction
        // have completed. Transposing it makes each B fragment a coalesced 32-bit
        // shared load, as in the prefill MMA kernel.
        for (int index = tid;
             index < kDecodeMmaKvTile * kDecodeMmaSteps;
             index += kDecodeMmaThreads) {
            const int slot = index / kDecodeMmaSteps;
            const int chunk = index - slot * kDecodeMmaSteps;
            uint4 value = make_uint4(0u, 0u, 0u, 0u);
            if (slot < count) {
                value = *reinterpret_cast<const uint4*>(v_cache +
                    static_cast<size_t>(base + slot) * kv_groups * kv_stride +
                    kv_offset + chunk * 8);
            }
            const uint16_t* source = reinterpret_cast<const uint16_t*>(&value);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                vt[chunk * 8 + i][slot] = source[i];
            }
        }
        __syncthreads();

        // Each warp covers a distinct 64-channel output slice. The B operand is
        // column-major in the transposed V staging buffer; `gid` selects one of
        // the eight output columns in the MMA fragment while `tig` selects its
        // two values written by this lane. Every output element is therefore
        // owned by one lane and the shared accumulator needs no atomics.
        const int slice = warp;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int channel = slice * 64 + j * 8;
            float value_fragment[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
            for (int kt = 0; kt < kDecodeMmaKvTile / 8; ++kt) {
                // The second row half is inactive, so its A fragment stays zero
                // and is never read out of the six-row probability tile.
                uint32_t a[2] = {0u, 0u};
                if (lrow0 < kDecodeMmaValidRows) {
                    a[0] = *reinterpret_cast<const uint32_t*>(
                        &probability[lrow0][kt * 8 + tig * 2]);
                }
                const uint32_t b = *reinterpret_cast<const uint32_t*>(
                    &vt[channel + gid][kt * 8 + tig * 2]);
                mma_m16n8k8_f16_f32(value_fragment, a, b);
            }
            if (lrow0 < kDecodeMmaValidRows) {
                const float row_rescale = rescale[lrow0];
                accumulator[j][0] =
                    accumulator[j][0] * row_rescale + value_fragment[0];
                accumulator[j][1] =
                    accumulator[j][1] * row_rescale + value_fragment[1];
            }
        }
        __syncthreads();
    }

    const int partial_stride = kDecodeMmaDim + 2;
    if (tid < kDecodeMmaValidRows) {
        float* destination = partial_output + partial_group_offset +
            (static_cast<size_t>(tid) * splits + split) * partial_stride;
        destination[0] = running_max[tid];
        destination[1] = running_sum[tid];
    }
    if (lrow0 < kDecodeMmaValidRows) {
        float* destination = partial_output + partial_group_offset +
            (static_cast<size_t>(lrow0) * splits + split) * partial_stride + 2;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int d0 = warp * 64 + j * 8 + tig * 2;
            *reinterpret_cast<float2*>(destination + d0) =
                make_float2(accumulator[j][0], accumulator[j][1]);
        }
    }
}

// Speculative verification scans one context split per CTA. A CTA owns one
// KV head, one group of three Q heads, and every query row, so each K/V element
// is loaded once for up to rows * 3 outputs. Splitting the history restores
// enough parallelism for the 4-32K context regime where a single CTA per group
// leaves SM75 mostly idle.
template <int kRows, int kHPG>
__global__ void gqa_verify_split_f16_kernel(
    const uint16_t* __restrict__ q_rows,
    const uint16_t* __restrict__ k_cache,
    const uint16_t* __restrict__ v_cache,
    float* __restrict__ partial_output,
    int rows,
    int q_heads,
    int kv_heads,
    int head_dim,
    int position_offset,
    int splits,
    int positions_per_split) {
    constexpr int kCombos = kRows * kHPG;
    const int q_per_kv = q_heads / kv_heads;
    const int groups_per_kv =
        (q_per_kv + kHPG - 1) / kHPG;
    const int grouped = static_cast<int>(blockIdx.x) / splits;
    const int split = static_cast<int>(blockIdx.x) % splits;
    const int kv_head = grouped / groups_per_kv;
    const int group = grouped % groups_per_kv;
    const int first_head = kv_head * q_per_kv + group * kHPG;
    const int split_start = split * positions_per_split;
    const int split_end = min(
        position_offset + rows, split_start + positions_per_split);
    if (kv_head >= kv_heads || split_start >= split_end) return;

    __shared__ float warp_sums[kCombos][kWarps];
    __shared__ float scores[kCombos];
    __shared__ float running_max[kCombos];
    __shared__ float running_sum[kCombos];
    __shared__ float rescale[kCombos];
    __shared__ float probability[kCombos];

    const int tid = static_cast<int>(threadIdx.x);
    float query[kCombos][kValuesPerThread];
    float accumulator[kCombos][kValuesPerThread];
    bool active[kCombos];
    int context_limit[kCombos];
#pragma unroll
    for (int combo = 0; combo < kCombos; ++combo) {
        const int row = combo / kHPG;
        const int head_in_group = combo % kHPG;
        const int head = first_head + head_in_group;
        active[combo] = row < rows &&
            head < (kv_head + 1) * q_per_kv && head < q_heads;
        context_limit[combo] = position_offset + row + 1;
#pragma unroll
        for (int i = 0; i < kValuesPerThread; ++i) {
            const int d = tid + i * kThreads;
            query[combo][i] = active[combo] && d < head_dim
                ? half_to_float(q_rows[
                      (static_cast<size_t>(row) * q_heads + head) * head_dim + d])
                : 0.0f;
            accumulator[combo][i] = 0.0f;
        }
        if (tid == combo) {
            running_max[combo] = -INFINITY;
            running_sum[combo] = 0.0f;
        }
    }
    __syncthreads();

    const float attention_scale = rsqrtf(static_cast<float>(head_dim));
    const size_t kv_stride = static_cast<size_t>(kv_heads) * head_dim;
    for (int position = split_start; position < split_end; ++position) {
        const size_t kv_base = static_cast<size_t>(position) * kv_stride +
            static_cast<size_t>(kv_head) * head_dim;
        float key[kValuesPerThread];
        float value[kValuesPerThread];
#pragma unroll
        for (int i = 0; i < kValuesPerThread; ++i) {
            const int d = tid + i * kThreads;
            key[i] = d < head_dim ? half_to_float(k_cache[kv_base + d]) : 0.0f;
            value[i] = d < head_dim ? half_to_float(v_cache[kv_base + d]) : 0.0f;
        }
        float dot[kCombos] = {};
#pragma unroll
        for (int combo = 0; combo < kCombos; ++combo) {
            if (active[combo] && position < context_limit[combo]) {
#pragma unroll
                for (int i = 0; i < kValuesPerThread; ++i) {
                    dot[combo] += query[combo][i] * key[i];
                }
            }
        }
        reduce_dot_products(dot, warp_sums, scores, attention_scale);
        if (tid < kCombos) {
            if (active[tid] && position < context_limit[tid]) {
                const float old_max = running_max[tid];
                const float new_max = fmaxf(old_max, scores[tid]);
                const float old_scale = old_max == -INFINITY
                    ? 0.0f : expf(old_max - new_max);
                const float weight = expf(scores[tid] - new_max);
                running_max[tid] = new_max;
                running_sum[tid] = running_sum[tid] * old_scale + weight;
                rescale[tid] = old_scale;
                probability[tid] = weight;
            } else {
                rescale[tid] = 1.0f;
                probability[tid] = 0.0f;
            }
        }
        __syncthreads();
#pragma unroll
        for (int combo = 0; combo < kCombos; ++combo) {
#pragma unroll
            for (int i = 0; i < kValuesPerThread; ++i) {
                accumulator[combo][i] = accumulator[combo][i] * rescale[combo] +
                    probability[combo] * value[i];
            }
        }
        __syncthreads();
    }

    const int partial_stride = head_dim + 2;
#pragma unroll
    for (int combo = 0; combo < kCombos; ++combo) {
        if (!active[combo]) continue;
        const int row = combo / kHPG;
        const int head_in_group = combo % kHPG;
        const int head = first_head + head_in_group;
        float* destination = partial_output +
            ((static_cast<size_t>(row) * q_heads + head) * splits + split) *
                partial_stride;
        if (tid == 0) {
            destination[0] = running_max[combo];
            destination[1] = running_sum[combo];
        }
#pragma unroll
        for (int i = 0; i < kValuesPerThread; ++i) {
            const int d = tid + i * kThreads;
            if (d < head_dim) destination[2 + d] = accumulator[combo][i];
        }
    }
}

__global__ void gqa_verify_merge_f16_kernel(
    const float* __restrict__ partial_output,
    uint16_t* __restrict__ output,
    int rows,
    int q_heads,
    int head_dim,
    int splits) {
    const int row = static_cast<int>(blockIdx.y);
    const int head = static_cast<int>(blockIdx.x);
    if (row >= rows || head >= q_heads) return;
    __shared__ float weights[kVerifyMaxSplits];
    if (threadIdx.x == 0) {
        const int stride = head_dim + 2;
        const size_t base =
            (static_cast<size_t>(row) * q_heads + head) * splits * stride;
        float maximum = -INFINITY;
        for (int split = 0; split < splits; ++split) {
            maximum = fmaxf(maximum,
                partial_output[base + static_cast<size_t>(split) * stride]);
        }
        float denominator = 0.0f;
        for (int split = 0; split < splits; ++split) {
            const float* source = partial_output + base +
                static_cast<size_t>(split) * stride;
            weights[split] = expf(source[0] - maximum);
            denominator += weights[split] * source[1];
        }
        const float inverse = denominator > 0.0f ? 1.0f / denominator : 0.0f;
        for (int split = 0; split < splits; ++split) weights[split] *= inverse;
    }
    __syncthreads();
    const int stride = head_dim + 2;
    const size_t base =
        (static_cast<size_t>(row) * q_heads + head) * splits * stride;
    for (int d = static_cast<int>(threadIdx.x); d < head_dim; d += kThreads) {
        float value = 0.0f;
        for (int split = 0; split < splits; ++split) {
            const float* source = partial_output + base +
                static_cast<size_t>(split) * stride;
            value += weights[split] * source[2 + d];
        }
        output[(static_cast<size_t>(row) * q_heads + head) * head_dim + d] =
            float_to_half(value);
    }
}

template <int kRows>
__global__ void gqa_verify_scores_exact_f16_kernel(
    const uint16_t* __restrict__ q_rows,
    const uint16_t* __restrict__ k_cache,
    float* __restrict__ scores,
    int rows,
    int q_heads,
    int kv_heads,
    int head_dim,
    int position_offset,
    int context_len) {
    constexpr int kCombos = kRows * kHeadsPerGroup;
    const int q_per_kv = q_heads / kv_heads;
    const int groups_per_kv =
        (q_per_kv + kHeadsPerGroup - 1) / kHeadsPerGroup;
    const int grouped = static_cast<int>(blockIdx.x) / context_len;
    const int position = static_cast<int>(blockIdx.x) % context_len;
    const int kv_head = grouped / groups_per_kv;
    const int group = grouped % groups_per_kv;
    const int first_head = kv_head * q_per_kv + group * kHeadsPerGroup;
    if (kv_head >= kv_heads) return;

    __shared__ float warp_sums[kCombos][kWarps];
    __shared__ float reduced[kCombos];
    const int tid = static_cast<int>(threadIdx.x);
    const size_t kv_base =
        (static_cast<size_t>(position) * kv_heads + kv_head) * head_dim;
    float key[kValuesPerThread];
#pragma unroll
    for (int i = 0; i < kValuesPerThread; ++i) {
        const int d = tid + i * kThreads;
        key[i] = d < head_dim ? half_to_float(k_cache[kv_base + d]) : 0.0f;
    }
    float partial[kCombos] = {};
#pragma unroll
    for (int combo = 0; combo < kCombos; ++combo) {
        const int row = combo / kHeadsPerGroup;
        const int head = first_head + combo % kHeadsPerGroup;
        if (row >= rows || head >= (kv_head + 1) * q_per_kv ||
            head >= q_heads || position >= position_offset + row + 1) continue;
        const uint16_t* query = q_rows +
            (static_cast<size_t>(row) * q_heads + head) * head_dim;
#pragma unroll
        for (int i = 0; i < kValuesPerThread; ++i) {
            const int d = tid + i * kThreads;
            if (d < head_dim) partial[combo] += half_to_float(query[d]) * key[i];
        }
    }
    reduce_dot_products(partial, warp_sums, reduced,
                        rsqrtf(static_cast<float>(head_dim)));
    if (tid < kCombos) {
        const int row = tid / kHeadsPerGroup;
        const int head = first_head + tid % kHeadsPerGroup;
        if (row < rows && head < (kv_head + 1) * q_per_kv && head < q_heads) {
            scores[(static_cast<size_t>(row) * q_heads + head) * context_len +
                   position] = position < position_offset + row + 1
                ? reduced[tid] : -INFINITY;
        }
    }
}

__global__ void gqa_verify_softmax_exact_f16_kernel(
    float* __restrict__ scores, int rows, int q_heads, int context_len,
    int position_offset) {
    const int row = static_cast<int>(blockIdx.y);
    const int head = static_cast<int>(blockIdx.x);
    if (row >= rows || head >= q_heads) return;
    float* line = scores +
        (static_cast<size_t>(row) * q_heads + head) * context_len;
    const int context_limit = position_offset + row + 1;
    __shared__ float reduce[kThreads];
    float local_max = -INFINITY;
    for (int position = static_cast<int>(threadIdx.x); position < context_len;
         position += kThreads) {
        const float value = position < context_limit ? line[position] : -INFINITY;
        local_max = fmaxf(local_max, value);
    }
    reduce[threadIdx.x] = local_max;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            reduce[threadIdx.x] =
                fmaxf(reduce[threadIdx.x], reduce[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const float maximum = reduce[0];
    float sum = 0.0f;
    for (int position = static_cast<int>(threadIdx.x); position < context_len;
         position += kThreads) {
        line[position] = position < context_limit
            ? expf(line[position] - maximum) : 0.0f;
        sum += line[position];
    }
    reduce[threadIdx.x] = sum;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) reduce[threadIdx.x] += reduce[threadIdx.x + stride];
        __syncthreads();
    }
    const float inverse = reduce[0] > 0.0f ? 1.0f / reduce[0] : 0.0f;
    for (int position = static_cast<int>(threadIdx.x); position < context_len;
         position += kThreads) {
        line[position] *= inverse;
    }
}

// A warp owns one candidate row, three Q heads sharing a KV head, and a
// contiguous 32-channel value tile. Splitting rows and channels into separate
// CTAs preserves each output element's left-to-right FP32 accumulation order,
// while exposing enough independent work to occupy all SMs. The previous kernel
// kept every candidate row in one CTA; for Qwen TP4 that launched only four
// long-running CTAs per layer at head_dim=256.
__global__ void gqa_verify_values_exact_f16_kernel(
    const float* __restrict__ probabilities,
    const uint16_t* __restrict__ v_cache,
    uint16_t* __restrict__ output,
    int rows,
    int q_heads,
    int kv_heads,
    int head_dim,
    int context_len) {
    const int q_per_kv = q_heads / kv_heads;
    const int groups_per_kv =
        (q_per_kv + kHeadsPerGroup - 1) / kHeadsPerGroup;
    const int kv_head = static_cast<int>(blockIdx.x) / groups_per_kv;
    const int group = static_cast<int>(blockIdx.x) % groups_per_kv;
    const int row = static_cast<int>(blockIdx.y);
    const int first_head = kv_head * q_per_kv + group * kHeadsPerGroup;
    const int d = static_cast<int>(blockIdx.z) * 32 + threadIdx.x;
    if (kv_head >= kv_heads || row >= rows || d >= head_dim) return;

    float accumulators[kHeadsPerGroup] = {};
    for (int position = 0; position < context_len; ++position) {
        const float value = half_to_float(v_cache[
            (static_cast<size_t>(position) * kv_heads + kv_head) * head_dim + d]);
#pragma unroll
        for (int i = 0; i < kHeadsPerGroup; ++i) {
            const int head = first_head + i;
            if (head < (kv_head + 1) * q_per_kv && head < q_heads) {
                accumulators[i] += probabilities[
                    (static_cast<size_t>(row) * q_heads + head) * context_len +
                    position] * value;
            }
        }
    }
#pragma unroll
    for (int i = 0; i < kHeadsPerGroup; ++i) {
        const int head = first_head + i;
        if (head < (kv_head + 1) * q_per_kv && head < q_heads) {
            output[(static_cast<size_t>(row) * q_heads + head) * head_dim + d] =
                float_to_half(accumulators[i]);
        }
    }
}

__global__ void gqa_decode_merge_f16_kernel(
    const float* __restrict__ partial_output,
    uint16_t* __restrict__ output,
    int q_heads,
    int head_dim,
    int splits) {
    const int head = static_cast<int>(blockIdx.x);
    if (head >= q_heads) return;
    // blockIdx.y indexes the sequence when batched decode launches this; the
    // single-row grid leaves it at zero, so the offsets below vanish.
    {
        const size_t row = static_cast<size_t>(blockIdx.y);
        partial_output += row * q_heads * splits * (head_dim + 2);
        output += row * q_heads * head_dim;
    }
    __shared__ float weights[kDecodeSplitCeiling];
    __shared__ float shared_maximum;
    const int stride = head_dim + 2;
    // The running maxima are strided head_dim + 2 apart, so one thread walking
    // them serializes up to 1024 dependent loads before any output work can
    // start. fmaxf is associative and commutative, so spreading the scan over
    // the block and reducing in shared memory finds the same maximum.
    {
        float local = -INFINITY;
        for (int split = static_cast<int>(threadIdx.x); split < splits;
             split += kThreads) {
            local = fmaxf(local, partial_output[
                (static_cast<size_t>(head) * splits + split) * stride]);
        }
        local = warp_sum_max(local);
        __shared__ float warp_maxima[kWarps];
        const int lane = static_cast<int>(threadIdx.x) & 31;
        const int warp = static_cast<int>(threadIdx.x) >> 5;
        if (lane == 0) warp_maxima[warp] = local;
        __syncthreads();
        if (threadIdx.x == 0) {
            float maximum = warp_maxima[0];
#pragma unroll
            for (int w = 1; w < kWarps; ++w) {
                maximum = fmaxf(maximum, warp_maxima[w]);
            }
            shared_maximum = maximum;
        }
        __syncthreads();
    }
    // The weights themselves are independent per split, so only the denominator
    // needs the original left-to-right order; thread 0 sums the already-computed
    // weights instead of also paying for the expf chain and the strided loads.
    for (int split = static_cast<int>(threadIdx.x); split < splits;
         split += kThreads) {
        const float* source = partial_output +
            (static_cast<size_t>(head) * splits + split) * stride;
        weights[split] = expf(source[0] - shared_maximum);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float denominator = 0.0f;
        for (int split = 0; split < splits; ++split) {
            denominator += weights[split] * partial_output[
                (static_cast<size_t>(head) * splits + split) * stride + 1];
        }
        shared_maximum = denominator > 0.0f ? 1.0f / denominator : 0.0f;
    }
    __syncthreads();
    const float inverse = shared_maximum;
    for (int split = static_cast<int>(threadIdx.x); split < splits;
         split += kThreads) {
        weights[split] *= inverse;
    }
    __syncthreads();
    for (int d = static_cast<int>(threadIdx.x); d < head_dim; d += kThreads) {
        float value = 0.0f;
        for (int split = 0; split < splits; ++split) {
            const float* source = partial_output +
                (static_cast<size_t>(head) * splits + split) * stride;
            value += weights[split] * source[2 + d];
        }
        output[static_cast<size_t>(head) * head_dim + d] = float_to_half(value);
    }
}

bool valid_shape(int q_heads, int kv_heads, int head_dim) {
    return q_heads > 0 && kv_heads > 0 && q_heads % kv_heads == 0 &&
        head_dim > 0 && head_dim <= kMaxHeadDim;
}

// Never splits finer than one position per split, so short contexts fall back to
// however many splits they can actually fill. The automatic tensor-core target
// is expressed as a total grid size: one CTA covers one KV head, so divide that
// target across KV heads instead of launching an extra full wave for every head.
// The MMA-specific environment override and the explicit test override remain
// per KV head for reproducible A/B sweeps.
int decode_split_count_for_variant(int attended_positions, bool mma,
                                    int split_count_override, int kv_heads = 1) {
    if (attended_positions <= 0) return 1;
    const int forced = decode_forced_positions();
    const int total_target = decode_target_splits(attended_positions, mma);
    const int mma_override = mma ? decode_mma_target_splits() : 0;
    const int shape_target = mma && mma_override <= 0
        ? (total_target + std::max(kv_heads, 1) - 1) / std::max(kv_heads, 1)
        : total_target;
    const int requested = split_count_override > 0
        ? split_count_override
        : forced > 0
            ? (attended_positions + forced - 1) / forced
            : std::min(shape_target, attended_positions);
    const int splits = std::max(std::min(decode_max_splits(), requested), 1);
    // Correctness tests may deliberately preserve an oversized split count to
    // exercise the neutral partial written by an empty CTA. Production geometry
    // re-derives the count below so dead trailing entries are not launched.
    if (split_count_override > 0) return splits;
    // Rounding the slice up can leave trailing splits with nothing to scan (8193
    // positions over 256 splits is 33 each, which only fills 249). Re-deriving
    // the count from the slice length drops those, so the grid and the merge do
    // not walk dead entries. The engine sizes the partial scratch from this same
    // function, so the two cannot disagree.
    const int positions = (attended_positions + splits - 1) / splits;
    return std::max((attended_positions + positions - 1) / positions, 1);
}

int decode_split_count(int attended_positions, int kv_heads,
                       bool tensor_core_shape) {
    const bool mma = tensor_core_shape && decode_mma_enabled(attended_positions);
    return decode_split_count_for_variant(
        attended_positions, mma, 0, kv_heads);
}

struct GqaCublasWorkspace {
    int device = -1;
    cublasHandle_t handle = nullptr;
    ~GqaCublasWorkspace() {
        if (handle != nullptr) cublasDestroy(handle);
    }
};

GqaCublasWorkspace& gqa_cublas_workspace() {
    static GqaCublasWorkspace workspace;
    return workspace;
}

bool ensure_gqa_cublas_workspace(GqaCublasWorkspace& workspace) {
    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess) return false;
    if (workspace.handle != nullptr && workspace.device == device) return true;
    if (workspace.handle != nullptr) {
        cublasDestroy(workspace.handle);
        workspace.handle = nullptr;
    }
    if (cublasCreate(&workspace.handle) != CUBLAS_STATUS_SUCCESS) return false;
    (void)cublasSetMathMode(workspace.handle, CUBLAS_TENSOR_OP_MATH);
    workspace.device = device;
    return true;
}

}  // namespace

bool qwen_gqa_prefill_attention_f16_tiled_cuda(
    const uint16_t* q, const uint16_t* k_cache,
    const uint16_t* v_cache, uint16_t* output, int seq_len,
    int q_heads, int kv_heads, int head_dim, int position_offset,
    int max_context, int attention_window, int sink_tokens, void* stream) {
    if (!q || !k_cache || !v_cache || !output || seq_len <= 0 ||
        position_offset < 0 || position_offset + seq_len > max_context ||
        attention_window < 0 || sink_tokens < 0 ||
        !valid_shape(q_heads, kv_heads, head_dim)) return false;
    const int groups_per_kv =
        (q_heads / kv_heads + kHeadsPerGroup - 1) / kHeadsPerGroup;
    const dim3 grid(static_cast<unsigned>(kv_heads * groups_per_kv),
                    static_cast<unsigned>((seq_len + kQueryRows - 1) / kQueryRows), 1);
    // The batched-position kernel amortises the per-history-element barriers
    // over kPosTile positions while preserving the online-softmax order, so it
    // is bit-identical. Set POCKETLLM_QWEN_GQA_POS_TILE=0 to fall back.
    const char* pos_tile = std::getenv("POCKETLLM_QWEN_GQA_POS_TILE");
    const bool use_pos_tile = pos_tile == nullptr ||
        std::strcmp(pos_tile, "0") != 0;
    if (use_pos_tile) {
        // Choose the head grouping that divides q_per_kv exactly. With the fixed
        // group of three and the real q_per_kv of four, the second group carried a
        // single head yet still streamed the whole KV history for its KV head, so
        // each history element was read twice per layer. A group of four covers
        // all four heads in one CTA and halves that traffic. Shapes that do not
        // divide evenly keep the original grouping.
        const int q_per_kv = q_heads / kv_heads;
        int hpg = kHeadsPerGroup;
        // Wider groups cut history traffic but cost registers per CTA. Measured at
        // the real TP4 shape (6 q heads over 1 kv head, head_dim 256, 4096 rows),
        // groups of six spill and lose: 67.7/215.4 ms against 54.9/186.8 for two.
        if (q_per_kv % 4 == 0) hpg = 4;
        else if (q_per_kv % 2 == 0) hpg = 2;
        else if (q_per_kv == 1) hpg = 1;
        if (const char* hpg_env = std::getenv("POCKETLLM_QWEN_GQA_HEADS_PER_GROUP")) {
            const int requested = std::atoi(hpg_env);
            if (requested >= 1 && requested <= 6 && q_per_kv % requested == 0) {
                hpg = requested;
            }
        }
        const int batched_groups = (q_per_kv + hpg - 1) / hpg;
        // Every CTA streams the whole KV history for its KV head, so the number of
        // query-row tiles is a direct multiplier on history traffic. At 8192 with
        // two rows per CTA the 16 GQA layers re-read 544 GiB per rank and run at
        // ~118 GB/s of the ~600 GB/s the card can sustain. Widening the tile
        // divides that traffic; the accumulator lives in registers, so the width
        // is capped to keep occupancy.
        int qr = 2;
        const char* qr_env = std::getenv("POCKETLLM_QWEN_GQA_QUERY_ROWS");
        const bool query_rows_explicit = qr_env != nullptr;
        if (qr_env != nullptr) {
            const int requested = std::atoi(qr_env);
            if (requested == 2 || requested == 4 || requested == 8) qr = requested;
        } else if (hpg <= 2) {
            // Measured on the real 8192 TP4 prefill: full_attention 4.96 s at two
            // rows, 4.51 s at four, 6.60 s at eight. Past four rows the register
            // accumulator spills and the traffic saving is more than undone. At
            // head_dim 256 the accumulator is twice as wide, so four rows only pay
            // off for the narrow groups: 53.7/178.5 ms against 54.9/186.8 at two
            // rows for hpg=2, while hpg=3 regresses to 69.7/214.3.
            qr = 4;
        }
        const dim3 batched_grid(
            static_cast<unsigned>(kv_heads * batched_groups),
            static_cast<unsigned>((seq_len + qr - 1) / qr), 1);
        // One warp per Q head avoids re-reading the single TP4 KV head for each
        // hpg=2 group. It uses a different dot-product reassociation, so the
        // numerical and real-model token-parity gates are kept separate below.
        const int total_context = position_offset + seq_len;
        const char* long_tile = std::getenv("POCKETLLM_QWEN_GQA_LONG_TILE");
        // The TP4 Q=6/KV=1 path is the measured production candidate for exact
        // long-context prefill. Keep an explicit `=0` escape hatch for A/B and
        // fallback debugging; an unset variable selects the faster path.
        const bool long_tile_enabled =
            long_tile == nullptr || std::strcmp(long_tile, "0") != 0;
        const bool tp4_long_shape = attention_window <= 0 && q_heads == 6 &&
            kv_heads == 1 && head_dim == 256 && seq_len >= 128 &&
            total_context >= 2048;
        // Tensor-core exact path, and the production default for the shapes it
        // supports. The scalar kernels are scalar-issue bound at roughly 8.5% of
        // the FP32 CUDA-core peak, so moving both GEMMs onto the tensor cores is
        // the only lever that moves exact long-context prefill. Measured on the
        // real 64-layer TP4 checkpoint, strictly serial, one binary:
        //
        //   context   scalar default   mma      prefill
        //     8192     824.3 tok/s   1125.7    1.37x
        //    32768     442.7 tok/s   1058.9    2.39x
        //    65536     258.0 tok/s    887.6    3.44x
        //
        // decode is unchanged (25.15/19.90/19.25 against 25.12/20.78/19.54) and
        // the generated tokens are identical at all three lengths. It reassociates
        // both dot products and rounds the softmax numerator to half for the MMA
        // operand, so `=0` selects the scalar path for A/B and for any future
        // divergence hunt.
        const char* mma_tile = std::getenv("POCKETLLM_QWEN_GQA_MMA_TILE");
        const bool mma_enabled =
            mma_tile == nullptr || std::strcmp(mma_tile, "0") != 0;
        // Windowed and sink attention stay on the scalar kernels: this kernel
        // masks by causal limit only, so an approximate window would silently
        // become exact rather than fail.
        if (mma_enabled && attention_window <= 0 && sink_tokens == 0 &&
            head_dim == kMmaDim && seq_len >= kMmaQRows) {
            const dim3 mma_grid(
                static_cast<unsigned>(q_heads),
                static_cast<unsigned>((seq_len + kMmaQRows - 1) / kMmaQRows), 1);
            // The non-aliased kernel is the production default. After widening
            // the query tile to 64 rows, both variants are limited to one CTA/SM:
            // 512 threads already consume the full 1024-thread SM75 block budget,
            // so aliasing K/V staging no longer buys occupancy. It still pays an
            // extra barrier and a second address/load pass per KV tile. Serial
            // same-binary real TP4 65K A/B measured 1400.67/1396.26 tok/s for the
            // non-aliased path against 1368.73/1352.86 for aliasing, with rank
            // parity PASS. Set POCKETLLM_QWEN_GQA_MMA_OCC=1 to restore the alias path
            // for controlled A/B.
            // Pairing two Q heads of one KV head into a CTA was tried and
            // rejected: it halves the KV traffic on paper, but two sets of Q
            // fragments and P*V accumulators need ~192 registers per thread, so
            // ptxas spills 400 bytes and the kernel runs 3x slower (249.0 ms
            // against 83.6 at 4096 rows, offset 61440) despite bit-identical
            // output. Sharing a tile across heads needs the accumulators moved to
            // shared memory first.
            const char* mma_occ = std::getenv("POCKETLLM_QWEN_GQA_MMA_OCC");
            if (mma_occ != nullptr && std::strcmp(mma_occ, "0") != 0) {
                gqa_prefill_mma_occ_f16_kernel<<<
                    mma_grid, kMmaWarps * 32, 0,
                    static_cast<cudaStream_t>(stream)>>>(
                    q, k_cache, v_cache, output, seq_len, q_heads, kv_heads,
                    head_dim, position_offset);
                return cudaGetLastError() == cudaSuccess;
            }
            gqa_prefill_mma_f16_kernel<<<
                mma_grid, kMmaWarps * 32, 0,
                static_cast<cudaStream_t>(stream)>>>(
                q, k_cache, v_cache, output, seq_len, q_heads, kv_heads,
                head_dim, position_offset);
            return cudaGetLastError() == cudaSuccess;
        }
        // Two-phase flash tile: one warp reduction per 32 positions instead of
        // one per position. It remains opt-in because it is not faster than the
        // hpg6 path on the measured TP4 long-context cases.
        const char* flash_tile = std::getenv("POCKETLLM_QWEN_GQA_FLASH_TILE");
        if (flash_tile != nullptr && std::strcmp(flash_tile, "0") != 0 &&
            tp4_long_shape) {
            const int flash_qr = qr >= 4 ? 4 : 2;
            const dim3 flash_grid(
                static_cast<unsigned>(kv_heads),
                static_cast<unsigned>((seq_len + flash_qr - 1) / flash_qr), 1);
            if (flash_qr == 2) {
                gqa_prefill_flash_tile_f16_kernel<8, 2><<<
                    flash_grid, 6 * 32, 0, static_cast<cudaStream_t>(stream)>>>(
                    q, k_cache, v_cache, output, seq_len, q_heads, kv_heads,
                    head_dim, position_offset);
            } else {
                gqa_prefill_flash_tile_f16_kernel<8, 4><<<
                    flash_grid, 6 * 32, 0, static_cast<cudaStream_t>(stream)>>>(
                    q, k_cache, v_cache, output, seq_len, q_heads, kv_heads,
                    head_dim, position_offset);
            }
            return cudaGetLastError() == cudaSuccess;
        }
        if (long_tile_enabled && tp4_long_shape) {
            // The hpg6 kernel has a six-warp register footprint. On SM75, its
            // four-row specialization is the production default; a caller that
            // explicitly requests two rows is allowed for controlled A/B tests.
            // Do not silently reinterpret an explicit eight-row request as four:
            // the generic batched path is the honest fallback for that shape.
            if (query_rows_explicit && qr == 8) {
                // Fall through to the generic position-tiled dispatch below.
            } else {
                const int long_qr = qr == 8 ? 4 : qr;
                const dim3 long_grid(
                    static_cast<unsigned>(kv_heads),
                    static_cast<unsigned>((seq_len + long_qr - 1) / long_qr), 1);
                if (long_qr == 2) {
                    gqa_prefill_hpg6_f16_kernel<8, 2><<<
                        long_grid, 6 * 32, 0, static_cast<cudaStream_t>(stream)>>>(
                        q, k_cache, v_cache, output, seq_len, q_heads, kv_heads,
                        head_dim, position_offset, attention_window, sink_tokens);
                } else {
                    gqa_prefill_hpg6_f16_kernel<8, 4><<<
                        long_grid, 6 * 32, 0, static_cast<cudaStream_t>(stream)>>>(
                        q, k_cache, v_cache, output, seq_len, q_heads, kv_heads,
                        head_dim, position_offset, attention_window, sink_tokens);
                }
                return cudaGetLastError() == cudaSuccess;
            }
        }

        // Warp-per-combo variant. Opt-in: it reassociates the dot-product
        // reduction, so the last FP32 bits differ from the per-position kernel and
        // a near-tie greedy argmax could flip. Requires head_dim to split evenly
        // across a warp and the combo count to cover the warps.
        const char* warp_combo = std::getenv("POCKETLLM_QWEN_GQA_WARP_COMBO");
        if (warp_combo != nullptr && std::strcmp(warp_combo, "0") != 0 &&
            (head_dim == 128 || head_dim == 256) && attention_window <= 0) {
            const int combos = hpg * qr;
            const int dpl = head_dim / 32;
            if (combos % kWarps == 0) {
#define POCKET_LAUNCH_WARP_COMBO_D(HPG, QR, DPL) \
                gqa_prefill_warp_combo_f16_kernel<HPG, QR, DPL> \
                    <<<batched_grid, kThreads, 0, \
                       static_cast<cudaStream_t>(stream)>>>( \
                        q, k_cache, v_cache, output, seq_len, q_heads, kv_heads, \
                        head_dim, position_offset, attention_window, sink_tokens)
#define POCKET_LAUNCH_WARP_COMBO(HPG, QR) \
                do { \
                    if (dpl == 8) { POCKET_LAUNCH_WARP_COMBO_D(HPG, QR, 8); } \
                    else { POCKET_LAUNCH_WARP_COMBO_D(HPG, QR, 4); } \
                } while (0)
                bool launched = true;
                if (hpg == 4 && qr == 4) { POCKET_LAUNCH_WARP_COMBO(4, 4); }
                else if (hpg == 4 && qr == 2) { POCKET_LAUNCH_WARP_COMBO(4, 2); }
                else if (hpg == 4 && qr == 8) { POCKET_LAUNCH_WARP_COMBO(4, 8); }
                else if (hpg == 2 && qr == 2) { POCKET_LAUNCH_WARP_COMBO(2, 2); }
                else if (hpg == 2 && qr == 4) { POCKET_LAUNCH_WARP_COMBO(2, 4); }
                else if (hpg == 2 && qr == 8) { POCKET_LAUNCH_WARP_COMBO(2, 8); }
                else if (hpg == 1 && qr == 4) { POCKET_LAUNCH_WARP_COMBO(1, 4); }
                else if (hpg == 1 && qr == 8) { POCKET_LAUNCH_WARP_COMBO(1, 8); }
                else { launched = false; }
#undef POCKET_LAUNCH_WARP_COMBO
#undef POCKET_LAUNCH_WARP_COMBO_D
                if (launched) return cudaGetLastError() == cudaSuccess;
            }
        }
        // kValuesPerThread is sized for the largest supported head_dim. The real
        // GQA head_dim is 128, which leaves half of every register array and half
        // of each inner loop doing nothing, so specialise on the exact count.
        const int vpt = (head_dim + kThreads - 1) / kThreads;
#define POCKET_LAUNCH_POS_TILE(HPG, VPT, QR) \
        gqa_prefill_tiled_batched_f16_kernel<HPG, VPT, QR> \
            <<<batched_grid, kThreads, 0, static_cast<cudaStream_t>(stream)>>>( \
                q, k_cache, v_cache, output, seq_len, q_heads, kv_heads, \
                head_dim, position_offset, attention_window, sink_tokens)
#define POCKET_LAUNCH_POS_TILE_QR(HPG, VPT) \
        do { \
            if (qr == 8) { POCKET_LAUNCH_POS_TILE(HPG, VPT, 8); } \
            else if (qr == 4) { POCKET_LAUNCH_POS_TILE(HPG, VPT, 4); } \
            else { POCKET_LAUNCH_POS_TILE(HPG, VPT, 2); } \
        } while (0)
#define POCKET_LAUNCH_POS_TILE_HPG(HPG) \
        do { \
            if (vpt == 1) { POCKET_LAUNCH_POS_TILE_QR(HPG, 1); } \
            else { POCKET_LAUNCH_POS_TILE_QR(HPG, kValuesPerThread); } \
        } while (0)
        if (hpg == 6) { POCKET_LAUNCH_POS_TILE_HPG(6); }
        else if (hpg == 4) { POCKET_LAUNCH_POS_TILE_HPG(4); }
        else if (hpg == 2) { POCKET_LAUNCH_POS_TILE_HPG(2); }
        else if (hpg == 1) { POCKET_LAUNCH_POS_TILE_HPG(1); }
        else { POCKET_LAUNCH_POS_TILE_HPG(3); }
#undef POCKET_LAUNCH_POS_TILE_HPG
#undef POCKET_LAUNCH_POS_TILE_QR
#undef POCKET_LAUNCH_POS_TILE
        return cudaGetLastError() == cudaSuccess;
    }
    gqa_prefill_tiled_f16_kernel<<<grid, kThreads, 0,
        static_cast<cudaStream_t>(stream)>>>(q, k_cache, v_cache, output,
        seq_len, q_heads, kv_heads, head_dim, position_offset,
        attention_window, sink_tokens);
    return cudaGetLastError() == cudaSuccess;
}

bool qwen_gqa_verify_attention_f16_cublas_qk_cuda(
    const uint16_t* q, const uint16_t* k_cache,
    const uint16_t* v_cache, uint16_t* output, float* score_scratch,
    int rows, int q_heads, int kv_heads, int head_dim,
    int position_offset, int max_context, void* stream) {
    if (!q || !k_cache || !v_cache || !output || !score_scratch ||
        rows < 2 || rows > kVerifyMaxRows || kv_heads != 1 ||
        position_offset < 0 || position_offset + rows > max_context ||
        !valid_shape(q_heads, kv_heads, head_dim)) return false;
    const int context_len = position_offset + rows;
    GqaCublasWorkspace& workspace = gqa_cublas_workspace();
    if (!ensure_gqa_cublas_workspace(workspace)) return false;
    const cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    if (cublasSetStream(workspace.handle, cuda_stream) != CUBLAS_STATUS_SUCCESS) {
        return false;
    }

    // Row-major Q [M,K] times row-major K_cache^T [K,N]. cuBLAS is column
    // major, so compute C^T [N,M] = K_cache [N,K] * Q^T [K,M]. C's row-major
    // storage is exactly C^T's column-major storage.
    const int m = rows * q_heads;
    const int n = context_len;
    const int k = head_dim;
    const float alpha = rsqrtf(static_cast<float>(head_dim));
    const float beta = 0.0f;
    if (cublasGemmEx(
            workspace.handle, CUBLAS_OP_T, CUBLAS_OP_N,
            n, m, k, &alpha,
            k_cache, CUDA_R_16F, k,
            q, CUDA_R_16F, k,
            &beta, score_scratch, CUDA_R_32F, n,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP) !=
        CUBLAS_STATUS_SUCCESS) {
        return false;
    }

    const dim3 softmax_grid(static_cast<unsigned>(q_heads),
                            static_cast<unsigned>(rows), 1);
    gqa_verify_softmax_exact_f16_kernel<<<
        softmax_grid, kThreads, 0, cuda_stream>>>(
        score_scratch, rows, q_heads, context_len, position_offset);
    if (cudaGetLastError() != cudaSuccess) return false;

    const int groups_per_kv =
        (q_heads / kv_heads + kHeadsPerGroup - 1) / kHeadsPerGroup;
    const dim3 value_grid(static_cast<unsigned>(kv_heads * groups_per_kv),
                          static_cast<unsigned>(rows),
                          static_cast<unsigned>((head_dim + 31) / 32));
    gqa_verify_values_exact_f16_kernel<<<value_grid, 32, 0, cuda_stream>>>(
        score_scratch, v_cache, output, rows, q_heads, kv_heads, head_dim,
        context_len);
    return cudaGetLastError() == cudaSuccess;
}

bool qwen_gqa_verify_attention_f16_exact_cuda(
    const uint16_t* q, const uint16_t* k_cache,
    const uint16_t* v_cache, uint16_t* output, float* score_scratch,
    int rows, int q_heads, int kv_heads, int head_dim,
    int position_offset, int max_context, void* stream) {
    if (!q || !k_cache || !v_cache || !output || !score_scratch ||
        rows < 2 || rows > kVerifyMaxRows || position_offset < 0 ||
        position_offset + rows > max_context ||
        !valid_shape(q_heads, kv_heads, head_dim)) return false;
    const int context_len = position_offset + rows;
    const int groups_per_kv =
        (q_heads / kv_heads + kHeadsPerGroup - 1) / kHeadsPerGroup;
    const uint64_t score_blocks =
        static_cast<uint64_t>(kv_heads) * groups_per_kv * context_len;
    if (score_blocks > static_cast<uint64_t>(UINT32_MAX)) return false;
    const cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
#define POCKET_LAUNCH_EXACT_SCORES(ROWS) \
    gqa_verify_scores_exact_f16_kernel<ROWS> \
        <<<static_cast<unsigned>(score_blocks), kThreads, 0, cuda_stream>>>( \
            q, k_cache, score_scratch, rows, q_heads, kv_heads, head_dim, \
            position_offset, context_len)
    switch (rows) {
        case 2: POCKET_LAUNCH_EXACT_SCORES(2); break;
        case 3: POCKET_LAUNCH_EXACT_SCORES(3); break;
        case 4: POCKET_LAUNCH_EXACT_SCORES(4); break;
        case 5: POCKET_LAUNCH_EXACT_SCORES(5); break;
        case 6: POCKET_LAUNCH_EXACT_SCORES(6); break;
        case 7: POCKET_LAUNCH_EXACT_SCORES(7); break;
        case 8: POCKET_LAUNCH_EXACT_SCORES(8); break;
        default: return false;
    }
#undef POCKET_LAUNCH_EXACT_SCORES
    if (cudaGetLastError() != cudaSuccess) return false;
    const dim3 softmax_grid(static_cast<unsigned>(q_heads),
                            static_cast<unsigned>(rows), 1);
    gqa_verify_softmax_exact_f16_kernel<<<softmax_grid, kThreads, 0, cuda_stream>>>(
        score_scratch, rows, q_heads, context_len, position_offset);
    if (cudaGetLastError() != cudaSuccess) return false;
    const dim3 value_grid(static_cast<unsigned>(kv_heads * groups_per_kv),
                          static_cast<unsigned>(rows),
                          static_cast<unsigned>((head_dim + 31) / 32));
    gqa_verify_values_exact_f16_kernel<<<value_grid, 32, 0, cuda_stream>>>(
        score_scratch, v_cache, output, rows, q_heads, kv_heads, head_dim,
        context_len);
    return cudaGetLastError() == cudaSuccess;
}

bool qwen_gqa_verify_attention_f16_cuda(
    const uint16_t* q, const uint16_t* k_cache,
    const uint16_t* v_cache, uint16_t* output, float* partial_scratch,
    int rows, int q_heads, int kv_heads, int head_dim,
    int position_offset, int max_context, int splits, void* stream) {
    if (!q || !k_cache || !v_cache || !output || !partial_scratch ||
        rows < 2 || rows > kVerifyMaxRows || position_offset < 0 ||
        position_offset + rows > max_context || splits <= 0 ||
        splits > kVerifyMaxSplits ||
        !valid_shape(q_heads, kv_heads, head_dim)) return false;
    const int context_len = position_offset + rows;
    const int positions_per_split = (context_len + splits - 1) / splits;
    const int q_per_kv = q_heads / kv_heads;
    const int groups_per_kv =
        (q_per_kv + kHeadsPerGroup - 1) / kHeadsPerGroup;
    const int blocks = kv_heads * groups_per_kv * splits;
    const cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
#define POCKET_LAUNCH_VERIFY(ROWS) \
    gqa_verify_split_f16_kernel<ROWS, kHeadsPerGroup> \
        <<<blocks, kThreads, 0, cuda_stream>>>( \
            q, k_cache, v_cache, partial_scratch, rows, q_heads, kv_heads, \
            head_dim, position_offset, splits, positions_per_split)
    switch (rows) {
        case 2: POCKET_LAUNCH_VERIFY(2); break;
        case 3: POCKET_LAUNCH_VERIFY(3); break;
        case 4: POCKET_LAUNCH_VERIFY(4); break;
        case 5: POCKET_LAUNCH_VERIFY(5); break;
        case 6: POCKET_LAUNCH_VERIFY(6); break;
        case 7: POCKET_LAUNCH_VERIFY(7); break;
        case 8: POCKET_LAUNCH_VERIFY(8); break;
        default: return false;
    }
#undef POCKET_LAUNCH_VERIFY
    if (cudaGetLastError() != cudaSuccess) return false;
    const dim3 merge_grid(static_cast<unsigned>(q_heads),
                          static_cast<unsigned>(rows), 1);
    gqa_verify_merge_f16_kernel<<<merge_grid, kThreads, 0, cuda_stream>>>(
        partial_scratch, output, rows, q_heads, head_dim, splits);
    return cudaGetLastError() == cudaSuccess;
}

int qwen_gqa_decode_split_count(int attended_positions, int kv_heads,
                                bool tensor_core_shape) {
    return decode_split_count(attended_positions, kv_heads, tensor_core_shape);
}

int qwen_gqa_decode_split_count_variant(
    int attended_positions, QwenGqaDecodeVariant variant,
    int split_count_override) {
    const bool mma = variant == QwenGqaDecodeVariant::Selected
        ? decode_mma_enabled(attended_positions)
        : variant == QwenGqaDecodeVariant::TensorCore;
    return decode_split_count_for_variant(attended_positions, mma,
                                          split_count_override, 1);
}

// Widest Q group the decode split kernel is instantiated for. Six covers one
// full-attention KV group in both TP4 (6/1) and TP2 (12/2), which is what removes
// the duplicate KV stream.
constexpr int kDecodeMaxHeadsPerGroup = 6;

// Zero keeps the automatic choice: cover the whole KV group so its cache line is
// read once. An override only exists for A/B sweeps.
int decode_group_width_override() {
    static const int value = [] {
        const char* env = std::getenv("POCKETLLM_QWEN_DECODE_GROUP_HEADS");
        const int parsed = env != nullptr ? std::atoi(env) : 0;
        return parsed > 0 ? parsed : 0;
    }();
    return value;
}

int decode_group_width(int q_per_kv) {
    const int override_width = decode_group_width_override();
    const int requested = override_width > 0 ? override_width : q_per_kv;
    return std::max(std::min(requested, kDecodeMaxHeadsPerGroup), 1);
}

namespace {

bool launch_decode_fused(
    const uint16_t* q, const uint16_t* k_cache,
    const uint16_t* v_cache, uint16_t* output, float* partial_scratch,
    int q_heads, int kv_heads, int head_dim, int context_len,
    int max_context, int attention_window, int sink_tokens,
    QwenGqaDecodeVariant variant, int split_count_override, void* stream) {
    if (!q || !k_cache || !v_cache || !output || !partial_scratch ||
        context_len > max_context ||
        attention_window < 0 || sink_tokens < 0 ||
        (attention_window <= 0 && context_len < kDecodeMinContext) ||
        !valid_shape(q_heads, kv_heads, head_dim)) return false;
    const SparseRanges ranges =
        sparse_ranges(context_len, attention_window, sink_tokens);
    const int attended = ranges.total();
    // The candidate has one CTA per split and embeds the six TP4 Q heads in
    // one padded MMA matrix. Keep it restricted to its supported shape and
    // refuse unsupported requests so setting the variable cannot silently
    // change other attention paths.
    const bool want_mma = variant == QwenGqaDecodeVariant::Selected
        ? decode_mma_enabled(attended)
        : variant == QwenGqaDecodeVariant::TensorCore;
    // Geometry follows the variant being launched, and the same query is public
    // so the engine's scratch sizing cannot disagree with the grid.
    const int splits = decode_split_count_for_variant(
        attended, want_mma, split_count_override, kv_heads);
    const int positions_per_split = (attended + splits - 1) / splits;
    const int group_heads = decode_group_width(q_heads / kv_heads);
    const int groups_per_kv =
        (q_heads / kv_heads + group_heads - 1) / group_heads;
    const cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    const bool mma_supported = decode_mma_device_supported() &&
        attention_window <= 0 && sink_tokens == 0 &&
        q_heads == kv_heads * kDecodeMmaValidRows &&
        head_dim == kDecodeMmaDim && splits <= kDecodeSplitCeiling;
    // An explicit tensor-core request on an unsupported shape is an error, not a
    // silent fallback: a test that asked for the candidate must not measure the
    // scalar kernel instead.
    if (variant == QwenGqaDecodeVariant::TensorCore && !mma_supported) {
        return false;
    }
    const bool use_decode_mma = want_mma && mma_supported;
    if (use_decode_mma) {
        const dim3 grid(static_cast<unsigned>(splits),
                        static_cast<unsigned>(kv_heads), 1);
        gqa_decode_split_mma_f16_kernel<<<
            grid, kDecodeMmaThreads, 0, cuda_stream>>>(
            q, k_cache, v_cache, partial_scratch, kv_heads, context_len,
            splits, positions_per_split);
        if (cudaGetLastError() != cudaSuccess) return false;
        gqa_decode_merge_f16_kernel<<<q_heads, kThreads, 0, cuda_stream>>>(
            partial_scratch, output, q_heads, head_dim, splits);
        return cudaGetLastError() == cudaSuccess;
    }
    const int blocks = kv_heads * groups_per_kv * splits;
#define POCKET_LAUNCH_DECODE_SPLIT(HPG) \
    gqa_decode_split_f16_kernel<HPG><<<blocks, kThreads, 0, cuda_stream>>>( \
        q, k_cache, v_cache, partial_scratch, q_heads, kv_heads, head_dim, \
        context_len, splits, positions_per_split, attention_window, \
        sink_tokens)
    switch (group_heads) {
        case 1: POCKET_LAUNCH_DECODE_SPLIT(1); break;
        case 2: POCKET_LAUNCH_DECODE_SPLIT(2); break;
        case 3: POCKET_LAUNCH_DECODE_SPLIT(3); break;
        case 4: POCKET_LAUNCH_DECODE_SPLIT(4); break;
        case 5: POCKET_LAUNCH_DECODE_SPLIT(5); break;
        case 6: POCKET_LAUNCH_DECODE_SPLIT(6); break;
        default: return false;
    }
#undef POCKET_LAUNCH_DECODE_SPLIT
    if (cudaGetLastError() != cudaSuccess) return false;
    gqa_decode_merge_f16_kernel<<<q_heads, kThreads, 0, cuda_stream>>>(
        partial_scratch, output, q_heads, head_dim, splits);
    return cudaGetLastError() == cudaSuccess;
}

}  // namespace

bool qwen_gqa_decode_attention_f16_fused_cuda(
    const uint16_t* q, const uint16_t* k_cache,
    const uint16_t* v_cache, uint16_t* output, float* partial_scratch,
    int q_heads, int kv_heads, int head_dim, int context_len,
    int max_context, int attention_window, int sink_tokens, void* stream) {
    return launch_decode_fused(
        q, k_cache, v_cache, output, partial_scratch, q_heads, kv_heads,
        head_dim, context_len, max_context, attention_window, sink_tokens,
        QwenGqaDecodeVariant::Selected, 0, stream);
}

int qwen_gqa_decode_batched_split_count(
    int max_context_len, int kv_heads, int attention_window, int sink_tokens) {
    const SparseRanges ranges =
        sparse_ranges(max_context_len, attention_window, sink_tokens);
    // Always the scalar geometry: the batched path does not launch the MMA
    // variant, whose CTA embeds exactly one row's six Q heads.
    return decode_split_count_for_variant(ranges.total(), false, 0, kv_heads);
}

bool qwen_gqa_decode_attention_f16_batched_cuda(
    const uint16_t* q, const uint16_t* k_cache,
    const uint16_t* v_cache, uint16_t* output, float* partial_scratch,
    const int* d_context_lens, const int* d_slot_ids, int rows,
    int max_context_len, size_t kv_slot_stride, int q_heads, int kv_heads,
    int head_dim, int max_context, int attention_window, int sink_tokens,
    const int* d_block_table, int block_size, int max_blocks_per_seq,
    void* stream) {
    if (!q || !k_cache || !v_cache || !output || !partial_scratch ||
        !d_context_lens || rows <= 0 || max_context_len <= 0 ||
        attention_window < 0 ||
        sink_tokens < 0 || !valid_shape(q_heads, kv_heads, head_dim)) {
        return false;
    }
    // A paged cache holds no per-slot reservation, so `max_context` bounds the
    // logical sequence rather than the arena and the block table supplies the
    // extent. The contiguous path keeps its arena bound.
    const bool paged = d_block_table != nullptr;
    if (paged) {
        if (block_size <= 0 || max_blocks_per_seq <= 0 ||
            max_context_len > block_size * max_blocks_per_seq) {
            return false;
        }
    } else if (max_context_len > max_context) {
        return false;
    }
    // One grid covers every sequence. The split count comes from the longest
    // row so a single launch serves the whole batch; shorter rows leave their
    // trailing splits empty, which publishes the neutral partial.
    const SparseRanges ranges =
        sparse_ranges(max_context_len, attention_window, sink_tokens);
    const int attended = ranges.total();
    const int splits = decode_split_count_for_variant(
        attended, false, 0, kv_heads);
    if (splits > kDecodeSplitCeiling) return false;
    const int positions_per_split = (attended + splits - 1) / splits;
    const int group_heads = decode_group_width(q_heads / kv_heads);
    const int groups_per_kv =
        (q_heads / kv_heads + group_heads - 1) / group_heads;
    const cudaStream_t cuda_stream = static_cast<cudaStream_t>(stream);
    const dim3 grid(static_cast<unsigned>(kv_heads * groups_per_kv * splits),
                    static_cast<unsigned>(rows), 1);
#define POCKET_LAUNCH_DECODE_SPLIT_BATCHED(HPG, PAGED) \
    gqa_decode_split_f16_kernel<HPG, true, PAGED> \
        <<<grid, kThreads, 0, cuda_stream>>>( \
        q, k_cache, v_cache, partial_scratch, q_heads, kv_heads, head_dim, \
        max_context_len, splits, positions_per_split, attention_window, \
        sink_tokens, d_context_lens, d_slot_ids, kv_slot_stride, \
        d_block_table, block_size, max_blocks_per_seq)
#define POCKET_LAUNCH_DECODE_SPLIT_BATCHED_WIDTH(PAGED) \
    switch (group_heads) { \
        case 1: POCKET_LAUNCH_DECODE_SPLIT_BATCHED(1, PAGED); break; \
        case 2: POCKET_LAUNCH_DECODE_SPLIT_BATCHED(2, PAGED); break; \
        case 3: POCKET_LAUNCH_DECODE_SPLIT_BATCHED(3, PAGED); break; \
        case 4: POCKET_LAUNCH_DECODE_SPLIT_BATCHED(4, PAGED); break; \
        case 5: POCKET_LAUNCH_DECODE_SPLIT_BATCHED(5, PAGED); break; \
        case 6: POCKET_LAUNCH_DECODE_SPLIT_BATCHED(6, PAGED); break; \
        default: return false; \
    }
    if (paged) {
        POCKET_LAUNCH_DECODE_SPLIT_BATCHED_WIDTH(true);
    } else {
        POCKET_LAUNCH_DECODE_SPLIT_BATCHED_WIDTH(false);
    }
#undef POCKET_LAUNCH_DECODE_SPLIT_BATCHED_WIDTH
#undef POCKET_LAUNCH_DECODE_SPLIT_BATCHED
    if (cudaGetLastError() != cudaSuccess) return false;
    const dim3 merge_grid(static_cast<unsigned>(q_heads),
                          static_cast<unsigned>(rows), 1);
    gqa_decode_merge_f16_kernel<<<merge_grid, kThreads, 0, cuda_stream>>>(
        partial_scratch, output, q_heads, head_dim, splits);
    return cudaGetLastError() == cudaSuccess;
}

bool qwen_gqa_decode_attention_f16_fused_variant_cuda(
    const uint16_t* q, const uint16_t* k_cache,
    const uint16_t* v_cache, uint16_t* output, float* partial_scratch,
    int q_heads, int kv_heads, int head_dim, int context_len,
    int max_context, int attention_window, int sink_tokens,
    QwenGqaDecodeVariant variant, int split_count_override, void* stream) {
    return launch_decode_fused(
        q, k_cache, v_cache, output, partial_scratch, q_heads, kv_heads,
        head_dim, context_len, max_context, attention_window, sink_tokens,
        variant, split_count_override, stream);
}

}  // namespace pocket
