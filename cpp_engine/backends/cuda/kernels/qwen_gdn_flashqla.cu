// FlashQLA SM75 Gated DeltaNet kernel — subgroup-sharded state for D=128.
// Ported from vLLM-2080Ti-Definitive v0.1.15 tools/flashqla_sm75_patches/gdn_forward.cu
// for comparison against the baseline serial recurrence in qwen_attention_ops.cu.
//
// Key differences from the baseline:
// - State sharding: each subgroup (WIDTH=16 lanes) holds COLS=4 columns of the
//   [D, D] state matrix as `state_shard[COLS][rows_per_lane]`, rather than one
//   thread per value dimension holding the full key_dim-length state vector.
// - Launch geometry: grid.z dimension fans out when D/COLS exceeds the CTA count,
//   and blockDim.y gives multiple column-groups per CTA.
// - Still a serial `for (int t = 0; t < tokens; ++t)` recurrence — not chunked.
//   Occupancy and serialization remain, but memory access is better.

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdlib>

namespace pocket {

__device__ __forceinline__ float warp_reduce_sum(float value, int width) {
    for (int offset = width / 2; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffffU, value, offset, width);
    }
    return value;
}

__device__ __forceinline__ float warp_broadcast_lane0(float value, int width) {
    return __shfl_sync(0xffffffffU, value, 0, width);
}

// D: key_dim and value_dim (compile-time 128)
// COLS: state columns per subgroup (4 for D=128)
// WIDTH: subgroup size (16 for D=128)
template <int D, int COLS, int WIDTH>
__global__ void gdn_flashqla_kernel(
    const float* __restrict__ q_normalized,
    const float* __restrict__ k_normalized,
    const __half* __restrict__ v,
    const __half* __restrict__ gate,
    const __half* __restrict__ beta,
    const float* __restrict__ initial_state,
    __half* __restrict__ output,
    float* __restrict__ final_state,
    int tokens,
    int q_heads,
    int v_heads,
    float q_scale
) {
    static_assert(D == 128 && COLS == 4 && WIDTH == 16, "D=128 COLS=4 WIDTH=16 only");
    static_assert(D % (COLS * (32 / WIDTH)) == 0, "D must divide evenly");

    constexpr int subgroups_per_warp = 32 / WIDTH;  // 2
    constexpr int rows_per_lane = (D + WIDTH - 1) / WIDTH;  // 8

    const int hv = static_cast<int>(blockIdx.x);
    const int subgroup = threadIdx.x / WIDTH;
    const int lane = threadIdx.x % WIDTH;
    const int group_base =
        (static_cast<int>(blockIdx.z) * static_cast<int>(blockDim.y) +
         static_cast<int>(threadIdx.y)) * subgroups_per_warp + subgroup;
    const int col_base = group_base * COLS;
    const int hq = hv / (v_heads / q_heads);

    // Load initial state for this subgroup's column shard.
    float state_shard[COLS][rows_per_lane];
#pragma unroll
    for (int c = 0; c < COLS; ++c) {
        const int col = col_base + c;
#pragma unroll
        for (int r = 0; r < rows_per_lane; ++r) {
            const int row = r * WIDTH + lane;
            float value = 0.0f;
            if (row < D && col < D) {
                const size_t state_index =
                    static_cast<size_t>(hv) * D * D +
                    static_cast<size_t>(row) * D +
                    static_cast<size_t>(col);
                value = initial_state == nullptr ? 0.0f : initial_state[state_index];
            }
            state_shard[c][r] = value;
        }
    }

    // Software-pipelined operand loads.
    //
    // Nothing a token reads depends on the previous token's state, so every load
    // can be issued one iteration early and its latency overlapped with the
    // recurrence arithmetic. That matters here because the recurrence is serial in
    // tokens and the per-token critical path was dominated by memory latency
    // rather than math: `gate`/`beta` were fetched at the top of the body, and
    // `v[col]` was fetched by lane 0 *after* the K-state reduction, putting a full
    // dependent global load in the middle of the chain. Holding token t+1's
    // operands in registers while token t computes leaves only the shuffle
    // reductions on the path. The arithmetic, its order, and the reduction widths
    // are untouched, so results stay bit-identical.
    float q_next[rows_per_lane];
    float k_next[rows_per_lane];
    float v_next[COLS];
    float gate_next = 0.0f;
    float beta_next = 0.0f;

    // `gate`/`beta` are per-head scalars broadcast across the warp. Each y-row is
    // an independent warp working a different state-column group, so lane 0 of
    // every warp loads them; expf is applied at the load site exactly as before.
    auto load_scalars = [&](int t, float* gate_out, float* beta_out) {
        float g_value = 0.0f;
        float b_value = 0.0f;
        if (threadIdx.x == 0) {
            g_value = expf(__half2float(gate[static_cast<size_t>(t) * v_heads + hv]));
            b_value = __half2float(beta[static_cast<size_t>(t) * v_heads + hv]);
        }
        *gate_out = g_value;
        *beta_out = b_value;
    };

    // Pre-normalized Q and K. Q and K share the key head hq under GQA, so the
    // key row stride is q_heads * D.
    auto load_qk = [&](int t, float* q_out, float* k_out) {
#pragma unroll
        for (int r = 0; r < rows_per_lane; ++r) {
            const int row = r * WIDTH + lane;
            float q_value = 0.0f;
            float k_value = 0.0f;
            if (row < D) {
                const size_t qk_index =
                    static_cast<size_t>(t) * q_heads * D +
                    static_cast<size_t>(hq) * D +
                    static_cast<size_t>(row);
                q_value = q_normalized[qk_index];
                k_value = k_normalized[qk_index];
            }
            q_out[r] = q_value;
            k_out[r] = k_value;
        }
    };

    // Only lane 0 consumes v, matching the original delta computation.
    auto load_v = [&](int t, float* v_out) {
#pragma unroll
        for (int c = 0; c < COLS; ++c) {
            float value = 0.0f;
            if (lane == 0) {
                const int col = col_base + c;
                if (col < D) {
                    const size_t v_index =
                        static_cast<size_t>(t) * v_heads * D +
                        static_cast<size_t>(hv) * D +
                        static_cast<size_t>(col);
                    value = __half2float(v[v_index]);
                }
            }
            v_out[c] = value;
        }
    };

    load_scalars(0, &gate_next, &beta_next);
    load_qk(0, q_next, k_next);
    load_v(0, v_next);

    // Serial recurrence over tokens.
    for (int t = 0; t < tokens; ++t) {
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
        float v_reg[COLS];
#pragma unroll
        for (int r = 0; r < rows_per_lane; ++r) {
            q_reg[r] = q_next[r];
            k_reg[r] = k_next[r];
        }
#pragma unroll
        for (int c = 0; c < COLS; ++c) v_reg[c] = v_next[c];
        float gate_value = gate_next;
        float beta_value = beta_next;

        // Issue the next token's loads before consuming this one's, so their
        // latency hides behind the reductions below.
        if (t + 1 < tokens) {
            load_scalars(t + 1, &gate_next, &beta_next);
            load_qk(t + 1, q_next, k_next);
            load_v(t + 1, v_next);
        }

        gate_value = warp_broadcast_lane0(gate_value, 32);
        beta_value = warp_broadcast_lane0(beta_value, 32);

        // Compute K · state for each column: sum over rows (subgroup reduction).
        float kv_partial[COLS];
#pragma unroll
        for (int c = 0; c < COLS; ++c) {
            float sum = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; ++r) {
                sum += state_shard[c][r] * k_reg[r];
            }
            kv_partial[c] = warp_reduce_sum(sum, WIDTH);
        }

        // Compute delta = (v - gate * K·state) * beta for each column. v is already
        // in registers from the prefetch, so lane 0 only does arithmetic here.
        float delta[COLS];
#pragma unroll
        for (int c = 0; c < COLS; ++c) {
            float delta_value = 0.0f;
            if (lane == 0) {
                const int col = col_base + c;
                if (col < D) {
                    delta_value = (v_reg[c] - gate_value * kv_partial[c]) *
                                  beta_value;
                }
            }
            delta[c] = warp_broadcast_lane0(delta_value, WIDTH);
        }

        // Update state: state = gate * state + K ⊗ delta, and accumulate Q · state.
        float attn_partial[COLS];
#pragma unroll
        for (int c = 0; c < COLS; ++c) {
            float sum = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; ++r) {
                const float new_state =
                    fmaf(k_reg[r], delta[c], gate_value * state_shard[c][r]);
                state_shard[c][r] = new_state;
                sum += new_state * q_reg[r];
            }
            attn_partial[c] = warp_reduce_sum(sum, WIDTH);
        }

        // Lane 0 writes output for its columns.
        if (lane == 0) {
            const size_t out_base = static_cast<size_t>(t) * v_heads * D +
                                    static_cast<size_t>(hv) * D;
#pragma unroll
            for (int c = 0; c < COLS; ++c) {
                const int col = col_base + c;
                if (col < D) {
                    output[out_base + col] = __float2half(attn_partial[c] * q_scale);
                }
            }
        }
    }

    // Store final state for this subgroup's shard.
    if (final_state != nullptr) {
#pragma unroll
        for (int c = 0; c < COLS; ++c) {
            const int col = col_base + c;
#pragma unroll
            for (int r = 0; r < rows_per_lane; ++r) {
                const int row = r * WIDTH + lane;
                if (row < D && col < D) {
                    const size_t state_index =
                        static_cast<size_t>(hv) * D * D +
                        static_cast<size_t>(row) * D +
                        static_cast<size_t>(col);
                    final_state[state_index] = state_shard[c][r];
                }
            }
        }
    }
}

bool qwen_gated_delta_flashqla_sm75_f16_cuda(
    float* d_state,
    const float* d_q_normalized,
    const float* d_k_normalized,
    const uint16_t* d_v,
    const uint16_t* d_g,
    const uint16_t* d_beta,
    uint16_t* d_out,
    int rows,
    int heads,
    int key_heads,
    int key_dim,
    int value_dim,
    float q_scale,
    void* stream
) {
    if (!d_state || !d_q_normalized || !d_k_normalized || !d_v || !d_g ||
        !d_beta || !d_out || rows <= 0 || heads <= 0 || key_heads <= 0 ||
        heads % key_heads != 0 || key_dim != 128 || value_dim != 128 ||
        q_scale <= 0.0f) {
        return false;
    }

    constexpr int D = 128;
    constexpr int COLS = 4;
    constexpr int WIDTH = 16;
    constexpr int subgroups_per_warp = 32 / WIDTH;  // 2

    // Column groups per CTA. Every subgroup is fully independent -- it owns COLS
    // columns of the [D, D] state and never communicates -- so this only decides
    // how the fixed 32 groups per head are packed into CTAs, never the arithmetic.
    // Packing 8 groups per CTA leaves grid=(heads,1,2): on the real TP4 shape that
    // is 16 CTAs for 68 SMs, so three quarters of the device is idle through a
    // recurrence that is already serial in tokens. Fewer groups per CTA trades
    // redundant Q/K re-reads (each group reads all D rows of the same q/k row) for
    // SM coverage; the default is swept in bench_qwen_delta_f16.
    static const int column_groups_per_block = [] {
        const char* raw = std::getenv("QWEN_GDN_FLASHQLA_GROUPS_PER_CTA");
        const int value = raw == nullptr ? 1 : std::atoi(raw);
        return value >= 1 && value <= 8 ? value : 1;
    }();

    const int groups = D / COLS;  // 32
    const int z = (groups + column_groups_per_block * subgroups_per_warp - 1) /
                  (column_groups_per_block * subgroups_per_warp);

    const dim3 block(32, column_groups_per_block);  // 32 threads (1 warp), 8 y-groups
    const dim3 grid(heads, 1, z);

    const cudaStream_t s = stream == nullptr ? nullptr : static_cast<cudaStream_t>(stream);

    gdn_flashqla_kernel<D, COLS, WIDTH><<<grid, block, 0, s>>>(
        d_q_normalized,
        d_k_normalized,
        reinterpret_cast<const __half*>(d_v),
        reinterpret_cast<const __half*>(d_g),
        reinterpret_cast<const __half*>(d_beta),
        d_state,
        reinterpret_cast<__half*>(d_out),
        d_state,
        rows,
        key_heads,
        heads,
        q_scale
    );

    return cudaGetLastError() == cudaSuccess;
}

}  // namespace pocket
