#pragma once

#include <cstddef>
#include <cstdint>

namespace pocket {

bool cuda_runtime_available();

bool q8_0_matvec_cuda(
    const float* d_x,
    const uint8_t* d_w,
    float* d_y,
    int rows,
    int cols,
    void* stream = nullptr);

bool q8_0_matmul_rows_cuda(
    const float* d_x,
    const uint8_t* d_w,
    float* d_y,
    int batch,
    int rows,
    int cols,
    void* stream = nullptr);

bool q8_0_matmul_rows_strided_cuda(
    const float* d_x,
    const uint8_t* d_w,
    float* d_y,
    int batch,
    int rows,
    int cols,
    int x_stride,
    int y_stride,
    void* stream = nullptr);

// --- IQ2_XXS / Q2_K single-token MoE decode kernels (cuda/q2_ops.cu). ---

// Returns a device pointer to the lazy-initialized signed_grid lookup table
// (shape [256, 128, 8] int8, ~256 KiB). Returns nullptr on allocation failure.
const int8_t* q2_signed_grid_device();

// Quantize fp32 activations to Q8_1 with 32-element groups (one row per launch).
bool q2_quantize_x_q8_1_cuda(
    const float* d_x,
    int8_t* d_x_q,
    float* d_x_scale,
    int rows,
    int row_elems,
    void* stream = nullptr);

// Quantize fp32 hidden activations to Q8_1 with 16-element groups (post-SwiGLU).
bool q2_quantize_hidden_q8_1_cuda(
    const float* d_hidden,
    int8_t* d_hidden_q,
    float* d_hidden_scale,
    int routes,
    int inter_dim,
    void* stream = nullptr);

// Fused: SwiGLU(gate, up) * route_weight -> Q8_1 with 16-element groups.
bool q2_route_swiglu_quantize_hidden_q8_1_cuda(
    const float* d_gate,
    const float* d_up,
    const float* d_route_weights,
    int8_t* d_hidden_q,
    float* d_hidden_scale,
    int routes,
    int inter_dim,
    float swiglu_limit,
    void* stream = nullptr);

// IQ2_XXS W1+W3 single-token matvec (per-route gate/up outputs).
// d_w1_blocks/d_w3_blocks layout: per-expert [inter_dim, blocks_per_row*66] flat bytes
// where blocks_per_row = dim / 256. Output: d_gate/d_up [routes, inter_dim].
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
    void* stream = nullptr);

// Q2_K W2 single-token matvec. Accumulates onto d_y (caller must zero beforehand).
// d_w2_blocks layout: per-expert [dim, blocks_per_row*84] flat bytes
// where blocks_per_row = inter_dim / 256.
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
    void* stream = nullptr);

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
    void* stream = nullptr);

struct MoePrefillQ2GroupedWorkspace {
    float* d_x_route = nullptr;    // [routes_cap, dim]
    int8_t* d_x_q = nullptr;       // [routes_cap, dim]
    float* d_x_scale = nullptr;    // [routes_cap, ceil(dim/32)]
    int64_t* d_route_slots = nullptr; // [routes_cap]
    float* d_gate = nullptr;       // [routes_cap, inter_dim]
    float* d_up = nullptr;         // [routes_cap, inter_dim]
    int8_t* d_hidden_q = nullptr;  // [routes_cap, inter_dim]
    float* d_hidden_scale = nullptr; // [routes_cap, ceil(inter_dim/16)]
    int routes_cap = 0;
    int dim = 0;
    int inter_dim = 0;
};

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
    void* stream = nullptr);

// --- IQ1_M single-token MoE decode kernels (cuda/iq1_ops.cu). ---

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
    void* stream = nullptr);

// Fused IQ1_M W1/W3 matvec + SwiGLU + route weight for decode.
// Writes d_hidden [routes, inter_dim] directly and avoids gate/up intermediates.
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
    void* stream = nullptr);

bool iq1_route_swiglu_cuda(
    const float* d_gate,
    const float* d_up,
    const float* d_route_weights,
    float* d_hidden,
    int routes,
    int inter_dim,
    float swiglu_limit,
    void* stream = nullptr);

bool iq1_moe_single_w2_cuda(
    const float* d_hidden,
    const int64_t* d_route_slots,
    const uint8_t* d_w2_blocks,
    float* d_y,
    int routes,
    int n_experts,
    int dim,
    int inter_dim,
    void* stream = nullptr);

// IQ1_M W2 specialization for small top-k decode (routes <= 8): one CTA per
// output column reduces all active routes and writes d_y directly, avoiding
// per-route atomics.
bool iq1_moe_single_w2_reduce_cuda(
    const float* d_hidden,
    const int64_t* d_route_slots,
    const uint8_t* d_w2_blocks,
    float* d_y,
    int routes,
    int n_experts,
    int dim,
    int inter_dim,
    void* stream = nullptr);

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
    void* stream = nullptr);

struct MoePrefillIq1GroupedWorkspace {
    float* d_hidden = nullptr;       // [routes_cap, inter_dim]
    int8_t* d_hidden_q = nullptr;    // [routes_cap, inter_dim], for IQ1 grouped W2 q8 path and IQ1 W13 + Q2_K W2 recipe
    float* d_hidden_scale = nullptr; // [routes_cap, ceil(inter_dim/16)]
    int8_t* d_x_q = nullptr;         // [routes_cap, dim], IQ1 grouped-GEMM W13 q8 route rows
    float* d_x_scale = nullptr;      // [routes_cap, ceil(dim/16)]
    int32_t* d_tile_experts = nullptr; // [tile_cap], compact route tiles
    int32_t* d_tile_rows = nullptr;    // [tile_cap], row offset within expert segment
    int routes_cap = 0;
    int tile_cap = 0;
    int tile_count = 0;
    int dim = 0;
    int inter_dim = 0;
};

// IQ1_M grouped routed MoE for prefill. Routes are grouped by local expert via
// moe_group_routes_cuda: d_seg_starts[e]..d_seg_starts[e+1] are route ids for
// expert e, and d_route_tokens[route] selects the source/output token row.
// Consumes raw IQ1_M blocks directly and writes d_y_rows [tokens, dim].
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
    void* stream = nullptr);

bool fp4_e2m1_e8m0_matvec_cuda(
    const float* d_x,
    const uint8_t* d_weight,
    const uint8_t* d_scale,
    float* d_y,
    int rows,
    int cols,
    void* stream = nullptr);

struct MoeSingleTokenFp4Workspace {
    int8_t* d_x_q = nullptr;
    float* d_x_scale = nullptr;
    float* d_gate = nullptr;
    float* d_up = nullptr;
    int8_t* d_hidden_q = nullptr;
    float* d_hidden_scale = nullptr;
    float* d_route_y = nullptr;
    int topk = 0;
    int dim = 0;
    int inter_dim = 0;
};

// Persistent scratch for the active-slot small-batch FP4 MoE path. Route inputs
// are compacted by expert: slot_expert[s] names a local expert and
// [slot_starts[s], slot_starts[s + 1]) contains its (token, weight) pairs.
struct MoeMultiTokenFp4Workspace {
    int8_t* d_x_q = nullptr;
    float* d_x_scale = nullptr;
    float* d_gate = nullptr;
    float* d_up = nullptr;
    int8_t* d_hidden_q = nullptr;
    float* d_hidden_scale = nullptr;
    float* d_partials = nullptr;
    int tokens_cap = 0;
    int pairs_cap = 0;
    int dim = 0;
    int inter_dim = 0;
};

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
    void* stream = nullptr);

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
    void* stream = nullptr);

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
    void* stream = nullptr);

// d_token_slot_routes (optional, [tokens, topk]) receives the inverse of
// d_route_tokens: slot k of token t holds the route id assigned there, or -1
// when that expert is not local to this rank. Pass nullptr to skip it.
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
    int32_t* d_token_slot_routes = nullptr,
    void* stream = nullptr);

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
    void* stream = nullptr);

struct MoePrefillFp4GroupedWorkspace {
    float* d_x_sorted = nullptr;
    int8_t* d_x_q = nullptr;
    float* d_x_scale = nullptr;
    int8_t* d_x_pad = nullptr;
    float* d_x_scale_pad = nullptr;
    float* d_gate = nullptr;
    float* d_up = nullptr;
    int8_t* d_hidden_q = nullptr;
    float* d_hidden_scale = nullptr;
    int32_t* d_tile_experts = nullptr;
    int32_t* d_tile_rows = nullptr;
    // [routes_cap, dim] scratch for the deterministic reduction. Lives in the
    // workspace because a per-call cudaMalloc/cudaFree here costs ~19x prefill
    // throughput -- each pair synchronizes the device inside the layer loop.
    float* d_partials = nullptr;
    int routes_cap = 0;
    int padded_rows_cap = 0;
    int tile_cap = 0;
    int tile_count = 0;
    int dim = 0;
    int inter_dim = 0;
};

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
    // Optional [tokens, topk] slot->route map from moe_group_routes_cuda. When
    // present (and topk>=3) the w2 output is reduced in fixed slot order
    // instead of by atomicAdd, which makes the result reproducible.
    const int32_t* d_token_slot_routes = nullptr,
    void* stream = nullptr);

bool fp8_e4m3_e8m0_matvec_cuda(
    const float* d_x,
    const uint8_t* d_weight,
    const uint8_t* d_scale,
    float* d_y,
    int rows,
    int cols,
    void* stream = nullptr);

bool fp8_e4m3_e8m0_matmul_cuda(
    const float* d_x,
    const uint8_t* d_weight,
    const uint8_t* d_scale,
    float* d_y,
    int batch,
    int rows,
    int cols,
    void* stream = nullptr);

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
    void* stream = nullptr);

bool wo_a_int8_decode_cuda(
    const float* d_x,
    const int8_t* d_weight_q,
    const float* d_weight_scale,
    float* d_y,
    int groups,
    int group_rank,
    int group_dim,
    int8_t* d_x_q,
    float* d_x_scale,
    void* stream = nullptr);

bool rmsnorm_bf16_gamma_cuda(
    const float* d_x,
    const uint16_t* d_gamma_bf16,
    float* d_y,
    int cols,
    float eps,
    void* stream = nullptr);

bool rmsnorm_bf16_gamma_rows_cuda(
    const float* d_x,
    const uint16_t* d_gamma_bf16,
    float* d_y,
    int rows,
    int cols,
    float eps,
    void* stream = nullptr);

bool silu_mul_cuda(
    const float* d_gate,
    const float* d_up,
    float* d_y,
    int cols,
    void* stream = nullptr);

bool silu_mul_rows_cuda(
    const float* d_gate,
    const float* d_up,
    float* d_y,
    int rows,
    int cols,
    void* stream = nullptr);

bool silu_mul_clamped_cuda(
    const float* d_gate,
    const float* d_up,
    float* d_y,
    int cols,
    float limit,
    void* stream = nullptr);

bool bf16_row_to_float_cuda(
    const uint16_t* d_matrix_bf16,
    float* d_y,
    int row,
    int cols,
    void* stream = nullptr);

// GGUF embed/head are stored F16 (IEEE half), not BF16. This is the F16
// counterpart of bf16_row_to_float_cuda for the GGUF dense path.
bool f16_row_to_float_cuda(
    const uint16_t* d_matrix_f16,
    float* d_y,
    int row,
    int cols,
    void* stream = nullptr);

bool f16_rows_to_float_cuda(
    const uint16_t* d_matrix_f16,
    const int* d_rows,
    float* d_y,
    int rows,
    int cols,
    void* stream = nullptr);

bool f16_contiguous_rows_to_float_cuda(
    const uint16_t* d_matrix_f16,
    float* d_y,
    int rows,
    int cols,
    void* stream = nullptr);

bool bf16_rows_to_float_cuda(
    const uint16_t* d_matrix_bf16,
    const int* d_rows,
    float* d_y,
    int rows,
    int cols,
    void* stream = nullptr);

bool bf16_matvec_cuda(
    const float* d_x,
    const uint16_t* d_w_bf16,
    float* d_y,
    int rows,
    int cols,
    void* stream = nullptr);

// Batched over a small number of token rows: d_y[t, r] = sum_c d_x[t, c] * w[r, c].
// d_x is [tokens, cols], d_y is [tokens, rows]. `tokens` must be <= 16 -- the
// per-thread accumulators are registers, which is what lets the weight row be
// read once for all tokens instead of once per token.
bool bf16_matvec_rows_cuda(
    const float* d_x,
    const uint16_t* d_w_bf16,
    float* d_y,
    int tokens,
    int rows,
    int cols,
    void* stream = nullptr);

bool bf16_dual_matvec_cuda(
    const float* d_x,
    const uint16_t* d_w_a_bf16,
    const uint16_t* d_w_b_bf16,
    float* d_y_a,
    float* d_y_b,
    int rows,
    int cols,
    void* stream = nullptr);

// Same as bf16_dual_matvec_cuda, but the activation vector is already BF16
// encoded. This avoids a BF16->FP32 restore kernel for compressor paths that
// intentionally round activations to BF16 before matvec.
bool bf16_dual_matvec_bf16_x_cuda(
    const uint16_t* d_x_bf16,
    const uint16_t* d_w_a_bf16,
    const uint16_t* d_w_b_bf16,
    float* d_y_a,
    float* d_y_b,
    int rows,
    int cols,
    void* stream = nullptr);

bool bf16_matvec_cpu_order_cuda(
    const float* d_x,
    const uint16_t* d_w_bf16,
    float* d_y,
    int rows,
    int cols,
    void* stream = nullptr);

bool gate_topk_bf16_cuda(
    const float* d_x,
    const uint16_t* d_w_bf16,
    const float* d_bias,
    int64_t* d_indices,
    float* d_weights,
    int experts,
    int cols,
    int topk,
    float route_scale,
    void* stream = nullptr);

bool gate_topk_bf16_cuda_with_buffers(
    const float* d_x,
    const uint16_t* d_w_bf16,
    const float* d_bias,
    float* d_original,
    float* d_scored,
    int64_t* d_indices,
    float* d_weights,
    int experts,
    int cols,
    int topk,
    float route_scale,
    void* stream = nullptr);

bool gate_topk_bf16_rows_cuda(
    const float* d_x,
    const uint16_t* d_w_bf16,
    const float* d_bias,
    int64_t* d_indices,
    float* d_weights,
    int tokens,
    int experts,
    int cols,
    int topk,
    float route_scale,
    void* stream = nullptr);

bool gate_hash_bf16_rows_cuda(
    const float* d_x,
    const uint16_t* d_w_bf16,
    const int64_t* d_tid2eid,
    const int* d_token_ids,
    int64_t* d_indices,
    float* d_weights,
    int tokens,
    int cols,
    int table_topk,
    int topk,
    float route_scale,
    void* stream = nullptr);

bool gate_hash_bf16_cuda(
    const float* d_x,
    const uint16_t* d_w_bf16,
    const int64_t* d_tid2eid,
    float* d_original_scratch,
    int64_t* d_indices,
    float* d_weights,
    int token,
    int cols,
    int table_topk,
    int topk,
    float route_scale,
    void* stream = nullptr);

// Map global routed expert ids to this TP rank's resident-local expert slots.
// Experts outside [expert_start, expert_start + experts_per_rank) become -1.
bool gguf_route_slots_from_indices_cuda(
    const int64_t* d_indices,
    int64_t* d_route_slots,
    int topk,
    int expert_start,
    int experts_per_rank,
    void* stream = nullptr);

// Greedy top-1 over fp32 logits on device. Writes the global token id
// (local_vocab_start + local index) and its logit to one-element device outputs.
bool argmax_fp32_cuda(
    const float* d_logits,
    int* d_token,
    float* d_logit,
    int count,
    int token_offset,
    void* stream = nullptr);

// Row-wise greedy top-1 over a contiguous [rows, count] logits matrix.
// Each output token includes token_offset and ties prefer the lower token id.
bool argmax_fp32_rows_cuda(
    const float* d_logits,
    int* d_tokens,
    float* d_logits_out,
    int rows,
    int count,
    int token_offset,
    void* stream = nullptr);

bool vector_add_cuda(
    const float* d_a,
    const float* d_b,
    float* d_y,
    int cols,
    void* stream = nullptr);

bool vector_accum_cuda(
    const float* d_x,
    float* d_y,
    int cols,
    float scale,
    void* stream = nullptr);

bool vector_accum_rows_cuda(
    const float* d_x,
    float* d_y,
    int rows,
    int cols,
    float scale,
    void* stream = nullptr);

bool repeat_vector_cuda(
    const float* d_x,
    float* d_y,
    int cols,
    int repeats,
    void* stream = nullptr);

bool prefill_causal_attention_cuda(
    const float* d_q,
    const float* d_kv,
    const float* d_attn_sink,
    float* d_y,
    int tokens,
    int heads,
    int head_dim,
    int window_size,
    float scale,
    void* stream = nullptr);

bool prefill_causal_attention_chunk_cuda(
    const float* d_q,
    const float* d_kv,
    const float* d_attn_sink,
    float* d_y,
    int tokens,
    int heads,
    int kv_len,
    int head_dim,
    int window_size,
    int start_position,
    float scale,
    void* stream = nullptr);

bool build_prefill_window_indices_cuda(
    int32_t* d_indices,
    int tokens,
    int window_size,
    int topk,
    void* stream = nullptr);

bool build_decode_kv_indices_cuda(
    int* d_indices,
    int window_start,
    int window_len,
    int window_size,
    int compressed_count,
    int compressed_offset,
    void* stream = nullptr);

bool prefill_sparse_attention_indexed_cuda(
    const float* d_q,
    const float* d_kv,
    const float* d_attn_sink,
    const int32_t* d_topk_indices,
    float* d_y,
    int tokens,
    int heads,
    int kv_len,
    int topk,
    int head_dim,
    float scale,
    void* stream = nullptr);

bool prefill_sparse_attention_headpair_cuda(
    const float* d_q,
    const float* d_kv,
    const float* d_attn_sink,
    const int32_t* d_topk_indices,
    float* d_y,
    int tokens,
    int heads,
    int kv_len,
    int topk,
    int head_dim,
    float scale,
    void* stream = nullptr);

bool prefill_sparse_attention_headpair_serial_cuda(
    const float* d_q,
    const float* d_kv,
    const float* d_attn_sink,
    const int32_t* d_topk_indices,
    float* d_y,
    int tokens,
    int heads,
    int kv_len,
    int topk,
    int head_dim,
    float scale,
    void* stream = nullptr);

bool single_token_sparse_attention_cuda(
    const float* d_q,
    const float* d_kv,
    const float* d_attn_sink,
    float* d_y,
    int heads,
    int head_dim,
    float scale,
    void* stream = nullptr);

bool cached_single_token_attention_cuda(
    const float* d_q,
    const float* d_kv_cache,
    const float* d_attn_sink,
    float* d_y,
    int heads,
    int head_dim,
    int cache_len,
    float scale,
    void* stream = nullptr);

// Cached single-token attention using caller-provided global weight scratch
// [heads, cache_len]. Avoids dynamic shared memory growth with context length.
bool cached_single_token_attention_workspace_cuda(
    const float* d_q,
    const float* d_kv_cache,
    const float* d_attn_sink,
    float* d_weight_scratch,
    float* d_y,
    int heads,
    int head_dim,
    int cache_len,
    float scale,
    void* stream = nullptr);

bool indexed_cached_single_token_attention_cuda(
    const float* d_q,
    const float* d_kv_cache,
    const int* d_indices,
    const float* d_attn_sink,
    float* d_y,
    int heads,
    int head_dim,
    int index_count,
    float scale,
    void* stream = nullptr);

// Continuation attention for a small block of query rows. d_row_starts is a
// CSR offset array [rows + 1] into d_indices; each row can therefore expose a
// different causal window and compressed/indexer selection.
bool indexed_cached_attention_rows_cuda(
    const float* d_q,
    const float* d_kv_cache,
    const int32_t* d_row_starts,
    const int32_t* d_indices,
    const float* d_attn_sink,
    float* d_y,
    int rows,
    int heads,
    int head_dim,
    int max_index_count,
    float scale,
    void* stream = nullptr);

// Variant for continuation blocks whose current rows have not yet been written
// into the live ring. Non-negative indices address d_kv_cache; indices <= -2
// address d_batch_kv[-index - 2]. Keeping current rows separate prevents a later
// row in a wrapping block from overwriting history still visible to an earlier
// row. -1 remains an invalid/padding index.
bool indexed_cached_attention_rows_batch_kv_cuda(
    const float* d_q,
    const float* d_kv_cache,
    const float* d_batch_kv,
    const int32_t* d_row_starts,
    const int32_t* d_indices,
    const float* d_attn_sink,
    float* d_y,
    int rows,
    int heads,
    int head_dim,
    int max_index_count,
    float scale,
    void* stream = nullptr);

bool indexer_select_topk_cuda(
    const float* d_index_q,
    const float* d_index_kv,
    const uint16_t* d_weight_proj_bf16,
    const float* d_x,
    float* d_scores_scratch,
    int* d_out_indices,
    int compressed_count,
    int keep,
    int heads,
    int head_dim,
    int dim,
    int offset,
    void* stream = nullptr);

bool hadamard128_rows_cuda(
    const float* d_x,
    float* d_y,
    int rows,
    void* stream = nullptr);

bool fp4_fake_quant128_rows_cuda(
    float* d_x,
    int rows,
    void* stream = nullptr);

bool fp8_act_quant_dequant_cuda(
    float* d_x,
    int cols,
    int block_size,
    void* stream = nullptr);

bool compressor_update_state_cuda(
    const float* d_kv,
    const float* d_score,
    const float* d_ape,
    float* d_kv_state,
    float* d_score_state,
    int offset,
    int write_slot,
    int state_cols,
    void* stream = nullptr);

bool compressor_pool_cuda(
    const float* d_kv_state,
    const float* d_score_state,
    float* d_out,
    int ratio,
    int head_dim,
    int state_cols,
    bool overlap,
    void* stream = nullptr);

bool compressor_shift_overlap_state_cuda(
    float* d_kv_state,
    float* d_score_state,
    int ratio,
    int state_cols,
    void* stream = nullptr);

bool head_rmsnorm_rope_cuda(
    float* d_x,
    int heads,
    int head_dim,
    int rope_dim,
    int position,
    float theta,
    bool inverse,
    float eps,
    void* stream = nullptr);

bool head_rmsnorm_rope_rows_cuda(
    float* d_x,
    int tokens,
    int heads,
    int head_dim,
    int rope_dim,
    int start_position,
    float theta,
    bool inverse,
    float eps,
    void* stream = nullptr);

bool head_rmsnorm_rope_freqs_cuda(
    float* d_x,
    const float* d_inv_freqs,
    int heads,
    int head_dim,
    int rope_dim,
    int position,
    bool inverse,
    float eps,
    void* stream = nullptr);

bool head_rmsnorm_rope_freqs_rows_cuda(
    float* d_x,
    const float* d_inv_freqs,
    int tokens,
    int heads,
    int head_dim,
    int rope_dim,
    int start_position,
    bool inverse,
    float eps,
    void* stream = nullptr);

bool fp8_act_quant_dequant_rows_cuda(
    float* d_x,
    int rows,
    int cols,
    int block_size,
    void* stream = nullptr);

bool fp8_act_quant_dequant_rows_strided_cuda(
    float* d_x,
    int rows,
    int cols,
    int row_stride,
    int block_size,
    void* stream = nullptr);

bool copy_rows_to_kv_cache_cuda(
    const float* d_rows,
    float* d_cache,
    int rows,
    int cols,
    int window_size,
    int start_position = 0,
    void* stream = nullptr);

bool fp32_to_bf16_cuda(
    const float* d_x,
    uint16_t* d_y,
    int count,
    void* stream = nullptr);

bool bf16_to_fp32_cuda(
    const uint16_t* d_x,
    float* d_y,
    int count,
    void* stream = nullptr);

bool hc_repeat_rows_cuda(
    const float* d_x_rows,
    float* d_h4_rows,
    int rows,
    int dim,
    void* stream = nullptr);

bool hc_pre_float_rows_cuda(
    const float* d_h4_rows,
    const float* d_fn,
    const float* d_scale,
    const float* d_base,
    float* d_x_rows,
    float* d_post_rows,
    float* d_comb_rows,
    int rows,
    int dim,
    void* stream = nullptr);

bool hc_post_float_rows_cuda(
    const float* d_x_rows,
    const float* d_residual_h4_rows,
    const float* d_post_rows,
    const float* d_comb_rows,
    float* d_y_h4_rows,
    int rows,
    int dim,
    void* stream = nullptr);

// Collapse [rows, 4, dim] -> [rows, dim] with the head's gate. Distinct from
// hc_pre: only 4 mixes, one shared scale, and no post/comb outputs.
//   d_fn    [4, 4*dim]
//   d_scale [1]
//   d_base  [4]
bool hc_head_float_rows_cuda(
    const float* d_h4_rows,
    const float* d_fn,
    const float* d_scale,
    const float* d_base,
    float* d_y_rows,
    int rows,
    int dim,
    void* stream = nullptr);

// Mean over the hc dimension: [rows, 4, dim] -> [rows, dim], written at
// d_out_rows[row * out_stride + col_offset]. The strided destination lets the
// DSpark target layers be pooled straight into one concatenated
// [rows, n_target * dim] buffer, matching the reference's cat on the last axis.
bool hc_mean_pool_rows_cuda(
    const float* d_h4_rows,
    float* d_out_rows,
    int rows,
    int dim,
    int out_stride,
    int col_offset,
    void* stream = nullptr);


bool hc_pre_float_cuda(
    const float* d_h4,
    const float* d_fn,
    const float* d_scale,
    const float* d_base,
    float* d_x,
    float* d_post,
    float* d_comb,
    int dim,
    void* stream = nullptr);

bool hc_post_float_cuda(
    const float* d_x,
    const float* d_residual_h4,
    const float* d_post,
    const float* d_comb,
    float* d_y_h4,
    int dim,
    void* stream = nullptr);

}  // namespace pocket
