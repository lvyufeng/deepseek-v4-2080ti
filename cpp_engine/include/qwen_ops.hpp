#pragma once

// Backend-neutral names for the Qwen operators, so engine code can call
// qwen_rmsnorm_fp16_gamma_rows_f16(...) instead of naming a vendor. Each name
// forwards to the *_cuda symbol or the *_ascend one depending on the build.
//
// Only operators the dense FP16 path needs are here. Three groups are absent on
// purpose:
//
//   - Quantized paths (FP8, NVFP4, INT8, TurboQuant). Ascend 910 has no FP8 and
//     the Qwen3.8-27B checkpoint this backend targets is BF16, so there is
//     nothing to dequantize.
//   - Hardware-specific variants: the cuBLAS-backed matmuls, the SM75 FlashQLA
//     kernel, and the fused/tiled/exact attention forms. These are CUDA tuning
//     choices, not distinct operations, and each has a portable equivalent here.
//   - Speculative decoding (DSpark, DFlash2, MTP). Not yet ported.
//
// Call sites for the absent groups still name *_cuda directly and are guarded by
// their existing runtime checks, which is why this header shrinks the surface
// rather than mirroring qwen_cuda_ops.hpp one-for-one.

#include <cstdint>

#include "qwen_cuda_ops.hpp"
#ifdef POCKET_BACKEND_ASCEND
#include "qwen_ascend_ops.hpp"
#endif

namespace pocket {

#ifndef POCKET_BACKEND_ASCEND
// The legacy row-wise argmax is shared with the DeepSeek-V4 path and is declared in
// cuda_ops.hpp when that header is included. Keep a no-default declaration here
// for Qwen-only translation units that do not include the legacy header.
bool argmax_fp32_rows_cuda(const float* d_logits, int* d_tokens,
                           float* d_logits_out, int rows, int count,
                           int token_offset, void* stream);
#endif

inline bool qwen_add_inplace_f16(uint16_t* d_y_fp16, const uint16_t* d_x_fp16, int count, void* stream = nullptr) {
#ifdef POCKET_BACKEND_ASCEND
    return qwen_add_inplace_f16_ascend(d_y_fp16, d_x_fp16, count, stream);
#else
    return qwen_add_inplace_f16_cuda(d_y_fp16, d_x_fp16, count, stream);
#endif
}

// Greedy row-wise top-1 over FP32 logits. The CUDA symbol is spelled without the
// qwen_ prefix because it predates the Qwen engine and is shared with the legacy
// DeepSeek-V4 path; the neutral name follows this header's convention.
inline bool qwen_argmax_fp32_rows(const float* d_logits, int* d_tokens, float* d_logits_out, int rows, int count, int token_offset, void* stream = nullptr) {
#ifdef POCKET_BACKEND_ASCEND
    return qwen_argmax_fp32_rows_ascend(d_logits, d_tokens, d_logits_out, rows, count, token_offset, stream);
#else
    return argmax_fp32_rows_cuda(d_logits, d_tokens, d_logits_out, rows, count, token_offset, stream);
#endif
}

inline bool qwen_append_kv_cache_f16(const uint16_t* d_k_rows_fp16, const uint16_t* d_v_rows_fp16, uint16_t* d_k_cache_fp16, uint16_t* d_v_cache_fp16, int seq_len, int kv_heads, int head_dim, int start_pos, int max_context, void* stream = nullptr) {
#ifdef POCKET_BACKEND_ASCEND
    return qwen_append_kv_cache_f16_ascend(d_k_rows_fp16, d_v_rows_fp16, d_k_cache_fp16, d_v_cache_fp16, seq_len, kv_heads, head_dim, start_pos, max_context, stream);
#else
    return qwen_append_kv_cache_f16_cuda(d_k_rows_fp16, d_v_rows_fp16, d_k_cache_fp16, d_v_cache_fp16, seq_len, kv_heads, head_dim, start_pos, max_context, stream);
#endif
}

inline bool qwen_causal_depthwise_conv_silu_f16(const uint16_t* d_x_fp16, const uint16_t* d_weight_fp16, uint16_t* d_tail_fp16, uint16_t* d_y_fp16, int seq_len, int channels, int kernel, bool update_tail, void* stream = nullptr) {
#ifdef POCKET_BACKEND_ASCEND
    return qwen_causal_depthwise_conv_silu_f16_ascend(d_x_fp16, d_weight_fp16, d_tail_fp16, d_y_fp16, seq_len, channels, kernel, update_tail, stream);
#else
    return qwen_causal_depthwise_conv_silu_f16_cuda(d_x_fp16, d_weight_fp16, d_tail_fp16, d_y_fp16, seq_len, channels, kernel, update_tail, stream);
#endif
}

inline bool qwen_concat_rows_f16(const uint16_t* d_left, const uint16_t* d_right, uint16_t* d_out, int rows, int cols, void* stream = nullptr) {
#ifdef POCKET_BACKEND_ASCEND
    return qwen_concat_rows_f16_ascend(d_left, d_right, d_out, rows, cols, stream);
#else
    return qwen_concat_rows_f16_cuda(d_left, d_right, d_out, rows, cols, stream);
#endif
}

inline bool qwen_copy_rows_strided_f16(const uint16_t* d_source_fp16, int source_row_stride, uint16_t* d_destination_fp16, int destination_row_stride, int rows, int columns, void* stream = nullptr) {
#ifdef POCKET_BACKEND_ASCEND
    return qwen_copy_rows_strided_f16_ascend(d_source_fp16, source_row_stride, d_destination_fp16, destination_row_stride, rows, columns, stream);
#else
    return qwen_copy_rows_strided_f16_cuda(d_source_fp16, source_row_stride, d_destination_fp16, destination_row_stride, rows, columns, stream);
#endif
}

inline bool qwen_embedding_fp16_gather_f16(const uint16_t* d_table_fp16, const int* d_tokens, uint16_t* d_out_fp16, int count, int cols, int row_start, int row_count, void* stream = nullptr) {
#ifdef POCKET_BACKEND_ASCEND
    return qwen_embedding_fp16_gather_f16_ascend(d_table_fp16, d_tokens, d_out_fp16, count, cols, row_start, row_count, stream);
#else
    return qwen_embedding_fp16_gather_f16_cuda(d_table_fp16, d_tokens, d_out_fp16, count, cols, row_start, row_count, stream);
#endif
}

inline bool qwen_fp16_matmul_rows_f16(const uint16_t* d_x_fp16, const uint16_t* d_w_fp16, uint16_t* d_y_fp16, int batch, int rows, int cols, int x_stride, int y_stride, int weight_stride, void* stream = nullptr) {
#ifdef POCKET_BACKEND_ASCEND
    return qwen_fp16_matmul_rows_f16_ascend(d_x_fp16, d_w_fp16, d_y_fp16, batch, rows, cols, x_stride, y_stride, weight_stride, stream);
#else
    return qwen_fp16_matmul_rows_f16_cuda(d_x_fp16, d_w_fp16, d_y_fp16, batch, rows, cols, x_stride, y_stride, weight_stride, stream);
#endif
}

inline bool qwen_fp16_matmul_rows_f16_f32(const uint16_t* d_x_fp16, const uint16_t* d_w_fp16, float* d_y, int batch, int rows, int cols, int x_stride, int y_stride, int weight_stride, void* stream = nullptr) {
#ifdef POCKET_BACKEND_ASCEND
    return qwen_fp16_matmul_rows_f16_f32_ascend(d_x_fp16, d_w_fp16, d_y, batch, rows, cols, x_stride, y_stride, weight_stride, stream);
#else
    return qwen_fp16_matmul_rows_f16_f32_cuda(d_x_fp16, d_w_fp16, d_y, batch, rows, cols, x_stride, y_stride, weight_stride, stream);
#endif
}

inline bool qwen_fp16_swiglu_matmul_rows_f16(const uint16_t* d_x_fp16, const uint16_t* d_gate_fp16, const uint16_t* d_up_fp16, uint16_t* d_y_fp16, int batch, int rows, int cols, int x_stride, int y_stride, int weight_stride, void* stream = nullptr) {
#ifdef POCKET_BACKEND_ASCEND
    return qwen_fp16_swiglu_matmul_rows_f16_ascend(d_x_fp16, d_gate_fp16, d_up_fp16, d_y_fp16, batch, rows, cols, x_stride, y_stride, weight_stride, stream);
#else
    return qwen_fp16_swiglu_matmul_rows_f16_cuda(d_x_fp16, d_gate_fp16, d_up_fp16, d_y_fp16, batch, rows, cols, x_stride, y_stride, weight_stride, stream);
#endif
}

inline bool qwen_gated_delta_sequence_f16(float* d_state, const uint16_t* d_q_fp16, const uint16_t* d_k_fp16, const uint16_t* d_v_fp16, const uint16_t* d_g_fp16, const uint16_t* d_beta_fp16, uint16_t* d_out_fp16, int rows, int heads, int key_heads, int key_dim, int value_dim, float q_scale, void* stream = nullptr) {
#ifdef POCKET_BACKEND_ASCEND
    return qwen_gated_delta_sequence_f16_ascend(d_state, d_q_fp16, d_k_fp16, d_v_fp16, d_g_fp16, d_beta_fp16, d_out_fp16, rows, heads, key_heads, key_dim, value_dim, q_scale, stream);
#else
    return qwen_gated_delta_sequence_f16_cuda(d_state, d_q_fp16, d_k_fp16, d_v_fp16, d_g_fp16, d_beta_fp16, d_out_fp16, rows, heads, key_heads, key_dim, value_dim, q_scale, stream);
#endif
}

inline bool qwen_gated_delta_sequence_normalized_f16(float* d_state, const float* d_q_normalized, const float* d_k_normalized, const uint16_t* d_v_fp16, const uint16_t* d_g_fp16, const uint16_t* d_beta_fp16, uint16_t* d_out_fp16, int rows, int heads, int key_heads, int key_dim, int value_dim, float q_scale, void* stream = nullptr) {
#ifdef POCKET_BACKEND_ASCEND
    return qwen_gated_delta_sequence_normalized_f16_ascend(d_state, d_q_normalized, d_k_normalized, d_v_fp16, d_g_fp16, d_beta_fp16, d_out_fp16, rows, heads, key_heads, key_dim, value_dim, q_scale, stream);
#else
    return qwen_gated_delta_sequence_normalized_f16_cuda(d_state, d_q_normalized, d_k_normalized, d_v_fp16, d_g_fp16, d_beta_fp16, d_out_fp16, rows, heads, key_heads, key_dim, value_dim, q_scale, stream);
#endif
}

inline bool qwen_gated_delta_sequence_normalized_shared_f16(float* d_state, const float* d_q_normalized, const float* d_k_normalized, const uint16_t* d_v_fp16, const uint16_t* d_g_fp16, const uint16_t* d_beta_fp16, uint16_t* d_out_fp16, int rows, int heads, int key_heads, int key_dim, int value_dim, float q_scale, void* stream = nullptr) {
#ifdef POCKET_BACKEND_ASCEND
    return qwen_gated_delta_sequence_normalized_shared_f16_ascend(d_state, d_q_normalized, d_k_normalized, d_v_fp16, d_g_fp16, d_beta_fp16, d_out_fp16, rows, heads, key_heads, key_dim, value_dim, q_scale, stream);
#else
    return qwen_gated_delta_sequence_normalized_shared_f16_cuda(d_state, d_q_normalized, d_k_normalized, d_v_fp16, d_g_fp16, d_beta_fp16, d_out_fp16, rows, heads, key_heads, key_dim, value_dim, q_scale, stream);
#endif
}

inline bool qwen_gated_delta_step_f16(float* d_state, const uint16_t* d_q_fp16, const uint16_t* d_k_fp16, const uint16_t* d_v_fp16, const uint16_t* d_g_fp16, const uint16_t* d_beta_fp16, uint16_t* d_out_fp16, int heads, int key_heads, int key_dim, int value_dim, float q_scale, void* stream = nullptr) {
#ifdef POCKET_BACKEND_ASCEND
    return qwen_gated_delta_step_f16_ascend(d_state, d_q_fp16, d_k_fp16, d_v_fp16, d_g_fp16, d_beta_fp16, d_out_fp16, heads, key_heads, key_dim, value_dim, q_scale, stream);
#else
    return qwen_gated_delta_step_f16_cuda(d_state, d_q_fp16, d_k_fp16, d_v_fp16, d_g_fp16, d_beta_fp16, d_out_fp16, heads, key_heads, key_dim, value_dim, q_scale, stream);
#endif
}

inline bool qwen_gated_rmsnorm_fp16_gamma_rows_f16(const uint16_t* d_x_fp16, const uint16_t* d_gamma_fp16, const uint16_t* d_gate_fp16, uint16_t* d_y_fp16, int rows, int cols, float eps, void* stream = nullptr) {
#ifdef POCKET_BACKEND_ASCEND
    return qwen_gated_rmsnorm_fp16_gamma_rows_f16_ascend(d_x_fp16, d_gamma_fp16, d_gate_fp16, d_y_fp16, rows, cols, eps, stream);
#else
    return qwen_gated_rmsnorm_fp16_gamma_rows_f16_cuda(d_x_fp16, d_gamma_fp16, d_gate_fp16, d_y_fp16, rows, cols, eps, stream);
#endif
}

inline bool qwen_gather_copy_regions(const QwenCopyRegion* d_regions, int region_count, uint8_t* d_packed, uint64_t total_blocks, void* stream = nullptr) {
#ifdef POCKET_BACKEND_ASCEND
    return qwen_gather_copy_regions_ascend(d_regions, region_count, d_packed, total_blocks, stream);
#else
    return qwen_gather_copy_regions_cuda(d_regions, region_count, d_packed, total_blocks, stream);
#endif
}

inline bool qwen_gqa_decode_attention_f16(const uint16_t* d_q_fp16, const uint16_t* d_k_cache_fp16, const uint16_t* d_v_cache_fp16, uint16_t* d_out_fp16, float* d_score_scratch, int q_heads, int kv_heads, int head_dim, int context_len, int max_context, void* stream = nullptr) {
#ifdef POCKET_BACKEND_ASCEND
    return qwen_gqa_decode_attention_f16_ascend(d_q_fp16, d_k_cache_fp16, d_v_cache_fp16, d_out_fp16, d_score_scratch, q_heads, kv_heads, head_dim, context_len, max_context, stream);
#else
    return qwen_gqa_decode_attention_f16_cuda(d_q_fp16, d_k_cache_fp16, d_v_cache_fp16, d_out_fp16, d_score_scratch, q_heads, kv_heads, head_dim, context_len, max_context, stream);
#endif
}

// Streams tile_count 64x256 FP16 tiles out of HBM and does one op per element.
// This is a measurement tool, not part of any model path: it establishes the
// achievable read bandwidth that the bandwidth-bound decode attention kernels are
// compared against. Ascend only.
inline bool qwen_hbm_read_probe(const uint16_t* d_source, uint16_t* d_sink, int tile_count, void* stream = nullptr) {
#ifdef POCKET_BACKEND_ASCEND
    return qwen_hbm_read_probe_ascend(d_source, d_sink, tile_count, stream);
#else
    (void)d_source; (void)d_sink; (void)tile_count; (void)stream;
    return false;
#endif
}

inline bool qwen_gqa_prefill_attention_f16(const uint16_t* d_q_rows_fp16, const uint16_t* d_k_cache_fp16, const uint16_t* d_v_cache_fp16, uint16_t* d_out_rows_fp16, int seq_len, int q_heads, int kv_heads, int head_dim, int position_offset, int max_context, void* stream = nullptr) {
#ifdef POCKET_BACKEND_ASCEND
    return qwen_gqa_prefill_attention_f16_ascend(d_q_rows_fp16, d_k_cache_fp16, d_v_cache_fp16, d_out_rows_fp16, seq_len, q_heads, kv_heads, head_dim, position_offset, max_context, stream);
#else
    return qwen_gqa_prefill_attention_f16_cuda(d_q_rows_fp16, d_k_cache_fp16, d_v_cache_fp16, d_out_rows_fp16, seq_len, q_heads, kv_heads, head_dim, position_offset, max_context, stream);
#endif
}

inline bool qwen_gqa_verify_attention_f16(const uint16_t* d_q_rows_fp16, const uint16_t* d_k_cache_fp16, const uint16_t* d_v_cache_fp16, uint16_t* d_out_rows_fp16, float* d_partial_scratch, int rows, int q_heads, int kv_heads, int head_dim, int position_offset, int max_context, int splits, void* stream = nullptr) {
#ifdef POCKET_BACKEND_ASCEND
    return qwen_gqa_verify_attention_f16_ascend(d_q_rows_fp16, d_k_cache_fp16, d_v_cache_fp16, d_out_rows_fp16, d_partial_scratch, rows, q_heads, kv_heads, head_dim, position_offset, max_context, splits, stream);
#else
    return qwen_gqa_verify_attention_f16_cuda(d_q_rows_fp16, d_k_cache_fp16, d_v_cache_fp16, d_out_rows_fp16, d_partial_scratch, rows, q_heads, kv_heads, head_dim, position_offset, max_context, splits, stream);
#endif
}

inline bool qwen_linear_attn_gates_f16(const uint16_t* d_a_fp16, const uint16_t* d_b_fp16, const uint16_t* d_a_log_fp16, const uint16_t* d_dt_bias_fp16, uint16_t* d_g_fp16, uint16_t* d_beta_fp16, int rows, int heads, void* stream = nullptr) {
#ifdef POCKET_BACKEND_ASCEND
    return qwen_linear_attn_gates_f16_ascend(d_a_fp16, d_b_fp16, d_a_log_fp16, d_dt_bias_fp16, d_g_fp16, d_beta_fp16, rows, heads, stream);
#else
    return qwen_linear_attn_gates_f16_cuda(d_a_fp16, d_b_fp16, d_a_log_fp16, d_dt_bias_fp16, d_g_fp16, d_beta_fp16, rows, heads, stream);
#endif
}

inline bool qwen_normalize_gated_delta_qk_f16(const uint16_t* d_q_fp16, const uint16_t* d_k_fp16, float* d_q_normalized, float* d_k_normalized, int rows, int key_heads, int key_dim, void* stream = nullptr) {
#ifdef POCKET_BACKEND_ASCEND
    return qwen_normalize_gated_delta_qk_f16_ascend(d_q_fp16, d_k_fp16, d_q_normalized, d_k_normalized, rows, key_heads, key_dim, stream);
#else
    return qwen_normalize_gated_delta_qk_f16_cuda(d_q_fp16, d_k_fp16, d_q_normalized, d_k_normalized, rows, key_heads, key_dim, stream);
#endif
}

inline bool qwen_partial_rope_rows_f16(uint16_t* d_q_fp16, uint16_t* d_k_fp16, int start_position, int rows, int rotary_dim, float theta, int q_heads, int kv_heads, int head_dim, void* stream = nullptr) {
#ifdef POCKET_BACKEND_ASCEND
    return qwen_partial_rope_rows_f16_ascend(d_q_fp16, d_k_fp16, start_position, rows, rotary_dim, theta, q_heads, kv_heads, head_dim, stream);
#else
    return qwen_partial_rope_rows_f16_cuda(d_q_fp16, d_k_fp16, start_position, rows, rotary_dim, theta, q_heads, kv_heads, head_dim, stream);
#endif
}

inline bool qwen_residual_add_rmsnorm_fp16_gamma_rows_f16(const uint16_t* d_hidden_fp16, const uint16_t* d_delta_fp16, const uint16_t* d_gamma_fp16, uint16_t* d_residual_fp16, uint16_t* d_normalized_fp16, int rows, int cols, float eps, void* stream = nullptr) {
#ifdef POCKET_BACKEND_ASCEND
    return qwen_residual_add_rmsnorm_fp16_gamma_rows_f16_ascend(d_hidden_fp16, d_delta_fp16, d_gamma_fp16, d_residual_fp16, d_normalized_fp16, rows, cols, eps, stream);
#else
    return qwen_residual_add_rmsnorm_fp16_gamma_rows_f16_cuda(d_hidden_fp16, d_delta_fp16, d_gamma_fp16, d_residual_fp16, d_normalized_fp16, rows, cols, eps, stream);
#endif
}

inline bool qwen_rmsnorm_fp16_gamma_rows_f16(const uint16_t* d_x_fp16, const uint16_t* d_gamma_fp16, uint16_t* d_y_fp16, int rows, int cols, float eps, void* stream = nullptr) {
#ifdef POCKET_BACKEND_ASCEND
    return qwen_rmsnorm_fp16_gamma_rows_f16_ascend(d_x_fp16, d_gamma_fp16, d_y_fp16, rows, cols, eps, stream);
#else
    return qwen_rmsnorm_fp16_gamma_rows_f16_cuda(d_x_fp16, d_gamma_fp16, d_y_fp16, rows, cols, eps, stream);
#endif
}

inline bool qwen_scatter_copy_regions(const QwenCopyRegion* d_regions, int region_count, const uint8_t* d_packed, uint64_t total_blocks, void* stream = nullptr) {
#ifdef POCKET_BACKEND_ASCEND
    return qwen_scatter_copy_regions_ascend(d_regions, region_count, d_packed, total_blocks, stream);
#else
    return qwen_scatter_copy_regions_cuda(d_regions, region_count, d_packed, total_blocks, stream);
#endif
}

inline bool qwen_sigmoid_mul_f16(const uint16_t* d_x_fp16, const uint16_t* d_gate_fp16, uint16_t* d_y_fp16, int count, void* stream = nullptr) {
#ifdef POCKET_BACKEND_ASCEND
    return qwen_sigmoid_mul_f16_ascend(d_x_fp16, d_gate_fp16, d_y_fp16, count, stream);
#else
    return qwen_sigmoid_mul_f16_cuda(d_x_fp16, d_gate_fp16, d_y_fp16, count, stream);
#endif
}

inline bool qwen_silu_mul_rows_f16(const uint16_t* d_gate_fp16, const uint16_t* d_up_fp16, uint16_t* d_y_fp16, int rows, int cols, void* stream = nullptr) {
#ifdef POCKET_BACKEND_ASCEND
    return qwen_silu_mul_rows_f16_ascend(d_gate_fp16, d_up_fp16, d_y_fp16, rows, cols, stream);
#else
    return qwen_silu_mul_rows_f16_cuda(d_gate_fp16, d_up_fp16, d_y_fp16, rows, cols, stream);
#endif
}

inline bool qwen_split_packed_qkv_f16(const uint16_t* d_packed_fp16, uint16_t* d_q_fp16, uint16_t* d_k_fp16, uint16_t* d_v_fp16, int rows, int key_dim, int value_dim, void* stream = nullptr) {
#ifdef POCKET_BACKEND_ASCEND
    return qwen_split_packed_qkv_f16_ascend(d_packed_fp16, d_q_fp16, d_k_fp16, d_v_fp16, rows, key_dim, value_dim, stream);
#else
    return qwen_split_packed_qkv_f16_cuda(d_packed_fp16, d_q_fp16, d_k_fp16, d_v_fp16, rows, key_dim, value_dim, stream);
#endif
}

inline bool qwen_split_q_gate_f16(const uint16_t* d_q_proj_fp16, uint16_t* d_q_fp16, uint16_t* d_gate_fp16, int rows, int q_heads, int head_dim, void* stream = nullptr) {
#ifdef POCKET_BACKEND_ASCEND
    return qwen_split_q_gate_f16_ascend(d_q_proj_fp16, d_q_fp16, d_gate_fp16, rows, q_heads, head_dim, stream);
#else
    return qwen_split_q_gate_f16_cuda(d_q_proj_fp16, d_q_fp16, d_gate_fp16, rows, q_heads, head_dim, stream);
#endif
}

inline bool qwen_split_rows_pair_f16(const uint16_t* d_packed_fp16, uint16_t* d_first_fp16, uint16_t* d_second_fp16, int rows, int width, void* stream = nullptr) {
#ifdef POCKET_BACKEND_ASCEND
    return qwen_split_rows_pair_f16_ascend(d_packed_fp16, d_first_fp16, d_second_fp16, rows, width, stream);
#else
    return qwen_split_rows_pair_f16_cuda(d_packed_fp16, d_first_fp16, d_second_fp16, rows, width, stream);
#endif
}
}  // namespace pocket
