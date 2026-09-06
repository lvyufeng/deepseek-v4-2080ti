#pragma once

// AscendC / aclnn implementations of the neutral Qwen operator set. These are the
// declarations qwen_ops.hpp forwards to when POCKET_BACKEND_ASCEND is defined;
// the definitions live under backends/ascend/kernels.
//
// Contract, matching the CUDA side exactly so the forwarders stay trivial:
//   - Every uint16_t* buffer is FP16. Not BF16: first-generation 910 has no BF16
//     arithmetic, so weights are converted at load time.
//   - `stream` is an aclrtStream as void*. nullptr means the default stream.
//   - Return false on a launch or argument failure; the engine's require_launch
//     turns that into an exception naming the operator.
//   - Shapes are row-major and identical to the CUDA declarations.

#include <cstdint>

// QwenCopyRegion and kQwenCopyRegionBlockBytes are part of the shared operator
// vocabulary rather than anything CUDA-specific: the descriptor table is built on
// the host and the layout has to agree byte for byte across backends.
#include "qwen_cuda_ops.hpp"

namespace pocket {

bool qwen_add_inplace_f16_ascend(uint16_t* d_y_fp16, const uint16_t* d_x_fp16, int count, void* stream);

// Row-wise greedy top-1. Named for Qwen like the rest of this header even though
// the CUDA counterpart lives in cuda_ops.hpp as argmax_fp32_rows_cuda: the neutral
// forwarder in qwen_ops.hpp is what the engine calls, and it keeps the argmax with
// the operators the Qwen forward actually uses.
bool qwen_argmax_fp32_rows_ascend(const float* d_logits, int* d_tokens, float* d_logits_out, int rows, int count, int token_offset, void* stream);

bool qwen_append_kv_cache_f16_ascend(const uint16_t* d_k_rows_fp16, const uint16_t* d_v_rows_fp16, uint16_t* d_k_cache_fp16, uint16_t* d_v_cache_fp16, int seq_len, int kv_heads, int head_dim, int start_pos, int max_context, void* stream);

bool qwen_causal_depthwise_conv_silu_f16_ascend(const uint16_t* d_x_fp16, const uint16_t* d_weight_fp16, uint16_t* d_tail_fp16, uint16_t* d_y_fp16, int seq_len, int channels, int kernel, bool update_tail, void* stream);

bool qwen_concat_rows_f16_ascend(const uint16_t* d_left, const uint16_t* d_right, uint16_t* d_out, int rows, int cols, void* stream);

bool qwen_copy_rows_strided_f16_ascend(const uint16_t* d_source_fp16, int source_row_stride, uint16_t* d_destination_fp16, int destination_row_stride, int rows, int columns, void* stream);

bool qwen_embedding_fp16_gather_f16_ascend(const uint16_t* d_table_fp16, const int* d_tokens, uint16_t* d_out_fp16, int count, int cols, int row_start, int row_count, void* stream);

bool qwen_fp16_matmul_rows_f16_ascend(const uint16_t* d_x_fp16, const uint16_t* d_w_fp16, uint16_t* d_y_fp16, int batch, int rows, int cols, int x_stride, int y_stride, int weight_stride, void* stream);

bool qwen_fp16_matmul_rows_f16_f32_ascend(const uint16_t* d_x_fp16, const uint16_t* d_w_fp16, float* d_y, int batch, int rows, int cols, int x_stride, int y_stride, int weight_stride, void* stream);

bool qwen_fp16_swiglu_matmul_rows_f16_ascend(const uint16_t* d_x_fp16, const uint16_t* d_gate_fp16, const uint16_t* d_up_fp16, uint16_t* d_y_fp16, int batch, int rows, int cols, int x_stride, int y_stride, int weight_stride, void* stream);

bool qwen_gated_delta_sequence_f16_ascend(float* d_state, const uint16_t* d_q_fp16, const uint16_t* d_k_fp16, const uint16_t* d_v_fp16, const uint16_t* d_g_fp16, const uint16_t* d_beta_fp16, uint16_t* d_out_fp16, int rows, int heads, int key_heads, int key_dim, int value_dim, float q_scale, void* stream);

bool qwen_gated_delta_sequence_normalized_f16_ascend(float* d_state, const float* d_q_normalized, const float* d_k_normalized, const uint16_t* d_v_fp16, const uint16_t* d_g_fp16, const uint16_t* d_beta_fp16, uint16_t* d_out_fp16, int rows, int heads, int key_heads, int key_dim, int value_dim, float q_scale, void* stream);

bool qwen_gated_delta_sequence_normalized_shared_f16_ascend(float* d_state, const float* d_q_normalized, const float* d_k_normalized, const uint16_t* d_v_fp16, const uint16_t* d_g_fp16, const uint16_t* d_beta_fp16, uint16_t* d_out_fp16, int rows, int heads, int key_heads, int key_dim, int value_dim, float q_scale, void* stream);

bool qwen_gated_delta_step_f16_ascend(float* d_state, const uint16_t* d_q_fp16, const uint16_t* d_k_fp16, const uint16_t* d_v_fp16, const uint16_t* d_g_fp16, const uint16_t* d_beta_fp16, uint16_t* d_out_fp16, int heads, int key_heads, int key_dim, int value_dim, float q_scale, void* stream);

bool qwen_gated_rmsnorm_fp16_gamma_rows_f16_ascend(const uint16_t* d_x_fp16, const uint16_t* d_gamma_fp16, const uint16_t* d_gate_fp16, uint16_t* d_y_fp16, int rows, int cols, float eps, void* stream);

bool qwen_gather_copy_regions_ascend(const QwenCopyRegion* d_regions, int region_count, uint8_t* d_packed, uint64_t total_blocks, void* stream);

bool qwen_gqa_decode_attention_f16_ascend(const uint16_t* d_q_fp16, const uint16_t* d_k_cache_fp16, const uint16_t* d_v_cache_fp16, uint16_t* d_out_fp16, float* d_score_scratch, int q_heads, int kv_heads, int head_dim, int context_len, int max_context, void* stream);
bool qwen_hbm_read_probe_ascend(const uint16_t* d_source, uint16_t* d_sink, int tile_count, void* stream);

bool qwen_gqa_prefill_attention_f16_ascend(const uint16_t* d_q_rows_fp16, const uint16_t* d_k_cache_fp16, const uint16_t* d_v_cache_fp16, uint16_t* d_out_rows_fp16, int seq_len, int q_heads, int kv_heads, int head_dim, int position_offset, int max_context, void* stream);

bool qwen_gqa_verify_attention_f16_ascend(const uint16_t* d_q_rows_fp16, const uint16_t* d_k_cache_fp16, const uint16_t* d_v_cache_fp16, uint16_t* d_out_rows_fp16, float* d_partial_scratch, int rows, int q_heads, int kv_heads, int head_dim, int position_offset, int max_context, int splits, void* stream);

bool qwen_linear_attn_gates_f16_ascend(const uint16_t* d_a_fp16, const uint16_t* d_b_fp16, const uint16_t* d_a_log_fp16, const uint16_t* d_dt_bias_fp16, uint16_t* d_g_fp16, uint16_t* d_beta_fp16, int rows, int heads, void* stream);

bool qwen_normalize_gated_delta_qk_f16_ascend(const uint16_t* d_q_fp16, const uint16_t* d_k_fp16, float* d_q_normalized, float* d_k_normalized, int rows, int key_heads, int key_dim, void* stream);

bool qwen_partial_rope_rows_f16_ascend(uint16_t* d_q_fp16, uint16_t* d_k_fp16, int start_position, int rows, int rotary_dim, float theta, int q_heads, int kv_heads, int head_dim, void* stream);

bool qwen_residual_add_rmsnorm_fp16_gamma_rows_f16_ascend(const uint16_t* d_hidden_fp16, const uint16_t* d_delta_fp16, const uint16_t* d_gamma_fp16, uint16_t* d_residual_fp16, uint16_t* d_normalized_fp16, int rows, int cols, float eps, void* stream);

bool qwen_rmsnorm_fp16_gamma_rows_f16_ascend(const uint16_t* d_x_fp16, const uint16_t* d_gamma_fp16, uint16_t* d_y_fp16, int rows, int cols, float eps, void* stream);

bool qwen_scatter_copy_regions_ascend(const QwenCopyRegion* d_regions, int region_count, const uint8_t* d_packed, uint64_t total_blocks, void* stream);

bool qwen_sigmoid_mul_f16_ascend(const uint16_t* d_x_fp16, const uint16_t* d_gate_fp16, uint16_t* d_y_fp16, int count, void* stream);

bool qwen_silu_mul_rows_f16_ascend(const uint16_t* d_gate_fp16, const uint16_t* d_up_fp16, uint16_t* d_y_fp16, int rows, int cols, void* stream);

bool qwen_split_packed_qkv_f16_ascend(const uint16_t* d_packed_fp16, uint16_t* d_q_fp16, uint16_t* d_k_fp16, uint16_t* d_v_fp16, int rows, int key_dim, int value_dim, void* stream);

bool qwen_split_q_gate_f16_ascend(const uint16_t* d_q_proj_fp16, uint16_t* d_q_fp16, uint16_t* d_gate_fp16, int rows, int q_heads, int head_dim, void* stream);

bool qwen_split_rows_pair_f16_ascend(const uint16_t* d_packed_fp16, uint16_t* d_first_fp16, uint16_t* d_second_fp16, int rows, int width, void* stream);
}  // namespace pocket
