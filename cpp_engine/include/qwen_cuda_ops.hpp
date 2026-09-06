#pragma once

#include <cstddef>
#include <cstdint>

namespace pocket {

// Qwen DSpark uses standard Qwen3 RMSNorm affine weights (gamma directly),
// unlike the Qwen3.5 target runtime's (1 + weight) convention.
bool qwen_dspark_rmsnorm_f16_cuda(const uint16_t* d_x_fp16,
                                  const uint16_t* d_gamma_fp16,
                                  uint16_t* d_y_fp16, int rows, int cols,
                                  float eps, void* stream = nullptr);

// Apply full NeoX-style YaRN RoPE in-place to [rows, heads, head_dim]. The
// inverse-frequency table has head_dim/2 FP32 elements; attention_factor scales
// both cos and sin exactly like Transformers Qwen3RotaryEmbedding.
bool qwen_dspark_yarn_rope_f16_cuda(uint16_t* d_x_fp16,
                                    const float* d_inv_freqs, int rows,
                                    int heads, int head_dim,
                                    int start_position,
                                    float attention_factor,
                                    void* stream = nullptr);

// Non-causal dual-source block attention used by the fixed-width DSpark
// proposal: every query sees committed context [0, context_len) and all block
// K/V rows [0, block_rows).
bool qwen_dspark_dual_source_gqa_f16_cuda(
    const uint16_t* d_q_fp16, const uint16_t* d_context_k_fp16,
    const uint16_t* d_context_v_fp16, const uint16_t* d_block_k_fp16,
    const uint16_t* d_block_v_fp16, uint16_t* d_output_fp16,
    int block_rows, int q_heads, int kv_heads, int head_dim,
    int context_len, int max_context, void* stream = nullptr);

// In-place local-vocab Markov correction for one proposal row.
bool qwen_dspark_add_markov_bias_f32_cuda(
    float* d_logits, const uint16_t* d_markov_embedding_fp16,
    const uint16_t* d_markov_w2_fp16, int local_vocab, int markov_rank,
    void* stream = nullptr);

// Gather rows from a replicated FP16 table; unlike target TP embedding gather,
// every token must be locally present.
bool qwen_dspark_embedding_gather_f16_cuda(
    const uint16_t* d_table_fp16, const int* d_tokens,
    uint16_t* d_output_fp16, int count, int cols, int table_rows,
    void* stream = nullptr);

bool qwen_dspark_confidence_f16_cuda(
    const uint16_t* d_hidden_fp16,
    const uint16_t* d_markov_embeddings_fp16,
    const uint16_t* d_weight_fp16, const uint16_t* d_bias_fp16,
    float* d_confidence, int rows, int hidden_size, int markov_rank,
    void* stream = nullptr);

// DFlash2 draft primitives. Convolution and attention preserve the
// reference's block-local zero padding and non-causal block visibility.
bool qwen_dflash2_f16_to_f32_cuda(
    const uint16_t* d_input_fp16, float* d_output_f32, int count,
    void* stream = nullptr);
bool qwen_dflash2_rmsnorm_f32_f16_cuda(
    const float* d_input_f32, const uint16_t* d_gamma_fp16,
    uint16_t* d_output_fp16, int rows, int columns, float eps,
    void* stream = nullptr);
bool qwen_dflash2_add_f32_cuda(
    float* d_output_f32, const float* d_input_f32, int count,
    void* stream = nullptr);
bool qwen_dflash2_grouped_dynamic_conv_f16_cuda(
    const uint16_t* d_hidden_fp16, const uint16_t* d_dynamic_fp16,
    const uint16_t* d_base_fp16, uint16_t* d_output_fp16, int rows,
    int hidden_size, int groups, int group_size, int kernel_size,
    void* stream = nullptr);
bool qwen_dflash2_grouped_dynamic_conv_strided_f16_cuda(
    const uint16_t* d_hidden_fp16, const uint16_t* d_dynamic_fp16,
    const uint16_t* d_base_fp16, uint16_t* d_output_fp16, int rows,
    int hidden_size, int groups, int group_size, int kernel_size,
    int dynamic_row_stride, int dynamic_offset, void* stream = nullptr);
bool qwen_dflash2_grouped_dynamic_conv_strided_f32_cuda(
    const float* d_hidden_f32, const uint16_t* d_dynamic_fp16,
    const uint16_t* d_base_fp16, float* d_output_f32, int rows,
    int hidden_size, int groups, int group_size, int kernel_size,
    int dynamic_row_stride, int dynamic_offset, void* stream = nullptr);
bool qwen_dflash2_rmsnorm_heads_f16_cuda(
    const uint16_t* d_input_fp16, const uint16_t* d_gamma_fp16,
    uint16_t* d_output_fp16, int rows, int heads, int head_dim,
    float eps, void* stream = nullptr);
bool qwen_dflash2_rope_rows_f16_cuda(
    uint16_t* d_q_fp16, uint16_t* d_k_fp16, int rows, int q_heads,
    int kv_heads, int head_dim, int start_position, float theta,
    void* stream = nullptr);
bool qwen_dflash2_rope_k_rows_f16_cuda(
    uint16_t* d_k_fp16, int rows, int kv_heads, int head_dim,
    int start_position, float theta, void* stream = nullptr);
bool qwen_dflash2_attention_f16_cuda(
    const uint16_t* d_q_fp16, const uint16_t* d_context_k_fp16,
    const uint16_t* d_context_v_fp16, const uint16_t* d_block_k_fp16,
    const uint16_t* d_block_v_fp16, uint16_t* d_output_fp16, int block_rows,
    int q_heads, int kv_heads, int head_dim, int context_len, int max_context,
    int sliding_window, void* stream = nullptr);
bool qwen_dflash2_attention_grouped_f16_cuda(
    const uint16_t* d_q_fp16, const uint16_t* d_context_k_fp16,
    const uint16_t* d_context_v_fp16, const uint16_t* d_block_k_fp16,
    const uint16_t* d_block_v_fp16, uint16_t* d_output_fp16, int block_rows,
    int q_heads, int kv_heads, int head_dim, int context_len, int max_context,
    int sliding_window, void* stream = nullptr);
bool qwen_dflash2_local_topk_f32_cuda(
    const float* d_logits, int* d_tokens, float* d_values, int rows,
    int vocab, int vocab_start, int top_k, void* stream = nullptr);
// Split variant. The single-block-per-row kernel launches only `rows` blocks, so
// on a 68-SM device the seven DFlash2 rows leave the GPU ~90% idle while each
// thread carries a spilling top_k register array. This partitions each row's
// vocabulary across `splits` blocks, then merges the partial lists with the same
// descending-logit / ascending-token comparator, so the result is bit-identical.
// `d_partial_tokens` and `d_partial_values` must hold rows * splits * top_k.
bool qwen_dflash2_local_topk_split_f32_cuda(
    const float* d_logits, int* d_partial_tokens, float* d_partial_values,
    int* d_tokens, float* d_values, int rows, int vocab, int vocab_start,
    int top_k, int splits, void* stream = nullptr);
// Merge the world-major NCCL all-gathered local top-k rows on device. The
// comparator is identical to local_topk: descending logit, ascending token.
bool qwen_dflash2_merge_topk_f32_cuda(
    const int* d_gathered_tokens, const float* d_gathered_logits,
    int* d_tokens, float* d_values, int world, int rows, int top_k,
    void* stream = nullptr);
// Pack/unpack one deterministic global top-1 candidate per row. The packed key
// is suitable for NCCL uint64 Max: larger logit wins, equal logit picks the
// smaller token id, and NaN loses to every finite or -inf candidate.
bool qwen_dflash2_pack_top1_key_f32_cuda(
    const int* d_tokens, const float* d_values, uint64_t* d_keys, int rows,
    void* stream = nullptr);
bool qwen_dflash2_unpack_top1_key_f32_cuda(
    const uint64_t* d_keys, int* d_tokens, float* d_values, int rows,
    void* stream = nullptr);
bool qwen_dflash2_selector_path_f16_cuda(
    const uint16_t* d_hidden_fp16, const int* d_candidates,
    const float* d_unary, const uint16_t* d_predecessor,
    const uint16_t* d_successor, const uint16_t* d_projection, int* d_output,
    int rows, int vocab, int hidden_size, int rank, int top_k,
    int anchor_token, void* stream = nullptr);
bool qwen_dflash2_selector_path_projected_f16_cuda(
    const float* d_projected, const int* d_candidates,
    const float* d_unary, const uint16_t* d_predecessor,
    const uint16_t* d_successor, int* d_output, int rows, int vocab,
    int rank, int top_k, int anchor_token, void* stream = nullptr);
// Exact-MAP selector. The projected selector score is a first-order chain over
// positions -- unary(position, token) plus a pairwise term that depends only on
// (previous token, token) -- so the greedy left-to-right argmax can sacrifice a
// whole suffix for a small gain at one position. This runs Viterbi over the same
// scores and returns the highest-scoring complete path. The score function, its
// FP32 accumulation order and the descending-score / ascending-token tie-break
// are identical to the greedy kernel, so with top_k=1 the two agree exactly.
// Draft content is a heuristic the target re-verifies, so a different path can
// only change acceptance, never emitted tokens.
bool qwen_dflash2_selector_path_viterbi_f16_cuda(
    const float* d_projected, const int* d_candidates,
    const float* d_unary, const uint16_t* d_predecessor,
    const uint16_t* d_successor, int* d_output, int rows, int vocab,
    int rank, int top_k, int anchor_token, void* stream = nullptr);
bool qwen_dflash2_selector_project_f16_cuda(
    const uint16_t* d_hidden_fp16, const uint16_t* d_projection_fp16,
    float* d_output, int rows, int hidden_size, int rank,
    void* stream = nullptr);



// Qwen FP8 weights use E4M3 codes and 128x128 block scales. On Turing,
// checkpoint BF16 scales are converted to IEEE FP16 before device upload.
// These kernels decode each code at the point of use; no expanded copy of the
// weight matrix is allocated.
bool qwen_fp8_e4m3_fp16scale_matvec_cuda(
    const float* d_x,
    const uint8_t* d_weight,
    const uint16_t* d_scale_fp16,
    float* d_y,
    int rows,
    int cols,
    int weight_stride,
    int scale_stride,
    void* stream = nullptr);

// Decode-only fused gate/up projection. The two matvecs retain independent FP32
// accumulation, then write silu(gate) * up without materializing either input.
bool qwen_fp8_e4m3_fp16scale_swiglu_matvec_cuda(
    const float* d_x,
    const uint8_t* d_gate_weight,
    const uint16_t* d_gate_scale_fp16,
    const uint8_t* d_up_weight,
    const uint16_t* d_up_scale_fp16,
    float* d_y,
    int rows,
    int cols,
    int weight_stride,
    int scale_stride,
    void* stream = nullptr);

bool qwen_fp8_e4m3_fp16scale_matmul_rows_cuda(
    const float* d_x,
    const uint8_t* d_weight,
    const uint16_t* d_scale_fp16,
    float* d_y,
    int batch,
    int rows,
    int cols,
    int x_stride,
    int y_stride,
    int weight_stride,
    int scale_stride,
    void* stream = nullptr);

// FP16 activation-storage variants. Dot products and normalization statistics
// still accumulate in FP32; only global activation traffic and workspace storage
// use IEEE FP16.
bool qwen_fp8_e4m3_fp16scale_matvec_f16_cuda(
    const uint16_t* d_x_fp16,
    const uint8_t* d_weight,
    const uint16_t* d_scale_fp16,
    uint16_t* d_y_fp16,
    int rows,
    int cols,
    int weight_stride,
    int scale_stride,
    void* stream = nullptr);

bool qwen_fp8_e4m3_fp16scale_matvec_dual_f16_cuda(
    const uint16_t* d_x_fp16,
    const uint8_t* d_first_weight,
    const uint16_t* d_first_scale_fp16,
    uint16_t* d_first_y_fp16,
    int first_rows,
    int first_weight_stride,
    int first_scale_stride,
    const uint8_t* d_second_weight,
    const uint16_t* d_second_scale_fp16,
    uint16_t* d_second_y_fp16,
    int second_rows,
    int second_weight_stride,
    int second_scale_stride,
    int cols,
    void* stream = nullptr);

bool qwen_fp8_e4m3_fp16scale_matvec_triple_f16_cuda(
    const uint16_t* d_x_fp16,
    const uint8_t* d_first_weight,
    const uint16_t* d_first_scale_fp16,
    uint16_t* d_first_y_fp16,
    int first_rows,
    int first_weight_stride,
    int first_scale_stride,
    const uint8_t* d_second_weight,
    const uint16_t* d_second_scale_fp16,
    uint16_t* d_second_y_fp16,
    int second_rows,
    int second_weight_stride,
    int second_scale_stride,
    const uint8_t* d_third_weight,
    const uint16_t* d_third_scale_fp16,
    uint16_t* d_third_y_fp16,
    int third_rows,
    int third_weight_stride,
    int third_scale_stride,
    int cols,
    void* stream = nullptr);

bool qwen_fp8_e4m3_fp16scale_swiglu_matvec_f16_cuda(
    const uint16_t* d_x_fp16,
    const uint8_t* d_gate_weight,
    const uint16_t* d_gate_scale_fp16,
    const uint8_t* d_up_weight,
    const uint16_t* d_up_scale_fp16,
    uint16_t* d_y_fp16,
    int rows,
    int cols,
    int weight_stride,
    int scale_stride,
    void* stream = nullptr);

bool qwen_fp8_e4m3_fp16scale_swiglu_small_batch_f16_cuda(
    const uint16_t* d_x_fp16,
    const uint8_t* d_gate_weight,
    const uint16_t* d_gate_scale_fp16,
    const uint8_t* d_up_weight,
    const uint16_t* d_up_scale_fp16,
    uint16_t* d_y_fp16,
    int batch,
    int rows,
    int cols,
    int x_stride,
    int y_stride,
    int weight_stride,
    int scale_stride,
    void* stream = nullptr);

bool qwen_fp8_e4m3_fp16scale_matmul_rows_f16_cuda(
    const uint16_t* d_x_fp16,
    const uint8_t* d_weight,
    const uint16_t* d_scale_fp16,
    uint16_t* d_y_fp16,
    int batch,
    int rows,
    int cols,
    int x_stride,
    int y_stride,
    int weight_stride,
    int scale_stride,
    void* stream = nullptr);

// Materialize a dense FP16 matrix from the checkpoint's raw FP8 codes and block
// scales. This is intended for long-lived verify caches; decode and prefill keep
// using the raw quantized storage paths.
bool qwen_fp8_e4m3_fp16scale_dequantize_f16_cuda(
    const uint8_t* d_weight, const uint16_t* d_scale_fp16,
    uint16_t* d_output_fp16, int rows, int cols, int weight_stride,
    int scale_stride, void* stream = nullptr);

// FP16 input/weight tensor-op GEMM with FP16 output. Weight storage is row-major
// [output_rows, columns], matching the target projection layout.
bool qwen_fp16_matmul_rows_f16_cublas_cuda(
    const uint16_t* d_x_fp16, const uint16_t* d_weight_fp16,
    uint16_t* d_y_fp16, int batch, int rows, int cols, int x_stride,
    int y_stride, int weight_stride, void* stream = nullptr);

// Per-output-channel FP8 used by compressed-tensors Qwen checkpoints. Scale
// lookup is scale[row], unlike the legacy 128x128 block-scale APIs above.
bool qwen_fp8_e4m3_channel_matvec_f16_cuda(
    const uint16_t* d_x_fp16,
    const uint8_t* d_weight,
    const uint16_t* d_scale_fp16,
    uint16_t* d_y_fp16,
    int rows,
    int cols,
    int weight_stride,
    void* stream = nullptr);

bool qwen_fp8_e4m3_channel_matmul_rows_f16_cuda(
    const uint16_t* d_x_fp16,
    const uint8_t* d_weight,
    const uint16_t* d_scale_fp16,
    uint16_t* d_y_fp16,
    int batch,
    int rows,
    int cols,
    int x_stride,
    int y_stride,
    int weight_stride,
    void* stream = nullptr);

bool qwen_fp8_e4m3_channel_matmul_rows_f16_f32_cuda(
    const uint16_t* d_x_fp16,
    const uint8_t* d_weight,
    const uint16_t* d_scale_fp16,
    float* d_y_fp32,
    int batch,
    int rows,
    int cols,
    int x_stride,
    int y_stride,
    int weight_stride,
    void* stream = nullptr);

// Qwen compressed-tensors NVFP4. `d_blocks` is a row-major array of 36-byte
// block64 records produced by qwen_materialize_nvfp4_host_linear(). The global
// factor is reciprocal(checkpoint weight_global_scale); input_global_scale is
// calibration metadata and is deliberately not applied by the SM75 fallback.
bool qwen_nvfp4_group16_matvec_f16_cuda(
    const uint16_t* d_x_fp16, const uint8_t* d_blocks,
    uint16_t* d_y_fp16, int rows, int cols, int blocks_per_row,
    float weight_global_factor, void* stream = nullptr);

bool qwen_nvfp4_group16_matmul_rows_f16_cuda(
    const uint16_t* d_x_fp16, const uint8_t* d_blocks,
    uint16_t* d_y_fp16, int batch, int rows, int cols,
    int x_stride, int y_stride, int blocks_per_row,
    float weight_global_factor, void* stream = nullptr);

bool qwen_nvfp4_group16_matmul_rows_f16_f32_cuda(
    const uint16_t* d_x_fp16, const uint8_t* d_blocks,
    float* d_y_fp32, int batch, int rows, int cols,
    int x_stride, int y_stride, int blocks_per_row,
    float weight_global_factor, void* stream = nullptr);

// SM75 decode/small-batch fallback. Activations are dynamically quantized to
// one symmetric Q8 scale per logical group of 32 values, then packed E2M1 codes
// are consumed by DP4A. `d_q8` holds batch*cols bytes and `d_q8_scale` holds
// batch*(cols/32) FP32 scales; callers may reuse the buffers across linears.
bool qwen_nvfp4_quantize_q8_group32_f16_cuda(
    const uint16_t* d_x_fp16, int8_t* d_q8, float* d_q8_scale,
    int batch, int cols, int x_stride, void* stream = nullptr);

bool qwen_nvfp4_group16_matmul_q8_f16_cuda(
    const int8_t* d_q8, const float* d_q8_scale,
    const uint8_t* d_blocks, uint16_t* d_y_fp16,
    int batch, int rows, int cols, int q8_stride, int y_stride,
    int blocks_per_row, float weight_global_factor,
    void* stream = nullptr);

bool qwen_nvfp4_group16_matmul_q8_f32_cuda(
    const int8_t* d_q8, const float* d_q8_scale,
    const uint8_t* d_blocks, float* d_y_fp32,
    int batch, int rows, int cols, int q8_stride, int y_stride,
    int blocks_per_row, float weight_global_factor,
    void* stream = nullptr);

// SM75 INT8 tensor-core prefill. Every K=16 MMA is rescaled independently so
// the checkpoint's per-output-row group-16 E4M3 scales remain exact relative to
// the shared Q8 activation representation.
bool qwen_nvfp4_group16_matmul_q8_wmma_f16_cuda(
    const int8_t* d_q8, const float* d_q8_scale,
    const uint8_t* d_blocks, uint16_t* d_y_fp16,
    int batch, int rows, int cols, int q8_stride, int y_stride,
    int blocks_per_row, float weight_global_factor,
    void* stream = nullptr);

bool qwen_nvfp4_group16_matmul_q8_wmma_f32_cuda(
    const int8_t* d_q8, const float* d_q8_scale,
    const uint8_t* d_blocks, float* d_y_fp32,
    int batch, int rows, int cols, int q8_stride, int y_stride,
    int blocks_per_row, float weight_global_factor,
    void* stream = nullptr);

// Wide prefill path for real prompt batches. A 128x64 block reuses each packed
// weight tile across 128 activation rows and consumes each K=16 subgroup with
// PRMT-unpacked INT8 DP4A. It keeps Qwen's group-16 scale boundaries exact while
// avoiding the per-K16 WMMA store/barrier sequence of the narrow fallback.
bool qwen_nvfp4_group16_matmul_q8_wide_n64_f16_cuda(
    const int8_t* d_q8, const float* d_q8_scale,
    const uint8_t* d_blocks, uint16_t* d_y_fp16,
    int batch, int rows, int cols, int q8_stride, int y_stride,
    int blocks_per_row, float weight_global_factor,
    void* stream = nullptr);

// Paired MLP prefill path. Gate and up share the same Q8 activation buffer and
// are computed in one WMMA launch before SiLU multiplication, avoiding a second
// dynamic quantization pass and the two materialized projection tensors.
bool qwen_nvfp4_group16_swiglu_q8_wmma_f16_cuda(
    const int8_t* d_q8, const float* d_q8_scale,
    const uint8_t* d_gate_blocks, float gate_weight_global_factor,
    const uint8_t* d_up_blocks, float up_weight_global_factor,
    uint16_t* d_y_fp16, int batch, int rows, int cols,
    int q8_stride, int y_stride, int blocks_per_row,
    void* stream = nullptr);

// FP16 input/weight tensor-op GEMM with FP32 output. This is kept separate from
// the exact reduction-order reference path and enabled only by runtime A/B gates.
bool qwen_fp16_matmul_rows_f16_f32_cublas_cuda(
    const uint16_t* d_x_fp16,
    const uint16_t* d_weight_fp16,
    float* d_y_fp32,
    int batch,
    int rows,
    int cols,
    int x_stride,
    int y_stride,
    int weight_stride,
    void* stream = nullptr);

// Explicit experimental prefill path used by correctness/performance A/B.
// It decodes only a block-local K tile and preserves FP32 accumulation.
bool qwen_fp8_e4m3_fp16scale_matmul_simt_cuda(
    const float* d_x,
    const uint8_t* d_weight,
    const uint16_t* d_scale_fp16,
    float* d_y,
    int batch,
    int rows,
    int cols,
    int x_stride,
    int y_stride,
    int weight_stride,
    int scale_stride,
    void* stream = nullptr);

// Compatibility/reference path for callers that still keep the scale in BF16
// bits. Qwen Turing loading should use the FP16-scale APIs above.
bool qwen_fp8_e4m3_bf16_matvec_cuda(
    const float* d_x,
    const uint8_t* d_weight,
    const uint16_t* d_scale_bf16,
    float* d_y,
    int rows,
    int cols,
    int weight_stride,
    int scale_stride,
    void* stream = nullptr);

bool qwen_fp8_e4m3_bf16_matmul_rows_cuda(
    const float* d_x,
    const uint8_t* d_weight,
    const uint16_t* d_scale_bf16,
    float* d_y,
    int batch,
    int rows,
    int cols,
    int x_stride,
    int y_stride,
    int weight_stride,
    int scale_stride,
    void* stream = nullptr);

// Qwen RMSNorm has a non-standard affine parameter convention: output is
// normalized * (1 + weight).  The gated variant deliberately has no +1.
bool qwen_rmsnorm_f32_cuda(const float* d_x, const float* d_weight,
                           float* d_y, int rows, int cols, float eps,
                           void* stream = nullptr);
bool qwen_gated_rmsnorm_f32_cuda(const float* d_x, const float* d_weight,
                                 const float* d_gate, float* d_y,
                                 int rows, int cols, float eps,
                                 void* stream = nullptr);
bool qwen_l2_norm_f32_cuda(const float* d_x, float* d_y, int rows, int cols,
                           float eps = 1.0e-6f, void* stream = nullptr);

// --- Dense FP16 weights (checkpoint BF16 materialized for Turing) -----------

// y[b, r] = sum_c x[b, c] * fp16(w[r, c]).
bool qwen_fp16_matmul_rows_cuda(const float* d_x, const uint16_t* d_w_fp16,
                                float* d_y, int batch, int rows, int cols,
                                int x_stride, int y_stride, int weight_stride,
                                void* stream = nullptr);

// Vocab-parallel embedding lookup. Tokens outside [row_start, row_start +
// row_count) produce a zero row so the caller can all-reduce the result.
bool qwen_embedding_fp16_gather_cuda(const uint16_t* d_table_fp16,
                                     const int* d_tokens, float* d_out,
                                     int count, int cols, int row_start,
                                     int row_count, void* stream = nullptr);

// RMSNorm variants whose affine parameter is stored as FP16 on device.
bool qwen_rmsnorm_fp16_gamma_rows_cuda(const float* d_x, const uint16_t* d_gamma_fp16,
                                       float* d_y, int rows, int cols, float eps,
                                       void* stream = nullptr);
bool qwen_gated_rmsnorm_fp16_gamma_rows_cuda(const float* d_x, const uint16_t* d_gamma_fp16,
                                             const float* d_gate, float* d_y,
                                             int rows, int cols, float eps,
                                             void* stream = nullptr);

// --- Gated DeltaNet linear attention ---------------------------------------

// Depthwise causal convolution over the packed QKV channels followed by SiLU.
// `d_tail` holds the previous kernel-1 inputs per channel and is updated in
// place when `update_tail` is set, which is what makes decode continue a prefill.
bool qwen_split_packed_qkv_cuda(const float* d_packed, float* d_q, float* d_k,
                                float* d_v, int rows, int key_dim, int value_dim,
                                void* stream = nullptr);

bool qwen_causal_depthwise_conv_silu_cuda(const float* d_x_rows,
                                          const uint16_t* d_weight_fp16,
                                          float* d_tail, float* d_y_rows,
                                          int seq_len, int channels, int kernel,
                                          bool update_tail, void* stream = nullptr);

// beta = sigmoid(b); g = -exp(A_log) * softplus(a + dt_bias).
bool qwen_linear_attn_gates_cuda(const float* d_a_rows, const float* d_b_rows,
                                 const uint16_t* d_a_log_fp16,
                                 const uint16_t* d_dt_bias_fp16,
                                 float* d_g_rows, float* d_beta_rows,
                                 int rows, int heads, void* stream = nullptr);

// One recurrent step of the gated delta rule over a persistent FP32 state
// [heads, key_dim, value_dim]. Query/key carry `key_heads` heads and are
// repeated `heads / key_heads` times to cover the value heads.
bool qwen_gated_delta_step_cuda(float* d_state, const float* d_q, const float* d_k,
                                const float* d_v, const float* d_g, const float* d_beta,
                                float* d_out, int heads, int key_heads, int key_dim,
                                int value_dim, float q_scale, void* stream = nullptr);

// Batched causal recurrent pass. Rows are processed strictly in order while
// the persistent state is updated in place; this is the prefill-only path.
bool qwen_gated_delta_sequence_cuda(float* d_state, const float* d_q, const float* d_k,
                                    const float* d_v, const float* d_g, const float* d_beta,
                                    float* d_out, int rows, int heads, int key_heads,
                                    int key_dim, int value_dim, float q_scale,
                                    void* stream = nullptr);

bool qwen_partial_rope_cuda(float* d_q, float* d_k, int position, int rotary_dim,
                            float theta, int q_heads, int kv_heads, int head_dim,
                            void* stream = nullptr);

// Prefill form of the above over `rows` consecutive positions in one launch.
bool qwen_partial_rope_rows_cuda(float* d_q, float* d_k, int start_position, int rows,
                                 int rotary_dim, float theta, int q_heads, int kv_heads,
                                 int head_dim, void* stream = nullptr);

// q_proj stores [q, gate] pairs per head; split them into contiguous matrices.
bool qwen_split_q_gate_cuda(const float* d_q_proj, float* d_q, float* d_gate,
                            int rows, int q_heads, int head_dim,
                            void* stream = nullptr);

// Select the rank-local KV head range from a replicated K/V projection.
bool qwen_select_kv_heads_cuda(const float* d_src, float* d_dst, int rows,
                               int total_kv_heads, int local_kv_heads,
                               int head_dim, int head_offset,
                               void* stream = nullptr);

// Append one or more tokens to a per-head contiguous KV cache.
bool qwen_append_kv_cache_cuda(const float* d_k_rows, const float* d_v_rows,
                               float* d_k_cache, float* d_v_cache, int seq_len,
                               int kv_heads, int head_dim, int start_pos,
                               int max_context, void* stream = nullptr);

// Single-token GQA attention against the cache. Query head h reads KV head
// h / (q_heads / kv_heads). d_score_scratch holds q_heads * context_len floats.
bool qwen_gqa_decode_attention_cuda(const float* d_q, const float* d_k_cache,
                                    const float* d_v_cache, float* d_out,
                                    float* d_score_scratch,
                                    int q_heads, int kv_heads, int head_dim,
                                    int context_len, int max_context,
                                    void* stream = nullptr);

// Multi-token causal GQA attention. Query row t is at absolute position
// `position_offset + t` and may attend to cache entries up to that position.
bool qwen_gqa_prefill_attention_cuda(const float* d_q_rows, const float* d_k_cache,
                                     const float* d_v_cache, float* d_out_rows,
                                     int seq_len, int q_heads, int kv_heads,
                                     int head_dim, int position_offset,
                                     int max_context, void* stream = nullptr);

// --- Small elementwise helpers ---------------------------------------------

bool qwen_sigmoid_mul_cuda(const float* d_x, const float* d_gate, float* d_y,
                           int count, void* stream = nullptr);
bool qwen_add_inplace_cuda(float* d_y, const float* d_x, int count,
                           void* stream = nullptr);
bool qwen_silu_mul_rows_cuda(const float* d_gate, const float* d_up, float* d_y,
                             int rows, int cols, void* stream = nullptr);

// --- FP16 activation-storage runtime ----------------------------------------

// cuBLAS-backed FP16 matrix product for large DSpark batches. Weight storage is
// row-major [output_rows, columns]; output is row-major [batch, output_rows].
bool qwen_dspark_fp16_gemm_rows_f16_cuda(
    const uint16_t* d_x_fp16, const uint16_t* d_weight_fp16,
    uint16_t* d_y_fp16, int batch, int output_rows, int columns,
    void* stream = nullptr);
bool qwen_fp16_matmul_rows_f16_cuda(const uint16_t* d_x_fp16,
                                    const uint16_t* d_w_fp16,
                                    uint16_t* d_y_fp16, int batch, int rows,
                                    int cols, int x_stride, int y_stride,
                                    int weight_stride, void* stream = nullptr);
bool qwen_fp16_matmul_rows_f16_f32_cuda(const uint16_t* d_x_fp16,
                                        const uint16_t* d_w_fp16,
                                        float* d_y, int batch, int rows,
                                        int cols, int x_stride, int y_stride,
                                        int weight_stride, void* stream = nullptr);
// Decode form of the call above. A warp owns two output rows and each lane takes
// eight consecutive columns, which turns the weight stream into 16-byte loads and
// replaces the per-row block reduction with a shuffle. Products and the reduction
// stay FP32, but each lane sums a strided column subset, so the result is close to
// the call above rather than bit-identical; greedy argmax can turn on a near tie,
// so callers select this explicitly instead of inheriting it. Returns false for
// anything other than a single row with 4-element-aligned strides, rather than
// falling back, so an A/B cannot silently measure the reference path.
bool qwen_fp16_matvec_rows_f16_f32_cuda(const uint16_t* d_x_fp16,
                                        const uint16_t* d_w_fp16,
                                        float* d_y, int batch, int rows,
                                        int cols, int x_stride, int y_stride,
                                        int weight_stride, void* stream = nullptr);
// Fused FP16 gate/up projections and SiLU multiplication for small batches.
// The output is [batch, rows] with the supplied row strides.
bool qwen_fp16_swiglu_matmul_rows_f16_cuda(
    const uint16_t* d_x_fp16, const uint16_t* d_gate_fp16,
    const uint16_t* d_up_fp16, uint16_t* d_y_fp16, int batch, int rows,
    int cols, int x_stride, int y_stride, int weight_stride,
    void* stream = nullptr);

bool qwen_embedding_fp16_gather_f16_cuda(const uint16_t* d_table_fp16,
                                         const int* d_tokens,
                                         uint16_t* d_out_fp16, int count,
                                         int cols, int row_start, int row_count,
                                         void* stream = nullptr);
bool qwen_concat_rows_f16_cuda(const uint16_t* d_left, const uint16_t* d_right,
                               uint16_t* d_out, int rows, int cols,
                               void* stream = nullptr);
// Copies a contiguous column range from every source row into a strided
// destination. Pitches and width are expressed in FP16 elements.
bool qwen_copy_rows_strided_f16_cuda(
    const uint16_t* d_source_fp16, int source_row_stride,
    uint16_t* d_destination_fp16, int destination_row_stride,
    int rows, int columns, void* stream = nullptr);

// Device-resident scatter/gather for independently allocated byte regions. The
// host builds a descriptor table once; every block copies one fixed-size chunk
// of a region to or from its offset in a contiguous transaction buffer.
struct QwenCopyRegion {
    uint64_t device_address = 0;
    uint64_t packed_offset = 0;
    uint64_t bytes = 0;
    uint64_t first_block = 0;
};

constexpr uint64_t kQwenCopyRegionBlockBytes = 64 * 1024;
constexpr uint64_t qwen_copy_region_blocks(uint64_t bytes) {
    return (bytes + kQwenCopyRegionBlockBytes - 1) /
           kQwenCopyRegionBlockBytes;
}

bool qwen_gather_copy_regions_cuda(
    const QwenCopyRegion* d_regions, int region_count, uint8_t* d_packed,
    uint64_t total_blocks, void* stream = nullptr);
bool qwen_scatter_copy_regions_cuda(
    const QwenCopyRegion* d_regions, int region_count, const uint8_t* d_packed,
    uint64_t total_blocks, void* stream = nullptr);

bool qwen_rmsnorm_fp16_gamma_rows_f16_cuda(const uint16_t* d_x_fp16,
                                           const uint16_t* d_gamma_fp16,
                                           uint16_t* d_y_fp16, int rows,
                                           int cols, float eps,
                                           void* stream = nullptr);
// Fuses the attention residual boundary with post-attention RMSNorm. The sum is
// rounded to FP16 before it contributes to the RMS statistic, exactly matching
// copy(hidden) -> FP16 add(attention) -> RMSNorm(residual).
bool qwen_residual_add_rmsnorm_fp16_gamma_rows_f16_cuda(
    const uint16_t* d_hidden_fp16, const uint16_t* d_delta_fp16,
    const uint16_t* d_gamma_fp16, uint16_t* d_residual_fp16,
    uint16_t* d_normalized_fp16, int rows, int cols, float eps,
    void* stream = nullptr);
bool qwen_gated_rmsnorm_fp16_gamma_rows_f16_cuda(
    const uint16_t* d_x_fp16, const uint16_t* d_gamma_fp16,
    const uint16_t* d_gate_fp16, uint16_t* d_y_fp16, int rows, int cols,
    float eps, void* stream = nullptr);
bool qwen_split_packed_qkv_f16_cuda(const uint16_t* d_packed_fp16,
                                    uint16_t* d_q_fp16, uint16_t* d_k_fp16,
                                    uint16_t* d_v_fp16, int rows,
                                    int key_dim, int value_dim,
                                    void* stream = nullptr);
// Splits a [rows, 2*width] row-major tensor into two [rows, width] tensors.
// Used to unpack a fused projection whose weight was row-concatenated.
bool qwen_split_rows_pair_f16_cuda(const uint16_t* d_packed_fp16,
                                   uint16_t* d_first_fp16,
                                   uint16_t* d_second_fp16, int rows,
                                   int width, void* stream = nullptr);
bool qwen_causal_depthwise_conv_silu_f16_cuda(
    const uint16_t* d_x_fp16, const uint16_t* d_weight_fp16,
    uint16_t* d_tail_fp16, uint16_t* d_y_fp16, int seq_len, int channels,
    int kernel, bool update_tail, void* stream = nullptr);
// Batched decode variant: `rows` sequences each contribute one new token and
// carry their own tail at `row * tail_slot_stride` elements. The tail is always
// updated, since a decode step by definition consumes its token.
bool qwen_causal_depthwise_conv_silu_f16_batched_cuda(
    const uint16_t* d_x_fp16, const uint16_t* d_weight_fp16,
    uint16_t* d_tail_fp16, uint16_t* d_y_fp16, const int* d_slot_ids, int rows,
    int channels, int kernel, size_t tail_slot_stride, void* stream = nullptr);
bool qwen_linear_attn_gates_f16_cuda(
    const uint16_t* d_a_fp16, const uint16_t* d_b_fp16,
    const uint16_t* d_a_log_fp16, const uint16_t* d_dt_bias_fp16,
    uint16_t* d_g_fp16, uint16_t* d_beta_fp16, int rows, int heads,
    void* stream = nullptr);
bool qwen_gated_delta_step_f16_cuda(
    float* d_state, const uint16_t* d_q_fp16, const uint16_t* d_k_fp16,
    const uint16_t* d_v_fp16, const uint16_t* d_g_fp16,
    const uint16_t* d_beta_fp16, uint16_t* d_out_fp16, int heads,
    int key_heads, int key_dim, int value_dim, float q_scale,
    void* stream = nullptr);
// Batched single step: `rows` sequences each advance one token against their own
// recurrent state slot at `row * state_slot_stride` elements. Q/K/V/gates and
// the output are row-major over rows.
// `d_slot_ids` maps each row to the state slot it owns; pass null when the rows
// occupy slots 0..rows-1 in order.
bool qwen_gated_delta_step_batched_f16_cuda(
    float* d_state, const uint16_t* d_q_fp16, const uint16_t* d_k_fp16,
    const uint16_t* d_v_fp16, const uint16_t* d_g_fp16,
    const uint16_t* d_beta_fp16, uint16_t* d_out_fp16, const int* d_slot_ids,
    int rows, int heads, int key_heads, int key_dim, int value_dim,
    float q_scale, size_t state_slot_stride, void* stream = nullptr);
bool qwen_gated_delta_sequence_f16_cuda(
    float* d_state, const uint16_t* d_q_fp16, const uint16_t* d_k_fp16,
    const uint16_t* d_v_fp16, const uint16_t* d_g_fp16,
    const uint16_t* d_beta_fp16, uint16_t* d_out_fp16, int rows,
    int heads, int key_heads, int key_dim, int value_dim, float q_scale,
    void* stream = nullptr);
// Same recurrence with Q/K already L2-normalized as FP32. Separating the
// normalization lets one key head serve all repeated value heads.
bool qwen_gated_delta_sequence_normalized_f16_cuda(
    float* d_state, const float* d_q_normalized, const float* d_k_normalized,
    const uint16_t* d_v_fp16, const uint16_t* d_g_fp16,
    const uint16_t* d_beta_fp16, uint16_t* d_out_fp16, int rows,
    int heads, int key_heads, int key_dim, int value_dim, float q_scale,
    void* stream = nullptr);
// Optional verify path that keeps the recurrent state in shared memory instead
// of a 128-element per-thread register vector. Set
// QWEN_GATED_DELTA_SHARED_STATE=1 to compare it with the fallback.
bool qwen_gated_delta_sequence_normalized_shared_f16_cuda(
    float* d_state, const float* d_q_normalized, const float* d_k_normalized,
    const uint16_t* d_v_fp16, const uint16_t* d_g_fp16,
    const uint16_t* d_beta_fp16, uint16_t* d_out_fp16, int rows,
    int heads, int key_heads, int key_dim, int value_dim, float q_scale,
    void* stream = nullptr);
bool qwen_normalize_gated_delta_qk_f16_cuda(
    const uint16_t* d_q_fp16, const uint16_t* d_k_fp16,
    float* d_q_normalized, float* d_k_normalized, int rows, int key_heads,
    int key_dim, void* stream = nullptr);
bool qwen_gated_delta_flashqla_sm75_f16_cuda(
    float* d_state, const float* d_q_normalized,
    const float* d_k_normalized, const uint16_t* d_v,
    const uint16_t* d_g, const uint16_t* d_beta, uint16_t* d_out,
    int rows, int heads, int key_heads, int key_dim, int value_dim,
    float q_scale, void* stream = nullptr);
bool qwen_partial_rope_f16_cuda(uint16_t* d_q_fp16, uint16_t* d_k_fp16,
                                int position, int rotary_dim, float theta,
                                int q_heads, int kv_heads, int head_dim,
                                void* stream = nullptr);
bool qwen_partial_rope_rows_f16_cuda(
    uint16_t* d_q_fp16, uint16_t* d_k_fp16, int start_position, int rows,
    int rotary_dim, float theta, int q_heads, int kv_heads, int head_dim,
    void* stream = nullptr);
// Batched variant: each row uses its own position from d_positions[row].
// When d_positions is null, falls back to start_position + row.
bool qwen_partial_rope_rows_f16_batched_cuda(
    uint16_t* d_q_fp16, uint16_t* d_k_fp16, const int* d_positions, int rows,
    int rotary_dim, float theta, int q_heads, int kv_heads, int head_dim,
    void* stream = nullptr);
bool qwen_split_q_gate_f16_cuda(const uint16_t* d_q_proj_fp16,
                                uint16_t* d_q_fp16,
                                uint16_t* d_gate_fp16, int rows,
                                int q_heads, int head_dim,
                                void* stream = nullptr);
bool qwen_sigmoid_mul_f16_cuda(const uint16_t* d_x_fp16,
                               const uint16_t* d_gate_fp16,
                               uint16_t* d_y_fp16, int count,
                               void* stream = nullptr);
bool qwen_add_inplace_f16_cuda(uint16_t* d_y_fp16,
                               const uint16_t* d_x_fp16, int count,
                               void* stream = nullptr);
bool qwen_silu_mul_rows_f16_cuda(const uint16_t* d_gate_fp16,
                                 const uint16_t* d_up_fp16,
                                 uint16_t* d_y_fp16, int rows, int cols,
                                 void* stream = nullptr);

// FP16 cache is the precision baseline. FP8 cache stores E4M3 codes plus one
// IEEE FP16 scale for each token/head/64-channel block.
bool qwen_append_kv_cache_f16_cuda(
    const uint16_t* d_k_rows_fp16, const uint16_t* d_v_rows_fp16,
    uint16_t* d_k_cache_fp16, uint16_t* d_v_cache_fp16, int seq_len,
    int kv_heads, int head_dim, int start_pos, int max_context,
    void* stream = nullptr);
// Batched variant: row `r` writes its K/V into its own cache slot at
// `r * kv_slot_stride` elements, at position d_start_positions[r] within that
// slot. Every sequence contributes exactly one token, so `rows` is the batch
// width rather than a sequence length.
// `d_slot_ids` maps each row to the cache slot it owns, because slots are
// allocated as requests arrive: a four-row batch can hold slots 1/3/5/6. Pass
// null when the rows occupy slots 0..rows-1 in order.
// A non-null `d_block_table` selects the paged cache: the arena is shared and
// `[max_slots, max_blocks_per_seq]` int32 block ids replace the per-slot stride,
// so position `p` of slot `s` lives at
// `(table[s][p / block_size] * block_size + p % block_size) * kv_heads *
// head_dim`. `kv_slot_stride` is then unused.
bool qwen_append_kv_cache_f16_batched_cuda(
    const uint16_t* d_k_rows_fp16, const uint16_t* d_v_rows_fp16,
    uint16_t* d_k_cache_fp16, uint16_t* d_v_cache_fp16, int rows,
    int kv_heads, int head_dim, const int* d_start_positions,
    const int* d_slot_ids, int max_context, size_t kv_slot_stride,
    const int* d_block_table = nullptr, int block_size = 0,
    int max_blocks_per_seq = 0, void* stream = nullptr);
// Paged single-sequence append. `d_block_table` is one sequence's row of block
// ids, so consecutive tokens land in whichever blocks that row names.
bool qwen_append_kv_cache_f16_paged_cuda(
    const uint16_t* d_k_rows_fp16, const uint16_t* d_v_rows_fp16,
    uint16_t* d_k_cache_fp16, uint16_t* d_v_cache_fp16, int seq_len,
    int kv_heads, int head_dim, int start_pos, const int* d_block_table,
    int block_size, void* stream = nullptr);
// Collects one sequence's paged history into a dense
// `[context_len, kv_heads, head_dim]` buffer, which is the layout every prefill
// and single-row decode kernel already indexes. Gathering once per layer keeps
// those five tuned variants unchanged, the same way the FP8 and TurboQuant
// caches dequantize once before calling them.
bool qwen_gather_kv_cache_f16_paged_cuda(
    const uint16_t* d_k_cache_fp16, const uint16_t* d_v_cache_fp16,
    uint16_t* d_k_dense_fp16, uint16_t* d_v_dense_fp16, int context_len,
    int kv_heads, int head_dim, const int* d_block_table, int block_size,
    void* stream = nullptr);
bool qwen_append_kv_cache_fp8_cuda(
    const uint16_t* d_k_rows_fp16, const uint16_t* d_v_rows_fp16,
    uint8_t* d_k_cache_fp8, uint8_t* d_v_cache_fp8,
    uint16_t* d_k_scale_fp16, uint16_t* d_v_scale_fp16, int seq_len,
    int kv_heads, int head_dim, int scale_block, int start_pos,
    int max_context, void* stream = nullptr);
// Batched variant for FP8 cache. `kv_slot_stride` is in cache elements; the
// scale array is strided by kv_slot_stride / scale_block to match.
bool qwen_append_kv_cache_fp8_batched_cuda(
    const uint16_t* d_k_rows_fp16, const uint16_t* d_v_rows_fp16,
    uint8_t* d_k_cache_fp8, uint8_t* d_v_cache_fp8,
    uint16_t* d_k_scale_fp16, uint16_t* d_v_scale_fp16, int rows,
    int kv_heads, int head_dim, int scale_block, const int* d_start_positions,
    const int* d_slot_ids, int max_context, size_t kv_slot_stride,
    void* stream = nullptr);
bool qwen_gqa_decode_attention_f16_cuda(
    const uint16_t* d_q_fp16, const uint16_t* d_k_cache_fp16,
    const uint16_t* d_v_cache_fp16, uint16_t* d_out_fp16,
    float* d_score_scratch, int q_heads, int kv_heads, int head_dim,
    int context_len, int max_context, void* stream = nullptr);
bool qwen_gqa_prefill_attention_f16_cuda(
    const uint16_t* d_q_rows_fp16, const uint16_t* d_k_cache_fp16,
    const uint16_t* d_v_cache_fp16, uint16_t* d_out_rows_fp16,
    int seq_len, int q_heads, int kv_heads, int head_dim,
    int position_offset, int max_context, void* stream = nullptr);
// Exact optimized paths. Decode reuses the existing score scratch for compact
// split partials; prefill shares each K/V load across GQA heads and query rows.
bool qwen_gqa_decode_attention_f16_fused_cuda(
    const uint16_t* d_q_fp16, const uint16_t* d_k_cache_fp16,
    const uint16_t* d_v_cache_fp16, uint16_t* d_out_fp16,
    float* d_partial_scratch, int q_heads, int kv_heads, int head_dim,
    int context_len, int max_context, int attention_window = 0,
    int sink_tokens = 0, void* stream = nullptr);
// Splits the fused decode kernel will use for this many attended positions. The
// caller must size `d_partial_scratch` as q_heads * splits * (head_dim + 2), so
// this has to be the same number the launch computes.
int qwen_gqa_decode_split_count(int attended_positions, int kv_heads = 1,
                                bool tensor_core_shape = false);

// Which split kernel the fused decode path runs. Production selects this from
// the environment once per process, which makes an in-process comparison of the
// two impossible; the explicit variant below exists so one test can drive both
// against the same inputs. `Selected` is exactly what the production entry point
// would do.
enum class QwenGqaDecodeVariant {
    Selected = 0,
    Scalar = 1,
    TensorCore = 2,
};

// Splits the given variant would use. Scalar and tensor-core geometry differ, so
// scratch has to be sized from the variant actually being launched.
int qwen_gqa_decode_split_count_variant(
    int attended_positions, QwenGqaDecodeVariant variant,
    int split_count_override = 0);

// Batched decode: one launch serves `rows` sequences, each reading its own KV
// slot at `row * kv_slot_stride` elements and attending its own context length
// from `d_context_lens`. Split geometry is derived from `max_context_len`, the
// longest row in the batch, so every row shares one grid. Q, output and the
// partial scratch are row-major over rows.
//
// Scratch must be sized rows * q_heads * splits * (head_dim + 2) using the
// split count below, which is the same number the launch computes.
int qwen_gqa_decode_batched_split_count(
    int max_context_len, int kv_heads, int attention_window = 0,
    int sink_tokens = 0);
// `d_slot_ids` maps each row to the KV slot it owns; pass null when the rows
// occupy slots 0..rows-1 in order.
// A non-null `d_block_table` selects the paged cache, translating each scanned
// position through the same block table the append above writes into. The
// translation is loop-invariant within a block, so it costs one table read per
// `block_size` positions rather than one per position. `max_context` then bounds
// only the logical sequence; the arena extent comes from the pool.
bool qwen_gqa_decode_attention_f16_batched_cuda(
    const uint16_t* d_q_fp16, const uint16_t* d_k_cache_fp16,
    const uint16_t* d_v_cache_fp16, uint16_t* d_out_fp16,
    float* d_partial_scratch, const int* d_context_lens,
    const int* d_slot_ids, int rows, int max_context_len,
    size_t kv_slot_stride, int q_heads, int kv_heads, int head_dim,
    int max_context, int attention_window = 0, int sink_tokens = 0,
    const int* d_block_table = nullptr, int block_size = 0,
    int max_blocks_per_seq = 0, void* stream = nullptr);

// Fused decode with the split kernel chosen explicitly. Returns false when the
// requested variant does not support the shape (the tensor-core kernel requires
// six Q heads per KV head, head_dim=256, and full causal attention),
// so a test cannot silently measure a fallback. A positive split override is a
// correctness-test seam: unlike production geometry, it preserves the requested
// count so trailing empty splits reach the kernel and merge.
bool qwen_gqa_decode_attention_f16_fused_variant_cuda(
    const uint16_t* d_q_fp16, const uint16_t* d_k_cache_fp16,
    const uint16_t* d_v_cache_fp16, uint16_t* d_out_fp16,
    float* d_partial_scratch, int q_heads, int kv_heads, int head_dim,
    int context_len, int max_context, int attention_window, int sink_tokens,
    QwenGqaDecodeVariant variant, int split_count_override = 0,
    void* stream = nullptr);
bool qwen_gqa_prefill_attention_f16_tiled_cuda(
    const uint16_t* d_q_rows_fp16, const uint16_t* d_k_cache_fp16,
    const uint16_t* d_v_cache_fp16, uint16_t* d_out_rows_fp16,
    int seq_len, int q_heads, int kv_heads, int head_dim,
    int position_offset, int max_context, int attention_window = 0,
    int sink_tokens = 0, void* stream = nullptr);
// Batched verifier preserving the reference decode reduction order. Each K row
// is shared across all candidate rows and the Q heads belonging to one KV head;
// scratch contains rows * q_heads * (position_offset + rows) FP32 scores.
bool qwen_gqa_verify_attention_f16_exact_cuda(
    const uint16_t* d_q_rows_fp16, const uint16_t* d_k_cache_fp16,
    const uint16_t* d_v_cache_fp16, uint16_t* d_out_rows_fp16,
    float* d_score_scratch, int rows, int q_heads, int kv_heads,
    int head_dim, int position_offset, int max_context,
    void* stream = nullptr);
// Experimental exact-full-attention path for speculative target verification.
// Context splits run in parallel while every K/V load serves every verify row
// in the CTA. Its online-softmax order may change near-tie downstream logits.
// Scratch contains rows * q_heads * splits * (head_dim + 2) FP32 elements.
// Tensor-core QK + FP32 masked softmax + deterministic SIMT PV. Optimized for
// TP shards with one KV head, where all local Q heads share a contiguous K/V
// matrix and QK becomes one large GEMM.
bool qwen_gqa_verify_attention_f16_cublas_qk_cuda(
    const uint16_t* d_q_fp16, const uint16_t* d_k_cache_fp16,
    const uint16_t* d_v_cache_fp16, uint16_t* d_output_fp16,
    float* d_score_scratch, int rows, int q_heads, int kv_heads,
    int head_dim, int position_offset, int max_context,
    void* stream = nullptr);

bool qwen_gqa_verify_attention_f16_cuda(
    const uint16_t* d_q_rows_fp16, const uint16_t* d_k_cache_fp16,
    const uint16_t* d_v_cache_fp16, uint16_t* d_out_rows_fp16,
    float* d_partial_scratch, int rows, int q_heads, int kv_heads,
    int head_dim, int position_offset, int max_context, int splits,
    void* stream = nullptr);
bool qwen_gqa_decode_attention_fp8_cuda(
    const uint16_t* d_q_fp16, const uint8_t* d_k_cache_fp8,
    const uint8_t* d_v_cache_fp8, const uint16_t* d_k_scale_fp16,
    const uint16_t* d_v_scale_fp16, uint16_t* d_out_fp16,
    float* d_score_scratch, int q_heads, int kv_heads, int head_dim,
    int scale_block, int context_len, int max_context,
    void* stream = nullptr);
bool qwen_gqa_prefill_attention_fp8_cuda(
    const uint16_t* d_q_rows_fp16, const uint8_t* d_k_cache_fp8,
    const uint8_t* d_v_cache_fp8, const uint16_t* d_k_scale_fp16,
    const uint16_t* d_v_scale_fp16, uint16_t* d_out_rows_fp16,
    int seq_len, int q_heads, int kv_heads, int head_dim, int scale_block,
    int position_offset, int max_context, void* stream = nullptr);

// Bulk dequantize FP8 KV cache to dense FP16. Used to amortize dequant cost
// before calling tensor-core prefill kernels on quantized KV history.
bool qwen_fp8_dequant_kv_cache_cuda(
    const uint8_t* d_k_cache_fp8, const uint8_t* d_v_cache_fp8,
    const uint16_t* d_k_scale_fp16, const uint16_t* d_v_scale_fp16,
    uint16_t* d_k_dense_fp16, uint16_t* d_v_dense_fp16,
    int context_len, int kv_heads, int head_dim, int scale_block,
    int max_context, void* stream = nullptr);

// INT8 per-token-head KV cache: INT8 K/V arrays with per-token per-head FP16
// scales. Dynamic quantization per token and KV head, proved faster than FP16
// in vLLM-2080Ti benchmarks (+16% decode at 65K).
bool qwen_append_kv_cache_int8_per_token_head_cuda(
    const uint16_t* d_k_rows_fp16, const uint16_t* d_v_rows_fp16,
    int8_t* d_k_cache_int8, int8_t* d_v_cache_int8,
    uint16_t* d_k_scale_fp16, uint16_t* d_v_scale_fp16,
    int seq_len, int kv_heads, int head_dim, int start_pos, int max_context,
    void* stream = nullptr);

bool qwen_int8_dequant_kv_cache_cuda(
    const int8_t* d_k_cache_int8, const int8_t* d_v_cache_int8,
    const uint16_t* d_k_scale_fp16, const uint16_t* d_v_scale_fp16,
    uint16_t* d_k_dense_fp16, uint16_t* d_v_dense_fp16,
    int context_len, int kv_heads, int head_dim, int max_context,
    void* stream = nullptr);

bool qwen_gqa_decode_attention_int8_per_token_head_cuda(
    const uint16_t* d_q_fp16, const int8_t* d_k_cache_int8,
    const int8_t* d_v_cache_int8, const uint16_t* d_k_scale_fp16,
    const uint16_t* d_v_scale_fp16, uint16_t* d_out_fp16, float* d_score_scratch,
    int q_heads, int kv_heads, int head_dim, int context_len, int max_context,
    int attention_window, int attention_sink_tokens, void* stream = nullptr);

// TurboQuant K8V4 KV-cache: one combined slot per token and KV head holding an
// FP8 E5M2 key (one byte per channel), a 4-bit uniformly quantized value (two
// channels per byte), and the value's FP16 scale and minimum. The slot size
// follows head_dim, so it is 196 bytes at head_dim=128 and 388 at head_dim=256.
constexpr int kTurboQuantK8V4MetadataBytes = 4;

constexpr int qwen_turboquant_k8v4_slot_bytes(int head_dim) {
    return head_dim + head_dim / 2 + kTurboQuantK8V4MetadataBytes;
}

bool qwen_append_kv_cache_turboquant_k8v4_cuda(
    const uint16_t* d_k_rows_fp16, const uint16_t* d_v_rows_fp16,
    uint8_t* d_combined_cache, int seq_len, int kv_heads, int head_dim,
    int start_pos, int max_context, void* stream = nullptr);

bool qwen_gqa_decode_attention_turboquant_k8v4_cuda(
    const uint16_t* d_q_fp16, const uint8_t* d_combined_cache,
    uint16_t* d_out_fp16, float* d_score_scratch, int q_heads, int kv_heads,
    int head_dim, int context_len, int max_context, int attention_window,
    int attention_sink_tokens, void* stream = nullptr);

bool qwen_gqa_prefill_attention_turboquant_k8v4_cuda(
    const uint16_t* d_q_rows_fp16, const uint8_t* d_combined_cache,
    uint16_t* d_out_rows_fp16, int seq_len, int q_heads, int kv_heads,
    int head_dim, int position_offset, int max_context, int attention_window,
    int attention_sink_tokens, void* stream = nullptr);

// Expand [0, context_len) of the combined cache into dense FP16 K/V laid out
// exactly like the FP16 cache ([max_context, kv_heads, head_dim], so the same
// max_context stride). Dequantizing the whole range once and handing the result
// to the tensor-core prefill kernel costs O(context) instead of the
// O(rows * context) of dequantizing inside the attention loop, which is what
// makes a quantized cache competitive on prefill.
bool qwen_turboquant_k8v4_dequant_kv_cuda(
    const uint8_t* d_combined_cache, uint16_t* d_k_dense_fp16,
    uint16_t* d_v_dense_fp16, int context_len, int kv_heads, int head_dim,
    int max_context, void* stream = nullptr);

}  // namespace pocket
