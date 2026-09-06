#pragma once

#include <cstdint>
#include <vector>

namespace pocket {

// Score one FlashMemory CSA layer against all compressed-K chunks for a single
// decode token. Reproduces retriever.py forward_and_score (fp32 weights).
//
// Inputs (device):
//   d_hidden        [hidden_dim]                      decode-token hidden state
//   d_wq_a          [q_lora_rank, hidden_dim]         row-major weight
//   d_wq_b          [n_heads*head_dim, q_lora_rank]   row-major weight
//   d_q_norm        [q_lora_rank]                     RMSNorm gamma
//   d_weights_proj  [n_heads, hidden_dim]             row-major weight
//   d_inv_freqs     [rope_dim/2]                      YaRN inverse frequencies
//   d_compressed_k  [n_chunks, head_dim + 4] uint8    fp8 K + f32 scale per chunk
// Scratch (device, caller-owned):
//   d_q_scratch     [n_heads*head_dim]
//   d_qlora_scratch [q_lora_rank]
//   d_fused_w       [n_heads]
// Output (device):
//   d_scores        [n_chunks]  sigmoid in [0,1] if apply_sigmoid else raw logits
bool flashmemory_score_layer_cuda(
    const float* d_hidden,
    const float* d_wq_a,
    const float* d_wq_b,
    const float* d_q_norm,
    const float* d_weights_proj,
    const float* d_inv_freqs,
    const uint8_t* d_compressed_k,
    float* d_q_scratch,
    float* d_qlora_scratch,
    float* d_fused_w,
    float* d_scores,
    int n_chunks,
    int n_heads,
    int head_dim,
    int q_lora_rank,
    int hidden_dim,
    int rope_dim,
    float rms_norm_eps,
    long long position,
    bool apply_sigmoid,
    void* stream = nullptr);

// Compute YaRN RoPE inverse frequencies on host (matches retriever.py
// precompute_freqs_cis mixed-freqs construction). Returns [rope_dim/2] floats:
// the per-pair angular frequency (mixed = freq*(1-ramp) + (freq/factor)*ramp).
std::vector<float> flashmemory_yarn_inv_freqs(
    int rope_dim,
    double base,
    double factor,
    int original_seq_len,
    double beta_fast,
    double beta_slow);

// Float-key variant: score against pre-dequantized float keys instead of fp8.
// Inputs identical to flashmemory_score_layer_cuda except:
//   d_float_keys  [n_chunks, head_dim] float  (already Hadamard-rotated, e.g. from indexer_kv_cache)
// No fp8 dequant step; directly use float keys in scoring. Reproduces the same
// q-side path (wq_a/wq_b/RMSNorm/RoPE/Hadamard) and score math (relu(q·k)×fused_w).
bool flashmemory_score_layer_floatk_cuda(
    const float* d_hidden,
    const float* d_wq_a,
    const float* d_wq_b,
    const float* d_q_norm,
    const float* d_weights_proj,
    const float* d_inv_freqs,
    const float* d_float_keys,
    float* d_q_scratch,
    float* d_qlora_scratch,
    float* d_fused_w,
    float* d_scores,
    int n_chunks,
    int n_heads,
    int head_dim,
    int q_lora_rank,
    int hidden_dim,
    int rope_dim,
    float rms_norm_eps,
    long long position,
    bool apply_sigmoid,
    void* stream = nullptr);

// Ensemble scores from multiple layers (e.g. l10/l12/l20) into a single global score per chunk.
// Inputs:
//   d_layer_scores  [n_layers, score_stride]  per-layer scores (already sigmoid if needed)
//   n_layers        number of scorer layers (e.g. 3)
//   n_chunks        number of valid chunks to ensemble
//   score_stride    stride between scorer layers in d_layer_scores (>= n_chunks)
//   use_max         true=max ensemble, false=mean ensemble
// Output:
//   d_ensemble_scores  [n_chunks]  max or mean across layers per chunk
bool flashmemory_ensemble_cuda(
    const float* d_layer_scores,
    int n_layers,
    int n_chunks,
    int score_stride,
    bool use_max,
    float* d_ensemble_scores,
    void* stream = nullptr);

// Select top-k highest-scoring chunks from ensemble scores.
// Inputs:
//   d_scores     [n_chunks]  ensemble scores
//   n_chunks     total chunk count
//   keep         top-k count
// Outputs:
//   d_out_indices  [keep]  logical chunk indices (0..n_chunks-1) of top-k, sorted descending by score
//   d_out_scores   [keep]  corresponding scores (optional, can be nullptr)
// Uses a simple iterative argmax (same strategy as indexer_topk_kernel).
bool flashmemory_topk_cuda(
    const float* d_scores,
    int n_chunks,
    int keep,
    int* d_out_indices,
    float* d_out_scores,
    void* stream = nullptr);

// Write global top-k logical chunk indices into a layer's kv_indices array with a physical offset.
// Inputs:
//   d_logical    [keep]  logical chunk IDs (0..compressed_cap-1)
//   keep         count
//   offset       physical offset (e.g. window size) to add to each logical ID
// Output:
//   d_out_indices  [keep]  d_out[i] = offset + d_logical[i]
// Tiny kernel, used to map global top-k into per-layer d_kv_indices format.
bool flashmemory_write_global_indices_cuda(
    const int* d_logical,
    int keep,
    int offset,
    int* d_out_indices,
    void* stream = nullptr);

}  // namespace pocket
