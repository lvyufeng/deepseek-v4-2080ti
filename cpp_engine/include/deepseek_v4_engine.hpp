#pragma once

#include "gguf_reader.hpp"
#include "model_config.hpp"

#include <string>
#include <vector>

namespace pocket {

class DeepSeekV4Engine {
public:
    explicit DeepSeekV4Engine(const std::string& model_path);

    const GGUFFile& gguf() const { return gguf_; }
    const ModelConfig& config() const { return config_; }

private:
    GGUFFile gguf_;
    ModelConfig config_;
};

struct ForwardSmokeResult {
    int token = 0;
    int dim = 0;
    int inter = 0;
    int logits = 0;
    int layers = 0;
    int top_token = 0;
    float top_logit = 0.0f;
    float checksum = 0.0f;
};

struct GenerateSmokeResult {
    std::vector<ForwardSmokeResult> tokens;
    double wall_seconds = 0.0;
    double prefill_seconds = 0.0;
    double decode_seconds = 0.0;
    int prompt_tokens = 0;
    int decode_tokens = 0;
};

struct ForwardSmokeOptions {
    int tp_world = 1;
    int tp_rank = 0;
    int device = 0;
    bool skip_fp4_host_prepare = false;
    std::string nccl_id_path;
};

ForwardSmokeResult run_safetensors_min_layer_smoke(const std::string& ckpt_dir);
ForwardSmokeResult run_safetensors_layer_loop_smoke(const std::string& ckpt_dir, int layer_count);
ForwardSmokeResult run_safetensors_token_forward(const std::string& ckpt_dir, int token, int layer_count);
ForwardSmokeResult run_safetensors_token_forward_at_position(const std::string& ckpt_dir, int token, int layer_count, int position);
ForwardSmokeResult run_safetensors_token_forward_with_options(const std::string& ckpt_dir, int token, int layer_count, int position, const ForwardSmokeOptions& options);
ForwardSmokeResult run_safetensors_prompt_forward(const std::string& ckpt_dir, const std::vector<int>& tokens, int layer_count);
ForwardSmokeResult run_safetensors_prompt_forward_with_options(const std::string& ckpt_dir, const std::vector<int>& tokens, int layer_count, const ForwardSmokeOptions& options);
std::vector<ForwardSmokeResult> run_safetensors_generate_tokens(const std::string& ckpt_dir, const std::vector<int>& seed_tokens, int layer_count, int max_new_tokens);
std::vector<ForwardSmokeResult> run_safetensors_generate_tokens_with_options(const std::string& ckpt_dir, const std::vector<int>& seed_tokens, int layer_count, int max_new_tokens, const ForwardSmokeOptions& options);
GenerateSmokeResult run_safetensors_generate_tokens_timed_with_options(const std::string& ckpt_dir, const std::vector<int>& seed_tokens, int layer_count, int max_new_tokens, const ForwardSmokeOptions& options);

// --- GGUF Q2 forward entry points ------------------------------------------
//
// Sibling to the safetensors path. The full forward chain isn't wired in yet
// (Phase 3); this initial entry point only verifies that we can construct a
// GgufForwardContext and that the GGUF mapping table resolves the dense
// metadata we'll need. Implementation is in deepseek_v4_engine.cpp alongside the
// safetensors path so future GGUF forward operators can share helpers.

struct GgufSmokeResult {
    int n_layers = 0;
    int n_hash_layers = 0;
    int dim = 0;
    int moe_inter_dim = 0;
    int n_routed_experts = 0;
    int n_activated_experts = 0;
    int vocab = 0;
};

GgufSmokeResult run_gguf_min_layer_smoke(const std::string& ckpt_path);

// Phase 3 step: exercise the dense input chain for layer 0 attention's
// q_a branch. Embed F16 lookup -> attn_norm F32->BF16 RMSNorm -> wq_a Q8_0
// matvec. Validates the F32 norm gamma conversion path and the first
// Q8_0 attention projection against real GGUF data.
struct GgufAttnNormWqaResult {
    int dim = 0;
    int q_a_dim = 0;
    float embed_rms = 0.0f;
    float normed_rms = 0.0f;
    float q_a_rms = 0.0f;
    float q_a_first[4] = {0.0f, 0.0f, 0.0f, 0.0f};
};

GgufAttnNormWqaResult run_gguf_attn_norm_wq_a_smoke(const std::string& ckpt_path, int token);

// Phase 3 step: full attention Q path for layer 0. Extends the q_a smoke
// through q_norm RMSNorm, wq_b Q8_0 projection, and head_rmsnorm_rope.
// Output is the final per-head Q tensor ready to feed attention compute.
struct GgufAttnQPathResult {
    int dim = 0;
    int q_a_dim = 0;
    int heads = 0;
    int head_dim = 0;
    int rope_dim = 0;
    float q_normed_rms = 0.0f;     // RMS of q_a after q_norm
    float q_pre_rope_rms = 0.0f;   // RMS of q (heads*head_dim) after wq_b, before head norm/rope
    float q_post_rope_rms = 0.0f;  // RMS of q after head_rmsnorm_rope
    float q_first[4] = {0.0f, 0.0f, 0.0f, 0.0f};
};

GgufAttnQPathResult run_gguf_attn_q_path_smoke(const std::string& ckpt_path, int token, int position);

// Phase 3 step: attention KV path for layer 0. Embed -> attn_norm -> wkv Q8_0
// projection (dim -> kv_dim) -> kv_norm RMSNorm -> head_rmsnorm_rope with
// heads=1, head_dim=kv_dim, rope_dim. RoPE is a rotation so the final L2 norm
// equals the post-rmsnorm L2 norm (sqrt(kv_dim) for unit-RMS output).
struct GgufAttnKvPathResult {
    int dim = 0;
    int kv_dim = 0;
    int rope_dim = 0;
    float kv_a_rms = 0.0f;       // RMS of kv_a after wkv (before rmsnorm)
    float kv_norm_rms = 0.0f;    // RMS of kv after rmsnorm with kv_norm gamma
    float kv_post_rope_rms = 0.0f; // RMS of kv after RoPE; rotation preserves L2
    float kv_first[4] = {0.0f, 0.0f, 0.0f, 0.0f};
};

GgufAttnKvPathResult run_gguf_attn_kv_path_smoke(const std::string& ckpt_path, int token, int position);

// Phase 3 step: full single-token dense attention for layer 0.
// Embed -> attn_norm -> Q-path (wq_a, q_norm, wq_b, head_rmsnorm_rope) ||
// KV-path (wkv, kv_norm, head_rmsnorm_rope) -> single_token_sparse_attention
// (with attn_sink) -> inverse RoPE on rope tail of each head -> grouped wo_a
// Q8_0 (o_groups separate matvecs) -> wo_b Q8_0 (-> dim).
// Output `attn_out` is the per-layer attention residual (length dim).
struct GgufAttnFullResult {
    int dim = 0;
    int q_a_dim = 0;
    int heads = 0;
    int head_dim = 0;
    int kv_dim = 0;
    int rope_dim = 0;
    int o_groups = 0;
    int o_lora_rank = 0;
    int attn_mid = 0;
    float q_rms = 0.0f;
    float kv_rms = 0.0f;
    float attn_value_rms = 0.0f;       // RMS of sparse-attention output
    float attn_value_post_inv_rms = 0.0f; // after inverse RoPE
    float attn_mid_rms = 0.0f;
    float attn_out_rms = 0.0f;
    float attn_out_first[4] = {0.0f, 0.0f, 0.0f, 0.0f};
};

GgufAttnFullResult run_gguf_attn_full_smoke(const std::string& ckpt_path, int token, int position);

// Phase 3 step: shared-expert FFN smoke for layer 0.
// Embed -> ffn_norm RMSNorm -> shared w1 / w3 Q8_0 matvec -> silu_mul ->
// shared w2 Q8_0 matvec. All three shared-expert projections are stored as
// Q8_0 in the GGUF model. Output is the shared-expert contribution to the
// FFN residual (length dim).
struct GgufSharedExpertResult {
    int dim = 0;
    int moe_inter_dim = 0;
    float ffn_normed_rms = 0.0f;
    float gate_rms = 0.0f;
    float up_rms = 0.0f;
    float hidden_rms = 0.0f;   // after silu_mul
    float shared_out_rms = 0.0f;
    float shared_out_first[4] = {0.0f, 0.0f, 0.0f, 0.0f};
};

GgufSharedExpertResult run_gguf_shared_expert_smoke(const std::string& ckpt_path, int token);

// Phase 3 step: single routed expert smoke for layer 0.
// Embed -> ffn_norm -> q8_1 quantize -> IQ2_XXS w1/w3 matvec -> SwiGLU
// (clamped + route-weight) -> q8_1 quantize hidden -> Q2_K w2 matvec.
// Loads only the single expert's slice from the routed 3D GGUF tensor, so
// the kernel sees n_experts=1 + route_slots=[0]. This wires real GGUF
// per-expert byte offsets into the existing Q2 single-token kernels.
struct GgufRoutedExpertResult {
    int dim = 0;
    int moe_inter_dim = 0;
    int expert_id = 0;
    float ffn_normed_rms = 0.0f;
    float gate_rms = 0.0f;
    float up_rms = 0.0f;
    float hidden_rms = 0.0f; // after SwiGLU + route weight (post quantize-dequantize)
    float route_out_rms = 0.0f;
    float route_out_first[4] = {0.0f, 0.0f, 0.0f, 0.0f};
};

GgufRoutedExpertResult run_gguf_routed_expert_smoke(const std::string& ckpt_path, int token, int expert_id);

// Phase 3 step: multi-active-expert routed Q2 MoE smoke for layer 0.
// Embed -> ffn_norm -> hash-gate lookup (tid2eid for hash layers) producing
// top-k expert ids and uniform 1/k weights -> stage top-k experts' bytes
// (w1/w3 IQ2_XXS + w2 Q2_K) into packed device buffers -> q8_1 quantize x
// broadcast to routes -> batched IQ2_XXS w1/w3 matvec -> SwiGLU+route_weight
// -> Q2_K w2 matvec accumulated across routes. Output: MoE contribution
// (length dim) summed over the top-k active experts for this token.
struct GgufRoutedMoeResult {
    int dim = 0;
    int moe_inter_dim = 0;
    int n_active = 0;          // number of active experts (top-k)
    int expert_ids[8] = {0,0,0,0,0,0,0,0};
    // Real route weights from sqrt_softplus(W gate · x_normed), gathered at
    // the hash-selected expert ids, normalized to sum=1, then ×route_scale.
    float route_weights[8] = {0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f,0.0f};
    float route_weights_sum = 0.0f; // sanity: should equal route_scale (1.5 for DeepSeek-V4-Flash)
    float ffn_normed_rms = 0.0f;
    float moe_out_rms = 0.0f;
    float moe_out_first[4] = {0.0f, 0.0f, 0.0f, 0.0f};
};

GgufRoutedMoeResult run_gguf_routed_moe_smoke(const std::string& ckpt_path, int token);

// Phase 3 step: full layer-0 forward composition with residuals.
// Embed -> save x_pre_attn -> attn_norm -> Q/KV path -> sparse attention ->
// inverse RoPE -> grouped wo_a -> wo_b -> +x_pre_attn (residual)
// -> save x_pre_ffn -> ffn_norm -> shared expert (Q8_0 w1/w3 + silu_mul + Q8_0 w2)
// -> hash-gate scores via gate W -> stage top-k Q2 experts -> Q8_1 quantize x
// -> batched IQ2_XXS w1/w3 -> SwiGLU + route_weight + Q8_1 quantize hidden
// -> batched Q2_K w2 (atomicAdd) -> +x_pre_ffn (residual) -> x_out.
// Validates the residual flow through one full layer (attention + FFN).
struct GgufLayer0FullResult {
    int dim = 0;
    int moe_inter_dim = 0;
    int heads = 0;
    int head_dim = 0;
    int n_active = 0;
    int expert_ids[8] = {0,0,0,0,0,0,0,0};
    float embed_rms = 0.0f;
    float attn_out_rms = 0.0f;        // wo_b output (before residual)
    float x_post_attn_rms = 0.0f;     // x_pre_attn + attn_out
    float shared_out_rms = 0.0f;
    float moe_out_rms = 0.0f;
    float ffn_combined_rms = 0.0f;    // shared_out + moe_out (before residual)
    float x_post_ffn_rms = 0.0f;      // x_post_attn + ffn_combined
    float route_weights_sum = 0.0f;   // sanity: should equal route_scale (1.5)
    float x_post_ffn_first[4] = {0.0f, 0.0f, 0.0f, 0.0f};
};

GgufLayer0FullResult run_gguf_layer0_full_smoke(const std::string& ckpt_path, int token, int position);

// Phase 3 step: 43-layer GGUF forward + final norm + Q8_0 head + argmax.
// Loads all dense weights resident on device, per-layer stages top-k routed
// Q2 experts (hash-gate layers use tid2eid lookup; non-hash layers compute
// the gate then D2H copy expert ids back to host for staging), runs the
// layer-forward helper across all 43 layers, applies output_norm RMSNorm,
// runs the Q8_0 head matvec, and reports the argmax token + top logit + a
// debug checksum.
struct GgufFullForwardResult {
    int n_layers = 0;
    int dim = 0;
    int vocab = 0;
    int top_token = 0;
    float top_logit = 0.0f;
    float checksum = 0.0f;       // sum of logits, like FP4 path
    float final_x_rms = 0.0f;
    float final_normed_rms = 0.0f;
    float logits_rms = 0.0f;
    float logits_first[4] = {0.0f, 0.0f, 0.0f, 0.0f};
};

GgufFullForwardResult run_gguf_full_forward_smoke(const std::string& ckpt_path, int token, int position);

// Phase 3 step: multi-token greedy decode smoke. Loads all dense weights once,
// allocates a per-layer KV cache sized for the full sequence (seeds + decode),
// then runs forward once per position. The token at position p is either the
// seed at p (when p < seeds.size()) or the argmax from position p-1. Returns
// the full sequence of generated tokens after the seeds, plus per-step top
// logits and rough timing breakdown.
struct GgufDecodeResult {
    int n_layers = 0;
    int dim = 0;
    int vocab = 0;
    std::vector<int> generated_tokens;  // length = max_new_tokens
    std::vector<float> top_logits;      // length = seeds.size() + max(0, max_new_tokens - 1)
    double load_seconds = 0.0;
    double forward_seconds = 0.0;       // total time across all forward steps
    double prefill_seconds = 0.0;       // current GGUF path: one forward per prompt token
    double decode_seconds = 0.0;        // current GGUF path: one forward per generated step
    int prompt_tokens = 0;
    int decode_tokens = 0;
};

GgufDecodeResult run_gguf_generate_smoke(const std::string& ckpt_path,
                                          const std::vector<int>& seed_tokens,
                                          int max_new_tokens,
                                          const ForwardSmokeOptions& options);
GgufDecodeResult run_gguf_generate_smoke(const std::string& ckpt_path,
                                          const std::vector<int>& seed_tokens,
                                          int max_new_tokens);

}  // namespace pocket
