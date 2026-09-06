#pragma once

#include "gguf_reader.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace pocket {

struct ModelConfig {
    uint64_t vocab_size = 0;
    uint64_t dim = 0;
    uint64_t n_layers = 0;
    uint64_t n_heads = 0;
    uint64_t head_dim = 0;
    uint64_t value_dim = 0;
    uint64_t kv_heads = 0;
    uint64_t q_lora_rank = 0;
    uint64_t o_lora_rank = 0;
    uint64_t o_groups = 0;
    uint64_t index_n_heads = 0;
    uint64_t index_head_dim = 0;
    uint64_t index_topk = 0;
    uint64_t n_routed_experts = 0;
    uint64_t n_shared_experts = 0;
    uint64_t n_activated_experts = 0;
    uint64_t moe_inter_dim = 0;
    double route_scale = 0.0;
    double swiglu_limit = 0.0;
    uint64_t context_length = 0;
    uint64_t original_context_length = 0;
    uint64_t rope_dim = 0;
    double rope_theta = 0.0;
    double rope_factor = 0.0;
    double beta_fast = 0.0;
    double beta_slow = 0.0;
    uint64_t window_size = 0;
    double compress_rope_theta = 0.0;
    uint64_t n_hash_layers = 0;
    uint64_t hc_mult = 0;
    uint64_t hc_sinkhorn_iters = 0;
    std::vector<uint64_t> compress_ratios;
    std::vector<double> swiglu_clamp_exp;
    std::string architecture;

    static ModelConfig from_gguf(const GGUFFile& file);
    static ModelConfig from_hf_config(const std::string& ckpt_dir);
    std::string to_string() const;
};

}  // namespace pocket
