#include "model_config.hpp"

#include "json_lite.hpp"

#include <cmath>
#include <fstream>
#include <sstream>

namespace pocket {

namespace {

uint64_t first_present_u64(const GGUFFile& file, std::initializer_list<std::string> keys) {
    for (const auto& key : keys) {
        if (auto value = file.metadata_u64(key)) {
            return *value;
        }
    }
    return 0;
}

double first_present_f64(const GGUFFile& file, std::initializer_list<std::string> keys) {
    for (const auto& key : keys) {
        if (auto value = file.metadata_f64(key)) {
            return *value;
        }
    }
    return 0.0;
}

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("failed to open file: " + path);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

uint64_t optional_u64(const JsonObject& obj, const std::string& key) {
    const auto* value = object_get(obj, key);
    return value == nullptr ? 0 : static_cast<uint64_t>(std::round(value->number()));
}

double optional_f64(const JsonObject& obj, const std::string& key) {
    const auto* value = object_get(obj, key);
    return value == nullptr ? 0.0 : value->number();
}

std::string optional_string(const JsonObject& obj, const std::string& key) {
    const auto* value = object_get(obj, key);
    return value == nullptr ? std::string() : value->string();
}

std::vector<uint64_t> optional_u64_array(const JsonObject& obj, const std::string& key) {
    const auto* value = object_get(obj, key);
    if (value == nullptr) return {};
    std::vector<uint64_t> out;
    for (const auto& item : value->array()) {
        out.push_back(static_cast<uint64_t>(std::round(item.number())));
    }
    return out;
}

std::string join_u64(const std::vector<uint64_t>& xs) {
    std::ostringstream oss;
    oss << '[';
    for (size_t i = 0; i < xs.size(); ++i) {
        if (i) oss << ',';
        oss << xs[i];
    }
    oss << ']';
    return oss.str();
}

std::string join_f64_prefix(const std::vector<double>& xs, size_t limit = 8) {
    std::ostringstream oss;
    oss << '[';
    const size_t n = std::min(limit, xs.size());
    for (size_t i = 0; i < n; ++i) {
        if (i) oss << ',';
        oss << xs[i];
    }
    if (xs.size() > n) oss << ",...(" << xs.size() << ')';
    oss << ']';
    return oss.str();
}

}  // namespace

ModelConfig ModelConfig::from_gguf(const GGUFFile& file) {
    ModelConfig cfg;
    cfg.architecture = file.metadata_string("general.architecture").value_or("");
    const std::string prefix = cfg.architecture.empty() ? std::string("deepseek4") : cfg.architecture;

    cfg.vocab_size = first_present_u64(file, {"tokenizer.ggml.tokens", prefix + ".vocab_size", "deepseek4.vocab_size", "llama.vocab_size"});
    if (cfg.vocab_size == 0) {
        auto it = file.metadata().find("tokenizer.ggml.tokens");
        if (it != file.metadata().end()) {
            if (auto* tokens = std::get_if<std::vector<std::string>>(&it->second)) {
                cfg.vocab_size = tokens->size();
            }
        }
    }

    cfg.dim = first_present_u64(file, {prefix + ".embedding_length", "deepseek4.embedding_length", "llama.embedding_length"});
    cfg.n_layers = first_present_u64(file, {prefix + ".block_count", "deepseek4.block_count", "llama.block_count"});
    cfg.n_heads = first_present_u64(file, {prefix + ".attention.head_count", "deepseek4.attention.head_count", "llama.attention.head_count"});
    cfg.kv_heads = first_present_u64(file, {prefix + ".attention.head_count_kv", "deepseek4.attention.head_count_kv", "llama.attention.head_count_kv"});
    cfg.context_length = first_present_u64(file, {prefix + ".context_length", "deepseek4.context_length", "llama.context_length"});
    cfg.head_dim = first_present_u64(file, {prefix + ".attention.key_length", "deepseek4.attention.key_length", "llama.attention.key_length"});
    cfg.value_dim = first_present_u64(file, {prefix + ".attention.value_length", "deepseek4.attention.value_length", "llama.attention.value_length"});
    cfg.q_lora_rank = first_present_u64(file, {prefix + ".attention.q_lora_rank", "deepseek4.attention.q_lora_rank"});
    cfg.o_lora_rank = first_present_u64(file, {prefix + ".attention.output_lora_rank", "deepseek4.attention.output_lora_rank"});
    cfg.o_groups = first_present_u64(file, {prefix + ".attention.output_group_count", "deepseek4.attention.output_group_count"});
    cfg.index_n_heads = first_present_u64(file, {prefix + ".attention.indexer.head_count", "deepseek4.attention.indexer.head_count"});
    cfg.index_head_dim = first_present_u64(file, {prefix + ".attention.indexer.key_length", "deepseek4.attention.indexer.key_length"});
    cfg.index_topk = first_present_u64(file, {prefix + ".attention.indexer.top_k", "deepseek4.attention.indexer.top_k"});
    cfg.n_routed_experts = first_present_u64(file, {prefix + ".expert_count", "deepseek4.expert_count", "llama.expert_count"});
    cfg.n_shared_experts = first_present_u64(file, {prefix + ".expert_shared_count", "deepseek4.expert_shared_count"});
    cfg.n_activated_experts = first_present_u64(file, {prefix + ".expert_used_count", "deepseek4.expert_used_count", "llama.expert_used_count"});
    cfg.moe_inter_dim = first_present_u64(file, {prefix + ".expert_feed_forward_length", "deepseek4.expert_feed_forward_length"});
    cfg.route_scale = first_present_f64(file, {prefix + ".expert_weights_scale", "deepseek4.expert_weights_scale"});
    cfg.swiglu_limit = 10.0;
    cfg.original_context_length = first_present_u64(file, {prefix + ".rope.scaling.original_context_length", "deepseek4.rope.scaling.original_context_length"});
    cfg.rope_dim = first_present_u64(file, {prefix + ".rope.dimension_count", "deepseek4.rope.dimension_count"});
    cfg.rope_theta = first_present_f64(file, {prefix + ".rope.freq_base", "deepseek4.rope.freq_base"});
    cfg.rope_factor = first_present_f64(file, {prefix + ".rope.scaling.factor", "deepseek4.rope.scaling.factor"});
    cfg.beta_fast = first_present_f64(file, {prefix + ".rope.scaling.yarn_beta_fast", "deepseek4.rope.scaling.yarn_beta_fast"});
    cfg.beta_slow = first_present_f64(file, {prefix + ".rope.scaling.yarn_beta_slow", "deepseek4.rope.scaling.yarn_beta_slow"});
    cfg.window_size = first_present_u64(file, {prefix + ".attention.sliding_window", "deepseek4.attention.sliding_window"});
    cfg.compress_rope_theta = first_present_f64(file, {prefix + ".attention.compress_rope_freq_base", "deepseek4.attention.compress_rope_freq_base"});
    cfg.n_hash_layers = first_present_u64(file, {prefix + ".hash_layer_count", "deepseek4.hash_layer_count"});
    cfg.hc_mult = first_present_u64(file, {prefix + ".hyper_connection.count", "deepseek4.hyper_connection.count"});
    cfg.hc_sinkhorn_iters = first_present_u64(file, {prefix + ".hyper_connection.sinkhorn_iterations", "deepseek4.hyper_connection.sinkhorn_iterations"});
    cfg.compress_ratios = file.metadata_u64_array(prefix + ".attention.compress_ratios");
    cfg.swiglu_clamp_exp = file.metadata_f64_array(prefix + ".swiglu_clamp_exp");
    if (cfg.head_dim == 0 && cfg.dim > 0 && cfg.n_heads > 0) {
        cfg.head_dim = cfg.dim / cfg.n_heads;
    }
    if (cfg.value_dim == 0) {
        cfg.value_dim = cfg.head_dim;
    }
    return cfg;
}

ModelConfig ModelConfig::from_hf_config(const std::string& ckpt_dir) {
    ModelConfig cfg;
    JsonValue root_value = parse_json(read_file(ckpt_dir + "/config.json"));
    const JsonObject& obj = root_value.object();
    cfg.architecture = optional_string(obj, "model_type");
    cfg.vocab_size = optional_u64(obj, "vocab_size");
    cfg.dim = optional_u64(obj, "hidden_size");
    cfg.n_layers = optional_u64(obj, "num_hidden_layers");
    cfg.n_heads = optional_u64(obj, "num_attention_heads");
    cfg.kv_heads = optional_u64(obj, "num_key_value_heads");
    cfg.head_dim = optional_u64(obj, "head_dim");
    cfg.value_dim = optional_u64(obj, "v_head_dim");
    cfg.q_lora_rank = optional_u64(obj, "q_lora_rank");
    cfg.o_lora_rank = optional_u64(obj, "o_lora_rank");
    cfg.o_groups = optional_u64(obj, "o_groups");
    cfg.index_n_heads = optional_u64(obj, "index_n_heads");
    cfg.index_head_dim = optional_u64(obj, "index_head_dim");
    cfg.index_topk = optional_u64(obj, "index_topk");
    cfg.n_routed_experts = optional_u64(obj, "n_routed_experts");
    cfg.n_shared_experts = optional_u64(obj, "n_shared_experts");
    cfg.n_activated_experts = optional_u64(obj, "num_experts_per_tok");
    cfg.moe_inter_dim = optional_u64(obj, "moe_intermediate_size");
    cfg.route_scale = optional_f64(obj, "routed_scaling_factor");
    cfg.swiglu_limit = optional_f64(obj, "swiglu_limit");
    cfg.context_length = optional_u64(obj, "max_position_embeddings");
    cfg.rope_dim = optional_u64(obj, "qk_rope_head_dim");
    cfg.rope_theta = optional_f64(obj, "rope_theta");
    cfg.window_size = optional_u64(obj, "sliding_window");
    cfg.compress_rope_theta = optional_f64(obj, "compress_rope_theta");
    cfg.n_hash_layers = optional_u64(obj, "num_hash_layers");
    cfg.hc_mult = optional_u64(obj, "hc_mult");
    cfg.hc_sinkhorn_iters = optional_u64(obj, "hc_sinkhorn_iters");
    cfg.compress_ratios = optional_u64_array(obj, "compress_ratios");
    if (const auto* scaling = object_get(obj, "rope_scaling")) {
        if (scaling->is_object()) {
            const JsonObject& sc = scaling->object();
            cfg.rope_factor = optional_f64(sc, "factor");
            cfg.beta_fast = optional_f64(sc, "beta_fast");
            cfg.beta_slow = optional_f64(sc, "beta_slow");
            cfg.original_context_length = optional_u64(sc, "original_max_position_embeddings");
        }
    }
    if (cfg.head_dim == 0 && cfg.dim > 0 && cfg.n_heads > 0) {
        cfg.head_dim = cfg.dim / cfg.n_heads;
    }
    if (cfg.value_dim == 0) {
        cfg.value_dim = cfg.head_dim;
    }
    return cfg;
}

std::string ModelConfig::to_string() const {
    std::ostringstream oss;
    oss << "architecture=" << architecture << '\n';
    oss << "vocab_size=" << vocab_size << '\n';
    oss << "dim=" << dim << '\n';
    oss << "n_layers=" << n_layers << '\n';
    oss << "n_heads=" << n_heads << '\n';
    oss << "kv_heads=" << kv_heads << '\n';
    oss << "head_dim=" << head_dim << '\n';
    oss << "value_dim=" << value_dim << '\n';
    oss << "q_lora_rank=" << q_lora_rank << '\n';
    oss << "o_lora_rank=" << o_lora_rank << '\n';
    oss << "o_groups=" << o_groups << '\n';
    oss << "index_n_heads=" << index_n_heads << '\n';
    oss << "index_head_dim=" << index_head_dim << '\n';
    oss << "index_topk=" << index_topk << '\n';
    oss << "n_routed_experts=" << n_routed_experts << '\n';
    oss << "n_shared_experts=" << n_shared_experts << '\n';
    oss << "n_activated_experts=" << n_activated_experts << '\n';
    oss << "moe_inter_dim=" << moe_inter_dim << '\n';
    oss << "route_scale=" << route_scale << '\n';
    oss << "swiglu_limit=" << swiglu_limit << '\n';
    oss << "context_length=" << context_length << '\n';
    oss << "original_context_length=" << original_context_length << '\n';
    oss << "rope_dim=" << rope_dim << '\n';
    oss << "rope_theta=" << rope_theta << '\n';
    oss << "rope_factor=" << rope_factor << '\n';
    oss << "beta_fast=" << beta_fast << '\n';
    oss << "beta_slow=" << beta_slow << '\n';
    oss << "window_size=" << window_size << '\n';
    oss << "compress_rope_theta=" << compress_rope_theta << '\n';
    oss << "n_hash_layers=" << n_hash_layers << '\n';
    oss << "hc_mult=" << hc_mult << '\n';
    oss << "hc_sinkhorn_iters=" << hc_sinkhorn_iters << '\n';
    oss << "compress_ratios=" << join_u64(compress_ratios) << '\n';
    oss << "swiglu_clamp_exp=" << join_f64_prefix(swiglu_clamp_exp) << '\n';
    return oss.str();
}

}  // namespace pocket
