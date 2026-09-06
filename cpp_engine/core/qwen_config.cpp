#include "qwen_config.hpp"

#include "json_lite.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace pocket {
namespace {

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("failed to open file: " + path);
    std::ostringstream out;
    out << in.rdbuf();
    return out.str();
}

// generation_config.json is optional: a checkpoint without one simply has no
// stop tokens, so this returns false rather than throwing.
bool try_read_file(const std::string& path, std::string* out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    std::ostringstream buffer;
    buffer << in.rdbuf();
    *out = buffer.str();
    return true;
}

int checked_token_id(const JsonValue& value, uint64_t vocab_size) {
    if (!value.is_number()) {
        throw std::runtime_error("Qwen eos_token_id entry is not a number");
    }
    const double number = value.number();
    const double rounded = std::round(number);
    if (number < 0.0 || std::fabs(number - rounded) > 1.0e-9) {
        throw std::runtime_error("Qwen eos_token_id is not a non-negative integer");
    }
    const uint64_t id = static_cast<uint64_t>(rounded);
    // An out-of-range stop id can never be produced by sampling, so it would
    // silently never fire. Fail loudly instead of shipping a dead stop token.
    if (id >= vocab_size) {
        throw std::runtime_error("Qwen eos_token_id is outside the vocabulary");
    }
    return static_cast<int>(id);
}

// Accepts both shapes HF uses: a bare id, or a list of them.
std::vector<int> parse_eos_token_ids(const std::string& ckpt_dir,
                                    uint64_t vocab_size) {
    std::string text;
    if (!try_read_file(ckpt_dir + "/generation_config.json", &text)) return {};
    const JsonValue root_value = parse_json(text);
    if (!root_value.is_object()) return {};
    const JsonValue* eos = object_get(root_value.object(), "eos_token_id");
    if (eos == nullptr || eos->is_null()) return {};

    std::vector<int> ids;
    if (eos->is_array()) {
        for (const JsonValue& entry : eos->array()) {
            const int id = checked_token_id(entry, vocab_size);
            if (std::find(ids.begin(), ids.end(), id) == ids.end()) ids.push_back(id);
        }
    } else {
        ids.push_back(checked_token_id(*eos, vocab_size));
    }
    return ids;
}

const JsonObject& text_config(const JsonObject& root) {
    const JsonValue* nested = object_get(root, "text_config");
    if (nested == nullptr) return root;
    if (!nested->is_object()) throw std::runtime_error("Qwen config text_config must be an object");
    return nested->object();
}

const JsonObject* optional_object(const JsonObject& obj, const std::string& key) {
    const JsonValue* value = object_get(obj, key);
    if (value == nullptr || value->is_null()) return nullptr;
    if (!value->is_object()) throw std::runtime_error("Qwen config key must be an object: " + key);
    return &value->object();
}

uint64_t required_u64(const JsonObject& obj, const std::string& key) {
    const JsonValue* value = object_get(obj, key);
    if (value == nullptr || !value->is_number()) throw std::runtime_error("missing Qwen integer: " + key);
    const double number = value->number();
    const double rounded = std::round(number);
    if (number < 0.0 || std::fabs(number - rounded) > 1.0e-9) {
        throw std::runtime_error("Qwen integer is not integral: " + key);
    }
    return static_cast<uint64_t>(rounded);
}

uint64_t optional_u64(const JsonObject& obj, const std::string& key, uint64_t fallback) {
    const JsonValue* value = object_get(obj, key);
    if (value == nullptr || value->is_null()) return fallback;
    if (!value->is_number()) throw std::runtime_error("Qwen config integer has wrong type: " + key);
    const double number = value->number();
    const double rounded = std::round(number);
    if (number < 0.0 || std::fabs(number - rounded) > 1.0e-9) {
        throw std::runtime_error("Qwen integer is not integral: " + key);
    }
    return static_cast<uint64_t>(rounded);
}

double optional_f64(const JsonObject& obj, const std::string& key, double fallback) {
    const JsonValue* value = object_get(obj, key);
    if (value == nullptr || value->is_null()) return fallback;
    if (!value->is_number()) throw std::runtime_error("Qwen config number has wrong type: " + key);
    return value->number();
}

bool required_bool(const JsonObject& obj, const std::string& key) {
    const JsonValue* value = object_get(obj, key);
    if (value == nullptr || !value->is_bool()) throw std::runtime_error("missing Qwen boolean: " + key);
    return value->boolean();
}

bool optional_bool(const JsonObject& obj, const std::string& key, bool fallback) {
    const JsonValue* value = object_get(obj, key);
    if (value == nullptr || value->is_null()) return fallback;
    if (!value->is_bool()) throw std::runtime_error("Qwen config boolean has wrong type: " + key);
    return value->boolean();
}

std::string optional_string(const JsonObject& obj, const std::string& key, const std::string& fallback) {
    const JsonValue* value = object_get(obj, key);
    if (value == nullptr || value->is_null()) return fallback;
    if (!value->is_string()) throw std::runtime_error("Qwen config string has wrong type: " + key);
    return value->string();
}

std::string root_model_type(const JsonObject& root) {
    return optional_string(root, "model_type", "");
}

void validate_positive(double value, const std::string& name) {
    if (!(value > 0.0) || !std::isfinite(value)) {
        throw std::runtime_error("invalid Qwen positive number: " + name);
    }
}

}  // namespace

QwenConfig QwenConfig::from_hf_config(const std::string& ckpt_dir) {
    const JsonValue root_value = parse_json(read_file(ckpt_dir + "/config.json"));
    if (!root_value.is_object()) throw std::runtime_error("Qwen config root must be an object");
    const JsonObject& root = root_value.object();
    const JsonObject& text = text_config(root);

    QwenConfig cfg;
    cfg.architecture = root_model_type(root);
    cfg.model_type = optional_string(text, "model_type", cfg.architecture);
    cfg.vocab_size = required_u64(text, "vocab_size");
    cfg.hidden_size = required_u64(text, "hidden_size");
    cfg.num_hidden_layers = required_u64(text, "num_hidden_layers");
    cfg.max_position_embeddings = required_u64(text, "max_position_embeddings");
    cfg.mtp_num_hidden_layers = optional_u64(text, "mtp_num_hidden_layers", 0);
    cfg.mtp_use_dedicated_embeddings = optional_bool(
        text, "mtp_use_dedicated_embeddings", false);
    cfg.rms_norm_eps = optional_f64(text, "rms_norm_eps", 1.0e-6);
    cfg.partial_rotary_factor = optional_f64(text, "partial_rotary_factor", 1.0);
    cfg.fp8_block_size = 128;

    const JsonObject* rope_parameters = optional_object(text, "rope_parameters");
    if (rope_parameters != nullptr) {
        cfg.rope_theta = optional_f64(*rope_parameters, "rope_theta", cfg.rope_theta);
        cfg.partial_rotary_factor = optional_f64(*rope_parameters, "partial_rotary_factor", cfg.partial_rotary_factor);
    } else {
        cfg.rope_theta = optional_f64(text, "rope_theta", cfg.rope_theta);
    }

    cfg.linear_attention.key_heads = required_u64(text, "linear_num_key_heads");
    cfg.linear_attention.value_heads = required_u64(text, "linear_num_value_heads");
    cfg.linear_attention.key_head_dim = required_u64(text, "linear_key_head_dim");
    cfg.linear_attention.value_head_dim = required_u64(text, "linear_value_head_dim");
    cfg.linear_attention.conv_kernel_dim = required_u64(text, "linear_conv_kernel_dim");

    cfg.full_attention.num_heads = required_u64(text, "num_attention_heads");
    cfg.full_attention.num_key_value_heads = required_u64(text, "num_key_value_heads");
    cfg.full_attention.head_dim = optional_u64(text, "head_dim", cfg.hidden_size / cfg.full_attention.num_heads);
    cfg.full_attention.output_gate = required_bool(text, "attn_output_gate");

    cfg.mlp.intermediate_size = optional_u64(text, "intermediate_size", 0);
    if (cfg.mlp.intermediate_size == 0) {
        throw std::runtime_error("missing Qwen dense MLP intermediate_size");
    }

    const JsonValue* raw_layers = object_get(text, "layer_types");
    if (raw_layers == nullptr || !raw_layers->is_array() || raw_layers->array().size() != cfg.num_hidden_layers) {
        throw std::runtime_error("Qwen layer_types must contain one entry per layer");
    }
    cfg.layer_types.reserve(cfg.num_hidden_layers);
    for (size_t i = 0; i < raw_layers->array().size(); ++i) {
        const JsonValue& value = raw_layers->array()[i];
        if (!value.is_string()) throw std::runtime_error("Qwen layer_types entry is not a string");
        if (value.string() == "linear_attention") {
            cfg.layer_types.push_back(QwenLayerType::LinearAttention);
        } else if (value.string() == "full_attention") {
            cfg.layer_types.push_back(QwenLayerType::FullAttention);
        } else {
            throw std::runtime_error("unsupported Qwen layer type at " + std::to_string(i) + ": " + value.string());
        }
    }

    cfg.eos_token_ids = parse_eos_token_ids(ckpt_dir, cfg.vocab_size);

    if (!cfg.is_qwen3_5()) throw std::runtime_error("config is not a Qwen3.5 text checkpoint");
    if (cfg.hidden_size == 0 || cfg.num_hidden_layers == 0 || cfg.vocab_size == 0 || cfg.max_position_embeddings == 0) {
        throw std::runtime_error("invalid zero-sized Qwen config");
    }
    validate_positive(cfg.rms_norm_eps, "rms_norm_eps");
    validate_positive(cfg.rope_theta, "rope_theta");
    if (!(cfg.partial_rotary_factor > 0.0 && cfg.partial_rotary_factor <= 1.0)) {
        throw std::runtime_error("Qwen partial_rotary_factor must be in (0, 1]");
    }
    if (cfg.linear_attention.key_heads == 0 || cfg.linear_attention.value_heads == 0 ||
        cfg.linear_attention.key_head_dim == 0 || cfg.linear_attention.value_head_dim == 0 ||
        cfg.linear_attention.conv_kernel_dim == 0) {
        throw std::runtime_error("invalid Qwen linear-attention dimensions");
    }
    if (cfg.full_attention.num_heads == 0 || cfg.full_attention.num_key_value_heads == 0 || cfg.full_attention.head_dim == 0) {
        throw std::runtime_error("invalid Qwen full-attention dimensions");
    }
    if (cfg.full_attention.num_heads % cfg.full_attention.num_key_value_heads != 0) {
        throw std::runtime_error("Qwen attention heads must be divisible by key/value heads");
    }
    if (cfg.partial_rotary_dim() == 0 || cfg.partial_rotary_dim() > cfg.full_attention.head_dim ||
        (cfg.partial_rotary_dim() % 2) != 0) {
        throw std::runtime_error("Qwen partial rotary dimension must be a positive even head dimension");
    }
    if (cfg.linear_attention.value_heads % cfg.linear_attention.key_heads != 0) {
        throw std::runtime_error("Qwen linear value heads must be divisible by key heads");
    }
    return cfg;
}

bool QwenConfig::is_qwen3_5() const {
    return architecture == "qwen3_5" || architecture == "qwen3_5_text" ||
           model_type == "qwen3_5" || model_type == "qwen3_5_text";
}

uint64_t QwenConfig::linear_attention_layers() const {
    uint64_t count = 0;
    for (QwenLayerType type : layer_types) {
        if (type == QwenLayerType::LinearAttention) ++count;
    }
    return count;
}

uint64_t QwenConfig::full_attention_layers() const {
    uint64_t count = 0;
    for (QwenLayerType type : layer_types) {
        if (type == QwenLayerType::FullAttention) ++count;
    }
    return count;
}

uint64_t QwenConfig::partial_rotary_dim() const {
    return static_cast<uint64_t>(std::llround(static_cast<double>(full_attention.head_dim) * partial_rotary_factor));
}

std::string QwenConfig::layer_type_name(uint64_t layer) const {
    if (layer >= layer_types.size()) throw std::out_of_range("Qwen layer index out of range");
    return layer_types[layer] == QwenLayerType::LinearAttention ? "linear_attention" : "full_attention";
}

std::string QwenConfig::to_string() const {
    std::ostringstream out;
    out << "architecture=" << architecture << '\n'
        << "model_type=" << model_type << '\n'
        << "vocab_size=" << vocab_size << '\n'
        << "hidden_size=" << hidden_size << '\n'
        << "num_hidden_layers=" << num_hidden_layers << '\n'
        << "max_position_embeddings=" << max_position_embeddings << '\n'
        << "mtp_num_hidden_layers=" << mtp_num_hidden_layers << '\n'
        << "mtp_use_dedicated_embeddings=" << (mtp_use_dedicated_embeddings ? 1 : 0) << '\n'
        << "rms_norm_eps=" << rms_norm_eps << '\n'
        << "rope_theta=" << rope_theta << '\n'
        << "partial_rotary_factor=" << partial_rotary_factor << '\n'
        << "partial_rotary_dim=" << partial_rotary_dim() << '\n'
        << "linear_key_heads=" << linear_attention.key_heads << '\n'
        << "linear_value_heads=" << linear_attention.value_heads << '\n'
        << "linear_key_head_dim=" << linear_attention.key_head_dim << '\n'
        << "linear_value_head_dim=" << linear_attention.value_head_dim << '\n'
        << "linear_conv_kernel_dim=" << linear_attention.conv_kernel_dim << '\n'
        << "full_num_heads=" << full_attention.num_heads << '\n'
        << "full_num_key_value_heads=" << full_attention.num_key_value_heads << '\n'
        << "full_head_dim=" << full_attention.head_dim << '\n'
        << "attn_output_gate=" << (full_attention.output_gate ? 1 : 0) << '\n'
        << "intermediate_size=" << mlp.intermediate_size << '\n'
        << "linear_attention_layers=" << linear_attention_layers() << '\n'
        << "full_attention_layers=" << full_attention_layers() << '\n';
    return out.str();
}

bool is_qwen3_5_checkpoint(const std::string& ckpt_dir) {
    const JsonValue root_value = parse_json(read_file(ckpt_dir + "/config.json"));
    if (!root_value.is_object()) return false;
    const JsonObject& root = root_value.object();
    const std::string arch = root_model_type(root);
    if (arch == "qwen3_5" || arch == "qwen3_5_text") return true;
    const JsonValue* nested = object_get(root, "text_config");
    if (nested == nullptr || !nested->is_object()) return false;
    const std::string type = optional_string(nested->object(), "model_type", "");
    return type == "qwen3_5" || type == "qwen3_5_text";
}

}  // namespace pocket
