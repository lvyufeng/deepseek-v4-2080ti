#include "qwen_dflash2.hpp"

#include "device_runtime.hpp"
#include "json_lite.hpp"
#include "qwen_ops.hpp"
#include "tp_comm.hpp"


#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace pocket {
namespace {

const JsonValue& required(const JsonObject& object, const std::string& key) {
    const JsonValue* value = object_get(object, key);
    if (value == nullptr) throw std::runtime_error("Qwen DFlash2 config is missing " + key);
    return *value;
}

const JsonValue* optional(const JsonObject& object, const std::string& key) {
    return object_get(object, key);
}

int number_int(const JsonValue& value, const std::string& key) {
    if (!value.is_number()) throw std::runtime_error("Qwen DFlash2 field is not numeric: " + key);
    return static_cast<int>(value.number());
}

int required_int(const JsonObject& object, const std::string& key) {
    return number_int(required(object, key), key);
}

int optional_int(const JsonObject& object, const std::string& key, int fallback) {
    const JsonValue* value = optional(object, key);
    return value == nullptr ? fallback : number_int(*value, key);
}

double number_value(const JsonValue& value, const std::string& key) {
    if (!value.is_number()) throw std::runtime_error("Qwen DFlash2 field is not numeric: " + key);
    return value.number();
}

double optional_number(const JsonObject& object, const std::string& key, double fallback) {
    const JsonValue* value = optional(object, key);
    return value == nullptr ? fallback : number_value(*value, key);
}

bool optional_bool(const JsonObject& object, const std::string& key, bool fallback) {
    const JsonValue* value = optional(object, key);
    if (value == nullptr) return fallback;
    if (!value->is_bool()) throw std::runtime_error("Qwen DFlash2 field is not boolean: " + key);
    return value->boolean();
}

std::string optional_string(const JsonObject& object, const std::string& key,
                           const std::string& fallback) {
    const JsonValue* value = optional(object, key);
    if (value == nullptr) return fallback;
    if (!value->is_string()) throw std::runtime_error("Qwen DFlash2 field is not textual: " + key);
    return value->string();
}

std::vector<int> required_int_array(const JsonObject& object, const std::string& key) {
    const JsonValue& value = required(object, key);
    if (!value.is_array()) throw std::runtime_error("Qwen DFlash2 field is not an array: " + key);
    std::vector<int> result;
    result.reserve(value.array().size());
    for (const JsonValue& item : value.array()) result.push_back(number_int(item, key));
    return result;
}

std::vector<std::string> required_string_array(const JsonObject& object,
                                               const std::string& key) {
    const JsonValue& value = required(object, key);
    if (!value.is_array()) throw std::runtime_error("Qwen DFlash2 field is not an array: " + key);
    std::vector<std::string> result;
    result.reserve(value.array().size());
    for (const JsonValue& item : value.array()) {
        if (!item.is_string()) throw std::runtime_error("Qwen DFlash2 array item is not textual: " + key);
        result.push_back(item.string());
    }
    return result;
}

std::string read_text(const std::string& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open Qwen DFlash2 config: " + path);
    std::ostringstream output;
    output << input.rdbuf();
    return output.str();
}

std::string shape_string(const std::vector<uint64_t>& shape) {
    std::ostringstream output;
    output << '[';
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i != 0) output << ',';
        output << shape[i];
    }
    output << ']';
    return output.str();
}

// Tensor-parallel drafter sharding. Default on; set POCKETLLM_DFLASH2_SHARD=0 to fall
// back to the replicated layout for one-click rollback or parity comparison.
bool dflash2_shard_enabled() {
    const char* value = std::getenv("POCKETLLM_DFLASH2_SHARD");
    if (value == nullptr || *value == '\0') return true;
    const std::string text(value);
    return text != "0" && text != "false" && text != "FALSE" && text != "off" &&
           text != "OFF";
}

void validate_config(const QwenDFlash2Config& config) {
    if (config.architecture != "DFlash2DraftModel" || config.block_size != 8 ||
        config.conv_group_size != 16 || config.conv_kernel_size != 2 ||
        config.mask_token_id != 248070 || config.selector_rank != 256 ||
        config.selector_top_k != 16 || config.hidden_size != 5120 ||
        config.intermediate_size != 17408 || config.num_hidden_layers != 5 ||
        config.num_attention_heads != 32 || config.num_key_value_heads != 8 ||
        config.head_dim != 128 || config.vocab_size != 248320 ||
        config.num_target_layers != 64 || config.max_position_embeddings <= 0 ||
        config.sliding_window != 2048 || config.rms_norm_eps <= 0.0f ||
        config.rope_theta != 10000000.0 || config.attention_bias || config.is_causal ||
        config.hidden_act != "silu" || config.rope_type != "default" ||
        config.target_layer_ids != std::vector<int>({5, 19, 33, 47, 61}) ||
        config.layer_types != std::vector<std::string>(5, "sliding_attention")) {
        throw std::runtime_error("unsupported Qwen DFlash2 checkpoint architecture");
    }
    if (config.hidden_size % config.conv_group_size != 0 ||
        config.num_attention_heads % config.num_key_value_heads != 0 ||
        config.head_dim * config.num_attention_heads != 4096) {
        throw std::runtime_error("invalid Qwen DFlash2 dimensions");
    }
}

struct DeviceLinear {
    QwenDeviceTensor weight;
};

struct DeviceConv {
    QwenDeviceTensor base;
    DeviceLinear projection;
};

struct DeviceLayer {
    QwenDeviceTensor input_norm;
    QwenDeviceTensor post_norm;
    DeviceLinear q;
    DeviceLinear k;
    DeviceLinear v;
    DeviceLinear out;
    QwenDeviceTensor q_norm;
    QwenDeviceTensor k_norm;
    DeviceLinear gate;
    DeviceLinear up;
    DeviceLinear down;
    DeviceConv attention_conv;
    DeviceConv mlp_conv;
    QwenDeviceTensor context_k;
    QwenDeviceTensor context_v;
};

void check_device(bool ok, const char* what) {
    if (!ok) {
        const std::string detail = device_last_error();
        throw std::runtime_error(detail.empty() ? std::string(what)
                                                : std::string(what) + ": " + detail);
    }
}

void require_launch(bool ok, const char* what) {
    if (!ok) throw std::runtime_error(std::string("Qwen DFlash2 device launch failed: ") + what);
}

class DFlash2StageProfiler {
public:
    DFlash2StageProfiler(int device, int rank)
        : device_(device), rank_(rank) {
        const char* value = std::getenv("POCKETLLM_DFLASH2_PROFILE");
        enabled_ = value != nullptr && value[0] != '\0' &&
                   std::string(value) != "0";
    }

    ~DFlash2StageProfiler() {
        for (const Entry& entry : entries_) {
            if (entry.start != nullptr) event_destroy(entry.start);
            if (entry.stop != nullptr) event_destroy(entry.stop);
        }
    }

    void begin(const std::string& name) {
        if (!enabled_) return;
        if (active_) throw std::runtime_error("nested DFlash2 profile stages");
        Entry entry;
        entry.name = name;
        entry.start = event_create(/*with_timing=*/true);
        check_device(entry.start != nullptr, "event_create DFlash2 profile start");
        entry.stop = event_create(/*with_timing=*/true);
        check_device(entry.stop != nullptr, "event_create DFlash2 profile stop");
        entry.host_start = std::chrono::steady_clock::now();
        check_device(event_record(entry.start, nullptr),
                     "event_record DFlash2 profile start");
        entries_.push_back(std::move(entry));
        active_ = true;
    }

    void end() {
        if (!enabled_) return;
        if (!active_ || entries_.empty()) {
            throw std::runtime_error("unbalanced DFlash2 profile stage");
        }
        Entry& entry = entries_.back();
        check_device(event_record(entry.stop, nullptr),
                     "event_record DFlash2 profile stop");
        entry.host_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - entry.host_start).count();
        active_ = false;
    }

    void flush(const char* operation) {
        if (!enabled_ || entries_.empty()) return;
        if (active_) throw std::runtime_error("unfinished DFlash2 profile stage");
        check_device(device_synchronize(), "sync DFlash2 profile");
        std::cout << "dflash2_profile rank=" << rank_
                  << " op=" << operation << " stages=" << entries_.size()
                  << "\n";
        for (Entry& entry : entries_) {
            float device_ms = 0.0f;
            check_device(event_elapsed_ms(entry.start, entry.stop, &device_ms),
                         "event_elapsed_ms DFlash2 profile");
            std::cout << "  " << entry.name << " device_ms=" << device_ms
                      << " host_ms=" << entry.host_ms << "\n";
            event_destroy(entry.start);
            event_destroy(entry.stop);
            entry.start = nullptr;
            entry.stop = nullptr;
        }
        entries_.clear();
    }

private:
    struct Entry {
        std::string name;
        void* start = nullptr;
        void* stop = nullptr;
        std::chrono::steady_clock::time_point host_start;
        double host_ms = 0.0;
    };

    int device_ = 0;
    int rank_ = 0;
    bool enabled_ = false;
    bool active_ = false;
    std::vector<Entry> entries_;
};

void allocate(QwenDeviceTensor& tensor, size_t elements,
              const std::vector<uint64_t>& shape, SafeDType dtype) {
    const size_t bytes = elements * safe_dtype_size(dtype);
    if (bytes == 0) throw std::runtime_error("Qwen DFlash2 attempted an empty allocation");
    if (tensor.data != nullptr && tensor.capacity >= bytes) {
        tensor.device_dtype = dtype;
        tensor.shape = shape;
        tensor.nbytes = bytes;
        return;
    }
    if (tensor.data != nullptr) {
        device_free(tensor.data);
        tensor.data = nullptr;
    }
    tensor.data = device_malloc(bytes);
    check_device(tensor.data != nullptr, "device_malloc DFlash2 workspace");
    tensor.device_dtype = dtype;
    tensor.shape = shape;
    tensor.nbytes = bytes;
    tensor.capacity = bytes;
}

void allocate_half(QwenDeviceTensor& tensor, size_t elements,
                   const std::vector<uint64_t>& shape) {
    allocate(tensor, elements, shape, SafeDType::F16);
}

void allocate_int(QwenDeviceTensor& tensor, size_t elements,
                  const std::vector<uint64_t>& shape) {
    allocate(tensor, elements, shape, SafeDType::I64);
}

void allocate_float(QwenDeviceTensor& tensor, size_t elements,
                    const std::vector<uint64_t>& shape) {
    allocate(tensor, elements, shape, SafeDType::F32);
}

void allocate_i32(QwenDeviceTensor& tensor, size_t elements,
                  const std::vector<uint64_t>& shape) {
    allocate(tensor, elements, shape, SafeDType::I64);
}

}  // namespace

QwenDFlash2Config QwenDFlash2Config::from_directory(const std::string& checkpoint_dir) {
    const JsonValue root = parse_json(read_text(checkpoint_dir + "/config.json"));
    if (!root.is_object()) throw std::runtime_error("Qwen DFlash2 config root is not an object");
    const JsonObject& object = root.object();
    const JsonObject* dflash = &object;
    if (const JsonValue* value = optional(object, "dflash_config")) {
        if (!value->is_object()) throw std::runtime_error("Qwen DFlash2 dflash_config is malformed");
        dflash = &value->object();
    }
    QwenDFlash2Config config;
    const JsonValue* architecture = optional(object, "architectures");
    if (architecture == nullptr || !architecture->is_array() || architecture->array().empty() ||
        !architecture->array().front().is_string()) {
        throw std::runtime_error("Qwen DFlash2 architectures is malformed");
    }
    config.architecture = architecture->array().front().string();
    config.block_size = optional_int(*dflash, "block_size", optional_int(object, "block_size", 0));
    config.conv_group_size = required_int(*dflash, "conv_group_size");
    config.conv_kernel_size = required_int(*dflash, "conv_kernel_size");
    config.mask_token_id = required_int(*dflash, "mask_token_id");
    config.selector_rank = required_int(*dflash, "selector_rank");
    config.selector_top_k = required_int(*dflash, "selector_top_k");
    config.target_layer_ids = required_int_array(*dflash, "target_layer_ids");
    config.hidden_size = required_int(object, "hidden_size");
    config.intermediate_size = required_int(object, "intermediate_size");
    config.num_hidden_layers = required_int(object, "num_hidden_layers");
    config.num_attention_heads = required_int(object, "num_attention_heads");
    config.num_key_value_heads = required_int(object, "num_key_value_heads");
    config.head_dim = required_int(object, "head_dim");
    config.vocab_size = required_int(object, "vocab_size");
    config.num_target_layers = required_int(object, "num_target_layers");
    config.max_position_embeddings = required_int(object, "max_position_embeddings");
    config.sliding_window = required_int(object, "sliding_window");
    config.rms_norm_eps = static_cast<float>(number_value(required(object, "rms_norm_eps"), "rms_norm_eps"));
    config.hidden_act = optional_string(object, "hidden_act", "");
    config.attention_bias = optional_bool(object, "attention_bias", false);
    config.is_causal = optional_bool(object, "is_causal", false);
    config.layer_types = required_string_array(object, "layer_types");
    const JsonValue* rope_value = optional(object, "rope_parameters");
    if (rope_value != nullptr) {
        if (!rope_value->is_object()) throw std::runtime_error("Qwen DFlash2 rope_parameters is malformed");
        const JsonObject& rope = rope_value->object();
        config.rope_theta = optional_number(rope, "rope_theta", config.rope_theta);
        config.rope_type = optional_string(rope, "rope_type", "default");
    } else {
        config.rope_theta = optional_number(object, "rope_theta", config.rope_theta);
        config.rope_type = optional_string(object, "rope_type", "default");
    }
    validate_config(config);
    return config;
}

void QwenDFlash2Config::validate_for_target(uint64_t target_hidden_size,
                                            uint64_t target_vocab_size,
                                            uint64_t target_layers) const {
    if (static_cast<uint64_t>(hidden_size) != target_hidden_size ||
        static_cast<uint64_t>(vocab_size) != target_vocab_size ||
        static_cast<uint64_t>(num_target_layers) != target_layers) {
        throw std::runtime_error("Qwen DFlash2 checkpoint does not match target model");
    }
}

QwenTensorRef QwenDFlash2WeightMap::require_tensor(
    const std::string& name, const std::vector<uint64_t>& shape,
    QwenShardRule rule, int shard_dim) const {
    const std::string* shard_name = index_.shard_for_tensor(name);
    if (shard_name == nullptr) throw std::runtime_error("missing Qwen DFlash2 tensor: " + name);
    SafeTensorsShard shard(index_.shard_path(*shard_name));
    const SafeTensorInfo* info = shard.find_tensor(name);
    if (info == nullptr || info->dtype != SafeDType::BF16 || info->shape != shape) {
        throw std::runtime_error("invalid Qwen DFlash2 tensor " + name + ", expected BF16 " + shape_string(shape));
    }
    QwenTensorRef ref;
    ref.name = name;
    ref.shard_name = *shard_name;
    ref.dtype = SafeDType::BF16;
    ref.device_dtype = SafeDType::F16;
    ref.full_shape = shape;
    ref.local_shape = shape;
    ref.rule = QwenShardRule::Replicated;
    ref.nbytes = info->nbytes;
    ref.device_nbytes = safe_tensor_numel(shape) * sizeof(uint16_t);
    ref.found = true;
    if (!shard_ || rule == QwenShardRule::Replicated || tp_world_ == 1) return ref;
    if (shard_dim < 0 || static_cast<size_t>(shard_dim) >= shape.size()) {
        throw std::runtime_error("invalid Qwen DFlash2 shard dimension for " + name);
    }
    const uint64_t total = shape[static_cast<size_t>(shard_dim)];
    if (total % static_cast<uint64_t>(tp_world_) != 0) {
        throw std::runtime_error("Qwen DFlash2 tensor is not divisible by TP world: " + name);
    }
    ref.rule = rule;
    ref.shard_dim = shard_dim;
    ref.shard_size = total / static_cast<uint64_t>(tp_world_);
    ref.shard_start = ref.shard_size * static_cast<uint64_t>(tp_rank_);
    ref.local_shape[static_cast<size_t>(shard_dim)] = ref.shard_size;
    ref.nbytes = safe_tensor_numel(ref.local_shape) * sizeof(uint16_t);
    ref.device_nbytes = ref.nbytes;
    return ref;
}

QwenLinearRef QwenDFlash2WeightMap::require_linear(
    const std::string& name, const std::vector<uint64_t>& shape,
    QwenShardRule rule, int shard_dim) const {
    QwenLinearRef result;
    result.weight = require_tensor(name, shape, rule, shard_dim);
    return result;
}

void QwenDFlash2WeightMap::record(const QwenTensorRef& ref) {
    if (!ref.found) return;
    ++tensor_count_;
    local_device_bytes_ += ref.device_nbytes;
}

QwenDFlash2WeightMap::QwenDFlash2WeightMap(const SafeTensorsIndex& index,
                                           const QwenDFlash2Config& config,
                                           int tp_world, int tp_rank)
    : index_(index), config_(config), tp_world_(tp_world), tp_rank_(tp_rank) {
    if (tp_world <= 0 || tp_rank < 0 || tp_rank >= tp_world) throw std::runtime_error("invalid Qwen DFlash2 TP rank");
    // The drafter was originally replicated on every rank, so each rank streamed
    // the whole 3.1 GiB of layer weight per proposal while the target streamed
    // only its quarter. Splitting the projections the same way the target does
    // trades that traffic for two small collectives per layer, which measures at
    // 0.046 ms per 8x5120 FP16 reduce against roughly 4 ms of saved reads.
    shard_ = tp_world > 1 && dflash2_shard_enabled();
    if (shard_) {
        const int heads = config.num_attention_heads;
        const int kv_heads = config.num_key_value_heads;
        if (heads % tp_world != 0 || kv_heads % tp_world != 0 ||
            config.intermediate_size % tp_world != 0) {
            shard_ = false;
        }
    }
    const uint64_t hidden = static_cast<uint64_t>(config.hidden_size);
    const uint64_t intermediate = static_cast<uint64_t>(config.intermediate_size);
    const uint64_t q_dim = static_cast<uint64_t>(config.num_attention_heads) * config.head_dim;
    const uint64_t kv_dim = static_cast<uint64_t>(config.num_key_value_heads) * config.head_dim;
    const uint64_t groups = hidden / config.conv_group_size;
    projector_ = require_tensor("fc.weight", {hidden, hidden * config.target_layer_ids.size()});
    hidden_norm_ = require_tensor("hidden_norm.weight", {hidden});
    final_norm_ = require_tensor("norm.weight", {hidden});
    predecessor_codebook_ = require_tensor("candidate_selector.predecessor_codebook", {static_cast<uint64_t>(config.vocab_size), static_cast<uint64_t>(config.selector_rank)});
    successor_codebook_ = require_tensor("candidate_selector.successor_codebook", {static_cast<uint64_t>(config.vocab_size), static_cast<uint64_t>(config.selector_rank)});
    selector_projection_ = require_tensor("candidate_selector.hidden_projection.weight", {static_cast<uint64_t>(config.selector_rank), hidden});
    for (const QwenTensorRef* ref : {&projector_, &hidden_norm_, &final_norm_, &predecessor_codebook_, &successor_codebook_, &selector_projection_}) record(*ref);
    layers_.resize(static_cast<size_t>(config.num_hidden_layers));
    for (int layer_id = 0; layer_id < config.num_hidden_layers; ++layer_id) {
        const std::string prefix = "layers." + std::to_string(layer_id) + ".";
        QwenDFlash2LayerWeights& layer = layers_[static_cast<size_t>(layer_id)];
        layer.input_layernorm = require_tensor(prefix + "input_layernorm.weight", {hidden});
        layer.post_attention_layernorm = require_tensor(prefix + "post_attention_layernorm.weight", {hidden});
        // Q/K/V split by whole head over dim 0 so each rank owns entire heads and
        // the per-head RMSNorm/RoPE stay local. O and MLP down split their
        // reduction dimension and finish with one all-reduce each.
        layer.q_proj = require_linear(prefix + "self_attn.q_proj.weight", {q_dim, hidden}, QwenShardRule::ColumnParallel, 0);
        layer.k_proj = require_linear(prefix + "self_attn.k_proj.weight", {kv_dim, hidden}, QwenShardRule::ColumnParallel, 0);
        layer.v_proj = require_linear(prefix + "self_attn.v_proj.weight", {kv_dim, hidden}, QwenShardRule::ColumnParallel, 0);
        layer.o_proj = require_linear(prefix + "self_attn.o_proj.weight", {hidden, q_dim}, QwenShardRule::RowParallel, 1);
        layer.q_norm = require_tensor(prefix + "self_attn.q_norm.weight", {static_cast<uint64_t>(config.head_dim)});
        layer.k_norm = require_tensor(prefix + "self_attn.k_norm.weight", {static_cast<uint64_t>(config.head_dim)});
        layer.gate_proj = require_linear(prefix + "mlp.gate_proj.weight", {intermediate, hidden}, QwenShardRule::ColumnParallel, 0);
        layer.up_proj = require_linear(prefix + "mlp.up_proj.weight", {intermediate, hidden}, QwenShardRule::ColumnParallel, 0);
        layer.down_proj = require_linear(prefix + "mlp.down_proj.weight", {hidden, intermediate}, QwenShardRule::RowParallel, 1);
        for (QwenDFlash2ConvWeights* conv : {&layer.attention_conv, &layer.mlp_conv}) {
            const std::string name = conv == &layer.attention_conv ? "attention_conv" : "mlp_conv";
            conv->base_kernel = require_tensor(prefix + name + ".base_kernel", {2, static_cast<uint64_t>(config.conv_kernel_size), hidden});
            conv->kernel_projection = require_linear(prefix + name + ".kernel_projection.weight", {2 * static_cast<uint64_t>(config.conv_kernel_size) * groups, hidden});
        }
        for (const QwenTensorRef* ref : {&layer.input_layernorm, &layer.post_attention_layernorm, &layer.q_proj.weight, &layer.k_proj.weight, &layer.v_proj.weight, &layer.o_proj.weight, &layer.q_norm, &layer.k_norm, &layer.gate_proj.weight, &layer.up_proj.weight, &layer.down_proj.weight, &layer.attention_conv.base_kernel, &layer.attention_conv.kernel_projection.weight, &layer.mlp_conv.base_kernel, &layer.mlp_conv.kernel_projection.weight}) record(*ref);
    }
    if (tensor_count_ != 81 || tensor_count_ != index_.tensor_count()) throw std::runtime_error("Qwen DFlash2 checkpoint must contain exactly the 81 expected tensors");
}

struct QwenDFlash2Runtime::Impl {
    std::string checkpoint_dir;
    QwenDFlash2Config config;
    const QwenDFlash2WeightMap& weight_map;
    SafeTensorsIndex index;
    const QwenDeviceTensor& target_embedding;
    QwenTargetHeadAdapter target_head;
    int tp_world = 1;
    int tp_rank = 0;
    int device = 0;
    std::string nccl_id_path;
    int max_context = 0;
    int committed = 0;
    // Raw target taps can be appended while target prefill/verification runs;
    // projection and DFlash2 K/V preparation are flushed once before drafting.
    int prepared_context = 0;
    uint64_t weight_bytes = 0;
    uint64_t cache_bytes = 0;
    std::vector<DeviceLayer> layers;
    QwenDeviceTensor projector;
    QwenDeviceTensor hidden_norm;
    QwenDeviceTensor final_norm;
    QwenDeviceTensor predecessor;
    QwenDeviceTensor successor;
    QwenDeviceTensor selector_projection;
    QwenDeviceTensor context_taps;
    QwenDeviceTensor tokens;
    QwenDeviceTensor hidden_a;
    QwenDeviceTensor hidden_b;
    QwenDeviceTensor normalized;
    QwenDeviceTensor q;
    QwenDeviceTensor k;
    QwenDeviceTensor v;
    QwenDeviceTensor attention;
    QwenDeviceTensor residual_f32;
    QwenDeviceTensor branch_f32;
    QwenDeviceTensor conv_result_f32;
    QwenDeviceTensor conv_hidden;
    QwenDeviceTensor gate;
    QwenDeviceTensor up;
    QwenDeviceTensor intermediate;
    QwenDeviceTensor logits;
    // prepare_context() scratch. These were function locals, so every proposal
    // paid five device malloc/free pairs (~13 ms) before any drafting work.
    QwenDeviceTensor context_projected;
    QwenDeviceTensor context_normalized;
    QwenDeviceTensor context_k;
    QwenDeviceTensor context_v;
    QwenDeviceTensor context_knorm;
    QwenDeviceTensor local_candidates;
    QwenDeviceTensor local_topk_partial_tokens;
    QwenDeviceTensor local_topk_partial_values;
    QwenDeviceTensor local_unary;
    QwenDeviceTensor global_candidates;
    QwenDeviceTensor global_unary;
    QwenDeviceTensor path;
    QwenDeviceTensor selector_projected;
    QwenDeviceTensor dynamic;
    QwenDFlash2DebugCallback debug_callback;
    DFlash2StageProfiler profiler;
    bool grouped_attention = false;
    bool fused_swiglu = false;
    bool cublas_fp32 = false;
    bool small_batch_gemm = false;
    bool split_local_topk = false;
    bool viterbi_selector = false;
    // Denoising passes per proposal. One pass is the shipped behaviour; extra
    // passes re-run the block with the previous drafts replacing the mask tokens.
    int refine_passes = 1;
    // Local projection extents. These equal the config values when the drafter is
    // replicated and the per-rank shard when it is tensor-parallel, so every
    // launch below is written against the local shapes.
    bool sharded = false;
    int local_q_heads = 0;
    int local_kv_heads = 0;
    int local_q_dim = 0;
    int local_kv_dim = 0;
    int local_intermediate = 0;

    static bool env_enabled(const char* name) {
        const char* value = std::getenv(name);
        if (value == nullptr || *value == '\0') return false;
        return std::string(value) != "0" && std::string(value) != "false" &&
               std::string(value) != "FALSE" && std::string(value) != "off" &&
               std::string(value) != "OFF";
    }

    bool use_grouped_attention() const {
        // The grouped kernel needs four Q heads per KV head and a 128 head dim.
        // The head-wise shard divides both counts by the TP world, so the ratio
        // is preserved and the kernel applies to either layout.
        return grouped_attention && local_kv_heads > 0 &&
               local_q_heads == local_kv_heads * 4 && config.head_dim == 128;
    }

    Impl(const std::string& checkpoint_dir_, const QwenDFlash2Config& config_,
         const QwenDFlash2WeightMap& weights_, const QwenDeviceTensor& embedding,
         QwenTargetHeadAdapter target_head_, int world, int rank, int device_,
         std::string id_path, int max_context_)
        : checkpoint_dir(checkpoint_dir_), config(config_), weight_map(weights_),
          index(SafeTensorsIndex::from_single_file(checkpoint_dir_)),
          target_embedding(embedding), target_head(target_head_),
          tp_world(world), tp_rank(rank), device(device_),
          nccl_id_path(std::move(id_path)), max_context(max_context_),
          profiler(device_, rank) {
        if (device < 0 || max_context <= 0 ||
            target_embedding.device_dtype != SafeDType::F16 ||
            target_embedding.shape.size() != 2 ||
            target_embedding.shape[1] != static_cast<uint64_t>(config.hidden_size) ||
            !target_head.valid() || target_head.hidden_size != config.hidden_size) {
            throw std::runtime_error("invalid Qwen DFlash2 runtime target tensors");
        }
        check_device(device_set(device), "select Qwen DFlash2 device");
        grouped_attention = env_enabled("POCKETLLM_DFLASH2_GROUPED_ATTN");
        fused_swiglu = env_enabled("POCKETLLM_DFLASH2_FUSED_SWIGLU");
        cublas_fp32 = env_enabled("POCKETLLM_DFLASH2_CUBLAS_FP32");
        target_head.cublas_fp32 = cublas_fp32;
        small_batch_gemm = env_enabled("POCKETLLM_DFLASH2_SMALL_BATCH_GEMM");
        // The single-block top-k launches one block per draft row, leaving a
        // 68-SM device almost idle. The split path partitions each row's shard
        // and merges with the identical comparator.
        split_local_topk = env_enabled("POCKETLLM_DFLASH2_SPLIT_TOPK");
        // Exact MAP over the selector chain instead of greedy argmax. Draft
        // content is re-verified by the target, so this can only move acceptance.
        viterbi_selector = env_enabled("POCKETLLM_DFLASH2_VITERBI_SELECTOR");
        if (const char* passes = std::getenv("POCKETLLM_DFLASH2_REFINE_PASSES")) {
            const int parsed = std::atoi(passes);
            if (parsed > 1) refine_passes = parsed;
        }
        if (small_batch_gemm && !cublas_fp32) {
            small_batch_gemm = false;
        }
        if (fused_swiglu && (config.block_size != 8 || config.hidden_size % 4 != 0)) {
            fused_swiglu = false;
        }
        sharded = weight_map.sharded();
        if (sharded && tp_world <= 1) throw std::runtime_error("Qwen DFlash2 shard requires TP world > 1");
        const int divisor = sharded ? tp_world : 1;
        local_q_heads = config.num_attention_heads / divisor;
        local_kv_heads = config.num_key_value_heads / divisor;
        local_q_dim = local_q_heads * config.head_dim;
        local_kv_dim = local_kv_heads * config.head_dim;
        local_intermediate = config.intermediate_size / divisor;
        if (local_q_heads <= 0 || local_kv_heads <= 0 || local_intermediate <= 0 ||
            local_q_heads % local_kv_heads != 0) {
            throw std::runtime_error("invalid Qwen DFlash2 local shard extents");
        }
        projector = qwen_upload_tensor(index, weight_map.projector());
        hidden_norm = qwen_upload_tensor(index, weight_map.hidden_norm());
        final_norm = qwen_upload_tensor(index, weight_map.final_norm());
        predecessor = qwen_upload_tensor(index, weight_map.predecessor_codebook());
        successor = qwen_upload_tensor(index, weight_map.successor_codebook());
        selector_projection = qwen_upload_tensor(index, weight_map.selector_projection());
        weight_bytes = weight_map.local_device_bytes();
        const size_t tap_elements = static_cast<size_t>(max_context) * config.hidden_size * config.target_layer_ids.size();
        allocate_half(context_taps, tap_elements, {static_cast<uint64_t>(max_context), static_cast<uint64_t>(config.hidden_size * config.target_layer_ids.size())});
        layers.reserve(weight_map.layers().size());
        const size_t context_elements = static_cast<size_t>(max_context) * local_kv_dim;
        for (const QwenDFlash2LayerWeights& source : weight_map.layers()) {
            DeviceLayer layer;
            layer.input_norm = qwen_upload_tensor(index, source.input_layernorm);
            layer.post_norm = qwen_upload_tensor(index, source.post_attention_layernorm);
            layer.q.weight = qwen_upload_tensor(index, source.q_proj.weight);
            layer.k.weight = qwen_upload_tensor(index, source.k_proj.weight);
            layer.v.weight = qwen_upload_tensor(index, source.v_proj.weight);
            layer.out.weight = qwen_upload_tensor(index, source.o_proj.weight);
            layer.q_norm = qwen_upload_tensor(index, source.q_norm);
            layer.k_norm = qwen_upload_tensor(index, source.k_norm);
            layer.gate.weight = qwen_upload_tensor(index, source.gate_proj.weight);
            layer.up.weight = qwen_upload_tensor(index, source.up_proj.weight);
            layer.down.weight = qwen_upload_tensor(index, source.down_proj.weight);
            layer.attention_conv.base = qwen_upload_tensor(index, source.attention_conv.base_kernel);
            layer.attention_conv.projection.weight = qwen_upload_tensor(index, source.attention_conv.kernel_projection.weight);
            layer.mlp_conv.base = qwen_upload_tensor(index, source.mlp_conv.base_kernel);
            layer.mlp_conv.projection.weight = qwen_upload_tensor(index, source.mlp_conv.kernel_projection.weight);
            allocate_half(layer.context_k, context_elements, {static_cast<uint64_t>(max_context), static_cast<uint64_t>(local_kv_heads), static_cast<uint64_t>(config.head_dim)});
            allocate_half(layer.context_v, context_elements, layer.context_k.shape);
            cache_bytes += layer.context_k.nbytes + layer.context_v.nbytes;
            layers.push_back(std::move(layer));
        }
    }

    void dump(const std::string& name, const void* data,
              QwenDFlash2DebugDType dtype, size_t item_size,
              const std::vector<uint64_t>& shape) const {
        if (!debug_callback) return;
        const uint64_t elements = safe_tensor_numel(shape);
        const uint64_t bytes = elements * item_size;
        if (data == nullptr || bytes == 0) {
            throw std::runtime_error("invalid Qwen DFlash2 debug tensor: " + name);
        }
        QwenDFlash2DebugTensor tensor;
        tensor.name = name;
        tensor.dtype = dtype;
        tensor.shape = shape;
        tensor.bytes.resize(static_cast<size_t>(bytes));
        check_device(memcpy_d2h(tensor.bytes.data(), data, tensor.bytes.size()),
                     "download Qwen DFlash2 debug tensor");
        debug_callback(tensor);
    }

    void dump_sharded(const std::string& name, const void* data,
                      QwenDFlash2DebugDType dtype, size_t item_size,
                      const std::vector<uint64_t>& shape) const {
        dump(name + ".tp_rank_" + std::to_string(tp_rank), data, dtype,
             item_size, shape);
    }

    void dump_half_sharded(const std::string& name, const uint16_t* data,
                           const std::vector<uint64_t>& shape) const {
        dump_sharded(name, data, QwenDFlash2DebugDType::F16,
                     sizeof(uint16_t), shape);
    }

    void dump_float_sharded(const std::string& name, const float* data,
                            const std::vector<uint64_t>& shape) const {
        dump_sharded(name, data, QwenDFlash2DebugDType::F32, sizeof(float),
                     shape);
    }

    void dump_i32_sharded(const std::string& name, const int* data,
                          const std::vector<uint64_t>& shape) const {
        dump_sharded(name, data, QwenDFlash2DebugDType::I32, sizeof(int),
                     shape);
    }

    void dump_half(const std::string& name, const uint16_t* data,
                   const std::vector<uint64_t>& shape) const {
        dump(name, data, QwenDFlash2DebugDType::F16, sizeof(uint16_t), shape);
    }

    void dump_float(const std::string& name, const float* data,
                    const std::vector<uint64_t>& shape) const {
        dump(name, data, QwenDFlash2DebugDType::F32, sizeof(float), shape);
    }

    void dump_i32(const std::string& name, const int* data,
                  const std::vector<uint64_t>& shape) const {
        dump(name, data, QwenDFlash2DebugDType::I32, sizeof(int), shape);
    }

    void all_reduce_half(uint16_t* values, int count) {
        if (tp_world == 1) return;
#ifdef POCKET_HAVE_TP_COMM
        if (nccl_id_path.empty()) throw std::runtime_error("Qwen DFlash2 TP requires NCCL ID path");
        tp_all_reduce_sum_f16_inplace(tp_world, tp_rank, device, nccl_id_path.c_str(), values, count);
#else
        (void)values; (void)count;
        throw std::runtime_error("Qwen DFlash2 TP requires NCCL-enabled build");
#endif
    }

    void all_reduce_float(float* values, int count) {
        if (tp_world == 1) return;
#ifdef POCKET_HAVE_TP_COMM
        if (nccl_id_path.empty()) throw std::runtime_error("Qwen DFlash2 TP requires NCCL ID path");
        tp_all_reduce_sum_float_inplace(tp_world, tp_rank, device, nccl_id_path.c_str(), values, count);
#else
        (void)values; (void)count;
        throw std::runtime_error("Qwen DFlash2 TP requires NCCL-enabled build");
#endif
    }

    void projection(const DeviceLinear& linear, const uint16_t* input,
                    uint16_t* output, int rows, int output_rows, int columns) {
        if (small_batch_gemm && rows > 1 && rows <= 8) {
            const char* previous = std::getenv("QWEN_FP16_SMALL_BATCH");
            std::string saved = previous != nullptr ? previous : "";
            setenv("QWEN_FP16_SMALL_BATCH", "1", 1);
            const bool launched = qwen_fp16_matmul_rows_f16(
                input, linear.weight.f16_data(), output, rows, output_rows,
                columns, columns, output_rows, columns);
            if (previous != nullptr) {
                setenv("QWEN_FP16_SMALL_BATCH", saved.c_str(), 1);
            } else {
                unsetenv("QWEN_FP16_SMALL_BATCH");
            }
            require_launch(launched, "DFlash2 small-batch projection");
            return;
        }
        require_launch(qwen_dspark_fp16_gemm_rows_f16_cuda(
            input, linear.weight.f16_data(), output, rows, output_rows,
            columns), "DFlash2 projection");
    }

    void append_target_taps(const uint16_t* taps, int rows, int position_offset) {
        if (!taps || rows <= 0 || position_offset != committed || position_offset + rows > max_context) throw std::runtime_error("invalid Qwen DFlash2 target tap append");
        const int width = config.hidden_size * static_cast<int>(config.target_layer_ids.size());
        const std::vector<uint64_t> tap_shape = {
            static_cast<uint64_t>(rows), static_cast<uint64_t>(width)};
        profiler.begin("context_append");
        check_device(memcpy_d2d(context_taps.f16_data() + static_cast<size_t>(position_offset) * width,
                                taps, static_cast<size_t>(rows) * width * sizeof(uint16_t)), "append DFlash2 target taps");
        profiler.end();
        dump_half("target_taps", taps, tap_shape);
        committed += rows;
    }

    void prepare_context() {
        if (prepared_context == committed) return;
        if (prepared_context < 0 || prepared_context > committed) {
            throw std::runtime_error("invalid Qwen DFlash2 prepared context");
        }
        const int position_offset = prepared_context;
        const int rows = committed - prepared_context;
        const int width = config.hidden_size * static_cast<int>(config.target_layer_ids.size());
        const uint16_t* taps = context_taps.f16_data() +
            static_cast<size_t>(position_offset) * width;
        const std::vector<uint64_t> tap_shape = {
            static_cast<uint64_t>(rows), static_cast<uint64_t>(width)};
        dump_half("context.pending_target_taps", taps, tap_shape);
        const int kv_dim = local_kv_dim;
        profiler.begin("context_projection");
        QwenDeviceTensor& projected = context_projected;
        QwenDeviceTensor& normalized = context_normalized;
        QwenDeviceTensor& k = context_k;
        QwenDeviceTensor& v = context_v;
        QwenDeviceTensor& knorm = context_knorm;
        allocate_half(projected, static_cast<size_t>(rows) * config.hidden_size, {static_cast<uint64_t>(rows), static_cast<uint64_t>(config.hidden_size)});
        allocate_half(normalized, projected.nbytes / sizeof(uint16_t), projected.shape);
        require_launch(qwen_dspark_fp16_gemm_rows_f16_cuda(taps, projector.f16_data(), projected.f16_data(), rows, config.hidden_size, width), "DFlash2 context projector");
        profiler.end();
        dump_half("context.projected", projected.f16_data(), projected.shape);
        profiler.begin("context_norm");
        require_launch(qwen_dspark_rmsnorm_f16_cuda(projected.f16_data(), hidden_norm.f16_data(), normalized.f16_data(), rows, config.hidden_size, config.rms_norm_eps), "DFlash2 context norm");
        profiler.end();
        dump_half("context.normalized", normalized.f16_data(), normalized.shape);
        profiler.begin("context_kv_cache");
        allocate_half(k, static_cast<size_t>(rows) * kv_dim, {static_cast<uint64_t>(rows), static_cast<uint64_t>(kv_dim)});
        allocate_half(v, k.nbytes / sizeof(uint16_t), k.shape);
        allocate_half(knorm, k.nbytes / sizeof(uint16_t), k.shape);
        for (size_t layer_index = 0; layer_index < layers.size(); ++layer_index) {
            DeviceLayer& layer = layers[layer_index];
            const std::string prefix = "layer." + std::to_string(layer_index) + ".context.";
            projection(layer.k, normalized.f16_data(), k.f16_data(), rows, kv_dim, config.hidden_size);
            dump_half_sharded(prefix + "k_projection", k.f16_data(), k.shape);
            projection(layer.v, normalized.f16_data(), v.f16_data(), rows, kv_dim, config.hidden_size);
            dump_half_sharded(prefix + "v", v.f16_data(), v.shape);
            require_launch(qwen_dflash2_rmsnorm_heads_f16_cuda(k.f16_data(), layer.k_norm.f16_data(), knorm.f16_data(), rows, local_kv_heads, config.head_dim, config.rms_norm_eps), "DFlash2 context K norm");
            dump_half_sharded(prefix + "k_norm", knorm.f16_data(), knorm.shape);
            require_launch(qwen_dflash2_rope_k_rows_f16_cuda(knorm.f16_data(), rows, local_kv_heads, config.head_dim, position_offset, static_cast<float>(config.rope_theta)), "DFlash2 context K RoPE");
            dump_half_sharded(prefix + "k_rope", knorm.f16_data(), knorm.shape);
            check_device(memcpy_d2d(layer.context_k.f16_data() + static_cast<size_t>(position_offset) * kv_dim,
                                    knorm.f16_data(), k.nbytes), "append DFlash2 context K");
            check_device(memcpy_d2d(layer.context_v.f16_data() + static_cast<size_t>(position_offset) * kv_dim,
                                    v.f16_data(), v.nbytes), "append DFlash2 context V");
        }
        profiler.end();
        profiler.flush("prepare_context");
        prepared_context = committed;
    }

    // One denoising pass over the block. `host_tokens` supplies the row inputs:
    // row 0 is the committed anchor and the remaining rows are either mask tokens
    // (first pass) or the previous pass's drafts (refinement passes). The selected
    // path is left in `path` on the device.
    void denoise_pass(int anchor_token, const std::vector<int>& host_tokens) {
        const int rows = config.block_size;
        const int hidden = config.hidden_size;
        const int q_dim = local_q_dim;
        const int kv_dim = local_kv_dim;
        const int local_vocab = target_head.local_vocab;
        allocate_int(tokens, rows, {static_cast<uint64_t>(rows)});
        check_device(memcpy_h2d(tokens.data, host_tokens.data(), host_tokens.size() * sizeof(int)), "upload DFlash2 tokens");
        const size_t hidden_elements = static_cast<size_t>(rows) * hidden;
        allocate_half(hidden_a, hidden_elements, {static_cast<uint64_t>(rows), static_cast<uint64_t>(hidden)});
        allocate_half(hidden_b, hidden_elements, hidden_a.shape);
        allocate_float(residual_f32, hidden_elements, hidden_a.shape);
        allocate_float(branch_f32, hidden_elements, hidden_a.shape);
        allocate_float(conv_result_f32, hidden_elements, hidden_a.shape);
        const int conv_groups = hidden / config.conv_group_size;
        const int dynamic_taps = config.conv_kernel_size * conv_groups;
        const int dynamic_row_stride = 2 * dynamic_taps;
        allocate_half(dynamic, static_cast<size_t>(rows) * dynamic_row_stride,
                      {static_cast<uint64_t>(rows),
                       static_cast<uint64_t>(dynamic_row_stride)});
        profiler.begin("embedding");
        require_launch(qwen_embedding_fp16_gather_f16(target_embedding.f16_data(), static_cast<const int*>(tokens.data), hidden_a.f16_data(), rows, hidden, static_cast<int>(target_head.vocab_start), static_cast<int>(target_embedding.shape.at(0))), "DFlash2 embedding gather");
        all_reduce_half(hidden_a.f16_data(), static_cast<int>(hidden_elements));
        profiler.end();
        profiler.begin("initial_residual");
        require_launch(qwen_dflash2_f16_to_f32_cuda(
            hidden_a.f16_data(), residual_f32.f32_data(),
            static_cast<int>(hidden_elements)), "DFlash2 initial residual");
        profiler.end();

        dump_i32("noise.tokens", static_cast<const int*>(tokens.data),
                 {static_cast<uint64_t>(rows)});
        dump_half("noise.embedding", hidden_a.f16_data(), hidden_a.shape);
        dump_float("residual.initial", residual_f32.f32_data(), hidden_a.shape);
        for (size_t layer_index = 0; layer_index < layers.size(); ++layer_index) {
            DeviceLayer& layer = layers[layer_index];
            const std::string prefix = "layer." + std::to_string(layer_index) + ".";
            profiler.begin(prefix + "input_norm");
            allocate_half(normalized, hidden_elements, hidden_a.shape);
            require_launch(qwen_dflash2_rmsnorm_f32_f16_cuda(
                residual_f32.f32_data(), layer.input_norm.f16_data(),
                normalized.f16_data(), rows, hidden, config.rms_norm_eps),
                "DFlash2 input norm");
            profiler.end();
            dump_half(prefix + "input_norm", normalized.f16_data(), normalized.shape);
            profiler.begin(prefix + "attention_dynamic_prepare");

            allocate_half(conv_hidden, hidden_elements, hidden_a.shape);
            const uint16_t* attention_projection =
                layer.attention_conv.projection.weight.f16_data();
            require_launch(qwen_dspark_fp16_gemm_rows_f16_cuda(
                normalized.f16_data(), attention_projection, dynamic.f16_data(),
                rows, dynamic_row_stride, hidden),
                "DFlash2 attention dynamic projection");
            dump_half(prefix + "attention.dynamic", dynamic.f16_data(), dynamic.shape);
            require_launch(qwen_dflash2_grouped_dynamic_conv_strided_f16_cuda(
                normalized.f16_data(), dynamic.f16_data(),
                layer.attention_conv.base.f16_data(), conv_hidden.f16_data(),
                rows, hidden, conv_groups, config.conv_group_size,
                config.conv_kernel_size, dynamic_row_stride, 0),
                "DFlash2 attention convolution prepare");
            profiler.end();
            dump_half(prefix + "attention.prepare", conv_hidden.f16_data(), conv_hidden.shape);

            profiler.begin(prefix + "qkv_projection");
            allocate_half(q, static_cast<size_t>(rows) * q_dim,
                          {static_cast<uint64_t>(rows), static_cast<uint64_t>(q_dim)});
            allocate_half(k, static_cast<size_t>(rows) * kv_dim,
                          {static_cast<uint64_t>(rows), static_cast<uint64_t>(kv_dim)});
            allocate_half(v, k.nbytes / sizeof(uint16_t), k.shape);
            projection(layer.q, conv_hidden.f16_data(), q.f16_data(), rows, q_dim, hidden);
            dump_half_sharded(prefix + "attention.q_projection", q.f16_data(), q.shape);
            projection(layer.k, conv_hidden.f16_data(), k.f16_data(), rows, kv_dim, hidden);
            dump_half_sharded(prefix + "attention.k_projection", k.f16_data(), k.shape);
            projection(layer.v, conv_hidden.f16_data(), v.f16_data(), rows, kv_dim, hidden);
            profiler.end();
            dump_half_sharded(prefix + "attention.v", v.f16_data(), v.shape);
            profiler.begin(prefix + "qk_norm_rope");
            require_launch(qwen_dflash2_rmsnorm_heads_f16_cuda(
                q.f16_data(), layer.q_norm.f16_data(), q.f16_data(), rows,
                local_q_heads, config.head_dim, config.rms_norm_eps),
                "DFlash2 Q norm");
            dump_half_sharded(prefix + "attention.q_norm", q.f16_data(), q.shape);
            require_launch(qwen_dflash2_rmsnorm_heads_f16_cuda(
                k.f16_data(), layer.k_norm.f16_data(), k.f16_data(), rows,
                local_kv_heads, config.head_dim, config.rms_norm_eps),
                "DFlash2 K norm");
            dump_half_sharded(prefix + "attention.k_norm", k.f16_data(), k.shape);
            require_launch(qwen_dflash2_rope_rows_f16_cuda(
                q.f16_data(), k.f16_data(), rows, local_q_heads,
                local_kv_heads, config.head_dim, committed,
                static_cast<float>(config.rope_theta)), "DFlash2 noise RoPE");
            profiler.end();
            dump_half_sharded(prefix + "attention.q_rope", q.f16_data(), q.shape);
            dump_half_sharded(prefix + "attention.k_rope", k.f16_data(), k.shape);
            profiler.begin(prefix + "attention");
            allocate_half(attention, static_cast<size_t>(rows) * q_dim, q.shape);
            const bool grouped = use_grouped_attention();
            const bool attention_ok = grouped
                ? qwen_dflash2_attention_grouped_f16_cuda(
                    q.f16_data(), layer.context_k.f16_data(),
                    layer.context_v.f16_data(), k.f16_data(), v.f16_data(),
                    attention.f16_data(), rows, local_q_heads,
                    local_kv_heads, config.head_dim, committed,
                    max_context, config.sliding_window)
                : qwen_dflash2_attention_f16_cuda(
                    q.f16_data(), layer.context_k.f16_data(),
                    layer.context_v.f16_data(), k.f16_data(), v.f16_data(),
                    attention.f16_data(), rows, local_q_heads,
                    local_kv_heads, config.head_dim, committed,
                    max_context, config.sliding_window);
            require_launch(attention_ok, grouped
                ? "DFlash2 grouped attention" : "DFlash2 attention");
            profiler.end();
            dump_half_sharded(prefix + "attention.output", attention.f16_data(), attention.shape);
            profiler.begin(prefix + "o_projection_finish_residual");
            allocate_half(hidden_b, hidden_elements, hidden_a.shape);
            projection(layer.out, attention.f16_data(), hidden_b.f16_data(),
                       rows, hidden, q_dim);
            // Row-parallel O holds only a partial sum over the local heads. The
            // reduce must land before the finish convolution, which mixes the
            // hidden channels and would otherwise combine partial sums.
            if (sharded) {
                all_reduce_half(hidden_b.f16_data(),
                                static_cast<int>(hidden_elements));
            }
            dump_half(prefix + "attention.o_projection", hidden_b.f16_data(), hidden_b.shape);
            require_launch(qwen_dflash2_grouped_dynamic_conv_strided_f16_cuda(
                hidden_b.f16_data(), dynamic.f16_data(),
                layer.attention_conv.base.f16_data() +
                    static_cast<size_t>(config.conv_kernel_size) * hidden,
                conv_hidden.f16_data(), rows, hidden, conv_groups,
                config.conv_group_size, config.conv_kernel_size,
                dynamic_row_stride, dynamic_taps),
                "DFlash2 attention convolution finish");
            dump_half(prefix + "attention.finish", conv_hidden.f16_data(), conv_hidden.shape);
            require_launch(qwen_dflash2_f16_to_f32_cuda(
                conv_hidden.f16_data(), branch_f32.f32_data(),
                static_cast<int>(hidden_elements)), "DFlash2 attention branch");
            dump_float(prefix + "attention.branch_f32", branch_f32.f32_data(), hidden_a.shape);
            require_launch(qwen_dflash2_add_f32_cuda(
                residual_f32.f32_data(), branch_f32.f32_data(),
                static_cast<int>(hidden_elements)), "DFlash2 attention residual");
            dump_float(prefix + "attention.residual", residual_f32.f32_data(), hidden_a.shape);
            profiler.end();
            profiler.begin(prefix + "post_norm");
            require_launch(qwen_dflash2_rmsnorm_f32_f16_cuda(
                residual_f32.f32_data(), layer.post_norm.f16_data(),
                normalized.f16_data(), rows, hidden, config.rms_norm_eps),
                "DFlash2 post norm");
            profiler.end();
            dump_half(prefix + "post_norm", normalized.f16_data(), normalized.shape);
            profiler.begin(prefix + "mlp_dynamic_prepare");
            const uint16_t* mlp_projection =
                layer.mlp_conv.projection.weight.f16_data();
            require_launch(qwen_dspark_fp16_gemm_rows_f16_cuda(
                normalized.f16_data(), mlp_projection, dynamic.f16_data(),
                rows, dynamic_row_stride, hidden), "DFlash2 MLP dynamic projection");
            dump_half(prefix + "mlp.dynamic", dynamic.f16_data(), dynamic.shape);
            require_launch(qwen_dflash2_grouped_dynamic_conv_strided_f16_cuda(
                normalized.f16_data(), dynamic.f16_data(),
                layer.mlp_conv.base.f16_data(), conv_hidden.f16_data(), rows,
                hidden, conv_groups, config.conv_group_size,
                config.conv_kernel_size, dynamic_row_stride, 0),
                "DFlash2 MLP convolution prepare");
            dump_half(prefix + "mlp.prepare", conv_hidden.f16_data(), conv_hidden.shape);
            require_launch(qwen_dflash2_f16_to_f32_cuda(
                conv_hidden.f16_data(), branch_f32.f32_data(),
                static_cast<int>(hidden_elements)),
                "DFlash2 MLP convolution prepare conversion");
            profiler.end();
            profiler.begin(prefix + "mlp_gate_up_swiglu");
            const size_t intermediate_elements =
                static_cast<size_t>(rows) * local_intermediate;
            allocate_half(gate, intermediate_elements,
                          {static_cast<uint64_t>(rows),
                           static_cast<uint64_t>(local_intermediate)});
            allocate_half(up, intermediate_elements, gate.shape);
            allocate_half(intermediate, intermediate_elements, gate.shape);
            if (fused_swiglu) {
                require_launch(qwen_fp16_swiglu_matmul_rows_f16(
                    conv_hidden.f16_data(), layer.gate.weight.f16_data(),
                    layer.up.weight.f16_data(), intermediate.f16_data(), rows,
                    local_intermediate, hidden, hidden,
                    local_intermediate, hidden),
                    "DFlash2 fused SwiGLU");
            } else {
                projection(layer.gate, conv_hidden.f16_data(), gate.f16_data(),
                           rows, local_intermediate, hidden);
                dump_half_sharded(prefix + "mlp.gate", gate.f16_data(), gate.shape);
                projection(layer.up, conv_hidden.f16_data(), up.f16_data(),
                           rows, local_intermediate, hidden);
                dump_half_sharded(prefix + "mlp.up", up.f16_data(), up.shape);
                require_launch(qwen_silu_mul_rows_f16(
                    gate.f16_data(), up.f16_data(), intermediate.f16_data(), rows,
                    local_intermediate), "DFlash2 SwiGLU");
            }
            dump_half_sharded(prefix + "mlp.swiglu", intermediate.f16_data(), intermediate.shape);
            profiler.end();
            profiler.begin(prefix + "mlp_down_finish_residual");
            require_launch(cublas_fp32
                ? qwen_fp16_matmul_rows_f16_f32_cublas_cuda(
                      intermediate.f16_data(), layer.down.weight.f16_data(),
                      branch_f32.f32_data(), rows, hidden,
                      local_intermediate, local_intermediate, hidden,
                      local_intermediate)
                : qwen_fp16_matmul_rows_f16_f32(
                      intermediate.f16_data(), layer.down.weight.f16_data(),
                      branch_f32.f32_data(), rows, hidden,
                      local_intermediate, local_intermediate, hidden,
                      local_intermediate),
                "DFlash2 down projection");
            // Row-parallel down keeps its FP32 accumulator, so the partial sums
            // are reduced in FP32 rather than round-tripping through FP16.
            if (sharded) {
                all_reduce_float(branch_f32.f32_data(),
                                 static_cast<int>(hidden_elements));
            }
            dump_float(prefix + "mlp.down", branch_f32.f32_data(), hidden_a.shape);
            require_launch(qwen_dflash2_grouped_dynamic_conv_strided_f32_cuda(
                branch_f32.f32_data(), dynamic.f16_data(),
                layer.mlp_conv.base.f16_data() +
                    static_cast<size_t>(config.conv_kernel_size) * hidden,
                conv_result_f32.f32_data(), rows, hidden, conv_groups,
                config.conv_group_size, config.conv_kernel_size,
                dynamic_row_stride, dynamic_taps),
                "DFlash2 MLP convolution finish");
            dump_float(prefix + "mlp.finish", conv_result_f32.f32_data(), hidden_a.shape);
            require_launch(qwen_dflash2_add_f32_cuda(
                residual_f32.f32_data(), conv_result_f32.f32_data(),
                static_cast<int>(hidden_elements)), "DFlash2 MLP residual");
            dump_float(prefix + "residual", residual_f32.f32_data(), hidden_a.shape);
            profiler.end();
        }
        profiler.begin("final_norm");
        require_launch(qwen_dflash2_rmsnorm_f32_f16_cuda(
            residual_f32.f32_data(), final_norm.f16_data(),
            normalized.f16_data(), rows, hidden, config.rms_norm_eps),
            "DFlash2 final norm");
        profiler.end();
        dump_float("final.residual", residual_f32.f32_data(), hidden_a.shape);
        dump_half("final.norm", normalized.f16_data(), normalized.shape);
        const int draft_rows = rows - 1;
        allocate_int(local_candidates, static_cast<size_t>(draft_rows) * config.selector_top_k, {static_cast<uint64_t>(draft_rows), static_cast<uint64_t>(config.selector_top_k)});
        allocate_int(global_candidates, static_cast<size_t>(draft_rows) * config.selector_top_k, {static_cast<uint64_t>(draft_rows), static_cast<uint64_t>(config.selector_top_k)});
        allocate_float(global_unary, static_cast<size_t>(draft_rows) * config.selector_top_k, {static_cast<uint64_t>(draft_rows), static_cast<uint64_t>(config.selector_top_k)});
        allocate_float(local_unary, static_cast<size_t>(draft_rows) * config.selector_top_k, {static_cast<uint64_t>(draft_rows), static_cast<uint64_t>(config.selector_top_k)});
        allocate_float(logits, static_cast<size_t>(draft_rows) * local_vocab,
                       {static_cast<uint64_t>(draft_rows),
                        static_cast<uint64_t>(local_vocab)});
        allocate_int(path, draft_rows, {static_cast<uint64_t>(draft_rows)});
        profiler.begin("lm_head");
        require_launch(target_head.project_f16_to_f32(
            normalized.f16_data() + hidden, logits.f32_data(), draft_rows),
            "DFlash2 local logits");
        profiler.end();
        dump_float_sharded("logits.local", logits.f32_data(), logits.shape);
        profiler.begin("local_topk");
        if (split_local_topk) {
            // Measured on SM75 at the real 7x62080 shard: 8 splits is the
            // optimum (2.185 -> 1.012 ms). Cost rises again past that because
            // each block's top-k merge is serial in one thread, so more splits
            // lengthens the final merge list faster than it adds parallelism.
            constexpr int kTopKSplitVocabPerBlock = 8192;
            const int splits = std::max(1, std::min(8,
                (local_vocab + kTopKSplitVocabPerBlock - 1) /
                    kTopKSplitVocabPerBlock));
            const size_t partial_elements =
                static_cast<size_t>(draft_rows) * splits * config.selector_top_k;
            allocate_int(local_topk_partial_tokens, partial_elements,
                         {static_cast<uint64_t>(draft_rows),
                          static_cast<uint64_t>(splits),
                          static_cast<uint64_t>(config.selector_top_k)});
            allocate_float(local_topk_partial_values, partial_elements,
                           local_topk_partial_tokens.shape);
            require_launch(qwen_dflash2_local_topk_split_f32_cuda(
                logits.f32_data(),
                static_cast<int*>(local_topk_partial_tokens.data),
                local_topk_partial_values.f32_data(),
                static_cast<int*>(local_candidates.data),
                local_unary.f32_data(), draft_rows, local_vocab,
                static_cast<int>(target_head.vocab_start), config.selector_top_k,
                splits), "DFlash2 split local top-k");
        } else {
            require_launch(qwen_dflash2_local_topk_f32_cuda(
                logits.f32_data(), static_cast<int*>(local_candidates.data),
                local_unary.f32_data(), draft_rows, local_vocab,
                static_cast<int>(target_head.vocab_start), config.selector_top_k),
                "DFlash2 local top-k");
        }
        profiler.end();
        dump_i32_sharded(
            "topk.local.tokens", static_cast<const int*>(local_candidates.data),
            local_candidates.shape);
        dump_float_sharded("topk.local.logits", local_unary.f32_data(),
                           local_unary.shape);
        profiler.begin("tp_global_topk");
#ifdef POCKET_HAVE_TP_COMM
        if (tp_world > 1) {
            if (nccl_id_path.empty()) throw std::runtime_error("Qwen DFlash2 TP requires NCCL ID path");
            tp_global_topk_rows_device(
                tp_world, tp_rank, device, nccl_id_path.c_str(),
                static_cast<const int*>(local_candidates.data),
                local_unary.f32_data(), draft_rows, config.selector_top_k,
                static_cast<int*>(global_candidates.data),
                global_unary.f32_data());
        } else {
            check_device(memcpy_d2d(global_candidates.data, local_candidates.data,
                                    global_candidates.nbytes),
                         "copy DFlash2 single-rank candidates");
            check_device(memcpy_d2d(global_unary.data, local_unary.data, local_unary.nbytes),
                         "copy DFlash2 single-rank logits");
        }
#else
        if (tp_world > 1) throw std::runtime_error("Qwen DFlash2 TP requires NCCL-enabled build");
        check_device(memcpy_d2d(global_candidates.data, local_candidates.data,
                                global_candidates.nbytes),
                     "copy DFlash2 single-rank candidates");
        check_device(memcpy_d2d(global_unary.data, local_unary.data, local_unary.nbytes),
                     "copy DFlash2 single-rank logits");
#endif
        profiler.end();
        dump_i32("topk.global.tokens", static_cast<const int*>(global_candidates.data),
                 global_candidates.shape);
        dump_float("topk.global.logits", global_unary.f32_data(), global_unary.shape);
        allocate_float(selector_projected,
                       static_cast<size_t>(draft_rows) * config.selector_rank,
                       {static_cast<uint64_t>(draft_rows),
                        static_cast<uint64_t>(config.selector_rank)});
        profiler.begin("selector_projection");
        require_launch(qwen_dflash2_selector_project_f16_cuda(
            normalized.f16_data() + hidden, selector_projection.f16_data(),
            selector_projected.f32_data(), draft_rows, hidden,
            config.selector_rank), "DFlash2 selector projection");
        profiler.end();
        profiler.begin("selector");
        require_launch(viterbi_selector
            ? qwen_dflash2_selector_path_viterbi_f16_cuda(
                  selector_projected.f32_data(),
                  static_cast<const int*>(global_candidates.data),
                  global_unary.f32_data(), predecessor.f16_data(),
                  successor.f16_data(), static_cast<int*>(path.data), draft_rows,
                  config.vocab_size, config.selector_rank, config.selector_top_k,
                  anchor_token)
            : qwen_dflash2_selector_path_projected_f16_cuda(
                  selector_projected.f32_data(),
                  static_cast<const int*>(global_candidates.data),
                  global_unary.f32_data(), predecessor.f16_data(),
                  successor.f16_data(), static_cast<int*>(path.data), draft_rows,
                  config.vocab_size, config.selector_rank, config.selector_top_k,
                  anchor_token),
            "DFlash2 selector");
        profiler.end();
        profiler.flush("propose");
        dump_i32("selector.path", static_cast<const int*>(path.data), path.shape);
    }

    QwenDFlash2Proposal propose(int anchor_token) {
        if (anchor_token < 0 || anchor_token >= config.vocab_size ||
            committed >= max_context) {
            throw std::runtime_error("invalid Qwen DFlash2 proposal anchor");
        }
        prepare_context();
        const int rows = config.block_size;
        const int draft_rows = rows - 1;
        std::vector<int> host_tokens(static_cast<size_t>(rows),
                                    config.mask_token_id);
        host_tokens[0] = anchor_token;
        denoise_pass(anchor_token, host_tokens);
        // Block diffusion denoises iteratively. The first pass conditions every
        // row on mask tokens, so late rows predict from an all-mask left context
        // and are the weakest drafts; feeding the pass's own tokens back in place
        // of those masks gives each row a real predecessor. Draft content is a
        // heuristic the target re-verifies, so extra passes can only change
        // acceptance, never the emitted tokens.
        for (int pass = 1; pass < refine_passes; ++pass) {
            std::vector<int> refined(static_cast<size_t>(draft_rows));
            check_device(memcpy_d2h(refined.data(), path.data, refined.size() * sizeof(int)),
                         "download DFlash2 refinement seed");
            std::vector<int> next(static_cast<size_t>(rows));
            next[0] = anchor_token;
            for (int row = 0; row < draft_rows; ++row) {
                next[static_cast<size_t>(row) + 1] = refined[row];
            }
            if (next == host_tokens) break;
            host_tokens = next;
            denoise_pass(anchor_token, host_tokens);
        }
        QwenDFlash2Proposal proposal;
        proposal.tokens.resize(static_cast<size_t>(draft_rows));
        // The selector consumes global_candidates/global_unary entirely on the
        // device. Downloading those debug-only arrays here adds two synchronous
        // copies to every proposal while no production caller reads them. Keep
        // the legacy host view available only when explicitly requested for
        // diagnostics; the normal runtime transfers the selected path only.
        if (env_enabled("POCKETLLM_DFLASH2_DOWNLOAD_CANDIDATES")) {
            proposal.candidates.resize(
                static_cast<size_t>(draft_rows) * config.selector_top_k);
            proposal.candidate_logits.resize(proposal.candidates.size());
            check_device(memcpy_d2h(proposal.candidates.data(), global_candidates.data,
                                    proposal.candidates.size() * sizeof(int)),
                  "download DFlash2 global candidates");
            check_device(memcpy_d2h(proposal.candidate_logits.data(), global_unary.f32_data(),
                                    proposal.candidate_logits.size() * sizeof(float)),
                  "download DFlash2 global logits");
        }
        check_device(memcpy_d2h(proposal.tokens.data(), path.data,
                                proposal.tokens.size() * sizeof(int)),
                     "download DFlash2 path");
        return proposal;
    }

    uint64_t activation_workspace_bytes() const {
        uint64_t bytes = 0;
        for (const QwenDeviceTensor* tensor : {&context_taps, &tokens, &hidden_a, &hidden_b, &normalized, &q, &k, &v, &attention, &residual_f32, &branch_f32, &conv_result_f32, &conv_hidden, &gate, &up, &intermediate, &logits, &context_projected, &context_normalized, &context_k, &context_v, &context_knorm, &local_candidates, &local_topk_partial_tokens, &local_topk_partial_values, &local_unary, &global_candidates, &global_unary, &path, &selector_projected, &dynamic}) bytes += tensor->capacity;
        return bytes;
    }
};

QwenDFlash2Runtime::QwenDFlash2Runtime(
    const std::string& checkpoint_dir, const QwenDFlash2Config& config,
    const QwenDFlash2WeightMap& weights,
    const QwenDeviceTensor& target_embedding,
    QwenTargetHeadAdapter target_head, int tp_world, int tp_rank, int device,
    std::string nccl_id_path, int max_context)
    : impl_(std::make_unique<Impl>(
          checkpoint_dir, config, weights, target_embedding, target_head,
          tp_world, tp_rank, device, std::move(nccl_id_path), max_context)) {}

QwenDFlash2Runtime::~QwenDFlash2Runtime() = default;
void QwenDFlash2Runtime::reset() {
    impl_->committed = 0;
    impl_->prepared_context = 0;
}
int QwenDFlash2Runtime::committed_position() const { return impl_->committed; }
void QwenDFlash2Runtime::set_debug_callback(
    QwenDFlash2DebugCallback callback) {
    impl_->debug_callback = std::move(callback);
}
void QwenDFlash2Runtime::debug_load_target_taps(
    const std::vector<uint16_t>& taps, int rows, int position_offset) {
    const size_t width = static_cast<size_t>(impl_->config.hidden_size) *
        impl_->config.target_layer_ids.size();
    if (rows <= 0 || position_offset < 0 ||
        taps.size() != static_cast<size_t>(rows) * width) {
        throw std::runtime_error("invalid host Qwen DFlash2 target taps");
    }
    QwenDeviceTensor device_taps;
    allocate_half(device_taps, taps.size(),
                  {static_cast<uint64_t>(rows), static_cast<uint64_t>(width)});
    check_device(memcpy_h2d(device_taps.data, taps.data(), taps.size() * sizeof(uint16_t)),
                 "upload host Qwen DFlash2 target taps");
    impl_->append_target_taps(device_taps.f16_data(), rows, position_offset);
}
QwenDFlash2Proposal QwenDFlash2Runtime::debug_propose_from_host(
    const std::vector<uint16_t>& taps, int context_rows, int position_offset,
    int anchor_token) {
    if (impl_->committed != position_offset) {
        if (position_offset <= impl_->committed) {
            impl_->committed = position_offset;
            impl_->prepared_context = std::min(impl_->prepared_context, position_offset);
        } else {
            throw std::runtime_error("host Qwen DFlash2 fixture has a context gap");
        }
    }
    debug_load_target_taps(taps, context_rows, position_offset);
    return impl_->propose(anchor_token);
}
uint64_t QwenDFlash2Runtime::resident_weight_bytes() const { return impl_->weight_bytes; }
uint64_t QwenDFlash2Runtime::context_cache_bytes() const { return impl_->cache_bytes; }
uint64_t QwenDFlash2Runtime::activation_workspace_bytes() const { return impl_->activation_workspace_bytes(); }
void QwenDFlash2Runtime::append_target_taps(const uint16_t* taps, int rows, int position_offset) { impl_->append_target_taps(taps, rows, position_offset); }
void QwenDFlash2Runtime::crop_context(int position) {
    if (position < 0 || position > impl_->committed) throw std::runtime_error("invalid Qwen DFlash2 context crop");
    impl_->committed = position;
    impl_->prepared_context = std::min(impl_->prepared_context, position);
}
QwenDFlash2Proposal QwenDFlash2Runtime::propose(int anchor_token) { return impl_->propose(anchor_token); }

}  // namespace pocket
