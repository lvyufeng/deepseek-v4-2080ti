#include "qwen_dspark.hpp"

#include "cuda_ops.hpp"
#include "device_runtime.hpp"
#include "json_lite.hpp"
#include "qwen_ops.hpp"
#include "tp_comm.hpp"


#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace pocket {
namespace {

const JsonValue& required(const JsonObject& object, const std::string& key) {
    const JsonValue* value = object_get(object, key);
    if (value == nullptr) {
        throw std::runtime_error("Qwen DSpark config is missing " + key);
    }
    return *value;
}

int required_int(const JsonObject& object, const std::string& key) {
    const JsonValue& value = required(object, key);
    if (!value.is_number()) {
        throw std::runtime_error("Qwen DSpark config field is not numeric: " + key);
    }
    return static_cast<int>(value.number());
}

double required_number(const JsonObject& object, const std::string& key) {
    const JsonValue& value = required(object, key);
    if (!value.is_number()) {
        throw std::runtime_error("Qwen DSpark config field is not numeric: " + key);
    }
    return value.number();
}

bool required_bool(const JsonObject& object, const std::string& key) {
    const JsonValue& value = required(object, key);
    if (!value.is_bool()) {
        throw std::runtime_error("Qwen DSpark config field is not boolean: " + key);
    }
    return value.boolean();
}

std::string required_string(const JsonObject& object, const std::string& key) {
    return json_required_string(object, key);
}

std::vector<int> required_int_array(const JsonObject& object,
                                    const std::string& key) {
    const JsonValue& value = required(object, key);
    if (!value.is_array()) {
        throw std::runtime_error("Qwen DSpark config field is not an array: " + key);
    }
    std::vector<int> output;
    output.reserve(value.array().size());
    for (const JsonValue& item : value.array()) {
        if (!item.is_number()) {
            throw std::runtime_error("Qwen DSpark array is not numeric: " + key);
        }
        output.push_back(static_cast<int>(item.number()));
    }
    return output;
}

std::vector<std::string> required_string_array(const JsonObject& object,
                                               const std::string& key) {
    const JsonValue& value = required(object, key);
    if (!value.is_array()) {
        throw std::runtime_error("Qwen DSpark config field is not an array: " + key);
    }
    std::vector<std::string> output;
    output.reserve(value.array().size());
    for (const JsonValue& item : value.array()) {
        if (!item.is_string()) {
            throw std::runtime_error("Qwen DSpark array is not textual: " + key);
        }
        output.push_back(item.string());
    }
    return output;
}

std::string read_text(const std::string& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open Qwen DSpark config: " + path);
    std::ostringstream output;
    output << input.rdbuf();
    return output.str();
}

std::string shape_string(const std::vector<uint64_t>& shape) {
    std::ostringstream output;
    output << '[';
    for (size_t index = 0; index < shape.size(); ++index) {
        if (index != 0) output << ',';
        output << shape[index];
    }
    output << ']';
    return output.str();
}

void shard_range(uint64_t total, int world, int rank,
                 uint64_t* start, uint64_t* size) {
    if (world <= 0 || rank < 0 || rank >= world ||
        total % static_cast<uint64_t>(world) != 0) {
        throw std::runtime_error("Qwen DSpark vocab cannot be evenly TP-sharded");
    }
    *size = total / static_cast<uint64_t>(world);
    *start = *size * static_cast<uint64_t>(rank);
}

void check_device(bool ok, const std::string& what) {
    if (!ok) {
        const std::string detail = device_last_error();
        throw std::runtime_error(detail.empty() ? what : what + ": " + detail);
    }
}

void require_launch(bool ok, const std::string& what) {
    if (!ok) throw std::runtime_error("Qwen DSpark device launch failed: " + what);
}

void allocate_tensor(QwenDeviceTensor& tensor, size_t elements,
                     const std::vector<uint64_t>& shape, SafeDType dtype) {
    const uint64_t item_size = safe_dtype_size(dtype);
    if (elements == 0 || item_size == 0 ||
        elements > static_cast<size_t>(UINT64_MAX / item_size)) {
        throw std::runtime_error("invalid Qwen DSpark device tensor extent");
    }
    const uint64_t bytes = elements * item_size;
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
    check_device(tensor.data != nullptr, "allocate Qwen DSpark device tensor");
    tensor.device_dtype = dtype;
    tensor.shape = shape;
    tensor.nbytes = bytes;
    tensor.capacity = bytes;
}

void allocate_half(QwenDeviceTensor& tensor, size_t elements,
                   const std::vector<uint64_t>& shape) {
    allocate_tensor(tensor, elements, shape, SafeDType::F16);
}

void allocate_float(QwenDeviceTensor& tensor, size_t elements,
                    const std::vector<uint64_t>& shape) {
    allocate_tensor(tensor, elements, shape, SafeDType::F32);
}

struct DSparkDeviceLinear {
    QwenDeviceTensor weight;
};

struct DSparkDeviceLayer {
    QwenDeviceTensor input_norm;
    QwenDeviceTensor post_norm;
    DSparkDeviceLinear q;
    DSparkDeviceLinear k;
    DSparkDeviceLinear v;
    DSparkDeviceLinear out;
    QwenDeviceTensor q_norm;
    QwenDeviceTensor k_norm;
    DSparkDeviceLinear gate;
    DSparkDeviceLinear up;
    DSparkDeviceLinear down;
    QwenDeviceTensor context_k;
    QwenDeviceTensor context_v;
};

DSparkDeviceLinear upload_linear(const SafeTensorsIndex& index,
                                 const QwenLinearRef& ref) {
    DSparkDeviceLinear output;
    output.weight = qwen_upload_tensor(index, ref.weight);
    return output;
}

}  // namespace

QwenDSparkConfig QwenDSparkConfig::from_directory(
    const std::string& checkpoint_dir) {
    const JsonValue root = parse_json(read_text(checkpoint_dir + "/config.json"));
    if (!root.is_object()) {
        throw std::runtime_error("Qwen DSpark config root is not an object");
    }
    const JsonObject& object = root.object();
    const JsonValue& dflash_value = required(object, "dflash_config");
    const JsonValue& rope_value = required(object, "rope_parameters");
    if (!dflash_value.is_object() || !rope_value.is_object()) {
        throw std::runtime_error("Qwen DSpark nested config is malformed");
    }
    const JsonObject& dflash = dflash_value.object();
    const JsonObject& rope = rope_value.object();

    QwenDSparkConfig config;
    config.block_size = required_int(object, "block_size");
    config.hidden_size = required_int(object, "hidden_size");
    config.intermediate_size = required_int(object, "intermediate_size");
    config.num_hidden_layers = required_int(object, "num_hidden_layers");
    config.num_attention_heads = required_int(object, "num_attention_heads");
    config.num_key_value_heads = required_int(object, "num_key_value_heads");
    config.head_dim = required_int(object, "head_dim");
    config.vocab_size = required_int(object, "vocab_size");
    config.num_target_layers = required_int(object, "num_target_layers");
    config.max_position_embeddings = required_int(object, "max_position_embeddings");
    config.markov_rank = required_int(object, "markov_rank");
    config.mask_token_id = required_int(dflash, "mask_token_id");
    config.rms_norm_eps = static_cast<float>(required_number(object, "rms_norm_eps"));
    config.attention_bias = required_bool(object, "attention_bias");
    config.confidence_head_with_markov =
        required_bool(object, "confidence_head_with_markov");
    config.enable_confidence_head = required_bool(object, "enable_confidence_head");
    config.hidden_act = required_string(object, "hidden_act");
    config.markov_head_type = required_string(object, "markov_head_type");
    config.attention_mode = required_string(dflash, "attention_mode");
    config.projector_type = required_string(dflash, "projector_type");
    config.target_layer_ids = required_int_array(dflash, "target_layer_ids");
    config.layer_types = required_string_array(object, "layer_types");
    config.rope.theta = required_number(rope, "rope_theta");
    config.rope.factor = required_number(rope, "factor");
    config.rope.beta_fast = required_number(rope, "beta_fast");
    config.rope.beta_slow = required_number(rope, "beta_slow");
    config.rope.original_max_positions =
        required_int(rope, "original_max_position_embeddings");
    if (const JsonValue* value = object_get(rope, "attention_factor")) {
        if (!value->is_number()) {
            throw std::runtime_error("Qwen DSpark attention_factor is not numeric");
        }
        config.rope.attention_factor = value->number();
    } else {
        // Match Transformers' yarn_parameters default. Qwen3RotaryEmbedding
        // multiplies both cosine and sine tables by this factor.
        config.rope.attention_factor =
            0.1 * std::log(config.rope.factor) + 1.0;
    }

    if (config.block_size != 7 || config.num_hidden_layers != 5 ||
        config.hidden_size <= 0 || config.intermediate_size <= 0 ||
        config.num_attention_heads <= 0 || config.num_key_value_heads <= 0 ||
        config.num_attention_heads % config.num_key_value_heads != 0 ||
        config.head_dim <= 0 || config.head_dim % 2 != 0 ||
        config.markov_rank <= 0 || config.vocab_size <= 0 ||
        config.rope.theta <= 0.0 || config.rope.factor <= 0.0 ||
        config.rope.attention_factor <= 0.0 ||
        config.mask_token_id < 0 ||
        config.mask_token_id >= config.vocab_size ||
        config.target_layer_ids.empty() ||
        config.layer_types.size() != static_cast<size_t>(config.num_hidden_layers)) {
        throw std::runtime_error("unsupported Qwen DSpark checkpoint architecture");
    }
    if (config.hidden_act != "silu" || config.markov_head_type != "vanilla" ||
        config.attention_mode != "gqa" || config.projector_type != "dspark" ||
        config.attention_bias || !config.enable_confidence_head ||
        !config.confidence_head_with_markov) {
        throw std::runtime_error("unsupported Qwen DSpark checkpoint semantics");
    }
    if (std::any_of(config.layer_types.begin(), config.layer_types.end(),
                    [](const std::string& type) {
                        return type != "full_attention";
                    })) {
        throw std::runtime_error("Qwen DSpark requires five full-attention layers");
    }
    return config;
}

void QwenDSparkConfig::validate_for_target(
    uint64_t target_hidden_size, uint64_t target_vocab_size,
    uint64_t target_layers) const {
    if (static_cast<uint64_t>(hidden_size) != target_hidden_size ||
        static_cast<uint64_t>(vocab_size) != target_vocab_size ||
        static_cast<uint64_t>(num_target_layers) != target_layers) {
        throw std::runtime_error("Qwen DSpark checkpoint does not match target model");
    }
    if (target_layer_ids != std::vector<int>({4, 16, 28, 40, 52})) {
        throw std::runtime_error("Qwen DSpark target feature taps are unsupported");
    }
}

QwenTensorRef QwenDSparkWeightMap::require_tensor(
    const std::string& name, const std::vector<uint64_t>& shape,
    QwenShardRule rule, int shard_dim) const {
    const std::string* shard_name = index_.shard_for_tensor(name);
    if (shard_name == nullptr) {
        throw std::runtime_error("missing Qwen DSpark tensor: " + name);
    }
    SafeTensorsShard shard(index_.shard_path(*shard_name));
    const SafeTensorInfo* info = shard.find_tensor(name);
    if (info == nullptr || info->dtype != SafeDType::BF16 || info->shape != shape) {
        throw std::runtime_error(
            "invalid Qwen DSpark tensor " + name + ", expected BF16 " +
            shape_string(shape));
    }

    QwenTensorRef ref;
    ref.name = name;
    ref.shard_name = *shard_name;
    ref.dtype = SafeDType::BF16;
    ref.device_dtype = SafeDType::F16;
    ref.full_shape = shape;
    ref.local_shape = shape;
    ref.rule = rule;
    ref.shard_dim = shard_dim;
    ref.nbytes = info->nbytes;
    ref.found = true;
    if (rule == QwenShardRule::ParallelHead) {
        uint64_t start = 0;
        uint64_t size = 0;
        shard_range(shape.at(static_cast<size_t>(shard_dim)), tp_world_, tp_rank_,
                    &start, &size);
        ref.shard_start = start;
        ref.shard_size = size;
        ref.local_shape[static_cast<size_t>(shard_dim)] = size;
    }
    ref.device_nbytes = safe_tensor_numel(ref.local_shape) * sizeof(uint16_t);
    return ref;
}

QwenLinearRef QwenDSparkWeightMap::require_linear(
    const std::string& name, const std::vector<uint64_t>& shape,
    QwenShardRule rule, int shard_dim) const {
    QwenLinearRef linear;
    linear.weight = require_tensor(name, shape, rule, shard_dim);
    return linear;
}

void QwenDSparkWeightMap::record(const QwenTensorRef& ref) {
    if (!ref.found) return;
    ++tensor_count_;
    local_device_bytes_ += ref.device_nbytes;
}

QwenDSparkWeightMap::QwenDSparkWeightMap(
    const SafeTensorsIndex& index, const QwenDSparkConfig& config,
    int tp_world, int tp_rank)
    : index_(index), config_(config), tp_world_(tp_world), tp_rank_(tp_rank) {
    if (tp_world <= 0 || tp_rank < 0 || tp_rank >= tp_world) {
        throw std::runtime_error("invalid Qwen DSpark TP rank");
    }
    const uint64_t hidden = static_cast<uint64_t>(config.hidden_size);
    const uint64_t intermediate = static_cast<uint64_t>(config.intermediate_size);
    const uint64_t q_dim = static_cast<uint64_t>(config.num_attention_heads) *
                           config.head_dim;
    const uint64_t kv_dim = static_cast<uint64_t>(config.num_key_value_heads) *
                            config.head_dim;
    const uint64_t vocab = static_cast<uint64_t>(config.vocab_size);
    const uint64_t markov = static_cast<uint64_t>(config.markov_rank);

    projector_ = require_tensor(
        "fc.weight", {hidden, hidden * config.target_layer_ids.size()});
    hidden_norm_ = require_tensor("hidden_norm.weight", {hidden});
    final_norm_ = require_tensor("norm.weight", {hidden});
    markov_w1_ = require_tensor("markov_head.markov_w1.weight", {vocab, markov});
    markov_w2_ = require_tensor("markov_head.markov_w2.weight", {vocab, markov},
                                QwenShardRule::ParallelHead, 0);
    confidence_weight_ = require_tensor(
        "confidence_head.proj.weight", {1, hidden + markov});
    confidence_bias_ = require_tensor("confidence_head.proj.bias", {1});
    for (const QwenTensorRef* ref : {&projector_, &hidden_norm_, &final_norm_,
                                     &markov_w1_, &markov_w2_,
                                     &confidence_weight_, &confidence_bias_}) {
        record(*ref);
    }

    layers_.resize(static_cast<size_t>(config.num_hidden_layers));
    for (int index = 0; index < config.num_hidden_layers; ++index) {
        const std::string prefix = "layers." + std::to_string(index) + ".";
        QwenDSparkLayerWeights& layer = layers_[static_cast<size_t>(index)];
        layer.input_layernorm =
            require_tensor(prefix + "input_layernorm.weight", {hidden});
        layer.post_attention_layernorm =
            require_tensor(prefix + "post_attention_layernorm.weight", {hidden});
        layer.q_proj = require_linear(prefix + "self_attn.q_proj.weight",
                                      {q_dim, hidden});
        layer.k_proj = require_linear(prefix + "self_attn.k_proj.weight",
                                      {kv_dim, hidden});
        layer.v_proj = require_linear(prefix + "self_attn.v_proj.weight",
                                      {kv_dim, hidden});
        layer.o_proj = require_linear(prefix + "self_attn.o_proj.weight",
                                      {hidden, q_dim});
        layer.q_norm = require_tensor(prefix + "self_attn.q_norm.weight",
                                      {static_cast<uint64_t>(config.head_dim)});
        layer.k_norm = require_tensor(prefix + "self_attn.k_norm.weight",
                                      {static_cast<uint64_t>(config.head_dim)});
        layer.gate_proj = require_linear(prefix + "mlp.gate_proj.weight",
                                         {intermediate, hidden});
        layer.up_proj = require_linear(prefix + "mlp.up_proj.weight",
                                       {intermediate, hidden});
        layer.down_proj = require_linear(prefix + "mlp.down_proj.weight",
                                         {hidden, intermediate});
        for (const QwenTensorRef* ref : {
                 &layer.input_layernorm, &layer.post_attention_layernorm,
                 &layer.q_proj.weight, &layer.k_proj.weight, &layer.v_proj.weight,
                 &layer.o_proj.weight, &layer.q_norm, &layer.k_norm,
                 &layer.gate_proj.weight, &layer.up_proj.weight,
                 &layer.down_proj.weight}) {
            record(*ref);
        }
    }
    if (tensor_count_ != 62 || tensor_count_ != index_.tensor_count()) {
        throw std::runtime_error(
            "Qwen DSpark checkpoint must contain exactly the 62 expected tensors");
    }
}

struct QwenDSparkRuntime::Impl {
    std::string checkpoint_dir;
    QwenDSparkConfig config;
    const QwenDSparkWeightMap& weight_map;
    SafeTensorsIndex index;
    const QwenDeviceTensor& target_embedding;
    QwenTargetHeadAdapter target_head;
    int tp_world = 1;
    int tp_rank = 0;
    int device = 0;
    std::string nccl_id_path;
    int max_context = 0;
    int committed = 0;
    uint64_t weight_bytes = 0;
    uint64_t cache_bytes = 0;

    QwenDeviceTensor projector;
    QwenDeviceTensor hidden_norm;
    QwenDeviceTensor final_norm;
    QwenDeviceTensor markov_w1;
    QwenDeviceTensor markov_w2;
    QwenDeviceTensor confidence_weight;
    QwenDeviceTensor confidence_bias;
    std::vector<DSparkDeviceLayer> layers;
    QwenDeviceTensor inverse_frequencies;

    QwenDeviceTensor projected_context;
    QwenDeviceTensor normalized_context;
    QwenDeviceTensor tokens;
    QwenDeviceTensor hidden_a;
    QwenDeviceTensor hidden_b;
    QwenDeviceTensor normalized;
    QwenDeviceTensor q;
    QwenDeviceTensor block_k;
    QwenDeviceTensor block_v;
    QwenDeviceTensor attention;
    QwenDeviceTensor gate;
    QwenDeviceTensor up;
    QwenDeviceTensor intermediate;
    QwenDeviceTensor mlp;
    QwenDeviceTensor logits;
    QwenDeviceTensor markov_embeddings;
    QwenDeviceTensor local_argmax_tokens;
    QwenDeviceTensor local_argmax_logits;
    QwenDeviceTensor confidence;

    Impl(const std::string& checkpoint_dir_, const QwenDSparkConfig& config_,
         const QwenDSparkWeightMap& weights_,
         const QwenDeviceTensor& target_embedding_,
         QwenTargetHeadAdapter target_head_, int tp_world_, int tp_rank_,
         int device_, std::string nccl_id_path_, int max_context_)
        : checkpoint_dir(checkpoint_dir_), config(config_), weight_map(weights_),
          index(SafeTensorsIndex::from_single_file(checkpoint_dir_)),
          target_embedding(target_embedding_), target_head(target_head_),
          tp_world(tp_world_), tp_rank(tp_rank_), device(device_),
          nccl_id_path(std::move(nccl_id_path_)), max_context(max_context_) {
        if (tp_world <= 0 || tp_rank < 0 || tp_rank >= tp_world ||
            device < 0 || max_context <= 0 ||
            target_embedding.device_dtype != SafeDType::F16 ||
            target_embedding.shape.size() != 2 ||
            target_embedding.shape[1] != static_cast<uint64_t>(config.hidden_size) ||
            !target_head.valid() || target_head.hidden_size != config.hidden_size) {
            throw std::runtime_error("invalid Qwen DSpark runtime target tensors");
        }
        check_device(device_set(device), "select Qwen DSpark device");
        projector = qwen_upload_tensor(index, weight_map.projector());
        hidden_norm = qwen_upload_tensor(index, weight_map.hidden_norm());
        final_norm = qwen_upload_tensor(index, weight_map.final_norm());
        markov_w1 = qwen_upload_tensor(index, weight_map.markov_w1());
        markov_w2 = qwen_upload_tensor(index, weight_map.markov_w2());
        confidence_weight =
            qwen_upload_tensor(index, weight_map.confidence_weight());
        confidence_bias =
            qwen_upload_tensor(index, weight_map.confidence_bias());
        weight_bytes = weight_map.projector().device_nbytes +
            weight_map.hidden_norm().device_nbytes +
            weight_map.final_norm().device_nbytes +
            weight_map.markov_w1().device_nbytes +
            weight_map.markov_w2().device_nbytes +
            weight_map.confidence_weight().device_nbytes +
            weight_map.confidence_bias().device_nbytes;

        const int kv_heads = config.num_key_value_heads;
        const int head_dim = config.head_dim;
        const size_t context_elements = static_cast<size_t>(max_context) *
            kv_heads * head_dim;
        const std::vector<uint64_t> context_shape = {
            static_cast<uint64_t>(max_context), static_cast<uint64_t>(kv_heads),
            static_cast<uint64_t>(head_dim)};
        layers.reserve(weight_map.layers().size());
        for (const QwenDSparkLayerWeights& source : weight_map.layers()) {
            DSparkDeviceLayer layer;
            layer.input_norm = qwen_upload_tensor(index, source.input_layernorm);
            layer.post_norm = qwen_upload_tensor(index, source.post_attention_layernorm);
            layer.q = upload_linear(index, source.q_proj);
            layer.k = upload_linear(index, source.k_proj);
            layer.v = upload_linear(index, source.v_proj);
            layer.out = upload_linear(index, source.o_proj);
            layer.q_norm = qwen_upload_tensor(index, source.q_norm);
            layer.k_norm = qwen_upload_tensor(index, source.k_norm);
            layer.gate = upload_linear(index, source.gate_proj);
            layer.up = upload_linear(index, source.up_proj);
            layer.down = upload_linear(index, source.down_proj);
            allocate_half(layer.context_k, context_elements, context_shape);
            allocate_half(layer.context_v, context_elements, context_shape);
            cache_bytes += layer.context_k.nbytes + layer.context_v.nbytes;
            for (const QwenTensorRef* ref : {
                     &source.input_layernorm, &source.post_attention_layernorm,
                     &source.q_proj.weight, &source.k_proj.weight,
                     &source.v_proj.weight, &source.o_proj.weight,
                     &source.q_norm, &source.k_norm,
                     &source.gate_proj.weight, &source.up_proj.weight,
                     &source.down_proj.weight}) {
                weight_bytes += ref->device_nbytes;
            }
            layers.push_back(std::move(layer));
        }
        const std::vector<float> host_frequencies =
            qwen_dspark_yarn_inv_freqs(config);
        allocate_float(inverse_frequencies, host_frequencies.size(),
                       {static_cast<uint64_t>(host_frequencies.size())});
        check_device(memcpy_h2d(inverse_frequencies.data, host_frequencies.data(),
                                host_frequencies.size() * sizeof(float)),
                     "upload Qwen DSpark YaRN frequencies");
    }

    // The draft LM head is a batch=block_size by vocab by hidden GEMM. The
    // generic FP32 kernel launches a rows x batch grid, so it re-reads the
    // whole vocab-sized weight matrix once per draft row and lands about an
    // order of magnitude off the memory-bound floor. cuBLAS reuses the weights
    // across rows instead. Set POCKETLLM_DSPARK_LM_HEAD_CUBLAS=0 to fall back.
    static bool lm_head_cublas_enabled() {
        const char* value = std::getenv("POCKETLLM_DSPARK_LM_HEAD_CUBLAS");
        return value == nullptr || std::strcmp(value, "0") != 0;
    }

    void all_reduce_half(uint16_t* values, int count) {
        if (tp_world == 1) return;
#ifdef POCKET_HAVE_TP_COMM
        if (nccl_id_path.empty()) {
            throw std::runtime_error("Qwen DSpark TP requires NCCL ID path");
        }
        tp_all_reduce_sum_f16_inplace(
            tp_world, tp_rank, device, nccl_id_path.c_str(), values, count);
#else
        (void)values;
        (void)count;
        throw std::runtime_error("Qwen DSpark TP requires NCCL-enabled build");
#endif
    }

    void projection(const DSparkDeviceLinear& linear, const uint16_t* input,
                    uint16_t* output, int rows) {
        const int output_rows = static_cast<int>(linear.weight.shape.at(0));
        const int columns = static_cast<int>(linear.weight.shape.at(1));
        require_launch(qwen_dspark_fp16_gemm_rows_f16_cuda(
            input, linear.weight.f16_data(), output, rows, output_rows,
            columns), "FP16 projection");
    }

    void standard_norm(const QwenDeviceTensor& gamma, const uint16_t* input,
                       uint16_t* output, int rows, int columns) {
        require_launch(qwen_dspark_rmsnorm_f16_cuda(
            input, gamma.f16_data(), output, rows, columns,
            config.rms_norm_eps), "standard RMSNorm");
    }

    void head_norm(const QwenDeviceTensor& gamma, const uint16_t* input,
                   uint16_t* output, int rows, int heads) {
        standard_norm(gamma, input, output, rows * heads, config.head_dim);
    }

    void apply_rope(uint16_t* values, int rows, int heads,
                    int start_position) {
        require_launch(qwen_dspark_yarn_rope_f16_cuda(
            values, inverse_frequencies.f32_data(), rows, heads,
            config.head_dim, start_position,
            static_cast<float>(config.rope.attention_factor)),
            "full YaRN RoPE");
    }

    void append_target_taps(const uint16_t* target_taps, int rows,
                            int position_offset) {
        if (target_taps == nullptr || rows <= 0 || position_offset < 0 ||
            position_offset != committed || position_offset + rows > max_context) {
            throw std::runtime_error("invalid Qwen DSpark target tap append");
        }
        const int hidden = config.hidden_size;
        const int tap_columns = hidden * static_cast<int>(config.target_layer_ids.size());
        const size_t hidden_elements = static_cast<size_t>(rows) * hidden;
        allocate_half(projected_context, hidden_elements,
                      {static_cast<uint64_t>(rows), static_cast<uint64_t>(hidden)});
        allocate_half(normalized_context, hidden_elements, projected_context.shape);
        require_launch(qwen_dspark_fp16_gemm_rows_f16_cuda(
            target_taps, projector.f16_data(), projected_context.f16_data(),
            rows, hidden, tap_columns), "target feature projector");
        standard_norm(hidden_norm, projected_context.f16_data(),
                      normalized_context.f16_data(), rows, hidden);

        const int kv_heads = config.num_key_value_heads;
        const int kv_dim = kv_heads * config.head_dim;
        const size_t kv_elements = static_cast<size_t>(rows) * kv_dim;
        allocate_half(block_k, kv_elements,
                      {static_cast<uint64_t>(rows), static_cast<uint64_t>(kv_dim)});
        allocate_half(block_v, kv_elements, block_k.shape);
        allocate_half(normalized, kv_elements, block_k.shape);
        for (DSparkDeviceLayer& layer : layers) {
            projection(layer.k, normalized_context.f16_data(), block_k.f16_data(), rows);
            projection(layer.v, normalized_context.f16_data(), block_v.f16_data(), rows);
            head_norm(layer.k_norm, block_k.f16_data(), normalized.f16_data(),
                      rows, kv_heads);
            apply_rope(normalized.f16_data(), rows, kv_heads, position_offset);
            const size_t offset = static_cast<size_t>(position_offset) * kv_dim;
            check_device(memcpy_d2d(
                layer.context_k.f16_data() + offset, normalized.f16_data(),
                kv_elements * sizeof(uint16_t)),
                "append Qwen DSpark context K");
            check_device(memcpy_d2d(
                layer.context_v.f16_data() + offset, block_v.f16_data(),
                kv_elements * sizeof(uint16_t)),
                "append Qwen DSpark context V");
        }
        committed += rows;
    }

    std::pair<int, float> global_top1(float* local_logits, int local_vocab) {
        allocate_tensor(local_argmax_tokens, 1, {1}, SafeDType::I64);
        allocate_float(local_argmax_logits, 1, {1});
        require_launch(argmax_fp32_rows_cuda(
            local_logits, static_cast<int*>(local_argmax_tokens.data),
            local_argmax_logits.f32_data(), 1, local_vocab,
            static_cast<int>(target_head.vocab_start)), "local Markov argmax");
        int token = 0;
        float value = 0.0f;
#ifdef POCKET_HAVE_TP_COMM
        if (tp_world > 1) {
            if (nccl_id_path.empty()) {
                throw std::runtime_error("Qwen DSpark TP requires NCCL ID path");
            }
            tp_global_top1_rows(
                tp_world, tp_rank, device, nccl_id_path.c_str(),
                static_cast<const int*>(local_argmax_tokens.data),
                local_argmax_logits.f32_data(), 1, &token, &value);
            return {token, value};
        }
#else
        if (tp_world > 1) {
            throw std::runtime_error("Qwen DSpark TP requires NCCL-enabled build");
        }
#endif
        check_device(memcpy_d2h(&token, local_argmax_tokens.data, sizeof(int)),
                     "download Qwen DSpark token");
        check_device(memcpy_d2h(&value, local_argmax_logits.data, sizeof(float)),
                     "download Qwen DSpark logit");
        return {token, value};
    }

    uint64_t activation_workspace_bytes() const {
        return projected_context.capacity + normalized_context.capacity +
            tokens.capacity + hidden_a.capacity + hidden_b.capacity +
            normalized.capacity + q.capacity + block_k.capacity + block_v.capacity +
            attention.capacity + gate.capacity + up.capacity + intermediate.capacity +
            mlp.capacity + logits.capacity + markov_embeddings.capacity +
            local_argmax_tokens.capacity + local_argmax_logits.capacity +
            confidence.capacity;
    }

    QwenDSparkProposal propose(int anchor_token) {
        if (anchor_token < 0 || anchor_token >= config.vocab_size ||
            committed >= max_context) {
            throw std::runtime_error("invalid Qwen DSpark proposal anchor");
        }
        const int rows = config.block_size;
        const int hidden = config.hidden_size;
        const int q_heads = config.num_attention_heads;
        const int kv_heads = config.num_key_value_heads;
        const int q_dim = q_heads * config.head_dim;
        const int kv_dim = kv_heads * config.head_dim;
        const int intermediate_size = config.intermediate_size;
        const int local_vocab = target_head.local_vocab;
        std::vector<int> host_tokens(static_cast<size_t>(rows),
                                     config.mask_token_id);
        host_tokens[0] = anchor_token;
        allocate_tensor(tokens, rows, {static_cast<uint64_t>(rows)},
                        SafeDType::I64);
        check_device(memcpy_h2d(tokens.data, host_tokens.data(),
                                host_tokens.size() * sizeof(int)),
                     "upload Qwen DSpark noise tokens");
        const size_t hidden_elements = static_cast<size_t>(rows) * hidden;
        allocate_half(hidden_a, hidden_elements,
                      {static_cast<uint64_t>(rows), static_cast<uint64_t>(hidden)});
        allocate_half(hidden_b, hidden_elements, hidden_a.shape);
        require_launch(qwen_embedding_fp16_gather_f16(
            target_embedding.f16_data(), static_cast<const int*>(tokens.data),
            hidden_a.f16_data(), rows, hidden,
            static_cast<int>(target_head.vocab_start),
            static_cast<int>(target_embedding.shape.at(0))),
            "target embedding gather");
        all_reduce_half(hidden_a.f16_data(), static_cast<int>(hidden_elements));

        uint16_t* current = hidden_a.f16_data();
        uint16_t* output = hidden_b.f16_data();
        for (DSparkDeviceLayer& layer : layers) {
            allocate_half(normalized, hidden_elements, hidden_a.shape);
            standard_norm(layer.input_norm, current, normalized.f16_data(),
                          rows, hidden);
            allocate_half(q, static_cast<size_t>(rows) * q_dim,
                          {static_cast<uint64_t>(rows), static_cast<uint64_t>(q_dim)});
            allocate_half(block_k, static_cast<size_t>(rows) * kv_dim,
                          {static_cast<uint64_t>(rows), static_cast<uint64_t>(kv_dim)});
            allocate_half(block_v, static_cast<size_t>(rows) * kv_dim, block_k.shape);
            projection(layer.q, normalized.f16_data(), q.f16_data(), rows);
            projection(layer.k, normalized.f16_data(), block_k.f16_data(), rows);
            projection(layer.v, normalized.f16_data(), block_v.f16_data(), rows);
            allocate_half(attention, static_cast<size_t>(rows) * q_dim, q.shape);
            head_norm(layer.q_norm, q.f16_data(), attention.f16_data(), rows, q_heads);
            apply_rope(attention.f16_data(), rows, q_heads, committed);
            head_norm(layer.k_norm, block_k.f16_data(), normalized.f16_data(),
                      rows, kv_heads);
            apply_rope(normalized.f16_data(), rows, kv_heads, committed);
            require_launch(qwen_dspark_dual_source_gqa_f16_cuda(
                attention.f16_data(), layer.context_k.f16_data(),
                layer.context_v.f16_data(), normalized.f16_data(),
                block_v.f16_data(), q.f16_data(), rows, q_heads, kv_heads,
                config.head_dim, committed, max_context),
                "dual-source GQA");
            projection(layer.out, q.f16_data(), output, rows);
            require_launch(qwen_add_inplace_f16(
                output, current, static_cast<int>(hidden_elements)),
                "attention residual");
            standard_norm(layer.post_norm, output, normalized.f16_data(),
                          rows, hidden);
            allocate_half(gate, static_cast<size_t>(rows) * intermediate_size,
                          {static_cast<uint64_t>(rows),
                           static_cast<uint64_t>(intermediate_size)});
            allocate_half(up, static_cast<size_t>(rows) * intermediate_size,
                          gate.shape);
            allocate_half(intermediate,
                          static_cast<size_t>(rows) * intermediate_size,
                          gate.shape);
            projection(layer.gate, normalized.f16_data(), gate.f16_data(), rows);
            projection(layer.up, normalized.f16_data(), up.f16_data(), rows);
            require_launch(qwen_silu_mul_rows_f16(
                gate.f16_data(), up.f16_data(), intermediate.f16_data(), rows,
                intermediate_size), "dense SwiGLU");
            allocate_half(mlp, hidden_elements, hidden_a.shape);
            projection(layer.down, intermediate.f16_data(), mlp.f16_data(), rows);
            require_launch(qwen_add_inplace_f16(
                output, mlp.f16_data(), static_cast<int>(hidden_elements)),
                "MLP residual");
            current = output;
            output = output == hidden_a.f16_data()
                ? hidden_b.f16_data() : hidden_a.f16_data();
        }
        allocate_half(normalized, hidden_elements, hidden_a.shape);
        standard_norm(final_norm, current, normalized.f16_data(), rows, hidden);
        allocate_float(logits, static_cast<size_t>(rows) * local_vocab,
                       {static_cast<uint64_t>(rows),
                        static_cast<uint64_t>(local_vocab)});
        // The dense FP16 head reuses its weights across draft rows only through
        // cuBLAS, so keep that gate while routing through the adapter that also
        // handles an FP8 per-channel checkpoint head.
        static const bool lm_head_cublas = lm_head_cublas_enabled();
        QwenTargetHeadAdapter draft_head = target_head;
        draft_head.cublas_fp32 = lm_head_cublas && rows > 1;
        require_launch(draft_head.project_f16_to_f32(
            normalized.f16_data(), logits.f32_data(), rows),
            "draft base logits");
        allocate_half(markov_embeddings,
                      static_cast<size_t>(rows) * config.markov_rank,
                      {static_cast<uint64_t>(rows),
                       static_cast<uint64_t>(config.markov_rank)});
        allocate_float(confidence, rows, {static_cast<uint64_t>(rows)});

        QwenDSparkProposal proposal;
        proposal.tokens.reserve(static_cast<size_t>(rows));
        proposal.top_logits.reserve(static_cast<size_t>(rows));
        std::vector<int> previous_tokens(static_cast<size_t>(rows), anchor_token);
        for (int row = 0; row < rows; ++row) {
            const int previous = row == 0
                ? anchor_token : proposal.tokens.back();
            previous_tokens[static_cast<size_t>(row)] = previous;
            check_device(memcpy_h2d(tokens.data, &previous, sizeof(int)),
                         "upload Qwen DSpark Markov token");
            require_launch(qwen_dspark_embedding_gather_f16_cuda(
                markov_w1.f16_data(), static_cast<const int*>(tokens.data),
                markov_embeddings.f16_data() +
                    static_cast<size_t>(row) * config.markov_rank,
                1, config.markov_rank, config.vocab_size),
                "Markov embedding gather");
            float* row_logits = logits.f32_data() +
                static_cast<size_t>(row) * local_vocab;
            require_launch(qwen_dspark_add_markov_bias_f32_cuda(
                row_logits,
                markov_embeddings.f16_data() +
                    static_cast<size_t>(row) * config.markov_rank,
                markov_w2.f16_data(), local_vocab, config.markov_rank),
                "Markov logit correction");
            const auto [token, value] = global_top1(row_logits, local_vocab);
            proposal.tokens.push_back(token);
            proposal.top_logits.push_back(value);
        }
        require_launch(qwen_dspark_confidence_f16_cuda(
            normalized.f16_data(), markov_embeddings.f16_data(),
            confidence_weight.f16_data(), confidence_bias.f16_data(),
            confidence.f32_data(), rows, hidden, config.markov_rank),
            "confidence predictor");
        proposal.confidences.resize(static_cast<size_t>(rows));
        check_device(memcpy_d2h(proposal.confidences.data(), confidence.data,
                                proposal.confidences.size() * sizeof(float)),
                     "download Qwen DSpark confidence");
        return proposal;
    }
};

QwenDSparkRuntime::QwenDSparkRuntime(
    const std::string& checkpoint_dir, const QwenDSparkConfig& config,
    const QwenDSparkWeightMap& weights,
    const QwenDeviceTensor& target_embedding,
    QwenTargetHeadAdapter target_head, int tp_world, int tp_rank, int device,
    std::string nccl_id_path, int max_context)
    : impl_(std::make_unique<Impl>(
          checkpoint_dir, config, weights, target_embedding, target_head,
          tp_world, tp_rank, device, std::move(nccl_id_path), max_context)) {}

QwenDSparkRuntime::~QwenDSparkRuntime() = default;

void QwenDSparkRuntime::reset() { impl_->committed = 0; }

int QwenDSparkRuntime::committed_position() const { return impl_->committed; }

uint64_t QwenDSparkRuntime::resident_weight_bytes() const {
    return impl_->weight_bytes;
}

uint64_t QwenDSparkRuntime::context_cache_bytes() const {
    return impl_->cache_bytes;
}

uint64_t QwenDSparkRuntime::activation_workspace_bytes() const {
    return impl_->activation_workspace_bytes();
}

void QwenDSparkRuntime::append_target_taps(
    const uint16_t* target_taps, int rows, int position_offset) {
    impl_->append_target_taps(target_taps, rows, position_offset);
}

void QwenDSparkRuntime::crop_context(int position) {
    if (position < 0 || position > impl_->committed) {
        throw std::runtime_error("invalid Qwen DSpark context crop");
    }
    impl_->committed = position;
}

QwenDSparkProposal QwenDSparkRuntime::propose(int anchor_token) {
    return impl_->propose(anchor_token);
}

std::vector<float> qwen_dspark_yarn_inv_freqs(
    const QwenDSparkConfig& config) {
    const int dim = config.head_dim;
    if (dim <= 0 || dim % 2 != 0 || config.rope.theta <= 0.0 ||
        config.rope.factor <= 0.0 || config.rope.original_max_positions <= 0) {
        throw std::runtime_error("invalid Qwen DSpark YaRN parameters");
    }
    const auto correction_dim = [&](double rotations) {
        return dim * std::log(config.rope.original_max_positions /
                              (rotations * 2.0 * M_PI)) /
               (2.0 * std::log(config.rope.theta));
    };
    const double low = std::max(0.0, std::floor(correction_dim(config.rope.beta_fast)));
    const double high = std::min(
        static_cast<double>(dim - 1),
        std::ceil(correction_dim(config.rope.beta_slow)));
    const double denominator = low == high ? 0.001 : high - low;
    std::vector<float> output(static_cast<size_t>(dim / 2));
    for (int pair = 0; pair < dim / 2; ++pair) {
        const double base = std::pow(config.rope.theta,
                                     2.0 * pair / static_cast<double>(dim));
        const double ramp = std::clamp((pair - low) / denominator, 0.0, 1.0);
        const double extrapolation_factor = 1.0 - ramp;
        const double inverse_extrapolation = 1.0 / base;
        const double inverse_interpolation =
            inverse_extrapolation / config.rope.factor;
        output[static_cast<size_t>(pair)] = static_cast<float>(
            inverse_interpolation * (1.0 - extrapolation_factor) +
            inverse_extrapolation * extrapolation_factor);
    }
    return output;
}

}  // namespace pocket
