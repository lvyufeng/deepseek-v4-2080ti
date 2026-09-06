// Device-agnostic half of the Qwen weight loader: checkpoint mapping, TP shard
// descriptors, coverage accounting and host materialization. This translation
// unit must stay free of any vendor SDK so an official checkpoint can be
// audited on a host with no accelerator toolkit installed. Device residency and
// uploads live in engine/qwen_weights.cpp.

#include "qwen_weights.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <sstream>
#include <stdexcept>

namespace pocket {
namespace {

uint64_t ceil_div(uint64_t value, uint64_t divisor) {
    return (value + divisor - 1) / divisor;
}

std::string shape_string(const std::vector<uint64_t>& shape) {
    std::ostringstream out;
    out << '[';
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i) out << ',';
        out << shape[i];
    }
    out << ']';
    return out.str();
}

void require_tp(int world, int rank) {
    if (world <= 0) throw std::runtime_error("Qwen TP world must be positive");
    if (rank < 0 || rank >= world) throw std::runtime_error("Qwen TP rank is out of range");
}

void shard_range(uint64_t total, int world, int rank, uint64_t* start, uint64_t* size) {
    if (total == 0 || total % static_cast<uint64_t>(world) != 0) {
        throw std::runtime_error("Qwen tensor dimension is not divisible by TP world");
    }
    *size = total / static_cast<uint64_t>(world);
    *start = *size * static_cast<uint64_t>(rank);
}

void validate_model_tp(const QwenConfig& config, int world) {
    auto require_divisible = [world](uint64_t value, const char* name) {
        if (value == 0 || value % static_cast<uint64_t>(world) != 0) {
            throw std::runtime_error(
                std::string("Qwen ") + name + " is not divisible by TP world " +
                std::to_string(world));
        }
    };
    require_divisible(config.vocab_size, "vocabulary size");
    require_divisible(config.mlp.intermediate_size, "MLP intermediate size");
    require_divisible(config.full_attention.num_heads,
                      "full-attention query heads");
    require_divisible(config.full_attention.num_key_value_heads,
                      "full-attention KV heads");
    require_divisible(config.linear_attention.key_heads,
                      "linear-attention key heads");
    require_divisible(config.linear_attention.value_heads,
                      "linear-attention value heads");
}

}  // namespace

SafeDType qwen_device_dtype(SafeDType storage_dtype) {
    // Every checkpoint BF16 tensor is materialized as IEEE FP16 before upload.
    // This applies equally to the official BF16 Qwen3.8 checkpoint and to the BF16
    // scale metadata carried by FP8 checkpoints. FP8 and NVFP4 codes stay
    // compressed for online unpack.
    //
    // Both currently supported backends need this, for unrelated reasons:
    //
    //   - RTX 2080 Ti (SM75) has no native BF16 arithmetic or storage path.
    //   - First-generation Ascend 910 (Short_SoC_version=Ascend910, which includes
    //     the product named "910B" with no trailing digit) has no BF16 at all: its
    //     platform_config declares no support_bf16 and no BF16 conversion
    //     intrinsic. Second-generation 910B1-910B4 do.
    //
    // A backend with native BF16 must not reuse this without re-deriving it; see
    // CLAUDE.md for why the Ascend product name cannot be used to make that call.
    return storage_dtype == SafeDType::BF16 ? SafeDType::F16 : storage_dtype;
}

uint16_t qwen_bf16_to_fp16_bits(uint16_t bits) {
    // Widen the BF16 value to an IEEE FP32 bit pattern and use the standard
    // round-to-nearest-even FP32 -> FP16 conversion. This handles subnormals,
    // infinities, NaNs, and overflow without relying on host compiler half types.
    const uint32_t value = static_cast<uint32_t>(bits) << 16;
    const uint32_t sign = (value >> 16) & 0x8000u;
    const int exponent = static_cast<int>((value >> 23) & 0xffu) - 127 + 15;
    uint32_t mantissa = value & 0x007fffffu;
    if (exponent <= 0) {
        if (exponent < -10) return static_cast<uint16_t>(sign);
        mantissa |= 0x00800000u;
        const int shift = 14 - exponent;
        uint32_t half_mantissa = mantissa >> shift;
        const uint32_t remainder = mantissa & ((1u << shift) - 1u);
        const uint32_t halfway = 1u << (shift - 1);
        if (remainder > halfway || (remainder == halfway && (half_mantissa & 1u))) ++half_mantissa;
        return static_cast<uint16_t>(sign | half_mantissa);
    }
    if (exponent >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
    uint32_t half_mantissa = mantissa >> 13;
    const uint32_t remainder = mantissa & 0x1fffu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (half_mantissa & 1u))) {
        ++half_mantissa;
        if (half_mantissa == 0x400u) {
            half_mantissa = 0;
            if (exponent + 1 >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
            return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent + 1) << 10));
        }
    }
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) | half_mantissa);
}

void qwen_convert_bf16_to_fp16(const uint16_t* src, uint16_t* dst, size_t count) {
    if (src == nullptr || dst == nullptr) throw std::invalid_argument("null BF16/FP16 conversion buffer");
    for (size_t i = 0; i < count; ++i) dst[i] = qwen_bf16_to_fp16_bits(src[i]);
}

bool qwen_is_one_plus_norm_gamma(const std::string& name) {
    // linear_attn.norm.weight is deliberately excluded: the gated RMSNorm applies
    // it directly, so folding a +1 into it would change the result.
    if (name.find("linear_attn.norm.weight") != std::string::npos) return false;
    static const char* const suffixes[] = {
        "input_layernorm.weight",
        "post_attention_layernorm.weight",
        "self_attn.q_norm.weight",
        "self_attn.k_norm.weight",
    };
    for (const char* suffix : suffixes) {
        const std::string tail(suffix);
        if (name.size() >= tail.size() &&
            name.compare(name.size() - tail.size(), tail.size(), tail) == 0) {
            return true;
        }
    }
    // The two trunk-level norms, which have no layer prefix to key on.
    return name == "model.language_model.norm.weight" ||
           name == "mtp.norm.weight";
}

QwenHostTensor qwen_materialize_host_tensor(const SafeTensorsIndex& index,
                                            const QwenTensorRef& ref) {
    if (!ref.found) throw std::runtime_error("cannot materialize an absent Qwen tensor: " + ref.name);
    if (ref.full_shape.empty() || ref.local_shape.empty()) {
        throw std::runtime_error("cannot materialize empty Qwen tensor shape: " + ref.name);
    }
    SafeTensorsShard shard(index.shard_path(ref.shard_name));
    const SafeTensorInfo* info = shard.find_tensor(ref.name);
    if (info == nullptr) throw std::runtime_error("Qwen tensor missing while materializing: " + ref.name);
    if (info->shape != ref.full_shape || info->dtype != ref.dtype) {
        throw std::runtime_error("Qwen tensor metadata changed while materializing: " + ref.name);
    }

    QwenHostTensor out;
    out.storage_dtype = ref.dtype;
    out.device_dtype = ref.device_dtype;
    out.shape = ref.local_shape;
    const uint64_t output_numel = safe_tensor_numel(ref.local_shape);
    const uint64_t device_item_size = safe_dtype_size(ref.device_dtype);
    out.bytes.resize(static_cast<size_t>(output_numel * device_item_size));
    const uint64_t source_item_size = safe_dtype_size(ref.dtype);
    if (source_item_size == 0 || device_item_size == 0) {
        throw std::runtime_error("unsupported Qwen materialization dtype: " + ref.name);
    }

    const uint8_t* source = shard.tensor_data(*info);
    uint8_t* destination = out.bytes.data();
    const uint64_t source_row_numel = safe_tensor_numel(
        std::vector<uint64_t>(ref.full_shape.begin() + 1, ref.full_shape.end()));
    const uint64_t local_row_numel = safe_tensor_numel(
        std::vector<uint64_t>(ref.local_shape.begin() + 1, ref.local_shape.end()));
    const uint64_t source_row_bytes = source_row_numel * source_item_size;
    const uint64_t local_row_bytes = local_row_numel * device_item_size;

    auto copy_bytes = [&](const uint8_t* src, uint8_t* dst, uint64_t elements) {
        if (ref.dtype == SafeDType::BF16) {
            if (ref.device_dtype != SafeDType::F16) {
                throw std::runtime_error("Qwen BF16 tensor has non-FP16 device dtype: " + ref.name);
            }
            qwen_convert_bf16_to_fp16(reinterpret_cast<const uint16_t*>(src),
                                      reinterpret_cast<uint16_t*>(dst),
                                      static_cast<size_t>(elements));
        } else {
            std::memcpy(dst, src, static_cast<size_t>(elements * source_item_size));
        }
    };

    const bool packed = ref.rule == QwenShardRule::PackedQkvColumnParallel ||
                        ref.rule == QwenShardRule::PackedConvChannelParallel;
    if (packed) {
        if (ref.segments.empty() || ref.full_shape[0] == 0 || ref.local_shape[0] == 0) {
            throw std::runtime_error("Qwen packed tensor has no segments: " + ref.name);
        }
        uint64_t output_row = 0;
        for (const auto& segment : ref.segments) {
            const uint64_t source_row = segment.first;
            const uint64_t rows = segment.second;
            if (source_row + rows > ref.full_shape[0] || output_row + rows > ref.local_shape[0]) {
                throw std::runtime_error("Qwen packed tensor segment is out of bounds: " + ref.name);
            }
            copy_bytes(source + source_row * source_row_bytes,
                       destination + output_row * local_row_bytes,
                       rows * source_row_numel);
            output_row += rows;
        }
        if (output_row != ref.local_shape[0]) {
            throw std::runtime_error("Qwen packed tensor segments do not fill local shape: " + ref.name);
        }
        return out;
    }

    const int shard_dim = ref.shard_dim;
    if (ref.rule == QwenShardRule::Replicated || ref.shard_size == 0) {
        copy_bytes(source, destination, safe_tensor_numel(ref.full_shape));
        return out;
    }
    if (shard_dim < 0 || static_cast<size_t>(shard_dim) >= ref.full_shape.size()) {
        throw std::runtime_error("Qwen materialization has invalid shard dimension: " + ref.name);
    }
    if (shard_dim == 0) {
        const uint64_t rows = ref.shard_size;
        if (ref.shard_start + rows > ref.full_shape[0] || ref.local_shape[0] != rows) {
            throw std::runtime_error("Qwen dim-0 shard shape mismatch: " + ref.name);
        }
        copy_bytes(source + ref.shard_start * source_row_bytes, destination,
                   rows * source_row_numel);
        return out;
    }
    if (shard_dim == 1 && ref.full_shape.size() == 2) {
        const uint64_t rows = ref.full_shape[0];
        const uint64_t source_cols = ref.full_shape[1];
        const uint64_t local_cols = ref.local_shape[1];
        if (ref.shard_start + ref.shard_size > source_cols || local_cols != ref.shard_size) {
            throw std::runtime_error("Qwen dim-1 shard shape mismatch: " + ref.name);
        }
        for (uint64_t row = 0; row < rows; ++row) {
            const uint8_t* src = source + row * source_cols * source_item_size +
                                 ref.shard_start * source_item_size;
            uint8_t* dst = destination + row * local_cols * device_item_size;
            copy_bytes(src, dst, local_cols);
        }
        return out;
    }
    throw std::runtime_error("unsupported Qwen materialization rank/dimension: " + ref.name);
}

QwenNvfp4HostLinear qwen_materialize_nvfp4_host_linear(
    const SafeTensorsIndex& index, const QwenLinearRef& ref) {
    if (ref.kind != QwenLinearKind::NvFp4Group16 || !ref.has_scale ||
        !ref.has_weight_global_scale || !ref.has_input_global_scale ||
        ref.logical_local_shape.size() != 2) {
        throw std::runtime_error("invalid Qwen NVFP4 linear descriptor");
    }
    const uint64_t rows = ref.logical_local_shape[0];
    const uint64_t cols = ref.logical_local_shape[1];
    if (rows == 0 || cols == 0 || cols % 64 != 0) {
        throw std::runtime_error(
            "Qwen NVFP4 logical local shape is not block64 aligned");
    }
    const QwenHostTensor packed = qwen_materialize_host_tensor(index, ref.weight);
    const QwenHostTensor scales = qwen_materialize_host_tensor(index, ref.scale);
    const QwenHostTensor weight_global =
        qwen_materialize_host_tensor(index, ref.weight_global_scale);
    const QwenHostTensor input_global =
        qwen_materialize_host_tensor(index, ref.input_global_scale);
    const uint64_t expected_packed = rows * cols / 2;
    const uint64_t expected_scales = rows * cols / 16;
    if (packed.device_dtype != SafeDType::U8 ||
        scales.device_dtype != SafeDType::F8_E4M3 ||
        packed.bytes.size() != expected_packed ||
        scales.bytes.size() != expected_scales ||
        weight_global.device_dtype != SafeDType::F32 ||
        input_global.device_dtype != SafeDType::F32 ||
        weight_global.bytes.size() != sizeof(float) ||
        input_global.bytes.size() != sizeof(float)) {
        throw std::runtime_error("Qwen NVFP4 materialized metadata mismatch");
    }

    float checkpoint_weight_global = 0.0f;
    float checkpoint_input_global = 0.0f;
    std::memcpy(&checkpoint_weight_global, weight_global.bytes.data(),
                sizeof(float));
    std::memcpy(&checkpoint_input_global, input_global.bytes.data(),
                sizeof(float));
    if (!std::isfinite(checkpoint_weight_global) ||
        checkpoint_weight_global == 0.0f ||
        !std::isfinite(checkpoint_input_global) ||
        checkpoint_input_global == 0.0f) {
        throw std::runtime_error("Qwen NVFP4 global scale is non-finite or zero");
    }

    QwenNvfp4HostLinear output;
    output.logical_shape = ref.logical_local_shape;
    output.weight_global_factor = 1.0f / checkpoint_weight_global;
    output.input_global_scale = checkpoint_input_global;
    const uint64_t blocks_per_row = cols / 64;
    output.blocks.resize(static_cast<size_t>(rows * blocks_per_row));
    for (uint64_t row = 0; row < rows; ++row) {
        const uint8_t* packed_row = packed.bytes.data() + row * cols / 2;
        const uint8_t* scale_row = scales.bytes.data() + row * cols / 16;
        for (uint64_t block = 0; block < blocks_per_row; ++block) {
            QwenNvfp4Block64& destination =
                output.blocks[static_cast<size_t>(row * blocks_per_row + block)];
            std::memcpy(destination.d, scale_row + block * 4,
                        sizeof(destination.d));
            std::memcpy(destination.qs, packed_row + block * 32,
                        sizeof(destination.qs));
        }
    }
    return output;
}

namespace {

QwenTensorRef make_ref(const std::string& name, const std::string& shard_name,
                       const SafeTensorInfo& info, QwenShardRule rule,
                       int shard_dim, uint64_t start, uint64_t size,
                       const std::vector<uint64_t>& local_shape) {
    QwenTensorRef ref;
    ref.name = name;
    ref.shard_name = shard_name;
    ref.dtype = info.dtype;
    ref.device_dtype = qwen_device_dtype(info.dtype);
    ref.full_shape = info.shape;
    ref.local_shape = local_shape;
    ref.rule = rule;
    ref.shard_dim = shard_dim;
    ref.shard_start = start;
    ref.shard_size = size;
    ref.nbytes = safe_tensor_numel(local_shape) * safe_dtype_size(info.dtype);
    ref.device_nbytes = safe_tensor_numel(local_shape) * safe_dtype_size(ref.device_dtype);
    ref.found = true;
    return ref;
}

}  // namespace

const char* qwen_shard_rule_name(QwenShardRule rule) {
    switch (rule) {
        case QwenShardRule::Replicated: return "replicated";
        case QwenShardRule::ColumnParallel: return "column_parallel";
        case QwenShardRule::RowParallel: return "row_parallel";
        case QwenShardRule::ParallelEmbedding: return "parallel_embedding";
        case QwenShardRule::ParallelHead: return "parallel_head";
        case QwenShardRule::PackedQkvColumnParallel: return "packed_qkv_column_parallel";
        case QwenShardRule::PackedConvChannelParallel: return "packed_conv_channel_parallel";
    }
    return "unknown";
}

const char* qwen_linear_kind_name(QwenLinearKind kind) {
    switch (kind) {
        case QwenLinearKind::DenseF16: return "dense_f16";
        case QwenLinearKind::Fp8Block128: return "fp8_block128";
        case QwenLinearKind::Fp8Channel: return "fp8_channel";
        case QwenLinearKind::NvFp4Group16: return "nvfp4_group16";
    }
    return "unknown";
}

QwenWeightMap::QwenWeightMap(const SafeTensorsIndex& index, const QwenConfig& config,
                             int tp_world, int tp_rank)
    : index_(index), config_(config), tp_world_(tp_world), tp_rank_(tp_rank) {
    require_tp(tp_world_, tp_rank_);
    validate_model_tp(config_, tp_world_);
    embed_tokens_ = require_tensor(
        "model.language_model.embed_tokens.weight", SafeDType::BF16,
        {config_.vocab_size, config_.hidden_size}, QwenShardRule::ParallelEmbedding, 0);
    final_norm_ = require_tensor(
        "model.language_model.norm.weight", SafeDType::BF16,
        {config_.hidden_size}, QwenShardRule::Replicated, -1);
    lm_head_ = require_linear(
        "lm_head.weight", {config_.vocab_size, config_.hidden_size},
        QwenShardRule::ParallelHead, 0);

    layers_.reserve(config_.num_hidden_layers);
    for (uint64_t layer_id = 0; layer_id < config_.num_hidden_layers; ++layer_id) {
        const std::string prefix = "model.language_model.layers." + std::to_string(layer_id) + ".";
        QwenLayerWeights layer;
        layer.input_layernorm = require_tensor(prefix + "input_layernorm.weight", SafeDType::BF16,
                                               {config_.hidden_size});
        layer.post_attention_layernorm = require_tensor(prefix + "post_attention_layernorm.weight", SafeDType::BF16,
                                                        {config_.hidden_size});
        if (config_.layer_types[layer_id] == QwenLayerType::LinearAttention) {
            const auto& linear = config_.linear_attention;
            const uint64_t key_dim = linear.key_heads * linear.key_head_dim;
            const uint64_t value_dim = linear.value_heads * linear.value_head_dim;
            layer.linear_attention.in_proj_qkv = require_linear(
                prefix + "linear_attn.in_proj_qkv.weight",
                {2 * key_dim + value_dim, config_.hidden_size},
                QwenShardRule::PackedQkvColumnParallel, 0);
            layer.linear_attention.in_proj_z = require_linear(
                prefix + "linear_attn.in_proj_z.weight", {value_dim, config_.hidden_size},
                QwenShardRule::ColumnParallel, 0);
            layer.linear_attention.out_proj = require_linear(
                prefix + "linear_attn.out_proj.weight", {config_.hidden_size, value_dim},
                QwenShardRule::RowParallel, 1);
            layer.linear_attention.in_proj_a = require_linear(
                prefix + "linear_attn.in_proj_a.weight", {linear.value_heads, config_.hidden_size},
                QwenShardRule::ColumnParallel, 0);
            layer.linear_attention.in_proj_b = require_linear(
                prefix + "linear_attn.in_proj_b.weight", {linear.value_heads, config_.hidden_size},
                QwenShardRule::ColumnParallel, 0);
            layer.linear_attention.conv1d = require_tensor(
                prefix + "linear_attn.conv1d.weight", SafeDType::BF16,
                {2 * key_dim + value_dim, 1, linear.conv_kernel_dim},
                QwenShardRule::PackedConvChannelParallel, 0);
            layer.linear_attention.a_log = require_tensor(
                prefix + "linear_attn.A_log", SafeDType::BF16, {linear.value_heads},
                QwenShardRule::ColumnParallel, 0);
            layer.linear_attention.dt_bias = require_tensor(
                prefix + "linear_attn.dt_bias", SafeDType::BF16, {linear.value_heads},
                QwenShardRule::ColumnParallel, 0);
            layer.linear_attention.norm = require_tensor(
                prefix + "linear_attn.norm.weight", SafeDType::BF16, {linear.value_head_dim});
        } else {
            const auto& full = config_.full_attention;
            const uint64_t attention_dim = full.num_heads * full.head_dim;
            const uint64_t q_dim = attention_dim * (full.output_gate ? 2 : 1);
            const uint64_t kv_dim = full.num_key_value_heads * full.head_dim;
            layer.full_attention.q_proj = require_linear(
                prefix + "self_attn.q_proj.weight", {q_dim, config_.hidden_size},
                QwenShardRule::ColumnParallel, 0);
            layer.full_attention.k_proj = require_linear(
                prefix + "self_attn.k_proj.weight", {kv_dim, config_.hidden_size},
                QwenShardRule::ColumnParallel, 0);
            layer.full_attention.v_proj = require_linear(
                prefix + "self_attn.v_proj.weight", {kv_dim, config_.hidden_size},
                QwenShardRule::ColumnParallel, 0);
            layer.full_attention.o_proj = require_linear(
                prefix + "self_attn.o_proj.weight", {config_.hidden_size, attention_dim},
                QwenShardRule::RowParallel, 1);
            layer.full_attention.q_norm = require_tensor(
                prefix + "self_attn.q_norm.weight", SafeDType::BF16, {full.head_dim});
            layer.full_attention.k_norm = require_tensor(
                prefix + "self_attn.k_norm.weight", SafeDType::BF16, {full.head_dim});
        }

        const std::string mlp_prefix = prefix + "mlp.";
        layer.mlp.gate_proj = require_linear(
            mlp_prefix + "gate_proj.weight", {config_.mlp.intermediate_size, config_.hidden_size},
            QwenShardRule::ColumnParallel, 0);
        layer.mlp.up_proj = require_linear(
            mlp_prefix + "up_proj.weight", {config_.mlp.intermediate_size, config_.hidden_size},
            QwenShardRule::ColumnParallel, 0);
        layer.mlp.down_proj = require_linear(
            mlp_prefix + "down_proj.weight", {config_.hidden_size, config_.mlp.intermediate_size},
            QwenShardRule::RowParallel, 1);
        layers_.push_back(std::move(layer));
    }

    if (config_.mtp_num_hidden_layers != 0) {
        if (config_.mtp_num_hidden_layers != 1) {
            throw std::runtime_error(
                "Qwen native MTP currently supports exactly one MTP layer");
        }
        if (config_.mtp_use_dedicated_embeddings) {
            throw std::runtime_error(
                "Qwen native MTP dedicated embeddings are not supported");
        }
        mtp_.found = true;
        mtp_.pre_fc_norm_embedding = require_tensor(
            "mtp.pre_fc_norm_embedding.weight", SafeDType::BF16,
            {config_.hidden_size});
        mtp_.pre_fc_norm_hidden = require_tensor(
            "mtp.pre_fc_norm_hidden.weight", SafeDType::BF16,
            {config_.hidden_size});
        mtp_.norm = require_tensor(
            "mtp.norm.weight", SafeDType::BF16, {config_.hidden_size});
        mtp_.fc = require_linear(
            "mtp.fc.weight", {config_.hidden_size, 2 * config_.hidden_size},
            QwenShardRule::Replicated, -1);

        const std::string prefix = "mtp.layers.0.";
        mtp_.layer.input_layernorm = require_tensor(
            prefix + "input_layernorm.weight", SafeDType::BF16,
            {config_.hidden_size});
        mtp_.layer.post_attention_layernorm = require_tensor(
            prefix + "post_attention_layernorm.weight", SafeDType::BF16,
            {config_.hidden_size});
        const auto& full = config_.full_attention;
        const uint64_t attention_dim = full.num_heads * full.head_dim;
        const uint64_t q_dim = attention_dim * 2;
        const uint64_t kv_dim = full.num_key_value_heads * full.head_dim;
        mtp_.layer.full_attention.q_proj = require_linear(
            prefix + "self_attn.q_proj.weight", {q_dim, config_.hidden_size},
            QwenShardRule::ColumnParallel, 0);
        mtp_.layer.full_attention.k_proj = require_linear(
            prefix + "self_attn.k_proj.weight", {kv_dim, config_.hidden_size},
            QwenShardRule::ColumnParallel, 0);
        mtp_.layer.full_attention.v_proj = require_linear(
            prefix + "self_attn.v_proj.weight", {kv_dim, config_.hidden_size},
            QwenShardRule::ColumnParallel, 0);
        mtp_.layer.full_attention.o_proj = require_linear(
            prefix + "self_attn.o_proj.weight", {config_.hidden_size, attention_dim},
            QwenShardRule::RowParallel, 1);
        mtp_.layer.full_attention.q_norm = require_tensor(
            prefix + "self_attn.q_norm.weight", SafeDType::BF16,
            {full.head_dim});
        mtp_.layer.full_attention.k_norm = require_tensor(
            prefix + "self_attn.k_norm.weight", SafeDType::BF16,
            {full.head_dim});
        const std::string mlp_prefix = prefix + "mlp.";
        mtp_.layer.mlp.gate_proj = require_linear(
            mlp_prefix + "gate_proj.weight",
            {config_.mlp.intermediate_size, config_.hidden_size},
            QwenShardRule::ColumnParallel, 0);
        mtp_.layer.mlp.up_proj = require_linear(
            mlp_prefix + "up_proj.weight",
            {config_.mlp.intermediate_size, config_.hidden_size},
            QwenShardRule::ColumnParallel, 0);
        mtp_.layer.mlp.down_proj = require_linear(
            mlp_prefix + "down_proj.weight",
            {config_.hidden_size, config_.mlp.intermediate_size},
            QwenShardRule::RowParallel, 1);
    }

    auto count_ref = [this](const QwenTensorRef& ref, bool scale) {
        record(ref, scale);
    };
    const auto count_linear = [this](const QwenLinearRef& linear) {
        record_linear(linear);
    };
    count_ref(embed_tokens_, false);
    count_ref(final_norm_, false);
    count_linear(lm_head_);
    for (const auto& layer : layers_) {
        count_ref(layer.input_layernorm, false);
        count_ref(layer.post_attention_layernorm, false);
        if (layer.linear_attention.in_proj_qkv.weight.found) {
            count_linear(layer.linear_attention.in_proj_qkv);
            count_linear(layer.linear_attention.in_proj_z);
            count_linear(layer.linear_attention.out_proj);
            count_linear(layer.linear_attention.in_proj_a);
            count_linear(layer.linear_attention.in_proj_b);
            count_ref(layer.linear_attention.conv1d, false);
            count_ref(layer.linear_attention.a_log, false);
            count_ref(layer.linear_attention.dt_bias, false);
            count_ref(layer.linear_attention.norm, false);
        } else {
            count_linear(layer.full_attention.q_proj);
            count_linear(layer.full_attention.k_proj);
            count_linear(layer.full_attention.v_proj);
            count_linear(layer.full_attention.o_proj);
            count_ref(layer.full_attention.q_norm, false);
            count_ref(layer.full_attention.k_norm, false);
        }
        count_linear(layer.mlp.gate_proj);
        count_linear(layer.mlp.up_proj);
        count_linear(layer.mlp.down_proj);
    }
    if (mtp_.found) {
        count_ref(mtp_.pre_fc_norm_embedding, false);
        count_ref(mtp_.pre_fc_norm_hidden, false);
        count_linear(mtp_.fc);
        count_ref(mtp_.norm, false);
        const QwenLayerWeights& layer = mtp_.layer;
        count_ref(layer.input_layernorm, false);
        count_ref(layer.post_attention_layernorm, false);
        count_linear(layer.full_attention.q_proj);
        count_linear(layer.full_attention.k_proj);
        count_linear(layer.full_attention.v_proj);
        count_linear(layer.full_attention.o_proj);
        count_ref(layer.full_attention.q_norm, false);
        count_ref(layer.full_attention.k_norm, false);
        count_linear(layer.mlp.gate_proj);
        count_linear(layer.mlp.up_proj);
        count_linear(layer.mlp.down_proj);
    }
}
QwenTensorRef QwenWeightMap::require_tensor(const std::string& name, SafeDType dtype,
                                            const std::vector<uint64_t>& shape,
                                            QwenShardRule rule, int shard_dim) const {
    const std::string* shard_name = index_.shard_for_tensor(name);
    if (shard_name == nullptr) throw std::runtime_error("Qwen tensor not in checkpoint index: " + name);
    SafeTensorsShard shard(index_.shard_path(*shard_name));
    const SafeTensorInfo* info = shard.find_tensor(name);
    if (info == nullptr) throw std::runtime_error("Qwen tensor missing from shard header: " + name);
    if (info->dtype != dtype) {
        throw std::runtime_error("unexpected dtype for " + name + " expected=" + safe_dtype_name(dtype) +
                                 " actual=" + safe_dtype_name(info->dtype));
    }
    if (info->shape != shape) {
        throw std::runtime_error("unexpected shape for " + name + " expected=" + shape_string(shape) +
                                 " actual=" + shape_string(info->shape));
    }

    std::vector<uint64_t> local_shape = shape;
    uint64_t start = 0;
    uint64_t size = 0;
    if (rule == QwenShardRule::Replicated) {
        size = shard_dim < 0 ? 0 : shape[static_cast<size_t>(shard_dim)];
    } else if (rule == QwenShardRule::PackedQkvColumnParallel ||
               rule == QwenShardRule::PackedConvChannelParallel) {
        const uint64_t key_dim = config_.linear_attention.key_heads * config_.linear_attention.key_head_dim;
        const uint64_t value_dim = config_.linear_attention.value_heads * config_.linear_attention.value_head_dim;
        const uint64_t segments[] = {key_dim, key_dim, value_dim};
        uint64_t source_offset = 0;
        uint64_t local_total = 0;
        QwenTensorRef ref;
        for (uint64_t segment : segments) {
            uint64_t segment_start = 0;
            uint64_t segment_size = 0;
            shard_range(segment, tp_world_, tp_rank_, &segment_start, &segment_size);
            ref.segments.emplace_back(source_offset + segment_start, segment_size);
            source_offset += segment;
            local_total += segment_size;
        }
        local_shape[0] = local_total;
        size = local_total;
        ref = make_ref(name, *shard_name, *info, rule, shard_dim, 0, size, local_shape);
        ref.segments.clear();
        source_offset = 0;
        for (uint64_t segment : segments) {
            uint64_t segment_start = 0;
            uint64_t segment_size = 0;
            shard_range(segment, tp_world_, tp_rank_, &segment_start, &segment_size);
            ref.segments.emplace_back(source_offset + segment_start, segment_size);
            source_offset += segment;
        }
        return ref;
    } else if (shard_dim >= 0) {
        if (static_cast<size_t>(shard_dim) >= shape.size()) throw std::runtime_error("invalid Qwen shard dimension");
        shard_range(shape[static_cast<size_t>(shard_dim)], tp_world_, tp_rank_, &start, &size);
        local_shape[static_cast<size_t>(shard_dim)] = size;
    } else {
        throw std::runtime_error("Qwen non-replicated tensor has no shard dimension: " + name);
    }
    return make_ref(name, *shard_name, *info, rule, shard_dim, start, size, local_shape);
}

QwenLinearRef QwenWeightMap::require_linear(const std::string& name,
                                            const std::vector<uint64_t>& shape,
                                            QwenShardRule rule,
                                            int shard_dim) const {
    const std::string weight_suffix = ".weight";
    const bool conventional_name =
        name.size() >= weight_suffix.size() &&
        name.compare(name.size() - weight_suffix.size(), weight_suffix.size(),
                     weight_suffix) == 0;
    const std::string base = conventional_name
        ? name.substr(0, name.size() - weight_suffix.size()) : name;
    const std::string packed_name = base + ".weight_packed";
    const std::string block_scale_name = base + ".weight_scale_inv";
    const std::string channel_scale_name = base + ".weight_scale";
    const std::string weight_global_name = base + ".weight_global_scale";
    const std::string input_global_name = base + ".input_global_scale";

    QwenLinearRef result;
    result.logical_full_shape = shape;
    result.rule = rule;
    result.shard_dim = shard_dim;
    result.logical_local_shape = shape;
    if (rule != QwenShardRule::Replicated) {
        if (rule == QwenShardRule::PackedQkvColumnParallel) {
            const uint64_t key_dim = config_.linear_attention.key_heads *
                                     config_.linear_attention.key_head_dim;
            const uint64_t value_dim = config_.linear_attention.value_heads *
                                       config_.linear_attention.value_head_dim;
            const uint64_t segments[] = {key_dim, key_dim, value_dim};
            uint64_t local_rows = 0;
            for (uint64_t segment : segments) {
                uint64_t start = 0;
                uint64_t size = 0;
                shard_range(segment, tp_world_, tp_rank_, &start, &size);
                local_rows += size;
            }
            result.logical_local_shape[0] = local_rows;
        } else {
            uint64_t start = 0;
            uint64_t size = 0;
            shard_range(shape.at(static_cast<size_t>(shard_dim)), tp_world_,
                        tp_rank_, &start, &size);
            result.logical_local_shape[static_cast<size_t>(shard_dim)] = size;
        }
    }
    const uint64_t local_n = result.logical_local_shape.at(0);
    const uint64_t local_k = result.logical_local_shape.at(1);
    if ((rule == QwenShardRule::ColumnParallel ||
         rule == QwenShardRule::ParallelHead) &&
        result.logical_local_shape[1] != shape[1]) {
        throw std::runtime_error(
            "Qwen column-parallel linear changed logical K: " + base);
    }
    if (rule == QwenShardRule::RowParallel &&
        result.logical_local_shape[0] != shape[0]) {
        throw std::runtime_error(
            "Qwen row-parallel linear changed logical N: " + base);
    }
    (void)local_n;
    (void)local_k;

    const std::string* packed_shard = index_.shard_for_tensor(packed_name);
    const std::string* weight_shard = index_.shard_for_tensor(name);
    if (packed_shard != nullptr) {
        if (weight_shard != nullptr) {
            throw std::runtime_error(
                "ambiguous Qwen linear has both packed and dense weights: " + base);
        }
        if (shape.size() != 2 || shape[1] % 64 != 0) {
            throw std::runtime_error(
                "Qwen NVFP4 logical K must be divisible by 64: " + base);
        }
        result.kind = QwenLinearKind::NvFp4Group16;
        const std::vector<uint64_t> packed_shape = {shape[0], shape[1] / 2};
        result.weight = require_tensor(packed_name, SafeDType::U8, packed_shape,
                                       rule, shard_dim);
        if (result.weight.local_shape[1] % 32 != 0) {
            throw std::runtime_error(
                "Qwen NVFP4 local packed K is not block64 aligned: " + base);
        }

        const std::string* scale_shard = index_.shard_for_tensor(channel_scale_name);
        if (scale_shard == nullptr) {
            throw std::runtime_error(
                "Qwen NVFP4 local scale not in checkpoint index: " +
                channel_scale_name);
        }
        SafeTensorsShard scale_file(index_.shard_path(*scale_shard));
        const SafeTensorInfo* scale_info =
            scale_file.find_tensor(channel_scale_name);
        const std::vector<uint64_t> scale_shape = {shape[0], shape[1] / 16};
        if (scale_info == nullptr || scale_info->dtype != SafeDType::F8_E4M3 ||
            scale_info->shape != scale_shape) {
            throw std::runtime_error(
                "unexpected Qwen NVFP4 scale metadata for " +
                channel_scale_name);
        }
        if (rule == QwenShardRule::PackedQkvColumnParallel) {
            throw std::runtime_error(
                "Qwen NVFP4 packed QKV projections are not supported: " + base);
        }
        if (rule == QwenShardRule::RowParallel) {
            uint64_t logical_start = 0;
            uint64_t logical_size = 0;
            shard_range(shape[1], tp_world_, tp_rank_, &logical_start,
                        &logical_size);
            if (logical_start % 64 != 0 || logical_size % 64 != 0) {
                throw std::runtime_error(
                    "Qwen row-parallel NVFP4 shard is not block64 aligned: " +
                    base);
            }
            result.weight.shard_start = logical_start / 2;
            result.weight.shard_size = logical_size / 2;
            result.weight.local_shape[1] = logical_size / 2;
            result.weight.nbytes = safe_tensor_numel(result.weight.local_shape);
            result.weight.device_nbytes = result.weight.nbytes;
            result.scale = make_ref(
                channel_scale_name, *scale_shard, *scale_info, rule, 1,
                logical_start / 16, logical_size / 16,
                {shape[0], logical_size / 16});
        } else if (rule == QwenShardRule::Replicated) {
            result.scale = make_ref(channel_scale_name, *scale_shard, *scale_info,
                                    rule, -1, 0, 0, scale_shape);
        } else {
            result.scale = require_tensor(
                channel_scale_name, SafeDType::F8_E4M3, scale_shape, rule, 0);
        }
        result.has_scale = true;
        result.weight_global_scale = require_tensor(
            weight_global_name, SafeDType::F32, {1},
            QwenShardRule::Replicated, -1);
        result.input_global_scale = require_tensor(
            input_global_name, SafeDType::F32, {1},
            QwenShardRule::Replicated, -1);
        result.has_weight_global_scale = true;
        result.has_input_global_scale = true;
        return result;
    }

    if (weight_shard == nullptr) {
        throw std::runtime_error("Qwen linear not in checkpoint index: " + name);
    }
    SafeTensorsShard weight_file(index_.shard_path(*weight_shard));
    const SafeTensorInfo* weight_info = weight_file.find_tensor(name);
    if (weight_info == nullptr) {
        throw std::runtime_error("Qwen linear missing from shard header: " + name);
    }
    if (weight_info->dtype == SafeDType::BF16 ||
        weight_info->dtype == SafeDType::F16) {
        result.kind = QwenLinearKind::DenseF16;
        result.weight = require_tensor(name, weight_info->dtype, shape, rule,
                                       shard_dim);
        if (index_.shard_for_tensor(block_scale_name) != nullptr ||
            index_.shard_for_tensor(channel_scale_name) != nullptr) {
            throw std::runtime_error(
                "dense Qwen linear unexpectedly has quantization scale: " + name);
        }
        return result;
    }
    if (weight_info->dtype != SafeDType::F8_E4M3) {
        throw std::runtime_error("unsupported Qwen linear dtype for " + name +
                                 ": " + safe_dtype_name(weight_info->dtype));
    }
    result.weight = require_tensor(name, SafeDType::F8_E4M3, shape, rule,
                                   shard_dim);

    const std::string* block_shard = index_.shard_for_tensor(block_scale_name);
    const std::string* channel_shard = index_.shard_for_tensor(channel_scale_name);
    if ((block_shard != nullptr) == (channel_shard != nullptr)) {
        throw std::runtime_error(
            "Qwen FP8 linear must have exactly one recognized scale: " + name);
    }

    if (channel_shard != nullptr) {
        result.kind = QwenLinearKind::Fp8Channel;
        SafeTensorsShard scale_file(index_.shard_path(*channel_shard));
        const SafeTensorInfo* scale_info = scale_file.find_tensor(channel_scale_name);
        const std::vector<uint64_t> full_scale_shape = {shape[0], 1};
        if (scale_info == nullptr || scale_info->dtype != SafeDType::BF16 ||
            scale_info->shape != full_scale_shape) {
            throw std::runtime_error(
                "unexpected Qwen FP8 channel scale metadata for " +
                channel_scale_name);
        }
        if (rule == QwenShardRule::PackedQkvColumnParallel) {
            const uint64_t key_dim = config_.linear_attention.key_heads *
                                     config_.linear_attention.key_head_dim;
            const uint64_t value_dim = config_.linear_attention.value_heads *
                                       config_.linear_attention.value_head_dim;
            const uint64_t segments[] = {key_dim, key_dim, value_dim};
            uint64_t source_offset = 0;
            uint64_t local_rows = 0;
            std::vector<std::pair<uint64_t, uint64_t>> segments_out;
            for (uint64_t segment : segments) {
                uint64_t start = 0;
                uint64_t size = 0;
                shard_range(segment, tp_world_, tp_rank_, &start, &size);
                segments_out.emplace_back(source_offset + start, size);
                source_offset += segment;
                local_rows += size;
            }
            result.scale = make_ref(
                channel_scale_name, *channel_shard, *scale_info, rule, 0,
                0, local_rows, {local_rows, 1});
            result.scale.segments = std::move(segments_out);
        } else if (rule == QwenShardRule::RowParallel ||
                   rule == QwenShardRule::Replicated) {
            result.scale = make_ref(channel_scale_name, *channel_shard,
                                    *scale_info, QwenShardRule::Replicated, -1,
                                    0, 0, full_scale_shape);
        } else {
            uint64_t start = 0;
            uint64_t size = 0;
            shard_range(shape[0], tp_world_, tp_rank_, &start, &size);
            result.scale = make_ref(channel_scale_name, *channel_shard,
                                    *scale_info, rule, 0, start, size,
                                    {size, 1});
        }
        result.has_scale = true;
        return result;
    }

    result.kind = QwenLinearKind::Fp8Block128;
    SafeTensorsShard scale_file(index_.shard_path(*block_shard));
    const SafeTensorInfo* scale_info = scale_file.find_tensor(block_scale_name);
    const std::vector<uint64_t> full_scale_shape = {
        ceil_div(shape[0], config_.fp8_block_size),
        ceil_div(shape[1], config_.fp8_block_size)};
    if (scale_info == nullptr || scale_info->dtype != SafeDType::BF16 ||
        scale_info->shape != full_scale_shape) {
        throw std::runtime_error(
            "unexpected Qwen FP8 block scale metadata for " + block_scale_name);
    }

    std::vector<uint64_t> local_scale_shape = full_scale_shape;
    uint64_t scale_start = 0;
    uint64_t scale_size = 0;
    if (rule == QwenShardRule::Replicated) {
        result.scale = make_ref(block_scale_name, *block_shard, *scale_info,
                                rule, shard_dim, 0, 0, local_scale_shape);
    } else if (rule == QwenShardRule::RowParallel) {
        shard_range(shape[1], tp_world_, tp_rank_, &scale_start, &scale_size);
        if (scale_start % config_.fp8_block_size != 0 ||
            scale_size % config_.fp8_block_size != 0) {
            throw std::runtime_error(
                "Qwen row-parallel FP8 shard is not scale-block aligned: " + name);
        }
        local_scale_shape[1] = scale_size / config_.fp8_block_size;
        result.scale = make_ref(
            block_scale_name, *block_shard, *scale_info, rule, shard_dim,
            scale_start / config_.fp8_block_size,
            scale_size / config_.fp8_block_size, local_scale_shape);
    } else if (rule == QwenShardRule::PackedQkvColumnParallel) {
        const uint64_t key_dim = config_.linear_attention.key_heads *
                                 config_.linear_attention.key_head_dim;
        const uint64_t value_dim = config_.linear_attention.value_heads *
                                   config_.linear_attention.value_head_dim;
        const uint64_t segments[] = {key_dim, key_dim, value_dim};
        uint64_t source_offset = 0;
        uint64_t local_rows = 0;
        std::vector<std::pair<uint64_t, uint64_t>> scale_segments;
        for (uint64_t segment : segments) {
            uint64_t segment_start = 0;
            uint64_t segment_size = 0;
            shard_range(segment, tp_world_, tp_rank_, &segment_start,
                        &segment_size);
            if (segment_start % config_.fp8_block_size != 0 ||
                segment_size % config_.fp8_block_size != 0) {
                throw std::runtime_error(
                    "Qwen packed QKV shard is not scale-block aligned: " + name);
            }
            scale_segments.emplace_back(
                source_offset / config_.fp8_block_size +
                    segment_start / config_.fp8_block_size,
                segment_size / config_.fp8_block_size);
            source_offset += segment;
            local_rows += segment_size / config_.fp8_block_size;
        }
        local_scale_shape[0] = local_rows;
        result.scale = make_ref(block_scale_name, *block_shard, *scale_info,
                                rule, shard_dim, 0, local_rows,
                                local_scale_shape);
        result.scale.segments = std::move(scale_segments);
    } else {
        shard_range(shape[0], tp_world_, tp_rank_, &scale_start, &scale_size);
        if (scale_start % config_.fp8_block_size != 0 ||
            scale_size % config_.fp8_block_size != 0) {
            throw std::runtime_error(
                "Qwen column-parallel FP8 shard is not scale-block aligned: " +
                name);
        }
        local_scale_shape[0] = scale_size / config_.fp8_block_size;
        result.scale = make_ref(
            block_scale_name, *block_shard, *scale_info, rule, shard_dim,
            scale_start / config_.fp8_block_size,
            scale_size / config_.fp8_block_size, local_scale_shape);
    }
    result.has_scale = true;
    return result;
}

void QwenWeightMap::record(const QwenTensorRef& ref, bool scale) {
    ++tensor_count_;
    if (scale) local_scale_bytes_ += ref.nbytes;
    else local_weight_bytes_ += ref.nbytes;
    claim(ref);
}

void QwenWeightMap::claim(const QwenTensorRef& ref) {
    if (!ref.found || ref.name.empty()) return;
    // A descriptor claims exactly one checkpoint tensor. Duplicates would make
    // the coverage audit understate what the text path actually reads, so they
    // are a mapping bug rather than something to tolerate.
    if (!claimed_tensors_.insert(ref.name).second) {
        throw std::runtime_error("Qwen tensor is claimed twice by the weight map: " +
                                 ref.name);
    }
    const uint64_t full_bytes =
        safe_tensor_numel(ref.full_shape) * safe_dtype_size(ref.dtype);
    checkpoint_text_bytes_ += full_bytes;
    if (ref.rule == QwenShardRule::Replicated || ref.shard_size == 0) {
        replicated_local_bytes_ += ref.nbytes;
    } else {
        sharded_local_bytes_ += ref.nbytes;
    }
}

void QwenWeightMap::record_linear(const QwenLinearRef& ref) {
    if (ref.weight.found) record(ref.weight, false);
    if (ref.has_scale) record(ref.scale, true);
    if (ref.has_weight_global_scale) {
        ++tensor_count_;
        host_global_metadata_bytes_ += ref.weight_global_scale.nbytes;
        claim(ref.weight_global_scale);
    }
    if (ref.has_input_global_scale) {
        ++tensor_count_;
        host_global_metadata_bytes_ += ref.input_global_scale.nbytes;
        claim(ref.input_global_scale);
    }
    switch (ref.kind) {
        case QwenLinearKind::DenseF16:
            ++checkpoint_linear_kind_counts_.dense_f16;
            break;
        case QwenLinearKind::Fp8Block128:
            ++checkpoint_linear_kind_counts_.fp8_block128;
            break;
        case QwenLinearKind::Fp8Channel:
            ++checkpoint_linear_kind_counts_.fp8_channel;
            break;
        case QwenLinearKind::NvFp4Group16:
            ++checkpoint_linear_kind_counts_.nvfp4_group16;
            break;
    }
}

bool qwen_is_visual_tensor(const std::string& name) {
    // Official Qwen3.8 checkpoints ship the vision tower in the same shards as
    // the text model. The C++ runtime is text-only, so these tensors are never
    // mapped, never materialized and never uploaded.
    static const char* const kVisualPrefixes[] = {
        "model.visual.",
        "visual.",
    };
    for (const char* prefix : kVisualPrefixes) {
        const std::string candidate(prefix);
        if (name.size() >= candidate.size() &&
            name.compare(0, candidate.size(), candidate) == 0) {
            return true;
        }
    }
    return false;
}

QwenCoverage QwenWeightMap::coverage() const {
    QwenCoverage out;
    out.index_tensors = index_.tensor_count();
    out.mapped_tensors = claimed_tensors_.size();
    out.checkpoint_text_bytes = checkpoint_text_bytes_;
    out.replicated_local_bytes = replicated_local_bytes_;
    out.sharded_local_bytes = sharded_local_bytes_;

    for (const auto& [name, shard] : index_.weight_map()) {
        (void)shard;
        if (claimed_tensors_.count(name) != 0) continue;
        if (qwen_is_visual_tensor(name)) {
            ++out.visual_tensors;
            continue;
        }
        ++out.unexpected_tensors;
        if (out.unexpected_examples.size() < 8) out.unexpected_examples.push_back(name);
    }
    return out;
}

void QwenWeightMap::require_full_coverage() const {
    const QwenCoverage cover = coverage();
    if (cover.unexpected_tensors != 0) {
        std::ostringstream message;
        message << "Qwen checkpoint has " << cover.unexpected_tensors
                << " tensors the text runtime does not map and cannot classify as"
                   " vision weights:";
        for (const std::string& name : cover.unexpected_examples) {
            message << "\n  " << name;
        }
        throw std::runtime_error(message.str());
    }
    if (cover.mapped_tensors + cover.visual_tensors != cover.index_tensors) {
        throw std::runtime_error(
            "Qwen coverage accounting does not add up to the checkpoint index");
    }
}

}  // namespace pocket
