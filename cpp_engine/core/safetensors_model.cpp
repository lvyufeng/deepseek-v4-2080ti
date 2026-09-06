#include "safetensors_model.hpp"

#include <stdexcept>

namespace pocket {
namespace {

SafeTensorRef make_ref(const std::string& name, const std::string& shard, const SafeTensorInfo& info) {
    SafeTensorRef ref;
    ref.name = name;
    ref.shard_name = shard;
    ref.dtype = info.dtype;
    ref.shape = info.shape;
    return ref;
}

}  // namespace

SafeTensorsModelMap::SafeTensorsModelMap(const SafeTensorsIndex& index, const ModelConfig& config)
    : index_(index), config_(config) {
    embed_ = require_ref("embed.weight", SafeDType::BF16, 2);
    head_ = require_ref("head.weight", SafeDType::BF16, 2);

    if (config_.n_layers == 0) throw std::runtime_error("model config has zero layers");
    if (config_.n_routed_experts == 0) throw std::runtime_error("model config has zero routed experts");
    layers_.resize(config_.n_layers);
    for (uint64_t li = 0; li < config_.n_layers; ++li) {
        SafeLayerWeights& layer = layers_[li];
        const std::string prefix = "layers." + std::to_string(li) + ".";

        layer.attn_norm = require_ref(prefix + "attn_norm.weight", SafeDType::BF16, 1);
        layer.ffn_norm = require_ref(prefix + "ffn_norm.weight", SafeDType::BF16, 1);

        layer.attn.q_norm = require_ref(prefix + "attn.q_norm.weight", SafeDType::BF16, 1);
        layer.attn.kv_norm = require_ref(prefix + "attn.kv_norm.weight", SafeDType::BF16, 1);
        layer.attn.attn_sink = require_ref(prefix + "attn.attn_sink", SafeDType::F32, 1);
        layer.attn.wq_a = require_ref(prefix + "attn.wq_a.weight", SafeDType::F8_E4M3, 2);
        layer.attn.wq_b = require_ref(prefix + "attn.wq_b.weight", SafeDType::F8_E4M3, 2);
        layer.attn.wkv = require_ref(prefix + "attn.wkv.weight", SafeDType::F8_E4M3, 2);
        layer.attn.wo_a = require_ref(prefix + "attn.wo_a.weight", SafeDType::F8_E4M3, 2);
        layer.attn.wo_b = require_ref(prefix + "attn.wo_b.weight", SafeDType::F8_E4M3, 2);

        layer.ffn.gate_weight = require_ref(prefix + "ffn.gate.weight", SafeDType::BF16, 2);
        layer.ffn.gate_tid2eid = optional_ref(prefix + "ffn.gate.tid2eid", SafeDType::I64, 2);
        layer.ffn.experts.resize(config_.n_routed_experts);
        for (uint64_t e = 0; e < config_.n_routed_experts; ++e) {
            const std::string ep = prefix + "ffn.experts." + std::to_string(e) + ".";
            layer.ffn.experts[e].w1 = require_fp4_pair(ep + "w1.weight");
            layer.ffn.experts[e].w2 = require_fp4_pair(ep + "w2.weight");
            layer.ffn.experts[e].w3 = require_fp4_pair(ep + "w3.weight");
        }
    }
}

SafeTensorRef SafeTensorsModelMap::require_ref(const std::string& name, SafeDType dtype, size_t ndim) const {
    const std::string* shard_name = index_.shard_for_tensor(name);
    if (shard_name == nullptr) throw std::runtime_error("tensor not in checkpoint index: " + name);
    SafeTensorsShard shard(index_.shard_path(*shard_name));
    const SafeTensorInfo* info = shard.find_tensor(name);
    if (info == nullptr) throw std::runtime_error("tensor missing from shard header: " + name);
    if (info->dtype != dtype) {
        throw std::runtime_error("unexpected dtype for " + name + " expected=" + safe_dtype_name(dtype) + " actual=" + safe_dtype_name(info->dtype));
    }
    if (info->shape.size() != ndim) {
        throw std::runtime_error("unexpected ndim for " + name + " expected=" + std::to_string(ndim) + " actual=" + std::to_string(info->shape.size()));
    }
    return make_ref(name, *shard_name, *info);
}

SafeTensorRef SafeTensorsModelMap::optional_ref(const std::string& name, SafeDType dtype, size_t ndim) const {
    if (index_.shard_for_tensor(name) == nullptr) return {};
    return require_ref(name, dtype, ndim);
}

SafeFp4TensorPair SafeTensorsModelMap::require_fp4_pair(const std::string& weight_name) const {
    const std::string* shard_name = index_.shard_for_tensor(weight_name);
    if (shard_name == nullptr) throw std::runtime_error("FP4 weight not in checkpoint index: " + weight_name);
    SafeTensorsShard shard(index_.shard_path(*shard_name));
    return resolve_fp4_tensor_pair(index_, shard, weight_name);
}

}  // namespace pocket
