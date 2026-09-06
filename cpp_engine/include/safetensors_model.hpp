#pragma once

#include "model_config.hpp"
#include "safetensors_reader.hpp"

#include <string>
#include <vector>

namespace pocket {

struct SafeTensorRef {
    std::string name;
    std::string shard_name;
    SafeDType dtype = SafeDType::Unknown;
    std::vector<uint64_t> shape;
};

struct SafeAttentionWeights {
    SafeTensorRef q_norm;
    SafeTensorRef kv_norm;
    SafeTensorRef wq_a;
    SafeTensorRef wq_b;
    SafeTensorRef wkv;
    SafeTensorRef wo_a;
    SafeTensorRef wo_b;
    SafeTensorRef attn_sink;
};

struct SafeExpertWeights {
    SafeFp4TensorPair w1;
    SafeFp4TensorPair w2;
    SafeFp4TensorPair w3;
};

struct SafeMoeWeights {
    SafeTensorRef gate_weight;
    SafeTensorRef gate_tid2eid;
    std::vector<SafeExpertWeights> experts;
};

struct SafeLayerWeights {
    SafeTensorRef attn_norm;
    SafeTensorRef ffn_norm;
    SafeAttentionWeights attn;
    SafeMoeWeights ffn;
};

class SafeTensorsModelMap {
public:
    SafeTensorsModelMap(const SafeTensorsIndex& index, const ModelConfig& config);

    const SafeTensorRef& embed() const { return embed_; }
    const SafeTensorRef& head() const { return head_; }
    const std::vector<SafeLayerWeights>& layers() const { return layers_; }

private:
    SafeTensorRef require_ref(const std::string& name, SafeDType dtype, size_t ndim) const;
    SafeTensorRef optional_ref(const std::string& name, SafeDType dtype, size_t ndim) const;
    SafeFp4TensorPair require_fp4_pair(const std::string& weight_name) const;

    const SafeTensorsIndex& index_;
    ModelConfig config_;
    SafeTensorRef embed_;
    SafeTensorRef head_;
    std::vector<SafeLayerWeights> layers_;
};

}  // namespace pocket
