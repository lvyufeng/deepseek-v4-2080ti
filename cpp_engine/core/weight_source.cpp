#include "weight_source.hpp"

#include <algorithm>
#include <stdexcept>

namespace pocket {

namespace {

bool ends_with(const std::string& s, const std::string& suffix) {
    if (s.size() < suffix.size()) return false;
    return std::equal(suffix.rbegin(), suffix.rend(), s.rbegin());
}

DType safe_dtype_to_dtype(SafeDType s) {
    switch (s) {
        case SafeDType::BF16: return DType::BF16;
        case SafeDType::F16: return DType::F16;
        case SafeDType::F32: return DType::F32;
        case SafeDType::F8_E4M3: return DType::F8_E4M3;
        default: return DType::Unknown;
    }
}

uint64_t gguf_metadata_int(const GGUFFile& f, const std::string& key, uint64_t default_value) {
    if (auto v = f.metadata_u64(key)) return *v;
    if (auto v = f.metadata_i64(key)) return static_cast<uint64_t>(*v);
    return default_value;
}

}  // namespace

WeightView WeightSource::require(const std::string& name) const {
    WeightView view = get(name);
    if (!view.found) {
        throw std::runtime_error("WeightSource: required tensor missing: " + name);
    }
    return view;
}

bool is_gguf_path(const std::string& path) {
    return ends_with(path, ".gguf");
}

std::unique_ptr<WeightSource> open_weight_source(const std::string& path) {
    if (is_gguf_path(path)) {
        return std::make_unique<GGUFWeightSource>(path);
    }
    return std::make_unique<SafetensorsWeightSource>(path);
}

// ---------- SafetensorsWeightSource ----------

SafetensorsWeightSource::SafetensorsWeightSource(const std::string& ckpt_dir)
    : index_(ckpt_dir) {}

SafeTensorsShard& SafetensorsWeightSource::shard_for(const std::string& tensor_name) const {
    const std::string* shard_name = index_.shard_for_tensor(tensor_name);
    if (shard_name == nullptr) {
        throw std::runtime_error("safetensors index missing tensor: " + tensor_name);
    }
    auto it = shards_.find(*shard_name);
    if (it == shards_.end()) {
        auto shard = std::make_unique<SafeTensorsShard>(index_.shard_path(*shard_name));
        it = shards_.emplace(*shard_name, std::move(shard)).first;
    }
    return *it->second;
}

bool SafetensorsWeightSource::has(const std::string& name) const {
    return index_.shard_for_tensor(name) != nullptr;
}

WeightView SafetensorsWeightSource::get(const std::string& name) const {
    WeightView view;
    view.name = name;
    if (!has(name)) return view;
    SafeTensorsShard& shard = shard_for(name);
    const SafeTensorInfo* info = shard.find_tensor(name);
    if (info == nullptr) return view;
    view.dtype = safe_dtype_to_dtype(info->dtype);
    view.shape = info->shape;
    view.data = shard.tensor_data(*info);
    view.nbytes = info->nbytes;
    view.found = true;
    return view;
}

WeightView SafetensorsWeightSource::get_expert(const std::string& /*routed_name_3d*/,
                                                const std::string& per_expert_name,
                                                int /*expert_id*/) const {
    return get(per_expert_name);
}

// ---------- GGUFWeightSource ----------

GGUFWeightSource::GGUFWeightSource(const std::string& path) : file_(path) {
    int n_layers = static_cast<int>(gguf_metadata_int(file_, "deepseek4.block_count", 43));
    int n_hash_layers = static_cast<int>(gguf_metadata_int(file_, "deepseek4.hash_layer_count", 3));
    int n_experts = static_cast<int>(gguf_metadata_int(file_, "deepseek4.expert_count", 256));
    build_mapping(n_layers, n_hash_layers, n_experts);
}

void GGUFWeightSource::build_mapping(int n_layers, int n_hash_layers, int /*n_experts*/) {
    auto add = [&](const std::string& internal, const std::string& gguf, Transform transform) {
        by_internal_name_[internal] = Mapping{gguf, transform};
    };

    add("embed.weight", "token_embd.weight", Transform::Transpose2D);
    add("head.weight", "output.weight", Transform::Transpose2D);
    add("norm.weight", "output_norm.weight", Transform::Direct);
    add("hc_head_fn", "output_hc_fn.weight", Transform::Transpose2D);
    add("hc_head_base", "output_hc_base.weight", Transform::Direct);
    add("hc_head_scale", "output_hc_scale.weight", Transform::Direct);

    for (int layer = 0; layer < n_layers; ++layer) {
        const std::string gp = "blk." + std::to_string(layer);
        const std::string tp = "layers." + std::to_string(layer);

        add(tp + ".attn.attn_sink", gp + ".attn_sinks.weight", Transform::Direct);
        add(tp + ".attn.wq_a.weight", gp + ".attn_q_a.weight", Transform::Transpose2D);
        add(tp + ".attn.q_norm.weight", gp + ".attn_q_a_norm.weight", Transform::Direct);
        add(tp + ".attn.wq_b.weight", gp + ".attn_q_b.weight", Transform::Transpose2D);
        add(tp + ".attn.wkv.weight", gp + ".attn_kv.weight", Transform::Transpose2D);
        add(tp + ".attn.kv_norm.weight", gp + ".attn_kv_a_norm.weight", Transform::Direct);
        add(tp + ".attn.wo_a.weight", gp + ".attn_output_a.weight", Transform::Transpose2D);
        add(tp + ".attn.wo_b.weight", gp + ".attn_output_b.weight", Transform::Transpose2D);
        add(tp + ".attn_norm.weight", gp + ".attn_norm.weight", Transform::Direct);

        add(tp + ".ffn.gate.weight", gp + ".ffn_gate_inp.weight", Transform::Transpose2D);
        if (layer < n_hash_layers) {
            add(tp + ".ffn.gate.tid2eid", gp + ".ffn_gate_tid2eid.weight", Transform::Transpose2D);
        } else {
            add(tp + ".ffn.gate.bias", gp + ".exp_probs_b.bias", Transform::Direct);
        }

        // Routed experts share a 3D tensor in GGUF; per-expert slices resolve
        // through get_expert() rather than the per-expert internal name lookup.
        add(tp + ".ffn.experts.routed.w1", gp + ".ffn_gate_exps.weight", Transform::RoutedExpertTranspose);
        add(tp + ".ffn.experts.routed.w2", gp + ".ffn_down_exps.weight", Transform::RoutedExpertTranspose);
        add(tp + ".ffn.experts.routed.w3", gp + ".ffn_up_exps.weight", Transform::RoutedExpertTranspose);

        add(tp + ".ffn.shared_experts.w1.weight", gp + ".ffn_gate_shexp.weight", Transform::Transpose2D);
        add(tp + ".ffn.shared_experts.w2.weight", gp + ".ffn_down_shexp.weight", Transform::Transpose2D);
        add(tp + ".ffn.shared_experts.w3.weight", gp + ".ffn_up_shexp.weight", Transform::Transpose2D);
        add(tp + ".ffn_norm.weight", gp + ".ffn_norm.weight", Transform::Direct);

        add(tp + ".hc_attn_fn", gp + ".hc_attn_fn.weight", Transform::Transpose2D);
        add(tp + ".hc_attn_base", gp + ".hc_attn_base.weight", Transform::Direct);
        add(tp + ".hc_attn_scale", gp + ".hc_attn_scale.weight", Transform::Direct);
        add(tp + ".hc_ffn_fn", gp + ".hc_ffn_fn.weight", Transform::Transpose2D);
        add(tp + ".hc_ffn_base", gp + ".hc_ffn_base.weight", Transform::Direct);
        add(tp + ".hc_ffn_scale", gp + ".hc_ffn_scale.weight", Transform::Direct);

        add(tp + ".attn.compressor.ape", gp + ".attn_compressor_ape.weight", Transform::Transpose2D);
        add(tp + ".attn.compressor.wkv.weight", gp + ".attn_compressor_kv.weight", Transform::Transpose2D);
        add(tp + ".attn.compressor.wgate.weight", gp + ".attn_compressor_gate.weight", Transform::Transpose2D);
        add(tp + ".attn.compressor.norm.weight", gp + ".attn_compressor_norm.weight", Transform::Direct);

        add(tp + ".attn.indexer.wq_b.weight", gp + ".indexer.attn_q_b.weight", Transform::Transpose2D);
        add(tp + ".attn.indexer.weights_proj.weight", gp + ".indexer.proj.weight", Transform::Transpose2D);
        add(tp + ".attn.indexer.compressor.ape", gp + ".indexer_compressor_ape.weight", Transform::Transpose2D);
        add(tp + ".attn.indexer.compressor.wkv.weight", gp + ".indexer_compressor_kv.weight", Transform::Transpose2D);
        add(tp + ".attn.indexer.compressor.wgate.weight", gp + ".indexer_compressor_gate.weight", Transform::Transpose2D);
        add(tp + ".attn.indexer.compressor.norm.weight", gp + ".indexer_compressor_norm.weight", Transform::Direct);
    }
}

bool GGUFWeightSource::has(const std::string& name) const {
    auto it = by_internal_name_.find(name);
    if (it == by_internal_name_.end()) return false;
    return file_.find_tensor(it->second.gguf_name) != nullptr;
}

WeightView GGUFWeightSource::get(const std::string& name) const {
    WeightView view;
    view.name = name;
    auto it = by_internal_name_.find(name);
    if (it == by_internal_name_.end()) return view;
    const GGUFTensorInfo* info = file_.find_tensor(it->second.gguf_name);
    if (info == nullptr) return view;
    TensorView tv = file_.tensor_view(*info);
    view.dtype = tv.dtype;
    view.shape = tv.shape;
    view.data = tv.data;
    view.nbytes = tv.nbytes;
    view.found = true;
    return view;
}

WeightView GGUFWeightSource::get_expert(const std::string& routed_name_3d,
                                         const std::string& per_expert_name,
                                         int expert_id) const {
    WeightView view;
    view.name = per_expert_name;
    auto it = by_internal_name_.find(routed_name_3d);
    if (it == by_internal_name_.end()) return view;
    const GGUFTensorInfo* info = file_.find_tensor(it->second.gguf_name);
    if (info == nullptr) return view;
    if (info->shape.size() != 3) return view;
    const uint64_t a = info->shape[0];
    const uint64_t b = info->shape[1];
    const uint64_t n_experts = info->shape[2];
    if (static_cast<uint64_t>(expert_id) >= n_experts) return view;
    if (info->nbytes % n_experts != 0) return view;
    const uint64_t per_expert_nbytes = info->nbytes / n_experts;
    TensorView tv = file_.tensor_view(*info);
    view.dtype = info->dtype;
    view.shape = {b, a};  // routed_expert_transpose: [a,b,n_experts] -> per-expert [b,a]
    view.data = tv.data + per_expert_nbytes * static_cast<uint64_t>(expert_id);
    view.nbytes = per_expert_nbytes;
    view.found = true;
    return view;
}

}  // namespace pocket
