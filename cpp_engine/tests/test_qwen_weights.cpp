// Qwen Safetensors mapping test with a small synthetic TP4 checkpoint.
//
// The fixture preserves the real 128-row FP8 scale block contract while using
// one linear-attention and one full-attention layer, so it can verify packed
// QKV/conv segments and local full-attention KV heads without a model download.

#include "cuda_ops.hpp"
#include "qwen_config.hpp"
#include "qwen_weights.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

struct TensorSpec {
    std::string name;
    std::string dtype;
    std::vector<uint64_t> shape;
};

uint64_t numel(const std::vector<uint64_t>& shape) {
    uint64_t total = 1;
    for (uint64_t dim : shape) total *= dim;
    return total;
}

uint64_t dtype_size(const std::string& dtype) {
    if (dtype == "F8_E4M3" || dtype == "U8") return 1;
    if (dtype == "BF16") return 2;
    if (dtype == "F32") return 4;
    throw std::runtime_error("unsupported fixture dtype: " + dtype);
}

std::string shape_json(const std::vector<uint64_t>& shape) {
    std::ostringstream out;
    out << '[';
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i) out << ',';
        out << shape[i];
    }
    out << ']';
    return out.str();
}

std::string fixture_dir() {
    const char* base = std::getenv("CLAUDE_JOB_DIR");
    const std::string root = base != nullptr ? std::string(base) + "/tmp" : std::string("/tmp");
    return root + "/qwen_weights_fixture";
}

std::string config_json() {
    return R"JSON({
  "architectures": ["Qwen3_5ForConditionalGeneration"],
  "model_type": "qwen3_5",
  "text_config": {
    "model_type": "qwen3_5_text",
    "vocab_size": 512,
    "hidden_size": 128,
    "intermediate_size": 512,
    "num_hidden_layers": 2,
    "mtp_num_hidden_layers": 1,
    "mtp_use_dedicated_embeddings": false,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "head_dim": 128,
    "attn_output_gate": true,
    "linear_num_key_heads": 4,
    "linear_num_value_heads": 4,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "max_position_embeddings": 1024,
    "rms_norm_eps": 1e-6,
    "partial_rotary_factor": 0.25,
    "rope_parameters": {"rope_type": "default", "rope_theta": 10000000},
    "layer_types": ["linear_attention", "full_attention"]
  }
})JSON";
}

std::vector<TensorSpec> tensor_specs() {
    const std::vector<uint64_t> hidden = {128};
    const std::vector<uint64_t> vocab_weight = {512, 128};
    const std::vector<uint64_t> qkv = {1536, 128};
    const std::vector<uint64_t> qkv_scale = {12, 1};
    const std::vector<uint64_t> value_proj = {512, 128};
    const std::vector<uint64_t> value_proj_scale = {4, 1};
    const std::vector<uint64_t> out_proj = {128, 512};
    const std::vector<uint64_t> out_proj_scale = {1, 4};
    const std::vector<uint64_t> vector4 = {4};
    const std::vector<uint64_t> value_heads_weight = {4, 128};
    const std::vector<uint64_t> mlp = {512, 128};
    const std::vector<uint64_t> mlp_scale = {4, 1};
    const std::vector<uint64_t> down = {128, 512};
    const std::vector<uint64_t> down_scale = {1, 4};

    std::vector<TensorSpec> out;
    auto add = [&out](const std::string& name, const std::string& dtype,
                      const std::vector<uint64_t>& shape) {
        out.push_back({name, dtype, shape});
    };
    add("model.language_model.embed_tokens.weight", "BF16", vocab_weight);
    add("model.language_model.norm.weight", "BF16", hidden);
    add("lm_head.weight", "BF16", vocab_weight);

    const std::string linear_prefix = "model.language_model.layers.0.";
    add(linear_prefix + "input_layernorm.weight", "BF16", hidden);
    add(linear_prefix + "post_attention_layernorm.weight", "BF16", hidden);
    add(linear_prefix + "linear_attn.in_proj_qkv.weight", "F8_E4M3", qkv);
    add(linear_prefix + "linear_attn.in_proj_qkv.weight_scale_inv", "BF16", qkv_scale);
    add(linear_prefix + "linear_attn.in_proj_z.weight", "F8_E4M3", value_proj);
    add(linear_prefix + "linear_attn.in_proj_z.weight_scale_inv", "BF16", value_proj_scale);
    add(linear_prefix + "linear_attn.out_proj.weight", "F8_E4M3", out_proj);
    add(linear_prefix + "linear_attn.out_proj.weight_scale_inv", "BF16", out_proj_scale);
    add(linear_prefix + "linear_attn.in_proj_a.weight", "BF16", value_heads_weight);
    add(linear_prefix + "linear_attn.in_proj_b.weight", "BF16", value_heads_weight);
    add(linear_prefix + "linear_attn.conv1d.weight", "BF16", {1536, 1, 4});
    add(linear_prefix + "linear_attn.A_log", "BF16", vector4);
    add(linear_prefix + "linear_attn.dt_bias", "BF16", vector4);
    add(linear_prefix + "linear_attn.norm.weight", "BF16", {128});

    const std::string full_prefix = "model.language_model.layers.1.";
    const std::vector<uint64_t> q_proj = {1024, 128};
    const std::vector<uint64_t> q_proj_scale = {8, 1};
    add(full_prefix + "input_layernorm.weight", "BF16", hidden);
    add(full_prefix + "post_attention_layernorm.weight", "BF16", hidden);
    add(full_prefix + "self_attn.q_proj.weight", "F8_E4M3", q_proj);
    add(full_prefix + "self_attn.q_proj.weight_scale_inv", "BF16", q_proj_scale);
    add(full_prefix + "self_attn.k_proj.weight", "F8_E4M3", value_proj);
    add(full_prefix + "self_attn.k_proj.weight_scale_inv", "BF16", value_proj_scale);
    add(full_prefix + "self_attn.v_proj.weight", "F8_E4M3", value_proj);
    add(full_prefix + "self_attn.v_proj.weight_scale_inv", "BF16", value_proj_scale);
    add(full_prefix + "self_attn.o_proj.weight", "F8_E4M3", out_proj);
    add(full_prefix + "self_attn.o_proj.weight_scale_inv", "BF16", out_proj_scale);
    add(full_prefix + "self_attn.q_norm.weight", "BF16", {128});
    add(full_prefix + "self_attn.k_norm.weight", "BF16", {128});

    auto add_mlp = [&add, &mlp, &mlp_scale, &down, &down_scale](const std::string& prefix) {
        add(prefix + "mlp.gate_proj.weight", "F8_E4M3", mlp);
        add(prefix + "mlp.gate_proj.weight_scale_inv", "BF16", mlp_scale);
        add(prefix + "mlp.up_proj.weight", "F8_E4M3", mlp);
        add(prefix + "mlp.up_proj.weight_scale_inv", "BF16", mlp_scale);
        add(prefix + "mlp.down_proj.weight", "F8_E4M3", down);
        add(prefix + "mlp.down_proj.weight_scale_inv", "BF16", down_scale);
    };
    add_mlp(linear_prefix);
    add_mlp(full_prefix);

    add("mtp.pre_fc_norm_embedding.weight", "BF16", hidden);
    add("mtp.pre_fc_norm_hidden.weight", "BF16", hidden);
    add("mtp.norm.weight", "BF16", hidden);
    add("mtp.fc.weight", "BF16", {128, 256});
    const std::string mtp_prefix = "mtp.layers.0.";
    add(mtp_prefix + "input_layernorm.weight", "BF16", hidden);
    add(mtp_prefix + "post_attention_layernorm.weight", "BF16", hidden);
    add(mtp_prefix + "self_attn.q_proj.weight", "F8_E4M3", q_proj);
    add(mtp_prefix + "self_attn.q_proj.weight_scale_inv", "BF16", q_proj_scale);
    add(mtp_prefix + "self_attn.k_proj.weight", "F8_E4M3", value_proj);
    add(mtp_prefix + "self_attn.k_proj.weight_scale_inv", "BF16", value_proj_scale);
    add(mtp_prefix + "self_attn.v_proj.weight", "F8_E4M3", value_proj);
    add(mtp_prefix + "self_attn.v_proj.weight_scale_inv", "BF16", value_proj_scale);
    add(mtp_prefix + "self_attn.o_proj.weight", "F8_E4M3", out_proj);
    add(mtp_prefix + "self_attn.o_proj.weight_scale_inv", "BF16", out_proj_scale);
    add(mtp_prefix + "self_attn.q_norm.weight", "BF16", {128});
    add(mtp_prefix + "self_attn.k_norm.weight", "BF16", {128});
    add_mlp(mtp_prefix);
    return out;
}

bool write_file(const std::string& path, const std::string& contents) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) return false;
    out.write(contents.data(), static_cast<std::streamsize>(contents.size()));
    return static_cast<bool>(out);
}

bool write_specs_fixture(const std::string& dir,
                         const std::vector<TensorSpec>& specs,
                         bool sentinel) {
    const std::string mkdir_cmd = "mkdir -p '" + dir + "'";
    if (std::system(mkdir_cmd.c_str()) != 0) return false;
    if (!write_file(dir + "/config.json", config_json())) return false;

    std::ostringstream header;
    header << '{';
    uint64_t offset = 0;
    for (size_t i = 0; i < specs.size(); ++i) {
        if (i) header << ',';
        const TensorSpec& spec = specs[i];
        const uint64_t bytes = numel(spec.shape) * dtype_size(spec.dtype);
        header << '"' << spec.name << "\":{\"dtype\":\"" << spec.dtype
               << "\",\"shape\":" << shape_json(spec.shape)
               << ",\"data_offsets\":[" << offset << ',' << (offset + bytes) << "]}";
        offset += bytes;
    }
    header << '}';

    std::ofstream shard(dir + "/model.safetensors",
                        std::ios::binary | std::ios::trunc);
    if (!shard) return false;
    const uint64_t header_len = header.str().size();
    shard.write(reinterpret_cast<const char*>(&header_len), sizeof(header_len));
    shard.write(header.str().data(), static_cast<std::streamsize>(header.str().size()));
    for (size_t tensor = 0; tensor < specs.size(); ++tensor) {
        const TensorSpec& spec = specs[tensor];
        const size_t bytes = static_cast<size_t>(
            numel(spec.shape) * dtype_size(spec.dtype));
        std::vector<uint8_t> payload(bytes, 0);
        if (sentinel) {
            for (size_t i = 0; i < bytes; ++i) {
                payload[i] = static_cast<uint8_t>(
                    (tensor * 29 + i + i / 251) & 0xffu);
            }
            if (spec.dtype == "F32" && bytes == sizeof(float)) {
                const float value = spec.name.find("weight_global") !=
                        std::string::npos ? 4.0f : 2.0f;
                std::memcpy(payload.data(), &value, sizeof(value));
            }
        }
        shard.write(reinterpret_cast<const char*>(payload.data()),
                    static_cast<std::streamsize>(payload.size()));
    }
    if (!shard) return false;

    std::ostringstream index;
    index << "{\"metadata\":{\"total_size\":" << offset
          << "},\"weight_map\":{";
    for (size_t i = 0; i < specs.size(); ++i) {
        if (i) index << ',';
        index << '"' << specs[i].name << "\":\"model.safetensors\"";
    }
    index << "}}";
    return write_file(dir + "/model.safetensors.index.json", index.str());
}

bool write_fixture(const std::string& dir) {
    return write_specs_fixture(dir, tensor_specs(), false);
}

std::string mixed_fixture_dir() {
    const char* base = std::getenv("CLAUDE_JOB_DIR");
    const std::string root = base != nullptr
        ? std::string(base) + "/tmp" : std::string("/tmp");
    return root + "/qwen_weights_mixed_tp2_fixture";
}

std::vector<TensorSpec> mixed_tensor_specs() {
    std::vector<TensorSpec> specs = tensor_specs();
    auto erase = [&specs](const std::string& name) {
        specs.erase(std::remove_if(
            specs.begin(), specs.end(),
            [&](const TensorSpec& spec) { return spec.name == name; }),
            specs.end());
    };
    erase("lm_head.weight");
    specs.push_back({"lm_head.weight", "F8_E4M3", {512, 128}});
    specs.push_back({"lm_head.weight_scale", "BF16", {512, 1}});
    for (int layer = 0; layer < 2; ++layer) {
        const std::string prefix = "model.language_model.layers." +
            std::to_string(layer) + ".mlp.";
        for (const std::string projection : {
                 "gate_proj", "up_proj", "down_proj"}) {
            const std::string base = prefix + projection;
            erase(base + ".weight");
            erase(base + ".weight_scale_inv");
            const std::vector<uint64_t> logical = projection == "down_proj"
                ? std::vector<uint64_t>{128, 512}
                : std::vector<uint64_t>{512, 128};
            specs.push_back({base + ".weight_packed", "U8",
                             {logical[0], logical[1] / 2}});
            specs.push_back({base + ".weight_scale", "F8_E4M3",
                             {logical[0], logical[1] / 16}});
            specs.push_back({base + ".weight_global_scale", "F32", {1}});
            specs.push_back({base + ".input_global_scale", "F32", {1}});
        }
    }
    return specs;
}

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void require_segment(const pocket::QwenTensorRef& ref, size_t index,
                     uint64_t start, uint64_t size, const std::string& label) {
    require(index < ref.segments.size(), label + " missing segment");
    require(ref.segments[index].first == start && ref.segments[index].second == size,
            label + " wrong segment");
}

void check_rank(const pocket::SafeTensorsIndex& index, const pocket::QwenConfig& config, int rank) {
    pocket::QwenWeightMap map(index, config, 4, rank);
    require(map.layers().size() == 2, "expected two mapped layers");
    const auto& linear = map.layers()[0].linear_attention;
    const auto& full = map.layers()[1].full_attention;

    require(linear.in_proj_a.weight.dtype == pocket::SafeDType::BF16,
            "in_proj_a must remain BF16 in storage");
    require(linear.in_proj_a.weight.local_shape == std::vector<uint64_t>({1, 128}),
            "in_proj_a local shape");
    require(linear.in_proj_a.weight.device_dtype == pocket::SafeDType::F16,
            "in_proj_a must upload as FP16 on Turing");
    require(linear.in_proj_b.weight.dtype == pocket::SafeDType::BF16,
            "in_proj_b must remain BF16 in storage");
    require(linear.in_proj_b.weight.device_dtype == pocket::SafeDType::F16,
            "in_proj_b must upload as FP16 on Turing");
    require(linear.in_proj_qkv.kind == pocket::QwenLinearKind::Fp8Block128,
            "legacy QKV must use block-128 FP8");
    require(linear.in_proj_qkv.weight.dtype == pocket::SafeDType::F8_E4M3 &&
                linear.in_proj_qkv.weight.device_dtype == pocket::SafeDType::F8_E4M3,
            "QKV FP8 must remain compressed on device");
    require(linear.in_proj_qkv.scale.dtype == pocket::SafeDType::BF16 &&
                linear.in_proj_qkv.scale.device_dtype == pocket::SafeDType::F16,
            "FP8 scale must upload as FP16 on Turing");
    require(!linear.in_proj_a.has_scale && !linear.in_proj_b.has_scale,
            "BF16 in_proj_a/b must not have FP8 scales");

    require(linear.in_proj_qkv.weight.local_shape == std::vector<uint64_t>({384, 128}),
            "QKV local shape");
    require(linear.in_proj_qkv.weight.segments.size() == 3, "QKV weight segment count");
    require(linear.in_proj_qkv.scale.local_shape == std::vector<uint64_t>({3, 1}),
            "QKV scale local shape");
    require(linear.in_proj_qkv.scale.segments.size() == 3, "QKV scale segment count");

    const uint64_t row_start = static_cast<uint64_t>(rank) * 128;
    require_segment(linear.in_proj_qkv.weight, 0, row_start, 128, "QKV weight Q");
    require_segment(linear.in_proj_qkv.weight, 1, 512 + row_start, 128, "QKV weight K");
    require_segment(linear.in_proj_qkv.weight, 2, 1024 + row_start, 128, "QKV weight V");
    const uint64_t block_start = static_cast<uint64_t>(rank);
    require_segment(linear.in_proj_qkv.scale, 0, block_start, 1, "QKV scale Q");
    require_segment(linear.in_proj_qkv.scale, 1, 4 + block_start, 1, "QKV scale K");
    require_segment(linear.in_proj_qkv.scale, 2, 8 + block_start, 1, "QKV scale V");

    require(linear.conv1d.local_shape == std::vector<uint64_t>({384, 1, 4}),
            "conv local shape");
    require(linear.conv1d.segments.size() == 3, "conv segment count");
    require_segment(linear.conv1d, 0, row_start, 128, "conv Q");
    require_segment(linear.conv1d, 1, 512 + row_start, 128, "conv K");
    require_segment(linear.conv1d, 2, 1024 + row_start, 128, "conv V");

    require(linear.in_proj_z.weight.local_shape == std::vector<uint64_t>({128, 128}),
            "in_proj_z local shape");
    require(linear.in_proj_z.scale.local_shape == std::vector<uint64_t>({1, 1}),
            "in_proj_z scale local shape");
    require(linear.in_proj_z.weight.shard_start == row_start &&
                linear.in_proj_z.scale.shard_start == static_cast<uint64_t>(rank),
            "in_proj_z shard offsets");

    require(linear.out_proj.weight.local_shape == std::vector<uint64_t>({128, 128}),
            "out_proj local shape");
    require(linear.out_proj.scale.local_shape == std::vector<uint64_t>({1, 1}),
            "out_proj scale local shape");
    require(linear.out_proj.weight.shard_start == row_start &&
                linear.out_proj.scale.shard_start == static_cast<uint64_t>(rank),
            "out_proj shard offsets");

    for (const auto* kv : {&full.k_proj, &full.v_proj}) {
        require(kv->weight.rule == pocket::QwenShardRule::ColumnParallel,
                "full KV weight must be column parallel");
        require(kv->weight.local_shape == std::vector<uint64_t>({128, 128}),
                "full KV local shape");
        require(kv->scale.local_shape == std::vector<uint64_t>({1, 1}),
                "full KV scale local shape");
        require(kv->weight.shard_start == row_start &&
                    kv->scale.shard_start == static_cast<uint64_t>(rank),
                "full KV shard offsets");
    }

    require(map.embed_tokens().local_shape == std::vector<uint64_t>({128, 128}),
            "embedding local shape");
    require(map.embed_tokens().device_dtype == pocket::SafeDType::F16,
            "embedding must upload as FP16");
    require(map.lm_head().kind == pocket::QwenLinearKind::DenseF16,
            "head linear kind");
    require(map.lm_head().logical_local_shape ==
                std::vector<uint64_t>({128, 128}) &&
                map.lm_head().weight.local_shape ==
                    std::vector<uint64_t>({128, 128}),
            "head local shape");
    require(map.lm_head().weight.device_dtype == pocket::SafeDType::F16,
            "head must upload as FP16");

    const pocket::QwenMtpWeights& mtp = map.mtp();
    require(mtp.found, "native MTP tensors must be mapped");
    require(mtp.fc.weight.dtype == pocket::SafeDType::BF16 &&
                mtp.fc.weight.device_dtype == pocket::SafeDType::F16 &&
                !mtp.fc.has_scale,
            "MTP fusion projection must stay dense FP16 on Turing");
    require(mtp.fc.weight.rule == pocket::QwenShardRule::Replicated &&
                mtp.fc.weight.local_shape == std::vector<uint64_t>({128, 256}),
            "MTP fusion projection must be replicated");
    require(mtp.pre_fc_norm_embedding.device_dtype == pocket::SafeDType::F16 &&
                mtp.pre_fc_norm_hidden.device_dtype == pocket::SafeDType::F16 &&
                mtp.norm.device_dtype == pocket::SafeDType::F16,
            "MTP norms must materialize as FP16");
    const auto& mtp_full = mtp.layer.full_attention;
    require(mtp_full.q_proj.weight.local_shape ==
                std::vector<uint64_t>({256, 128}),
            "MTP Q projection local shape");
    require(mtp_full.k_proj.weight.local_shape ==
                std::vector<uint64_t>({128, 128}) &&
                mtp_full.v_proj.weight.local_shape ==
                    std::vector<uint64_t>({128, 128}),
            "MTP KV projection local shapes");
    require(mtp_full.o_proj.weight.local_shape ==
                std::vector<uint64_t>({128, 128}),
            "MTP output projection local shape");

    const pocket::QwenHostTensor qkv_host =
        pocket::qwen_materialize_host_tensor(index, linear.in_proj_qkv.weight);
    require(qkv_host.storage_dtype == pocket::SafeDType::F8_E4M3 &&
                qkv_host.device_dtype == pocket::SafeDType::F8_E4M3,
            "QKV host materializer must preserve FP8");
    require(qkv_host.bytes.size() == 384u * 128u, "QKV host bytes");
    const pocket::QwenHostTensor scale_host =
        pocket::qwen_materialize_host_tensor(index, linear.in_proj_qkv.scale);
    require(scale_host.storage_dtype == pocket::SafeDType::BF16 &&
                scale_host.device_dtype == pocket::SafeDType::F16,
            "scale host materializer must convert to FP16");
    require(scale_host.bytes.size() == 3u * sizeof(uint16_t), "scale host bytes");
    const pocket::QwenHostTensor a_host =
        pocket::qwen_materialize_host_tensor(index, linear.in_proj_a.weight);
    require(a_host.device_dtype == pocket::SafeDType::F16 &&
                a_host.bytes.size() == 128u * sizeof(uint16_t),
            "in_proj_a host materializer must convert local BF16 shard");

    require(pocket::qwen_bf16_to_fp16_bits(0x3f80u) == 0x3c00u,
            "BF16 1.0 converts to FP16 1.0");
    require(pocket::qwen_bf16_to_fp16_bits(0xbf80u) == 0xbc00u,
            "BF16 -1.0 converts to FP16 -1.0");
    if (pocket::cuda_runtime_available()) {
        pocket::QwenDeviceTensor device_scale =
            pocket::qwen_upload_tensor(index, linear.in_proj_qkv.scale);
        require(device_scale.device_dtype == pocket::SafeDType::F16,
                "uploaded scale dtype");
        std::vector<uint8_t> device_bytes(scale_host.bytes.size(), 0);
        require(cudaDeviceSynchronize() == cudaSuccess, "scale upload sync");
        require(cudaMemcpy(device_bytes.data(), device_scale.data, device_bytes.size(),
                           cudaMemcpyDeviceToHost) == cudaSuccess,
                "scale device readback");
        require(device_bytes == scale_host.bytes,
                "scale device bytes are FP16 materialized bytes");
    }
    require(map.local_weight_bytes() > 0 && map.local_scale_bytes() > 0,
            "weight byte accounting");
}

void check_mixed_tp2() {
    const std::string dir = mixed_fixture_dir();
    require(write_specs_fixture(dir, mixed_tensor_specs(), true),
            "could not create mixed TP2 fixture");
    const pocket::QwenConfig config = pocket::QwenConfig::from_hf_config(dir);
    pocket::SafeTensorsIndex index(dir);
    pocket::QwenWeightMap rank0(index, config, 2, 0);
    pocket::QwenWeightMap rank1(index, config, 2, 1);
    for (const pocket::QwenWeightMap* map : {&rank0, &rank1}) {
        require(map->lm_head().kind == pocket::QwenLinearKind::Fp8Channel,
                "mixed head kind");
        require(map->lm_head().logical_local_shape ==
                    std::vector<uint64_t>({256, 128}) &&
                map->lm_head().scale.local_shape ==
                    std::vector<uint64_t>({256, 1}),
                "mixed head local shape");
        require(map->mtp().fc.kind == pocket::QwenLinearKind::DenseF16,
                "mixed MTP remains dense");
        const auto kinds = map->checkpoint_linear_kind_counts();
        require(kinds.dense_f16 == 3 && kinds.fp8_block128 == 14 &&
                    kinds.fp8_channel == 1 && kinds.nvfp4_group16 == 6,
                "mixed checkpoint linear kind counts dense=" +
                    std::to_string(kinds.dense_f16) + " block=" +
                    std::to_string(kinds.fp8_block128) + " channel=" +
                    std::to_string(kinds.fp8_channel) + " nvfp4=" +
                    std::to_string(kinds.nvfp4_group16));
        const auto& mlp = map->layers()[0].mlp;
        require(mlp.gate_proj.kind == pocket::QwenLinearKind::NvFp4Group16 &&
                    mlp.up_proj.kind == pocket::QwenLinearKind::NvFp4Group16 &&
                    mlp.down_proj.kind == pocket::QwenLinearKind::NvFp4Group16,
                "mixed MLP NVFP4 kinds");
        require(mlp.gate_proj.logical_local_shape ==
                    std::vector<uint64_t>({256, 128}) &&
                mlp.gate_proj.weight.local_shape ==
                    std::vector<uint64_t>({256, 64}) &&
                mlp.gate_proj.scale.local_shape ==
                    std::vector<uint64_t>({256, 8}),
                "NVFP4 column shard shapes");
        require(mlp.down_proj.logical_local_shape ==
                    std::vector<uint64_t>({128, 256}) &&
                mlp.down_proj.weight.local_shape ==
                    std::vector<uint64_t>({128, 128}) &&
                mlp.down_proj.scale.local_shape ==
                    std::vector<uint64_t>({128, 16}),
                "NVFP4 row shard shapes");
    }
    require(rank0.layers()[0].mlp.down_proj.weight.shard_start == 0 &&
                rank1.layers()[0].mlp.down_proj.weight.shard_start == 128 &&
                rank1.layers()[0].mlp.down_proj.scale.shard_start == 16,
            "NVFP4 logical K shard offsets");
    const auto host0 = pocket::qwen_materialize_nvfp4_host_linear(
        index, rank0.layers()[0].mlp.gate_proj);
    const auto host1 = pocket::qwen_materialize_nvfp4_host_linear(
        index, rank1.layers()[0].mlp.gate_proj);
    require(host0.logical_shape == std::vector<uint64_t>({256, 128}) &&
                host0.blocks.size() == 512 &&
                host0.weight_global_factor == 0.25f &&
                host0.input_global_scale == 2.0f,
            "NVFP4 host interleave metadata");
    require(host0.blocks.front().qs[1] != host1.blocks.front().qs[1],
            "NVFP4 nonzero rank sentinel");
    const auto packed0 = pocket::qwen_materialize_host_tensor(
        index, rank0.layers()[0].mlp.gate_proj.weight);
    const auto scales0 = pocket::qwen_materialize_host_tensor(
        index, rank0.layers()[0].mlp.gate_proj.scale);
    const uint64_t interleaved_bytes = host0.blocks.size() *
        sizeof(pocket::QwenNvfp4Block64);
    require(packed0.bytes.size() == 256u * 128u / 2u &&
                scales0.bytes.size() == 256u * 128u / 16u &&
                interleaved_bytes == packed0.bytes.size() + scales0.bytes.size(),
            "NVFP4 resident weight/scale accounting");
    require(rank0.host_global_metadata_bytes() == 12u * sizeof(float),
            "NVFP4 host-only global metadata accounting");
    require(std::equal(std::begin(host0.blocks.front().qs),
                       std::end(host0.blocks.front().qs),
                       packed0.bytes.begin()),
            "NVFP4 block packed bytes");
    require(std::equal(std::begin(host0.blocks.front().d),
                       std::end(host0.blocks.front().d),
                       scales0.bytes.begin()),
            "NVFP4 block scale bytes");
}

}  // namespace

int main() {
    try {
        const std::string dir = fixture_dir();
        if (!write_fixture(dir)) {
            std::cout << "[SKIP] could not create Qwen weight fixture\n";
            return 0;
        }
        const pocket::QwenConfig config = pocket::QwenConfig::from_hf_config(dir);
        pocket::SafeTensorsIndex index(dir);
        for (int rank = 0; rank < 4; ++rank) check_rank(index, config, rank);
        check_mixed_tp2();
        std::cout << "[PASS] test_qwen_weights tp=4 legacy tp=2 mixed layers="
                  << config.num_hidden_layers << "\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cout << "[FAIL] test_qwen_weights " << ex.what() << "\n";
        return 1;
    }
}
