#pragma once

// Shared zero-weight Qwen3.5 checkpoint fixture for engine-level parity tests.
//
// The two-layer alternating config (one linear_attention, one full_attention)
// over a 64-token vocab loads in well under a second, which is what lets a
// parity test stand up two whole engine configurations instead of needing a real
// checkpoint. Callers pass their own directory so concurrent tests do not share
// one fixture.
//
// Extracted verbatim from test_qwen_batched_engine_parity.cpp, which now
// includes this header, so the two parity tests cannot drift onto different
// model geometries.

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace qwen_fixture {

struct TensorSpec {
    std::string name;
    std::string dtype;
    std::vector<uint64_t> shape;
};

uint64_t numel(const std::vector<uint64_t>& shape) {
    uint64_t out = 1;
    for (uint64_t dim : shape) out *= dim;
    return out;
}

uint64_t dtype_size(const std::string& dtype) {
    if (dtype == "F8_E4M3") return 1;
    if (dtype == "BF16") return 2;
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

// `name` distinguishes one test's fixture from another's, since two tests
// running at once would otherwise write the same shard.
std::string fixture_dir(const std::string& name) {
    const char* base = std::getenv("CLAUDE_JOB_DIR");
    const std::string root = base != nullptr ? std::string(base) + "/tmp"
                                              : std::string("/tmp");
    return root + "/" + name;
}

std::string config_json() {
    // Two-layer alternating: one linear_attention, one full_attention.
    // Context 8192 so decode at positions 4096, 5000, 6144, 7500 all fall inside
    // the kDecodeMinContext=4096 threshold where the fused split path runs.
    return R"JSON({
  "architectures": ["Qwen3_5ForConditionalGeneration"],
  "model_type": "qwen3_5",
  "text_config": {
    "model_type": "qwen3_5_text",
    "vocab_size": 64,
    "hidden_size": 128,
    "intermediate_size": 128,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "head_dim": 128,
    "attn_output_gate": true,
    "linear_num_key_heads": 4,
    "linear_num_value_heads": 4,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "max_position_embeddings": 8192,
    "rms_norm_eps": 1e-6,
    "partial_rotary_factor": 0.25,
    "rope_parameters": {"rope_type": "default", "rope_theta": 10000000},
    "layer_types": ["linear_attention", "full_attention"]
  }
})JSON";
}

std::vector<TensorSpec> specs(bool bf16_mlp = false) {
    const std::vector<uint64_t> hidden = {128};
    const std::vector<uint64_t> vocab = {64, 128};
    // linear_attention QKV: (q_heads + 2*kv_heads) * head_dim = (4+2*4)*128 = 1536
    const std::vector<uint64_t> qkv = {1536, 128};
    // FP8 block scale: ceil_div(1536, 128) × ceil_div(128, 128) = 12 × 1
    const std::vector<uint64_t> qkv_scale = {12, 1};
    const std::vector<uint64_t> value_proj = {512, 128};
    // FP8 block scale: ceil_div(512, 128) × ceil_div(128, 128) = 4 × 1
    const std::vector<uint64_t> value_scale = {4, 1};
    const std::vector<uint64_t> out_proj = {128, 512};
    // FP8 block scale: ceil_div(128, 128) × ceil_div(512, 128) = 1 × 4
    const std::vector<uint64_t> out_scale = {1, 4};
    const std::vector<uint64_t> heads = {4, 128};
    const std::vector<uint64_t> vector4 = {4};
    const std::vector<uint64_t> mlp = {128, 128};
    // FP8 block scale: ceil_div(128, 128) × ceil_div(128, 128) = 1 × 1
    const std::vector<uint64_t> mlp_scale = {1, 1};
    // full_attention Q projection: 2x heads for GQA = 8 * 128 = 1024
    const std::vector<uint64_t> q_proj = {1024, 128};
    // FP8 block scale: ceil_div(1024, 128) × ceil_div(128, 128) = 8 × 1
    const std::vector<uint64_t> q_proj_scale = {8, 1};
    const std::string linear = "model.language_model.layers.0.";
    const std::string full = "model.language_model.layers.1.";
    std::vector<TensorSpec> out = {
        {"model.language_model.embed_tokens.weight", "BF16", vocab},
        {"model.language_model.norm.weight", "BF16", hidden},
        {"lm_head.weight", "BF16", vocab},
        {linear + "input_layernorm.weight", "BF16", hidden},
        {linear + "post_attention_layernorm.weight", "BF16", hidden},
        {linear + "linear_attn.in_proj_qkv.weight", "F8_E4M3", qkv},
        {linear + "linear_attn.in_proj_qkv.weight_scale_inv", "BF16", qkv_scale},
        {linear + "linear_attn.in_proj_z.weight", "F8_E4M3", value_proj},
        {linear + "linear_attn.in_proj_z.weight_scale_inv", "BF16", value_scale},
        {linear + "linear_attn.out_proj.weight", "F8_E4M3", out_proj},
        {linear + "linear_attn.out_proj.weight_scale_inv", "BF16", out_scale},
        {linear + "linear_attn.in_proj_a.weight", "BF16", heads},
        {linear + "linear_attn.in_proj_b.weight", "BF16", heads},
        {linear + "linear_attn.conv1d.weight", "BF16", {1536, 1, 4}},
        {linear + "linear_attn.A_log", "BF16", vector4},
        {linear + "linear_attn.dt_bias", "BF16", vector4},
        {linear + "linear_attn.norm.weight", "BF16", hidden},
        {linear + "mlp.gate_proj.weight", bf16_mlp ? "BF16" : "F8_E4M3", mlp},
        {linear + "mlp.up_proj.weight", bf16_mlp ? "BF16" : "F8_E4M3", mlp},
        {linear + "mlp.down_proj.weight", bf16_mlp ? "BF16" : "F8_E4M3", mlp},
    };
    if (!bf16_mlp) {
        // FP8 block scales, swept into the list order above. BF16 MLPs need no
        // scale tensor: require_linear only takes the FP8 path when a matching
        // weight_scale_inv exists.
        out.push_back({linear + "mlp.gate_proj.weight_scale_inv", "BF16", mlp_scale});
        out.push_back({linear + "mlp.up_proj.weight_scale_inv", "BF16", mlp_scale});
        out.push_back({linear + "mlp.down_proj.weight_scale_inv", "BF16", mlp_scale});
    }
    out.insert(out.end(), {
        {full + "input_layernorm.weight", "BF16", hidden},
        {full + "post_attention_layernorm.weight", "BF16", hidden},
        {full + "self_attn.q_proj.weight", "F8_E4M3", q_proj},
        {full + "self_attn.q_proj.weight_scale_inv", "BF16", q_proj_scale},
        {full + "self_attn.k_proj.weight", "F8_E4M3", value_proj},
        {full + "self_attn.k_proj.weight_scale_inv", "BF16", value_scale},
        {full + "self_attn.v_proj.weight", "F8_E4M3", value_proj},
        {full + "self_attn.v_proj.weight_scale_inv", "BF16", value_scale},
        {full + "self_attn.o_proj.weight", "F8_E4M3", out_proj},
        {full + "self_attn.o_proj.weight_scale_inv", "BF16", out_scale},
        {full + "self_attn.q_norm.weight", "BF16", hidden},
        {full + "self_attn.k_norm.weight", "BF16", hidden},
        {full + "mlp.gate_proj.weight", bf16_mlp ? "BF16" : "F8_E4M3", mlp},
        {full + "mlp.up_proj.weight", bf16_mlp ? "BF16" : "F8_E4M3", mlp},
        {full + "mlp.down_proj.weight", bf16_mlp ? "BF16" : "F8_E4M3", mlp},
    });
    if (!bf16_mlp) {
        out.push_back({full + "mlp.gate_proj.weight_scale_inv", "BF16", mlp_scale});
        out.push_back({full + "mlp.up_proj.weight_scale_inv", "BF16", mlp_scale});
        out.push_back({full + "mlp.down_proj.weight_scale_inv", "BF16", mlp_scale});
    }
    return out;
}

bool write_fixture(const std::string& dir, bool bf16_mlp = false) {
    const std::string mkdir_cmd = "mkdir -p '" + dir + "'";
    if (std::system(mkdir_cmd.c_str()) != 0) return false;
    std::ofstream config(dir + "/config.json",
                        std::ios::binary | std::ios::trunc);
    if (!config) return false;
    config << config_json();
    if (!config) return false;

    const auto tensors = specs(bf16_mlp);
    std::ostringstream header;
    header << '{';
    uint64_t offset = 0;
    for (size_t i = 0; i < tensors.size(); ++i) {
        if (i) header << ',';
        const auto& tensor = tensors[i];
        const uint64_t bytes = numel(tensor.shape) * dtype_size(tensor.dtype);
        header << '"' << tensor.name << "\":{\"dtype\":\"" << tensor.dtype
               << "\",\"shape\":" << shape_json(tensor.shape)
               << ",\"data_offsets\":[" << offset << ',' << offset + bytes
               << "]}";
        offset += bytes;
    }
    header << '}';
    const std::string header_text = header.str();
    std::ofstream shard(dir + "/model.safetensors",
                       std::ios::binary | std::ios::trunc);
    if (!shard) return false;
    const uint64_t header_len = header_text.size();
    shard.write(reinterpret_cast<const char*>(&header_len),
               sizeof(header_len));
    shard.write(header_text.data(),
               static_cast<std::streamsize>(header_text.size()));
    std::vector<uint8_t> data(static_cast<size_t>(offset), 0);
    uint64_t data_offset = 0;
    for (const TensorSpec& tensor : tensors) {
        const uint64_t bytes = numel(tensor.shape) * dtype_size(tensor.dtype);
        if (tensor.dtype == "F8_E4M3") {
            std::fill(data.begin() + static_cast<ptrdiff_t>(data_offset),
                     data.begin() + static_cast<ptrdiff_t>(data_offset + bytes),
                     static_cast<uint8_t>(0x38));
        } else {
            const bool embedding =
                tensor.name.find("embed_tokens.weight") != std::string::npos;
            const uint64_t elements = bytes / 2;
            for (uint64_t element = 0; element < elements; ++element) {
                uint16_t bits = 0x3f80;
                if (embedding && !tensor.shape.empty()) {
                    const uint64_t row = element / tensor.shape.back();
                    bits = static_cast<uint16_t>(0x3f80u + (row & 0x1fu));
                }
                const uint64_t at = data_offset + element * 2;
                data[static_cast<size_t>(at)] =
                    static_cast<uint8_t>(bits & 0xffu);
                data[static_cast<size_t>(at + 1)] =
                    static_cast<uint8_t>(bits >> 8);
            }
        }
        data_offset += bytes;
    }
    shard.write(reinterpret_cast<const char*>(data.data()),
               static_cast<std::streamsize>(data.size()));
    if (!shard) return false;

    std::ofstream index(dir + "/model.safetensors.index.json",
                       std::ios::binary | std::ios::trunc);
    if (!index) return false;
    index << "{\"metadata\":{\"total_size\":" << offset
          << "},\"weight_map\":{";
    for (size_t i = 0; i < tensors.size(); ++i) {
        if (i) index << ',';
        index << '"' << tensors[i].name << "\":\"model.safetensors\"";
    }
    index << "}}";
    return static_cast<bool>(index);
}

}  // namespace qwen_fixture
