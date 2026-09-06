// Official BF16 Qwen3.8 checkpoint mapping test. Host-only: it links the
// device-agnostic core, so it runs on a machine with no accelerator toolkit.
//
// Two halves. The first writes a synthetic dense-BF16 TP4 fixture that carries
// the official tensor naming plus a bundled vision tower, and asserts the text
// map covers exactly the text tensors while classifying the vision ones as
// deliberately ignored. The second runs the same assertions against the real
// checkpoint when QWEN38_CKPT points at one, using the authoritative dimensions
// of Qwen/Qwen3.8-27B.

#include "qwen_config.hpp"
#include "qwen_weights.hpp"
#include "safetensors_reader.hpp"

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace {

struct TensorSpec {
    std::string name;
    std::vector<uint64_t> shape;
};

uint64_t numel(const std::vector<uint64_t>& shape) {
    uint64_t total = 1;
    for (uint64_t dim : shape) total *= dim;
    return total;
}

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
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
    const std::string root = base != nullptr ? std::string(base) + "/tmp"
                                            : std::string("/tmp");
    return root + "/qwen_bf16_checkpoint_fixture";
}

// Mirrors the official root layout: multimodal architecture at the root with the
// text model nested under text_config, alongside a vision_config the text
// runtime must ignore.
std::string config_json() {
    return R"JSON({
  "architectures": ["Qwen3_5ForConditionalGeneration"],
  "model_type": "qwen3_5",
  "language_model_only": false,
  "tie_word_embeddings": false,
  "image_token_id": 248056,
  "vision_config": {"model_type": "qwen3_5_vit", "depth": 2, "hidden_size": 64},
  "text_config": {
    "model_type": "qwen3_5_text",
    "vocab_size": 512,
    "hidden_size": 128,
    "intermediate_size": 512,
    "num_hidden_layers": 2,
    "mtp_num_hidden_layers": 1,
    "mtp_use_dedicated_embeddings": false,
    "num_attention_heads": 8,
    "num_key_value_heads": 4,
    "head_dim": 32,
    "attn_output_gate": true,
    "linear_num_key_heads": 8,
    "linear_num_value_heads": 16,
    "linear_key_head_dim": 32,
    "linear_value_head_dim": 32,
    "linear_conv_kernel_dim": 4,
    "max_position_embeddings": 1024,
    "rms_norm_eps": 1e-06,
    "partial_rotary_factor": 0.25,
    "rope_parameters": {"rope_type": "default", "rope_theta": 10000000},
    "layer_types": ["linear_attention", "full_attention"]
  }
})JSON";
}

// Every tensor is dense BF16, as in the official checkpoint: no weight_scale_inv,
// no weight_packed, no global scales.
std::vector<TensorSpec> text_specs() {
    const uint64_t hidden = 128;
    const uint64_t vocab = 512;
    const uint64_t intermediate = 512;
    const uint64_t key_dim = 8 * 32;    // 256
    const uint64_t value_dim = 16 * 32; // 512
    const uint64_t attention_dim = 8 * 32;
    const uint64_t q_dim = attention_dim * 2;  // output gate doubles the rows
    const uint64_t kv_dim = 4 * 32;

    std::vector<TensorSpec> out;
    auto add = [&out](const std::string& name, std::vector<uint64_t> shape) {
        out.push_back({name, std::move(shape)});
    };

    add("model.language_model.embed_tokens.weight", {vocab, hidden});
    add("model.language_model.norm.weight", {hidden});
    add("lm_head.weight", {vocab, hidden});

    const std::string linear = "model.language_model.layers.0.";
    add(linear + "input_layernorm.weight", {hidden});
    add(linear + "post_attention_layernorm.weight", {hidden});
    add(linear + "linear_attn.in_proj_qkv.weight", {2 * key_dim + value_dim, hidden});
    add(linear + "linear_attn.in_proj_z.weight", {value_dim, hidden});
    add(linear + "linear_attn.out_proj.weight", {hidden, value_dim});
    add(linear + "linear_attn.in_proj_a.weight", {16, hidden});
    add(linear + "linear_attn.in_proj_b.weight", {16, hidden});
    add(linear + "linear_attn.conv1d.weight", {2 * key_dim + value_dim, 1, 4});
    add(linear + "linear_attn.A_log", {16});
    add(linear + "linear_attn.dt_bias", {16});
    add(linear + "linear_attn.norm.weight", {32});

    const std::string full = "model.language_model.layers.1.";
    add(full + "input_layernorm.weight", {hidden});
    add(full + "post_attention_layernorm.weight", {hidden});
    add(full + "self_attn.q_proj.weight", {q_dim, hidden});
    add(full + "self_attn.k_proj.weight", {kv_dim, hidden});
    add(full + "self_attn.v_proj.weight", {kv_dim, hidden});
    add(full + "self_attn.o_proj.weight", {hidden, attention_dim});
    add(full + "self_attn.q_norm.weight", {32});
    add(full + "self_attn.k_norm.weight", {32});

    const std::string mtp_layer = "mtp.layers.0.";
    add("mtp.pre_fc_norm_embedding.weight", {hidden});
    add("mtp.pre_fc_norm_hidden.weight", {hidden});
    add("mtp.norm.weight", {hidden});
    add("mtp.fc.weight", {hidden, 2 * hidden});
    add(mtp_layer + "input_layernorm.weight", {hidden});
    add(mtp_layer + "post_attention_layernorm.weight", {hidden});
    add(mtp_layer + "self_attn.q_proj.weight", {q_dim, hidden});
    add(mtp_layer + "self_attn.k_proj.weight", {kv_dim, hidden});
    add(mtp_layer + "self_attn.v_proj.weight", {kv_dim, hidden});
    add(mtp_layer + "self_attn.o_proj.weight", {hidden, attention_dim});
    add(mtp_layer + "self_attn.q_norm.weight", {32});
    add(mtp_layer + "self_attn.k_norm.weight", {32});

    for (const std::string& prefix : {linear, full, mtp_layer}) {
        add(prefix + "mlp.gate_proj.weight", {intermediate, hidden});
        add(prefix + "mlp.up_proj.weight", {intermediate, hidden});
        add(prefix + "mlp.down_proj.weight", {hidden, intermediate});
    }
    return out;
}

std::vector<TensorSpec> visual_specs() {
    std::vector<TensorSpec> out;
    out.push_back({"model.visual.patch_embed.proj.weight", {64, 3, 2, 2}});
    out.push_back({"model.visual.patch_embed.proj.bias", {64}});
    out.push_back({"model.visual.pos_embed.weight", {16, 64}});
    for (int block = 0; block < 2; ++block) {
        const std::string prefix =
            "model.visual.blocks." + std::to_string(block) + ".";
        out.push_back({prefix + "attn.qkv.weight", {192, 64}});
        out.push_back({prefix + "attn.qkv.bias", {192}});
        out.push_back({prefix + "attn.proj.weight", {64, 64}});
        out.push_back({prefix + "attn.proj.bias", {64}});
        out.push_back({prefix + "norm1.weight", {64}});
        out.push_back({prefix + "norm1.bias", {64}});
    }
    out.push_back({"model.visual.merger.linear_fc1.weight", {128, 64}});
    out.push_back({"model.visual.merger.linear_fc1.bias", {128}});
    return out;
}

bool write_file(const std::string& path, const std::string& contents) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) return false;
    out.write(contents.data(), static_cast<std::streamsize>(contents.size()));
    return static_cast<bool>(out);
}

// Writes the fixture as two shards with a standard index, so the audit exercises
// multi-shard resolution the way the official 18-shard layout does.
bool write_fixture(const std::string& dir, const std::vector<TensorSpec>& text,
                   const std::vector<TensorSpec>& visual) {
    if (std::system(("mkdir -p '" + dir + "'").c_str()) != 0) return false;
    if (!write_file(dir + "/config.json", config_json())) return false;

    struct Shard {
        std::string file;
        std::vector<TensorSpec> specs;
    };
    const std::vector<Shard> shards = {
        {"model-00001-of-00002.safetensors", text},
        {"model-00002-of-00002.safetensors", visual},
    };

    std::ostringstream index;
    index << "{\"metadata\":{\"total_size\":";
    uint64_t total = 0;
    for (const Shard& shard : shards) {
        for (const TensorSpec& spec : shard.specs) total += numel(spec.shape) * 2;
    }
    index << total << "},\"weight_map\":{";

    bool first = true;
    for (const Shard& shard : shards) {
        std::ostringstream header;
        header << '{';
        uint64_t offset = 0;
        for (size_t i = 0; i < shard.specs.size(); ++i) {
            if (i) header << ',';
            const TensorSpec& spec = shard.specs[i];
            const uint64_t bytes = numel(spec.shape) * 2;
            header << '"' << spec.name << "\":{\"dtype\":\"BF16\",\"shape\":"
                   << shape_json(spec.shape) << ",\"data_offsets\":[" << offset
                   << ',' << (offset + bytes) << "]}";
            offset += bytes;
            if (!first) index << ',';
            first = false;
            index << '"' << spec.name << "\":\"" << shard.file << '"';
        }
        header << '}';

        std::ofstream out(dir + "/" + shard.file,
                          std::ios::binary | std::ios::trunc);
        if (!out) return false;
        const uint64_t header_len = header.str().size();
        out.write(reinterpret_cast<const char*>(&header_len), sizeof(header_len));
        out.write(header.str().data(),
                  static_cast<std::streamsize>(header.str().size()));
        // BF16 1.0 payload keeps the bytes meaningful for the conversion check.
        const std::vector<uint8_t> pattern = {0x80, 0x3f};
        for (uint64_t written = 0; written < offset; written += 2) {
            out.write(reinterpret_cast<const char*>(pattern.data()), 2);
        }
        if (!out) return false;
    }
    index << "}}";
    return write_file(dir + "/model.safetensors.index.json", index.str());
}

void check_dense_bf16_tp4(const std::string& dir, size_t text_count,
                          size_t visual_count) {
    const pocket::QwenConfig config = pocket::QwenConfig::from_hf_config(dir);
    const pocket::SafeTensorsIndex index(dir);
    require(index.shard_count() == 2, "fixture must span two shards");
    require(index.tensor_count() == text_count + visual_count,
            "index must hold both text and vision tensors");

    for (int rank = 0; rank < 4; ++rank) {
        const pocket::QwenWeightMap map(index, config, 4, rank);
        map.require_full_coverage();
        const pocket::QwenCoverage cover = map.coverage();
        require(cover.mapped_tensors == text_count,
                "text map must claim exactly the text tensors, got " +
                    std::to_string(cover.mapped_tensors));
        require(cover.visual_tensors == visual_count,
                "vision tensors must be reported as deliberately ignored");
        require(cover.unexpected_tensors == 0, "no tensor may be unclassified");

        const pocket::QwenLinearKindCounts kinds =
            map.checkpoint_linear_kind_counts();
        require(kinds.fp8_block128 == 0 && kinds.fp8_channel == 0 &&
                    kinds.nvfp4_group16 == 0,
                "an official BF16 checkpoint has no quantized linears");
        require(kinds.dense_f16 > 0, "dense linears must be detected");

        // TP4 shard contract. Rows split, hidden stays whole, and row-parallel
        // projections shard their K instead.
        require(map.embed_tokens().local_shape ==
                    std::vector<uint64_t>({128, 128}),
                "embedding rows must split by TP world");
        require(map.lm_head().logical_local_shape ==
                    std::vector<uint64_t>({128, 128}),
                "lm_head rows must split by TP world");
        const auto& mlp = map.layers().front().mlp;
        require(mlp.gate_proj.logical_local_shape ==
                    std::vector<uint64_t>({128, 128}),
                "MLP intermediate rows must split by TP world");
        require(mlp.down_proj.logical_local_shape ==
                    std::vector<uint64_t>({128, 128}),
                "row-parallel down_proj must shard K, not N");
        const auto& full = map.layers().at(1).full_attention;
        require(full.q_proj.logical_local_shape ==
                    std::vector<uint64_t>({128, 128}),
                "gated Q rows must split by TP world");
        require(full.k_proj.logical_local_shape ==
                    std::vector<uint64_t>({32, 128}),
                "one KV head per rank");
        require(full.o_proj.logical_local_shape ==
                    std::vector<uint64_t>({128, 64}),
                "row-parallel o_proj must shard its input");

        // BF16 storage, FP16 device residency: the CUDA/SM75 policy.
        require(map.embed_tokens().dtype == pocket::SafeDType::BF16 &&
                    map.embed_tokens().device_dtype == pocket::SafeDType::F16,
                "BF16 storage must materialize as FP16 on Turing");
        require(map.mtp().found, "native MTP must be mapped");
        require(map.mtp().fc.weight.rule == pocket::QwenShardRule::Replicated,
                "MTP fusion projection stays replicated");

        // Packed QKV and the depthwise conv share one fused row axis. Each rank
        // must take its own slice of all three segments, in order, rather than a
        // single contiguous range of the fused tensor.
        const auto& linear_attn = map.layers().front().linear_attention;
        const pocket::QwenTensorRef& qkv = linear_attn.in_proj_qkv.weight;
        require(qkv.rule == pocket::QwenShardRule::PackedQkvColumnParallel,
                "fused in_proj_qkv must use the packed column-parallel rule");
        require(qkv.segments.size() == 3, "packed QKV must expose three segments");
        const uint64_t key_shard = 8u * 32u / 4u;    // 64 rows of K per rank
        const uint64_t value_shard = 16u * 32u / 4u; // 128 rows of V per rank
        require(qkv.local_shape.at(0) == 2 * key_shard + value_shard,
                "packed QKV local rows must sum the three sharded segments");
        require(qkv.segments[0].first == rank * key_shard,
                "packed Q segment must start at the rank's offset");
        require(qkv.segments[1].first == 8u * 32u + rank * key_shard,
                "packed K segment must start after the whole Q segment");
        require(qkv.segments[2].first == 2u * 8u * 32u + rank * value_shard,
                "packed V segment must start after Q and K");
        require(qkv.segments[0].second == qkv.segments[1].second,
                "Q and K segments shard identically");

        const pocket::QwenTensorRef& conv = linear_attn.conv1d;
        require(conv.rule == pocket::QwenShardRule::PackedConvChannelParallel,
                "conv1d must shard by channel alongside packed QKV");
        require(conv.segments.size() == 3, "conv1d must follow the QKV segments");
        require(conv.local_shape.at(0) == qkv.local_shape.at(0),
                "conv1d channels must match the packed QKV rows on this rank");
        require(conv.local_shape.at(2) == 4, "conv kernel width stays whole");

        require(linear_attn.out_proj.rule == pocket::QwenShardRule::RowParallel,
                "linear attention out_proj is row parallel");
        require(linear_attn.out_proj.logical_local_shape.at(1) == value_shard,
                "out_proj must shard its input by value heads");

        require(cover.replicated_local_bytes > 0 && cover.sharded_local_bytes > 0,
                "coverage must separate replicated from sharded residency");
        require(cover.sharded_local_bytes < cover.checkpoint_text_bytes,
                "a single rank must hold less than the whole text checkpoint");

        const pocket::QwenHostTensor host =
            pocket::qwen_materialize_host_tensor(index, map.embed_tokens());
        require(host.storage_dtype == pocket::SafeDType::BF16 &&
                    host.device_dtype == pocket::SafeDType::F16,
                "materializer must convert BF16 to FP16");
        require(host.bytes.size() == 128u * 128u * sizeof(uint16_t),
                "materialized embedding shard size");
        require(host.bytes[0] == 0x00 && host.bytes[1] == 0x3c,
                "BF16 1.0 must materialize as FP16 1.0");
    }
}

// Rejects a checkpoint whose extra tensors are neither mapped nor vision, so an
// unrecognized variant fails loudly instead of loading a partial model.
void check_unexpected_tensor_is_rejected(const std::string& base_dir) {
    const std::string dir = base_dir + "_unexpected";
    std::vector<TensorSpec> visual = visual_specs();
    visual.push_back({"model.language_model.layers.0.linear_attn.mystery", {8}});
    require(write_fixture(dir, text_specs(), visual),
            "could not write unexpected-tensor fixture");

    const pocket::QwenConfig config = pocket::QwenConfig::from_hf_config(dir);
    const pocket::SafeTensorsIndex index(dir);
    const pocket::QwenWeightMap map(index, config, 4, 0);
    require(map.coverage().unexpected_tensors == 1,
            "an unrecognized tensor must be counted as unexpected");
    bool threw = false;
    try {
        map.require_full_coverage();
    } catch (const std::exception&) {
        threw = true;
    }
    require(threw, "strict coverage must reject an unrecognized tensor");
}

bool path_exists(const std::string& path) {
    struct stat st;
    return ::stat(path.c_str(), &st) == 0;
}

// Authoritative Qwen/Qwen3.8-27B facts, verified against the published index.
void check_real_checkpoint(const std::string& dir) {
    const pocket::QwenConfig config = pocket::QwenConfig::from_hf_config(dir);
    require(config.vocab_size == 248320, "official vocab_size");
    require(config.hidden_size == 5120, "official hidden_size");
    require(config.num_hidden_layers == 64, "official layer count");
    require(config.linear_attention_layers() == 48, "official linear layers");
    require(config.full_attention_layers() == 16, "official full layers");
    require(config.mlp.intermediate_size == 17408, "official intermediate_size");
    require(config.full_attention.head_dim == 256, "official head_dim");
    require(config.max_position_embeddings == 262144, "official max positions");
    require(config.mtp_num_hidden_layers == 1, "official MTP layer count");

    const pocket::SafeTensorsIndex index(dir);
    require(index.shard_count() == 18, "official shard count");
    require(index.tensor_count() == 1199, "official index tensor count");
    require(index.total_size() == 55562855904ull, "official total_size");

    for (const std::string& shard : index.shards()) {
        const pocket::SafeTensorsShard opened(index.shard_path(shard));
        require(!opened.tensors().empty(), "shard header must parse: " + shard);
    }

    uint64_t previous_text_bytes = 0;
    for (int rank = 0; rank < 4; ++rank) {
        const pocket::QwenWeightMap map(index, config, 4, rank);
        map.require_full_coverage();
        const pocket::QwenCoverage cover = map.coverage();
        require(cover.mapped_tensors == 866, "official text tensor count");
        require(cover.visual_tensors == 333, "official vision tensor count");
        require(cover.checkpoint_text_bytes == 54641395712ull,
                "official text weight bytes");
        if (rank != 0) {
            require(cover.checkpoint_text_bytes == previous_text_bytes,
                    "logical text bytes must not depend on rank");
        }
        previous_text_bytes = cover.checkpoint_text_bytes;

        require(map.embed_tokens().local_shape.at(0) == 62080,
                "official TP4 vocabulary shard");
        require(map.lm_head().logical_local_shape.at(0) == 62080,
                "official TP4 head shard");
        const auto& mlp = map.layers().front().mlp;
        require(mlp.gate_proj.logical_local_shape.at(0) == 4352,
                "official TP4 MLP intermediate shard");
        require(mlp.down_proj.logical_local_shape.at(1) == 4352,
                "official TP4 row-parallel MLP shard");
        require(map.checkpoint_linear_kind_counts().fp8_block128 == 0 &&
                    map.checkpoint_linear_kind_counts().nvfp4_group16 == 0,
                "official checkpoint must map as dense BF16 only");
    }
    std::cout << "real checkpoint audited: " << dir << '\n';
}

}  // namespace

int main() {
    try {
        const std::string dir = fixture_dir();
        const std::vector<TensorSpec> text = text_specs();
        const std::vector<TensorSpec> visual = visual_specs();
        require(write_fixture(dir, text, visual),
                "could not write BF16 checkpoint fixture");
        check_dense_bf16_tp4(dir, text.size(), visual.size());
        check_unexpected_tensor_is_rejected(dir);

        const char* real = std::getenv("QWEN38_CKPT");
        if (real != nullptr && path_exists(std::string(real) + "/config.json")) {
            check_real_checkpoint(real);
        } else {
            std::cout << "[SKIP] real checkpoint audit (set QWEN38_CKPT)\n";
        }

        std::cout << "[PASS] test_qwen_bf16_checkpoint text=" << text.size()
                  << " visual=" << visual.size() << " tp=4\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cout << "[FAIL] test_qwen_bf16_checkpoint " << ex.what() << '\n';
        return 1;
    }
}
