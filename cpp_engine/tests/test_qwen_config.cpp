// Qwen3.8 config parsing test. Writes a config.json fixture holding the
// authoritative Qwen/Qwen3.8-27B-FP8 text_config values and asserts the parsed
// QwenConfig matches, so a checkpoint is not required to run this.

#include "qwen_config.hpp"

#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

namespace {

int failures = 0;

void check(bool condition, const std::string& what) {
    if (!condition) {
        std::cout << "[FAIL] " << what << "\n";
        ++failures;
    }
}

template <typename A, typename B>
void check_eq(const A& actual, const B& expected, const std::string& what) {
    if (!(actual == static_cast<A>(expected))) {
        std::cout << "[FAIL] " << what << " actual=" << actual << " expected=" << expected << "\n";
        ++failures;
        return;
    }
}

std::string layer_types_json(int layers, int full_interval) {
    std::ostringstream out;
    out << '[';
    for (int i = 0; i < layers; ++i) {
        if (i) out << ',';
        const bool full = ((i + 1) % full_interval) == 0;
        out << (full ? "\"full_attention\"" : "\"linear_attention\"");
    }
    out << ']';
    return out.str();
}

std::string qwen38_config_json() {
    std::ostringstream out;
    out << "{\n"
        << "  \"architectures\": [\"Qwen3_5ForConditionalGeneration\"],\n"
        << "  \"model_type\": \"qwen3_5\",\n"
        // The official checkpoint is multimodal at the root: a vision tower sits
        // beside text_config and language_model_only is false. The text runtime
        // has to parse straight through these instead of tripping over them.
        << "  \"language_model_only\": false,\n"
        << "  \"tie_word_embeddings\": false,\n"
        << "  \"image_token_id\": 248056,\n"
        << "  \"bos_token_id\": 248044,\n"
        << "  \"eos_token_id\": 248044,\n"
        << "  \"dtype\": \"bfloat16\",\n"
        << "  \"vision_config\": {\n"
        << "    \"model_type\": \"qwen3_5_vit\",\n"
        << "    \"depth\": 27,\n"
        << "    \"hidden_size\": 1152,\n"
        << "    \"num_heads\": 16,\n"
        << "    \"out_hidden_size\": 5120,\n"
        << "    \"patch_size\": 16,\n"
        << "    \"temporal_patch_size\": 2\n"
        << "  },\n"
        << "  \"text_config\": {\n"
        << "    \"model_type\": \"qwen3_5_text\",\n"
        << "    \"vocab_size\": 248320,\n"
        << "    \"hidden_size\": 5120,\n"
        << "    \"intermediate_size\": 17408,\n"
        << "    \"num_hidden_layers\": 64,\n"
        << "    \"mtp_num_hidden_layers\": 1,\n"
        << "    \"mtp_use_dedicated_embeddings\": false,\n"
        << "    \"num_attention_heads\": 24,\n"
        << "    \"num_key_value_heads\": 4,\n"
        << "    \"head_dim\": 256,\n"
        << "    \"attn_output_gate\": true,\n"
        << "    \"linear_num_key_heads\": 16,\n"
        << "    \"linear_num_value_heads\": 48,\n"
        << "    \"linear_key_head_dim\": 128,\n"
        << "    \"linear_value_head_dim\": 128,\n"
        << "    \"linear_conv_kernel_dim\": 4,\n"
        << "    \"max_position_embeddings\": 262144,\n"
        << "    \"rms_norm_eps\": 1e-06,\n"
        << "    \"partial_rotary_factor\": 0.25,\n"
        << "    \"rope_parameters\": {\n"
        << "      \"rope_type\": \"default\",\n"
        << "      \"rope_theta\": 10000000,\n"
        << "      \"partial_rotary_factor\": 0.25,\n"
        << "      \"mrope_interleaved\": true,\n"
        << "      \"mrope_section\": [11, 11, 10]\n"
        << "    },\n"
        << "    \"layer_types\": " << layer_types_json(64, 4) << "\n"
        << "  }\n"
        << "}\n";
    return out.str();
}

bool write_file(const std::string& path, const std::string& contents) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) return false;
    out << contents;
    return static_cast<bool>(out);
}

std::string temp_dir() {
    const char* base = std::getenv("CLAUDE_JOB_DIR");
    std::string root = base != nullptr ? std::string(base) + "/tmp" : std::string("/tmp");
    return root + "/qwen_config_fixture";
}

}  // namespace

int main() {
    const std::string dir = temp_dir();
    const std::string mkdir_cmd = "mkdir -p '" + dir + "'";
    if (std::system(mkdir_cmd.c_str()) != 0) {
        std::cout << "[FAIL] could not create fixture dir " << dir << "\n";
        return 1;
    }
    if (!write_file(dir + "/config.json", qwen38_config_json())) {
        std::cout << "[FAIL] could not write fixture config.json\n";
        return 1;
    }

    check(pocket::is_qwen3_5_checkpoint(dir), "is_qwen3_5_checkpoint detects qwen3_5 root model_type");
    check(pocket::is_qwen3_5_checkpoint(dir),
          "a multimodal root with language_model_only=false still dispatches to Qwen");

    const pocket::QwenConfig cfg = pocket::QwenConfig::from_hf_config(dir);
    check(cfg.is_qwen3_5(), "config reports qwen3_5 architecture");
    check_eq(cfg.hidden_size, 5120u, "hidden_size");
    check_eq(cfg.vocab_size, 248320u, "vocab_size");
    check_eq(cfg.num_hidden_layers, 64u, "num_hidden_layers");
    check_eq(cfg.mtp_num_hidden_layers, 1u, "mtp_num_hidden_layers");
    check(!cfg.mtp_use_dedicated_embeddings,
          "MTP shares the main embedding table");
    check_eq(cfg.mlp.intermediate_size, 17408u, "dense MLP intermediate_size");
    check_eq(cfg.max_position_embeddings, 262144u, "max_position_embeddings");
    check_eq(cfg.full_attention.num_heads, 24u, "num_attention_heads");
    check_eq(cfg.full_attention.num_key_value_heads, 4u, "num_key_value_heads");
    check(cfg.full_attention.num_heads % 2 == 0 &&
              cfg.full_attention.num_key_value_heads % 2 == 0,
          "real attention heads are TP2-compatible");
    check_eq(cfg.full_attention.head_dim, 256u, "head_dim");
    check(cfg.full_attention.output_gate, "attn_output_gate is true");
    check_eq(cfg.full_attention.attention_dim(), 6144u, "attention_dim = 24*256");
    check_eq(cfg.full_attention.q_dim(), 12288u, "q_dim doubles for the output gate");
    check_eq(cfg.full_attention.kv_dim(), 1024u, "kv_dim = 4*256");
    check_eq(cfg.linear_attention.key_heads, 16u, "linear_num_key_heads");
    check_eq(cfg.linear_attention.value_heads, 48u, "linear_num_value_heads");
    check_eq(cfg.linear_attention.key_head_dim, 128u, "linear_key_head_dim");
    check_eq(cfg.linear_attention.value_head_dim, 128u, "linear_value_head_dim");
    check_eq(cfg.linear_attention.conv_kernel_dim, 4u, "linear_conv_kernel_dim");
    check_eq(cfg.linear_attention.qkv_dim(), 10240u, "qkv_dim = 2*2048 + 6144");
    check_eq(cfg.linear_attention.value_state_dim(), 6144u, "value_state_dim = 48*128");
    check_eq(cfg.partial_rotary_dim(), 64u, "partial rotary dim = 256 * 0.25");
    check_eq(cfg.fp8_block_size, 128u, "fp8 block size");
    check(cfg.rope_theta > 9.9e6 && cfg.rope_theta < 1.01e7, "rope_theta comes from rope_parameters");
    check(cfg.rms_norm_eps > 0.0 && cfg.rms_norm_eps < 1.0e-5, "rms_norm_eps");
    check_eq(cfg.full_attention_layers(), 16u, "one full-attention layer per four");
    check_eq(cfg.linear_attention_layers(), 48u, "three linear-attention layers per four");
    check_eq(cfg.layer_type_name(0), std::string("linear_attention"), "layer 0 is linear attention");
    check_eq(cfg.layer_type_name(3), std::string("full_attention"), "layer 3 is full attention");
    check_eq(cfg.layer_type_name(63), std::string("full_attention"), "last layer is full attention");

    // Parsing is topology-agnostic. A valid Qwen config whose KV-head count is
    // not divisible by four must still parse; the runtime validates the actual
    // requested TP world later.
    std::string tp2_only = qwen38_config_json();
    const std::string kv_heads = "\"num_key_value_heads\": 4";
    const size_t kv_at = tp2_only.find(kv_heads);
    check(kv_at != std::string::npos, "fixture contains num_key_value_heads");
    if (kv_at != std::string::npos) {
        tp2_only.replace(kv_at, kv_heads.size(), "\"num_key_value_heads\": 2");
        const std::string tp2_dir = dir + "_tp2_only";
        const std::string tp2_mkdir = "mkdir -p '" + tp2_dir + "'";
        if (std::system(tp2_mkdir.c_str()) == 0 &&
            write_file(tp2_dir + "/config.json", tp2_only)) {
            bool parsed = true;
            try {
                const pocket::QwenConfig tp2_cfg =
                    pocket::QwenConfig::from_hf_config(tp2_dir);
                parsed = tp2_cfg.full_attention.num_key_value_heads == 2;
            } catch (const std::exception&) {
                parsed = false;
            }
            check(parsed, "config parser does not impose TP4 divisibility");
        }
    }

    // A dense text config must not be mistaken for MoE: dropping
    // intermediate_size has to fail loudly rather than default to zero.
    std::string no_inter = qwen38_config_json();
    const std::string needle = "\"intermediate_size\": 17408,\n";
    const size_t at = no_inter.find(needle);
    check(at != std::string::npos, "fixture contains intermediate_size");
    if (at != std::string::npos) {
        no_inter.erase(at, needle.size());
        const std::string bad_dir = dir + "_no_inter";
        const std::string bad_mkdir = "mkdir -p '" + bad_dir + "'";
        if (std::system(bad_mkdir.c_str()) == 0 && write_file(bad_dir + "/config.json", no_inter)) {
            bool threw = false;
            try {
                (void)pocket::QwenConfig::from_hf_config(bad_dir);
            } catch (const std::exception&) {
                threw = true;
            }
            check(threw, "missing intermediate_size throws instead of defaulting");
        }
    }

    if (failures != 0) {
        std::cout << "test_qwen_config failures=" << failures << "\n";
        return 1;
    }
    std::cout << "[PASS] test_qwen_config layers=" << cfg.num_hidden_layers
              << " full=" << cfg.full_attention_layers()
              << " linear=" << cfg.linear_attention_layers()
              << " rotary_dim=" << cfg.partial_rotary_dim() << "\n";
    return 0;
}
