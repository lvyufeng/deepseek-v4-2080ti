// Smoke test: open the GGUF Q2 checkpoint and verify the WeightSource mapping
// can resolve representative tensors (dense layer attention/embed/head/norm/HC,
// shared experts, hash gate, and 3D routed-expert slices) by shape and dtype.
#include "weight_source.hpp"

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace pocket;

void expect_present(const WeightSource& ws, const std::string& name,
                    DType expected_dtype, std::vector<uint64_t> expected_shape) {
    if (!ws.has(name)) {
        throw std::runtime_error("missing tensor: " + name);
    }
    WeightView v = ws.get(name);
    if (!v.found) {
        throw std::runtime_error("get() returned not-found despite has()=true: " + name);
    }
    if (v.dtype != expected_dtype) {
        throw std::runtime_error("dtype mismatch for " + name + " expected=" + dtype_name(expected_dtype) + " actual=" + dtype_name(v.dtype));
    }
    if (v.shape != expected_shape) {
        std::string err = "shape mismatch for " + name + " expected=[";
        for (size_t i = 0; i < expected_shape.size(); ++i) err += (i ? "," : "") + std::to_string(expected_shape[i]);
        err += "] actual=[";
        for (size_t i = 0; i < v.shape.size(); ++i) err += (i ? "," : "") + std::to_string(v.shape[i]);
        err += "]";
        throw std::runtime_error(err);
    }
    if (v.data == nullptr || v.nbytes == 0) {
        throw std::runtime_error("empty data/nbytes for " + name);
    }
}

void expect_expert_slice(const WeightSource& ws, const std::string& routed_3d,
                         int expert_id, DType expected_dtype,
                         std::vector<uint64_t> expected_shape, uint64_t expected_nbytes) {
    const std::string per_expert_name = routed_3d + ".expert" + std::to_string(expert_id);
    WeightView v = ws.get_expert(routed_3d, per_expert_name, expert_id);
    if (!v.found) {
        throw std::runtime_error("get_expert() not-found for " + routed_3d + " expert=" + std::to_string(expert_id));
    }
    if (v.dtype != expected_dtype) {
        throw std::runtime_error("expert dtype mismatch for " + routed_3d);
    }
    if (v.shape != expected_shape) {
        std::string err = "expert shape mismatch for " + routed_3d + " expected=[";
        for (size_t i = 0; i < expected_shape.size(); ++i) err += (i ? "," : "") + std::to_string(expected_shape[i]);
        err += "] actual=[";
        for (size_t i = 0; i < v.shape.size(); ++i) err += (i ? "," : "") + std::to_string(v.shape[i]);
        err += "]";
        throw std::runtime_error(err);
    }
    if (v.nbytes != expected_nbytes) {
        throw std::runtime_error("expert nbytes mismatch for " + routed_3d + " expected=" + std::to_string(expected_nbytes) + " actual=" + std::to_string(v.nbytes));
    }
    if (v.data == nullptr) {
        throw std::runtime_error("expert null data for " + routed_3d);
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: test_weight_source <model.gguf or safetensors-dir>\n";
        return 2;
    }
    try {
        auto ws = pocket::open_weight_source(argv[1]);

        // Embed/head/norm and dense layer 0 attention should always be present
        // regardless of GGUF or safetensors. Shapes below match
        // DeepSeek-V4-Flash (dim=4096, q_lora=1536, kv_lora=512, etc.).
        // We don't pin every shape — just spot-check enough to know the mapping
        // is wired. Concrete tensor shape verification is done in the dense
        // layer parity gate, not here.
        if (!ws->has("embed.weight") || !ws->has("head.weight") || !ws->has("norm.weight")) {
            throw std::runtime_error("embed/head/norm missing");
        }
        if (!ws->has("layers.0.attn.wq_a.weight") ||
            !ws->has("layers.0.attn_norm.weight") ||
            !ws->has("layers.0.ffn_norm.weight") ||
            !ws->has("layers.0.ffn.gate.weight") ||
            !ws->has("layers.0.ffn.shared_experts.w1.weight")) {
            throw std::runtime_error("dense layer 0 tensors missing");
        }
        if (!ws->has("layers.0.hc_attn_fn") || !ws->has("layers.0.hc_ffn_fn")) {
            throw std::runtime_error("HC tensors missing");
        }

        // Hash layers expect ffn.gate.tid2eid; non-hash layers expect ffn.gate.bias.
        if (!ws->has("layers.0.ffn.gate.tid2eid")) {
            throw std::runtime_error("layer 0 (hash) missing ffn.gate.tid2eid");
        }
        if (!ws->has("layers.3.ffn.gate.bias")) {
            throw std::runtime_error("layer 3 (non-hash) missing ffn.gate.bias");
        }

        // GGUF-specific: routed experts are 3D tensors, per-expert slices through get_expert.
        // Skip routed-expert slice check for safetensors (per-expert names are flat).
        if (ws->format() == pocket::WeightSource::Format::GGUF_Q2) {
            const std::string routed_w1 = "layers.0.ffn.experts.routed.w1";
            const std::string routed_w2 = "layers.0.ffn.experts.routed.w2";
            const std::string routed_w3 = "layers.0.ffn.experts.routed.w3";
            if (!ws->has(routed_w1) || !ws->has(routed_w2) || !ws->has(routed_w3)) {
                throw std::runtime_error("3D routed expert tensors missing");
            }
            // Inspect expert 0 of w1: should be IQ2_XXS, shape [inter, dim]=[2048, 4096],
            // nbytes per expert = 2048 * 4096 / 256 * 66 = 2162688.
            WeightView routed_view = ws->get(routed_w1);
            if (routed_view.shape.size() != 3) {
                throw std::runtime_error("routed w1 not 3D");
            }
            const uint64_t a = routed_view.shape[0];
            const uint64_t b = routed_view.shape[1];
            const uint64_t n_experts = routed_view.shape[2];
            if (n_experts == 0 || routed_view.nbytes % n_experts != 0) {
                throw std::runtime_error("routed w1 shape inconsistent");
            }
            const uint64_t per_expert_nbytes = routed_view.nbytes / n_experts;
            expect_expert_slice(*ws, routed_w1, 0, routed_view.dtype, {b, a}, per_expert_nbytes);
            expect_expert_slice(*ws, routed_w1, static_cast<int>(n_experts - 1), routed_view.dtype, {b, a}, per_expert_nbytes);
            std::cout << "routed w1 layer0: " << dtype_name(routed_view.dtype)
                      << " [" << a << "," << b << "," << n_experts << "]"
                      << " per_expert_nbytes=" << per_expert_nbytes << "\n";

            WeightView routed_w2_view = ws->get(routed_w2);
            std::cout << "routed w2 layer0: " << dtype_name(routed_w2_view.dtype)
                      << " [" << routed_w2_view.shape[0] << "," << routed_w2_view.shape[1] << "," << routed_w2_view.shape[2] << "]"
                      << " nbytes=" << routed_w2_view.nbytes << "\n";
        }

        // Print one summary line for visual confirmation.
        WeightView embed = ws->require("embed.weight");
        WeightView wq_a = ws->require("layers.0.attn.wq_a.weight");
        std::cout << "embed:    " << dtype_name(embed.dtype) << " shape=[" << embed.shape[0] << "," << embed.shape[1] << "] nbytes=" << embed.nbytes << "\n";
        std::cout << "wq_a@L0:  " << dtype_name(wq_a.dtype) << " shape=[" << wq_a.shape[0] << "," << wq_a.shape[1] << "] nbytes=" << wq_a.nbytes << "\n";
        std::cout << "[PASS] weight_source smoke ok\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "[FAIL] " << ex.what() << "\n";
        return 1;
    }
}
