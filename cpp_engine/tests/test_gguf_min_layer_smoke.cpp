// Phase 3 entry-point smoke test for the GGUF Q2 forward path.
//
// Verifies that run_gguf_min_layer_smoke() can construct a GgufForwardContext
// from a real .gguf checkpoint and surface the model dimensions we'll need
// for subsequent dense + MoE wiring (n_layers, dim, n_experts, vocab, etc.).
//
// This is the parallel of test_min_layer_smoke for the GGUF path; future
// commits will add real forward verification (embed → attn → dense layer
// output parity, then full layer forward).

#include "deepseek_v4_engine.hpp"

#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: test_gguf_min_layer_smoke <model.gguf>\n";
        return 2;
    }
    try {
        auto r = pocket::run_gguf_min_layer_smoke(argv[1]);
        std::printf("[PASS] gguf_min_layer_smoke "
                    "layers=%d hash_layers=%d dim=%d moe_inter=%d "
                    "experts=%d topk=%d vocab=%d\n",
                    r.n_layers, r.n_hash_layers, r.dim, r.moe_inter_dim,
                    r.n_routed_experts, r.n_activated_experts, r.vocab);
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "[FAIL] " << ex.what() << "\n";
        return 1;
    }
}
