// Phase 3 step: GGUF single routed Q2 expert smoke for layer 0.
// Embed -> ffn_norm -> q8_1 quantize -> IQ2_XXS w1/w3 -> SwiGLU+q8_1 ->
// Q2_K w2. Loads only the single expert's bytes from the routed 3D tensor.

#include "deepseek_v4_engine.hpp"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: test_gguf_routed_expert <model.gguf> [token] [expert_id]\n";
        return 2;
    }
    const int token = argc >= 3 ? std::atoi(argv[2]) : 1234;
    const int expert_id = argc >= 4 ? std::atoi(argv[3]) : 0;
    try {
        auto r = pocket::run_gguf_routed_expert_smoke(argv[1], token, expert_id);
        std::printf("token=%d expert_id=%d dim=%d moe_inter_dim=%d\n",
                    token, r.expert_id, r.dim, r.moe_inter_dim);
        std::printf("ffn_normed_rms = %.4f\n", r.ffn_normed_rms);
        std::printf("gate_rms       = %.4f\n", r.gate_rms);
        std::printf("up_rms         = %.4f\n", r.up_rms);
        std::printf("hidden_rms     = %.4f  (post swiglu+route_weight, Q8_1 view)\n",
                    r.hidden_rms);
        std::printf("route_out_rms  = %.4f\n", r.route_out_rms);
        std::printf("route_out[0..3]= %.4f %.4f %.4f %.4f\n",
                    r.route_out_first[0], r.route_out_first[1],
                    r.route_out_first[2], r.route_out_first[3]);

        if (!(r.gate_rms > 1e-4f && r.gate_rms < 1000.0f)) {
            std::cerr << "[FAIL] gate_rms\n"; return 1;
        }
        if (!(r.up_rms > 1e-4f && r.up_rms < 1000.0f)) {
            std::cerr << "[FAIL] up_rms\n"; return 1;
        }
        if (!(r.hidden_rms > 1e-6f && r.hidden_rms < 1000.0f)) {
            std::cerr << "[FAIL] hidden_rms\n"; return 1;
        }
        if (!(r.route_out_rms > 1e-6f && r.route_out_rms < 1000.0f)) {
            std::cerr << "[FAIL] route_out_rms\n"; return 1;
        }
        std::cout << "[PASS] gguf routed expert smoke\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "[FAIL] " << ex.what() << "\n";
        return 1;
    }
}
