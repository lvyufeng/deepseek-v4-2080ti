// Phase 3 step: GGUF shared-expert FFN smoke for layer 0.
// Embed -> ffn_norm -> shared w1 / w3 Q8_0 -> silu_mul -> shared w2 Q8_0.

#include "deepseek_v4_engine.hpp"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: test_gguf_shared_expert <model.gguf> [token]\n";
        return 2;
    }
    const int token = argc >= 3 ? std::atoi(argv[2]) : 1234;
    try {
        auto r = pocket::run_gguf_shared_expert_smoke(argv[1], token);
        std::printf("token=%d dim=%d moe_inter_dim=%d\n",
                    token, r.dim, r.moe_inter_dim);
        std::printf("ffn_normed_rms = %.4f\n", r.ffn_normed_rms);
        std::printf("gate_rms       = %.4f\n", r.gate_rms);
        std::printf("up_rms         = %.4f\n", r.up_rms);
        std::printf("hidden_rms     = %.4f  (post silu_mul)\n", r.hidden_rms);
        std::printf("shared_out_rms = %.4f\n", r.shared_out_rms);
        std::printf("shared_out[0..3] = %.4f %.4f %.4f %.4f\n",
                    r.shared_out_first[0], r.shared_out_first[1],
                    r.shared_out_first[2], r.shared_out_first[3]);

        if (!(r.ffn_normed_rms > 1e-4f && r.ffn_normed_rms < 100.0f)) {
            std::cerr << "[FAIL] ffn_normed_rms\n"; return 1;
        }
        if (!(r.gate_rms > 1e-4f && r.gate_rms < 1000.0f)) {
            std::cerr << "[FAIL] gate_rms\n"; return 1;
        }
        if (!(r.up_rms > 1e-4f && r.up_rms < 1000.0f)) {
            std::cerr << "[FAIL] up_rms\n"; return 1;
        }
        if (!(r.hidden_rms > 1e-6f && r.hidden_rms < 1000.0f)) {
            std::cerr << "[FAIL] hidden_rms\n"; return 1;
        }
        if (!(r.shared_out_rms > 1e-6f && r.shared_out_rms < 1000.0f)) {
            std::cerr << "[FAIL] shared_out_rms\n"; return 1;
        }
        std::cout << "[PASS] gguf shared expert smoke\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "[FAIL] " << ex.what() << "\n";
        return 1;
    }
}
