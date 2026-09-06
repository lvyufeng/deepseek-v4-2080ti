// Phase 3 step: GGUF dense q_a chain (embed -> attn_norm RMSNorm -> wq_a).
// Validates the F32 norm gamma conversion path and the first Q8_0 attention
// projection against real GGUF data, paving the way for the rest of the
// attention forward.

#include "deepseek_v4_engine.hpp"

#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: test_gguf_attn_norm_wqa <model.gguf> [token]\n";
        return 2;
    }
    const int token = argc >= 3 ? std::atoi(argv[2]) : 1234;
    try {
        auto r = pocket::run_gguf_attn_norm_wq_a_smoke(argv[1], token);
        std::printf("token=%d dim=%d q_a_dim=%d\n", token, r.dim, r.q_a_dim);
        std::printf("embed_rms  = %.4f\n", r.embed_rms);
        std::printf("normed_rms = %.4f  (RMSNorm should bring this to ~|gamma|_rms)\n",
                    r.normed_rms);
        std::printf("q_a_rms    = %.4f\n", r.q_a_rms);
        std::printf("q_a[0..3]  = %.4f %.4f %.4f %.4f\n",
                    r.q_a_first[0], r.q_a_first[1], r.q_a_first[2], r.q_a_first[3]);
        if (!(r.embed_rms > 1e-6f && r.embed_rms < 100.0f)) {
            std::cerr << "[FAIL] embed_rms out of plausible range\n";
            return 1;
        }
        if (!(r.normed_rms > 1e-3f && r.normed_rms < 100.0f)) {
            std::cerr << "[FAIL] normed_rms out of plausible range\n";
            return 1;
        }
        if (!(r.q_a_rms > 1e-3f && r.q_a_rms < 1000.0f)) {
            std::cerr << "[FAIL] q_a_rms out of plausible range\n";
            return 1;
        }
        std::cout << "[PASS] gguf attn_norm + wq_a smoke\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "[FAIL] " << ex.what() << "\n";
        return 1;
    }
}
