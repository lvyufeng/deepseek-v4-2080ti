// Phase 3 milestone: full 43-layer GGUF forward + final norm + Q8_0 head +
// argmax. Validates that the per-layer helper composes correctly across all
// layers, both hash (0..n_hash-1) and non-hash (n_hash..n_layers-1) gate
// paths, and ends with a sane top token + logit.

#include "deepseek_v4_engine.hpp"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: test_gguf_full_forward <model.gguf> [token] [position]\n";
        return 2;
    }
    const int token = argc >= 3 ? std::atoi(argv[2]) : 1234;
    const int position = argc >= 4 ? std::atoi(argv[3]) : 0;
    try {
        auto r = pocket::run_gguf_full_forward_smoke(argv[1], token, position);
        std::printf("token=%d position=%d n_layers=%d dim=%d vocab=%d\n",
                    token, position, r.n_layers, r.dim, r.vocab);
        std::printf("final_x_rms      = %.4f\n", r.final_x_rms);
        std::printf("final_normed_rms = %.4f\n", r.final_normed_rms);
        std::printf("logits_rms       = %.4f\n", r.logits_rms);
        std::printf("logits[0..3]     = %.4f %.4f %.4f %.4f\n",
                    r.logits_first[0], r.logits_first[1],
                    r.logits_first[2], r.logits_first[3]);
        std::printf("top_token=%d top_logit=%.4f checksum=%.4f\n",
                    r.top_token, r.top_logit, r.checksum);

        if (!(r.final_x_rms > 0.01f && r.final_x_rms < 1000.0f)) {
            std::cerr << "[FAIL] final_x_rms out of range\n"; return 1;
        }
        if (!(r.logits_rms > 0.01f && r.logits_rms < 1000.0f)) {
            std::cerr << "[FAIL] logits_rms out of range\n"; return 1;
        }
        if (r.top_token < 0 || r.top_token >= r.vocab) {
            std::cerr << "[FAIL] top_token out of range\n"; return 1;
        }
        if (!(r.top_logit > -1e6f && r.top_logit < 1e6f)) {
            std::cerr << "[FAIL] top_logit out of range\n"; return 1;
        }
        std::cout << "[PASS] gguf full 43-layer forward smoke\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "[FAIL] " << ex.what() << "\n";
        return 1;
    }
}
