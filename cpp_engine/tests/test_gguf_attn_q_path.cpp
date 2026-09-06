// Phase 3 step: full GGUF attention Q-path smoke for layer 0.
// Embed -> attn_norm -> wq_a -> q_norm -> wq_b -> head_rmsnorm_rope.

#include "deepseek_v4_engine.hpp"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: test_gguf_attn_q_path <model.gguf> [token] [position]\n";
        return 2;
    }
    const int token = argc >= 3 ? std::atoi(argv[2]) : 1234;
    const int position = argc >= 4 ? std::atoi(argv[3]) : 0;
    try {
        auto r = pocket::run_gguf_attn_q_path_smoke(argv[1], token, position);
        std::printf("token=%d position=%d dim=%d q_a_dim=%d heads=%d head_dim=%d rope_dim=%d\n",
                    token, position, r.dim, r.q_a_dim, r.heads, r.head_dim, r.rope_dim);
        std::printf("q_normed_rms    = %.4f\n", r.q_normed_rms);
        std::printf("q_pre_rope_rms  = %.4f\n", r.q_pre_rope_rms);
        std::printf("q_post_rope_rms = %.4f  (per-head norm makes head-RMS == 1)\n",
                    r.q_post_rope_rms);
        std::printf("q[0..3]         = %.4f %.4f %.4f %.4f\n",
                    r.q_first[0], r.q_first[1], r.q_first[2], r.q_first[3]);
        if (!(r.q_normed_rms > 1e-3f && r.q_normed_rms < 100.0f)) {
            std::cerr << "[FAIL] q_normed_rms\n"; return 1;
        }
        if (!(r.q_pre_rope_rms > 1e-3f && r.q_pre_rope_rms < 1000.0f)) {
            std::cerr << "[FAIL] q_pre_rope_rms\n"; return 1;
        }
        // After per-head RMSNorm each head has RMS==1, so the full-tensor RMS
        // is also ~1 (RoPE is a rotation that preserves L2 norm).
        if (!(r.q_post_rope_rms > 0.5f && r.q_post_rope_rms < 2.0f)) {
            std::cerr << "[FAIL] q_post_rope_rms expected ~1, got "
                      << r.q_post_rope_rms << "\n";
            return 1;
        }
        std::cout << "[PASS] gguf attn q path smoke\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "[FAIL] " << ex.what() << "\n";
        return 1;
    }
}
