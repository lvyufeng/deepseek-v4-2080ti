// Phase 3 step: full single-token dense attention chain for layer 0.
// Embed -> attn_norm -> Q & KV projections -> sparse attention with attn_sink
// -> inverse RoPE -> grouped wo_a Q8_0 -> wo_b Q8_0 -> attn_out[dim].

#include "deepseek_v4_engine.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: test_gguf_attn_full <model.gguf> [token] [position]\n";
        return 2;
    }
    const int token = argc >= 3 ? std::atoi(argv[2]) : 1234;
    const int position = argc >= 4 ? std::atoi(argv[3]) : 0;
    try {
        auto r = pocket::run_gguf_attn_full_smoke(argv[1], token, position);
        std::printf("token=%d position=%d dim=%d heads=%d head_dim=%d kv_dim=%d rope_dim=%d\n",
                    token, position, r.dim, r.heads, r.head_dim, r.kv_dim, r.rope_dim);
        std::printf("o_groups=%d o_lora_rank=%d attn_mid=%d\n",
                    r.o_groups, r.o_lora_rank, r.attn_mid);
        std::printf("q_rms                  = %.4f\n", r.q_rms);
        std::printf("kv_rms                 = %.4f\n", r.kv_rms);
        std::printf("attn_value_rms         = %.4f\n", r.attn_value_rms);
        std::printf("attn_value_post_inv_rms= %.4f  (inverse RoPE preserves L2)\n",
                    r.attn_value_post_inv_rms);
        std::printf("attn_mid_rms           = %.4f\n", r.attn_mid_rms);
        std::printf("attn_out_rms           = %.4f\n", r.attn_out_rms);
        std::printf("attn_out[0..3]         = %.4f %.4f %.4f %.4f\n",
                    r.attn_out_first[0], r.attn_out_first[1],
                    r.attn_out_first[2], r.attn_out_first[3]);

        if (!(r.q_rms > 1e-4f && r.q_rms < 1000.0f)) {
            std::cerr << "[FAIL] q_rms\n"; return 1;
        }
        if (!(r.kv_rms > 1e-4f && r.kv_rms < 1000.0f)) {
            std::cerr << "[FAIL] kv_rms\n"; return 1;
        }
        if (!(r.attn_value_rms > 1e-6f && r.attn_value_rms < 1000.0f)) {
            std::cerr << "[FAIL] attn_value_rms\n"; return 1;
        }
        // Inverse RoPE is an orthogonal rotation per pair; preserves total L2.
        const float ratio =
            r.attn_value_post_inv_rms / std::max(r.attn_value_rms, 1e-12f);
        if (!(ratio > 0.99f && ratio < 1.01f)) {
            std::cerr << "[FAIL] inverse RoPE changed RMS: ratio=" << ratio << "\n";
            return 1;
        }
        if (!(r.attn_mid_rms > 1e-6f && r.attn_mid_rms < 1000.0f)) {
            std::cerr << "[FAIL] attn_mid_rms\n"; return 1;
        }
        if (!(r.attn_out_rms > 1e-6f && r.attn_out_rms < 1000.0f)) {
            std::cerr << "[FAIL] attn_out_rms\n"; return 1;
        }
        std::cout << "[PASS] gguf attn full smoke\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "[FAIL] " << ex.what() << "\n";
        return 1;
    }
}
