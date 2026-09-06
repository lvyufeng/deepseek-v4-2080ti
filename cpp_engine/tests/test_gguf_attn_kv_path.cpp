// Phase 3 step: GGUF attention KV-path smoke for layer 0.
// Embed -> attn_norm -> wkv -> kv_norm -> head_rmsnorm_rope (heads=1).
//
// kv_dim = 512 (= kv_lora_rank 448 + rope_dim 64). RoPE is a rotation so the
// post-rope L2 norm equals the post-rmsnorm L2 norm. Since RMSNorm with gamma
// gives RMS ~ |gamma|_avg, we check the rope step preserves the kv RMS.

#include "deepseek_v4_engine.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: test_gguf_attn_kv_path <model.gguf> [token] [position]\n";
        return 2;
    }
    const int token = argc >= 3 ? std::atoi(argv[2]) : 1234;
    const int position = argc >= 4 ? std::atoi(argv[3]) : 0;
    try {
        auto r = pocket::run_gguf_attn_kv_path_smoke(argv[1], token, position);
        std::printf("token=%d position=%d dim=%d kv_dim=%d rope_dim=%d\n",
                    token, position, r.dim, r.kv_dim, r.rope_dim);
        std::printf("kv_a_rms         = %.4f\n", r.kv_a_rms);
        std::printf("kv_norm_rms      = %.4f\n", r.kv_norm_rms);
        std::printf("kv_post_rope_rms = %.4f  (RoPE preserves L2 norm)\n",
                    r.kv_post_rope_rms);
        std::printf("kv[0..3]         = %.4f %.4f %.4f %.4f\n",
                    r.kv_first[0], r.kv_first[1], r.kv_first[2], r.kv_first[3]);

        if (!(r.kv_a_rms > 1e-4f && r.kv_a_rms < 1000.0f)) {
            std::cerr << "[FAIL] kv_a_rms\n"; return 1;
        }
        if (!(r.kv_norm_rms > 1e-4f && r.kv_norm_rms < 1000.0f)) {
            std::cerr << "[FAIL] kv_norm_rms\n"; return 1;
        }
        // RoPE is an orthogonal rotation per pair; full-tensor L2 (and hence RMS)
        // is preserved.
        const float ratio = r.kv_post_rope_rms / std::max(r.kv_norm_rms, 1e-12f);
        if (!(ratio > 0.99f && ratio < 1.01f)) {
            std::cerr << "[FAIL] RoPE changed kv RMS: ratio=" << ratio << "\n";
            return 1;
        }
        std::cout << "[PASS] gguf attn kv path smoke\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "[FAIL] " << ex.what() << "\n";
        return 1;
    }
}
