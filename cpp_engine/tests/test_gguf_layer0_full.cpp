// Phase 3 step: GGUF layer-0 full forward composition with residuals.
// Embed -> attn_norm -> attention (Q/KV/sparse_attn/inv_rope/wo_a/wo_b) ->
// +x (residual) -> ffn_norm -> (Q8_0 shared expert + Q2 routed MoE with real
// hash-gate weights) -> +x (residual) -> x_out.

#include "deepseek_v4_engine.hpp"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: test_gguf_layer0_full <model.gguf> [token] [position]\n";
        return 2;
    }
    const int token = argc >= 3 ? std::atoi(argv[2]) : 1234;
    const int position = argc >= 4 ? std::atoi(argv[3]) : 0;
    try {
        auto r = pocket::run_gguf_layer0_full_smoke(argv[1], token, position);
        std::printf("token=%d position=%d dim=%d moe_inter=%d heads=%d head_dim=%d n_active=%d\n",
                    token, position, r.dim, r.moe_inter_dim, r.heads, r.head_dim, r.n_active);
        std::printf("expert_ids        = [");
        for (int k = 0; k < r.n_active; ++k) {
            std::printf("%d%s", r.expert_ids[k], k + 1 < r.n_active ? ", " : "");
        }
        std::printf("]\n");
        std::printf("embed_rms         = %.4f\n", r.embed_rms);
        std::printf("attn_out_rms      = %.4f\n", r.attn_out_rms);
        std::printf("x_post_attn_rms   = %.4f  (embed + attn_out)\n", r.x_post_attn_rms);
        std::printf("shared_out_rms    = %.4f\n", r.shared_out_rms);
        std::printf("moe_out_rms       = %.4f\n", r.moe_out_rms);
        std::printf("ffn_combined_rms  = %.4f  (shared_out + moe_out)\n", r.ffn_combined_rms);
        std::printf("x_post_ffn_rms    = %.4f  (x_post_attn + ffn_combined)\n", r.x_post_ffn_rms);
        std::printf("x_post_ffn[0..3]  = %.4f %.4f %.4f %.4f\n",
                    r.x_post_ffn_first[0], r.x_post_ffn_first[1],
                    r.x_post_ffn_first[2], r.x_post_ffn_first[3]);
        std::printf("route_weights_sum = %.4f\n", r.route_weights_sum);

        if (!(r.embed_rms > 1e-4f && r.embed_rms < 1000.0f)) {
            std::cerr << "[FAIL] embed_rms\n"; return 1;
        }
        if (!(r.attn_out_rms > 1e-6f && r.attn_out_rms < 1000.0f)) {
            std::cerr << "[FAIL] attn_out_rms\n"; return 1;
        }
        if (!(r.x_post_attn_rms > 1e-4f && r.x_post_attn_rms < 1000.0f)) {
            std::cerr << "[FAIL] x_post_attn_rms\n"; return 1;
        }
        if (!(r.shared_out_rms > 1e-6f && r.shared_out_rms < 1000.0f)) {
            std::cerr << "[FAIL] shared_out_rms\n"; return 1;
        }
        if (!(r.moe_out_rms > 1e-6f && r.moe_out_rms < 1000.0f)) {
            std::cerr << "[FAIL] moe_out_rms\n"; return 1;
        }
        if (!(r.x_post_ffn_rms > 1e-4f && r.x_post_ffn_rms < 1000.0f)) {
            std::cerr << "[FAIL] x_post_ffn_rms\n"; return 1;
        }
        if (!(r.route_weights_sum > 0.5f && r.route_weights_sum < 5.0f)) {
            std::cerr << "[FAIL] route_weights_sum\n"; return 1;
        }
        std::cout << "[PASS] gguf layer-0 full forward smoke\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "[FAIL] " << ex.what() << "\n";
        return 1;
    }
}
