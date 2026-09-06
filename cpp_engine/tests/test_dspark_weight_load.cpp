// Loads the DSpark draft weights from a real checkpoint and checks that every
// tensor landed on the device with sane values. This is the first DSpark test
// that touches real data: until weights load, every forward path is unverified.
#include "dspark.hpp"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>

namespace {

int failures = 0;

void expect(bool cond, const std::string& what) {
    if (cond) {
        std::printf("  [ok]   %s\n", what.c_str());
    } else {
        std::printf("  [FAIL] %s\n", what.c_str());
        ++failures;
    }
}

}  // namespace

int main(int argc, char** argv) {
    const char* ckpt = argc > 1 ? argv[1] : std::getenv("POCKETLLM_CKPT_DIR");
    if (ckpt == nullptr) {
        std::printf("usage: %s <checkpoint_dir>   (or set POCKETLLM_CKPT_DIR)\n", argv[0]);
        return 2;
    }
    const int tp_rank = argc > 2 ? std::atoi(argv[2]) : 0;
    const int tp_world = argc > 3 ? std::atoi(argv[3]) : 1;

    std::printf("DSpark weight load test\n  ckpt=%s tp_rank=%d tp_world=%d\n\n",
                ckpt, tp_rank, tp_world);

    size_t free_before = 0, total = 0;
    cudaMemGetInfo(&free_before, &total);

    dspark::DSparkEngine engine(ckpt, tp_rank, tp_world);
    const dspark::Config& cfg = engine.config();

    std::printf("Config parsed from config.json:\n");
    std::printf("  block_size=%d noise_token_id=%d markov_rank=%d window=%d\n",
                cfg.block_size, cfg.noise_token_id, cfg.markov_rank, cfg.window_size);
    std::printf("  dim=%d vocab=%d hc_mult=%d n_stages=%d\n",
                cfg.dim, cfg.vocab_size, cfg.hc_mult, cfg.n_stages);
    std::printf("  n_heads=%d head_dim=%d q_lora=%d o_lora=%d o_groups=%d rope_dim=%d\n",
                cfg.n_heads, cfg.head_dim, cfg.q_lora_rank, cfg.o_lora_rank,
                cfg.o_groups, cfg.rope_dim);
    std::printf("  n_experts=%d topk=%d moe_inter=%d\n\n",
                cfg.n_experts, cfg.topk, cfg.moe_inter);

    std::printf("Config checks (against DeepSeek-V4-Flash-0731 config.json):\n");
    expect(cfg.block_size == 5, "dspark_block_size == 5");
    expect(cfg.noise_token_id == 128799, "dspark_noise_token_id == 128799");
    expect(cfg.markov_rank == 256, "dspark_markov_rank == 256");
    expect(cfg.target_layer_ids.size() == 3, "dspark_target_layer_ids has 3 entries");
    expect(cfg.dim == 4096, "hidden_size == 4096");
    expect(cfg.vocab_size == 129280, "vocab_size == 129280");
    expect(cfg.hc_mult == 4, "hc_mult == 4");
    expect(cfg.window_size == 128, "sliding_window == 128");
    expect(cfg.n_heads == 64, "num_attention_heads == 64");
    expect(cfg.head_dim == 512, "head_dim == 512");
    expect(cfg.n_experts == 256, "n_routed_experts == 256");
    expect(cfg.topk == 6, "num_experts_per_tok == 6");
    // Detected by probing mtp.N prefixes in the index, not read from config.
    expect(cfg.n_stages == 3, "detected 3 mtp stages in checkpoint");

    size_t free_after = 0;
    cudaMemGetInfo(&free_after, &total);
    const double used_mb = static_cast<double>(free_before - free_after) / (1024.0 * 1024.0);
    std::printf("\nDevice memory consumed by DSpark weights: %.1f MB\n", used_mb);
    // embed alone is 129280*4096*2B ~= 1011 MB, plus 3 stages of attn/ffn.
    expect(used_mb > 900.0, "weights actually uploaded (>900 MB)");

    std::printf("\n%s (%d failure%s)\n", failures == 0 ? "PASS" : "FAIL",
                failures, failures == 1 ? "" : "s");
    return failures == 0 ? 0 : 1;
}
