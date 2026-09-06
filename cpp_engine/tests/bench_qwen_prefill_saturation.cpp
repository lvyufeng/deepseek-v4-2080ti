// Measures how prefill throughput scales with prompt length on a single slot.
//
// Batched prefill only pays off where one prompt leaves the GPU idle. Prefill
// already chunks at options.prefill_chunk_tokens (8192 by default), so a prompt
// at or above that runs full-width chunks and merging prompts would add nothing.
// The open question is the short end: if 512 tokens already reaches the same
// tok/s as 8192, the GEMMs are saturated and batched prefill is not where the
// next win is. This sweep answers that before any of it is written.
//
// Usage: ./bench_qwen_prefill_saturation <checkpoint_dir> [--tp-world N]
//                                        [--tp-rank N] [--device N]
//                                        [--nccl-id PATH] [--iters N]
#include "qwen_engine.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

namespace {

struct Options {
    std::string checkpoint;
    int tp_world = 1;
    int tp_rank = 0;
    int device = -1;  // -1 means "use tp_rank"
    int iters = 3;
    std::string nccl_id_path = "/tmp/qwen_prefill_sat_nccl_id";
};

void usage(const char* argv0) {
    std::fprintf(stderr,
                 "usage: %s <checkpoint_dir> [options]\n"
                 "  --tp-world N       TP world size (default 1)\n"
                 "  --tp-rank N        TP rank (default 0)\n"
                 "  --device N         CUDA device ordinal (default: tp-rank)\n"
                 "  --iters N          timed repeats per length (default 3)\n"
                 "  --nccl-id PATH     NCCL id rendezvous file\n",
                 argv0);
}

// Distinct pseudo-random ids per run so nothing can be served from cache.
std::vector<int> make_prompt(int length, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::vector<int> tokens(static_cast<size_t>(length));
    for (int i = 0; i < length; ++i) {
        tokens[static_cast<size_t>(i)] =
            1000 + static_cast<int>(rng() % 100000ULL);
    }
    return tokens;
}

const int kLengths[] = {128, 256, 512, 1024, 2048, 4096, 8192};
constexpr int kMaxLength = 8192;

}  // namespace

int main(int argc, char** argv) {
    Options opts;
    if (argc < 2) {
        usage(argv[0]);
        return 2;
    }
    opts.checkpoint = argv[1];
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--tp-world" && i + 1 < argc) {
            opts.tp_world = std::atoi(argv[++i]);
        } else if (arg == "--tp-rank" && i + 1 < argc) {
            opts.tp_rank = std::atoi(argv[++i]);
        } else if (arg == "--device" && i + 1 < argc) {
            opts.device = std::atoi(argv[++i]);
        } else if (arg == "--iters" && i + 1 < argc) {
            opts.iters = std::atoi(argv[++i]);
        } else if (arg == "--nccl-id" && i + 1 < argc) {
            opts.nccl_id_path = argv[++i];
        } else {
            usage(argv[0]);
            return 2;
        }
    }

    pocket::QwenEngineOptions engine_opts;
    engine_opts.device = opts.device >= 0 ? opts.device : opts.tp_rank;
    engine_opts.max_batch_size = 1;
    // Prefix reuse would let a repeat return cached logits instead of computing,
    // which is exactly what this benchmark must not measure. It also keeps every
    // prefill starting from position 0 with a zeroed recurrent state.
    engine_opts.prefix_cache = false;
    if (opts.tp_world > 1) {
        engine_opts.tp_world = opts.tp_world;
        engine_opts.tp_rank = opts.tp_rank;
        engine_opts.nccl_id_path = opts.nccl_id_path;
    }

    pocket::QwenEngine engine(opts.checkpoint, engine_opts, 0, kMaxLength + 64);
    engine.allocate_batch_slots(1);

    const bool is_rank0 = opts.tp_rank == 0;
    if (is_rank0) {
        std::printf("\nQwen prefill saturation sweep\n");
        std::printf("=============================\n");
        std::printf("Checkpoint: %s\n", opts.checkpoint.c_str());
        std::printf("TP: %d (rank %d)\n", opts.tp_world, opts.tp_rank);
        std::printf("Timed repeats per length: %d\n\n", opts.iters);
        std::printf("%8s %12s %12s\n", "tokens", "ms", "tok/s");
    }

    std::vector<double> tok_per_s;
    for (int length : kLengths) {
        const uint64_t seed =
            0x9E3779B97F4A7C15ULL ^ static_cast<uint64_t>(length);

        // One untimed pass so allocator growth and any autotuning are not
        // charged to the first measured iteration.
        (void)engine.prefill(make_prompt(length, seed), 0);

        double total_seconds = 0.0;
        for (int iter = 0; iter < opts.iters; ++iter) {
            const std::vector<int> prompt =
                make_prompt(length, seed + 1 + static_cast<uint64_t>(iter));
            const auto start = std::chrono::steady_clock::now();
            (void)engine.prefill(prompt, 0);
            const auto end = std::chrono::steady_clock::now();
            total_seconds += std::chrono::duration<double>(end - start).count();
        }

        const double seconds = total_seconds / opts.iters;
        tok_per_s.push_back(static_cast<double>(length) / seconds);
        if (is_rank0) {
            std::printf("%8d %12.3f %12.1f\n", length, seconds * 1000.0,
                        tok_per_s.back());
            std::fflush(stdout);
        }
    }

    if (is_rank0) {
        const double saturated = tok_per_s.back();
        const size_t count = tok_per_s.size();
        std::printf("\nThroughput relative to the %d-token chunk\n", kMaxLength);
        std::printf("----------------------------------------\n");
        for (size_t i = 0; i < count; ++i) {
            std::printf("%8d %11.1f tok/s  %5.2fx\n", kLengths[i], tok_per_s[i],
                        tok_per_s[i] / saturated);
        }

        // tok_per_s[2] is the 512-token row: a plausible chat prompt, and the
        // length where a scheduler would most want to merge requests.
        const double short_ratio = tok_per_s[2] / saturated;
        std::printf("\nInterpretation\n--------------\n");
        if (short_ratio < 0.75) {
            std::printf(
                "512-token prefill runs at %.2fx the 8192-token rate, so short\n"
                "prompts leave the GEMMs underfed. Merging several into one\n"
                "forward has roughly %.2fx of headroom to recover.\n",
                short_ratio, saturated / tok_per_s[2]);
        } else {
            std::printf(
                "512-token prefill already reaches %.2fx the 8192-token rate,\n"
                "so a single short prompt nearly saturates the GEMMs and batched\n"
                "prefill would mostly add bookkeeping. Prefer paged KV instead.\n",
                short_ratio);
        }
    }
    return 0;
}
