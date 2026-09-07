// Measures whether a short request can get its first token while a long prompt
// is still prefilling.
//
// This is the reason chunked prefill exists, and it is not visible in the
// equivalence test: that one proves splitting a prompt does not change the
// output, this one proves splitting actually yields the device. Without a
// budget, a short request submitted just after a long one waits out the entire
// long prefill before its own can start, so its TTFT tracks the long prompt's
// length. With a budget, it should only wait for the chunk in flight.
//
// Reports the short request's TTFT under a budget and with chunking disabled;
// the ratio between them is the interleaving win.
//
// Usage: ./bench_qwen_prefill_interleave <checkpoint_dir> [--long N] [--budget N]
//                                        [--layers N] [--tp-world N] [--tp-rank N]
//                                        [--device N] [--nccl-id PATH]
#include "batch_scheduler.hpp"
#include "qwen_engine.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace {

std::vector<int> make_prompt(int length, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::vector<int> tokens(static_cast<size_t>(length));
    for (int i = 0; i < length; ++i) {
        tokens[static_cast<size_t>(i)] = 1000 + static_cast<int>(rng() % 100000ULL);
    }
    return tokens;
}

struct Trial {
    double short_ttft = 0.0;
    double long_total = 0.0;
};

// Submits one long request, then a short one just behind it, and reports when
// each got its first token.
Trial run_trial(pocket::QwenEngine& engine, int budget, int long_len,
                int short_len, int max_new) {
    pocket::BatchScheduler scheduler(&engine, 4);
    scheduler.set_prefill_token_budget(budget);

    pocket::BatchSamplingParams sampling;
    sampling.temperature = 0.0f;  // greedy
    sampling.max_new_tokens = max_new;

    std::mutex m;
    pocket::SchedulerGenerationResult long_result;
    pocket::SchedulerGenerationResult short_result;
    std::atomic<int> done{0};

    const std::vector<int> long_prompt = make_prompt(long_len, 0xAAAA);
    const std::vector<int> short_prompt = make_prompt(short_len, 0xBBBB);

    scheduler.submit_request(long_prompt, sampling,
                             [&](const pocket::SchedulerGenerationResult& r) {
                                 std::lock_guard<std::mutex> lock(m);
                                 long_result = r;
                                 done.fetch_add(1);
                             });
    // Far enough behind that the long prefill is definitely underway, so the
    // short request has to preempt rather than simply arrive first.
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    scheduler.submit_request(short_prompt, sampling,
                             [&](const pocket::SchedulerGenerationResult& r) {
                                 std::lock_guard<std::mutex> lock(m);
                                 short_result = r;
                                 done.fetch_add(1);
                             });

    const auto deadline = std::chrono::steady_clock::now() + std::chrono::minutes(5);
    while (done.load() < 2 && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    scheduler.stop();

    std::lock_guard<std::mutex> lock(m);
    Trial t;
    t.short_ttft = short_result.ttft_seconds;
    t.long_total = long_result.total_seconds;
    return t;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <checkpoint_dir> [--long N] [--budget N] "
                             "[--layers N] [--tp-world N] [--tp-rank N] "
                             "[--device N] [--nccl-id PATH]\n", argv[0]);
        return 2;
    }
    const std::string checkpoint = argv[1];
    int long_len = 16384;
    int budget = 4096;
    int layers = 0;
    int tp_world = 1;
    int tp_rank = 0;
    int device = -1;
    std::string nccl_id = "/tmp/qwen_interleave_nccl_id";
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--long" && i + 1 < argc) long_len = std::atoi(argv[++i]);
        else if (arg == "--budget" && i + 1 < argc) budget = std::atoi(argv[++i]);
        else if (arg == "--layers" && i + 1 < argc) layers = std::atoi(argv[++i]);
        else if (arg == "--tp-world" && i + 1 < argc) tp_world = std::atoi(argv[++i]);
        else if (arg == "--tp-rank" && i + 1 < argc) tp_rank = std::atoi(argv[++i]);
        else if (arg == "--device" && i + 1 < argc) device = std::atoi(argv[++i]);
        else if (arg == "--nccl-id" && i + 1 < argc) nccl_id = argv[++i];
        else {
            std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
            return 2;
        }
    }

    constexpr int kShortLen = 64;
    constexpr int kMaxNew = 8;

    pocket::QwenEngineOptions opts;
    opts.device = device >= 0 ? device : tp_rank;
    opts.max_batch_size = 4;
    if (tp_world > 1) {
        opts.tp_world = tp_world;
        opts.tp_rank = tp_rank;
        opts.nccl_id_path = nccl_id;
    }

    pocket::QwenEngine engine(checkpoint, opts, layers,
                            long_len + kShortLen + kMaxNew + 64);
    engine.allocate_batch_slots(4);
    engine.warmup_tp();
    if (tp_rank != 0) {
        engine.run_worker_loop();
        return 0;
    }

    std::printf("\nPrefill interleaving\n");
    std::printf("====================\n");
    std::printf("Long prompt %d tokens, short prompt %d tokens\n\n", long_len,
                kShortLen);

    const Trial chunked = run_trial(engine, budget, long_len, kShortLen, kMaxNew);
    std::printf("budget %5d: short TTFT %7.3f s  (long request total %7.3f s)\n",
                budget, chunked.short_ttft, chunked.long_total);

    // Both trials use the same two prompts, so without clearing the cache the
    // second one resumes the first one's prefix and never prefills at all --
    // which shows up as an impossibly fast unchunked baseline.
    if (tp_world > 1) engine.worker_command_reset();
    engine.clear_prefix_cache();

    // Budget 0 disables chunking, reproducing the previous behaviour.
    const Trial blocking = run_trial(engine, 0, long_len, kShortLen, kMaxNew);
    std::printf("unchunked  : short TTFT %7.3f s  (long request total %7.3f s)\n",
                blocking.short_ttft, blocking.long_total);

    if (chunked.short_ttft > 0.0 && blocking.short_ttft > 0.0) {
        std::printf("\nshort-request TTFT improvement: %.2fx\n",
                    blocking.short_ttft / chunked.short_ttft);
    }

    if (tp_world > 1) engine.worker_command_shutdown();
    return 0;
}
