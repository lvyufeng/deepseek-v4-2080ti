// Throughput of the paged-KV admission rule against the contiguous arena, at a
// fixed KV byte budget.
//
// The contiguous arena reserves max_context per slot up front, so a byte budget
// buys a fixed number of slots regardless of how long requests actually are.
// Paging hands the same bytes out per block, so short requests cost only what
// they span and more of them run at once. The workload here is mixed-length,
// which is where that difference shows: a run of short requests should reach a
// concurrency the arena cannot, while the long ones still get served.
//
// Both arms are given the same bytes, the same prompts in the same order, and
// the same slot ceiling, so the only variable is how capacity is divided.
//
// Usage: ./bench_qwen_paged_admission <checkpoint_dir> [--requests N]
//        [--max-context N] [--short N] [--long N] [--long-every N]
//        [--decode N] [--slots N] [--block-size N] [--budget-mb N]
//        [--prefill-budget N] [--layers N]

#include "qwen_batch_scheduler.hpp"
#include "qwen_engine.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <numeric>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace {

struct Options {
    std::string checkpoint;
    int requests = 64;
    int max_context = 8192;
    int short_len = 256;
    int long_len = 6144;
    // One in every N requests is a long one; the rest are short.
    int long_every = 8;
    int decode = 32;
    int slots = 16;
    int block_size = 256;
    // KV budget both arms must live within. 0 derives a budget that gives the
    // contiguous arena exactly 4 slots, so paging has something to beat.
    int budget_mb = 0;
    int prefill_budget = 2048;
    int layers = 0;
};

std::vector<int> make_prompt(int length, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::vector<int> tokens(static_cast<size_t>(length));
    for (int i = 0; i < length; ++i) {
        tokens[static_cast<size_t>(i)] = 1000 + static_cast<int>(rng() % 50000ULL);
    }
    return tokens;
}

// The request mix, built once and replayed identically by both arms.
std::vector<std::vector<int>> build_workload(const Options& opts) {
    std::vector<std::vector<int>> prompts;
    prompts.reserve(static_cast<size_t>(opts.requests));
    for (int i = 0; i < opts.requests; ++i) {
        const bool is_long = opts.long_every > 0 && (i % opts.long_every) == 0;
        prompts.push_back(make_prompt(is_long ? opts.long_len : opts.short_len,
                                      0x1000ULL + static_cast<uint64_t>(i)));
    }
    return prompts;
}

struct Result {
    double wall_seconds = 0.0;
    int completed = 0;
    int rejected = 0;
    long long generated_tokens = 0;
    long long prompt_tokens = 0;
    double mean_ttft = 0.0;
    double p99_ttft = 0.0;
    // Peak requests running at once, sampled by the submitting thread. This is
    // the mechanism under test: the arena's ceiling is its slot count, paging's
    // is whatever the block budget allows.
    int peak_concurrency = 0;
    int total_blocks = 0;
};

double percentile(std::vector<double> values, double q) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const size_t idx = static_cast<size_t>(q * (values.size() - 1) + 0.5);
    return values[std::min(idx, values.size() - 1)];
}

// Submits the whole workload as fast as it is accepted, then waits for it to
// drain. Rejections are counted rather than retried: under a fixed budget an
// oversized request is a permanent no, and retrying would hide that.
Result run_arm(pocket::QwenEngine& engine, const Options& opts,
               const std::vector<std::vector<int>>& prompts) {
    pocket::QwenBatchScheduler scheduler(&engine, opts.slots);
    scheduler.set_prefill_token_budget(opts.prefill_budget);

    pocket::QwenBatchSamplingParams sampling;
    sampling.temperature = 0.0f;  // greedy, so both arms do the same work
    sampling.max_new_tokens = opts.decode;
    sampling.ignore_eos = true;   // every request runs its full decode length

    std::mutex mu;
    std::vector<double> ttfts;
    std::atomic<int> done{0};
    long long generated = 0;
    long long prompt_total = 0;

    Result res;
    res.total_blocks = engine.kv_total_blocks();

    const auto start = std::chrono::steady_clock::now();

    int submitted = 0;
    for (const auto& prompt : prompts) {
        const uint64_t id = scheduler.submit_request(
            prompt, sampling, [&](const pocket::SchedulerGenerationResult& r) {
                std::lock_guard<std::mutex> lock(mu);
                ttfts.push_back(r.ttft_seconds);
                generated += r.completion_tokens;
                prompt_total += r.prompt_tokens;
                done.fetch_add(1);
            });
        if (id == 0) {
            ++res.rejected;
            continue;
        }
        ++submitted;
        res.peak_concurrency =
            std::max(res.peak_concurrency, scheduler.get_stats().running_requests);
    }

    // Generous: the arena arm serialises far more of the workload than paging.
    const auto deadline = start + std::chrono::minutes(30);
    while (done.load() < submitted && std::chrono::steady_clock::now() < deadline) {
        res.peak_concurrency =
            std::max(res.peak_concurrency, scheduler.get_stats().running_requests);
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    const auto end = std::chrono::steady_clock::now();
    scheduler.stop();

    std::lock_guard<std::mutex> lock(mu);
    res.wall_seconds = std::chrono::duration<double>(end - start).count();
    res.completed = done.load();
    res.generated_tokens = generated;
    res.prompt_tokens = prompt_total;
    if (!ttfts.empty()) {
        res.mean_ttft =
            std::accumulate(ttfts.begin(), ttfts.end(), 0.0) / ttfts.size();
        res.p99_ttft = percentile(ttfts, 0.99);
    }
    return res;
}

// A short discarded run so the measured pass does not pay first-touch costs.
// The prefix cache is cleared afterwards: these are the same prompts the
// measured pass uses, and a warm prefix would let it skip prefill entirely.
void warmup(pocket::QwenEngine& engine, const Options& opts,
            const std::vector<std::vector<int>>& prompts) {
    Options warm = opts;
    warm.decode = std::min(opts.decode, 4);
    const size_t n = std::min<size_t>(prompts.size(),
                                      static_cast<size_t>(opts.slots));
    const std::vector<std::vector<int>> subset(prompts.begin(),
                                               prompts.begin() + n);
    run_arm(engine, warm, subset);
    engine.clear_prefix_cache();
}

void report(const char* label, const Options& opts, const Result& r) {
    const double decode_tps =
        r.wall_seconds > 0.0 ? r.generated_tokens / r.wall_seconds : 0.0;
    const double req_ps =
        r.wall_seconds > 0.0 ? r.completed / r.wall_seconds : 0.0;
    std::printf("%-12s %8.2f %9.1f %8.2f %8d %8d %9.3f %9.3f", label,
                r.wall_seconds, decode_tps, req_ps, r.peak_concurrency,
                r.rejected, r.mean_ttft, r.p99_ttft);
    if (r.total_blocks > 0) {
        std::printf("  (%d blocks of %d)", r.total_blocks, opts.block_size);
    }
    std::printf("\n");
    // A throughput comparison only means something if both arms did the same
    // work, so print what was actually consumed and produced rather than
    // trusting that the same submissions imply the same forwards.
    std::printf("%-12s prompt_tokens=%lld generated_tokens=%lld completed=%d\n",
                "", r.prompt_tokens, r.generated_tokens, r.completed);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
                     "usage: %s <checkpoint_dir> [--requests N] [--max-context N]\n"
                     "       [--short N] [--long N] [--long-every N] [--decode N]\n"
                     "       [--slots N] [--block-size N] [--budget-mb N]\n"
                     "       [--prefill-budget N] [--layers N]\n",
                     argv[0]);
        return 2;
    }

    Options opts;
    opts.checkpoint = argv[1];
    for (int i = 2; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&](int& out) { if (i + 1 < argc) out = std::atoi(argv[++i]); };
        if (arg == "--requests") next(opts.requests);
        else if (arg == "--max-context") next(opts.max_context);
        else if (arg == "--short") next(opts.short_len);
        else if (arg == "--long") next(opts.long_len);
        else if (arg == "--long-every") next(opts.long_every);
        else if (arg == "--decode") next(opts.decode);
        else if (arg == "--slots") next(opts.slots);
        else if (arg == "--block-size") next(opts.block_size);
        else if (arg == "--budget-mb") next(opts.budget_mb);
        else if (arg == "--prefill-budget") next(opts.prefill_budget);
        else if (arg == "--layers") next(opts.layers);
        else {
            std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
            return 2;
        }
    }

    if (opts.long_len + opts.decode > opts.max_context) {
        std::fprintf(stderr,
                     "long prompt (%d) plus decode (%d) exceeds max_context (%d)\n",
                     opts.long_len, opts.decode, opts.max_context);
        return 2;
    }

    // Both arms get this many bytes. Derived from the engine's own per-slot cost
    // so the default is a round number of arena slots rather than a guess.
    pocket::QwenEngineOptions probe;
    probe.max_batch_size = 1;
    probe.kv_paged = false;
    pocket::QwenEngine sizer(opts.checkpoint, probe, opts.layers, opts.max_context);
    const uint64_t bytes_per_slot = sizer.kv_cache_bytes();
    const int arena_slots =
        opts.budget_mb > 0
            ? std::max(1, static_cast<int>(
                              (static_cast<uint64_t>(opts.budget_mb) << 20) /
                              std::max<uint64_t>(1, bytes_per_slot)))
            : 4;
    const uint64_t budget = static_cast<uint64_t>(arena_slots) * bytes_per_slot;

    const std::vector<std::vector<int>> prompts = build_workload(opts);
    const long long workload_tokens = std::accumulate(
        prompts.begin(), prompts.end(), 0LL,
        [](long long acc, const std::vector<int>& p) {
            return acc + static_cast<long long>(p.size());
        });

    std::printf("\nPaged KV admission throughput\n");
    std::printf("=============================\n");
    std::printf("%d requests: %d-token short, %d-token long every %d, %d decode each\n",
                opts.requests, opts.short_len, opts.long_len, opts.long_every,
                opts.decode);
    std::printf("max_context %d, slot ceiling %d, prefill budget %d\n",
                opts.max_context, opts.slots, opts.prefill_budget);
    std::printf("KV budget %.0f MiB = %d contiguous slots of %.0f MiB each\n",
                static_cast<double>(budget) / (1 << 20), arena_slots,
                static_cast<double>(bytes_per_slot) / (1 << 20));
    std::printf("workload %lld prompt tokens\n\n", workload_tokens);

    std::printf("%-12s %8s %9s %8s %8s %8s %9s %9s\n", "arm", "wall_s",
                "decode/s", "req/s", "peakcc", "reject", "ttft_avg", "ttft_p99");

    // Arena arm: the byte budget becomes a slot count, and the slot count is the
    // whole admission rule.
    {
        pocket::QwenEngineOptions o;
        o.max_batch_size = arena_slots;
        o.kv_paged = false;
        pocket::QwenEngine engine(opts.checkpoint, o, opts.layers, opts.max_context);
        engine.allocate_batch_slots(arena_slots);
        Options arm = opts;
        // Cannot exceed what the arena allocated, whatever the ceiling asks for.
        arm.slots = std::min(opts.slots, arena_slots);
        // Discarded: this arm runs first, so without it the contiguous number
        // absorbs one-time CUDA context and autotune cost that paging never pays.
        warmup(engine, arm, prompts);
        report("contiguous", arm, run_arm(engine, arm, prompts));
    }

    // Paged arm: same bytes, handed out per block against the full slot ceiling.
    {
        pocket::QwenEngineOptions o;
        o.max_batch_size = opts.slots;
        o.kv_paged = true;
        o.kv_block_size = opts.block_size;
        o.kv_cache_bytes = budget;
        pocket::QwenEngine engine(opts.checkpoint, o, opts.layers, opts.max_context);
        engine.allocate_batch_slots(opts.slots);
        warmup(engine, opts, prompts);
        report("paged", opts, run_arm(engine, opts, prompts));
    }

    return 0;
}
