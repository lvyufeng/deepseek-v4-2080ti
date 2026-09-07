// Verifies that generation stops on a stop token instead of running to
// max_new_tokens.
//
// Before this, batched decode's only finish condition was the length cap, so
// every request occupied its slot for the full max_new_tokens and the tokens
// past the model's EOS were kept in the output. The slots are a fixed pool, so
// that directly caps concurrency.
//
// The model will not emit EOS on cue, so these checks drive the stop through
// BatchSamplingParams::stop_token_ids: run once unrestricted, note a token
// the model actually produces at a known step, then re-run with that token as
// the stop and require generation to end exactly there. That tests the stop
// machinery, which is what changed; it does not depend on the checkpoint's own
// eos ids, which are covered separately by the config check below.
//
// Usage: ./test_qwen_eos_stop <checkpoint_dir> [--layers N] [--tp-world N]
//                             [--tp-rank N] [--device N] [--nccl-id PATH]
#include "qwen_engine.hpp"
#include "qwen_config.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace {

struct Options {
    std::string checkpoint;
    int tp_world = 1;
    int tp_rank = 0;
    int device = -1;
    int layers = 0;
    std::string nccl_id_path = "/tmp/qwen_eos_stop_nccl_id";
};

int failures = 0;

void expect(bool ok, const std::string& what) {
    std::printf("  %s %s\n", ok ? "[ok]  " : "[FAIL]", what.c_str());
    if (!ok) ++failures;
}

std::vector<int> make_prompt(int length, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::vector<int> tokens(static_cast<size_t>(length));
    for (int i = 0; i < length; ++i) {
        tokens[static_cast<size_t>(i)] = 1000 + static_cast<int>(rng() % 100000ULL);
    }
    return tokens;
}

// Drives one request through batch_prefill plus batch_decode_step, the same way
// the scheduler does, and returns every token including the prefill's.
struct RunOutcome {
    std::vector<int> tokens;
    bool finished_early = false;   // stopped before the length cap
    bool stop_flag_seen = false;   // hit_stop_token reported by the engine
};

// Each run takes its own slot: a slot carries both KV state and a prefix cache
// after a run, so re-prefilling the same prompt on a used slot would resume from
// that state instead of starting clean.
RunOutcome run_request(pocket::QwenEngine& engine, const std::vector<int>& prompt,
                       int slot_id,
                       const pocket::BatchSamplingParams& sampling) {
    RunOutcome out;
    auto req = std::make_unique<pocket::BatchedRequest>();
    req->request_id = static_cast<uint64_t>(slot_id) + 1;
    req->prompt_tokens = prompt;
    req->slot_id = slot_id;
    req->sampling = sampling;

    std::vector<pocket::BatchedRequest*> batch{req.get()};
    const pocket::BatchPrefillResult prefilled = engine.batch_prefill(batch, 0);
    out.tokens.push_back(prefilled.results[0].top_token);

    // batch_prefill marks the request finished when the prompt's first predicted
    // token is itself a stop token.
    if (req->finished) {
        out.finished_early = true;
        out.stop_flag_seen = true;
        return out;
    }

    while (static_cast<int>(out.tokens.size()) < sampling.max_new_tokens) {
        // batch_decode_step and batch_prefill announce their own collectives to
        // the workers, so this loop is the same at TP1 and TP4.
        const pocket::BatchDecodeResult step = engine.batch_decode_step(batch);
        out.tokens.push_back(step.next_tokens[0]);
        if (step.hit_stop_token[0]) out.stop_flag_seen = true;
        if (step.finished[0]) {
            out.finished_early =
                static_cast<int>(out.tokens.size()) < sampling.max_new_tokens;
            break;
        }
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    Options opts;
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <checkpoint_dir> [--layers N] "
                             "[--tp-world N] [--tp-rank N] [--device N] "
                             "[--nccl-id PATH]\n", argv[0]);
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
        } else if (arg == "--nccl-id" && i + 1 < argc) {
            opts.nccl_id_path = argv[++i];
        } else if (arg == "--layers" && i + 1 < argc) {
            opts.layers = std::atoi(argv[++i]);
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
            return 2;
        }
    }

    constexpr int kPromptLength = 64;
    constexpr int kMaxNewTokens = 24;
    constexpr int kMaxContext = 2048;

    pocket::QwenEngineOptions engine_opts;
    engine_opts.device = opts.device >= 0 ? opts.device : opts.tp_rank;
    // Three slots: one per run, since a used slot carries KV and prefix state.
    engine_opts.max_batch_size = 3;
    if (opts.tp_world > 1) {
        engine_opts.tp_world = opts.tp_world;
        engine_opts.tp_rank = opts.tp_rank;
        engine_opts.nccl_id_path = opts.nccl_id_path;
    }

    pocket::QwenEngine engine(opts.checkpoint, engine_opts, opts.layers, kMaxContext);
    engine.allocate_batch_slots(3);
    engine.warmup_tp();

    if (opts.tp_rank != 0) {
        engine.run_worker_loop();
        return 0;
    }

    std::printf("\nStop-token handling\n");
    std::printf("===================\n\n");

    // The checkpoint's own eos ids, which is what a request with no explicit
    // stop_token_ids falls back to.
    const std::vector<int>& eos_ids = engine.config().eos_token_ids;
    std::printf("checkpoint eos_token_ids:");
    for (int id : eos_ids) std::printf(" %d", id);
    std::printf("%s\n\n", eos_ids.empty() ? " (none)" : "");
    expect(!eos_ids.empty(), "checkpoint declares at least one eos token");

    const std::vector<int> prompt = make_prompt(kPromptLength, 0xE05);

    // Baseline: ignore_eos, so this must produce exactly max_new_tokens and
    // gives the reference token sequence.
    pocket::BatchSamplingParams baseline_params;
    baseline_params.max_new_tokens = kMaxNewTokens;
    baseline_params.ignore_eos = true;
    const RunOutcome baseline = run_request(engine, prompt, 0, baseline_params);
    expect(static_cast<int>(baseline.tokens.size()) == kMaxNewTokens,
           "ignore_eos runs to max_new_tokens");
    expect(!baseline.finished_early, "ignore_eos does not stop early");
    expect(!baseline.stop_flag_seen, "ignore_eos never reports a stop token");

    // Stop on a token from the baseline, and expect the stop at its *first*
    // occurrence. Picking baseline.tokens[k] and expecting to stop at k would be
    // wrong whenever that token also appeared earlier, which is common: a
    // layer-truncated model repeats itself heavily (token 220 filled most of the
    // baseline on 4 layers).
    int stop_index = -1;
    for (size_t i = 1; i < baseline.tokens.size(); ++i) {
        const auto first = std::find(baseline.tokens.begin(), baseline.tokens.end(),
                                     baseline.tokens[i]);
        if (first == baseline.tokens.begin() + static_cast<long>(i)) {
            stop_index = static_cast<int>(i);
            break;
        }
    }
    if (stop_index < 0) {
        // Every token after the first is a repeat, so no interior stop point
        // exists to test. Rather than assert something vacuous, say so: run with
        // more --layers for a model that does not collapse to one token.
        std::printf("  [FAIL] baseline has no token whose first occurrence is "
                    "past index 0; cannot place a stop\n");
        return 1;
    }
    const int kStopIndex = stop_index;
    const int stop_token = baseline.tokens[static_cast<size_t>(kStopIndex)];
    std::printf("\nstop token %d, expected at output index %d\n", stop_token,
                kStopIndex);

    pocket::BatchSamplingParams stop_params;
    stop_params.max_new_tokens = kMaxNewTokens;
    stop_params.stop_token_ids = {stop_token};
    const RunOutcome stopped = run_request(engine, prompt, 1, stop_params);

    // The token stream must be identical up to the stop: the stop check decides
    // when to halt and must not perturb the arithmetic.
    const bool prefix_matches =
        static_cast<int>(stopped.tokens.size()) == kStopIndex + 1 &&
        std::equal(stopped.tokens.begin(), stopped.tokens.end(),
                   baseline.tokens.begin());
    std::printf("  produced %zu tokens\n", stopped.tokens.size());
    expect(static_cast<int>(stopped.tokens.size()) == kStopIndex + 1,
           "stopped at the stop token, not at max_new_tokens");
    expect(prefix_matches, "tokens up to the stop match the baseline exactly");
    expect(stopped.finished_early, "reported finished before the length cap");
    expect(stopped.stop_flag_seen, "reported hit_stop_token");
    expect(!stopped.tokens.empty() && stopped.tokens.back() == stop_token,
           "stop token is kept as the last output token");

    // A stop id the model does not produce must not shorten anything: this
    // catches a stop check that fires on the wrong comparison.
    pocket::BatchSamplingParams absent_params;
    absent_params.max_new_tokens = kMaxNewTokens;
    absent_params.stop_token_ids = {-424242};
    const RunOutcome absent = run_request(engine, prompt, 2, absent_params);
    std::printf("\n");
    expect(static_cast<int>(absent.tokens.size()) == kMaxNewTokens,
           "an unreachable stop id runs to max_new_tokens");
    expect(absent.tokens == baseline.tokens,
           "an unreachable stop id leaves the tokens unchanged");

    if (opts.tp_world > 1) engine.worker_command_shutdown();

    std::printf("\n%s\n", failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
