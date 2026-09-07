// Verifies that bounded (chunked) prefill computes the same thing as one
// unbounded prefill of the same prompt.
//
// Chunked prefill exists so a long prompt cannot hold the GPU while other
// requests wait, but it is only usable if splitting a prompt is unobservable in
// the output. The risk is entirely in resumption: prefill_partial leaves the
// slot position and the cached prompt at the boundary it reached, and the next
// call finds its resume point by matching against exactly those. An off-by-one
// there silently skips or repeats prompt tokens, which shows up as different
// logits rather than as a crash.
//
// Usage: ./test_qwen_chunked_prefill <checkpoint_dir> [--tp-world N]
//                                    [--tp-rank N] [--device N] [--nccl-id PATH]
#include "qwen_engine.hpp"

#include <cmath>
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
    int device = -1;
    // 0 loads every layer. A smaller count lets this run on a single card.
    int layers = 0;
    // Prompt tokens per bounded step. The engine's chunk width is set to match,
    // so the reference prefill and the chunked run do identical arithmetic.
    int budget = 512;
    std::string nccl_id_path = "/tmp/qwen_chunked_prefill_nccl_id";
};

std::vector<int> make_prompt(int length, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::vector<int> tokens(static_cast<size_t>(length));
    for (int i = 0; i < length; ++i) {
        tokens[static_cast<size_t>(i)] =
            1000 + static_cast<int>(rng() % 100000ULL);
    }
    return tokens;
}

int failures = 0;

void expect(bool ok, const std::string& what) {
    std::printf("  %s %s\n", ok ? "[ok]  " : "[FAIL]", what.c_str());
    if (!ok) ++failures;
}

// Compares the sampled token plus checksum and top logit. The token alone would
// miss a drift that has not yet moved the argmax; the checksum covers the whole
// logit row, so it catches a difference anywhere in the vocabulary.
void expect_same_result(const pocket::ForwardResult& chunked,
                        const pocket::ForwardResult& whole,
                        const std::string& label) {
    if (chunked.top_token != whole.top_token) {
        std::printf("  [FAIL] %s: token %d vs %d\n", label.c_str(),
                    chunked.top_token, whole.top_token);
        ++failures;
        return;
    }
    // Both paths run the same kernels over the same rows in the same order, so
    // the expectation is exact equality rather than a tolerance.
    if (chunked.checksum != whole.checksum ||
        chunked.top_logit != whole.top_logit) {
        std::printf("  [FAIL] %s: checksum %.9g vs %.9g, top logit %.9g vs %.9g\n",
                    label.c_str(), chunked.checksum, whole.checksum,
                    chunked.top_logit, whole.top_logit);
        ++failures;
        return;
    }
    if (chunked.position != whole.position) {
        std::printf("  [FAIL] %s: position %d vs %d\n", label.c_str(),
                    chunked.position, whole.position);
        ++failures;
        return;
    }
    std::printf("  [ok]   %s: token %d, checksum %.9g identical\n", label.c_str(),
                chunked.top_token, chunked.checksum);
}

}  // namespace

int main(int argc, char** argv) {
    Options opts;
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <checkpoint_dir> [--budget N] "
                             "[--layers N] [--tp-world N] [--tp-rank N] "
                             "[--device N] [--nccl-id PATH]\n",
                     argv[0]);
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
        } else if (arg == "--budget" && i + 1 < argc) {
            opts.budget = std::max(1, std::atoi(argv[++i]));
        } else {
            std::fprintf(stderr, "unknown argument: %s\n", arg.c_str());
            return 2;
        }
    }

    constexpr int kPromptLength = 2600;  // Not a multiple of any chunk tested.
    constexpr int kMaxContext = 4096;

    pocket::QwenEngineOptions engine_opts;
    engine_opts.device = opts.device >= 0 ? opts.device : opts.tp_rank;
    engine_opts.max_batch_size = 2;
    // Match the chunk width to the budget so the only difference between the
    // reference and the chunked run is whether prefill returned between chunks.
    engine_opts.prefill_chunk_tokens = opts.budget;
    if (opts.tp_world > 1) {
        engine_opts.tp_world = opts.tp_world;
        engine_opts.tp_rank = opts.tp_rank;
        engine_opts.nccl_id_path = opts.nccl_id_path;
    }

    pocket::QwenEngine engine(opts.checkpoint, engine_opts, opts.layers, kMaxContext);
    engine.allocate_batch_slots(2);
    engine.warmup_tp();

    // Workers spend the rest of their life in the command loop, executing the
    // prefills rank 0 announces. Only rank 0 runs the checks below.
    const bool is_rank0 = opts.tp_rank == 0;
    if (!is_rank0) {
        engine.run_worker_loop();
        return 0;
    }

    const std::vector<int> prompt = make_prompt(kPromptLength, 0xC0FFEE);

    std::printf("\nChunked prefill equivalence\n");
    std::printf("===========================\n");
    std::printf("Prompt: %d tokens, TP %d\n\n", kPromptLength, opts.tp_world);

    // The reference is an unbounded prefill on this same engine, so it runs the
    // configured chunk width -- which the caller sets equal to the budget under
    // test via --budget.
    //
    // Why the widths must match, since it is easy to mistake for a bug here:
    // chunk width alone already moves the logits. With this prompt on 4 layers,
    // an unmodified engine at prefill_chunk_tokens=512 returns checksum
    // 6.58258438 where 8192 returns 6.58369541. The FP8 projections dequantize
    // per call and reduce over a different row count per chunk, so a narrower
    // chunk reassociates the arithmetic. That predates this change, and is why
    // prefill_chunk_tokens is a tuning knob and not a correctness one.
    //
    // The invariant that actually pins resumption down is therefore: stopping and
    // resuming at token N equals never having stopped, at one fixed chunk width.
    const int budget = opts.budget;
    std::printf("Budget %d, engine chunk width %d\n\n", budget,
                engine_opts.prefill_chunk_tokens);

    if (opts.tp_world > 1) engine.worker_command_prefill(prompt, 1, 0);
    const pocket::ForwardResult whole = engine.prefill(prompt, 1);

    int steps = 0;
    int consumed = 0;
    pocket::PartialPrefillResult step;
    do {
        if (opts.tp_world > 1) {
            engine.worker_command_prefill(prompt, 0, budget);
        }
        step = engine.prefill_partial(prompt, 0, budget);
        ++steps;
        // Progress must be strictly monotonic, or a scheduler driving this loop
        // would spin forever.
        if (step.consumed_tokens <= consumed) {
            std::printf("  [FAIL] stalled at %d tokens\n", step.consumed_tokens);
            ++failures;
            break;
        }
        consumed = step.consumed_tokens;
        if (steps > kPromptLength) {
            std::printf("  [FAIL] did not terminate\n");
            ++failures;
            break;
        }
    } while (!step.complete);

    std::printf("chunked prefill: %d steps\n", steps);
    expect(step.complete, "reported complete");
    expect(step.consumed_tokens == kPromptLength, "consumed the whole prompt");
    // A budget below the prompt length must actually have yielded, or nothing
    // was interleavable and the feature does not work.
    if (budget < kPromptLength) {
        expect(steps > 1, "took more than one step");
    }
    expect_same_result(step.result, whole, "final logits");
    std::printf("\n");

    // Release the workers from their command loop, or they never exit.
    if (opts.tp_world > 1) engine.worker_command_shutdown();

    std::printf("%s\n", failures == 0 ? "PASS" : "FAIL");
    return failures == 0 ? 0 : 1;
}
