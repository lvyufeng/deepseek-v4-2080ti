// End-to-end test of the speculative scheduler: does a multi-round speculative
// loop produce the same tokens as plain decode, while committing only the
// positions up to the first draft mismatch?
//
// The scheduler calls draft() and verifies one position at a time. This test
// confirms:
//
//   1. sequential speculative_step() exactly matches decode_step; batched mode
//      reports the first expected numerical divergence instead
//   2. A high-acceptance prompt generates the same tokens in fewer rounds
//   3. A rejection-heavy prompt takes the early-exit/rollback path without
//      corrupting compressor state, hidden capture, draft-ring state, or the
//      next round
//
// It does NOT measure end-to-end tok/s -- that needs a production serving loop
// with real request batching and is deferred to a separate benchmark.
//
//   test_speculative_scheduler <ckpt_dir> [tokens=8] [layers=43] [tp_world=1] [tp_rank=0] [nccl_id_path]

#include "persistent_engine.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using namespace pocket;

namespace {

int failures = 0;

void fail(const std::string& msg) {
    std::cout << "  [FAIL] " << msg << "\n";
    ++failures;
}

std::string join(const std::vector<int>& v) {
    std::string s;
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) s += " ";
        s += std::to_string(v[i]);
    }
    return s;
}

struct TrialResult {
    std::vector<int> reference;
    std::vector<int> speculative;
    int spec_rounds = 0;
    int total_accepted = 0;
    bool saw_rejection = false;
};

TrialResult run_trial(PersistentEngine& engine, const std::string& name,
                      const std::vector<int>& prompt, int token_count,
                      int full_draft_len, const SamplingParams& sp,
                      bool require_plain_exact, bool verbose) {
    TrialResult result;

    // Reference: plain decode from the same prompt. prefill/decode_step do not
    // fan out to workers on their own -- the caller drives them, as main.cpp does.
    engine.worker_command_reset();
    engine.reset_session();
    engine.worker_command_prefill(prompt);
    int token = engine.prefill(prompt, sp);
    int pos = static_cast<int>(prompt.size());
    for (int i = 0; i < token_count; ++i) {
        engine.worker_command_decode(token, pos);
        token = engine.decode_step(token, pos++, sp);
        result.reference.push_back(token);
    }

    // Speculative path: speculative_step drives every TP rank itself.
    engine.worker_command_reset();
    engine.reset_session();
    engine.worker_command_prefill(prompt);
    token = engine.prefill(prompt, sp);
    pos = static_cast<int>(prompt.size());
    while (static_cast<int>(result.speculative.size()) < token_count) {
        const std::vector<int> generated = engine.speculative_step(token, pos, sp);
        if (generated.empty()) {
            fail(name + ": speculative_step returned empty vector");
            break;
        }
        ++result.spec_rounds;
        const int n_generated = static_cast<int>(generated.size());
        const int remaining = token_count - static_cast<int>(result.speculative.size());
        const int n_committed = std::min(n_generated, remaining);
        result.total_accepted += n_committed - 1;
        result.saw_rejection |= n_generated < full_draft_len;

        result.speculative.insert(result.speculative.end(), generated.begin(),
                                  generated.begin() + n_committed);
        pos += n_committed;
        token = generated[n_committed - 1];

        if (verbose) {
            std::cout << "  " << name << " round " << result.spec_rounds
                      << " pos=" << pos << " generated=" << n_generated
                      << " accepted=" << n_generated - 1 << "\n";
        }
    }

    if (verbose) {
        std::cout << name << " reference  : " << join(result.reference) << "\n"
                  << name << " speculative: " << join(result.speculative) << "\n"
                  << name << " rounds=" << result.spec_rounds
                  << " accepted=" << result.total_accepted
                  << " avg=" << (result.spec_rounds > 0
                                      ? static_cast<double>(result.total_accepted) /
                                            result.spec_rounds
                                      : 0.0)
                  << " rejected=" << (result.saw_rejection ? "yes" : "no") << "\n";
    }

    if (result.speculative.size() != result.reference.size()) {
        fail(name + ": generated " + std::to_string(result.speculative.size()) +
             " tokens, expected " + std::to_string(result.reference.size()));
        return result;
    }

    const auto mismatch = std::mismatch(result.speculative.begin(), result.speculative.end(),
                                        result.reference.begin());
    if (mismatch.first != result.speculative.end()) {
        const int i = static_cast<int>(mismatch.first - result.speculative.begin());
        std::cout << "  " << name << ": first plain divergence at position " << i
                  << " (spec=" << result.speculative[i]
                  << " ref=" << result.reference[i] << ")\n";
        if (require_plain_exact) {
            fail(name + ": tokens diverged at position " + std::to_string(i) +
                 " (spec=" + std::to_string(result.speculative[i]) +
                 " ref=" + std::to_string(result.reference[i]) + ")");
        }
    }
    return result;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0]
                  << " <ckpt_dir> [tokens=8] [layers=43] [tp_world=1] [tp_rank=0]"
                     " [nccl_id_path]\n";
        return 2;
    }
    const std::string ckpt_dir = argv[1];
    const int token_count = argc > 2 ? std::atoi(argv[2]) : 8;
    const int layer_count = argc > 3 ? std::atoi(argv[3]) : 43;

    ForwardSmokeOptions opts;
    opts.tp_world = argc > 4 ? std::atoi(argv[4]) : 1;
    opts.tp_rank = argc > 5 ? std::atoi(argv[5]) : 0;
    opts.device = opts.tp_rank;
    if (argc > 6) opts.nccl_id_path = argv[6];
    if (opts.tp_world > 1 && opts.nccl_id_path.empty()) {
        std::cerr << "tp_world > 1 needs an nccl_id_path\n";
        return 2;
    }
    const bool verbose = opts.tp_rank == 0;

    if (cudaSetDevice(opts.device) != cudaSuccess) {
        std::cerr << "failed to set CUDA device " << opts.device << "\n";
        return 2;
    }

    PersistentEngine engine(ckpt_dir, opts, layer_count, 2048);
    engine.load_dspark(ckpt_dir);
    if (!engine.dspark_loaded()) {
        std::cout << "[FAIL] load_dspark did not take effect\n";
        return 1;
    }

    // Rank 0 drives; workers block on the command channel. The draft, verify,
    // and draft-ring write all fan out from speculative_step().
    engine.warmup_tp();
    if (opts.tp_rank != 0) {
        engine.run_worker_loop();
        return 0;
    }

    SamplingParams sp;
    sp.greedy = true;
    sp.temperature = 1.0f;
    sp.seed = 12345;

    // draft_tokens returns [committed, d0, ..., d4] for block_size=5.
    const int full_draft_len = 6;

    // A batched GEMM changes reduction order, so batched greedy successors can
    // differ from plain decode even when transaction commit/rollback is correct.
    const bool require_plain_exact =
        std::getenv("POCKETLLM_CPP_BATCHED_VERIFY") == nullptr ||
        std::atoi(std::getenv("POCKETLLM_CPP_BATCHED_VERIFY")) == 0;

    // Near-certain continuation: exercises full acceptance and amortization.
    const std::vector<int> cyclic = {
        16, 18, 16, 18, 16, 18, 16, 18, 16, 18, 16, 18,
    };
    const TrialResult accepted = run_trial(engine, "cyclic", cyclic, token_count,
                                            full_draft_len, sp,
                                            require_plain_exact, verbose);
    if (accepted.spec_rounds >= token_count) {
        fail("cyclic: speculative rounds " + std::to_string(accepted.spec_rounds) +
             " >= decode rounds " + std::to_string(token_count) +
             " (no amortization)");
    }

    // Ordinary English: previously measured 0/30 acceptance. This is the
    // important correctness branch -- rejection must stop before forwarding the
    // tail, then the following rounds must still match plain decode exactly.
    const std::vector<int> english = {0, 17665, 31114, 12, 526, 318, 264, 4017, 30};
    const TrialResult rejected = run_trial(engine, "english", english, token_count,
                                            full_draft_len, sp,
                                            require_plain_exact, verbose);
    if (!rejected.saw_rejection) {
        fail("english: expected at least one early-exit rejection");
    }

    engine.worker_command_shutdown();
    if (failures != 0) {
        std::cout << "[FAIL] speculative_scheduler: " << failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "[PASS] speculative_scheduler (accept and reject paths verified)\n";
    return 0;
}
