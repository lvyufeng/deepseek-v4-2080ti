// Perfect-draft test for PersistentEngine::verify_step.
//
// Speculative decoding is only correct if verifying a draft block yields the
// same continuation the model would have produced one token at a time. This
// test removes draft quality from the question entirely: it decodes normally,
// then feeds that exact output back in as a "perfect" draft. A faithful verify
// must then agree everywhere -- any mismatch is the verify path's own fault,
// not the drafter's.
//
// Semantics being checked: verify_step(draft, pos) returns, for each draft
// token, what the model samples *after* consuming it. So with a perfect draft
// starting at plain_tokens[i], the k-th returned token must equal
// plain_tokens[i + k + 1] -- the successor, not the draft token itself.
//
//   test_perfect_draft <ckpt_dir> [max_tokens=20] [draft_len=3] [layers=3]
//
// Exits non-zero on any mismatch. Sequential verify is expected to match
// exactly (it runs the same per-token forward as decode); a batched verify
// would not, which is the reason verify_step is sequential today.

#include "persistent_engine.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using namespace pocket;

namespace {

void print_tokens(const char* label, const std::vector<int>& v) {
    std::cout << label << " [";
    for (size_t i = 0; i < v.size(); ++i) {
        std::cout << v[i];
        if (i + 1 < v.size()) std::cout << ", ";
    }
    std::cout << "]\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0]
                  << " <ckpt_dir> [max_tokens=20] [draft_len=3] [layers=3]\n";
        return 2;
    }

    const std::string ckpt_dir = argv[1];
    const int max_tokens = argc > 2 ? std::atoi(argv[2]) : 20;
    const int draft_len = argc > 3 ? std::atoi(argv[3]) : 3;
    const int layer_count = argc > 4 ? std::atoi(argv[4]) : 3;

    if (max_tokens < 2 || draft_len < 1) {
        std::cerr << "max_tokens must be >= 2 and draft_len >= 1\n";
        return 2;
    }

    std::cout << "perfect_draft ckpt=" << ckpt_dir << " max_tokens=" << max_tokens
              << " draft_len=" << draft_len << " layers=" << layer_count << "\n";

    ForwardSmokeOptions opts;
    opts.tp_world = 1;
    opts.tp_rank = 0;
    opts.device = 0;

    const int max_context = 2048;
    PersistentEngine engine(ckpt_dir, opts, layer_count, max_context);

    SamplingParams sp;
    sp.greedy = true;  // a sampled comparison would prove nothing
    sp.temperature = 1.0f;
    sp.seed = 12345;

    const std::vector<int> prompt = {1, 15043, 5312};

    // --- Ground truth: plain one-token-at-a-time decode ---
    engine.reset_session();
    int token = engine.prefill(prompt, sp);
    std::vector<int> plain = {token};
    for (int i = 1; i < max_tokens; ++i) {
        const int position = static_cast<int>(prompt.size()) + i - 1;
        token = engine.decode_step(token, position, sp);
        plain.push_back(token);
    }
    print_tokens("plain  =", plain);

    // --- Same continuation, fed back as a perfect draft ---
    engine.reset_session();
    std::vector<int> got = {engine.prefill(prompt, sp)};

    int mismatches = 0;
    int checked = 0;
    int pos = static_cast<int>(prompt.size());

    while (static_cast<int>(got.size()) < max_tokens) {
        const int have = static_cast<int>(got.size());
        // Draft the tokens the model already committed to, starting at the last
        // one we hold. n_draft is capped so every prediction has a ground truth
        // successor to compare against.
        const int n_draft = std::min(draft_len, max_tokens - have);

        std::vector<int> draft;
        draft.reserve(static_cast<size_t>(n_draft));
        for (int i = 0; i < n_draft; ++i) draft.push_back(plain[have - 1 + i]);

        const std::vector<int> next = engine.verify_step(draft, pos, sp);
        if (static_cast<int>(next.size()) != n_draft) {
            std::cerr << "verify_step returned " << next.size() << " tokens, want "
                      << n_draft << "\n";
            return 1;
        }

        for (int i = 0; i < n_draft; ++i) {
            const int expect_idx = have + i;
            if (expect_idx >= max_tokens) break;
            ++checked;
            if (next[i] != plain[expect_idx]) {
                ++mismatches;
                std::cout << "  mismatch at token " << expect_idx
                          << ": plain=" << plain[expect_idx] << " verify=" << next[i]
                          << " (draft slot " << i << ", pos " << pos + i << ")\n";
            }
        }

        for (int i = 0; i < n_draft && static_cast<int>(got.size()) < max_tokens; ++i) {
            got.push_back(next[i]);
        }
        pos += n_draft;
    }
    print_tokens("verify =", got);

    std::cout << "checked=" << checked << " mismatches=" << mismatches << "\n";
    if (mismatches != 0) {
        std::cout << "[FAIL] perfect_draft: verify disagrees with plain decode\n";
        return 1;
    }
    std::cout << "[PASS] perfect_draft: verify reproduces plain decode exactly\n";
    return 0;
}
