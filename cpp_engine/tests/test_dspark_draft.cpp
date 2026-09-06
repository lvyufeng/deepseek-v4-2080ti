// End-to-end DSpark draft test: does the draft module, seeded from the main
// model's captured hidden, actually predict what the main model does next?
//
// The trap here is that every wrong version still looks right. A draft seeded
// with the wrong hidden, the wrong position, or a hidden from the wrong layers
// still emits fluent-looking token ids of the correct shape; the only symptom
// is a lower accept rate, which nothing in a shape or finiteness check can see.
//
// So the test needs a prompt whose continuation the main model is near-certain
// about. On such a prompt a working draft accepts most of its block and a broken
// one accepts nothing, and the difference is unambiguous. On ordinary text it is
// not: measured here, the same build reads 0/30 on a short English prompt and
// 30/30 on the cyclic default, because the English continuation is genuinely
// uncertain and even a correct draft loses. A low rate on a hard prompt says
// nothing. Hence the default prompt is cyclic and the pass bar is an absolute
// rate.
//
// The corrupted seeds below are reported but *not* asserted on, and the reason
// is the same fact from the other side: on a prompt predictable enough to make
// the absolute rate meaningful, the ring and the committed token already carry
// the answer, so a zeroed seed drafts just as well (measured: real 30/30, zeros
// 30/30). The seed's contribution is real -- test_dspark_draft_sensitivity.cpp
// shows a zeroed seed changing 5/5 drafted tokens once the ring is zeroed too --
// but it is not separable *here*, and an assertion that only holds on prompts
// where the absolute rate is uninformative would be worse than none.
//
//   real    the captured hidden for the committed token
//   zeros   no signal at all -- the floor for "the draft module alone"
//   stale   a hidden captured at an earlier position, i.e. the mistake an
//           off-by-one in the plumbing would actually produce
//
// Block alignment: after prefill(ctx) the KV holds positions 0..ctx-1 and
// `committed` is the model's token for position ctx, not yet consumed. So the
// block handed to verify_step is [committed, drafts...] starting at ctx, and
// verified[i] -- what the model samples after consuming block[i] -- is the
// truth for block[i+1]. Verifying the drafts alone would silently shift every
// comparison by one position and read as a near-zero accept rate.
//
//   test_dspark_draft <ckpt_dir> [rounds=6] [layers=43] [tp_world=1] [tp_rank=0] [nccl_id_path] [prompt_ids]
//
// prompt_ids is an optional comma-separated token id list, defaulting to a
// cyclic pattern for the reason above. Pass ordinary text ids to measure a
// realistic accept rate, but expect it to be low and do not read that as a
// broken draft.
//
// The first rounds of a short cyclic prompt score 0 until the pattern is
// established in the ring; the 12-token default is long enough to avoid that.
// The bar is on the total either way, and the per-round trace is printed so a
// run that never converges is distinguishable from one that merely starts slow.
//
// TP=1 does not fit on a 22 GB card: the draft's ~12.4 GB sits on top of the
// main model's FP4 arena, and the layer count cannot be cut because the draft
// reads layers 40/41/42. Run it at TP=4 (3.8 GB of draft weights per rank),
// which also exercises the draft's own all-reduces. Every rank drafts the same
// tokens -- the head is replicated -- so any rank's output is the answer.

#include "persistent_engine.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

using namespace pocket;

namespace {

int failures = 0;

// Device memory is the binding constraint for this test, so report it at each
// stage rather than leaving an OOM to be diagnosed from a bare allocation name.
void report_mem(const char* stage, int rank) {
    size_t freeb = 0, totalb = 0;
    if (cudaMemGetInfo(&freeb, &totalb) != cudaSuccess) return;
    std::cout << "  [mem] rank " << rank << " " << stage << ": used="
              << (totalb - freeb) / (1024 * 1024) << " MiB free="
              << freeb / (1024 * 1024) << " MiB\n"
              << std::flush;
}

void fail(const std::string& msg) {
    std::cout << "  [FAIL] " << msg << "\n";
    ++failures;
}

// Leading tokens of `block` (= [committed, drafts...]) the main model agrees
// with. verified[i] is the model's token after consuming block[i], so it is the
// truth for block[i + 1].
int accepted_prefix(const std::vector<int>& block, const std::vector<int>& verified) {
    int n = 0;
    for (size_t i = 0; i + 1 < block.size() && i < verified.size(); ++i) {
        if (block[i + 1] != verified[i]) break;
        ++n;
    }
    return n;
}

struct Trial {
    const char* name;
    int accepted = 0;
    int offered = 0;
    double rate() const { return offered > 0 ? static_cast<double>(accepted) / offered : 0.0; }
};

// Comma-separated token ids, or `fallback` when the argument is absent.
std::vector<int> parse_ids(const char* s, const std::vector<int>& fallback) {
    if (s == nullptr || *s == '\0') return fallback;
    std::vector<int> out;
    const char* p = s;
    while (*p != '\0') {
        char* end = nullptr;
        const long v = std::strtol(p, &end, 10);
        if (end == p) break;
        out.push_back(static_cast<int>(v));
        p = end;
        while (*p == ',' || *p == ' ') ++p;
    }
    return out.empty() ? fallback : out;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0]
                  << " <ckpt_dir> [rounds=6] [layers=43] [tp_world=1] [tp_rank=0]"
                     " [nccl_id_path] [prompt_ids]\n";
        return 2;
    }
    const std::string ckpt_dir = argv[1];
    const int rounds = argc > 2 ? std::atoi(argv[2]) : 6;
    const int layer_count = argc > 3 ? std::atoi(argv[3]) : 43;

    ForwardSmokeOptions opts;
    opts.tp_world = argc > 4 ? std::atoi(argv[4]) : 1;
    opts.tp_rank = argc > 5 ? std::atoi(argv[5]) : 0;
    opts.device = opts.tp_rank;  // one rank per card, as main.cpp defaults
    if (argc > 6) opts.nccl_id_path = argv[6];
    if (opts.tp_world > 1 && opts.nccl_id_path.empty()) {
        std::cerr << "tp_world > 1 needs an nccl_id_path\n";
        return 2;
    }
    const bool verbose = opts.tp_rank == 0;

    // Selecting the device is the caller's job (main.cpp does the same); skip
    // it and every rank allocates onto GPU 0, which OOMs after two ranks and
    // looks like the draft being too large rather than a placement bug.
    if (cudaSetDevice(opts.device) != cudaSuccess) {
        std::cerr << "failed to set CUDA device " << opts.device << "\n";
        return 2;
    }

    report_mem("start", opts.tp_rank);
    PersistentEngine engine(ckpt_dir, opts, layer_count, 2048);
    report_mem("main model", opts.tp_rank);
    engine.load_dspark(ckpt_dir);
    report_mem("draft loaded", opts.tp_rank);
    if (!engine.dspark_loaded()) {
        std::cout << "[FAIL] load_dspark did not take effect\n";
        return 1;
    }

    SamplingParams sp;
    sp.greedy = true;
    sp.temperature = 1.0f;
    sp.seed = 12345;

    // A cyclic default: the main model's continuation is near-deterministic
    // here, which is what makes an absolute accept rate readable. See the
    // header for why an ordinary-text prompt cannot serve as the pass bar.
    const std::vector<int> prompt = parse_ids(
        argc > 7 ? argv[7] : nullptr,
        {16, 18, 16, 18, 16, 18, 16, 18, 16, 18, 16, 18});

    Trial real{"real"}, zeros{"zeros"}, stale{"stale"};
    std::vector<float> prev_hidden;
    int block_len = 0;

    // Context grows by the accepted prefix each round; every round re-prefills
    // it so the KV never holds a rejected draft.
    std::vector<int> ctx = prompt;

    for (int r = 0; r < rounds; ++r) {
        engine.reset_session();
        const int committed = engine.prefill(ctx, sp);
        const std::vector<float>& captured = engine.last_dspark_hidden();
        const int n_pos = engine.last_dspark_hidden_positions();
        const int pos = static_cast<int>(ctx.size());
        if (captured.empty() || n_pos <= 0) {
            fail("load_dspark did not enable capture");
            break;
        }

        // prefill() primes every captured prompt position into the draft ring
        // when DSpark is loaded; writing them again would duplicate work.

        // Seed the draft from the last captured position -- the one the
        // committed token was predicted *from*, at pos - 1. draft_tokens takes
        // that position, not the committed token's own: the committed token
        // goes into draft slot 0, which the draft ropes at start_pos + 1.
        // Passing `pos` instead shifts every draft position by one and adds a
        // bogus ring slot, which still drafts fluently and matches nothing.
        const size_t stride = captured.size() / static_cast<size_t>(n_pos);
        const std::vector<float> hidden(captured.end() - stride, captured.end());

        const dspark::DraftOutput out = engine.draft_tokens(committed, pos - 1, hidden);
        if (r == 0) {
            block_len = static_cast<int>(out.tokens.size());
            if (verbose) {
                std::cout << "hidden positions=" << n_pos << " stride=" << stride
                          << " block=" << out.tokens.size()
                          << " confidence=" << out.confidence.size() << "\n";
            }
            if (out.tokens.size() != out.confidence.size() + 1) {
                fail("tokens should be one longer than confidence (the seed token)");
            }
        }
        if (out.tokens.empty() || out.tokens[0] != committed) {
            fail("draft output does not start with the committed token");
            break;
        }
        bool sane = true;
        for (int t : out.tokens) sane = sane && (t >= 0 && t < 129280);
        for (float c : out.confidence) sane = sane && std::isfinite(c);
        if (!sane) {
            fail("draft produced out-of-range tokens or non-finite confidence");
            break;
        }

        // Truth for this block. Verifying [committed, drafts...] at `pos` keeps
        // every draft on the position it was drafted for.
        const std::vector<int> verified = engine.verify_step(out.tokens, pos, sp);
        const int n_ok = accepted_prefix(out.tokens, verified);
        real.accepted += n_ok;
        real.offered += static_cast<int>(out.tokens.size()) - 1;

        // Corrupted seeds, same committed token, position, and primed ring,
        // scored against the same truth: the seed hidden is the only thing that
        // differs. Reported for the record, not asserted on -- see the header
        // for why they cannot discriminate on a prompt this predictable.
        {
            const std::vector<float> z(hidden.size(), 0.0f);
            const dspark::DraftOutput o = engine.draft_tokens(committed, pos - 1, z);
            zeros.accepted += accepted_prefix(o.tokens, verified);
            zeros.offered += static_cast<int>(o.tokens.size()) - 1;
        }
        if (!prev_hidden.empty()) {
            const dspark::DraftOutput o = engine.draft_tokens(committed, pos - 1, prev_hidden);
            stale.accepted += accepted_prefix(o.tokens, verified);
            stale.offered += static_cast<int>(o.tokens.size()) - 1;
        }

        if (verbose) {
            std::cout << "  round " << r << " pos=" << pos << " accepted=" << n_ok
                      << "/" << out.tokens.size() - 1 << "\n";
            std::cout << "    block   =";
            for (int t : out.tokens) std::cout << " " << t;
            std::cout << "\n    verified=";
            for (int t : verified) std::cout << " " << t;
            std::cout << "\n    conf    =";
            for (float c : out.confidence) std::cout << " " << c;
            std::cout << "\n" << std::flush;
        }

        prev_hidden = hidden;
        // Commit the accepted prefix plus the bonus token, as a real
        // speculative loop would -- so the context advances even when nothing
        // was accepted and the rounds do not repeat the same position.
        ctx.push_back(committed);
        for (int i = 0; i < n_ok; ++i) ctx.push_back(out.tokens[i + 1]);
    }

    std::cout << "\nrank " << opts.tp_rank << " seed  accepted/offered   rate\n";
    for (const Trial* t : {&real, &zeros, &stale}) {
        std::cout << "  " << t->name << "\t" << t->accepted << "/" << t->offered
                  << "\t\t" << t->rate() << "\n";
    }

    if (block_len <= 1) fail("draft produced no tokens");
    if (real.offered == 0) fail("no draft rounds completed");

    // The bar is an absolute accept rate on a prompt the main model is
    // near-certain about. 0.5 sits between the two regimes this build actually
    // produces -- 1.0 on the cyclic default (5/5 every round) and 0.0 when the
    // seed or the ring is not reaching the draft -- so it separates working from
    // broken with room for a shorter prompt's slow start, without pinning the
    // test to the exact number.
    //
    // The corruption trials are printed but not asserted on: on a prompt this
    // predictable the ring and the committed token already determine the
    // continuation, so a zeroed seed scores the same. That the seed matters is
    // established by test_dspark_draft_sensitivity, which isolates it properly.
    constexpr double kMinAcceptRate = 0.5;
    if (real.rate() < kMinAcceptRate) {
        fail("accept rate " + std::to_string(real.rate()) + " below " +
             std::to_string(kMinAcceptRate) + " on a near-deterministic prompt -- "
             "the draft is not tracking the main model");
    }

    if (failures != 0) {
        std::cout << "[FAIL] dspark_draft: " << failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "[PASS] dspark_draft\n";
    return 0;
}
