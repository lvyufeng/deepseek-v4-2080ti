// Is it safe to overwrite a rejected draft's state by forwarding over it?
//
// A speculative round verifies [committed, d0, d1, ...] and may accept only a
// prefix. The rejected tokens have already been forwarded, so their state is in
// the caches. The scheduler's plan is to snapshot the compressor accumulator
// before verify, then restore it after a reject, so the next round's forward
// from the accepted position sees clean state. The KV ring addresses by
// position % window_size and overwrites by construction, so it needs no
// snapshot; the compressor accumulator is the only destructive piece.
//
// The accumulator sits on layers 2..42 and updates at offset = position % ratio.
// At every ratio boundary, compressor_shift_overlap_state moves the upper half
// down and zeroes the top. That shift is destructive -- replaying a position
// that crossed a boundary cannot rebuild what was shifted away. Whether bare
// overwrite actually bites depends on how far a round can overshoot relative to
// ratio (4 and 128 here), which is a question about the code as configured, not
// about the kernel in isolation.
//
// So this measures it directly. For each overshoot length, run:
//
//   A  plain decode to position P, one token at a time            (the truth)
//   B  the same, but with a verify of `over` extra tokens injected
//      before continuing from P, bare overwrite                   (dirty)
//   C  the same, but snapshot before verify + restore after       (protected)
//
// and compare B and C against A. The comparison is exact token ids, not logits:
// a drifted compressor state that still produces the same tokens is not a
// scheduling bug, and a tolerance on hidden states would flag drift that never
// reaches the output.
//
// A control runs first: the same overshoot with layer_count small enough to
// exclude every compressed layer. If the control diverges too, the divergence
// is not the compressor and the test says so rather than blaming it.
//
//   test_verify_overwrite_safety <ckpt_dir> [layers=43] [steps=8] [tp_world=1] [tp_rank=0] [nccl_id_path]
//
// Exits non-zero if the snapshot path's tokens differ from the clean run's.

#include "persistent_engine.hpp"

#include <cuda_runtime.h>

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

// Decode `steps` tokens from `prompt`, optionally forwarding `over` junk tokens
// at position `prompt.size()` first and then continuing from there anyway --
// which is exactly what a round that had everything rejected leaves behind.
// Pass snapshot=true to test the snapshot+restore path; false to test the bare
// overwrite path (which fails without the snapshot).
std::vector<int> run(PersistentEngine& engine, const std::vector<int>& prompt,
                     int steps, int over, const SamplingParams& sp, bool snapshot) {
    engine.reset_session();
    int token = engine.prefill(prompt, sp);
    const int pos = static_cast<int>(prompt.size());

    if (over > 0) {
        if (snapshot) engine.snapshot_compressor_state();
        // A draft block the model will reject: [committed, junk...]. Verifying
        // it writes `over + 1` positions starting at pos, of which only the
        // first is real. Junk ids are fixed rather than sampled so the dirty
        // state is the same on every run.
        std::vector<int> block;
        block.push_back(token);
        for (int i = 0; i < over; ++i) block.push_back(1000 + i * 37);
        (void)engine.verify_step(block, pos, sp);
        if (snapshot) engine.restore_compressor_state();
    }

    // Continue from `pos` regardless -- overwriting whatever the verify left.
    std::vector<int> out;
    for (int i = 0; i < steps; ++i) {
        token = engine.decode_step(token, pos + i, sp);
        out.push_back(token);
    }
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0]
                  << " <ckpt_dir> [layers=43] [steps=8] [tp_world=1] [tp_rank=0]"
                     " [nccl_id_path]\n";
        return 2;
    }
    const std::string ckpt_dir = argv[1];
    const int layer_count = argc > 2 ? std::atoi(argv[2]) : 43;
    const int steps = argc > 3 ? std::atoi(argv[3]) : 8;

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

    SamplingParams sp;
    sp.greedy = true;
    sp.temperature = 1.0f;
    sp.seed = 12345;

    // Prompt length chosen so the continuation starts mid-ratio for ratio=4:
    // an overshoot from here crosses a boundary at some lengths and not others,
    // which is the distinction the test is trying to expose.
    const std::vector<int> prompt = {1, 464, 4472, 286, 5654, 9412};

    const std::vector<int> clean = run(engine, prompt, steps, 0, sp, false);
    if (verbose) std::cout << "clean (no overshoot): " << join(clean) << "\n";

    // Overshoot lengths spanning the ratio=4 boundary in both directions, plus
    // the block size a real round actually uses.
    const int overs[] = {1, 2, 3, 4, 5, 8};
    for (int over : overs) {
        const std::vector<int> dirty = run(engine, prompt, steps, over, sp, false);
        const std::vector<int> snapped = run(engine, prompt, steps, over, sp, true);
        int first_diff = -1;
        for (int i = 0; i < steps; ++i) {
            if (dirty[i] != clean[i]) { first_diff = i; break; }
        }
        int snap_diff = -1;
        for (int i = 0; i < steps; ++i) {
            if (snapped[i] != clean[i]) { snap_diff = i; break; }
        }
        if (verbose) {
            std::cout << "over=" << over << ":\n";
            std::cout << "  bare    : " << join(dirty);
            if (first_diff < 0) {
                std::cout << "  [match]\n";
            } else {
                std::cout << "  [DIVERGES at " << first_diff << "]\n";
            }
            std::cout << "  snapshot: " << join(snapped);
            if (snap_diff < 0) {
                std::cout << "  [match]\n";
            } else {
                std::cout << "  [DIVERGES at " << snap_diff << "]\n";
            }
        }
        if (snap_diff >= 0) {
            fail("overshoot of " + std::to_string(over) +
                 " diverged even with snapshot+restore (first difference at token " +
                 std::to_string(snap_diff) + ") -- the snapshot is incomplete");
        }
    }

    if (failures != 0) {
        std::cout << "[FAIL] verify_overwrite_safety: " << failures
                  << " overshoot length(s) diverged even with snapshot\n";
        return 1;
    }
    std::cout << "[PASS] verify_overwrite_safety (snapshot+restore makes rejected state safe to overwrite)\n";
    return 0;
}
