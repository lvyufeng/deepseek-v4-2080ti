// DSpark main-hidden capture test.
//
// The draft module is seeded from the main model's block output at a few late
// layers, mean-pooled over the hc dimension and concatenated. Every way of
// getting that wrong still produces a finite [n_target * dim] vector of
// plausible magnitude -- a wrong layer, a sum instead of a mean, a wrong
// position, or slots written in the wrong order all look correct from the
// outside and only show up as a quietly worse accept rate. So this checks the
// pieces separately:
//
//   Part A (no checkpoint): hc_mean_pool_rows_cuda against a CPU mean. Catches
//     the scale factor (mean vs sum is a clean 4x), the strided destination,
//     and row indexing, exactly rather than within a tolerance.
//
//   Part B (needs a checkpoint):
//     B1  capture off leaves the hidden empty and does not perturb the tokens
//     B2  slot k of a multi-layer capture equals a single-layer capture of that
//         layer -- i.e. the layer index is honoured and the slots are ordered
//     B3  the target layers are actually distinct, so B2 is not vacuous
//     B4  verify_step's per-draft-token rows match plain decode's hiddens at
//         the same positions (verify runs the identical per-token forward)
//     B5  prefill captures the *last* prompt position, checked against a decode
//         step landing on that same position, with a wrong position reported
//         alongside as the discrimination margin
//     B6  a windowed prefill capture returns the right count, in position
//         order, with the last row still equal to the single-position capture
//
//   test_dspark_hidden_capture [ckpt_dir] [layers=3]
//
// With no ckpt_dir only Part A runs.

#include "cuda_ops.hpp"
#include "persistent_engine.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <cstring>
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

double rel_l2(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) return INFINITY;
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        num += d * d;
        den += static_cast<double>(b[i]) * static_cast<double>(b[i]);
    }
    if (den == 0.0) return num == 0.0 ? 0.0 : INFINITY;
    return std::sqrt(num / den);
}

float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float m = 0.0f;
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
        m = std::max(m, std::fabs(a[i] - b[i]));
    }
    return m;
}

// Slot `slot` of a [n_target * dim] concatenated hidden.
std::vector<float> slot_of(const std::vector<float>& h, int slot, int dim) {
    const size_t off = static_cast<size_t>(slot) * dim;
    if (h.size() < off + dim) return {};
    return std::vector<float>(h.begin() + off, h.begin() + off + dim);
}

void part_a() {
    std::cout << "Part A: hc_mean_pool_rows_cuda vs CPU mean\n";

    const int rows = 3;
    const int dim = 37;          // deliberately not a multiple of the block size
    const int n_target = 4;
    const int out_stride = n_target * dim;
    const int col_offset = 2 * dim;

    std::vector<float> h4(static_cast<size_t>(rows) * 4 * dim);
    for (size_t i = 0; i < h4.size(); ++i) {
        h4[i] = std::sin(static_cast<float>(i) * 0.37f) * 3.0f;
    }

    float* d_h4 = nullptr;
    float* d_out = nullptr;
    cudaMalloc(&d_h4, h4.size() * sizeof(float));
    cudaMalloc(&d_out, static_cast<size_t>(rows) * out_stride * sizeof(float));
    cudaMemcpy(d_h4, h4.data(), h4.size() * sizeof(float), cudaMemcpyHostToDevice);
    // Poison the destination so an unwritten slot cannot pass as a zero.
    cudaMemset(d_out, 0x7f, static_cast<size_t>(rows) * out_stride * sizeof(float));

    if (!hc_mean_pool_rows_cuda(d_h4, d_out, rows, dim, out_stride, col_offset)) {
        fail("hc_mean_pool_rows_cuda launch failed");
        cudaFree(d_h4);
        cudaFree(d_out);
        return;
    }
    cudaDeviceSynchronize();

    std::vector<float> got(static_cast<size_t>(rows) * out_stride);
    cudaMemcpy(got.data(), d_out, got.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // Rejections: an offset that would run past the row. Checked while the
    // buffers are still alive so a wrapper that dereferenced first would fault
    // here rather than silently reading freed memory.
    const bool rejected = !hc_mean_pool_rows_cuda(d_h4, d_out, rows, dim, dim, dim);
    std::cout << "  out-of-range col_offset rejected=" << (rejected ? "yes" : "no") << "\n";
    if (!rejected) fail("hc_mean_pool_rows_cuda accepted an out-of-range col_offset");

    cudaFree(d_h4);
    cudaFree(d_out);

    int wrong = 0;
    float worst = 0.0f;
    for (int r = 0; r < rows; ++r) {
        for (int d = 0; d < dim; ++d) {
            float acc = 0.0f;
            for (int h = 0; h < 4; ++h) {
                acc += h4[(static_cast<size_t>(r) * 4 + h) * dim + d];
            }
            const float want = acc * 0.25f;
            const float have = got[static_cast<size_t>(r) * out_stride + col_offset + d];
            worst = std::max(worst, std::fabs(want - have));
            if (want != have) ++wrong;
        }
    }
    std::cout << "  pooled cells mismatched=" << wrong << " max_abs=" << worst << "\n";
    if (wrong != 0) fail("pooled values disagree with a CPU mean (sum instead of mean?)");

    // Everything outside the written column slice must still hold the poison.
    const uint32_t poison_bits = 0x7f7f7f7fu;
    float poison;
    std::memcpy(&poison, &poison_bits, sizeof(float));
    int clobbered = 0;
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < out_stride; ++c) {
            if (c >= col_offset && c < col_offset + dim) continue;
            const float v = got[static_cast<size_t>(r) * out_stride + c];
            if (std::memcmp(&v, &poison, sizeof(float)) != 0) ++clobbered;
        }
    }
    std::cout << "  cells written outside the slice=" << clobbered << "\n";
    if (clobbered != 0) fail("pooling wrote outside its column slice");
}

void part_b(const std::string& ckpt_dir, int layer_count) {
    std::cout << "Part B: engine capture, ckpt=" << ckpt_dir
              << " layers=" << layer_count << "\n";

    ForwardSmokeOptions opts;
    opts.tp_world = 1;
    opts.tp_rank = 0;
    opts.device = 0;

    PersistentEngine engine(ckpt_dir, opts, layer_count, 2048);

    SamplingParams sp;
    sp.greedy = true;
    sp.temperature = 1.0f;
    sp.seed = 12345;

    const std::vector<int> prompt = {1, 15043, 5312};
    const int steps = 4;

    // --- B1: capture off ---
    engine.reset_session();
    int token = engine.prefill(prompt, sp);
    std::vector<int> tokens_off = {token};
    if (!engine.last_dspark_hidden().empty()) {
        fail("hidden is non-empty with capture off");
    }
    for (int i = 1; i < steps; ++i) {
        token = engine.decode_step(token, static_cast<int>(prompt.size()) + i - 1, sp);
        tokens_off.push_back(token);
    }

    // Capture the last `n_target` layers, the ones the draft actually reads.
    const int n_target = std::min(3, layer_count);
    std::vector<int> targets;
    for (int i = n_target; i >= 1; --i) targets.push_back(layer_count - i);
    engine.set_dspark_capture_layers(targets);

    engine.reset_session();
    token = engine.prefill(prompt, sp);
    std::vector<int> tokens_on = {token};
    const std::vector<float> prefill_hidden = engine.last_dspark_hidden();
    if (prefill_hidden.empty()) {
        fail("hidden is empty with capture on");
        return;
    }
    const int dim = static_cast<int>(prefill_hidden.size()) / n_target;
    std::cout << "  n_target=" << n_target << " dim=" << dim
              << " hidden_len=" << prefill_hidden.size() << "\n";
    if (static_cast<int>(prefill_hidden.size()) != n_target * dim) {
        fail("hidden length is not n_target * dim");
    }
    bool finite = true;
    for (float v : prefill_hidden) finite = finite && std::isfinite(v);
    if (!finite) fail("prefill hidden is not finite");

    std::vector<std::vector<float>> decode_hidden;
    for (int i = 1; i < steps; ++i) {
        token = engine.decode_step(token, static_cast<int>(prompt.size()) + i - 1, sp);
        tokens_on.push_back(token);
        decode_hidden.push_back(engine.last_dspark_hidden());
    }

    int token_diffs = 0;
    for (int i = 0; i < steps; ++i) {
        if (tokens_off[i] != tokens_on[i]) ++token_diffs;
    }
    std::cout << "  tokens with capture on vs off: differ=" << token_diffs << "\n";
    if (token_diffs != 0) fail("capture perturbed the token stream");

    // --- B2/B3: one layer at a time, compared slot by slot ---
    std::vector<std::vector<float>> single;
    for (int li : targets) {
        engine.set_dspark_capture_layers({li});
        engine.reset_session();
        (void)engine.prefill(prompt, sp);
        single.push_back(engine.last_dspark_hidden());
        if (static_cast<int>(single.back().size()) != dim) {
            fail("single-layer capture is not dim-sized");
            return;
        }
    }

    for (int k = 0; k < n_target; ++k) {
        const std::vector<float> got = slot_of(prefill_hidden, k, dim);
        const float m = max_abs_diff(got, single[k]);
        std::cout << "  slot " << k << " (layer " << targets[k] << ") vs single-layer:"
                  << " max_abs=" << m << "\n";
        if (m != 0.0f) fail("slot does not match a single-layer capture of that layer");
    }

    for (int k = 1; k < n_target; ++k) {
        const double r = rel_l2(single[k], single[0]);
        std::cout << "  layer " << targets[k] << " vs layer " << targets[0]
                  << " rel_l2=" << r << "\n";
        if (!(r > 1e-3)) fail("target layers are indistinguishable; the slot check is vacuous");
    }

    // --- B4: verify_step rows vs plain decode at the same positions ---
    engine.set_dspark_capture_layers(targets);
    engine.reset_session();
    const int first = engine.prefill(prompt, sp);
    std::vector<int> draft;
    draft.push_back(first);
    for (int i = 0; i + 1 < static_cast<int>(tokens_on.size()) && static_cast<int>(draft.size()) < steps - 1; ++i) {
        draft.push_back(tokens_on[i + 1]);
    }
    (void)engine.verify_step(draft, static_cast<int>(prompt.size()), sp);
    const std::vector<float>& vh = engine.last_verify_dspark_hidden();
    const size_t want_len = draft.size() * static_cast<size_t>(n_target) * dim;
    std::cout << "  verify hidden rows=" << draft.size()
              << " len=" << vh.size() << " want=" << want_len << "\n";
    if (vh.size() != want_len) {
        fail("verify hidden is not [draft_len, n_target * dim]");
    } else {
        for (size_t i = 0; i < draft.size() && i < decode_hidden.size(); ++i) {
            const size_t off = i * static_cast<size_t>(n_target) * dim;
            const std::vector<float> row(vh.begin() + off,
                                         vh.begin() + off + static_cast<size_t>(n_target) * dim);
            const float m = max_abs_diff(row, decode_hidden[i]);
            std::cout << "    verify row " << i << " vs decode: max_abs=" << m << "\n";
            if (m != 0.0f) {
                fail("verify hidden disagrees with plain decode at the same position");
            }
        }
    }

    // --- B5: prefill captures the last prompt position ---
    // Prefill the prompt minus its last token, then decode that token: the
    // forward now ends at the same position the full prefill ended at. The
    // batched and per-token paths drift (docs/dspark.md), so this is a loose
    // bound -- what makes it a real check is the wrong-position comparison
    // printed beside it, which is O(1) rather than O(1e-2).
    std::vector<int> head(prompt.begin(), prompt.end() - 1);
    engine.reset_session();
    (void)engine.prefill(head, sp);
    (void)engine.decode_step(prompt.back(), static_cast<int>(head.size()), sp);
    const std::vector<float> same_pos = engine.last_dspark_hidden();

    const double r_same = rel_l2(same_pos, prefill_hidden);
    const double r_wrong = decode_hidden.empty() ? INFINITY
                                                 : rel_l2(decode_hidden[0], prefill_hidden);
    std::cout << "  prefill hidden vs same position rel_l2=" << r_same
              << ", vs the next position rel_l2=" << r_wrong << "\n";
    if (!(r_same < 5e-2)) {
        fail("prefill did not capture the last prompt position");
    }
    if (!(r_wrong > 10.0 * r_same)) {
        fail("positions are not separable; the position check proves nothing");
    }

    // --- B6: windowed prefill capture ---
    // load_dspark() keeps the last window_size positions rather than one,
    // because priming the draft's attention ring needs every committed
    // position. The rows must come back in position order and the last one must
    // still be the one the single-position capture produced -- an off-by-one or
    // a reversed window would look identical in shape and magnitude.
    const int window = 4;
    engine.set_dspark_capture_layers(targets, window);
    engine.reset_session();
    (void)engine.prefill(prompt, sp);
    const std::vector<float> windowed = engine.last_dspark_hidden();
    const int n_pos = engine.last_dspark_hidden_positions();
    const size_t stride = static_cast<size_t>(n_target) * dim;
    const int want_pos = std::min<int>(window, static_cast<int>(prompt.size()));
    std::cout << "  windowed prefill positions=" << n_pos << " want=" << want_pos
              << " len=" << windowed.size() << "\n";
    if (n_pos != want_pos) fail("windowed prefill kept the wrong number of positions");
    if (windowed.size() != static_cast<size_t>(n_pos) * stride) {
        fail("windowed prefill length is not positions * n_target * dim");
    } else {
        // Last row == the single-position capture of the same prompt.
        const std::vector<float> last(windowed.end() - stride, windowed.end());
        const float m = max_abs_diff(last, prefill_hidden);
        std::cout << "  last windowed row vs single-position capture: max_abs=" << m << "\n";
        if (m != 0.0f) fail("the last windowed row is not the final prompt position");

        // Rows must be distinct and in order: row i is position
        // prompt.size() - n_pos + i. Comparing the first row against the last
        // catches a window written backwards, which a length check cannot.
        if (n_pos >= 2) {
            const std::vector<float> first(windowed.begin(), windowed.begin() + stride);
            const double r = rel_l2(first, last);
            std::cout << "  first windowed row vs last: rel_l2=" << r << "\n";
            if (!(r > 1e-3)) fail("windowed rows are not distinct positions");
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    const std::string ckpt_dir = argc > 1 ? argv[1] : "";
    const int layer_count = argc > 2 ? std::atoi(argv[2]) : 3;

    part_a();
    if (!ckpt_dir.empty()) {
        part_b(ckpt_dir, layer_count);
    } else {
        std::cout << "Part B skipped (no ckpt_dir given)\n";
    }

    if (failures != 0) {
        std::cout << "[FAIL] dspark_hidden_capture: " << failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "[PASS] dspark_hidden_capture\n";
    return 0;
}
