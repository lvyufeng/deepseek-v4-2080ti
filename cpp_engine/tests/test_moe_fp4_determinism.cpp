// Guards the fixed-order reduction in the FP4 prefill MoE against regression.
//
// The grouped w2 kernels used to atomicAdd each route's contribution straight
// into y. Float addition is not associative and blocks finish in whatever order
// the scheduler picks, so a token whose routes land in a different order gets a
// different sum -- with topk>=3 that diverges on essentially every run. The fix
// has each route write its own row of a [routes, dim] scratch buffer, then sums
// a token's routes in router-slot order, which is fixed.
//
// The test runs the same grouped MoE call repeatedly on identical inputs and
// compares bitwise. With POCKETLLM_CPP_MOE_DETERMINISTIC_REDUCE=1 (the default) every
// run must be identical for every topk -- that is the guard.
//
// Setting it to 0 restores the atomic path, and the test then requires that at
// least one topk>=3 case diverges. That direction is deliberately weak: whether
// a particular shape diverges depends on how the scheduler orders blocks, and at
// these sizes topk=3 is often stable while topk=6 is not. Asserting divergence
// per case would encode a scheduler accident; asserting it somewhere is what
// proves the test can still observe the bug it guards against. topk<=2 is stable
// either way -- two addends admit only one order.
//
//   test_moe_fp4_determinism            # the guard: all topk bitwise stable
//   POCKETLLM_CPP_MOE_DETERMINISTIC_REDUCE=0 test_moe_fp4_determinism   # must still see the bug
//
// Weights are random bytes rather than a real checkpoint: reduction order does
// not care what the numbers mean, and this keeps the test free of checkpoint
// dependencies.

#include "cuda_ops.hpp"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <vector>

#include <cuda_runtime.h>

namespace {

int failures = 0;

void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << what << ": " << cudaGetErrorString(err) << "\n";
        std::exit(1);
    }
}

// Small enough to run in a few seconds, large enough that a token's routes land
// in several different blocks -- which is what makes the atomic order vary.
constexpr int kDim = 1024;
constexpr int kInterDim = 1408;  // multiple of 32 for the FP4 scale layout
constexpr int kExperts = 32;
constexpr int kTokens = 256;
constexpr int kIters = 20;

struct DeviceBuffers {
    uint8_t *w1q = nullptr, *w1s = nullptr;
    uint8_t *w2q = nullptr, *w2s = nullptr;
    uint8_t *w3q = nullptr, *w3s = nullptr;
    float* x = nullptr;
    int64_t* indices = nullptr;
    float* weights = nullptr;
    int64_t* route_tokens = nullptr;
    float* route_weights = nullptr;
    int32_t *seg_starts = nullptr, *counts = nullptr, *offsets = nullptr;
    int32_t *total_routes = nullptr, *slot_routes = nullptr;
    float* y = nullptr;

    ~DeviceBuffers() {
        for (void* p : {(void*)w1q, (void*)w1s, (void*)w2q, (void*)w2s, (void*)w3q,
                        (void*)w3s, (void*)x, (void*)indices, (void*)weights,
                        (void*)route_tokens, (void*)route_weights, (void*)seg_starts,
                        (void*)counts, (void*)offsets, (void*)total_routes,
                        (void*)slot_routes, (void*)y}) {
            if (p != nullptr) cudaFree(p);
        }
    }
};

template <typename T>
void alloc_and_fill(T** dst, size_t count, std::mt19937& rng, bool random_bytes) {
    check_cuda(cudaMalloc(dst, count * sizeof(T)), "cudaMalloc");
    if (!random_bytes) return;
    std::vector<uint8_t> host(count * sizeof(T));
    for (auto& b : host) b = static_cast<uint8_t>(rng() & 0xff);
    check_cuda(cudaMemcpy(*dst, host.data(), host.size(), cudaMemcpyHostToDevice), "cudaMemcpy");
}

// Returns the number of runs (out of kIters) that differed from the first.
int run_topk(int topk, DeviceBuffers& buf, std::mt19937& rng, int& routes_out) {
    const int max_routes = kTokens * topk;

    // Router picks: spread across experts so routes for one token land in
    // different expert segments, i.e. different blocks.
    std::vector<int64_t> h_indices(max_routes);
    std::vector<float> h_weights(max_routes, 1.0f / static_cast<float>(topk));
    for (int t = 0; t < kTokens; ++t) {
        for (int k = 0; k < topk; ++k) {
            h_indices[t * topk + k] = (t * 7 + k * 11) % kExperts;
        }
    }
    check_cuda(cudaMemcpy(buf.indices, h_indices.data(), h_indices.size() * sizeof(int64_t),
                          cudaMemcpyHostToDevice), "copy indices");
    check_cuda(cudaMemcpy(buf.weights, h_weights.data(), h_weights.size() * sizeof(float),
                          cudaMemcpyHostToDevice), "copy weights");

    std::vector<float> h_x(static_cast<size_t>(kTokens) * kDim);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& v : h_x) v = dist(rng);
    check_cuda(cudaMemcpy(buf.x, h_x.data(), h_x.size() * sizeof(float), cudaMemcpyHostToDevice),
               "copy x");

    if (!pocket::moe_group_routes_cuda(buf.indices, buf.weights, buf.route_tokens,
                                     buf.route_weights, buf.seg_starts, buf.counts, buf.offsets,
                                     buf.total_routes, kTokens, topk, 0, kExperts,
                                     buf.slot_routes)) {
        std::cerr << "moe_group_routes_cuda failed\n";
        std::exit(1);
    }

    int routes = 0;
    check_cuda(cudaMemcpy(&routes, buf.total_routes, sizeof(int), cudaMemcpyDeviceToHost),
               "copy total_routes");
    routes_out = routes;

    std::vector<int32_t> h_counts(kExperts);
    check_cuda(cudaMemcpy(h_counts.data(), buf.counts, h_counts.size() * sizeof(int32_t),
                          cudaMemcpyDeviceToHost), "copy counts");
    int max_count = 0;
    for (int c : h_counts) max_count = std::max(max_count, c);

    pocket::MoePrefillFp4GroupedWorkspace ws;
    // The standalone moe_prefill_fp4_grouped_cuda wrapper hardcodes
    // d_token_slot_routes=nullptr and so can never take the deterministic path;
    // this test has to build the workspace itself to pass the slot map through.
    ws.dim = kDim;
    ws.inter_dim = kInterDim;
    ws.routes_cap = routes;
    ws.padded_rows_cap = kExperts * max_count;
    const size_t routes_dim = static_cast<size_t>(routes) * kDim;
    const size_t padded_dim = static_cast<size_t>(ws.padded_rows_cap) * kDim;
    const size_t padded_inter = static_cast<size_t>(ws.padded_rows_cap) * kInterDim;
    check_cuda(cudaMalloc(&ws.d_x_sorted, routes_dim * sizeof(float)), "alloc x_sorted");
    check_cuda(cudaMalloc(&ws.d_partials, routes_dim * sizeof(float)), "alloc partials");
    check_cuda(cudaMalloc(&ws.d_x_q, routes_dim), "alloc x_q");
    check_cuda(cudaMalloc(&ws.d_x_scale, static_cast<size_t>(routes) * sizeof(float)), "alloc x_scale");
    check_cuda(cudaMalloc(&ws.d_x_pad, padded_dim), "alloc x_pad");
    check_cuda(cudaMalloc(&ws.d_x_scale_pad, static_cast<size_t>(ws.padded_rows_cap) * sizeof(float)), "alloc x_scale_pad");
    check_cuda(cudaMalloc(&ws.d_gate, padded_inter * sizeof(float)), "alloc gate");
    check_cuda(cudaMalloc(&ws.d_up, padded_inter * sizeof(float)), "alloc up");
    check_cuda(cudaMalloc(&ws.d_hidden_q, padded_inter), "alloc hidden_q");
    check_cuda(cudaMalloc(&ws.d_hidden_scale, static_cast<size_t>(ws.padded_rows_cap) * sizeof(float)), "alloc hidden_scale");

    std::vector<float> first(static_cast<size_t>(kTokens) * kDim);
    std::vector<float> current(first.size());
    int diverged = 0;

    for (int iter = 0; iter < kIters; ++iter) {
        check_cuda(cudaMemset(buf.y, 0, first.size() * sizeof(float)), "memset y");
        if (!pocket::moe_prefill_fp4_grouped_cuda_with_workspace(
                buf.x, buf.route_tokens, buf.route_weights, buf.seg_starts, buf.w1q, buf.w1s,
                buf.w2q, buf.w2s, buf.w3q, buf.w3s, buf.y, kTokens, topk, routes, kExperts,
                max_count, kDim, kInterDim, 6.0f, ws, buf.slot_routes)) {
            std::cerr << "moe_prefill_fp4_grouped_cuda_with_workspace failed\n";
            std::exit(1);
        }
        check_cuda(cudaMemcpy(current.data(), buf.y, current.size() * sizeof(float),
                              cudaMemcpyDeviceToHost), "copy y");
        if (iter == 0) {
            first = current;
        } else if (std::memcmp(first.data(), current.data(), first.size() * sizeof(float)) != 0) {
            ++diverged;
        }
    }

    for (void* p : {(void*)ws.d_x_sorted, (void*)ws.d_partials, (void*)ws.d_x_q,
                    (void*)ws.d_x_scale, (void*)ws.d_x_pad, (void*)ws.d_x_scale_pad,
                    (void*)ws.d_gate, (void*)ws.d_up, (void*)ws.d_hidden_q,
                    (void*)ws.d_hidden_scale}) {
        cudaFree(p);
    }
    return diverged;
}

}  // namespace

int main() {
    const char* env = std::getenv("POCKETLLM_CPP_MOE_DETERMINISTIC_REDUCE");
    // Matches the default in fp4_ops.cu: unset means on.
    const bool deterministic = (env == nullptr) || (std::atoi(env) != 0);

    std::cout << "POCKETLLM_CPP_MOE_DETERMINISTIC_REDUCE=" << (env ? env : "(unset)")
              << "  -> expecting " << (deterministic ? "fixed-order reduction" : "atomics")
              << "\n";

    std::mt19937 rng(20260810);
    DeviceBuffers buf;
    const size_t w_q = static_cast<size_t>(kExperts) * kDim * (kInterDim / 2);
    const size_t w_s = static_cast<size_t>(kExperts) * kDim * (kInterDim / 32);
    alloc_and_fill(&buf.w1q, w_q, rng, true);
    alloc_and_fill(&buf.w2q, w_q, rng, true);
    alloc_and_fill(&buf.w3q, w_q, rng, true);
    // Scales are e8m0; keep them mid-range so products stay finite.
    for (uint8_t** s : {&buf.w1s, &buf.w2s, &buf.w3s}) {
        check_cuda(cudaMalloc(s, w_s), "cudaMalloc scale");
        check_cuda(cudaMemset(*s, 127, w_s), "memset scale");
    }

    const int max_topk = 6;
    alloc_and_fill(&buf.x, static_cast<size_t>(kTokens) * kDim, rng, false);
    alloc_and_fill(&buf.indices, static_cast<size_t>(kTokens) * max_topk, rng, false);
    alloc_and_fill(&buf.weights, static_cast<size_t>(kTokens) * max_topk, rng, false);
    alloc_and_fill(&buf.route_tokens, static_cast<size_t>(kTokens) * max_topk, rng, false);
    alloc_and_fill(&buf.route_weights, static_cast<size_t>(kTokens) * max_topk, rng, false);
    alloc_and_fill(&buf.slot_routes, static_cast<size_t>(kTokens) * max_topk, rng, false);
    alloc_and_fill(&buf.seg_starts, static_cast<size_t>(kExperts) + 1, rng, false);
    alloc_and_fill(&buf.counts, kExperts, rng, false);
    alloc_and_fill(&buf.offsets, kExperts, rng, false);
    alloc_and_fill(&buf.total_routes, 1, rng, false);
    alloc_and_fill(&buf.y, static_cast<size_t>(kTokens) * kDim, rng, false);

    int atomic_divergent_cases = 0;
    for (int topk : {1, 2, 3, 6}) {
        int routes = 0;
        const int diverged = run_topk(topk, buf, rng, routes);

        std::cout << "  topk=" << topk << " routes=" << routes << " diverged=" << diverged << "/"
                  << (kIters - 1);
        if (deterministic) {
            // The actual guard: the fix must make every topk bitwise stable.
            if (diverged == 0) {
                std::cout << "  [ok]";
            } else {
                std::cout << "  [FAIL] (fixed-order reduction must be bitwise stable)";
                ++failures;
            }
        } else {
            // Atomics: whether a given shape diverges depends on how the scheduler
            // happens to order blocks, so a per-case assertion would encode an
            // accident. topk<=2 is stable by construction; for topk>=3 we only
            // require that at least one case diverges (checked after the loop),
            // which is what proves the test can see the bug at all.
            if (topk >= 3 && diverged > 0) ++atomic_divergent_cases;
            std::cout << "  (informational)";
        }
        std::cout << "\n";
    }

    if (!deterministic && atomic_divergent_cases == 0) {
        std::cout
            << "\n[FAIL] atomic path produced identical results everywhere -- this test can no\n"
               "       longer observe the bug it guards against (shapes too small, or the\n"
               "       atomic path was removed). Re-tune kTokens/kExperts before trusting it.\n";
        ++failures;
    }

    if (failures != 0) {
        std::cout << "\nFAIL: " << failures << " case(s)\n";
        return 1;
    }
    std::cout << "\nPASS\n";
    return 0;
}
