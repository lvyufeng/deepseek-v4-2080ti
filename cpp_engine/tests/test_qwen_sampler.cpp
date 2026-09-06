// Correctness tests for the device top-k -> temperature -> top-p sampler.
//
// The sampler is the only nondeterministic component in the Qwen decode path,
// so these tests pin the properties the benchmark protocol depends on: greedy
// equivalence at temperature 0, exact agreement with a CPU reference over the
// top-k set, correct nucleus truncation, and identical draws across TP ranks
// when the host supplies the uniforms.

#include "qwen_sampler.hpp"

#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

namespace {

int g_failures = 0;

void expect(bool condition, const char* what) {
    if (!condition) {
        std::printf("FAIL %s\n", what);
        ++g_failures;
    } else {
        std::printf("ok   %s\n", what);
    }
}

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        std::printf("FAIL %s: %s\n", what, cudaGetErrorString(status));
        ++g_failures;
    }
}

// The sampler entry points follow the codebase's bool-returning launch
// convention rather than returning cudaError_t.
void check(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL %s: launch reported failure\n", what);
        ++g_failures;
    }
}

struct DeviceRun {
    std::vector<int> tokens;
    std::vector<float> logits;
};

// Run the kernel over `rows` x `vocab` host logits.
DeviceRun run_sampler(const std::vector<float>& host_logits, int rows,
                      int vocab, int vocab_start, float temperature,
                      float top_p, int top_k,
                      const std::vector<float>& uniforms) {
    float* d_logits = nullptr;
    int* d_tokens = nullptr;
    float* d_out_logits = nullptr;
    float* d_uniforms = nullptr;
    pocket::DeviceRngState* d_states = nullptr;

    check(cudaMalloc(&d_logits, host_logits.size() * sizeof(float)), "malloc logits");
    check(cudaMalloc(&d_tokens, static_cast<size_t>(rows) * sizeof(int)), "malloc tokens");
    check(cudaMalloc(&d_out_logits, static_cast<size_t>(rows) * sizeof(float)), "malloc out logits");
    check(cudaMalloc(&d_states, static_cast<size_t>(rows) * pocket::qwen_sampler_rng_state_size()), "malloc states");
    check(cudaMemcpy(d_logits, host_logits.data(),
                     host_logits.size() * sizeof(float), cudaMemcpyHostToDevice),
          "copy logits");
    check(pocket::qwen_init_rng_states(d_states, rows, 1234ULL, nullptr), "init states");

    if (!uniforms.empty()) {
        check(cudaMalloc(&d_uniforms, uniforms.size() * sizeof(float)), "malloc uniforms");
        check(cudaMemcpy(d_uniforms, uniforms.data(),
                         uniforms.size() * sizeof(float), cudaMemcpyHostToDevice),
              "copy uniforms");
    }

    check(pocket::qwen_sample_top_k_top_p_rows(
              d_logits, d_tokens, d_out_logits, rows, vocab, vocab_start,
              temperature, top_p, top_k, d_states, d_uniforms, nullptr),
          "launch sampler");
    check(cudaDeviceSynchronize(), "sync sampler");

    DeviceRun out;
    out.tokens.resize(static_cast<size_t>(rows));
    out.logits.resize(static_cast<size_t>(rows));
    check(cudaMemcpy(out.tokens.data(), d_tokens,
                     static_cast<size_t>(rows) * sizeof(int), cudaMemcpyDeviceToHost),
          "copy tokens");
    check(cudaMemcpy(out.logits.data(), d_out_logits,
                     static_cast<size_t>(rows) * sizeof(float), cudaMemcpyDeviceToHost),
          "copy out logits");

    cudaFree(d_logits);
    cudaFree(d_tokens);
    cudaFree(d_out_logits);
    cudaFree(d_states);
    if (d_uniforms != nullptr) cudaFree(d_uniforms);
    return out;
}

// CPU reference: top-k, then softmax at temperature, then top-p, then inverse
// CDF against the same uniform the device consumed.
int reference_sample(const float* logits, int vocab, float temperature,
                     float top_p, int top_k, float u) {
    std::vector<int> order(static_cast<size_t>(vocab));
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        if (logits[a] != logits[b]) return logits[a] > logits[b];
        return a < b;
    });
    const int k = std::min(top_k, vocab);
    if (temperature <= 1e-5f) return order[0];

    const float inv_t = 1.0f / temperature;
    const float max_scaled = logits[order[0]] * inv_t;
    std::vector<float> probs(static_cast<size_t>(k));
    float sum = 0.0f;
    for (int i = 0; i < k; ++i) {
        probs[static_cast<size_t>(i)] = std::exp(logits[order[static_cast<size_t>(i)]] * inv_t - max_scaled);
        sum += probs[static_cast<size_t>(i)];
    }
    for (int i = 0; i < k; ++i) probs[static_cast<size_t>(i)] /= sum;

    int keep = k;
    if (top_p > 0.0f && top_p < 1.0f) {
        float cum = 0.0f;
        for (int i = 0; i < k; ++i) {
            cum += probs[static_cast<size_t>(i)];
            if (cum >= top_p) { keep = i + 1; break; }
        }
    }
    float kept = 0.0f;
    for (int i = 0; i < keep; ++i) kept += probs[static_cast<size_t>(i)];

    const float target = u * kept;
    float accum = 0.0f;
    for (int i = 0; i < keep; ++i) {
        accum += probs[static_cast<size_t>(i)];
        if (accum >= target) return order[static_cast<size_t>(i)];
    }
    return order[static_cast<size_t>(keep - 1)];
}

std::vector<float> random_logits(int rows, int vocab, unsigned seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 4.0f);
    std::vector<float> out(static_cast<size_t>(rows) * vocab);
    for (auto& v : out) v = dist(rng);
    return out;
}

void test_topk40_world_major_merge() {
    constexpr int world = 4;
    constexpr int rows = 3;
    constexpr int top_k = 40;
    const size_t local_count = static_cast<size_t>(rows) * top_k;
    std::vector<int> gathered_tokens(static_cast<size_t>(world) * local_count);
    std::vector<float> gathered_logits(gathered_tokens.size());

    for (int rank = 0; rank < world; ++rank) {
        for (int row = 0; row < rows; ++row) {
            for (int candidate = 0; candidate < top_k; ++candidate) {
                const size_t at = static_cast<size_t>(rank) * local_count +
                                  static_cast<size_t>(row) * top_k + candidate;
                gathered_tokens[at] = rank * 10000 + row * 1000 + candidate;
                gathered_logits[at] =
                    static_cast<float>((candidate * 17 + rank * 11 + row * 5) % 53);
            }
        }
    }

    // Force equal logits across ranks and distant list positions so the lower
    // global token must win independently of world-major scan order.
    gathered_logits[0 * local_count + 0 * top_k + 39] = 100.0f;
    gathered_tokens[0 * local_count + 0 * top_k + 39] = 400;
    gathered_logits[3 * local_count + 0 * top_k + 0] = 100.0f;
    gathered_tokens[3 * local_count + 0 * top_k + 0] = 17;
    gathered_logits[1 * local_count + 1 * top_k + 7] = NAN;
    gathered_tokens[2 * local_count + 2 * top_k + 4] = -1;

    std::vector<int> expected_tokens(static_cast<size_t>(rows) * top_k);
    std::vector<float> expected_logits(static_cast<size_t>(rows) * top_k);
    for (int row = 0; row < rows; ++row) {
        std::vector<std::pair<float, int>> candidates;
        for (int rank = 0; rank < world; ++rank) {
            const size_t row_base = static_cast<size_t>(rank) * local_count +
                                    static_cast<size_t>(row) * top_k;
            for (int candidate = 0; candidate < top_k; ++candidate) {
                const size_t at = row_base + candidate;
                if (gathered_tokens[at] < 0 || std::isnan(gathered_logits[at])) continue;
                candidates.emplace_back(gathered_logits[at], gathered_tokens[at]);
            }
        }
        std::sort(candidates.begin(), candidates.end(), [](const auto& left,
                                                           const auto& right) {
            if (left.first != right.first) return left.first > right.first;
            return left.second < right.second;
        });
        for (int candidate = 0; candidate < top_k; ++candidate) {
            const size_t out = static_cast<size_t>(row) * top_k + candidate;
            expected_logits[out] = candidates[static_cast<size_t>(candidate)].first;
            expected_tokens[out] = candidates[static_cast<size_t>(candidate)].second;
        }
    }

    int* d_gathered_tokens = nullptr;
    float* d_gathered_logits = nullptr;
    int* d_tokens = nullptr;
    float* d_logits = nullptr;
    check(cudaMalloc(&d_gathered_tokens,
                     gathered_tokens.size() * sizeof(int)),
          "merge malloc gathered tokens");
    check(cudaMalloc(&d_gathered_logits,
                     gathered_logits.size() * sizeof(float)),
          "merge malloc gathered logits");
    check(cudaMalloc(&d_tokens, expected_tokens.size() * sizeof(int)),
          "merge malloc output tokens");
    check(cudaMalloc(&d_logits, expected_logits.size() * sizeof(float)),
          "merge malloc output logits");
    check(cudaMemcpy(d_gathered_tokens, gathered_tokens.data(),
                     gathered_tokens.size() * sizeof(int), cudaMemcpyHostToDevice),
          "merge copy gathered tokens");
    check(cudaMemcpy(d_gathered_logits, gathered_logits.data(),
                     gathered_logits.size() * sizeof(float), cudaMemcpyHostToDevice),
          "merge copy gathered logits");
    check(pocket::qwen_merge_topk_candidates(
              d_gathered_tokens, d_gathered_logits, d_tokens, d_logits, world,
              rows, top_k, nullptr),
          "merge top-k=40 launch");
    check(cudaDeviceSynchronize(), "merge top-k=40 sync");

    std::vector<int> actual_tokens(expected_tokens.size());
    std::vector<float> actual_logits(expected_logits.size());
    check(cudaMemcpy(actual_tokens.data(), d_tokens,
                     actual_tokens.size() * sizeof(int), cudaMemcpyDeviceToHost),
          "merge copy output tokens");
    check(cudaMemcpy(actual_logits.data(), d_logits,
                     actual_logits.size() * sizeof(float), cudaMemcpyDeviceToHost),
          "merge copy output logits");
    expect(actual_tokens == expected_tokens,
           "top-k=40 merge preserves world-major rows and tie-breaks");
    expect(actual_logits == expected_logits,
           "top-k=40 merge preserves descending logits");

    cudaFree(d_gathered_tokens);
    cudaFree(d_gathered_logits);
    cudaFree(d_tokens);
    cudaFree(d_logits);
}

}  // namespace

int main() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::printf("no CUDA device available\n");
        return 1;
    }

    // Realistic shard width for TP4 over a 248,320-token vocabulary.
    const int vocab = 62080;
    const int rows = 8;

    test_topk40_world_major_merge();

    // 1. Temperature 0 must reproduce argmax exactly.
    {
        auto logits = random_logits(rows, vocab, 11);
        auto run = run_sampler(logits, rows, vocab, 0, 0.0f, 0.95f, 20, {});
        bool all_match = true;
        for (int r = 0; r < rows; ++r) {
            const float* row = logits.data() + static_cast<size_t>(r) * vocab;
            const int expected = static_cast<int>(
                std::max_element(row, row + vocab) - row);
            if (run.tokens[static_cast<size_t>(r)] != expected) all_match = false;
        }
        expect(all_match, "temperature 0 reproduces argmax");
    }

    // 2. Sampled draws must match the CPU reference bit for bit.
    {
        auto logits = random_logits(rows, vocab, 22);
        std::vector<float> uniforms{0.01f, 0.2f, 0.35f, 0.5f, 0.66f, 0.8f, 0.93f, 0.999f};
        auto run = run_sampler(logits, rows, vocab, 0, 1.0f, 0.95f, 20, uniforms);
        bool all_match = true;
        for (int r = 0; r < rows; ++r) {
            const float* row = logits.data() + static_cast<size_t>(r) * vocab;
            const int expected = reference_sample(row, vocab, 1.0f, 0.95f, 20,
                                                  uniforms[static_cast<size_t>(r)]);
            if (run.tokens[static_cast<size_t>(r)] != expected) {
                std::printf("  row %d device=%d reference=%d\n", r,
                            run.tokens[static_cast<size_t>(r)], expected);
                all_match = false;
            }
        }
        expect(all_match, "sampled draws match CPU reference");
    }

    // 3. vocab_start must be added so TP shards emit global ids.
    {
        auto logits = random_logits(1, vocab, 33);
        const int shard_base = 62080;
        auto run = run_sampler(logits, 1, vocab, shard_base, 0.0f, 1.0f, 20, {});
        const int local_argmax = static_cast<int>(
            std::max_element(logits.begin(), logits.end()) - logits.begin());
        expect(run.tokens[0] == local_argmax + shard_base,
               "vocab_start offset produces global token id");
    }

    // 4. A tight nucleus must collapse onto the argmax token.
    {
        auto logits = random_logits(rows, vocab, 44);
        std::vector<float> uniforms(static_cast<size_t>(rows), 0.999f);
        auto run = run_sampler(logits, rows, vocab, 0, 1.0f, 1e-6f, 20, uniforms);
        bool all_top = true;
        for (int r = 0; r < rows; ++r) {
            const float* row = logits.data() + static_cast<size_t>(r) * vocab;
            const int expected = static_cast<int>(
                std::max_element(row, row + vocab) - row);
            if (run.tokens[static_cast<size_t>(r)] != expected) all_top = false;
        }
        expect(all_top, "vanishing top_p keeps only the argmax token");
    }

    // 5. Host-supplied uniforms must make repeated runs identical. This is what
    //    keeps a TP group's sampled token consistent across ranks.
    {
        auto logits = random_logits(rows, vocab, 55);
        std::vector<float> uniforms{0.11f, 0.22f, 0.33f, 0.44f, 0.55f, 0.66f, 0.77f, 0.88f};
        auto a = run_sampler(logits, rows, vocab, 0, 1.0f, 0.95f, 20, uniforms);
        auto b = run_sampler(logits, rows, vocab, 0, 1.0f, 0.95f, 20, uniforms);
        expect(a.tokens == b.tokens, "host uniforms give reproducible draws");
    }

    // 6. The empirical distribution must track the intended probabilities.
    {
        const int small_vocab = 2048;
        auto logits = random_logits(1, small_vocab, 66);
        const int trials = 4000;
        std::vector<float> uniforms(static_cast<size_t>(trials));
        std::mt19937 rng(99);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        for (auto& u : uniforms) u = dist(rng);

        // Replicate the single row so each trial samples it with its own uniform.
        std::vector<float> repeated(static_cast<size_t>(trials) * small_vocab);
        for (int t = 0; t < trials; ++t) {
            std::copy(logits.begin(), logits.end(),
                      repeated.begin() + static_cast<size_t>(t) * small_vocab);
        }
        auto run = run_sampler(repeated, trials, small_vocab, 0, 1.0f, 0.95f, 20,
                               uniforms);

        // Expected probabilities over the top-k set.
        std::vector<int> order(static_cast<size_t>(small_vocab));
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
            return logits[static_cast<size_t>(a)] > logits[static_cast<size_t>(b)];
        });
        const int top = order[0];
        int top_hits = 0;
        for (int t = 0; t < trials; ++t) {
            if (run.tokens[static_cast<size_t>(t)] == top) ++top_hits;
        }
        const float observed = static_cast<float>(top_hits) / trials;

        float expected_top = 0.0f;
        {
            const float max_l = logits[static_cast<size_t>(top)];
            std::vector<float> probs;
            float sum = 0.0f;
            for (int i = 0; i < 20; ++i) {
                const float p = std::exp(logits[static_cast<size_t>(order[static_cast<size_t>(i)])] - max_l);
                probs.push_back(p);
                sum += p;
            }
            float cum = 0.0f;
            int keep = 20;
            for (int i = 0; i < 20; ++i) {
                cum += probs[static_cast<size_t>(i)] / sum;
                if (cum >= 0.95f) { keep = i + 1; break; }
            }
            float kept = 0.0f;
            for (int i = 0; i < keep; ++i) kept += probs[static_cast<size_t>(i)] / sum;
            expected_top = (probs[0] / sum) / kept;
        }
        std::printf("  top-token frequency observed=%.4f expected=%.4f\n",
                    observed, expected_top);
        expect(std::fabs(observed - expected_top) < 0.03f,
               "empirical frequency tracks intended probability");
    }

    // 7. Every sampled token must be inside the top-k set.
    {
        auto logits = random_logits(rows, vocab, 77);
        std::vector<float> uniforms{0.05f, 0.15f, 0.25f, 0.45f, 0.6f, 0.75f, 0.85f, 0.95f};
        auto run = run_sampler(logits, rows, vocab, 0, 1.5f, 0.99f, 20, uniforms);
        bool inside = true;
        for (int r = 0; r < rows; ++r) {
            const float* row = logits.data() + static_cast<size_t>(r) * vocab;
            std::vector<float> sorted(row, row + vocab);
            std::nth_element(sorted.begin(), sorted.begin() + 20, sorted.end(),
                             std::greater<float>());
            const float kth = sorted[20];
            if (run.logits[static_cast<size_t>(r)] < kth) inside = false;
        }
        expect(inside, "sampled token always lies within the top-k set");
    }

    // 8. The two-stage TP path must equal sampling the unsharded row. This is
    //    the property TP4 decoding depends on: each rank emits its local top-k
    //    as global ids, NCCL merges them, and stage 2 draws from the merge with
    //    a shared uniform. Here the merge is done on the host to isolate the
    //    kernels from NCCL.
    {
        const int tp = 4;
        const int shard = 4096;
        const int full_vocab = shard * tp;
        const int top_k = 20;
        auto full = random_logits(rows, full_vocab, 88);
        std::vector<float> uniforms{0.03f, 0.18f, 0.31f, 0.47f, 0.62f, 0.79f, 0.88f, 0.97f};

        // Reference: sample each full row as a single unsharded shard.
        auto reference = run_sampler(full, rows, full_vocab, 0, 1.0f, 0.95f,
                                     top_k, uniforms);

        // Stage 1 per rank, over that rank's contiguous slice of the vocabulary.
        std::vector<int> merged_tokens(static_cast<size_t>(rows) * top_k * tp);
        std::vector<float> merged_logits(static_cast<size_t>(rows) * top_k * tp);
        for (int rank = 0; rank < tp; ++rank) {
            std::vector<float> shard_logits(static_cast<size_t>(rows) * shard);
            for (int r = 0; r < rows; ++r) {
                const float* src = full.data() + static_cast<size_t>(r) * full_vocab +
                                   static_cast<size_t>(rank) * shard;
                std::copy(src, src + shard,
                          shard_logits.begin() + static_cast<size_t>(r) * shard);
            }

            float* d_logits = nullptr;
            int* d_tokens = nullptr;
            float* d_out = nullptr;
            check(cudaMalloc(&d_logits, shard_logits.size() * sizeof(float)), "tp malloc logits");
            check(cudaMalloc(&d_tokens, static_cast<size_t>(rows) * top_k * sizeof(int)), "tp malloc tokens");
            check(cudaMalloc(&d_out, static_cast<size_t>(rows) * top_k * sizeof(float)), "tp malloc logits out");
            check(cudaMemcpy(d_logits, shard_logits.data(),
                             shard_logits.size() * sizeof(float), cudaMemcpyHostToDevice),
                  "tp copy logits");
            check(pocket::qwen_local_topk_candidates(
                      d_logits, d_tokens, d_out, rows, shard, rank * shard,
                      top_k, nullptr),
                  "tp stage1 launch");
            check(cudaDeviceSynchronize(), "tp stage1 sync");

            std::vector<int> rank_tokens(static_cast<size_t>(rows) * top_k);
            std::vector<float> rank_logits(static_cast<size_t>(rows) * top_k);
            check(cudaMemcpy(rank_tokens.data(), d_tokens,
                             rank_tokens.size() * sizeof(int), cudaMemcpyDeviceToHost),
                  "tp copy tokens back");
            check(cudaMemcpy(rank_logits.data(), d_out,
                             rank_logits.size() * sizeof(float), cudaMemcpyDeviceToHost),
                  "tp copy logits back");
            cudaFree(d_logits);
            cudaFree(d_tokens);
            cudaFree(d_out);

            // Concatenate this rank's block, mirroring an NCCL all-gather.
            for (int r = 0; r < rows; ++r) {
                for (int i = 0; i < top_k; ++i) {
                    const size_t dst = static_cast<size_t>(r) * top_k * tp +
                                       static_cast<size_t>(rank) * top_k + i;
                    const size_t src = static_cast<size_t>(r) * top_k + i;
                    merged_tokens[dst] = rank_tokens[src];
                    merged_logits[dst] = rank_logits[src];
                }
            }
        }

        // Stage 2 over the gathered candidates.
        int* d_cand_tok = nullptr;
        float* d_cand_lg = nullptr;
        int* d_tokens = nullptr;
        float* d_logits_out = nullptr;
        float* d_uniforms = nullptr;
        pocket::DeviceRngState* d_states = nullptr;
        check(cudaMalloc(&d_cand_tok, merged_tokens.size() * sizeof(int)), "tp2 malloc cand tok");
        check(cudaMalloc(&d_cand_lg, merged_logits.size() * sizeof(float)), "tp2 malloc cand lg");
        check(cudaMalloc(&d_tokens, static_cast<size_t>(rows) * sizeof(int)), "tp2 malloc tokens");
        check(cudaMalloc(&d_logits_out, static_cast<size_t>(rows) * sizeof(float)), "tp2 malloc logits");
        check(cudaMalloc(&d_uniforms, uniforms.size() * sizeof(float)), "tp2 malloc uniforms");
        check(cudaMalloc(&d_states, static_cast<size_t>(rows) * pocket::qwen_sampler_rng_state_size()), "tp2 malloc states");
        check(cudaMemcpy(d_cand_tok, merged_tokens.data(),
                         merged_tokens.size() * sizeof(int), cudaMemcpyHostToDevice),
              "tp2 copy cand tok");
        check(cudaMemcpy(d_cand_lg, merged_logits.data(),
                         merged_logits.size() * sizeof(float), cudaMemcpyHostToDevice),
              "tp2 copy cand lg");
        check(cudaMemcpy(d_uniforms, uniforms.data(),
                         uniforms.size() * sizeof(float), cudaMemcpyHostToDevice),
              "tp2 copy uniforms");
        check(pocket::qwen_init_rng_states(d_states, rows, 7ULL, nullptr), "tp2 init states");
        check(pocket::qwen_sample_from_candidates(
                  d_cand_tok, d_cand_lg, d_tokens, d_logits_out, rows,
                  top_k * tp, 1.0f, 0.95f, top_k, d_states, d_uniforms, nullptr),
              "tp2 launch");
        check(cudaDeviceSynchronize(), "tp2 sync");

        std::vector<int> tp_tokens(static_cast<size_t>(rows));
        check(cudaMemcpy(tp_tokens.data(), d_tokens,
                         tp_tokens.size() * sizeof(int), cudaMemcpyDeviceToHost),
              "tp2 copy tokens");
        cudaFree(d_cand_tok);
        cudaFree(d_cand_lg);
        cudaFree(d_tokens);
        cudaFree(d_logits_out);
        cudaFree(d_uniforms);
        cudaFree(d_states);

        bool same = true;
        for (int r = 0; r < rows; ++r) {
            if (tp_tokens[static_cast<size_t>(r)] != reference.tokens[static_cast<size_t>(r)]) {
                std::printf("  row %d sharded=%d unsharded=%d\n", r,
                            tp_tokens[static_cast<size_t>(r)],
                            reference.tokens[static_cast<size_t>(r)]);
                same = false;
            }
        }
        expect(same, "sharded two-stage sampling equals unsharded sampling");
    }

    std::printf("%s\n", g_failures == 0 ? "PASS" : "FAILURES PRESENT");
    return g_failures == 0 ? 0 : 1;
}
