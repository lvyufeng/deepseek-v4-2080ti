#include "sampler.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

namespace pocket {

int sample_token_top_p(const float* logits, int vocab, float temperature, float top_p, std::mt19937& rng) {
    if (logits == nullptr || vocab <= 0) throw std::runtime_error("sampler: invalid logits");
    if (temperature <= 1.0e-5f) {
        int best = 0;
        float best_logit = logits[0];
        for (int i = 1; i < vocab; ++i) {
            if (logits[i] > best_logit) { best_logit = logits[i]; best = i; }
        }
        return best;
    }
    const float inv_t = 1.0f / temperature;
    float max_v = logits[0] * inv_t;
    for (int i = 1; i < vocab; ++i) {
        const float v = logits[i] * inv_t;
        if (v > max_v) max_v = v;
    }
    std::vector<float> probs(static_cast<size_t>(vocab));
    double denom = 0.0;
    for (int i = 0; i < vocab; ++i) {
        const float p = std::exp(logits[i] * inv_t - max_v);
        probs[static_cast<size_t>(i)] = p;
        denom += p;
    }
    const float inv_denom = static_cast<float>(1.0 / denom);
    for (int i = 0; i < vocab; ++i) probs[static_cast<size_t>(i)] *= inv_denom;

    if (top_p > 0.0f && top_p < 1.0f) {
        std::vector<int> idx(static_cast<size_t>(vocab));
        for (int i = 0; i < vocab; ++i) idx[static_cast<size_t>(i)] = i;
        std::sort(idx.begin(), idx.end(), [&](int a, int b) { return probs[static_cast<size_t>(a)] > probs[static_cast<size_t>(b)]; });
        double cum = 0.0;
        int keep = vocab;
        for (int rank = 0; rank < vocab; ++rank) {
            cum += probs[static_cast<size_t>(idx[static_cast<size_t>(rank)])];
            if (cum >= top_p) { keep = rank + 1; break; }
        }
        double kept_sum = 0.0;
        for (int r = 0; r < keep; ++r) kept_sum += probs[static_cast<size_t>(idx[static_cast<size_t>(r)])];
        const float inv_kept = static_cast<float>(1.0 / kept_sum);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        const float u = dist(rng);
        double accum = 0.0;
        for (int r = 0; r < keep; ++r) {
            accum += probs[static_cast<size_t>(idx[static_cast<size_t>(r)])] * inv_kept;
            if (static_cast<float>(accum) >= u) return idx[static_cast<size_t>(r)];
        }
        return idx[static_cast<size_t>(keep - 1)];
    }

    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    const float u = dist(rng);
    double accum = 0.0;
    for (int i = 0; i < vocab; ++i) {
        accum += probs[static_cast<size_t>(i)];
        if (static_cast<float>(accum) >= u) return i;
    }
    return vocab - 1;
}

}  // namespace pocket
