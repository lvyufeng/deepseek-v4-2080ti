#pragma once

#include <cstddef>
#include <random>

namespace pocket {

// Sample one token from `logits` of length `vocab` using temperature scaling
// followed by top-p (nucleus) truncation. If temperature is <= 1e-5 falls back
// to argmax. If top_p <= 0 or >= 1, no nucleus truncation is applied.
int sample_token_top_p(const float* logits, int vocab, float temperature, float top_p, std::mt19937& rng);

}  // namespace pocket
