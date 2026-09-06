// The (1 + weight) RMSNorm gamma policy: which tensors it applies to, and what it
// costs in precision.
//
// Qwen's RMSNorm scales by (1 + weight). CUDA kernels add the 1 at use time in
// FP32. Ascend cannot: aclnnRmsNorm applies gamma directly and rejects an FP32
// gamma against FP16 activations (161002, measured), so the +1 is folded into the
// FP16 weight at load time. That trades a documented precision loss for being able
// to use the fused op at all, and this test pins down both halves of the trade.
//
// No device is needed: the policy is host arithmetic on the staged bytes.

#include "device_runtime.hpp"
#include "qwen_weights.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

int failures = 0;
int checks = 0;

void expect(bool ok, const std::string& what) {
    ++checks;
    if (!ok) {
        ++failures;
        std::cout << "  FAIL " << what << "\n";
    }
}

float half_to_float(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    const uint32_t exponent = (h >> 10) & 0x1fu;
    const uint32_t mantissa = h & 0x3ffu;
    uint32_t bits = 0;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            uint32_t shifted = mantissa;
            int shift = 0;
            while ((shifted & 0x400u) == 0) {
                shifted <<= 1;
                ++shift;
            }
            bits = sign | ((127u - 15u - static_cast<uint32_t>(shift) + 1u) << 23) |
                   ((shifted & 0x3ffu) << 13);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7f800000u | (mantissa << 13);
    } else {
        bits = sign | ((exponent + 127u - 15u) << 23) | (mantissa << 13);
    }
    float out = 0.0f;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

// Apply the policy to one tensor and hand back the resulting FP16 words.
std::vector<uint16_t> apply(const std::string& name,
                            const std::vector<uint16_t>& gamma,
                            pocket::SafeDType device_dtype) {
    pocket::QwenTensorRef ref;
    ref.name = name;
    pocket::QwenHostTensor host;
    host.device_dtype = device_dtype;
    host.bytes.resize(gamma.size() * sizeof(uint16_t));
    std::memcpy(host.bytes.data(), gamma.data(), host.bytes.size());
    pocket::qwen_apply_norm_gamma_policy(ref, host);
    std::vector<uint16_t> out(gamma.size());
    std::memcpy(out.data(), host.bytes.data(), host.bytes.size());
    return out;
}

const bool folds = pocket::device_backend() == pocket::DeviceBackend::Ascend;

// Which names the predicate claims. Getting this wrong is silent: a norm that
// misses the fold is scaled by gamma instead of 1+gamma, which for the usual
// near-zero gamma nearly zeroes the layer's output.
void test_name_coverage() {
    const char* const folded[] = {
        "model.language_model.layers.0.input_layernorm.weight",
        "model.language_model.layers.47.post_attention_layernorm.weight",
        "model.language_model.layers.3.self_attn.q_norm.weight",
        "model.language_model.layers.3.self_attn.k_norm.weight",
        "model.language_model.norm.weight",
        "mtp.norm.weight",
    };
    for (const char* name : folded) {
        expect(pocket::qwen_is_one_plus_norm_gamma(name),
               std::string("expected the fold for ") + name);
    }

    const char* const untouched[] = {
        // The gated RMSNorm applies this one directly.
        "model.language_model.layers.0.linear_attn.norm.weight",
        // Not norms at all.
        "model.language_model.layers.0.mlp.gate_proj.weight",
        "model.language_model.layers.0.self_attn.q_proj.weight",
        "model.language_model.layers.0.linear_attn.conv1d.weight",
        "model.language_model.layers.0.linear_attn.A_log",
        "model.language_model.embed_tokens.weight",
        "lm_head.weight",
    };
    for (const char* name : untouched) {
        expect(!pocket::qwen_is_one_plus_norm_gamma(name),
               std::string("expected no fold for ") + name);
    }
}

// Only FP16 device tensors are folded. A weight staged as FP32 would be scaled
// twice if the policy fired on it, so the dtype guard matters.
void test_dtype_and_name_guards() {
    const std::vector<uint16_t> gamma = {0x3000u, 0xB000u, 0x0000u};
    const std::string norm = "model.language_model.norm.weight";

    expect(apply(norm, gamma, pocket::SafeDType::F32) == gamma,
           "an F32 device tensor is left alone");
    expect(apply("model.language_model.layers.0.linear_attn.norm.weight", gamma,
                 pocket::SafeDType::F16) == gamma,
           "the gated norm weight is left alone");
    expect(apply("model.language_model.layers.0.mlp.up_proj.weight", gamma,
                 pocket::SafeDType::F16) == gamma,
           "a projection weight is left alone");
    expect((apply(norm, gamma, pocket::SafeDType::F16) != gamma) == folds,
           "the fold fires exactly on the Ascend backend");
}

// The bound the header claims: the stored value is the correctly-rounded FP16 of
// the exact sum, so its error against 1+gamma is at most half an ULP there.
void test_precision_bound() {
    if (!folds) {
        std::cout << "  [skip] precision bound is Ascend-only\n";
        return;
    }
    // Every FP16 bit pattern in the magnitude range a norm weight occupies, both
    // signs, rather than a sampled few.
    std::vector<uint16_t> gamma;
    for (uint32_t exponent = 0; exponent <= 16; ++exponent) {  // up to ~2.0
        for (uint32_t mantissa = 0; mantissa < 1024u; ++mantissa) {
            const uint16_t bits =
                static_cast<uint16_t>((exponent << 10) | mantissa);
            gamma.push_back(bits);
            gamma.push_back(static_cast<uint16_t>(bits | 0x8000u));
        }
    }
    const std::vector<uint16_t> folded =
        apply("model.language_model.norm.weight", gamma, pocket::SafeDType::F16);

    double worst_absolute = 0.0;
    int over_half_ulp = 0;
    for (size_t i = 0; i < gamma.size(); ++i) {
        const double exact = 1.0 + static_cast<double>(half_to_float(gamma[i]));
        const double stored = half_to_float(folded[i]);
        const double error = std::fabs(stored - exact);
        worst_absolute = std::max(worst_absolute, error);
        // ULP at the exact sum. Sums here land in [-1, 3], so the exponent is
        // between -1 and 1 and the ULP is 2^(exponent - 10).
        int exponent = 0;
        std::frexp(exact == 0.0 ? 1.0 : exact, &exponent);
        const double ulp = std::ldexp(1.0, exponent - 1 - 10);
        // Half an ULP is the bound for round-to-nearest, with a small slack for
        // the tie case landing on the far side.
        if (error > 0.5 * ulp * (1.0 + 1e-9)) ++over_half_ulp;
    }
    expect(over_half_ulp == 0,
           std::to_string(over_half_ulp) +
               " folded values exceed half an ULP of the exact sum");

    // The half-ULP result above is the correctness statement and holds across the
    // whole sweep. The absolute bound is range-dependent, so state it only over the
    // magnitudes a trained norm weight actually has. |gamma| <= 0.25 puts 1+gamma
    // in [0.75, 1.25], where an FP16 ULP is at most 2^-10 and so the rounding error
    // is at most 2^-11: the header's "2^-11 versus 2^-24" claim, measured.
    double worst_realistic = 0.0;
    for (size_t i = 0; i < gamma.size(); ++i) {
        const double value = half_to_float(gamma[i]);
        if (std::fabs(value) > 0.25) continue;
        worst_realistic = std::max(
            worst_realistic, std::fabs(half_to_float(folded[i]) - (1.0 + value)));
    }
    expect(worst_realistic <= std::ldexp(1.0, -11),
           "worst absolute error for |gamma| <= 0.25 is " +
               std::to_string(worst_realistic));
    std::cout << "  worst absolute error " << worst_realistic << " for |gamma| <= "
              << "0.25, " << worst_absolute << " over the full sweep (2^-11 = "
              << std::ldexp(1.0, -11) << ")\n";

    // A fold applied twice would be a real bug, and it is not idempotent, so make
    // sure the test's own reasoning about that is stated: folding the folded value
    // moves it again.
    const std::vector<uint16_t> twice =
        apply("model.language_model.norm.weight", folded, pocket::SafeDType::F16);
    expect(twice != folded, "the fold is not idempotent, as expected");
}

}  // namespace

int main() {
    std::cout << "backend=" << pocket::device_backend_name() << "\n";
    test_name_coverage();
    test_dtype_and_name_guards();
    test_precision_bound();
    if (failures != 0) {
        std::cout << "FAIL " << failures << " of " << checks << " checks\n";
        return 1;
    }
    std::cout << "PASS (" << checks << " checks)\n";
    return 0;
}
