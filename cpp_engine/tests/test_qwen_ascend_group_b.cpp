// Correctness tests for the hand-written Group B Qwen operators.
//
// Host references follow the operator definitions in double precision rather than
// copying either vendor implementation. The calls use qwen_ops.hpp's neutral names,
// so this also verifies that the engine-facing dispatch seam reaches the Ascend
// launchers.
//
//   ./tests/test_qwen_ascend_group_b [--device N] [--only a,b,c]

#include "device_runtime.hpp"
#include "qwen_ops.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr int kKeyDim = 128;
constexpr int kValueDim = 128;

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
            bits = sign |
                   ((127u - 15u - static_cast<uint32_t>(shift) + 1u) << 23) |
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

uint16_t float_to_half(float f) {
    uint32_t bits = 0;
    std::memcpy(&bits, &f, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    const int exponent = static_cast<int>((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t mantissa = bits & 0x7fffffu;
    if (exponent <= 0) {
        if (exponent < -10) return static_cast<uint16_t>(sign);
        mantissa |= 0x800000u;
        const int shift = 14 - exponent;
        uint32_t half = mantissa >> shift;
        const uint32_t remainder = mantissa & ((1u << shift) - 1u);
        const uint32_t halfway = 1u << (shift - 1);
        if (remainder > halfway || (remainder == halfway && (half & 1u))) ++half;
        return static_cast<uint16_t>(sign | half);
    }
    if (exponent >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
    uint32_t half = mantissa >> 13;
    const uint32_t remainder = mantissa & 0x1fffu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (half & 1u))) {
        ++half;
        if (half == 0x400u) {
            half = 0;
            if (exponent + 1 >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
            return static_cast<uint16_t>(
                sign | (static_cast<uint32_t>(exponent + 1) << 10));
        }
    }
    return static_cast<uint16_t>(
        sign | (static_cast<uint32_t>(exponent) << 10) | half);
}

template <typename T>
class DeviceBuffer {
public:
    explicit DeviceBuffer(size_t count) : count_(count) {
        if (count == 0) return;
        if (!pocket::device_malloc_into(ptr_, count * sizeof(T))) {
            throw std::runtime_error("device_malloc failed");
        }
        if (!pocket::device_memset(ptr_, 0, count * sizeof(T))) {
            throw std::runtime_error("device_memset failed");
        }
    }

    explicit DeviceBuffer(const std::vector<T>& host) : DeviceBuffer(host.size()) {
        upload(host);
    }

    ~DeviceBuffer() { pocket::device_free(ptr_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* get() const { return ptr_; }

    void upload(const std::vector<T>& host) {
        if (host.size() != count_) {
            throw std::runtime_error("DeviceBuffer upload size mismatch");
        }
        if (count_ != 0 &&
            !pocket::memcpy_h2d(ptr_, host.data(), count_ * sizeof(T))) {
            throw std::runtime_error("memcpy_h2d failed");
        }
    }

    std::vector<T> download() const {
        std::vector<T> host(count_);
        if (count_ != 0 &&
            !pocket::memcpy_d2h(host.data(), ptr_, count_ * sizeof(T))) {
            throw std::runtime_error("memcpy_d2h failed");
        }
        return host;
    }

private:
    T* ptr_ = nullptr;
    size_t count_ = 0;
};

std::vector<uint16_t> random_halves(size_t count, std::mt19937& rng,
                                    float scale) {
    std::uniform_real_distribution<float> dist(-scale, scale);
    std::vector<uint16_t> out(count);
    for (uint16_t& value : out) value = float_to_half(dist(rng));
    return out;
}

std::vector<float> random_floats(size_t count, std::mt19937& rng, float scale) {
    std::uniform_real_distribution<float> dist(-scale, scale);
    std::vector<float> out(count);
    for (float& value : out) value = dist(rng);
    return out;
}

void sync_or_throw(const std::string& what) {
    if (!pocket::device_synchronize()) {
        throw std::runtime_error("device_synchronize failed after " + what);
    }
}

struct ErrorStats {
    double worst = 0.0;
    int mismatches = 0;
};

ErrorStats half_error(const std::vector<uint16_t>& got,
                      const std::vector<double>& want, double relative,
                      double absolute) {
    if (got.size() != want.size()) {
        return {1.0e30, static_cast<int>(std::max(got.size(), want.size()))};
    }
    ErrorStats stats;
    for (size_t i = 0; i < want.size(); ++i) {
        const double actual = half_to_float(got[i]);
        const double error = std::fabs(actual - want[i]);
        const double scale = std::max(std::fabs(want[i]), absolute);
        stats.worst = std::max(stats.worst, error / scale);
        if (error > absolute + relative * std::fabs(want[i])) ++stats.mismatches;
    }
    return stats;
}

ErrorStats float_error(const std::vector<float>& got,
                       const std::vector<double>& want, double relative,
                       double absolute) {
    if (got.size() != want.size()) {
        return {1.0e30, static_cast<int>(std::max(got.size(), want.size()))};
    }
    ErrorStats stats;
    for (size_t i = 0; i < want.size(); ++i) {
        const double error = std::fabs(static_cast<double>(got[i]) - want[i]);
        const double scale = std::max(std::fabs(want[i]), absolute);
        stats.worst = std::max(stats.worst, error / scale);
        if (error > absolute + relative * std::fabs(want[i])) ++stats.mismatches;
    }
    return stats;
}

void expect_half_close(const std::vector<uint16_t>& got,
                       const std::vector<double>& want, double relative,
                       double absolute, const std::string& what) {
    const ErrorStats stats = half_error(got, want, relative, absolute);
    std::string detail;
    if (stats.mismatches != 0 && got.size() == want.size()) {
        for (size_t i = 0; i < want.size(); ++i) {
            const double actual = half_to_float(got[i]);
            const double error = std::fabs(actual - want[i]);
            if (error > absolute + relative * std::fabs(want[i])) {
                detail = " first=" + std::to_string(i) +
                         " got=" + std::to_string(actual) +
                         " want=" + std::to_string(want[i]);
                break;
            }
        }
    }
    expect(stats.mismatches == 0,
           what + " mismatches=" + std::to_string(stats.mismatches) +
               " worst_relative=" + std::to_string(stats.worst) + detail);
}

void expect_half_exact(const std::vector<uint16_t>& got,
                       const std::vector<uint16_t>& want,
                       const std::string& what) {
    int mismatches = 0;
    size_t first = 0;
    for (size_t i = 0; i < std::min(got.size(), want.size()); ++i) {
        if (got[i] != want[i]) {
            if (mismatches == 0) first = i;
            ++mismatches;
        }
    }
    if (got.size() != want.size()) {
        mismatches += static_cast<int>(std::max(got.size(), want.size()) -
                                       std::min(got.size(), want.size()));
    }
    std::string detail;
    if (mismatches != 0 && first < got.size() && first < want.size()) {
        detail = " first=" + std::to_string(first) +
                 " got=" + std::to_string(half_to_float(got[first])) +
                 " want=" + std::to_string(half_to_float(want[first]));
    }
    expect(mismatches == 0,
           what + " mismatches=" + std::to_string(mismatches) + detail);
}

void expect_float_close(const std::vector<float>& got,
                        const std::vector<double>& want, double relative,
                        double absolute, const std::string& what) {
    const ErrorStats stats = float_error(got, want, relative, absolute);
    expect(stats.mismatches == 0,
           what + " mismatches=" + std::to_string(stats.mismatches) +
               " worst_relative=" + std::to_string(stats.worst));
}

std::vector<double> normalize_reference(const std::vector<uint16_t>& source,
                                        int rows, int heads, int dim) {
    std::vector<double> out(source.size());
    for (int row = 0; row < rows; ++row) {
        for (int head = 0; head < heads; ++head) {
            const size_t base = (static_cast<size_t>(row) * heads + head) * dim;
            double sum = 0.0;
            for (int d = 0; d < dim; ++d) {
                const double value = half_to_float(source[base + d]);
                sum += value * value;
            }
            const double inverse = 1.0 / std::sqrt(sum + 1.0e-6);
            for (int d = 0; d < dim; ++d) {
                out[base + d] = half_to_float(source[base + d]) * inverse;
            }
        }
    }
    return out;
}

void test_normalize_qk(std::mt19937& rng) {
    const int rows = 3;
    const int key_heads = 7;
    const size_t count = static_cast<size_t>(rows) * key_heads * kKeyDim;
    const std::vector<uint16_t> q = random_halves(count, rng, 0.7f);
    const std::vector<uint16_t> k = random_halves(count, rng, 0.7f);
    DeviceBuffer<uint16_t> d_q(q);
    DeviceBuffer<uint16_t> d_k(k);
    DeviceBuffer<float> d_qn(count);
    DeviceBuffer<float> d_kn(count);

    expect(pocket::qwen_normalize_gated_delta_qk_f16(
               d_q.get(), d_k.get(), d_qn.get(), d_kn.get(), rows, key_heads,
               kKeyDim),
           "normalize qk launch");
    sync_or_throw("normalize qk");

    expect_float_close(d_qn.download(), normalize_reference(q, rows, key_heads,
                                                             kKeyDim),
                       3.0e-5, 2.0e-6, "normalized q");
    expect_float_close(d_kn.download(), normalize_reference(k, rows, key_heads,
                                                             kKeyDim),
                       3.0e-5, 2.0e-6, "normalized k");
}

struct RecurrenceReference {
    std::vector<double> output;
    std::vector<double> state;
};

RecurrenceReference recurrence_reference(
    const std::vector<float>& initial_state, const std::vector<double>& q,
    const std::vector<double>& k, const std::vector<uint16_t>& v,
    const std::vector<uint16_t>& g, const std::vector<uint16_t>& beta, int rows,
    int heads, int key_heads, double q_scale) {
    std::vector<double> state(initial_state.begin(), initial_state.end());
    std::vector<double> output(static_cast<size_t>(rows) * heads * kValueDim);
    const int repeat = heads / key_heads;
    std::vector<double> memory(kValueDim);
    std::vector<double> delta(kValueDim);
    for (int head = 0; head < heads; ++head) {
        const int key_head = head / repeat;
        const size_t state_base = static_cast<size_t>(head) * kKeyDim * kValueDim;
        for (int token = 0; token < rows; ++token) {
            const size_t key_base =
                (static_cast<size_t>(token) * key_heads + key_head) * kKeyDim;
            const size_t value_base =
                (static_cast<size_t>(token) * heads + head) * kValueDim;
            const double decay =
                std::exp(static_cast<double>(half_to_float(g[token * heads + head])));
            const double gain = half_to_float(beta[token * heads + head]);
            for (int i = 0; i < kKeyDim; ++i) {
                for (int d = 0; d < kValueDim; ++d) {
                    state[state_base + static_cast<size_t>(i) * kValueDim + d] *= decay;
                }
            }
            for (int d = 0; d < kValueDim; ++d) {
                double sum = 0.0;
                for (int i = 0; i < kKeyDim; ++i) {
                    sum += state[state_base + static_cast<size_t>(i) * kValueDim + d] *
                           k[key_base + i];
                }
                memory[d] = sum;
                delta[d] =
                    (half_to_float(v[value_base + d]) - memory[d]) * gain;
            }
            for (int i = 0; i < kKeyDim; ++i) {
                for (int d = 0; d < kValueDim; ++d) {
                    state[state_base + static_cast<size_t>(i) * kValueDim + d] +=
                        k[key_base + i] * delta[d];
                }
            }
            for (int d = 0; d < kValueDim; ++d) {
                double sum = 0.0;
                for (int i = 0; i < kKeyDim; ++i) {
                    sum += state[state_base + static_cast<size_t>(i) * kValueDim + d] *
                           q[key_base + i];
                }
                output[value_base + d] = sum * q_scale;
            }
        }
    }
    return {std::move(output), std::move(state)};
}

std::vector<double> normalize_as_double(const std::vector<uint16_t>& source,
                                        int rows, int heads) {
    return normalize_reference(source, rows, heads, kKeyDim);
}

void test_gated_delta_recurrence(std::mt19937& rng) {
    const int rows = 2;
    const int heads = 4;
    const int key_heads = 2;
    const float q_scale = 0.125f;
    const size_t key_count = static_cast<size_t>(rows) * key_heads * kKeyDim;
    const size_t value_count = static_cast<size_t>(rows) * heads * kValueDim;
    const size_t state_count = static_cast<size_t>(heads) * kKeyDim * kValueDim;
    const std::vector<uint16_t> q = random_halves(key_count, rng, 0.6f);
    const std::vector<uint16_t> k = random_halves(key_count, rng, 0.6f);
    const std::vector<uint16_t> v = random_halves(value_count, rng, 0.35f);
    const std::vector<uint16_t> g = random_halves(static_cast<size_t>(rows) * heads,
                                                  rng, 0.06f);
    std::vector<uint16_t> beta(static_cast<size_t>(rows) * heads);
    std::uniform_real_distribution<float> beta_dist(0.15f, 0.75f);
    for (uint16_t& value : beta) value = float_to_half(beta_dist(rng));
    const std::vector<float> initial_state = random_floats(state_count, rng, 0.002f);
    const std::vector<double> qn = normalize_as_double(q, rows, key_heads);
    const std::vector<double> kn = normalize_as_double(k, rows, key_heads);
    const RecurrenceReference reference = recurrence_reference(
        initial_state, qn, kn, v, g, beta, rows, heads, key_heads, q_scale);

    DeviceBuffer<float> d_state(initial_state);
    DeviceBuffer<uint16_t> d_q(q);
    DeviceBuffer<uint16_t> d_k(k);
    DeviceBuffer<uint16_t> d_v(v);
    DeviceBuffer<uint16_t> d_g(g);
    DeviceBuffer<uint16_t> d_beta(beta);
    DeviceBuffer<uint16_t> d_out(value_count);
    expect(pocket::qwen_gated_delta_sequence_f16(
               d_state.get(), d_q.get(), d_k.get(), d_v.get(), d_g.get(),
               d_beta.get(), d_out.get(), rows, heads, key_heads, kKeyDim,
               kValueDim, q_scale),
           "gated delta sequence launch");
    sync_or_throw("gated delta sequence");
    expect_half_close(d_out.download(), reference.output, 7.0e-3, 2.5e-3,
                      "gated delta sequence output");
    expect_float_close(d_state.download(), reference.state, 8.0e-4, 8.0e-6,
                       "gated delta sequence state");

    // Pre-normalized and shared entry points must implement the same recurrence.
    std::vector<float> qn32(qn.begin(), qn.end());
    std::vector<float> kn32(kn.begin(), kn.end());
    DeviceBuffer<float> d_qn(qn32);
    DeviceBuffer<float> d_kn(kn32);
    DeviceBuffer<float> d_norm_state(initial_state);
    DeviceBuffer<float> d_shared_state(initial_state);
    DeviceBuffer<uint16_t> d_norm_out(value_count);
    DeviceBuffer<uint16_t> d_shared_out(value_count);
    expect(pocket::qwen_gated_delta_sequence_normalized_f16(
               d_norm_state.get(), d_qn.get(), d_kn.get(), d_v.get(), d_g.get(),
               d_beta.get(), d_norm_out.get(), rows, heads, key_heads, kKeyDim,
               kValueDim, q_scale),
           "normalized gated delta launch");
    expect(pocket::qwen_gated_delta_sequence_normalized_shared_f16(
               d_shared_state.get(), d_qn.get(), d_kn.get(), d_v.get(), d_g.get(),
               d_beta.get(), d_shared_out.get(), rows, heads, key_heads, kKeyDim,
               kValueDim, q_scale),
           "shared normalized gated delta launch");
    sync_or_throw("normalized gated delta");
    const std::vector<uint16_t> norm_out = d_norm_out.download();
    const std::vector<uint16_t> shared_out = d_shared_out.download();
    expect(norm_out == shared_out, "normalized and shared outputs are bit-identical");
    expect(d_norm_state.download() == d_shared_state.download(),
           "normalized and shared states are bit-identical");
    expect_half_close(norm_out, reference.output, 7.0e-3, 2.5e-3,
                      "normalized gated delta output");

    // A one-token sequence and the step entry point start from the same state and
    // must finish identically. This is the actual decode geometry.
    const size_t one_key_count = static_cast<size_t>(key_heads) * kKeyDim;
    const size_t one_value_count = static_cast<size_t>(heads) * kValueDim;
    std::vector<uint16_t> q1(q.begin(), q.begin() + one_key_count);
    std::vector<uint16_t> k1(k.begin(), k.begin() + one_key_count);
    std::vector<uint16_t> v1(v.begin(), v.begin() + one_value_count);
    std::vector<uint16_t> g1(g.begin(), g.begin() + heads);
    std::vector<uint16_t> beta1(beta.begin(), beta.begin() + heads);
    DeviceBuffer<uint16_t> d_q1(q1), d_k1(k1), d_v1(v1), d_g1(g1), d_beta1(beta1);
    DeviceBuffer<float> d_step_state(initial_state), d_one_state(initial_state);
    DeviceBuffer<uint16_t> d_step_out(one_value_count), d_one_out(one_value_count);
    expect(pocket::qwen_gated_delta_step_f16(
               d_step_state.get(), d_q1.get(), d_k1.get(), d_v1.get(), d_g1.get(),
               d_beta1.get(), d_step_out.get(), heads, key_heads, kKeyDim,
               kValueDim, q_scale),
           "gated delta step launch");
    expect(pocket::qwen_gated_delta_sequence_f16(
               d_one_state.get(), d_q1.get(), d_k1.get(), d_v1.get(), d_g1.get(),
               d_beta1.get(), d_one_out.get(), 1, heads, key_heads, kKeyDim,
               kValueDim, q_scale),
           "one-row gated delta sequence launch");
    sync_or_throw("gated delta step parity");
    expect(d_step_out.download() == d_one_out.download(),
           "step and one-row sequence outputs are bit-identical");
    expect(d_step_state.download() == d_one_state.download(),
           "step and one-row sequence states are bit-identical");
}

void test_linear_attn_gates(std::mt19937& rng) {
    const int rows = 9;
    const int heads = 7;
    const size_t count = static_cast<size_t>(rows) * heads;
    std::vector<uint16_t> a = random_halves(count, rng, 4.0f);
    const std::vector<uint16_t> b = random_halves(count, rng, 7.0f);
    const std::vector<uint16_t> a_log = random_halves(heads, rng, 1.5f);
    const std::vector<uint16_t> dt_bias = random_halves(heads, rng, 2.0f);
    // Exercise the overflow-safe softplus branch explicitly.
    a[3] = float_to_half(100.0f);
    a[19] = float_to_half(40.0f);
    DeviceBuffer<uint16_t> d_a(a), d_b(b), d_a_log(a_log), d_dt(dt_bias);
    DeviceBuffer<uint16_t> d_g(count), d_beta(count);
    expect(pocket::qwen_linear_attn_gates_f16(
               d_a.get(), d_b.get(), d_a_log.get(), d_dt.get(), d_g.get(),
               d_beta.get(), rows, heads),
           "linear attention gates launch");
    sync_or_throw("linear attention gates");

    std::vector<double> want_g(count), want_beta(count);
    for (size_t i = 0; i < count; ++i) {
        const int head = static_cast<int>(i % heads);
        const double x = static_cast<double>(half_to_float(a[i])) +
                         half_to_float(dt_bias[head]);
        const double softplus = x > 0.0 ? x + std::log1p(std::exp(-x))
                                        : std::log1p(std::exp(x));
        want_g[i] = -std::exp(static_cast<double>(half_to_float(a_log[head]))) *
                    softplus;
        const double bv = half_to_float(b[i]);
        want_beta[i] = bv >= 0.0 ? 1.0 / (1.0 + std::exp(-bv))
                                 : std::exp(bv) / (1.0 + std::exp(bv));
    }
    expect_half_close(d_g.download(), want_g, 2.5e-3, 2.0e-3,
                      "linear attention g");
    // Hardware Reciprocal is intentionally FP16-grade here.
    expect_half_close(d_beta.download(), want_beta, 4.0e-3, 1.5e-3,
                      "linear attention beta");
}

std::vector<double> conv_reference(const std::vector<uint16_t>& x,
                                   const std::vector<uint16_t>& weight,
                                   const std::vector<uint16_t>* tail, int seq_len,
                                   int channels, int kernel) {
    const int tail_len = kernel - 1;
    std::vector<double> out(static_cast<size_t>(seq_len) * channels);
    for (int token = 0; token < seq_len; ++token) {
        for (int channel = 0; channel < channels; ++channel) {
            double sum = 0.0;
            for (int tap = 0; tap < kernel; ++tap) {
                const int source = token - tail_len + tap;
                const double value = source >= 0
                    ? half_to_float(x[static_cast<size_t>(source) * channels + channel])
                    : tail == nullptr
                        ? 0.0
                        : half_to_float((*tail)[static_cast<size_t>(source + tail_len) *
                                                      channels + channel]);
                sum += value *
                       half_to_float(weight[static_cast<size_t>(channel) * kernel + tap]);
            }
            out[static_cast<size_t>(token) * channels + channel] =
                sum / (1.0 + std::exp(-sum));
        }
    }
    return out;
}

std::vector<uint16_t> expected_tail(const std::vector<uint16_t>& old_tail,
                                    const std::vector<uint16_t>& x, int seq_len,
                                    int channels, int kernel) {
    const int tail_len = kernel - 1;
    std::vector<uint16_t> history = old_tail;
    history.insert(history.end(), x.begin(), x.end());
    std::vector<uint16_t> out(static_cast<size_t>(tail_len) * channels);
    const int available_rows = tail_len + seq_len;
    const int first = available_rows - tail_len;
    std::copy(history.begin() + static_cast<size_t>(first) * channels,
              history.begin() + static_cast<size_t>(first + tail_len) * channels,
              out.begin());
    return out;
}

void test_causal_conv(std::mt19937& rng) {
    const int seq_len = 5;
    const int channels = 517;
    const int kernel = 4;
    const int tail_len = kernel - 1;
    const std::vector<uint16_t> x =
        random_halves(static_cast<size_t>(seq_len) * channels, rng, 0.5f);
    const std::vector<uint16_t> weight =
        random_halves(static_cast<size_t>(channels) * kernel, rng, 0.35f);
    const std::vector<uint16_t> tail =
        random_halves(static_cast<size_t>(tail_len) * channels, rng, 0.4f);
    DeviceBuffer<uint16_t> d_x(x), d_weight(weight), d_tail(tail);
    DeviceBuffer<uint16_t> d_y(static_cast<size_t>(seq_len) * channels);
    expect(pocket::qwen_causal_depthwise_conv_silu_f16(
               d_x.get(), d_weight.get(), d_tail.get(), d_y.get(), seq_len,
               channels, kernel, true),
           "causal conv launch");
    sync_or_throw("causal conv");
    expect_half_close(d_y.download(),
                      conv_reference(x, weight, &tail, seq_len, channels, kernel),
                      6.0e-3, 2.5e-3, "causal conv output");
    expect(d_tail.download() ==
               expected_tail(tail, x, seq_len, channels, kernel),
           "causal conv updates carried tail");

    // update_tail=false must leave history untouched.
    DeviceBuffer<uint16_t> d_tail_unchanged(tail);
    DeviceBuffer<uint16_t> d_y2(static_cast<size_t>(seq_len) * channels);
    expect(pocket::qwen_causal_depthwise_conv_silu_f16(
               d_x.get(), d_weight.get(), d_tail_unchanged.get(), d_y2.get(),
               seq_len, channels, kernel, false),
           "causal conv no-tail-update launch");
    sync_or_throw("causal conv no-tail-update");
    expect(d_tail_unchanged.download() == tail,
           "causal conv leaves tail unchanged when requested");

    // Prefill followed by one decode token must equal one fused convolution.
    const int prefix = seq_len - 1;
    std::vector<uint16_t> prefix_x(x.begin(), x.begin() +
                                             static_cast<size_t>(prefix) * channels);
    std::vector<uint16_t> last_x(x.begin() + static_cast<size_t>(prefix) * channels,
                                 x.end());
    DeviceBuffer<uint16_t> d_prefix(prefix_x), d_last(last_x), d_chain_tail(tail);
    DeviceBuffer<uint16_t> d_prefix_y(static_cast<size_t>(prefix) * channels);
    DeviceBuffer<uint16_t> d_last_y(channels);
    expect(pocket::qwen_causal_depthwise_conv_silu_f16(
               d_prefix.get(), d_weight.get(), d_chain_tail.get(), d_prefix_y.get(),
               prefix, channels, kernel, true),
           "causal conv prefix launch");
    expect(pocket::qwen_causal_depthwise_conv_silu_f16(
               d_last.get(), d_weight.get(), d_chain_tail.get(), d_last_y.get(), 1,
               channels, kernel, true),
           "causal conv decode launch");
    sync_or_throw("causal conv continuity");
    std::vector<uint16_t> chained = d_prefix_y.download();
    const std::vector<uint16_t> last = d_last_y.download();
    chained.insert(chained.end(), last.begin(), last.end());
    expect(chained == d_y.download(), "causal conv prefill/decode continuity");

    DeviceBuffer<uint16_t> d_zero_y(static_cast<size_t>(seq_len) * channels);
    expect(pocket::qwen_causal_depthwise_conv_silu_f16(
               d_x.get(), d_weight.get(), nullptr, d_zero_y.get(), seq_len,
               channels, kernel, false),
           "causal conv null-tail launch");
    sync_or_throw("causal conv null-tail");
    expect_half_close(d_zero_y.download(),
                      conv_reference(x, weight, nullptr, seq_len, channels, kernel),
                      6.0e-3, 2.5e-3, "causal conv null-tail output");
}

void rope_reference(std::vector<uint16_t>& values, int rows, int heads,
                    int head_dim, int rotary_dim, int start_position,
                    double theta) {
    const int half = rotary_dim / 2;
    for (int row = 0; row < rows; ++row) {
        for (int index = 0; index < half; ++index) {
            const double frequency =
                std::pow(theta, -2.0 * static_cast<double>(index) / rotary_dim);
            const double angle = (start_position + row) * frequency;
            const double c = std::cos(angle);
            const double s = std::sin(angle);
            for (int head = 0; head < heads; ++head) {
                const size_t base =
                    (static_cast<size_t>(row) * heads + head) * head_dim;
                const double a = half_to_float(values[base + index]);
                const double b = half_to_float(values[base + index + half]);
                values[base + index] =
                    float_to_half(static_cast<float>(a * c - b * s));
                values[base + index + half] =
                    float_to_half(static_cast<float>(b * c + a * s));
            }
        }
    }
}

void test_partial_rope(std::mt19937& rng) {
    const int rows = 3;
    const int q_heads = 5;
    const int kv_heads = 2;
    const int head_dim = 80;
    const int rotary_dim = 64;
    const int start_position = 37;
    const float theta = 1000000.0f;
    const std::vector<uint16_t> q = random_halves(
        static_cast<size_t>(rows) * q_heads * head_dim, rng, 0.7f);
    const std::vector<uint16_t> k = random_halves(
        static_cast<size_t>(rows) * kv_heads * head_dim, rng, 0.7f);
    std::vector<uint16_t> want_q = q;
    std::vector<uint16_t> want_k = k;
    rope_reference(want_q, rows, q_heads, head_dim, rotary_dim, start_position,
                   theta);
    rope_reference(want_k, rows, kv_heads, head_dim, rotary_dim, start_position,
                   theta);
    DeviceBuffer<uint16_t> d_q(q), d_k(k);
    expect(pocket::qwen_partial_rope_rows_f16(
               d_q.get(), d_k.get(), start_position, rows, rotary_dim, theta,
               q_heads, kv_heads, head_dim),
           "partial rope launch");
    sync_or_throw("partial rope");
    const std::vector<uint16_t> got_q = d_q.download();
    const std::vector<uint16_t> got_k = d_k.download();
    expect_half_exact(got_q, want_q, "partial rope q matches rounded reference");
    expect_half_exact(got_k, want_k, "partial rope k matches rounded reference");
    int untouched_mismatches = 0;
    for (int row = 0; row < rows; ++row) {
        for (int head = 0; head < q_heads; ++head) {
            const size_t base =
                (static_cast<size_t>(row) * q_heads + head) * head_dim;
            for (int d = rotary_dim; d < head_dim; ++d) {
                if (got_q[base + d] != q[base + d]) ++untouched_mismatches;
            }
        }
        for (int head = 0; head < kv_heads; ++head) {
            const size_t base =
                (static_cast<size_t>(row) * kv_heads + head) * head_dim;
            for (int d = rotary_dim; d < head_dim; ++d) {
                if (got_k[base + d] != k[base + d]) ++untouched_mismatches;
            }
        }
    }
    expect(untouched_mismatches == 0,
           "partial rope preserves unrotated dimensions");
}

void test_append_kv(std::mt19937& rng) {
    const int seq_len = 3;
    const int kv_heads = 3;
    const int head_dim = 17;
    const int start_pos = 2;
    const int max_context = 8;
    const size_t rows_count = static_cast<size_t>(seq_len) * kv_heads * head_dim;
    const size_t cache_count =
        static_cast<size_t>(max_context) * kv_heads * head_dim;
    const std::vector<uint16_t> k_rows = random_halves(rows_count, rng, 0.8f);
    const std::vector<uint16_t> v_rows = random_halves(rows_count, rng, 0.8f);
    std::vector<uint16_t> k_cache(cache_count, float_to_half(7.0f));
    std::vector<uint16_t> v_cache(cache_count, float_to_half(-7.0f));
    std::vector<uint16_t> want_k = k_cache;
    std::vector<uint16_t> want_v = v_cache;
    for (int token = 0; token < seq_len; ++token) {
        const size_t src = static_cast<size_t>(token) * kv_heads * head_dim;
        const size_t dst =
            static_cast<size_t>(start_pos + token) * kv_heads * head_dim;
        std::copy(k_rows.begin() + src,
                  k_rows.begin() + src + kv_heads * head_dim,
                  want_k.begin() + dst);
        std::copy(v_rows.begin() + src,
                  v_rows.begin() + src + kv_heads * head_dim,
                  want_v.begin() + dst);
    }
    DeviceBuffer<uint16_t> d_k_rows(k_rows), d_v_rows(v_rows);
    DeviceBuffer<uint16_t> d_k_cache(k_cache), d_v_cache(v_cache);
    expect(pocket::qwen_append_kv_cache_f16(
               d_k_rows.get(), d_v_rows.get(), d_k_cache.get(), d_v_cache.get(),
               seq_len, kv_heads, head_dim, start_pos, max_context),
           "append kv launch");
    sync_or_throw("append kv");
    expect_half_exact(d_k_cache.download(), want_k,
                      "append kv writes K at the offset");
    expect_half_exact(d_v_cache.download(), want_v,
                      "append kv writes V at the offset");
}

std::vector<double> attention_reference(const std::vector<uint16_t>& q,
                                        const std::vector<uint16_t>& k_cache,
                                        const std::vector<uint16_t>& v_cache,
                                        int rows, int q_heads, int kv_heads,
                                        int head_dim, int position_offset) {
    std::vector<double> output(static_cast<size_t>(rows) * q_heads * head_dim);
    const int repeat = q_heads / kv_heads;
    const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));
    std::vector<double> scores;
    for (int row = 0; row < rows; ++row) {
        const int context_len = position_offset + row + 1;
        scores.resize(context_len);
        for (int head = 0; head < q_heads; ++head) {
            const int kv_head = head / repeat;
            const size_t q_base =
                (static_cast<size_t>(row) * q_heads + head) * head_dim;
            double maximum = -INFINITY;
            for (int pos = 0; pos < context_len; ++pos) {
                const size_t cache_base =
                    (static_cast<size_t>(pos) * kv_heads + kv_head) * head_dim;
                double score = 0.0;
                for (int d = 0; d < head_dim; ++d) {
                    score += static_cast<double>(half_to_float(q[q_base + d])) *
                             half_to_float(k_cache[cache_base + d]);
                }
                scores[pos] = score * scale;
                maximum = std::max(maximum, scores[pos]);
            }
            double denominator = 0.0;
            for (double& score : scores) {
                score = std::exp(score - maximum);
                denominator += score;
            }
            for (int d = 0; d < head_dim; ++d) {
                double value = 0.0;
                for (int pos = 0; pos < context_len; ++pos) {
                    const size_t cache_base =
                        (static_cast<size_t>(pos) * kv_heads + kv_head) * head_dim;
                    value += scores[pos] * half_to_float(v_cache[cache_base + d]);
                }
                output[q_base + d] = value / denominator;
            }
        }
    }
    return output;
}

void test_gqa_decode(std::mt19937& rng) {
    const int q_heads = 6;
    const int kv_heads = 2;
    const int head_dim = 32;
    const int context_len = 5;
    const int max_context = 9;
    const std::vector<uint16_t> q =
        random_halves(static_cast<size_t>(q_heads) * head_dim, rng, 0.45f);
    std::vector<uint16_t> k_cache = random_halves(
        static_cast<size_t>(max_context) * kv_heads * head_dim, rng, 0.5f);
    std::vector<uint16_t> v_cache = random_halves(
        static_cast<size_t>(max_context) * kv_heads * head_dim, rng, 0.5f);
    const size_t poison_start =
        static_cast<size_t>(context_len) * kv_heads * head_dim;
    std::fill(k_cache.begin() + poison_start, k_cache.end(), float_to_half(40.0f));
    std::fill(v_cache.begin() + poison_start, v_cache.end(), float_to_half(-40.0f));
    DeviceBuffer<uint16_t> d_q(q), d_k(k_cache), d_v(v_cache);
    DeviceBuffer<uint16_t> d_out(static_cast<size_t>(q_heads) * head_dim);
    DeviceBuffer<float> d_scores(static_cast<size_t>(q_heads) * context_len);
    expect(pocket::qwen_gqa_decode_attention_f16(
               d_q.get(), d_k.get(), d_v.get(), d_out.get(), d_scores.get(),
               q_heads, kv_heads, head_dim, context_len, max_context),
           "GQA decode launch");
    sync_or_throw("GQA decode");
    expect_half_close(d_out.download(),
                      attention_reference(q, k_cache, v_cache, 1, q_heads,
                                          kv_heads, head_dim, context_len - 1),
                      5.0e-3, 2.0e-3, "GQA decode output");
    const std::vector<float> scores = d_scores.download();
    int bad_probability_rows = 0;
    for (int head = 0; head < q_heads; ++head) {
        double sum = 0.0;
        for (int pos = 0; pos < context_len; ++pos) {
            const float value = scores[static_cast<size_t>(head) * context_len + pos];
            if (!(value >= 0.0f)) ++bad_probability_rows;
            sum += value;
        }
        if (std::fabs(sum - 1.0) > 2.0e-4) ++bad_probability_rows;
    }
    std::string score_detail;
    if (bad_probability_rows != 0) {
        score_detail = " rows=" + std::to_string(bad_probability_rows) + " sums=";
        for (int head = 0; head < q_heads; ++head) {
            double sum = 0.0;
            for (int pos = 0; pos < context_len; ++pos) {
                sum += scores[static_cast<size_t>(head) * context_len + pos];
            }
            score_detail += std::to_string(sum) + (head + 1 == q_heads ? "" : ",");
        }
    }
    expect(bad_probability_rows == 0,
           "GQA decode score scratch holds probabilities" + score_detail);
}

void test_gqa_decode_vectorized(std::mt19937& rng) {
    // Exercise the tuned Qwen3.8-27B TP4 geometry and a partial 64-position
    // tile. This must take the AscendC vectorized path rather than the scalar
    // fallback used by the general-shape decode case above.
    const int q_heads = 6;
    const int kv_heads = 1;
    const int head_dim = 256;
    const int context_len = 65;
    const int max_context = 80; // Also exercises the serialized scratch-row path.
    const std::vector<uint16_t> q = random_halves(
        static_cast<size_t>(q_heads) * head_dim, rng, 0.45f);
    std::vector<uint16_t> k_cache = random_halves(
        static_cast<size_t>(max_context) * kv_heads * head_dim, rng, 0.5f);
    std::vector<uint16_t> v_cache = random_halves(
        static_cast<size_t>(max_context) * kv_heads * head_dim, rng, 0.5f);
    const size_t poison_start =
        static_cast<size_t>(context_len) * kv_heads * head_dim;
    std::fill(k_cache.begin() + poison_start, k_cache.end(),
              float_to_half(40.0f));
    std::fill(v_cache.begin() + poison_start, v_cache.end(),
              float_to_half(-40.0f));

    DeviceBuffer<uint16_t> d_q(q), d_k(k_cache), d_v(v_cache);
    DeviceBuffer<uint16_t> d_out(static_cast<size_t>(q_heads) * head_dim);
    DeviceBuffer<float> d_scores(static_cast<size_t>(q_heads) * context_len);
    expect(pocket::qwen_gqa_decode_attention_f16(
               d_q.get(), d_k.get(), d_v.get(), d_out.get(), d_scores.get(),
               q_heads, kv_heads, head_dim, context_len, max_context),
           "vectorized GQA decode launch");
    sync_or_throw("vectorized GQA decode");
    expect_half_close(
        d_out.download(),
        attention_reference(q, k_cache, v_cache, 1, q_heads, kv_heads,
                            head_dim, context_len - 1),
        5.0e-3, 2.0e-3, "vectorized GQA decode output");

    const std::vector<float> scores = d_scores.download();
    int bad_probability_rows = 0;
    for (int head = 0; head < q_heads; ++head) {
        double sum = 0.0;
        for (int pos = 0; pos < context_len; ++pos) {
            const float value = scores[static_cast<size_t>(head) * context_len + pos];
            if (!(value >= 0.0f) || !std::isfinite(value)) {
                ++bad_probability_rows;
            }
            sum += value;
        }
        if (std::fabs(sum - 1.0) > 2.0e-4) ++bad_probability_rows;
    }
    expect(bad_probability_rows == 0,
           "vectorized GQA decode scratch holds probabilities");

    // Contexts divisible by five 64-position tiles use the context-split path:
    // five partial softmax/value blocks per query head, followed by merge and
    // scratch normalization. Keep this separate from the partial-tile check above
    // so both dispatch branches remain covered by the hardware suite.
    const int split_context_len = 640;
    const int split_max_context = 656;
    const std::vector<uint16_t> split_k_cache = random_halves(
        static_cast<size_t>(split_max_context) * kv_heads * head_dim, rng, 0.5f);
    const std::vector<uint16_t> split_v_cache = random_halves(
        static_cast<size_t>(split_max_context) * kv_heads * head_dim, rng, 0.5f);
    DeviceBuffer<uint16_t> split_k(split_k_cache), split_v(split_v_cache);
    DeviceBuffer<uint16_t> split_out(static_cast<size_t>(q_heads) * head_dim);
    DeviceBuffer<float> split_scores(
        static_cast<size_t>(q_heads) * split_context_len);
    expect(pocket::qwen_gqa_decode_attention_f16(
               d_q.get(), split_k.get(), split_v.get(), split_out.get(),
               split_scores.get(), q_heads, kv_heads, head_dim,
               split_context_len, split_max_context),
           "context-split GQA decode launch");
    sync_or_throw("context-split GQA decode");
    expect_half_close(
        split_out.download(),
        attention_reference(q, split_k_cache, split_v_cache, 1, q_heads,
                            kv_heads, head_dim, split_context_len - 1),
        5.0e-3, 2.0e-3, "context-split GQA decode output");
    const std::vector<float> split_probability = split_scores.download();
    int bad_split_probability_rows = 0;
    for (int head = 0; head < q_heads; ++head) {
        double sum = 0.0;
        for (int pos = 0; pos < split_context_len; ++pos) {
            const float value = split_probability[
                static_cast<size_t>(head) * split_context_len + pos];
            if (!(value >= 0.0f) || !std::isfinite(value)) {
                ++bad_split_probability_rows;
            }
            sum += value;
        }
        if (std::fabs(sum - 1.0) > 2.0e-4) {
            ++bad_split_probability_rows;
        }
    }
    expect(bad_split_probability_rows == 0,
           "context-split GQA scratch holds probabilities");

    // From 30 whole 64-position tiles the dispatch switches to the KV-sharing
    // geometry: one context range per AI core, every query head swept across the
    // K/V tile already in UB. That path keeps raw scaled scores in scratch during
    // the sweep and produces the probability plane in its own normalize stage, so
    // both the output and the scratch contract need separate coverage here.
    // Must clear the dispatch threshold in qwen_ascend_ops_launch.cpp, which is
    // set by measured crossover rather than by the kernel's 30-tile minimum.
    const int shared_context_len = 16384;
    const int shared_max_context = 16400;
    const std::vector<uint16_t> shared_k_cache = random_halves(
        static_cast<size_t>(shared_max_context) * kv_heads * head_dim, rng, 0.5f);
    const std::vector<uint16_t> shared_v_cache = random_halves(
        static_cast<size_t>(shared_max_context) * kv_heads * head_dim, rng, 0.5f);
    DeviceBuffer<uint16_t> shared_k(shared_k_cache), shared_v(shared_v_cache);
    DeviceBuffer<uint16_t> shared_out(static_cast<size_t>(q_heads) * head_dim);
    DeviceBuffer<float> shared_scores(
        static_cast<size_t>(q_heads) * shared_context_len);
    expect(pocket::qwen_gqa_decode_attention_f16(
               d_q.get(), shared_k.get(), shared_v.get(), shared_out.get(),
               shared_scores.get(), q_heads, kv_heads, head_dim,
               shared_context_len, shared_max_context),
           "KV-sharing GQA decode launch");
    sync_or_throw("KV-sharing GQA decode");
    expect_half_close(
        shared_out.download(),
        attention_reference(q, shared_k_cache, shared_v_cache, 1, q_heads,
                            kv_heads, head_dim, shared_context_len - 1),
        5.0e-3, 2.0e-3, "KV-sharing GQA decode output");
    const std::vector<float> shared_probability = shared_scores.download();
    int bad_shared_probability_rows = 0;
    for (int head = 0; head < q_heads; ++head) {
        double sum = 0.0;
        for (int pos = 0; pos < shared_context_len; ++pos) {
            const float value = shared_probability[
                static_cast<size_t>(head) * shared_context_len + pos];
            if (!(value >= 0.0f) || !std::isfinite(value)) {
                ++bad_shared_probability_rows;
            }
            sum += value;
        }
        if (std::fabs(sum - 1.0) > 2.0e-4) {
            ++bad_shared_probability_rows;
        }
    }
    expect(bad_shared_probability_rows == 0,
           "KV-sharing GQA scratch holds probabilities");
}

void test_argmax(std::mt19937& rng) {
    (void)rng;
    const int rows = 5;
    const int count = 263;
    const int token_offset = -17;
    std::vector<float> logits(static_cast<size_t>(rows) * count, -10.0f);
    logits[7] = 3.0f;
    logits[2] = 3.0f;
    logits[static_cast<size_t>(count) + 5] = 4.0f;
    logits[static_cast<size_t>(count) + 9] = 4.0f;
    logits[static_cast<size_t>(2) * count + 262] = 9.0f;
    logits[static_cast<size_t>(2) * count + 3] = 8.0f;
    logits[static_cast<size_t>(3) * count] = -1.0f;
    logits[static_cast<size_t>(3) * count + 11] = -1.0f;
    logits[static_cast<size_t>(4) * count + 129] = 0.5f;
    logits[static_cast<size_t>(4) * count + 130] = 0.5f;

    DeviceBuffer<float> d_logits(logits);
    DeviceBuffer<int> d_tokens(rows);
    DeviceBuffer<float> d_values(rows);
    expect(pocket::qwen_argmax_fp32_rows(
               d_logits.get(), d_tokens.get(), d_values.get(), rows, count,
               token_offset),
           "argmax launch");
    sync_or_throw("argmax");

    const std::vector<int> expected_tokens = {-15, -12, 245, -17, 112};
    const std::vector<float> expected_values = {3.0f, 4.0f, 9.0f, -1.0f, 0.5f};
    expect(d_tokens.download() == expected_tokens,
           "argmax selects lowest global id on ties");
    expect(d_values.download() == expected_values,
           "argmax returns winning logits");

    // The decode path calls this with rows == 1 and a vocabulary-wide row, and it
    // allocates the 4-byte token and logit outputs from one workspace arena, so
    // they can land in a single 32-byte cache line. Cover that shape explicitly:
    // the multi-row case above would not catch a single-row store defect.
    const int wide = 248320;
    std::vector<float> row(static_cast<size_t>(wide), -3.0f);
    row[wide - 1] = 11.5f;
    row[4096] = 11.0f;
    DeviceBuffer<float> d_row(row);
    DeviceBuffer<int> d_token(1);
    DeviceBuffer<float> d_value(1);
    expect(pocket::qwen_argmax_fp32_rows(d_row.get(), d_token.get(),
                                       d_value.get(), 1, wide, 0),
           "argmax single-row launch");
    sync_or_throw("argmax single row");
    expect(d_token.download() == std::vector<int>{wide - 1},
           "argmax single row selects the last index");
    expect(d_value.download() == std::vector<float>{11.5f},
           "argmax single row returns its logit");
}

void test_gqa_prefill(std::mt19937& rng) {
    const int rows = 3;
    const int q_heads = 6;
    const int kv_heads = 2;
    const int head_dim = 32;
    const int position_offset = 2;
    const int max_context = 9;
    const std::vector<uint16_t> q = random_halves(
        static_cast<size_t>(rows) * q_heads * head_dim, rng, 0.45f);
    std::vector<uint16_t> k_cache = random_halves(
        static_cast<size_t>(max_context) * kv_heads * head_dim, rng, 0.5f);
    std::vector<uint16_t> v_cache = random_halves(
        static_cast<size_t>(max_context) * kv_heads * head_dim, rng, 0.5f);
    const size_t poison_start = static_cast<size_t>(position_offset + rows) *
                                kv_heads * head_dim;
    std::fill(k_cache.begin() + poison_start, k_cache.end(), float_to_half(40.0f));
    std::fill(v_cache.begin() + poison_start, v_cache.end(), float_to_half(-40.0f));
    DeviceBuffer<uint16_t> d_q(q), d_k(k_cache), d_v(v_cache);
    DeviceBuffer<uint16_t> d_out(static_cast<size_t>(rows) * q_heads * head_dim);
    expect(pocket::qwen_gqa_prefill_attention_f16(
               d_q.get(), d_k.get(), d_v.get(), d_out.get(), rows, q_heads,
               kv_heads, head_dim, position_offset, max_context),
           "GQA prefill launch");
    sync_or_throw("GQA prefill");
    expect_half_close(d_out.download(),
                      attention_reference(q, k_cache, v_cache, rows, q_heads,
                                          kv_heads, head_dim, position_offset),
                      5.0e-3, 2.0e-3, "GQA prefill output");
}

void test_gqa_verify(std::mt19937& rng) {
    const int rows = 4;
    const int q_heads = 6;
    const int kv_heads = 2;
    const int head_dim = 32;
    const int position_offset = 3;
    const int max_context = 10;
    const int splits = 5;
    const std::vector<uint16_t> q = random_halves(
        static_cast<size_t>(rows) * q_heads * head_dim, rng, 0.45f);
    std::vector<uint16_t> k_cache = random_halves(
        static_cast<size_t>(max_context) * kv_heads * head_dim, rng, 0.5f);
    std::vector<uint16_t> v_cache = random_halves(
        static_cast<size_t>(max_context) * kv_heads * head_dim, rng, 0.5f);
    const size_t poison_start = static_cast<size_t>(position_offset + rows) *
                                kv_heads * head_dim;
    std::fill(k_cache.begin() + poison_start, k_cache.end(), float_to_half(40.0f));
    std::fill(v_cache.begin() + poison_start, v_cache.end(), float_to_half(-40.0f));
    DeviceBuffer<uint16_t> d_q(q), d_k(k_cache), d_v(v_cache);
    DeviceBuffer<uint16_t> d_out(static_cast<size_t>(rows) * q_heads * head_dim);
    DeviceBuffer<float> d_partial(static_cast<size_t>(rows) * q_heads * splits *
                                  (head_dim + 2));
    expect(pocket::qwen_gqa_verify_attention_f16(
               d_q.get(), d_k.get(), d_v.get(), d_out.get(), d_partial.get(),
               rows, q_heads, kv_heads, head_dim, position_offset, max_context,
               splits),
           "GQA verify launch");
    sync_or_throw("GQA verify");
    expect_half_close(d_out.download(),
                      attention_reference(q, k_cache, v_cache, rows, q_heads,
                                          kv_heads, head_dim, position_offset),
                      5.0e-3, 2.0e-3, "GQA verify output");
    const std::vector<float> partial = d_partial.download();
    int nonfinite = 0;
    for (size_t group = 0;
         group < static_cast<size_t>(rows) * q_heads * splits; ++group) {
        const float maximum = partial[group * (head_dim + 2)];
        const float denominator = partial[group * (head_dim + 2) + 1];
        if (!std::isfinite(maximum) && denominator != 0.0f) ++nonfinite;
        if (!std::isfinite(denominator) || denominator < 0.0f) ++nonfinite;
    }
    expect(nonfinite == 0, "GQA verify fills valid split partials");
}

void test_argument_rejection() {
    DeviceBuffer<uint16_t> half(1024);
    DeviceBuffer<float> real(65536);
    expect(!pocket::qwen_normalize_gated_delta_qk_f16(
               nullptr, half.get(), real.get(), real.get(), 1, 1, kKeyDim),
           "normalize rejects null input");
    expect(!pocket::qwen_gated_delta_sequence_f16(
               real.get(), half.get(), half.get(), half.get(), half.get(),
               half.get(), half.get(), 1, 3, 2, kKeyDim, kValueDim, 0.1f),
           "recurrence rejects a bad head ratio");
    expect(!pocket::qwen_gated_delta_step_f16(
               real.get(), half.get(), half.get(), half.get(), half.get(),
               half.get(), half.get(), 2, 1, 64, kValueDim, 0.1f),
           "recurrence rejects unsupported key width");
    expect(!pocket::qwen_gated_delta_step_f16(
               real.get(), half.get(), half.get(), half.get(), half.get(),
               half.get(), half.get(), 2, 1, kKeyDim, kValueDim, 0.0f),
           "recurrence rejects nonpositive q scale");
    expect(!pocket::qwen_linear_attn_gates_f16(
               half.get(), half.get(), half.get(), half.get(), half.get(),
               half.get(), 0, 4),
           "gates reject zero rows");
    expect(!pocket::qwen_causal_depthwise_conv_silu_f16(
               half.get(), half.get(), half.get(), half.get(), 1, 8, 9, true),
           "convolution rejects a kernel above eight");
    expect(!pocket::qwen_partial_rope_rows_f16(
               half.get(), half.get(), 0, 1, 63, 10000.0f, 2, 1, 64),
           "rope rejects an odd rotary dimension");
    expect(!pocket::qwen_append_kv_cache_f16(
               half.get(), half.get(), half.get(), half.get(), 2, 1, 16, 3, 4),
           "KV append rejects cache overflow");
    expect(!pocket::qwen_gqa_decode_attention_f16(
               half.get(), half.get(), half.get(), half.get(), real.get(), 3, 2,
               16, 2, 4),
           "GQA decode rejects a bad head ratio");
    expect(!pocket::qwen_gqa_prefill_attention_f16(
               half.get(), half.get(), half.get(), half.get(), 3, 2, 1, 16, 2,
               4),
           "GQA prefill rejects context overflow");
    expect(!pocket::qwen_gqa_verify_attention_f16(
               half.get(), half.get(), half.get(), half.get(), real.get(), 2, 2,
               1, 16, 1, 4, 0),
           "GQA verify rejects zero splits");
}

}  // namespace

int main(int argc, char** argv) {
    int device = 0;
    std::string only;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--device" && i + 1 < argc) {
            device = std::stoi(argv[++i]);
        } else if (arg == "--only" && i + 1 < argc) {
            only = argv[++i];
        } else {
            throw std::runtime_error("unknown or incomplete argument: " + arg);
        }
    }
    if (!pocket::device_runtime_available()) {
        std::cout << "[SKIP] no device runtime available\n";
        return 0;
    }
    if (!pocket::device_set(device)) {
        std::cout << "[SKIP] device_set failed for device " << device << "\n";
        return 0;
    }
    std::cout << "backend=" << pocket::device_backend_name() << " device=" << device
              << "\n";

    std::mt19937 rng(20260830u);
    const std::vector<std::pair<const char*, void (*)(std::mt19937&)>> suite = {
        {"normalize_qk", test_normalize_qk},
        {"gated_delta", test_gated_delta_recurrence},
        {"linear_attn_gates", test_linear_attn_gates},
        {"causal_conv", test_causal_conv},
        {"partial_rope", test_partial_rope},
        {"append_kv", test_append_kv},
        {"gqa_decode", test_gqa_decode},
        {"gqa_decode_vectorized", test_gqa_decode_vectorized},
        {"gqa_prefill", test_gqa_prefill},
        {"gqa_verify", test_gqa_verify},
        {"argmax", test_argmax},
    };
    for (const auto& entry : suite) {
        if (!only.empty() &&
            ("," + only + ",").find("," + std::string(entry.first) + ",") ==
                std::string::npos) {
            continue;
        }
        std::cout << "[run] " << entry.first << std::endl;
        try {
            entry.second(rng);
        } catch (const std::exception& error) {
            ++failures;
            std::cout << "  FAIL " << entry.first << " threw: " << error.what()
                      << "\n";
        }
    }
    if (only.empty() ||
        ("," + only + ",").find(",argument_rejection,") != std::string::npos) {
        std::cout << "[run] argument_rejection" << std::endl;
        try {
            test_argument_rejection();
        } catch (const std::exception& error) {
            ++failures;
            std::cout << "  FAIL argument_rejection threw: " << error.what() << "\n";
        }
    }

    if (failures != 0) {
        std::cout << "FAIL " << failures << " of " << checks << " checks\n";
        return 1;
    }
    std::cout << "PASS (" << checks << " checks)\n";
    return 0;
}
