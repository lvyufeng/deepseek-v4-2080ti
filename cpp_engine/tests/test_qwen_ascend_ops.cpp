// Correctness test for the aclnn-backed Qwen operators.
//
// References are computed on the host in double precision from the operator's
// definition, not from the CUDA kernels. Comparing two implementations of the same
// misunderstanding proves nothing; comparing against the math catches a convention
// error on either side.
//
// The operators are called through the neutral names in qwen_ops.hpp, so this test
// is written once and would be meaningful on either backend. It links
// pocket_cpp_core, so it is built only where those operators exist.
//
//   ./tests/test_qwen_ascend_ops [--device N]

#include "device_runtime.hpp"
#include "qwen_ops.hpp"
#include "qwen_weights.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <random>
#include <utility>
#include <stdexcept>
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
    ~DeviceBuffer() { pocket::device_free(ptr_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    T* get() const { return ptr_; }

    explicit DeviceBuffer(const std::vector<T>& host) : DeviceBuffer(host.size()) {
        upload(host);
    }

    void upload(const std::vector<T>& host) {
        if (host.empty()) return;
        if (!pocket::memcpy_h2d(ptr_, host.data(), host.size() * sizeof(T))) {
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

// Worst relative error against a double-precision reference. Absolute floor keeps
// values that reference near zero from dominating.
double worst_relative_error(const std::vector<uint16_t>& got,
                            const std::vector<double>& want) {
    double worst = 0.0;
    for (size_t i = 0; i < want.size(); ++i) {
        const double reference = want[i];
        const double actual = half_to_float(got[i]);
        const double scale = std::fabs(reference) > 1e-2 ? std::fabs(reference) : 1e-2;
        worst = std::max(worst, std::fabs(actual - reference) / scale);
    }
    return worst;
}

std::vector<uint16_t> random_halves(size_t count, std::mt19937& rng, float scale) {
    std::uniform_real_distribution<float> dist(-scale, scale);
    std::vector<uint16_t> out(count);
    for (uint16_t& value : out) value = float_to_half(dist(rng));
    return out;
}

void sync_or_throw(const std::string& what) {
    if (!pocket::device_synchronize()) {
        throw std::runtime_error("device_synchronize failed after " + what);
    }
}

// y[b, r] = sum_c x[b, c] * w[r, c]. Hidden 5120 is the real Qwen width, which is
// what makes the accumulated FP16 rounding representative.
void test_matmul(std::mt19937& rng) {
    const int batch = 4;
    const int rows = 96;
    const int cols = 5120;
    // A padded input stride, which is the case an understated storage shape used
    // to reject. The padding holds a recognizable value so a stride mistake shows
    // up as a wrong result rather than as harmless zeros.
    const int x_stride = cols + 96;
    std::vector<uint16_t> x_padded(static_cast<size_t>(batch) * x_stride,
                                   float_to_half(9.0f));
    const std::vector<uint16_t> x =
        random_halves(static_cast<size_t>(batch) * cols, rng, 0.5f);
    for (int b = 0; b < batch; ++b) {
        for (int c = 0; c < cols; ++c) {
            x_padded[static_cast<size_t>(b) * x_stride + c] =
                x[static_cast<size_t>(b) * cols + c];
        }
    }
    const std::vector<uint16_t> w =
        random_halves(static_cast<size_t>(rows) * cols, rng, 0.05f);
    DeviceBuffer<uint16_t> d_x_padded(x_padded);
    DeviceBuffer<uint16_t> d_y_padded(static_cast<size_t>(batch) * rows);
    DeviceBuffer<uint16_t> d_x(x);
    DeviceBuffer<uint16_t> d_w(w);
    DeviceBuffer<uint16_t> d_y(static_cast<size_t>(batch) * rows);
    DeviceBuffer<float> d_y32(static_cast<size_t>(batch) * rows);

    expect(pocket::qwen_fp16_matmul_rows_f16(d_x.get(), d_w.get(), d_y.get(), batch,
                                           rows, cols, cols, rows, cols),
           "matmul f16 launch");
    expect(pocket::qwen_fp16_matmul_rows_f16_f32(d_x.get(), d_w.get(), d_y32.get(),
                                               batch, rows, cols, cols, rows, cols),
           "matmul f32 launch");
    expect(pocket::qwen_fp16_matmul_rows_f16(d_x_padded.get(), d_w.get(),
                                           d_y_padded.get(), batch, rows, cols,
                                           x_stride, rows, cols),
           "matmul padded-input launch");
    sync_or_throw("matmul");

    std::vector<double> want(static_cast<size_t>(batch) * rows);
    for (int b = 0; b < batch; ++b) {
        for (int r = 0; r < rows; ++r) {
            double sum = 0.0;
            for (int c = 0; c < cols; ++c) {
                sum += static_cast<double>(
                           half_to_float(x[static_cast<size_t>(b) * cols + c])) *
                       static_cast<double>(
                           half_to_float(w[static_cast<size_t>(r) * cols + c]));
            }
            want[static_cast<size_t>(b) * rows + r] = sum;
        }
    }
    // A 5120-term FP16 dot product accumulated in FP32 lands within a few 1e-3
    // relative; the measured value on this hardware is 3.7e-04.
    const double f16_error = worst_relative_error(d_y.download(), want);
    expect(f16_error < 5e-3, "matmul f16 error " + std::to_string(f16_error));

    // The padded input must give bit-identical results to the dense one: same
    // values, same accumulation order, only the row pitch differs.
    expect(d_y_padded.download() == d_y.download(),
           "a padded input stride gives the same result as a dense one");

    const std::vector<float> got32 = d_y32.download();
    double worst32 = 0.0;
    for (size_t i = 0; i < want.size(); ++i) {
        const double scale = std::max(std::fabs(want[i]), 1e-2);
        worst32 = std::max(worst32, std::fabs(got32[i] - want[i]) / scale);
    }
    expect(worst32 < 5e-3, "matmul f32 error " + std::to_string(worst32));
}

// SwiGLU fused into the two projections: y = silu(x @ gate^T) * (x @ up^T). Run
// with a padded y_stride, which is the case the operator's per-row loop exists for
// and where a pitch mistake would corrupt a neighbouring row.
void test_swiglu_matmul(std::mt19937& rng) {
    const int batch = 3;
    const int rows = 128;
    const int cols = 512;
    const int y_stride = rows + 64;
    const std::vector<uint16_t> x =
        random_halves(static_cast<size_t>(batch) * cols, rng, 0.5f);
    const std::vector<uint16_t> gate =
        random_halves(static_cast<size_t>(rows) * cols, rng, 0.1f);
    const std::vector<uint16_t> up =
        random_halves(static_cast<size_t>(rows) * cols, rng, 0.1f);

    DeviceBuffer<uint16_t> d_x(x);
    DeviceBuffer<uint16_t> d_gate(gate);
    DeviceBuffer<uint16_t> d_up(up);
    DeviceBuffer<uint16_t> d_y(static_cast<size_t>(batch) * y_stride);
    expect(pocket::qwen_fp16_swiglu_matmul_rows_f16(d_x.get(), d_gate.get(),
                                                  d_up.get(), d_y.get(), batch,
                                                  rows, cols, cols, y_stride, cols),
           "swiglu launch");
    sync_or_throw("swiglu");

    const std::vector<uint16_t> got = d_y.download();
    std::vector<uint16_t> payload(static_cast<size_t>(batch) * rows);
    std::vector<double> want(payload.size());
    int spill = 0;
    for (int b = 0; b < batch; ++b) {
        for (int r = 0; r < rows; ++r) {
            double gate_sum = 0.0;
            double up_sum = 0.0;
            for (int c = 0; c < cols; ++c) {
                const double value =
                    half_to_float(x[static_cast<size_t>(b) * cols + c]);
                gate_sum += value *
                            half_to_float(gate[static_cast<size_t>(r) * cols + c]);
                up_sum +=
                    value * half_to_float(up[static_cast<size_t>(r) * cols + c]);
            }
            want[static_cast<size_t>(b) * rows + r] =
                (gate_sum / (1.0 + std::exp(-gate_sum))) * up_sum;
            payload[static_cast<size_t>(b) * rows + r] =
                got[static_cast<size_t>(b) * y_stride + r];
        }
        for (int c = rows; c < y_stride; ++c) {
            if (got[static_cast<size_t>(b) * y_stride + c] != 0) ++spill;
        }
    }
    const double error = worst_relative_error(payload, want);
    expect(error < 1e-2, "swiglu error " + std::to_string(error));
    expect(spill == 0, "swiglu wrote " + std::to_string(spill) +
                           " elements into the row padding");
}

// rms(x) * gamma, where gamma already carries the +1 fold on Ascend. The test
// applies the same policy to its reference so it checks the operator rather than
// re-deriving the convention.
void test_rmsnorm(std::mt19937& rng) {
    const int rows = 6;
    const int cols = 5120;
    const float eps = 1e-6f;
    const std::vector<uint16_t> x =
        random_halves(static_cast<size_t>(rows) * cols, rng, 1.0f);
    const std::vector<uint16_t> raw_gamma =
        random_halves(static_cast<size_t>(cols), rng, 0.1f);

    // What the loader would have uploaded: on Ascend, 1 + gamma folded in FP16.
    pocket::QwenTensorRef ref;
    ref.name = "model.language_model.layers.0.input_layernorm.weight";
    pocket::QwenHostTensor host;
    host.device_dtype = pocket::SafeDType::F16;
    host.bytes.resize(raw_gamma.size() * sizeof(uint16_t));
    std::memcpy(host.bytes.data(), raw_gamma.data(), host.bytes.size());
    pocket::qwen_apply_norm_gamma_policy(ref, host);
    std::vector<uint16_t> gamma(raw_gamma.size());
    std::memcpy(gamma.data(), host.bytes.data(), host.bytes.size());

    const bool folded = pocket::device_backend() == pocket::DeviceBackend::Ascend;
    expect(folded == (gamma != raw_gamma),
           "gamma fold applied exactly on the Ascend backend");

    DeviceBuffer<uint16_t> d_x(x);
    DeviceBuffer<uint16_t> d_gamma(gamma);
    DeviceBuffer<uint16_t> d_y(x.size());
    expect(pocket::qwen_rmsnorm_fp16_gamma_rows_f16(d_x.get(), d_gamma.get(),
                                                  d_y.get(), rows, cols, eps),
           "rmsnorm launch");
    sync_or_throw("rmsnorm");

    // Reference: the (1 + weight) convention against the *raw* checkpoint gamma,
    // which is the behaviour both backends have to produce however they get there.
    std::vector<double> want(x.size());
    for (int r = 0; r < rows; ++r) {
        double sum_squares = 0.0;
        for (int c = 0; c < cols; ++c) {
            const double value =
                half_to_float(x[static_cast<size_t>(r) * cols + c]);
            sum_squares += value * value;
        }
        const double inv =
            1.0 / std::sqrt(sum_squares / static_cast<double>(cols) + eps);
        for (int c = 0; c < cols; ++c) {
            const double value =
                half_to_float(x[static_cast<size_t>(r) * cols + c]);
            const double weight =
                1.0 + half_to_float(raw_gamma[static_cast<size_t>(c)]);
            want[static_cast<size_t>(r) * cols + c] = value * inv * weight;
        }
    }
    // Folding 1+gamma into FP16 costs a relative ~2^-11 on the weight, so the
    // tolerance here is wider than the matmul's on purpose.
    const double error = worst_relative_error(d_y.download(), want);
    expect(error < 2e-3, "rmsnorm error " + std::to_string(error));
}

// residual = hidden + delta; normalized = rms(residual) * gamma. Both outputs are
// checked: the residual is what the next layer consumes.
void test_residual_add_rmsnorm(std::mt19937& rng) {
    const int rows = 3;
    const int cols = 5120;
    const float eps = 1e-6f;
    const std::vector<uint16_t> hidden =
        random_halves(static_cast<size_t>(rows) * cols, rng, 1.0f);
    const std::vector<uint16_t> delta =
        random_halves(static_cast<size_t>(rows) * cols, rng, 0.5f);
    std::vector<uint16_t> gamma(static_cast<size_t>(cols),
                                float_to_half(1.0f));  // pre-folded (1 + 0)

    DeviceBuffer<uint16_t> d_hidden(hidden);
    DeviceBuffer<uint16_t> d_delta(delta);
    DeviceBuffer<uint16_t> d_gamma(gamma);
    DeviceBuffer<uint16_t> d_residual(hidden.size());
    DeviceBuffer<uint16_t> d_normalized(hidden.size());
    expect(pocket::qwen_residual_add_rmsnorm_fp16_gamma_rows_f16(
               d_hidden.get(), d_delta.get(), d_gamma.get(), d_residual.get(),
               d_normalized.get(), rows, cols, eps),
           "residual add rmsnorm launch");
    sync_or_throw("residual add rmsnorm");

    // The sum is rounded to FP16 before the sum of squares, matching the CUDA
    // kernel: the residual buffer is a real FP16 tensor, not an FP32 accumulator.
    std::vector<uint16_t> residual(hidden.size());
    for (size_t i = 0; i < hidden.size(); ++i) {
        residual[i] = float_to_half(half_to_float(hidden[i]) +
                                    half_to_float(delta[i]));
    }
    const std::vector<uint16_t> got_residual = d_residual.download();
    int residual_mismatches = 0;
    for (size_t i = 0; i < residual.size(); ++i) {
        const float want = half_to_float(residual[i]);
        const float got = half_to_float(got_residual[i]);
        if (std::fabs(got - want) > 1e-3f * std::max(std::fabs(want), 1.0f)) {
            ++residual_mismatches;
        }
    }
    expect(residual_mismatches == 0,
           "residual sum mismatches " + std::to_string(residual_mismatches));

    std::vector<double> want(hidden.size());
    for (int r = 0; r < rows; ++r) {
        double sum_squares = 0.0;
        for (int c = 0; c < cols; ++c) {
            const double value =
                half_to_float(residual[static_cast<size_t>(r) * cols + c]);
            sum_squares += value * value;
        }
        const double inv =
            1.0 / std::sqrt(sum_squares / static_cast<double>(cols) + eps);
        for (int c = 0; c < cols; ++c) {
            want[static_cast<size_t>(r) * cols + c] =
                half_to_float(residual[static_cast<size_t>(r) * cols + c]) * inv;
        }
    }
    const double error = worst_relative_error(d_normalized.download(), want);
    expect(error < 2e-3, "residual add rmsnorm error " + std::to_string(error));
}

// rms(x) * gamma * silu(gate), with gamma applied directly (no +1).
void test_gated_rmsnorm(std::mt19937& rng) {
    const int rows = 4;
    const int cols = 256;  // the DeltaNet value head dim
    const float eps = 1e-6f;
    const std::vector<uint16_t> x =
        random_halves(static_cast<size_t>(rows) * cols, rng, 1.0f);
    const std::vector<uint16_t> gate =
        random_halves(static_cast<size_t>(rows) * cols, rng, 1.5f);
    const std::vector<uint16_t> gamma =
        random_halves(static_cast<size_t>(cols), rng, 0.5f);

    DeviceBuffer<uint16_t> d_x(x);
    DeviceBuffer<uint16_t> d_gate(gate);
    DeviceBuffer<uint16_t> d_gamma(gamma);
    DeviceBuffer<uint16_t> d_y(x.size());
    expect(pocket::qwen_gated_rmsnorm_fp16_gamma_rows_f16(
               d_x.get(), d_gamma.get(), d_gate.get(), d_y.get(), rows, cols, eps),
           "gated rmsnorm launch");
    sync_or_throw("gated rmsnorm");

    std::vector<double> want(x.size());
    for (int r = 0; r < rows; ++r) {
        double sum_squares = 0.0;
        for (int c = 0; c < cols; ++c) {
            const double value = half_to_float(x[static_cast<size_t>(r) * cols + c]);
            sum_squares += value * value;
        }
        const double inv =
            1.0 / std::sqrt(sum_squares / static_cast<double>(cols) + eps);
        for (int c = 0; c < cols; ++c) {
            const size_t index = static_cast<size_t>(r) * cols + c;
            const double normalized = half_to_float(x[index]) * inv *
                                      half_to_float(gamma[static_cast<size_t>(c)]);
            const double g = half_to_float(gate[index]);
            want[index] = normalized * (g / (1.0 + std::exp(-g)));
        }
    }
    const double error = worst_relative_error(d_y.download(), want);
    expect(error < 5e-3, "gated rmsnorm error " + std::to_string(error));
}

void test_elementwise(std::mt19937& rng) {
    const int rows = 5;
    const int cols = 1024;
    const size_t count = static_cast<size_t>(rows) * cols;
    const std::vector<uint16_t> a = random_halves(count, rng, 2.0f);
    const std::vector<uint16_t> b = random_halves(count, rng, 2.0f);

    {
        DeviceBuffer<uint16_t> d_y(a);
        DeviceBuffer<uint16_t> d_x(b);
        expect(pocket::qwen_add_inplace_f16(d_y.get(), d_x.get(),
                                          static_cast<int>(count)),
               "add inplace launch");
        sync_or_throw("add inplace");
        std::vector<double> want(count);
        for (size_t i = 0; i < count; ++i) {
            want[i] = static_cast<double>(half_to_float(a[i])) +
                      static_cast<double>(half_to_float(b[i]));
        }
        const double error = worst_relative_error(d_y.download(), want);
        expect(error < 2e-3, "add inplace error " + std::to_string(error));
    }
    {
        DeviceBuffer<uint16_t> d_x(a);
        DeviceBuffer<uint16_t> d_gate(b);
        DeviceBuffer<uint16_t> d_y(count);
        expect(pocket::qwen_sigmoid_mul_f16(d_x.get(), d_gate.get(), d_y.get(),
                                          static_cast<int>(count)),
               "sigmoid mul launch");
        sync_or_throw("sigmoid mul");
        std::vector<double> want(count);
        for (size_t i = 0; i < count; ++i) {
            const double gate = half_to_float(b[i]);
            want[i] = half_to_float(a[i]) * (1.0 / (1.0 + std::exp(-gate)));
        }
        const double error = worst_relative_error(d_y.download(), want);
        expect(error < 5e-3, "sigmoid mul error " + std::to_string(error));
    }
    {
        DeviceBuffer<uint16_t> d_gate(a);
        DeviceBuffer<uint16_t> d_up(b);
        DeviceBuffer<uint16_t> d_y(count);
        expect(pocket::qwen_silu_mul_rows_f16(d_gate.get(), d_up.get(), d_y.get(),
                                            rows, cols),
               "silu mul launch");
        sync_or_throw("silu mul");
        std::vector<double> want(count);
        for (size_t i = 0; i < count; ++i) {
            const double gate = half_to_float(a[i]);
            want[i] = (gate / (1.0 + std::exp(-gate))) * half_to_float(b[i]);
        }
        const double error = worst_relative_error(d_y.download(), want);
        expect(error < 5e-3, "silu mul error " + std::to_string(error));
    }
}

// The reshape-shaped operators are exact: any mismatch is a layout error, so these
// compare bit patterns rather than magnitudes.
void test_concat_rows(std::mt19937& rng) {
    const int rows = 7;
    const int cols = 320;
    (void)cols;
    const std::vector<uint16_t> left =
        random_halves(static_cast<size_t>(rows) * cols, rng, 1.0f);
    const std::vector<uint16_t> right =
        random_halves(static_cast<size_t>(rows) * cols, rng, 1.0f);
    DeviceBuffer<uint16_t> d_left(left);
    DeviceBuffer<uint16_t> d_right(right);
    DeviceBuffer<uint16_t> d_out(static_cast<size_t>(rows) * cols * 2);
    expect(pocket::qwen_concat_rows_f16(d_left.get(), d_right.get(), d_out.get(),
                                      rows, cols),
           "concat launch");
    sync_or_throw("concat");
    const std::vector<uint16_t> got = d_out.download();
    int mismatches = 0;
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            const size_t src = static_cast<size_t>(r) * cols + c;
            const size_t dst = static_cast<size_t>(r) * 2 * cols + c;
            if (got[dst] != left[src]) ++mismatches;
            if (got[dst + cols] != right[src]) ++mismatches;
        }
    }
    expect(mismatches == 0, "concat mismatches " + std::to_string(mismatches));
}

void test_copy_rows_strided(std::mt19937& rng) {
    const int rows = 7;
    const int cols = 320;
    (void)cols;
    // Destination rows are wider than the payload, which is how the engine
    // interleaves speculative taps.
    const int destination_stride = cols * 3;
    const std::vector<uint16_t> source =
        random_halves(static_cast<size_t>(rows) * cols, rng, 1.0f);
    DeviceBuffer<uint16_t> d_source(source);
    DeviceBuffer<uint16_t> d_dest(static_cast<size_t>(rows) *
                                  destination_stride);
    expect(pocket::qwen_copy_rows_strided_f16(d_source.get(), cols, d_dest.get(),
                                            destination_stride, rows, cols),
           "strided copy launch");
    sync_or_throw("strided copy");
    const std::vector<uint16_t> got = d_dest.download();
    int mismatches = 0;
    int spill = 0;
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if (got[static_cast<size_t>(r) * destination_stride + c] !=
                source[static_cast<size_t>(r) * cols + c]) {
                ++mismatches;
            }
        }
        // The padding must be untouched; a pitch mistake would write here.
        for (int c = cols; c < destination_stride; ++c) {
            if (got[static_cast<size_t>(r) * destination_stride + c] != 0) {
                ++spill;
            }
        }
    }
    expect(mismatches == 0,
           "strided copy mismatches " + std::to_string(mismatches));
    expect(spill == 0, "strided copy wrote " + std::to_string(spill) +
                           " elements into the row padding");
}

void test_split_packed_qkv(std::mt19937& rng) {
    const int rows = 7;
    const int cols = 320;
    (void)cols;
    const int key_dim = 192;
    const int value_dim = 256;
    const int packed = 2 * key_dim + value_dim;
    const std::vector<uint16_t> source =
        random_halves(static_cast<size_t>(rows) * packed, rng, 1.0f);
    DeviceBuffer<uint16_t> d_packed(source);
    DeviceBuffer<uint16_t> d_q(static_cast<size_t>(rows) * key_dim);
    DeviceBuffer<uint16_t> d_k(static_cast<size_t>(rows) * key_dim);
    DeviceBuffer<uint16_t> d_v(static_cast<size_t>(rows) * value_dim);
    expect(pocket::qwen_split_packed_qkv_f16(d_packed.get(), d_q.get(), d_k.get(),
                                           d_v.get(), rows, key_dim, value_dim),
           "split qkv launch");
    sync_or_throw("split qkv");
    const std::vector<uint16_t> q = d_q.download();
    const std::vector<uint16_t> k = d_k.download();
    const std::vector<uint16_t> v = d_v.download();
    int mismatches = 0;
    for (int r = 0; r < rows; ++r) {
        const size_t base = static_cast<size_t>(r) * packed;
        for (int c = 0; c < key_dim; ++c) {
            if (q[static_cast<size_t>(r) * key_dim + c] != source[base + c]) {
                ++mismatches;
            }
            if (k[static_cast<size_t>(r) * key_dim + c] !=
                source[base + key_dim + c]) {
                ++mismatches;
            }
        }
        for (int c = 0; c < value_dim; ++c) {
            if (v[static_cast<size_t>(r) * value_dim + c] !=
                source[base + 2 * key_dim + c]) {
                ++mismatches;
            }
        }
    }
    expect(mismatches == 0,
           "split qkv mismatches " + std::to_string(mismatches));
}

void test_split_rows_pair(std::mt19937& rng) {
    const int rows = 7;
    const int cols = 320;
    (void)cols;
    const int width = 288;
    const std::vector<uint16_t> source =
        random_halves(static_cast<size_t>(rows) * width * 2, rng, 1.0f);
    DeviceBuffer<uint16_t> d_packed(source);
    DeviceBuffer<uint16_t> d_first(static_cast<size_t>(rows) * width);
    DeviceBuffer<uint16_t> d_second(static_cast<size_t>(rows) * width);
    expect(pocket::qwen_split_rows_pair_f16(d_packed.get(), d_first.get(),
                                          d_second.get(), rows, width),
           "split rows pair launch");
    sync_or_throw("split rows pair");
    const std::vector<uint16_t> first = d_first.download();
    const std::vector<uint16_t> second = d_second.download();
    int mismatches = 0;
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < width; ++c) {
            const size_t base = static_cast<size_t>(r) * 2 * width;
            if (first[static_cast<size_t>(r) * width + c] != source[base + c]) {
                ++mismatches;
            }
            if (second[static_cast<size_t>(r) * width + c] !=
                source[base + width + c]) {
                ++mismatches;
            }
        }
    }
    expect(mismatches == 0,
           "split rows pair mismatches " + std::to_string(mismatches));
}

void test_split_q_gate(std::mt19937& rng) {
    const int rows = 7;
    const int cols = 320;
    (void)cols;
    // q and gate are interleaved per head, not per row: [row, head, 2*dim].
    const int q_heads = 6;
    const int head_dim = 128;
    const size_t total =
        static_cast<size_t>(rows) * q_heads * head_dim;
    const std::vector<uint16_t> source = random_halves(total * 2, rng, 1.0f);
    DeviceBuffer<uint16_t> d_source(source);
    DeviceBuffer<uint16_t> d_q(total);
    DeviceBuffer<uint16_t> d_gate(total);
    expect(pocket::qwen_split_q_gate_f16(d_source.get(), d_q.get(), d_gate.get(),
                                       rows, q_heads, head_dim),
           "split q gate launch");
    sync_or_throw("split q gate");
    const std::vector<uint16_t> q = d_q.download();
    const std::vector<uint16_t> gate = d_gate.download();
    int mismatches = 0;
    for (int r = 0; r < rows; ++r) {
        for (int h = 0; h < q_heads; ++h) {
            const size_t base =
                ((static_cast<size_t>(r) * q_heads) + h) * head_dim * 2;
            const size_t out =
                ((static_cast<size_t>(r) * q_heads) + h) * head_dim;
            for (int d = 0; d < head_dim; ++d) {
                if (q[out + d] != source[base + d]) ++mismatches;
                if (gate[out + d] != source[base + head_dim + d]) ++mismatches;
            }
        }
    }
    expect(mismatches == 0,
           "split q gate mismatches " + std::to_string(mismatches));
}

// Vocab-sharded embedding: tokens inside this rank's slice gather a row, tokens
// outside it must produce exact zeros so the all-reduce sums to the right row.
void test_embedding(std::mt19937& rng) {
    const int row_start = 1000;
    const int row_count = 64;
    const int cols = 512;
    const std::vector<uint16_t> table =
        random_halves(static_cast<size_t>(row_count) * cols, rng, 1.0f);
    const std::vector<int> tokens = {
        row_start,                  // first resident row
        row_start + row_count - 1,  // last resident row
        row_start - 1,              // just below the slice
        row_start + row_count,      // just above the slice
        row_start + 7,
        0,                          // far below
        999999,                     // far above
    };
    const int count = static_cast<int>(tokens.size());

    DeviceBuffer<uint16_t> d_table(table);
    DeviceBuffer<int> d_tokens(tokens);
    DeviceBuffer<uint16_t> d_out(static_cast<size_t>(count) * cols);
    expect(pocket::qwen_embedding_fp16_gather_f16(d_table.get(), d_tokens.get(),
                                                d_out.get(), count, cols,
                                                row_start, row_count),
           "embedding launch");
    sync_or_throw("embedding");

    const std::vector<uint16_t> got = d_out.download();
    int mismatches = 0;
    int nonzero_outside = 0;
    for (int i = 0; i < count; ++i) {
        const int token = tokens[static_cast<size_t>(i)];
        const bool resident = token >= row_start && token < row_start + row_count;
        for (int c = 0; c < cols; ++c) {
            const uint16_t value = got[static_cast<size_t>(i) * cols + c];
            if (resident) {
                const uint16_t want =
                    table[static_cast<size_t>(token - row_start) * cols + c];
                if (value != want) ++mismatches;
            } else if (half_to_float(value) != 0.0f) {
                ++nonzero_outside;
            }
        }
    }
    expect(mismatches == 0, "embedding mismatches " + std::to_string(mismatches));
    expect(nonzero_outside == 0,
           "embedding wrote " + std::to_string(nonzero_outside) +
               " nonzero elements for out-of-shard tokens");
}

// Gather then scatter has to round-trip: the packed buffer is the transaction
// snapshot the engine rolls speculative state back from.
void test_copy_regions(std::mt19937& rng) {
    // Deliberately uneven sizes, including one that is not a whole number of
    // 64 KiB blocks, since that is where a block-count bug would show.
    const std::vector<size_t> region_elements = {4096, 100, 40960, 1};
    // DeviceBuffer is intentionally not movable, so hold the regions by pointer
    // rather than by value.
    std::vector<std::unique_ptr<DeviceBuffer<uint16_t>>> buffers;
    std::vector<std::vector<uint16_t>> host_values;
    std::vector<pocket::QwenCopyRegion> regions;
    uint64_t packed_offset = 0;
    uint64_t total_blocks = 0;
    for (size_t elements : region_elements) {
        host_values.push_back(random_halves(elements, rng, 1.0f));
        buffers.push_back(
            std::make_unique<DeviceBuffer<uint16_t>>(host_values.back()));
        pocket::QwenCopyRegion region;
        region.device_address =
            reinterpret_cast<uint64_t>(buffers.back()->get());
        region.packed_offset = packed_offset;
        region.bytes = elements * sizeof(uint16_t);
        region.first_block = total_blocks;
        regions.push_back(region);
        packed_offset += region.bytes;
        total_blocks += pocket::qwen_copy_region_blocks(region.bytes);
    }

    DeviceBuffer<pocket::QwenCopyRegion> d_regions(regions);
    DeviceBuffer<uint8_t> d_packed(static_cast<size_t>(packed_offset));
    expect(pocket::qwen_gather_copy_regions(d_regions.get(),
                                          static_cast<int>(regions.size()),
                                          d_packed.get(), total_blocks),
           "gather regions launch");
    sync_or_throw("gather regions");

    // Overwrite every region, then scatter the snapshot back and require the
    // original bytes. Zeroing would pass even if scatter did nothing.
    for (size_t i = 0; i < buffers.size(); ++i) {
        std::vector<uint16_t> clobber(host_values[i].size(),
                                      float_to_half(-7.5f));
        buffers[i]->upload(clobber);
    }
    expect(pocket::qwen_scatter_copy_regions(d_regions.get(),
                                           static_cast<int>(regions.size()),
                                           d_packed.get(), total_blocks),
           "scatter regions launch");
    sync_or_throw("scatter regions");

    int mismatches = 0;
    for (size_t i = 0; i < buffers.size(); ++i) {
        const std::vector<uint16_t> got = buffers[i]->download();
        for (size_t j = 0; j < got.size(); ++j) {
            if (got[j] != host_values[i][j]) ++mismatches;
        }
    }
    expect(mismatches == 0,
           "region round-trip mismatches " + std::to_string(mismatches));

    // A descriptor table whose block accounting disagrees with total_blocks is a
    // corrupted snapshot, and must be refused rather than partially applied.
    expect(!pocket::qwen_gather_copy_regions(d_regions.get(),
                                           static_cast<int>(regions.size()),
                                           d_packed.get(), total_blocks + 1),
           "gather rejects an inconsistent block count");
}

// Rejected arguments must return false rather than launching, because the engine
// turns false into an exception naming the operator.
void test_argument_rejection() {
    DeviceBuffer<uint16_t> buffer(64);
    expect(!pocket::qwen_rmsnorm_fp16_gamma_rows_f16(nullptr, buffer.get(),
                                                   buffer.get(), 1, 8, 1e-6f),
           "rmsnorm rejects a null input");
    expect(!pocket::qwen_rmsnorm_fp16_gamma_rows_f16(buffer.get(), buffer.get(),
                                                   buffer.get(), 0, 8, 1e-6f),
           "rmsnorm rejects zero rows");
    expect(!pocket::qwen_silu_mul_rows_f16(buffer.get(), buffer.get(), buffer.get(),
                                         4, 0),
           "silu mul rejects zero cols");
    expect(!pocket::qwen_copy_rows_strided_f16(buffer.get(), 4, buffer.get(), 8, 2, 8),
           "strided copy rejects a source stride below the payload width");
    expect(!pocket::qwen_fp16_matmul_rows_f16(buffer.get(), buffer.get(),
                                            buffer.get(), 1, 4, 8, 4, 4, 8),
           "matmul rejects an x_stride below cols");
    expect(!pocket::qwen_fp16_matmul_rows_f16(buffer.get(), buffer.get(),
                                            buffer.get(), 1, 4, 8, 8, 2, 8),
           "matmul rejects a y_stride below rows");
}

}  // namespace

int main(int argc, char** argv) {
    int device = 0;
    // Run a single named case. Useful when one operator aborts and the question
    // is whether it fails on its own or only after another op has run.
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

    std::mt19937 rng(20260829u);
    // Named so a hang or an abort inside an aclnn call identifies itself; several
    // of these ops fail by dying rather than by returning false.
    const std::vector<std::pair<const char*, void (*)(std::mt19937&)>> suite = {
        {"matmul", test_matmul},
        {"swiglu_matmul", test_swiglu_matmul},
        {"rmsnorm", test_rmsnorm},
        {"residual_add_rmsnorm", test_residual_add_rmsnorm},
        {"gated_rmsnorm", test_gated_rmsnorm},
        {"elementwise", test_elementwise},
        {"concat_rows", test_concat_rows},
        {"copy_rows_strided", test_copy_rows_strided},
        {"split_packed_qkv", test_split_packed_qkv},
        {"split_rows_pair", test_split_rows_pair},
        {"split_q_gate", test_split_q_gate},
        {"embedding", test_embedding},
        {"copy_regions", test_copy_regions},
    };
    // A throw here means a device call reported failure. Report it as a failed
    // check rather than letting it abort, so the rest of the suite still runs.
    for (const auto& entry : suite) {
        // --only takes a comma-separated list, so a suspected interaction between
        // two cases can be reproduced without the rest of the suite.
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
        test_argument_rejection();
    }

    if (failures != 0) {
        std::cout << "FAIL " << failures << " of " << checks << " checks\n";
        return 1;
    }
    std::cout << "PASS (" << checks << " checks)\n";
    return 0;
}
