// Qwen FP8 E4M3 + BF16 block-scale online-unpack numerics.
//
// Builds random FP8 codes and BF16 128x128 block scales on the host, computes
// the reference product in double precision from the same decoded values, and
// compares against the CUDA kernels. Shapes deliberately cross block
// boundaries and include a TP-style shard offset so a mis-indexed scale block
// shows up as an error rather than as noise.

#include "cuda_ops.hpp"
#include "qwen_ops.hpp"
#include "qwen_weights.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

int failures = 0;

void fail(const std::string& what) {
    std::cout << "[FAIL] " << what << "\n";
    ++failures;
}

float fp8_e4m3_to_float(uint8_t code) {
    const int sign = (code >> 7) & 1;
    const int exponent = (code >> 3) & 0xf;
    const int mantissa = code & 0x7;
    float value;
    if (exponent == 0) {
        value = std::ldexp(static_cast<float>(mantissa) * (1.0f / 8.0f), -6);
    } else {
        value = std::ldexp(1.0f + static_cast<float>(mantissa) * (1.0f / 8.0f), exponent - 7);
    }
    return sign ? -value : value;
}

uint16_t float_to_bf16_bits(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    // Round-to-nearest-even on the truncated mantissa so the host reference and
    // the kernel see bit-identical scales.
    const uint32_t rounding_bias = 0x7fff + ((bits >> 16) & 1);
    return static_cast<uint16_t>((bits + rounding_bias) >> 16);
}

float bf16_bits_to_float(uint16_t bits) {
    const uint32_t widened = static_cast<uint32_t>(bits) << 16;
    float value;
    std::memcpy(&value, &widened, sizeof(value));
    return value;
}

// IEEE FP16 -> FP32, matching the device-side decode of converted Qwen scales.
float fp16_bits_to_float(uint16_t bits) {
    const uint32_t sign = static_cast<uint32_t>(bits & 0x8000u) << 16;
    const uint32_t exponent = (bits >> 10) & 0x1fu;
    const uint32_t mantissa = bits & 0x03ffu;
    uint32_t widened;
    if (exponent == 0) {
        if (mantissa == 0) {
            widened = sign;
        } else {
            uint32_t normalized = mantissa;
            int exp = -14;
            while ((normalized & 0x400u) == 0) {
                normalized <<= 1;
                --exp;
            }
            widened = sign | (static_cast<uint32_t>(exp + 127) << 23) | ((normalized & 0x3ffu) << 13);
        }
    } else if (exponent == 0x1fu) {
        widened = sign | 0x7f800000u | (mantissa << 13);
    } else {
        widened = sign | ((exponent - 15 + 127) << 23) | (mantissa << 13);
    }
    float value;
    std::memcpy(&value, &widened, sizeof(value));
    return value;
}

struct DeviceBuffers {
    float* x = nullptr;
    uint8_t* weight = nullptr;
    uint16_t* scale = nullptr;
    float* y = nullptr;

    ~DeviceBuffers() {
        cudaFree(x);
        cudaFree(weight);
        cudaFree(scale);
        cudaFree(y);
    }
};

struct SwiGLUDeviceBuffers {
    float* x = nullptr;
    uint8_t* gate_weight = nullptr;
    uint8_t* up_weight = nullptr;
    uint16_t* gate_scale = nullptr;
    uint16_t* up_scale = nullptr;
    float* gate = nullptr;
    float* up = nullptr;
    float* reference = nullptr;
    float* fused = nullptr;

    ~SwiGLUDeviceBuffers() {
        cudaFree(x);
        cudaFree(gate_weight);
        cudaFree(up_weight);
        cudaFree(gate_scale);
        cudaFree(up_scale);
        cudaFree(gate);
        cudaFree(up);
        cudaFree(reference);
        cudaFree(fused);
    }
};

bool run_case(int batch, int rows, int cols, uint64_t seed, const std::string& label) {
    constexpr int kBlock = 128;
    const int scale_rows = (rows + kBlock - 1) / kBlock;
    const int scale_cols = (cols + kBlock - 1) / kBlock;

    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> code_dist(0, 255);
    std::uniform_real_distribution<float> x_dist(-1.5f, 1.5f);
    std::uniform_real_distribution<float> scale_dist(0.01f, 0.4f);

    std::vector<float> h_x(static_cast<size_t>(batch) * cols);
    for (float& value : h_x) value = x_dist(rng);

    std::vector<uint8_t> h_weight(static_cast<size_t>(rows) * cols);
    for (uint8_t& code : h_weight) {
        int value = code_dist(rng);
        // 0x7f/0xff are E4M3 NaN in the OCP encoding; the checkpoint never
        // stores them and including them would make the comparison meaningless.
        if ((value & 0x7f) == 0x7f) value = 0x70;
        code = static_cast<uint8_t>(value);
    }

    std::vector<uint16_t> h_scale_bf16(static_cast<size_t>(scale_rows) * scale_cols);
    std::vector<uint16_t> h_scale_fp16(h_scale_bf16.size());
    for (uint16_t& bits : h_scale_bf16) bits = float_to_bf16_bits(scale_dist(rng));
    for (size_t i = 0; i < h_scale_bf16.size(); ++i) {
        h_scale_fp16[i] = pocket::qwen_bf16_to_fp16_bits(h_scale_bf16[i]);
    }

    std::vector<double> expected(static_cast<size_t>(batch) * rows, 0.0);
    for (int sample = 0; sample < batch; ++sample) {
        for (int row = 0; row < rows; ++row) {
            double sum = 0.0;
            for (int col = 0; col < cols; ++col) {
                const float w = fp8_e4m3_to_float(h_weight[static_cast<size_t>(row) * cols + col]);
                const float s = fp16_bits_to_float(
                    h_scale_fp16[static_cast<size_t>(row / kBlock) * scale_cols + col / kBlock]);
                sum += static_cast<double>(h_x[static_cast<size_t>(sample) * cols + col]) *
                       static_cast<double>(w) * static_cast<double>(s);
            }
            expected[static_cast<size_t>(sample) * rows + row] = sum;
        }
    }

    DeviceBuffers dev;
    if (cudaMalloc(&dev.x, h_x.size() * sizeof(float)) != cudaSuccess) return false;
    if (cudaMalloc(&dev.weight, h_weight.size()) != cudaSuccess) return false;
    if (cudaMalloc(&dev.scale, h_scale_fp16.size() * sizeof(uint16_t)) != cudaSuccess) return false;
    if (cudaMalloc(&dev.y, static_cast<size_t>(batch) * rows * sizeof(float)) != cudaSuccess) return false;
    cudaMemcpy(dev.x, h_x.data(), h_x.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev.weight, h_weight.data(), h_weight.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dev.scale, h_scale_fp16.data(), h_scale_fp16.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);

    std::vector<float> actual(static_cast<size_t>(batch) * rows, 0.0f);
    if (batch == 1) {
        if (!pocket::qwen_fp8_e4m3_fp16scale_matvec_cuda(dev.x, dev.weight, dev.scale, dev.y,
                                                       rows, cols, cols, scale_cols)) {
            fail(label + ": FP16-scale matvec kernel launch failed");
            return true;
        }
    } else if (batch >= 32) {
        if (!pocket::qwen_fp8_e4m3_fp16scale_matmul_simt_cuda(dev.x, dev.weight, dev.scale, dev.y,
                                                            batch, rows, cols, cols, rows,
                                                            cols, scale_cols)) {
            fail(label + ": FP16-scale SIMT matmul kernel launch failed");
            return true;
        }
    } else {
        if (!pocket::qwen_fp8_e4m3_fp16scale_matmul_rows_cuda(dev.x, dev.weight, dev.scale, dev.y,
                                                            batch, rows, cols, cols, rows,
                                                            cols, scale_cols)) {
            fail(label + ": FP16-scale matmul kernel launch failed");
            return true;
        }
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        fail(label + ": kernel sync failed");
        return true;
    }
    cudaMemcpy(actual.data(), dev.y, actual.size() * sizeof(float), cudaMemcpyDeviceToHost);

    double max_abs = 0.0;
    double max_rel = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const double want = expected[i];
        const double got = static_cast<double>(actual[i]);
        const double abs_err = std::fabs(got - want);
        max_abs = std::max(max_abs, abs_err);
        const double denom = std::max(1.0, std::fabs(want));
        max_rel = std::max(max_rel, abs_err / denom);
    }
    // fp32 accumulation over up to 640 terms; 2e-4 relative is well inside
    // fp32 rounding while still catching a wrong scale block or stride.
    const bool ok = max_rel < 2.0e-4;
    if (!ok) {
        fail(label + ": max_rel=" + std::to_string(max_rel) + " max_abs=" + std::to_string(max_abs));
    } else {
        std::cout << "  " << label << " batch=" << batch << " rows=" << rows << " cols=" << cols
                  << " max_abs=" << max_abs << " max_rel=" << max_rel << "\n";
    }
    return true;
}

bool run_swiglu_case() {
    constexpr int rows = 4352;
    constexpr int cols = 5120;
    constexpr int block = 128;
    const int scale_cols = cols / block;
    const int scale_rows = rows / block;
    std::mt19937_64 rng(0x1234abcd);
    std::uniform_int_distribution<int> code_dist(0, 255);
    std::uniform_real_distribution<float> x_dist(-1.0f, 1.0f);
    SwiGLUDeviceBuffers dev;
    std::vector<float> x(cols);
    std::vector<uint8_t> gate_weight(static_cast<size_t>(rows) * cols);
    std::vector<uint8_t> up_weight(gate_weight.size());
    std::vector<uint16_t> gate_scale(static_cast<size_t>(scale_rows) * scale_cols, 0x3800);
    std::vector<uint16_t> up_scale(gate_scale.size(), 0x3a00);
    for (float& value : x) value = x_dist(rng);
    for (size_t i = 0; i < gate_weight.size(); ++i) {
        gate_weight[i] = static_cast<uint8_t>(code_dist(rng));
        up_weight[i] = static_cast<uint8_t>(code_dist(rng));
    }
    if (cudaMalloc(&dev.x, x.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dev.gate_weight, gate_weight.size()) != cudaSuccess ||
        cudaMalloc(&dev.up_weight, up_weight.size()) != cudaSuccess ||
        cudaMalloc(&dev.gate_scale, gate_scale.size() * sizeof(uint16_t)) != cudaSuccess ||
        cudaMalloc(&dev.up_scale, up_scale.size() * sizeof(uint16_t)) != cudaSuccess ||
        cudaMalloc(&dev.gate, rows * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dev.up, rows * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dev.reference, rows * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dev.fused, rows * sizeof(float)) != cudaSuccess) {
        return false;
    }
    cudaMemcpy(dev.x, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev.gate_weight, gate_weight.data(), gate_weight.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dev.up_weight, up_weight.data(), up_weight.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dev.gate_scale, gate_scale.data(), gate_scale.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(dev.up_scale, up_scale.data(), up_scale.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
    if (!pocket::qwen_fp8_e4m3_fp16scale_matvec_cuda(
            dev.x, dev.gate_weight, dev.gate_scale, dev.gate, rows, cols, cols, scale_cols) ||
        !pocket::qwen_fp8_e4m3_fp16scale_matvec_cuda(
            dev.x, dev.up_weight, dev.up_scale, dev.up, rows, cols, cols, scale_cols) ||
        !pocket::qwen_silu_mul_rows_cuda(dev.gate, dev.up, dev.reference, 1, rows) ||
        !pocket::qwen_fp8_e4m3_fp16scale_swiglu_matvec_cuda(
            dev.x, dev.gate_weight, dev.gate_scale, dev.up_weight, dev.up_scale, dev.fused,
            rows, cols, cols, scale_cols)) {
        fail("fused SwiGLU launch failed");
        return true;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        fail("fused SwiGLU sync failed");
        return true;
    }
    std::vector<float> reference(rows), fused(rows);
    cudaMemcpy(reference.data(), dev.reference, reference.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(fused.data(), dev.fused, fused.size() * sizeof(float), cudaMemcpyDeviceToHost);
    double max_abs = 0.0;
    double max_rel = 0.0;
    for (int i = 0; i < rows; ++i) {
        const double error = std::fabs(static_cast<double>(fused[i]) - reference[i]);
        max_abs = std::max(max_abs, error);
        max_rel = std::max(max_rel, error / std::max(1.0, std::fabs(static_cast<double>(reference[i]))));
    }
    if (max_rel > 2.0e-4) {
        fail("fused SwiGLU max_rel=" + std::to_string(max_rel) + " max_abs=" + std::to_string(max_abs));
    } else {
        std::cout << "  fused SwiGLU rows=" << rows << " cols=" << cols
                  << " max_abs=" << max_abs << " max_rel=" << max_rel << "\n";
    }
    return true;
}

bool run_norm_cases() {
    const int rows = 3;
    const int cols = 256;
    std::mt19937_64 rng(20260815);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    std::vector<float> h_x(static_cast<size_t>(rows) * cols);
    std::vector<float> h_gate(h_x.size());
    std::vector<float> h_w(cols);
    for (float& v : h_x) v = dist(rng);
    for (float& v : h_gate) v = dist(rng);
    for (float& v : h_w) v = dist(rng) * 0.1f;

    float *d_x = nullptr, *d_gate = nullptr, *d_w = nullptr, *d_y = nullptr;
    if (cudaMalloc(&d_x, h_x.size() * sizeof(float)) != cudaSuccess) return false;
    if (cudaMalloc(&d_gate, h_gate.size() * sizeof(float)) != cudaSuccess) return false;
    if (cudaMalloc(&d_w, h_w.size() * sizeof(float)) != cudaSuccess) return false;
    if (cudaMalloc(&d_y, h_x.size() * sizeof(float)) != cudaSuccess) return false;
    cudaMemcpy(d_x, h_x.data(), h_x.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_gate, h_gate.data(), h_gate.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, h_w.data(), h_w.size() * sizeof(float), cudaMemcpyHostToDevice);

    const float eps = 1.0e-6f;
    std::vector<float> got(h_x.size());

    // Qwen rms_norm multiplies by (1 + weight), unlike the DeepSeek-V4 convention.
    if (!pocket::qwen_rmsnorm_f32_cuda(d_x, d_w, d_y, rows, cols, eps)) {
        fail("qwen_rmsnorm launch failed");
    } else {
        cudaDeviceSynchronize();
        cudaMemcpy(got.data(), d_y, got.size() * sizeof(float), cudaMemcpyDeviceToHost);
        double max_err = 0.0;
        for (int r = 0; r < rows; ++r) {
            double sq = 0.0;
            for (int c = 0; c < cols; ++c) {
                const double v = h_x[static_cast<size_t>(r) * cols + c];
                sq += v * v;
            }
            const double inv = 1.0 / std::sqrt(sq / cols + eps);
            for (int c = 0; c < cols; ++c) {
                const double want = h_x[static_cast<size_t>(r) * cols + c] * inv * (1.0 + h_w[c]);
                max_err = std::max(max_err, std::fabs(want - got[static_cast<size_t>(r) * cols + c]));
            }
        }
        if (max_err > 1.0e-4) fail("qwen_rmsnorm max_err=" + std::to_string(max_err));
        else std::cout << "  rms_norm(1+w) max_err=" << max_err << "\n";
    }

    // Gated variant: weight * normalized * silu(gate), with no (1 + weight).
    if (!pocket::qwen_gated_rmsnorm_f32_cuda(d_x, d_w, d_gate, d_y, rows, cols, eps)) {
        fail("qwen_gated_rmsnorm launch failed");
    } else {
        cudaDeviceSynchronize();
        cudaMemcpy(got.data(), d_y, got.size() * sizeof(float), cudaMemcpyDeviceToHost);
        double max_err = 0.0;
        for (int r = 0; r < rows; ++r) {
            double sq = 0.0;
            for (int c = 0; c < cols; ++c) {
                const double v = h_x[static_cast<size_t>(r) * cols + c];
                sq += v * v;
            }
            const double inv = 1.0 / std::sqrt(sq / cols + eps);
            for (int c = 0; c < cols; ++c) {
                const double g = h_gate[static_cast<size_t>(r) * cols + c];
                const double silu = g / (1.0 + std::exp(-g));
                const double want = h_w[c] * h_x[static_cast<size_t>(r) * cols + c] * inv * silu;
                max_err = std::max(max_err, std::fabs(want - got[static_cast<size_t>(r) * cols + c]));
            }
        }
        if (max_err > 1.0e-4) fail("qwen_gated_rmsnorm max_err=" + std::to_string(max_err));
        else std::cout << "  gated_rms_norm(w*norm*silu(z)) max_err=" << max_err << "\n";
    }

    if (!pocket::qwen_l2_norm_f32_cuda(d_x, d_y, rows, cols)) {
        fail("qwen_l2_norm launch failed");
    } else {
        cudaDeviceSynchronize();
        cudaMemcpy(got.data(), d_y, got.size() * sizeof(float), cudaMemcpyDeviceToHost);
        double max_err = 0.0;
        for (int r = 0; r < rows; ++r) {
            double sq = 0.0;
            for (int c = 0; c < cols; ++c) {
                const double v = h_x[static_cast<size_t>(r) * cols + c];
                sq += v * v;
            }
            const double inv = 1.0 / std::sqrt(sq + 1.0e-6);
            for (int c = 0; c < cols; ++c) {
                const double want = h_x[static_cast<size_t>(r) * cols + c] * inv;
                max_err = std::max(max_err, std::fabs(want - got[static_cast<size_t>(r) * cols + c]));
            }
        }
        if (max_err > 1.0e-5) fail("qwen_l2_norm max_err=" + std::to_string(max_err));
        else std::cout << "  l2_norm max_err=" << max_err << "\n";
    }

    cudaFree(d_x);
    cudaFree(d_gate);
    cudaFree(d_w);
    cudaFree(d_y);
    return true;
}

}  // namespace

int main() {
    if (!pocket::cuda_runtime_available()) {
        std::cout << "[SKIP] test_qwen_fp8_online requires a CUDA device\n";
        return 0;
    }

    struct Case {
        int batch;
        int rows;
        int cols;
        const char* label;
    };
    // 640 crosses five scale blocks along k; 384 rows crosses three along the
    // output dim; the 130/300 case is deliberately not block-aligned.
    const Case cases[] = {
        {1, 384, 640, "decode matvec block-aligned"},
        {1, 130, 300, "decode matvec unaligned"},
        {4, 256, 512, "small-batch verify"},
        {9, 128, 256, "prefill multi-row"},
        {64, 130, 300, "prefill SIMT tiled"},
        {130, 130, 300, "prefill SIMT wide unaligned"},
        {256, 256, 512, "prefill SIMT wide aligned"},
    };
    for (const Case& c : cases) {
        if (!run_case(c.batch, c.rows, c.cols, 0x5eed1234u + static_cast<uint64_t>(c.rows), c.label)) {
            std::cout << "[SKIP] device allocation failed for " << c.label << "\n";
            return 0;
        }
    }
    if (!run_swiglu_case()) {
        std::cout << "[SKIP] device allocation failed for fused SwiGLU\n";
        return 0;
    }
    if (!run_norm_cases()) {
        std::cout << "[SKIP] device allocation failed for norm cases\n";
        return 0;
    }

    if (failures != 0) {
        std::cout << "test_qwen_fp8_online failures=" << failures << "\n";
        return 1;
    }
    std::cout << "[PASS] test_qwen_fp8_online\n";
    return 0;
}
