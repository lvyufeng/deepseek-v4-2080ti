// Causal depthwise conv: a single prefill over N tokens must produce exactly the
// same outputs as feeding those N tokens one at a time through the conv tail.
//
// This is the property that makes prefill->decode handoff correct. It caught a
// real bug: when seq_len < kernel-1 the tail rotation zero-filled the still-live
// history instead of shifting it, so decode diverged from token 2 onward.

#include "cuda_ops.hpp"
#include "qwen_ops.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

namespace {

int failures = 0;

uint16_t float_to_fp16_bits(float value) {
    uint32_t bits;
    __builtin_memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    int exponent = static_cast<int>((bits >> 23) & 0xff) - 127 + 15;
    const uint32_t mantissa = bits & 0x7fffffu;
    if (exponent <= 0) return static_cast<uint16_t>(sign);
    if (exponent >= 0x1f) return static_cast<uint16_t>(sign | 0x7c00u);
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) | (mantissa >> 13));
}

}  // namespace

int main() {
    if (!pocket::cuda_runtime_available()) {
        std::printf("[SKIP] test_qwen_conv_tail requires a CUDA device\n");
        return 0;
    }

    const int channels = 512;
    const int kernel = 4;   // Qwen3.8 linear_conv_kernel_dim
    const int seq_len = 6;

    std::mt19937 rng(13572468);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> x(static_cast<size_t>(seq_len) * channels);
    for (float& v : x) v = dist(rng);
    std::vector<uint16_t> w(static_cast<size_t>(channels) * kernel);
    for (uint16_t& v : w) v = float_to_fp16_bits(dist(rng) * 0.5f);

    float *d_x = nullptr, *d_y = nullptr, *d_tail = nullptr;
    uint16_t* d_w = nullptr;
    const size_t tail_elems = static_cast<size_t>(kernel - 1) * channels;
    if (cudaMalloc(&d_x, x.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_y, x.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_tail, tail_elems * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_w, w.size() * sizeof(uint16_t)) != cudaSuccess) {
        std::printf("[SKIP] device allocation failed\n");
        return 0;
    }
    cudaMemcpy(d_w, w.data(), w.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);

    // Pass 1: whole sequence at once, starting from a zero tail.
    cudaMemset(d_tail, 0, tail_elems * sizeof(float));
    cudaMemcpy(d_x, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice);
    if (!pocket::qwen_causal_depthwise_conv_silu_cuda(d_x, d_w, d_tail, d_y, seq_len, channels,
                                                    kernel, true)) {
        std::printf("[FAIL] prefill conv launch failed\n");
        return 1;
    }
    cudaDeviceSynchronize();
    std::vector<float> prefill(x.size());
    cudaMemcpy(prefill.data(), d_y, prefill.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // Pass 2: one token at a time, carrying the tail forward.
    cudaMemset(d_tail, 0, tail_elems * sizeof(float));
    std::vector<float> stepwise(x.size());
    for (int t = 0; t < seq_len; ++t) {
        cudaMemcpy(d_x, x.data() + static_cast<size_t>(t) * channels,
                   static_cast<size_t>(channels) * sizeof(float), cudaMemcpyHostToDevice);
        if (!pocket::qwen_causal_depthwise_conv_silu_cuda(d_x, d_w, d_tail, d_y, 1, channels,
                                                        kernel, true)) {
            std::printf("[FAIL] decode conv launch failed at t=%d\n", t);
            return 1;
        }
        cudaDeviceSynchronize();
        cudaMemcpy(stepwise.data() + static_cast<size_t>(t) * channels, d_y,
                   static_cast<size_t>(channels) * sizeof(float), cudaMemcpyDeviceToHost);
    }

    double worst = 0.0;
    int worst_t = -1;
    for (int t = 0; t < seq_len; ++t) {
        for (int c = 0; c < channels; ++c) {
            const size_t i = static_cast<size_t>(t) * channels + c;
            const double diff = std::fabs(static_cast<double>(prefill[i]) - stepwise[i]);
            if (diff > worst) {
                worst = diff;
                worst_t = t;
            }
        }
    }
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_tail);
    cudaFree(d_w);

    // Both passes run the identical kernel in the same order, so this must be
    // bit-exact, not merely close.
    if (worst != 0.0) {
        std::printf("[FAIL] prefill vs stepwise conv mismatch worst=%.3e at t=%d\n", worst, worst_t);
        ++failures;
    } else {
        std::printf("  conv tail continuity seq_len=%d kernel=%d channels=%d exact\n",
                    seq_len, kernel, channels);
    }
    if (failures != 0) return 1;
    std::printf("[PASS] test_qwen_conv_tail\n");
    return 0;
}
