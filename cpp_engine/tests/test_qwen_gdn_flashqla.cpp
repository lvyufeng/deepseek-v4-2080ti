// Numerical test for FlashQLA SM75 GDN kernel against baseline.
#include "qwen_ops.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

void check_cuda(cudaError_t status, const char* context) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", context, cudaGetErrorString(status));
        std::exit(1);
    }
}

uint16_t f16(float v) {
    __half x = __float2half(v);
    uint16_t b = 0;
    std::memcpy(&b, &x, 2);
    return b;
}

float f16_to_f32(uint16_t b) {
    __half x;
    std::memcpy(&x, &b, 2);
    return __half2float(x);
}

void run_test(int rows, int heads, int key_heads, int key_dim, int value_dim) {
    std::printf("Testing rows=%d heads=%d key_heads=%d key_dim=%d value_dim=%d\n",
                rows, heads, key_heads, key_dim, value_dim);

    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    const size_t state_size = static_cast<size_t>(heads) * key_dim * value_dim;
    const size_t qk_size = static_cast<size_t>(rows) * key_heads * key_dim;
    const size_t v_size = static_cast<size_t>(rows) * heads * value_dim;
    const size_t gate_size = static_cast<size_t>(rows) * heads;
    const size_t out_size = v_size;

    std::vector<float> h_state(state_size);
    for (float& x : h_state) x = dist(rng) * 0.1f;
    std::vector<uint16_t> h_q(qk_size);
    std::vector<uint16_t> h_k(qk_size);
    std::vector<uint16_t> h_v(v_size);
    std::vector<uint16_t> h_g(gate_size);
    std::vector<uint16_t> h_beta(gate_size);

    for (auto& x : h_q) x = f16(dist(rng) * 0.1f);
    for (auto& x : h_k) x = f16(dist(rng) * 0.1f);
    for (auto& x : h_v) x = f16(dist(rng) * 0.1f);
    for (auto& x : h_g) x = f16(dist(rng) * 0.5f);
    for (auto& x : h_beta) x = f16(0.5f + dist(rng) * 0.2f);

    float *d_state_baseline = nullptr, *d_state_flashqla = nullptr;
    float *d_state_split = nullptr;
    float *d_q_normalized = nullptr, *d_k_normalized = nullptr;
    uint16_t *d_q = nullptr, *d_k = nullptr, *d_v = nullptr;
    uint16_t *d_g = nullptr, *d_beta = nullptr;
    uint16_t *d_out_baseline = nullptr, *d_out_flashqla = nullptr,
             *d_out_split = nullptr;

    check_cuda(cudaMalloc(&d_state_baseline, state_size * sizeof(float)), "state_baseline");
    check_cuda(cudaMalloc(&d_state_flashqla, state_size * sizeof(float)), "state_flashqla");
    check_cuda(cudaMalloc(&d_state_split, state_size * sizeof(float)), "state_split");
    check_cuda(cudaMemset(d_state_split, 0, state_size * sizeof(float)), "zero state split");
    check_cuda(cudaMalloc(&d_q_normalized, qk_size * sizeof(float)), "normalized_q");
    check_cuda(cudaMalloc(&d_k_normalized, qk_size * sizeof(float)), "normalized_k");
    check_cuda(cudaMalloc(&d_q, qk_size * sizeof(uint16_t)), "q");
    check_cuda(cudaMalloc(&d_k, qk_size * sizeof(uint16_t)), "k");
    check_cuda(cudaMalloc(&d_v, v_size * sizeof(uint16_t)), "v");
    check_cuda(cudaMalloc(&d_g, gate_size * sizeof(uint16_t)), "g");
    check_cuda(cudaMalloc(&d_beta, gate_size * sizeof(uint16_t)), "beta");
    check_cuda(cudaMalloc(&d_out_baseline, out_size * sizeof(uint16_t)), "out_baseline");
    check_cuda(cudaMalloc(&d_out_flashqla, out_size * sizeof(uint16_t)), "out_flashqla");
    check_cuda(cudaMalloc(&d_out_split, out_size * sizeof(uint16_t)), "out_split");

    check_cuda(cudaMemcpy(d_state_baseline, h_state.data(), state_size * sizeof(float),
                          cudaMemcpyHostToDevice), "copy state baseline");
    check_cuda(cudaMemcpy(d_state_flashqla, h_state.data(), state_size * sizeof(float),
                          cudaMemcpyHostToDevice), "copy state flashqla");
    check_cuda(cudaMemcpy(d_q, h_q.data(), qk_size * sizeof(uint16_t),
                          cudaMemcpyHostToDevice), "copy q");
    check_cuda(cudaMemcpy(d_k, h_k.data(), qk_size * sizeof(uint16_t),
                          cudaMemcpyHostToDevice), "copy k");
    check_cuda(cudaMemcpy(d_v, h_v.data(), v_size * sizeof(uint16_t),
                          cudaMemcpyHostToDevice), "copy v");
    check_cuda(cudaMemcpy(d_g, h_g.data(), gate_size * sizeof(uint16_t),
                          cudaMemcpyHostToDevice), "copy g");
    check_cuda(cudaMemcpy(d_beta, h_beta.data(), gate_size * sizeof(uint16_t),
                          cudaMemcpyHostToDevice), "copy beta");

    // Baseline: existing serial kernel.
    const float q_scale = 1.0f / std::sqrt(static_cast<float>(key_dim));
    bool ok_baseline = pocket::qwen_gated_delta_sequence_f16(
        d_state_baseline, d_q, d_k, d_v, d_g, d_beta, d_out_baseline,
        rows, heads, key_heads, key_dim, value_dim, q_scale);
    if (!ok_baseline) {
        std::fprintf(stderr, "Baseline kernel launch failed\n");
        std::exit(1);
    }
    check_cuda(cudaDeviceSynchronize(), "sync baseline");

    // FlashQLA SM75 kernel, consuming the same normalized Q/K tensors as the
    // normalized baseline. This keeps the normalization reduction order equal.
    bool ok_normalize = pocket::qwen_normalize_gated_delta_qk_f16(
        d_q, d_k, d_q_normalized, d_k_normalized, rows, key_heads, key_dim);
    if (!ok_normalize) {
        std::fprintf(stderr, "Q/K normalization launch failed\n");
        std::exit(1);
    }
    check_cuda(cudaDeviceSynchronize(), "sync normalization");
    bool ok_flashqla = pocket::qwen_gated_delta_flashqla_sm75_f16_cuda(
        d_state_flashqla, d_q_normalized, d_k_normalized, d_v, d_g, d_beta,
        d_out_flashqla, rows, heads, key_heads, key_dim, value_dim,
        q_scale, nullptr);
    if (!ok_flashqla) {
        std::fprintf(stderr, "FlashQLA kernel launch failed\n");
        std::exit(1);
    }
    check_cuda(cudaDeviceSynchronize(), "sync flashqla");

    // State continuity: two consecutive halves resumed through the persistent
    // state must reproduce the single full-length run exactly. This is what a
    // chunked prefill followed by decode relies on, and it is where the earlier
    // transposed state layout silently produced wrong tokens.
    const int split = rows / 2;
    if (split > 0) {
        check_cuda(cudaMemcpy(d_state_split, h_state.data(),
                              state_size * sizeof(float),
                              cudaMemcpyHostToDevice), "copy state split");
        const size_t qk_rows = static_cast<size_t>(key_heads) * key_dim;
        const size_t v_rows = static_cast<size_t>(heads) * value_dim;
        const size_t gate_rows = static_cast<size_t>(heads);
        for (int begin = 0; begin < rows; begin += split) {
            const int extent = std::min(split, rows - begin);
            if (!pocket::qwen_gated_delta_flashqla_sm75_f16_cuda(
                    d_state_split,
                    d_q_normalized + static_cast<size_t>(begin) * qk_rows,
                    d_k_normalized + static_cast<size_t>(begin) * qk_rows,
                    d_v + static_cast<size_t>(begin) * v_rows,
                    d_g + static_cast<size_t>(begin) * gate_rows,
                    d_beta + static_cast<size_t>(begin) * gate_rows,
                    d_out_split + static_cast<size_t>(begin) * v_rows,
                    extent, heads, key_heads, key_dim, value_dim, q_scale)) {
                std::fprintf(stderr, "FlashQLA split launch failed\n");
                std::exit(1);
            }
        }
        check_cuda(cudaDeviceSynchronize(), "sync flashqla split");
    }

    std::vector<uint16_t> h_out_baseline(out_size);
    std::vector<uint16_t> h_out_flashqla(out_size);
    std::vector<uint16_t> h_out_split(out_size);
    std::vector<float> h_state_split_final(state_size);
    std::vector<float> h_state_baseline_final(state_size);
    std::vector<float> h_state_flashqla_final(state_size);

    check_cuda(cudaMemcpy(h_out_baseline.data(), d_out_baseline,
                          out_size * sizeof(uint16_t), cudaMemcpyDeviceToHost), "read out baseline");
    check_cuda(cudaMemcpy(h_out_flashqla.data(), d_out_flashqla,
                          out_size * sizeof(uint16_t), cudaMemcpyDeviceToHost), "read out flashqla");
    check_cuda(cudaMemcpy(h_state_baseline_final.data(), d_state_baseline,
                          state_size * sizeof(float), cudaMemcpyDeviceToHost), "read state baseline");
    check_cuda(cudaMemcpy(h_state_flashqla_final.data(), d_state_flashqla,
                          state_size * sizeof(float), cudaMemcpyDeviceToHost), "read state flashqla");
    check_cuda(cudaMemcpy(h_out_split.data(), d_out_split,
                          out_size * sizeof(uint16_t), cudaMemcpyDeviceToHost),
               "read out split");
    check_cuda(cudaMemcpy(h_state_split_final.data(), d_state_split,
                          state_size * sizeof(float), cudaMemcpyDeviceToHost),
               "read state split");

    cudaFree(d_state_baseline);
    cudaFree(d_state_flashqla);
    cudaFree(d_state_split);
    cudaFree(d_out_split);
    cudaFree(d_q_normalized);
    cudaFree(d_k_normalized);
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_g);
    cudaFree(d_beta);
    cudaFree(d_out_baseline);
    cudaFree(d_out_flashqla);

    float max_out_diff = 0.0f;
    float max_state_diff = 0.0f;

    for (size_t i = 0; i < out_size; ++i) {
        const float a = f16_to_f32(h_out_baseline[i]);
        const float b = f16_to_f32(h_out_flashqla[i]);
        if (!std::isfinite(a) || !std::isfinite(b)) {
            std::fprintf(stderr, "FAIL: non-finite output at index %zu\n", i);
            std::exit(1);
        }
        const float diff = std::fabs(a - b);
        if (diff > max_out_diff) max_out_diff = diff;
    }

    for (float value : h_state_baseline_final) {
        if (!std::isfinite(value)) {
            std::fprintf(stderr, "FAIL: non-finite baseline state\n");
            std::exit(1);
        }
    }
    for (float value : h_state_flashqla_final) {
        if (!std::isfinite(value)) {
            std::fprintf(stderr, "FAIL: non-finite FlashQLA state\n");
            std::exit(1);
        }
    }

    for (size_t i = 0; i < state_size; ++i) {
        const float diff = std::fabs(h_state_baseline_final[i] - h_state_flashqla_final[i]);
        if (diff > max_state_diff) max_state_diff = diff;
    }

    // The split run recurs through the same state in the same order, so it must
    // be bit-identical to the single call, not merely close.
    if (split > 0) {
        for (size_t i = 0; i < out_size; ++i) {
            if (h_out_split[i] != h_out_flashqla[i]) {
                std::fprintf(stderr,
                             "FAIL: split-run output differs at index %zu\n", i);
                std::exit(1);
            }
        }
        for (size_t i = 0; i < state_size; ++i) {
            if (h_state_split_final[i] != h_state_flashqla_final[i]) {
                std::fprintf(stderr,
                             "FAIL: split-run state differs at index %zu\n", i);
                std::exit(1);
            }
        }
    }

    std::printf("  max_out_diff=%.6e max_state_diff=%.6e split_continuity=exact\n",
                max_out_diff, max_state_diff);

    if (!std::isfinite(max_out_diff) || !std::isfinite(max_state_diff)) {
        std::fprintf(stderr, "FAIL: non-finite difference\n");
        std::exit(1);
    }
    // FlashQLA reduces each K/Q dot with a 16-lane shuffle tree while the
    // baseline accumulates it sequentially, so the difference is reduction
    // order only, not a semantic change. That leaves the output within one or
    // two FP16 ULP and the FP32 state within ~1e-5 even at 512 rows; anything
    // larger means the sharding or the state layout is wrong again.
    const float out_tol = rows <= 64 ? 1e-5f : 2e-3f;
    const float state_tol = rows <= 64 ? 1e-6f : 1e-4f;
    if (max_out_diff > out_tol || max_state_diff > state_tol) {
        std::fprintf(stderr, "FAIL: exceeds tolerance (out=%.6e state=%.6e)\n",
                     out_tol, state_tol);
        std::exit(1);
    }
    std::printf("  PASS\n");
}

}  // namespace

int main() {
    check_cuda(cudaSetDevice(0), "set device");
    // TP4 Qwen3.8 shape: heads=12, key_heads=4, key_dim=value_dim=128.
    // Include a long recurrence to expose state drift; full-model token parity
    // remains the production gate.
    run_test(8, 12, 4, 128, 128);
    run_test(32, 12, 4, 128, 128);
    run_test(64, 12, 4, 128, 128);
    run_test(512, 12, 4, 128, 128);
    std::printf("All tests PASS\n");
    return 0;
}
