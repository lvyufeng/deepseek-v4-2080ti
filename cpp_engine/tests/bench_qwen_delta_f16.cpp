#include "qwen_ops.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(status));
        std::exit(1);
    }
}

uint16_t to_half(float value) {
    const __half h = __float2half(value);
    uint16_t bits;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
}

template <typename Launch>
double time_launch(Launch launch, int iters) {
    if (!launch()) std::exit(1);
    check(cudaDeviceSynchronize(), "warmup");
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    check(cudaEventCreate(&start), "create start");
    check(cudaEventCreate(&stop), "create stop");
    check(cudaEventRecord(start), "record start");
    for (int i = 0; i < iters; ++i) {
        if (!launch()) std::exit(1);
    }
    check(cudaEventRecord(stop), "record stop");
    check(cudaEventSynchronize(stop), "sync stop");
    float elapsed = 0.0f;
    check(cudaEventElapsedTime(&elapsed, start, stop), "elapsed");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return elapsed / iters;
}

}  // namespace

int main(int argc, char** argv) {
    int rows = 8;
    int iters = 1000;
    if (argc > 1) rows = std::atoi(argv[1]);
    if (argc > 2) iters = std::atoi(argv[2]);
    const int heads = argc > 3 ? std::atoi(argv[3]) : 12;
    const int key_heads = argc > 4 ? std::atoi(argv[4]) : 4;
    constexpr int kDim = 128;
    if (heads <= 0 || key_heads <= 0 || heads % key_heads != 0) {
        std::fprintf(stderr, "invalid heads=%d key_heads=%d\n", heads, key_heads);
        return 1;
    }
    const size_t state_elements = static_cast<size_t>(heads) * kDim * kDim;
    const size_t key_elements = static_cast<size_t>(rows) * key_heads * kDim;
    const size_t value_elements = static_cast<size_t>(rows) * heads * kDim;
    const size_t gate_elements = static_cast<size_t>(rows) * heads;

    float* state = nullptr;
    uint16_t* q = nullptr;
    uint16_t* k = nullptr;
    uint16_t* v = nullptr;
    uint16_t* g = nullptr;
    uint16_t* beta = nullptr;
    uint16_t* output = nullptr;
    float* q_normalized = nullptr;
    float* k_normalized = nullptr;
    check(cudaMalloc(&state, state_elements * sizeof(float)), "malloc state");
    check(cudaMalloc(&q, key_elements * sizeof(uint16_t)), "malloc q");
    check(cudaMalloc(&k, key_elements * sizeof(uint16_t)), "malloc k");
    check(cudaMalloc(&v, value_elements * sizeof(uint16_t)), "malloc v");
    check(cudaMalloc(&g, gate_elements * sizeof(uint16_t)), "malloc g");
    check(cudaMalloc(&beta, gate_elements * sizeof(uint16_t)), "malloc beta");
    check(cudaMalloc(&output, value_elements * sizeof(uint16_t)), "malloc output");
    check(cudaMalloc(&q_normalized, key_elements * sizeof(float)),
          "malloc normalized q");
    check(cudaMalloc(&k_normalized, key_elements * sizeof(float)),
          "malloc normalized k");
    check(cudaMemset(state, 0, state_elements * sizeof(float)), "zero state");

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-0.2f, 0.2f);
    auto fill = [&](uint16_t* device, size_t elements, float bias) {
        std::vector<uint16_t> host(elements);
        for (uint16_t& x : host) x = to_half(dist(rng) + bias);
        check(cudaMemcpy(device, host.data(), elements * sizeof(uint16_t),
                         cudaMemcpyHostToDevice), "copy input");
    };
    fill(q, key_elements, 0.0f);
    fill(k, key_elements, 0.0f);
    fill(v, value_elements, 0.0f);
    fill(g, gate_elements, -0.2f);
    fill(beta, gate_elements, 0.5f);

    auto sequence = [&]() {
        return pocket::qwen_gated_delta_sequence_f16(
            state, q, k, v, g, beta, output, rows, heads, key_heads,
            kDim, kDim, 1.0f / 11.3137085f);
    };
    auto normalized = [&]() {
        return pocket::qwen_normalize_gated_delta_qk_f16(
                   q, k, q_normalized, k_normalized, rows, key_heads, kDim) &&
               pocket::qwen_gated_delta_sequence_normalized_f16(
                   state, q_normalized, k_normalized, v, g, beta, output,
                   rows, heads, key_heads, kDim, kDim,
                   1.0f / 11.3137085f);
    };
    auto shared_state_variant = [&]() {
        return pocket::qwen_normalize_gated_delta_qk_f16(
                   q, k, q_normalized, k_normalized, rows, key_heads, kDim) &&
               pocket::qwen_gated_delta_sequence_normalized_shared_f16(
                   state, q_normalized, k_normalized, v, g, beta, output,
                   rows, heads, key_heads, kDim, kDim,
                   1.0f / 11.3137085f);
    };
    // FlashQLA SM75 subgroup-sharded kernel: WIDTH=16 lanes hold COLS=4 state
    // columns each, so the [D, D] state is spread over 8 warps instead of one
    // thread per value dimension. Same serial recurrence, different sharding.
    auto flashqla = [&]() {
        return pocket::qwen_normalize_gated_delta_qk_f16(
                   q, k, q_normalized, k_normalized, rows, key_heads, kDim) &&
               pocket::qwen_gated_delta_flashqla_sm75_f16_cuda(
                   state, q_normalized, k_normalized, v, g, beta, output,
                   rows, heads, key_heads, kDim, kDim,
                   1.0f / 11.3137085f);
    };
    auto steps = [&]() {
        for (int row = 0; row < rows; ++row) {
            if (!pocket::qwen_gated_delta_step_f16(
                    state, q + static_cast<size_t>(row) * key_heads * kDim,
                    k + static_cast<size_t>(row) * key_heads * kDim,
                    v + static_cast<size_t>(row) * heads * kDim,
                    g + static_cast<size_t>(row) * heads,
                    beta + static_cast<size_t>(row) * heads,
                    output + static_cast<size_t>(row) * heads * kDim,
                    heads, key_heads, kDim, kDim,
                    1.0f / 11.3137085f)) return false;
        }
        return true;
    };
    const double seq_ms = time_launch(sequence, iters);
    check(cudaMemset(state, 0, state_elements * sizeof(float)), "reset state");
    const double normalized_ms = time_launch(normalized, iters);
    check(cudaMemset(state, 0, state_elements * sizeof(float)), "reset state");
    const double shared_ms = time_launch(shared_state_variant, iters);
    check(cudaMemset(state, 0, state_elements * sizeof(float)), "reset state");
    const double flashqla_ms = time_launch(flashqla, iters);
    check(cudaMemset(state, 0, state_elements * sizeof(float)), "reset state");
    const double step_ms = time_launch(steps, iters);
    std::printf("qwen_gated_delta_f16 rows=%d heads=%d key_heads=%d dim=%d "
                "sequence=%.6f ms normalized=%.6f ms shared=%.6f ms "
                "flashqla=%.6f ms steps=%.6f ms normalized_speedup=%.3f "
                "shared_speedup=%.3f flashqla_speedup=%.3f "
                "sequence_speedup=%.3f\n",
                rows, heads, key_heads, kDim, seq_ms, normalized_ms, shared_ms,
                flashqla_ms, step_ms, seq_ms / normalized_ms,
                normalized_ms / shared_ms, normalized_ms / flashqla_ms,
                step_ms / seq_ms);
    cudaFree(state);
    cudaFree(q);
    cudaFree(k);
    cudaFree(v);
    cudaFree(g);
    cudaFree(beta);
    cudaFree(output);
    cudaFree(q_normalized);
    cudaFree(k_normalized);
    return 0;
}
