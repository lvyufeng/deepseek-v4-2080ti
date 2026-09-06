#include "qwen_ops.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

void check(cudaError_t error, const char* what) {
    if (error != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(error));
        std::exit(1);
    }
}

uint16_t to_half(float value) {
    uint16_t bits = 0;
    const __half h = __float2half(value);
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
}

template <typename Launch>
double time_ms(Launch launch, int iters) {
    for (int i = 0; i < 10; ++i) {
        if (!launch()) {
            std::fprintf(stderr, "launch failed\n");
            std::exit(1);
        }
    }
    check(cudaDeviceSynchronize(), "warmup sync");
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    check(cudaEventCreate(&start), "create start");
    check(cudaEventCreate(&stop), "create stop");
    check(cudaEventRecord(start), "record start");
    for (int i = 0; i < iters; ++i) (void)launch();
    check(cudaEventRecord(stop), "record stop");
    check(cudaEventSynchronize(stop), "sync stop");
    float elapsed = 0.0f;
    check(cudaEventElapsedTime(&elapsed, start, stop), "event elapsed");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return static_cast<double>(elapsed) / iters;
}

}  // namespace

int main(int argc, char** argv) {
    int context = 8192;
    int rows = 8;
    int splits = 64;
    int iters = 200;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--context" && i + 1 < argc) context = std::atoi(argv[++i]);
        else if (arg == "--rows" && i + 1 < argc) rows = std::atoi(argv[++i]);
        else if (arg == "--splits" && i + 1 < argc) splits = std::atoi(argv[++i]);
        else if (arg == "--iters" && i + 1 < argc) iters = std::atoi(argv[++i]);
    }
    constexpr int q_heads = 6;
    constexpr int kv_heads = 1;
    constexpr int head_dim = 256;
    const int position_offset = context - rows;

    const size_t q_elements = static_cast<size_t>(rows) * q_heads * head_dim;
    const size_t kv_elements = static_cast<size_t>(context) * kv_heads * head_dim;
    const size_t output_elements = static_cast<size_t>(rows) * q_heads * head_dim;
    const size_t exact_scores = static_cast<size_t>(rows) * q_heads * context;
    const size_t partials = static_cast<size_t>(rows) * q_heads * splits *
        static_cast<size_t>(head_dim + 2);

    uint16_t* q = nullptr;
    uint16_t* k = nullptr;
    uint16_t* v = nullptr;
    uint16_t* split_output = nullptr;
    uint16_t* exact_output = nullptr;
    uint16_t* cublas_output = nullptr;
    float* split_scratch = nullptr;
    float* exact_scratch = nullptr;
    check(cudaMalloc(&q, q_elements * sizeof(uint16_t)), "malloc q");
    check(cudaMalloc(&k, kv_elements * sizeof(uint16_t)), "malloc k");
    check(cudaMalloc(&v, kv_elements * sizeof(uint16_t)), "malloc v");
    check(cudaMalloc(&split_output, output_elements * sizeof(uint16_t)), "malloc split output");
    check(cudaMalloc(&exact_output, output_elements * sizeof(uint16_t)), "malloc exact output");
    check(cudaMalloc(&cublas_output, output_elements * sizeof(uint16_t)), "malloc cublas output");
    check(cudaMalloc(&split_scratch, partials * sizeof(float)), "malloc partials");
    check(cudaMalloc(&exact_scratch, exact_scores * sizeof(float)), "malloc scores");

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    std::vector<uint16_t> host_q(q_elements);
    std::vector<uint16_t> host_k(kv_elements);
    std::vector<uint16_t> host_v(kv_elements);
    for (uint16_t& x : host_q) x = to_half(dist(rng));
    for (uint16_t& x : host_k) x = to_half(dist(rng));
    for (uint16_t& x : host_v) x = to_half(dist(rng));
    check(cudaMemcpy(q, host_q.data(), host_q.size() * sizeof(uint16_t), cudaMemcpyHostToDevice), "copy q");
    check(cudaMemcpy(k, host_k.data(), host_k.size() * sizeof(uint16_t), cudaMemcpyHostToDevice), "copy k");
    check(cudaMemcpy(v, host_v.data(), host_v.size() * sizeof(uint16_t), cudaMemcpyHostToDevice), "copy v");

    auto split = [&]() {
        return pocket::qwen_gqa_verify_attention_f16(
            q, k, v, split_output, split_scratch, rows, q_heads, kv_heads,
            head_dim, position_offset, context, splits);
    };
    auto exact = [&]() {
        return pocket::qwen_gqa_verify_attention_f16_exact_cuda(
            q, k, v, exact_output, exact_scratch, rows, q_heads, kv_heads,
            head_dim, position_offset, context);
    };
    auto cublas_qk = [&]() {
        return pocket::qwen_gqa_verify_attention_f16_cublas_qk_cuda(
            q, k, v, cublas_output, exact_scratch, rows, q_heads, kv_heads,
            head_dim, position_offset, context);
    };

    const double split_ms = time_ms(split, iters);
    const double exact_ms = time_ms(exact, iters);
    const double cublas_ms = time_ms(cublas_qk, iters);

    split();
    exact();
    cublas_qk();
    check(cudaDeviceSynchronize(), "result sync");
    std::vector<uint16_t> a(output_elements), b(output_elements), c(output_elements);
    check(cudaMemcpy(a.data(), split_output, a.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost), "copy split");
    check(cudaMemcpy(b.data(), exact_output, b.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost), "copy exact");
    check(cudaMemcpy(c.data(), cublas_output, c.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost), "copy cublas");
    double split_error = 0.0;
    double cublas_error = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        __half ah, bh, ch;
        std::memcpy(&ah, &a[i], 2);
        std::memcpy(&bh, &b[i], 2);
        std::memcpy(&ch, &c[i], 2);
        const float af = __half2float(ah);
        const float bf = __half2float(bh);
        const float cf = __half2float(ch);
        split_error = std::max(split_error, static_cast<double>(std::fabs(af - bf)));
        cublas_error = std::max(cublas_error, static_cast<double>(std::fabs(cf - bf)));
    }

    const double kv_gb = static_cast<double>(2 * kv_elements * sizeof(uint16_t)) / 1e9;
    std::printf("gqa_verify rows=%d context=%d splits=%d iters=%d\n", rows, context, splits, iters);
    std::printf("  split      %.4f ms  %.1f GB/s  max_err=%.3e\n",
                split_ms, kv_gb / (split_ms / 1000.0), split_error);
    std::printf("  exact      %.4f ms  %.1f GB/s\n",
                exact_ms, kv_gb / (exact_ms / 1000.0));
    std::printf("  cublas_qk  %.4f ms  %.1f GB/s  max_err=%.3e\n",
                cublas_ms, kv_gb / (cublas_ms / 1000.0), cublas_error);
    cudaFree(q);
    cudaFree(k);
    cudaFree(v);
    cudaFree(split_output);
    cudaFree(exact_output);
    cudaFree(cublas_output);
    cudaFree(split_scratch);
    cudaFree(exact_scratch);
    return 0;
}
