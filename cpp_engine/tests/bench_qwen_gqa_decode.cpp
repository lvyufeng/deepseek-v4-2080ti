// Decode-attention crossover at the real Qwen TP4 shape.
//
// The engine picks between the reference score/softmax/value kernels and the
// fused split/merge kernel by a context threshold. This measures where that
// crossover actually is, and checks the two agree numerically, at the shape a
// TP4 rank really runs: 6 Q heads, 1 KV head, head_dim 256.
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "qwen_ops.hpp"

namespace {

uint16_t to_half(float value) {
    uint32_t bits;
    __builtin_memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    int exponent = static_cast<int>((bits >> 23) & 0xFFu) - 127 + 15;
    uint32_t mantissa = bits & 0x7FFFFFu;
    if (exponent <= 0) return static_cast<uint16_t>(sign);
    if (exponent >= 31) return static_cast<uint16_t>(sign | 0x7C00u);
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) |
                                (mantissa >> 13));
}

float from_half(uint16_t value) {
    const uint32_t sign = static_cast<uint32_t>(value & 0x8000u) << 16;
    const uint32_t exponent = (value >> 10) & 0x1Fu;
    const uint32_t mantissa = value & 0x3FFu;
    uint32_t bits;
    if (exponent == 0) {
        bits = sign;
    } else if (exponent == 31) {
        bits = sign | 0x7F800000u | (mantissa << 13);
    } else {
        bits = sign | ((exponent - 15 + 127) << 23) | (mantissa << 13);
    }
    float result;
    __builtin_memcpy(&result, &bits, sizeof(result));
    return result;
}

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "[FAIL] %s: %s\n", what, cudaGetErrorString(status));
        std::exit(1);
    }
}

double time_ms(void (*body)(void*), void* arg, int iterations) {
    body(arg);
    check(cudaDeviceSynchronize(), "warmup");
    cudaEvent_t begin, finish;
    check(cudaEventCreate(&begin), "event create");
    check(cudaEventCreate(&finish), "event create");
    check(cudaEventRecord(begin), "event record");
    for (int i = 0; i < iterations; ++i) body(arg);
    check(cudaEventRecord(finish), "event record");
    check(cudaEventSynchronize(finish), "event sync");
    float elapsed = 0.0f;
    check(cudaEventElapsedTime(&elapsed, begin, finish), "event elapsed");
    check(cudaEventDestroy(begin), "event destroy");
    check(cudaEventDestroy(finish), "event destroy");
    return static_cast<double>(elapsed) / iterations;
}

struct Args {
    const uint16_t* q;
    const uint16_t* k;
    const uint16_t* v;
    uint16_t* out;
    float* scratch;
    int q_heads;
    int kv_heads;
    int head_dim;
    int context_len;
    int max_context;
};

void run_reference(void* raw) {
    const Args& a = *static_cast<Args*>(raw);
    if (!pocket::qwen_gqa_decode_attention_f16(
            a.q, a.k, a.v, a.out, a.scratch, a.q_heads, a.kv_heads, a.head_dim,
            a.context_len, a.max_context)) {
        std::fprintf(stderr, "[FAIL] reference decode launch\n");
        std::exit(1);
    }
}

void run_fused(void* raw) {
    const Args& a = *static_cast<Args*>(raw);
    if (!pocket::qwen_gqa_decode_attention_f16_fused_cuda(
            a.q, a.k, a.v, a.out, a.scratch, a.q_heads, a.kv_heads, a.head_dim,
            a.context_len, a.max_context, 0, 0)) {
        std::fprintf(stderr, "[FAIL] fused decode launch\n");
        std::exit(1);
    }
}

}  // namespace

int main(int argc, char** argv) {
    // Real six-Q-heads-per-KV-group shape unless overridden. TP4 uses 6/1;
    // TP2 uses 12/2 with the same per-group kernel geometry.
    const int q_heads = argc > 1 ? std::atoi(argv[1]) : 6;
    const int kv_heads = argc > 2 ? std::atoi(argv[2]) : 1;
    const int head_dim = argc > 3 ? std::atoi(argv[3]) : 256;
    const int max_context = argc > 4 ? std::atoi(argv[4]) + 16 : 65552;
    const int iterations = argc > 5 ? std::atoi(argv[5]) : 50;

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<uint16_t> host_q(static_cast<size_t>(q_heads) * head_dim);
    for (uint16_t& value : host_q) value = to_half(dist(rng));
    const size_t cache_elements =
        static_cast<size_t>(max_context) * kv_heads * head_dim;
    std::vector<uint16_t> host_k(cache_elements);
    std::vector<uint16_t> host_v(cache_elements);
    for (size_t i = 0; i < cache_elements; ++i) {
        host_k[i] = to_half(dist(rng));
        host_v[i] = to_half(dist(rng));
    }

    uint16_t *q = nullptr, *k = nullptr, *v = nullptr, *out_a = nullptr,
             *out_b = nullptr;
    float* scratch = nullptr;
    check(cudaMalloc(&q, host_q.size() * sizeof(uint16_t)), "malloc q");
    check(cudaMalloc(&k, cache_elements * sizeof(uint16_t)), "malloc k");
    check(cudaMalloc(&v, cache_elements * sizeof(uint16_t)), "malloc v");
    const size_t out_elements = static_cast<size_t>(q_heads) * head_dim;
    check(cudaMalloc(&out_a, out_elements * sizeof(uint16_t)), "malloc out");
    check(cudaMalloc(&out_b, out_elements * sizeof(uint16_t)), "malloc out");
    // Large enough for either layout: q_heads*context for the reference path,
    // q_heads*splits*(head_dim+2) for the fused one. The split count is tunable,
    // so ask the kernel rather than assuming it.
    const bool tensor_core_shape =
        q_heads == kv_heads * 6 && head_dim == 256;
    const size_t scratch_elements =
        static_cast<size_t>(q_heads) * max_context +
        static_cast<size_t>(q_heads) *
            pocket::qwen_gqa_decode_split_count(
                max_context, kv_heads, tensor_core_shape) *
            (head_dim + 2);
    check(cudaMalloc(&scratch, scratch_elements * sizeof(float)), "malloc scratch");
    check(cudaMemcpy(q, host_q.data(), host_q.size() * sizeof(uint16_t),
                     cudaMemcpyHostToDevice), "copy q");
    check(cudaMemcpy(k, host_k.data(), cache_elements * sizeof(uint16_t),
                     cudaMemcpyHostToDevice), "copy k");
    check(cudaMemcpy(v, host_v.data(), cache_elements * sizeof(uint16_t),
                     cudaMemcpyHostToDevice), "copy v");

    std::printf("decode attention crossover q_heads=%d kv_heads=%d head_dim=%d\n",
                q_heads, kv_heads, head_dim);
    std::printf("%9s %12s %12s %8s %12s\n", "context", "reference", "fused",
                "speedup", "max_diff");

    // The fused kernel declines contexts below its own minimum, so start there.
    const int contexts[] = {4096, 8192, 16384, 32768, 65536};
    const int context_count = max_context < 65552 ? 1 : 5;
    for (int context_index = 0; context_index < context_count; ++context_index) {
        const int context_len = contexts[context_index];
        if (context_len >= max_context) continue;
        Args args{q, k, v, out_a, scratch, q_heads, kv_heads, head_dim,
                  context_len, max_context};
        const double reference_ms = time_ms(run_reference, &args, iterations);
        std::vector<uint16_t> host_a(out_elements);
        check(cudaMemcpy(host_a.data(), out_a, out_elements * sizeof(uint16_t),
                         cudaMemcpyDeviceToHost), "copy out");

        args.out = out_b;
        const double fused_ms = time_ms(run_fused, &args, iterations);
        std::vector<uint16_t> host_b(out_elements);
        check(cudaMemcpy(host_b.data(), out_b, out_elements * sizeof(uint16_t),
                         cudaMemcpyDeviceToHost), "copy out");

        float worst = 0.0f;
        for (size_t i = 0; i < out_elements; ++i) {
            worst = std::fmax(worst,
                              std::fabs(from_half(host_a[i]) - from_half(host_b[i])));
        }
        std::printf("%9d %10.4fms %10.4fms %7.2fx %12.3e\n", context_len,
                    reference_ms, fused_ms, reference_ms / fused_ms, worst);
        if (worst > 5e-3f) {
            std::fprintf(stderr, "[FAIL] decode paths disagree at context %d\n",
                         context_len);
            return 1;
        }
    }

    check(cudaFree(q), "free");
    check(cudaFree(k), "free");
    check(cudaFree(v), "free");
    check(cudaFree(out_a), "free");
    check(cudaFree(out_b), "free");
    check(cudaFree(scratch), "free");
    std::printf("[PASS] bench_qwen_gqa_decode\n");
    return 0;
}
