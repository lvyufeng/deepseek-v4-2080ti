// Microbenchmark for the Qwen LM head shape (vocab 62080, hidden 5120). The
// batch-7 section measures the drafter path, where the generic FP32 matmul
// launches a rows x batch grid and re-reads the weight matrix once per draft
// row. The decode section compares its block-per-row kernel with the warp-per-row
// matvec and cuBLAS on the same shape.
#include "qwen_ops.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

using namespace pocket;

namespace {

uint16_t to_half(float value) {
    uint16_t out = 0;
    const __half h = __float2half(value);
    std::memcpy(&out, &h, sizeof(out));
    return out;
}

double median(std::vector<double>& values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

struct Timing {
    double ms;
    double gbps;
};

template <typename Fn>
Timing time_kernel(Fn fn, int rows, int cols, int iters = 30) {
    for (int i = 0; i < 5; ++i) {
        if (!fn()) {
            std::printf("    launch failed\n");
            return {0.0, 0.0};
        }
    }
    cudaDeviceSynchronize();
    std::vector<double> samples;
    samples.reserve(static_cast<size_t>(iters));
    for (int i = 0; i < iters; ++i) {
        const auto started = std::chrono::steady_clock::now();
        fn();
        cudaDeviceSynchronize();
        samples.push_back(std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count() * 1000.0);
    }
    const double ms = median(samples);
    const double bytes = static_cast<double>(rows) * cols * sizeof(uint16_t);
    return {ms, bytes / (ms / 1000.0) / 1e9};
}

// Decode LM head. Same weight shape, one activation row. The generic kernel gives
// each of the 62080 rows a 256-thread block and a shared-memory reduction; the
// matvec kernel gives each warp two rows and reduces by shuffle.
void bench_decode(const uint16_t* d_x, const uint16_t* d_w, int rows, int cols,
                  int weight_stride) {
    float* d_generic = nullptr;
    float* d_matvec = nullptr;
    float* d_cublas = nullptr;
    const size_t logits_bytes = static_cast<size_t>(rows) * sizeof(float);
    if (cudaMalloc(&d_generic, logits_bytes) != cudaSuccess ||
        cudaMalloc(&d_matvec, logits_bytes) != cudaSuccess ||
        cudaMalloc(&d_cublas, logits_bytes) != cudaSuccess) {
        std::printf("decode allocation failed\n");
        return;
    }
    std::printf("decode LM head batch=1 rows=%d cols=%d\n", rows, cols);

    const Timing generic = time_kernel([&]() {
        return qwen_fp16_matmul_rows_f16_f32_cuda(
            d_x, d_w, d_generic, 1, rows, cols, cols, rows, weight_stride);
    }, rows, cols);
    std::printf("  generic (block per row)      %7.3f ms  %6.1f GB/s\n",
                generic.ms, generic.gbps);

    const Timing matvec = time_kernel([&]() {
        return qwen_fp16_matvec_rows_f16_f32_cuda(
            d_x, d_w, d_matvec, 1, rows, cols, cols, rows, weight_stride);
    }, rows, cols);
    std::printf("  matvec (warp per two rows)   %7.3f ms  %6.1f GB/s\n",
                matvec.ms, matvec.gbps);

    const Timing cublas = time_kernel([&]() {
        return qwen_fp16_matmul_rows_f16_f32_cublas_cuda(
            d_x, d_w, d_cublas, 1, rows, cols, cols, rows, weight_stride);
    }, rows, cols);
    std::printf("  cublasGemmEx                 %7.3f ms  %6.1f GB/s\n",
                cublas.ms, cublas.gbps);
    if (generic.ms > 0.0 && matvec.ms > 0.0) {
        std::printf("  matvec speedup %.2fx, cublas speedup %.2fx\n",
                    generic.ms / matvec.ms,
                    cublas.ms > 0.0 ? generic.ms / cublas.ms : 0.0);
    }

    // The reduction order differs, so this reports the spread and whether the
    // greedy pick moves, not equality.
    std::vector<float> a(static_cast<size_t>(rows));
    std::vector<float> b(a.size());
    std::vector<float> c(a.size());
    cudaMemcpy(a.data(), d_generic, logits_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(b.data(), d_matvec, logits_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(c.data(), d_cublas, logits_bytes, cudaMemcpyDeviceToHost);
    double worst_matvec = 0.0;
    double worst_cublas = 0.0;
    int best_a = 0;
    int best_b = 0;
    int best_c = 0;
    for (int row = 0; row < rows; ++row) {
        worst_matvec = std::max(worst_matvec,
                                std::fabs(static_cast<double>(a[row]) - b[row]));
        worst_cublas = std::max(worst_cublas,
                               std::fabs(static_cast<double>(a[row]) - c[row]));
        if (a[row] > a[best_a]) best_a = row;
        if (b[row] > b[best_b]) best_b = row;
        if (c[row] > c[best_c]) best_c = row;
    }
    // Margin over the runner-up says how much slack the greedy pick has; a tiny
    // margin means a reduction-order change can flip the token on its own.
    float second = -std::numeric_limits<float>::infinity();
    for (int row = 0; row < rows; ++row) {
        if (row != best_a && a[row] > second) second = a[row];
    }
    std::printf("  max |generic - matvec| = %.3e, |generic - cublas| = %.3e\n",
                worst_matvec, worst_cublas);
    std::printf("  argmax generic=%d matvec=%d cublas=%d, top1-top2 margin=%.3e\n",
                best_a, best_b, best_c,
                static_cast<double>(a[best_a]) - second);

    cudaFree(d_generic);
    cudaFree(d_matvec);
    cudaFree(d_cublas);
}

}  // namespace

int main() {
    const int batch = 7;
    const int rows = 62080;   // local vocab shard
    const int cols = 5120;    // hidden
    const int x_stride = cols;
    const int y_stride = rows;
    const int weight_stride = cols;

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-0.05f, 0.05f);
    std::vector<uint16_t> host_x(static_cast<size_t>(batch) * x_stride);
    std::vector<uint16_t> host_w(static_cast<size_t>(rows) * weight_stride);
    for (uint16_t& v : host_x) v = to_half(dist(rng));
    for (uint16_t& v : host_w) v = to_half(dist(rng));

    uint16_t* d_x = nullptr;
    uint16_t* d_w = nullptr;
    float* d_generic = nullptr;
    float* d_cublas = nullptr;
    if (cudaMalloc(&d_x, host_x.size() * sizeof(uint16_t)) != cudaSuccess ||
        cudaMalloc(&d_w, host_w.size() * sizeof(uint16_t)) != cudaSuccess ||
        cudaMalloc(&d_generic, static_cast<size_t>(batch) * y_stride * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_cublas, static_cast<size_t>(batch) * y_stride * sizeof(float)) != cudaSuccess) {
        std::printf("allocation failed\n");
        return 1;
    }
    cudaMemcpy(d_x, host_x.data(), host_x.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, host_w.data(), host_w.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);

    const double weight_mb = static_cast<double>(rows) * cols * sizeof(uint16_t) / 1e6;
    std::printf("draft LM head batch=%d rows=%d cols=%d weight=%.1f MB\n",
                batch, rows, cols, weight_mb);

    // Current production path: FP32 output, small-batch reuse disabled.
    const Timing generic = time_kernel([&]() {
        return qwen_fp16_matmul_rows_f16_f32(
            d_x, d_w, d_generic, batch, rows, cols, x_stride, y_stride,
            weight_stride);
    }, rows, cols);
    std::printf("  generic (rows x batch grid)  %7.2f ms  %6.1f GB/s effective\n",
                generic.ms, generic.gbps);

    const Timing cublas = time_kernel([&]() {
        return qwen_fp16_matmul_rows_f16_f32_cublas_cuda(
            d_x, d_w, d_cublas, batch, rows, cols, x_stride, y_stride,
            weight_stride);
    }, rows, cols);
    std::printf("  cublasGemmEx                 %7.2f ms  %6.1f GB/s effective\n",
                cublas.ms, cublas.gbps);

    if (generic.ms > 0.0 && cublas.ms > 0.0) {
        std::printf("  cublas speedup %.2fx\n", generic.ms / cublas.ms);
    }

    // Numerical agreement between the two paths on this shape.
    std::vector<float> a(static_cast<size_t>(batch) * y_stride);
    std::vector<float> b(a.size());
    cudaMemcpy(a.data(), d_generic, a.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(b.data(), d_cublas, b.size() * sizeof(float), cudaMemcpyDeviceToHost);
    double worst = 0.0;
    size_t argmax_disagreements = 0;
    for (int sample = 0; sample < batch; ++sample) {
        int best_a = 0;
        int best_b = 0;
        for (int row = 0; row < rows; ++row) {
            const size_t at = static_cast<size_t>(sample) * y_stride + row;
            worst = std::max(worst, std::fabs(static_cast<double>(a[at]) - b[at]));
            if (a[at] > a[static_cast<size_t>(sample) * y_stride + best_a]) best_a = row;
            if (b[at] > b[static_cast<size_t>(sample) * y_stride + best_b]) best_b = row;
        }
        if (best_a != best_b) ++argmax_disagreements;
    }
    std::printf("  max |generic - cublas| = %.3e, argmax disagreements = %zu/%d\n",
                worst, argmax_disagreements, batch);

    cudaFree(d_generic);
    cudaFree(d_cublas);

    bench_decode(d_x, d_w, rows, cols, weight_stride);

    cudaFree(d_x);
    cudaFree(d_w);
    return 0;
}
