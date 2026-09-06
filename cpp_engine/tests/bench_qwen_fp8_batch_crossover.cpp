// Finds the batch at which the FP8 cuBLAS dequant-and-GEMM path overtakes the
// warp-per-output-channel kernel. The verify path runs batch 2..8, which the
// current batch>=96 gate never routes to cuBLAS.
#include "qwen_ops.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
namespace {
constexpr int kBlock = 128;
void ck(cudaError_t s, const char* w) {
    if (s != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(s)); std::exit(1); }
}
uint16_t h(float v) { const __half x = __float2half(v); uint16_t b = 0; std::memcpy(&b, &x, 2); return b; }
struct Shape { const char* name; int rows; int cols; };
const Shape kShapes[] = {
    {"mlp.gate", 4352, 5120}, {"mlp.down", 5120, 4352},
    {"linear.qkv", 2560, 5120}, {"full.q_gate", 3072, 5120},
    {"linear.out", 5120, 1536},
};
double run(bool cublas, const uint16_t* x, const uint8_t* w, const uint16_t* s,
           uint16_t* y, int batch, int rows, int cols, int sc, int iters) {
    setenv("QWEN_FP8_F16_PREFILL_CUBLAS", cublas ? "1" : "0", 1);
    setenv("QWEN_FP8_F16_SMALL_BATCH", cublas ? "0" : "1", 1);
    auto go = [&]() {
        return pocket::qwen_fp8_e4m3_fp16scale_matmul_rows_f16_cuda(
            x, w, s, y, batch, rows, cols, cols, rows, cols, sc);
    };
    if (!go()) { std::fprintf(stderr, "launch failed b=%d\n", batch); std::exit(1); }
    ck(cudaDeviceSynchronize(), "warmup");
    cudaEvent_t a, b; ck(cudaEventCreate(&a), "ev"); ck(cudaEventCreate(&b), "ev");
    ck(cudaEventRecord(a), "rec");
    for (int i = 0; i < iters; ++i) (void)go();
    ck(cudaEventRecord(b), "rec"); ck(cudaEventSynchronize(b), "sync");
    float ms = 0; ck(cudaEventElapsedTime(&ms, a, b), "el");
    cudaEventDestroy(a); cudaEventDestroy(b);
    return ms / iters;
}
}  // namespace
int main(int argc, char** argv) {
    int iters = 50;
    if (argc > 1) iters = std::atoi(argv[1]);
    ck(cudaSetDevice(0), "dev");
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> d(-1.f, 1.f);
    const int batches[] = {1, 8, 96, 128, 256, 512, 1024};
    constexpr int kMaxBatch = 1024;
    for (const Shape& sh : kShapes) {
        const int sc = (sh.cols + kBlock - 1) / kBlock;
        uint16_t *x = nullptr, *s = nullptr, *y = nullptr; uint8_t* w = nullptr;
        ck(cudaMalloc(&x, size_t(kMaxBatch) * sh.cols * 2), "x");
        ck(cudaMalloc(&w, size_t(sh.rows) * sh.cols), "w");
        ck(cudaMalloc(&s, size_t(sh.rows) * sc * 2), "s");
        ck(cudaMalloc(&y, size_t(kMaxBatch) * sh.rows * 2), "y");
        std::vector<uint16_t> hx(size_t(96) * sh.cols);
        for (auto& v : hx) v = h(d(rng) * 0.1f);
        ck(cudaMemcpy(x, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice), "cx");
        std::vector<uint8_t> hw(size_t(sh.rows) * sh.cols);
        for (auto& c : hw) c = uint8_t(32 + (rng() % 192));
        ck(cudaMemcpy(w, hw.data(), hw.size(), cudaMemcpyHostToDevice), "cw");
        std::vector<uint16_t> hs(size_t(sh.rows) * sc, h(0.02f));
        ck(cudaMemcpy(s, hs.data(), hs.size() * 2, cudaMemcpyHostToDevice), "cs");
        const double mib = double(sh.rows) * sh.cols / (1024.0 * 1024.0);
        std::printf("%s rows=%d cols=%d weight=%.1f MiB\n", sh.name, sh.rows, sh.cols, mib);
        for (int b : batches) {
            const double kern = run(false, x, w, s, y, b, sh.rows, sh.cols, sc, iters);
            const double cub = run(true, x, w, s, y, b, sh.rows, sh.cols, sc, iters);
            std::printf("  batch=%3d kernel=%7.4f ms (%5.0f GB/s) cublas=%7.4f ms (%5.0f GB/s) speedup=%5.2fx%s\n",
                        b, kern, double(sh.rows) * sh.cols / (kern * 1.0e6),
                        cub, double(sh.rows) * sh.cols / (cub * 1.0e6),
                        kern / cub, cub < kern ? "  <-- cublas wins" : "");
        }
        cudaFree(x); cudaFree(w); cudaFree(s); cudaFree(y);
    }
    return 0;
}
