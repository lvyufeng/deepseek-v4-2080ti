// Sizes the headroom a fused FP8 GEMM (Marlin-style) could recover on the real
// prefill projection shapes.
//
// The production prefill path for rows>8 dequantizes the FP8 weight into an FP16
// scratch buffer and then calls cuBLAS. A fused kernel that consumes the FP8
// codes directly removes the dequant kernel and its scratch write+read, but it
// cannot beat the GEMM itself. So the honest upper bound on a Marlin port is the
// dequant share of the combined path, measured here as:
//
//   combined  = production path (dequant + GEMM), the thing to beat
//   gemm_only = cuBLAS on an already-FP16 weight, the floor a fused kernel
//               approaches but cannot go below
//   implied dequant overhead = combined - gemm_only
//
// A large gap justifies the repack pipeline; a small one means the projections
// are GEMM bound and Marlin buys almost nothing at these shapes.
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
    if (s != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(s));
        std::exit(1);
    }
}

uint16_t h(float v) {
    const __half x = __float2half(v);
    uint16_t b = 0;
    std::memcpy(&b, &x, 2);
    return b;
}

struct Shape { const char* name; int rows; int cols; };

// The five projection shapes that dominate the 32K prefill leaf profile, in
// descending measured cost: mlp.down/up/gate then lin.qkv and lin.out.
const Shape kShapes[] = {
    {"mlp.down", 5120, 4352}, {"mlp.up", 4352, 5120},
    {"mlp.gate", 4352, 5120}, {"linear.qkv", 2560, 5120},
    {"linear.out", 5120, 1536},
};

template <typename F>
double timed(F&& fn, int iters) {
    if (!fn()) { std::fprintf(stderr, "launch failed\n"); std::exit(1); }
    ck(cudaDeviceSynchronize(), "warmup");
    cudaEvent_t a, b;
    ck(cudaEventCreate(&a), "ev");
    ck(cudaEventCreate(&b), "ev");
    ck(cudaEventRecord(a), "rec");
    for (int i = 0; i < iters; ++i) (void)fn();
    ck(cudaEventRecord(b), "rec");
    ck(cudaEventSynchronize(b), "sync");
    float ms = 0;
    ck(cudaEventElapsedTime(&ms, a, b), "el");
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    return ms / iters;
}

}  // namespace

int main(int argc, char** argv) {
    int iters = 20;
    if (argc > 1) iters = std::atoi(argv[1]);
    ck(cudaSetDevice(0), "dev");
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> d(-1.f, 1.f);
    // Prefill dispatches per 512-row slice, so 512 is the shape that matters.
    // 1024 and 2048 show whether the dequant cost amortises with batch.
    const int batches[] = {512, 1024, 2048};
    constexpr int kMaxBatch = 2048;

    for (const Shape& sh : kShapes) {
        const int sc = (sh.cols + kBlock - 1) / kBlock;
        uint16_t *x = nullptr, *s = nullptr, *y = nullptr, *wf16 = nullptr;
        uint8_t* w = nullptr;
        ck(cudaMalloc(&x, size_t(kMaxBatch) * sh.cols * 2), "x");
        ck(cudaMalloc(&w, size_t(sh.rows) * sh.cols), "w");
        ck(cudaMalloc(&wf16, size_t(sh.rows) * sh.cols * 2), "wf16");
        ck(cudaMalloc(&s, size_t(sh.rows) * sc * 2), "s");
        ck(cudaMalloc(&y, size_t(kMaxBatch) * sh.rows * 2), "y");

        std::vector<uint16_t> hx(size_t(kMaxBatch) * sh.cols);
        for (auto& v : hx) v = h(d(rng) * 0.1f);
        ck(cudaMemcpy(x, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice), "cx");
        std::vector<uint8_t> hw(size_t(sh.rows) * sh.cols);
        for (auto& c : hw) c = uint8_t(32 + (rng() % 192));
        ck(cudaMemcpy(w, hw.data(), hw.size(), cudaMemcpyHostToDevice), "cw");
        std::vector<uint16_t> hs(size_t(sh.rows) * sc, h(0.02f));
        ck(cudaMemcpy(s, hs.data(), hs.size() * 2, cudaMemcpyHostToDevice), "cs");

        // Materialize the FP16 weight once so gemm_only measures a pure GEMM.
        if (!pocket::qwen_fp8_e4m3_fp16scale_dequantize_f16_cuda(
                w, s, wf16, sh.rows, sh.cols, sh.cols, sc)) {
            std::fprintf(stderr, "dequant failed\n");
            return 1;
        }
        ck(cudaDeviceSynchronize(), "dq");

        std::printf("%s rows=%d cols=%d fp8=%.1f MiB fp16_scratch=%.1f MiB\n",
                    sh.name, sh.rows, sh.cols,
                    double(sh.rows) * sh.cols / (1024.0 * 1024.0),
                    double(sh.rows) * sh.cols * 2 / (1024.0 * 1024.0));

        const double dq = timed([&]() {
            return pocket::qwen_fp8_e4m3_fp16scale_dequantize_f16_cuda(
                w, s, wf16, sh.rows, sh.cols, sh.cols, sc);
        }, iters);

        for (int b : batches) {
            setenv("QWEN_FP8_F16_PREFILL_CUBLAS", "1", 1);
            const double combined = timed([&]() {
                return pocket::qwen_fp8_e4m3_fp16scale_matmul_rows_f16_cuda(
                    x, w, s, y, b, sh.rows, sh.cols, sh.cols, sh.rows, sh.cols,
                    sc);
            }, iters);
            const double gemm = timed([&]() {
                return pocket::qwen_fp16_matmul_rows_f16_cublas_cuda(
                    x, wf16, y, b, sh.rows, sh.cols, sh.cols, sh.rows, sh.cols);
            }, iters);
            const double flops = 2.0 * b * sh.rows * sh.cols;
            std::printf("  batch=%4d combined=%7.4f ms  gemm_only=%7.4f ms "
                        "(%6.2f TFLOP/s)  dequant_alone=%7.4f ms  "
                        "implied_overhead=%6.1f%%  max_fused_speedup=%5.2fx\n",
                        b, combined, gemm, flops / (gemm * 1.0e9), dq,
                        100.0 * (combined - gemm) / combined,
                        combined / gemm);
        }
        cudaFree(x); cudaFree(w); cudaFree(wf16); cudaFree(s); cudaFree(y);
    }
    return 0;
}
