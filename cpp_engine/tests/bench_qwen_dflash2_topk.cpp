// Times the two local top-k paths at the real DFlash2 shape in isolation.
#include "qwen_ops.hpp"
#include <cuda_runtime.h>
#include <cstdio>
#include <random>
#include <vector>
int main() {
    const int rows = 7, vocab = 62080, top_k = 16, iters = 200;
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> dist(-12.f, 12.f);
    std::vector<float> logits((size_t)rows * vocab);
    for (float& v : logits) v = dist(rng);
    float* d_logits; int* d_tok; float* d_val;
    cudaMalloc(&d_logits, logits.size() * 4);
    cudaMalloc(&d_tok, (size_t)rows * top_k * 4);
    cudaMalloc(&d_val, (size_t)rows * top_k * 4);
    cudaMemcpy(d_logits, logits.data(), logits.size() * 4, cudaMemcpyHostToDevice);
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    for (int i = 0; i < 20; ++i)
        pocket::qwen_dflash2_local_topk_f32_cuda(d_logits, d_tok, d_val, rows, vocab, 0, top_k);
    cudaDeviceSynchronize();
    cudaEventRecord(a);
    for (int i = 0; i < iters; ++i)
        pocket::qwen_dflash2_local_topk_f32_cuda(d_logits, d_tok, d_val, rows, vocab, 0, top_k);
    cudaEventRecord(b); cudaDeviceSynchronize();
    float ms = 0; cudaEventElapsedTime(&ms, a, b);
    std::printf("single-block  %.4f ms\n", ms / iters);
    for (int splits : {8, 16, 24, 32, 64}) {
        int* d_pt; float* d_pv;
        cudaMalloc(&d_pt, (size_t)rows * splits * top_k * 4);
        cudaMalloc(&d_pv, (size_t)rows * splits * top_k * 4);
        for (int i = 0; i < 20; ++i)
            pocket::qwen_dflash2_local_topk_split_f32_cuda(d_logits, d_pt, d_pv, d_tok, d_val, rows, vocab, 0, top_k, splits);
        cudaDeviceSynchronize();
        cudaEventRecord(a);
        for (int i = 0; i < iters; ++i)
            pocket::qwen_dflash2_local_topk_split_f32_cuda(d_logits, d_pt, d_pv, d_tok, d_val, rows, vocab, 0, top_k, splits);
        cudaEventRecord(b); cudaDeviceSynchronize();
        cudaEventElapsedTime(&ms, a, b);
        std::printf("splits=%3d    %.4f ms\n", splits, ms / iters);
        cudaFree(d_pt); cudaFree(d_pv);
    }
    return 0;
}
