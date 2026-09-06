// Bitwise parity between the two MMA prefill kernels.
//
// The occupancy variant only changes where K and V are staged and when: it
// aliases the two shared buffers and loads V after Q*K has parked its scores.
// The arithmetic, the accumulation order, and the online-softmax rescaling are
// untouched, so the outputs must match bit for bit, not merely within a
// tolerance. A tolerance-based check here would hide exactly the kind of
// synchronization bug the aliasing could introduce.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "cuda_ops.hpp"
#include "qwen_ops.hpp"

namespace {

// Qwen3.8 TP4 shard: 24 q heads over 4 kv heads at head_dim 256, sharded four
// ways, so one rank runs 6 q heads over 1 kv head.
constexpr int kQHeads = 6;
constexpr int kKvHeads = 1;
constexpr int kHeadDim = 256;

bool run_case(int rows, int position_offset, int& checked) {
    const int max_context = position_offset + rows;
    const size_t q_elements =
        static_cast<size_t>(rows) * kQHeads * kHeadDim;
    const size_t kv_elements =
        static_cast<size_t>(max_context) * kKvHeads * kHeadDim;

    std::mt19937 rng(20260828u + static_cast<unsigned>(rows + position_offset));
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<uint16_t> host_q(q_elements);
    std::vector<uint16_t> host_k(kv_elements);
    std::vector<uint16_t> host_v(kv_elements);
    auto fill = [&](std::vector<uint16_t>& out) {
        for (uint16_t& value : out) {
            const __half h = __float2half(dist(rng));
            std::memcpy(&value, &h, sizeof(value));
        }
    };
    fill(host_q);
    fill(host_k);
    fill(host_v);

    uint16_t* d_q = nullptr;
    uint16_t* d_k = nullptr;
    uint16_t* d_v = nullptr;
    uint16_t* d_base = nullptr;
    uint16_t* d_occ = nullptr;
    if (cudaMalloc(&d_q, q_elements * sizeof(uint16_t)) != cudaSuccess ||
        cudaMalloc(&d_k, kv_elements * sizeof(uint16_t)) != cudaSuccess ||
        cudaMalloc(&d_v, kv_elements * sizeof(uint16_t)) != cudaSuccess ||
        cudaMalloc(&d_base, q_elements * sizeof(uint16_t)) != cudaSuccess ||
        cudaMalloc(&d_occ, q_elements * sizeof(uint16_t)) != cudaSuccess) {
        std::printf("[SKIP] allocation failed for rows=%d offset=%d\n",
                    rows, position_offset);
        return true;
    }
    cudaMemcpy(d_q, host_q.data(), q_elements * sizeof(uint16_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, host_k.data(), kv_elements * sizeof(uint16_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, host_v.data(), kv_elements * sizeof(uint16_t),
               cudaMemcpyHostToDevice);

    setenv("POCKETLLM_QWEN_GQA_OPTIMIZED", "1", 1);
    setenv("POCKETLLM_QWEN_GQA_MMA_TILE", "1", 1);

    setenv("POCKETLLM_QWEN_GQA_MMA_OCC", "0", 1);
    const bool base_ok = pocket::qwen_gqa_prefill_attention_f16_tiled_cuda(
        d_q, d_k, d_v, d_base, rows, kQHeads, kKvHeads, kHeadDim,
        position_offset, max_context, 0, 0);
    setenv("POCKETLLM_QWEN_GQA_MMA_OCC", "1", 1);
    const bool occ_ok = pocket::qwen_gqa_prefill_attention_f16_tiled_cuda(
        d_q, d_k, d_v, d_occ, rows, kQHeads, kKvHeads, kHeadDim,
        position_offset, max_context, 0, 0);
    cudaDeviceSynchronize();

    std::vector<uint16_t> out_base(q_elements);
    std::vector<uint16_t> out_occ(q_elements);
    cudaMemcpy(out_base.data(), d_base, q_elements * sizeof(uint16_t),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(out_occ.data(), d_occ, q_elements * sizeof(uint16_t),
               cudaMemcpyDeviceToHost);
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_base);
    cudaFree(d_occ);

    if (!base_ok || !occ_ok) {
        std::printf("[FAIL] launch failed rows=%d offset=%d base=%d occ=%d\n",
                    rows, position_offset, static_cast<int>(base_ok),
                    static_cast<int>(occ_ok));
        return false;
    }

    size_t mismatches = 0;
    float worst = 0.0f;
    for (size_t i = 0; i < q_elements; ++i) {
        if (out_base[i] != out_occ[i]) {
            ++mismatches;
            __half a;
            __half b;
            std::memcpy(&a, &out_base[i], sizeof(a));
            std::memcpy(&b, &out_occ[i], sizeof(b));
            const float diff =
                std::fabs(__half2float(a) - __half2float(b));
            if (diff > worst) worst = diff;
        }
    }
    // Guard against both kernels writing nothing, which would make any
    // comparison trivially pass.
    size_t nonzero = 0;
    for (size_t i = 0; i < q_elements; ++i) {
        if (out_base[i] != 0) ++nonzero;
    }
    if (nonzero == 0) {
        std::printf("[FAIL] rows=%d offset=%d produced an all-zero reference\n",
                    rows, position_offset);
        return false;
    }

    std::printf("  mma occ rows=%5d offset=%6d mismatches=%zu worst=%.3e "
                "nonzero=%zu/%zu\n",
                rows, position_offset, mismatches, worst, nonzero, q_elements);
    ++checked;
    return mismatches == 0;
}

}  // namespace

int main() {
    if (!pocket::cuda_runtime_available()) {
        std::printf("[SKIP] test_qwen_gqa_mma_occ requires a CUDA device\n");
        return 0;
    }
    struct Case {
        int rows;
        int offset;
    };
    // Fresh chunk, second chunk, and the deep-history offsets that dominate
    // long-context prefill. 150 rows is not a multiple of the 32-row block, so
    // it exercises the ragged tail.
    const Case cases[] = {
        {  32,     0}, { 150,     0}, { 512,     0}, { 512,  4096},
        {1024,  2048}, {2048,  4096}, {4096,     0}, {4096,  4096},
        {4096, 16384}, {4096, 61440},
    };
    bool ok = true;
    int checked = 0;
    for (const Case& item : cases) {
        if (!run_case(item.rows, item.offset, checked)) ok = false;
    }
    if (checked == 0) {
        std::printf("[FAIL] no case ran\n");
        return 1;
    }
    std::printf("%s test_qwen_gqa_mma_occ (%d cases)\n",
                ok ? "[PASS]" : "[FAIL]", checked);
    return ok ? 0 : 1;
}
