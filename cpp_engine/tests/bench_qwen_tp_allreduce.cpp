// Prices the TP4 FP16 all-reduce at the exact shapes a DFlash2 draft block would
// issue if the drafter were tensor-parallel instead of replicated. Sharding the
// drafter trades replicated weight traffic for four extra collectives per layer,
// so the collective cost has to be measured before that trade is worth making.

#include "tp_comm.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(status));
        std::exit(1);
    }
}

}  // namespace

int main(int argc, char** argv) {
    int world = 4;
    int rank = 0;
    int device = 0;
    int iters = 200;
    std::string id_path;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--tp-world" && i + 1 < argc) world = std::atoi(argv[++i]);
        else if (arg == "--tp-rank" && i + 1 < argc) rank = std::atoi(argv[++i]);
        else if (arg == "--device" && i + 1 < argc) device = std::atoi(argv[++i]);
        else if (arg == "--iters" && i + 1 < argc) iters = std::atoi(argv[++i]);
        else if (arg == "--nccl-id-path" && i + 1 < argc) id_path = argv[++i];
    }
    if (id_path.empty() || world <= 1 || iters <= 0) {
        std::fprintf(stderr,
                     "usage: bench_qwen_tp_allreduce --nccl-id-path P "
                     "--tp-world N --tp-rank R [--device D] [--iters N]\n");
        return 1;
    }
    if (!pocket::tp_comm_available()) {
        std::fprintf(stderr, "this build has no NCCL support\n");
        return 1;
    }
    check(cudaSetDevice(device), "cudaSetDevice");

    // 8 rows is the DFlash2 verify/draft block; 5120 is hidden size. A sharded
    // drafter would add one reduce after MLP down and one after attention out,
    // per layer, for five layers.
    const int rows[] = {1, 8};
    const int widths[] = {5120};
    constexpr int kDraftReducesPerProposal = 5 * 2;

    for (int row_count : rows) {
        for (int width : widths) {
            const int count = row_count * width;
            uint16_t* buffer = nullptr;
            check(cudaMalloc(&buffer, static_cast<size_t>(count) * sizeof(uint16_t)),
                  "cudaMalloc reduce buffer");
            std::vector<uint16_t> host(static_cast<size_t>(count));
            const __half one = __float2half(1.0f);
            uint16_t one_bits = 0;
            std::memcpy(&one_bits, &one, sizeof(one_bits));
            for (uint16_t& value : host) value = one_bits;
            check(cudaMemcpy(buffer, host.data(),
                             host.size() * sizeof(uint16_t),
                             cudaMemcpyHostToDevice), "copy reduce buffer");
            // Warm up the communicator so connection setup is not timed.
            for (int i = 0; i < 5; ++i) {
                pocket::tp_all_reduce_sum_f16_inplace(
                    world, rank, device, id_path.c_str(), buffer, count);
            }
            check(cudaDeviceSynchronize(), "warmup sync");
            const auto started = std::chrono::steady_clock::now();
            for (int i = 0; i < iters; ++i) {
                pocket::tp_all_reduce_sum_f16_inplace(
                    world, rank, device, id_path.c_str(), buffer, count);
            }
            check(cudaDeviceSynchronize(), "reduce sync");
            const double total_ms = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - started).count();
            const double per_call = total_ms / iters;
            std::printf(
                "tp_allreduce rank=%d rows=%d width=%d bytes=%zu per_call=%.4f ms "
                "draft_block_cost=%.4f ms\n",
                rank, row_count, width,
                static_cast<size_t>(count) * sizeof(uint16_t), per_call,
                per_call * kDraftReducesPerProposal);
            cudaFree(buffer);
        }
    }
    return 0;
}
