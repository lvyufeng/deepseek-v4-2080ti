// Compares the current per-layer Qwen transaction snapshot copies against one
// descriptor-driven gather/scatter launch at the exact TP4 recurrent-state sizes.

#include "qwen_ops.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

namespace {

constexpr int kLayers = 48;
constexpr size_t kStateBytes = 12u * 128u * 128u * sizeof(float);
constexpr size_t kTailBytes = 3u * 2560u * sizeof(uint16_t);

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(status));
        std::exit(1);
    }
}

struct Buffers {
    std::vector<uint8_t*> state;
    std::vector<uint8_t*> tail;
    std::vector<uint8_t*> state_copy;
    std::vector<uint8_t*> tail_copy;
    uint8_t* packed = nullptr;
    pocket::QwenCopyRegion* descriptors = nullptr;

    Buffers() {
        state.resize(kLayers);
        tail.resize(kLayers);
        state_copy.resize(kLayers);
        tail_copy.resize(kLayers);
    }

    ~Buffers() {
        for (uint8_t* value : state) cudaFree(value);
        for (uint8_t* value : tail) cudaFree(value);
        for (uint8_t* value : state_copy) cudaFree(value);
        for (uint8_t* value : tail_copy) cudaFree(value);
        cudaFree(packed);
        cudaFree(descriptors);
    }
};

struct Timing {
    double event_ms = 0.0;
    double submit_ms = 0.0;
    double wall_ms = 0.0;
};

template <typename Launch>
Timing time_launch(Launch launch, int iters) {
    if (!launch()) std::exit(1);
    check(cudaDeviceSynchronize(), "warmup sync");

    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    check(cudaEventCreate(&start), "event create");
    check(cudaEventCreate(&stop), "event create");
    check(cudaEventRecord(start), "event start");
    for (int iteration = 0; iteration < iters; ++iteration) {
        if (!launch()) std::exit(1);
    }
    check(cudaEventRecord(stop), "event stop");
    check(cudaEventSynchronize(stop), "event sync");
    float event_ms = 0.0f;
    check(cudaEventElapsedTime(&event_ms, start, stop), "event elapsed");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    check(cudaDeviceSynchronize(), "host warmup sync");
    const auto wall_started = std::chrono::steady_clock::now();
    for (int iteration = 0; iteration < iters; ++iteration) {
        if (!launch()) std::exit(1);
    }
    const auto submitted = std::chrono::steady_clock::now();
    check(cudaDeviceSynchronize(), "host final sync");
    const auto completed = std::chrono::steady_clock::now();

    Timing result;
    result.event_ms = event_ms / iters;
    result.submit_ms = std::chrono::duration<double, std::milli>(
        submitted - wall_started).count() / iters;
    result.wall_ms = std::chrono::duration<double, std::milli>(
        completed - wall_started).count() / iters;
    return result;
}

void print_timing(const char* name, const Timing& timing, uint64_t bytes) {
    const double bandwidth = bytes / (timing.event_ms * 1.0e6);
    std::printf("  %-14s event=%8.4f ms (%6.1f GB/s) submit=%8.4f ms "
                "wall=%8.4f ms\n",
                name, timing.event_ms, bandwidth, timing.submit_ms,
                timing.wall_ms);
}

}  // namespace

int main(int argc, char** argv) {
    int iters = 200;
    if (argc > 1) iters = std::atoi(argv[1]);
    if (iters <= 0) {
        std::fprintf(stderr, "usage: bench_qwen_transaction_copy [iters]\n");
        return 1;
    }

    Buffers buffers;
    std::vector<pocket::QwenCopyRegion> host_descriptors;
    host_descriptors.reserve(kLayers * 2);
    uint64_t packed_bytes = 0;
    uint64_t total_blocks = 0;
    for (int layer = 0; layer < kLayers; ++layer) {
        check(cudaMalloc(&buffers.state[layer], kStateBytes), "malloc state");
        check(cudaMalloc(&buffers.tail[layer], kTailBytes), "malloc tail");
        check(cudaMalloc(&buffers.state_copy[layer], kStateBytes),
              "malloc state copy");
        check(cudaMalloc(&buffers.tail_copy[layer], kTailBytes),
              "malloc tail copy");
        check(cudaMemset(buffers.state[layer], layer + 1, kStateBytes),
              "init state");
        check(cudaMemset(buffers.tail[layer], layer + 17, kTailBytes),
              "init tail");
        for (auto item : {
                 std::pair<uint8_t*, size_t>{buffers.state[layer], kStateBytes},
                 std::pair<uint8_t*, size_t>{buffers.tail[layer], kTailBytes}}) {
            pocket::QwenCopyRegion descriptor;
            descriptor.device_address = reinterpret_cast<uint64_t>(item.first);
            descriptor.packed_offset = packed_bytes;
            descriptor.bytes = item.second;
            descriptor.first_block = total_blocks;
            host_descriptors.push_back(descriptor);
            packed_bytes += item.second;
            total_blocks += pocket::qwen_copy_region_blocks(item.second);
        }
    }
    check(cudaMalloc(&buffers.packed, packed_bytes), "malloc packed");
    check(cudaMalloc(&buffers.descriptors,
                     host_descriptors.size() * sizeof(host_descriptors[0])),
          "malloc descriptors");
    check(cudaMemcpy(buffers.descriptors, host_descriptors.data(),
                     host_descriptors.size() * sizeof(host_descriptors[0]),
                     cudaMemcpyHostToDevice),
          "copy descriptors");

    auto sync_copies = [&]() {
        for (int layer = 0; layer < kLayers; ++layer) {
            if (cudaMemcpy(buffers.state_copy[layer], buffers.state[layer],
                           kStateBytes, cudaMemcpyDeviceToDevice) != cudaSuccess ||
                cudaMemcpy(buffers.tail_copy[layer], buffers.tail[layer],
                           kTailBytes, cudaMemcpyDeviceToDevice) != cudaSuccess) {
                return false;
            }
        }
        return true;
    };
    auto async_copies = [&]() {
        for (int layer = 0; layer < kLayers; ++layer) {
            if (cudaMemcpyAsync(buffers.state_copy[layer], buffers.state[layer],
                                kStateBytes, cudaMemcpyDeviceToDevice) !=
                    cudaSuccess ||
                cudaMemcpyAsync(buffers.tail_copy[layer], buffers.tail[layer],
                                kTailBytes, cudaMemcpyDeviceToDevice) !=
                    cudaSuccess) {
                return false;
            }
        }
        return true;
    };
    auto gather = [&]() {
        return pocket::qwen_gather_copy_regions(
            buffers.descriptors, static_cast<int>(host_descriptors.size()),
            buffers.packed, total_blocks);
    };
    auto scatter = [&]() {
        return pocket::qwen_scatter_copy_regions(
            buffers.descriptors, static_cast<int>(host_descriptors.size()),
            buffers.packed, total_blocks);
    };

    const Timing sync = time_launch(sync_copies, iters);
    const Timing async = time_launch(async_copies, iters);
    const Timing gather_timing = time_launch(gather, iters);
    const Timing scatter_timing = time_launch(scatter, iters);
    std::printf("qwen_transaction_copy regions=%zu bytes=%llu blocks=%llu "
                "iters=%d\n",
                host_descriptors.size(),
                static_cast<unsigned long long>(packed_bytes),
                static_cast<unsigned long long>(total_blocks), iters);
    print_timing("sync_96", sync, packed_bytes);
    print_timing("async_96", async, packed_bytes);
    print_timing("gather", gather_timing, packed_bytes);
    print_timing("scatter", scatter_timing, packed_bytes);
    std::printf("  gather_speedup event=%.3f submit=%.3f wall=%.3f\n",
                sync.event_ms / gather_timing.event_ms,
                sync.submit_ms / gather_timing.submit_ms,
                sync.wall_ms / gather_timing.wall_ms);
    return 0;
}
