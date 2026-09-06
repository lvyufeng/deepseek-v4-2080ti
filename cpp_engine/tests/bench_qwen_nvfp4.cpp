#include "cuda_ops.hpp"
#include "qwen_ops.hpp"
#include "qwen_weights.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

uint16_t to_half(float value) {
    return __half_as_ushort(__float2half(value));
}

struct DeviceBuffer {
    void* data = nullptr;
    ~DeviceBuffer() { cudaFree(data); }

    template <typename T>
    T* allocate(size_t count) {
        if (cudaMalloc(&data, count * sizeof(T)) != cudaSuccess) return nullptr;
        return static_cast<T*>(data);
    }
};

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

double milliseconds(cudaEvent_t start, cudaEvent_t stop, int iterations) {
    float elapsed = 0.0f;
    require(cudaEventElapsedTime(&elapsed, start, stop) == cudaSuccess,
            "cudaEventElapsedTime");
    return static_cast<double>(elapsed) / iterations;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (!pocket::cuda_runtime_available()) {
            std::printf("[SKIP] bench_qwen_nvfp4 requires CUDA\n");
            return 0;
        }
        int batch = 1;
        int rows = 8704;
        int cols = 5120;
        int iterations = 20;
        for (int i = 1; i < argc; ++i) {
            if (std::strcmp(argv[i], "--batch") == 0 && i + 1 < argc) {
                batch = std::atoi(argv[++i]);
            } else if (std::strcmp(argv[i], "--rows") == 0 && i + 1 < argc) {
                rows = std::atoi(argv[++i]);
            } else if (std::strcmp(argv[i], "--cols") == 0 && i + 1 < argc) {
                cols = std::atoi(argv[++i]);
            } else if (std::strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
                iterations = std::atoi(argv[++i]);
            } else {
                throw std::runtime_error("unknown bench_qwen_nvfp4 argument");
            }
        }
        require(batch > 0 && rows > 0 && cols > 0 && iterations > 0,
                "invalid benchmark extent");
        require(cols % 64 == 0, "NVFP4 columns must be block64 aligned");
        const int blocks_per_row = cols / 64;
        std::vector<uint16_t> input(static_cast<size_t>(batch) * cols);
        std::vector<pocket::QwenNvfp4Block64> blocks(
            static_cast<size_t>(rows) * blocks_per_row);
        std::mt19937 rng(12345);
        std::uniform_real_distribution<float> activation(-2.0f, 2.0f);
        std::uniform_int_distribution<int> fp4(0, 15);
        const uint8_t scales[8] = {0x28, 0x30, 0x34, 0x38,
                                   0x3c, 0x40, 0x44, 0x48};
        for (uint16_t& value : input) value = to_half(activation(rng));
        for (size_t index = 0; index < blocks.size(); ++index) {
            auto& block = blocks[index];
            for (int group = 0; group < 4; ++group) {
                block.d[group] = scales[(index + group) % 8];
            }
            for (uint8_t& packed : block.qs) {
                packed = static_cast<uint8_t>(fp4(rng) | (fp4(rng) << 4));
            }
        }

        DeviceBuffer dx, dw, dy, dq8, dq8_scale;
        uint16_t* d_x = dx.allocate<uint16_t>(input.size());
        uint8_t* d_blocks = dw.allocate<uint8_t>(
            blocks.size() * sizeof(pocket::QwenNvfp4Block64));
        uint16_t* d_y = dy.allocate<uint16_t>(static_cast<size_t>(batch) * rows);
        int8_t* d_q8 = dq8.allocate<int8_t>(static_cast<size_t>(batch) * cols);
        float* d_q8_scale = dq8_scale.allocate<float>(
            static_cast<size_t>(batch) * cols / 32);
        require(d_x && d_blocks && d_y && d_q8 && d_q8_scale,
                "CUDA allocation failed");
        require(cudaMemcpy(d_x, input.data(), input.size() * sizeof(uint16_t),
                           cudaMemcpyHostToDevice) == cudaSuccess,
                "input upload");
        require(cudaMemcpy(d_blocks, blocks.data(),
                           blocks.size() * sizeof(pocket::QwenNvfp4Block64),
                           cudaMemcpyHostToDevice) == cudaSuccess,
                "weight upload");

        auto reference = [&] {
            return pocket::qwen_nvfp4_group16_matmul_rows_f16_cuda(
                d_x, d_blocks, d_y, batch, rows, cols, cols, rows,
                blocks_per_row, 1.0f);
        };
        auto quantize = [&] {
            return pocket::qwen_nvfp4_quantize_q8_group32_f16_cuda(
                d_x, d_q8, d_q8_scale, batch, cols, cols);
        };
        auto dp4a = [&] {
            return pocket::qwen_nvfp4_group16_matmul_q8_f16_cuda(
                d_q8, d_q8_scale, d_blocks, d_y, batch, rows, cols, cols,
                rows, blocks_per_row, 1.0f);
        };
        auto wmma = [&] {
            return pocket::qwen_nvfp4_group16_matmul_q8_wmma_f16_cuda(
                d_q8, d_q8_scale, d_blocks, d_y, batch, rows, cols, cols,
                rows, blocks_per_row, 1.0f);
        };
        for (int warmup = 0; warmup < 3; ++warmup) {
            require(reference() && quantize() && dp4a() && wmma(),
                    "warmup launch");
        }
        require(cudaDeviceSynchronize() == cudaSuccess, "warmup sync");

        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        require(cudaEventCreate(&start) == cudaSuccess &&
                    cudaEventCreate(&stop) == cudaSuccess,
                "event create");
        auto measure = [&](auto&& launch) {
            require(cudaEventRecord(start) == cudaSuccess, "event start");
            for (int i = 0; i < iterations; ++i) require(launch(), "bench launch");
            require(cudaEventRecord(stop) == cudaSuccess &&
                        cudaEventSynchronize(stop) == cudaSuccess,
                    "event stop");
            return milliseconds(start, stop, iterations);
        };
        const double reference_ms = measure(reference);
        const double quantize_ms = measure(quantize);
        require(quantize(), "integer kernel setup quantize");
        const double dp4a_ms = measure(dp4a);
        const double wmma_ms = measure(wmma);
        auto wide_n64 = [&] {
            return pocket::qwen_nvfp4_group16_matmul_q8_wide_n64_f16_cuda(
                d_q8, d_q8_scale, d_blocks, d_y, batch, rows, cols, cols,
                rows, blocks_per_row, 1.0f);
        };
        require(wide_n64(), "wide n64 warmup");
        require(cudaDeviceSynchronize() == cudaSuccess, "wide n64 sync");
        const double wide_n64_ms = measure(wide_n64);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        std::printf(
            "bench_qwen_nvfp4 batch=%d rows=%d cols=%d iterations=%d "
            "reference_ms=%.6f quantize_ms=%.6f dp4a_ms=%.6f "
            "dp4a_total_ms=%.6f dp4a_speedup=%.3f "
            "wmma_ms=%.6f wmma_total_ms=%.6f wmma_speedup=%.3f "
            "wide_n64_ms=%.6f wide_n64_total_ms=%.6f wide_n64_speedup=%.3f\n",
            batch, rows, cols, iterations, reference_ms, quantize_ms, dp4a_ms,
            quantize_ms + dp4a_ms,
            (quantize_ms + dp4a_ms) > 0.0
                ? reference_ms / (quantize_ms + dp4a_ms) : 0.0,
            wmma_ms, quantize_ms + wmma_ms,
            (quantize_ms + wmma_ms) > 0.0
                ? reference_ms / (quantize_ms + wmma_ms) : 0.0,
            wide_n64_ms, quantize_ms + wide_n64_ms,
            (quantize_ms + wide_n64_ms) > 0.0
                ? reference_ms / (quantize_ms + wide_n64_ms) : 0.0);
        return 0;
    } catch (const std::exception& ex) {
        std::printf("[FAIL] bench_qwen_nvfp4 %s\n", ex.what());
        return 1;
    }
}
