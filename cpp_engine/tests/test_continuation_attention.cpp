// Small-batch continuation attention agreement with per-row cached attention.

#include "cuda_ops.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
    }
}

template <typename T>
struct DeviceBuffer {
    T* ptr = nullptr;
    explicit DeviceBuffer(size_t count) {
        if (count != 0) check_cuda(cudaMalloc(&ptr, count * sizeof(T)), "cudaMalloc");
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer() { if (ptr != nullptr) cudaFree(ptr); }
};

template <typename T>
void upload(DeviceBuffer<T>& dst, const std::vector<T>& src) {
    if (!src.empty()) {
        check_cuda(cudaMemcpy(dst.ptr, src.data(), src.size() * sizeof(T),
                              cudaMemcpyHostToDevice), "upload");
    }
}

void run_case(int rows, int heads, int head_dim, int cache_slots,
              const std::vector<int32_t>& starts,
              const std::vector<int32_t>& indices) {
    if (starts.size() != static_cast<size_t>(rows + 1) || starts.back() != static_cast<int32_t>(indices.size())) {
        throw std::runtime_error("invalid CSR fixture");
    }
    int max_count = 0;
    for (int row = 0; row < rows; ++row) max_count = std::max(max_count, starts[row + 1] - starts[row]);

    std::vector<float> q(static_cast<size_t>(rows) * heads * head_dim);
    std::vector<float> kv(static_cast<size_t>(cache_slots) * head_dim);
    std::vector<float> sink(heads);
    for (size_t i = 0; i < q.size(); ++i) q[i] = std::sin(static_cast<float>(i) * 0.019f) * 0.4f;
    for (size_t i = 0; i < kv.size(); ++i) kv[i] = std::cos(static_cast<float>(i) * 0.013f) * 0.3f;
    for (int h = 0; h < heads; ++h) sink[h] = -0.2f + 0.03f * h;

    DeviceBuffer<float> d_q(q.size()), d_kv(kv.size()), d_sink(sink.size());
    DeviceBuffer<float> d_batch(q.size()), d_ref(q.size());
    DeviceBuffer<int32_t> d_starts(starts.size()), d_indices(indices.size());
    upload(d_q, q);
    upload(d_kv, kv);
    upload(d_sink, sink);
    upload(d_starts, starts);
    upload(d_indices, indices);

    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    if (!pocket::indexed_cached_attention_rows_cuda(
            d_q.ptr, d_kv.ptr, d_starts.ptr, d_indices.ptr, d_sink.ptr,
            d_batch.ptr, rows, heads, head_dim, max_count, scale)) {
        throw std::runtime_error("batched continuation attention launch failed");
    }
    for (int row = 0; row < rows; ++row) {
        const int begin = starts[row];
        const int count = starts[row + 1] - begin;
        if (!pocket::indexed_cached_single_token_attention_cuda(
                d_q.ptr + static_cast<size_t>(row) * heads * head_dim,
                d_kv.ptr, reinterpret_cast<const int*>(d_indices.ptr + begin),
                d_sink.ptr, d_ref.ptr + static_cast<size_t>(row) * heads * head_dim,
                heads, head_dim, count, scale)) {
            throw std::runtime_error("single-row reference launch failed");
        }
    }
    check_cuda(cudaDeviceSynchronize(), "sync attention");

    std::vector<float> batch(q.size()), ref(q.size());
    check_cuda(cudaMemcpy(batch.data(), d_batch.ptr, batch.size() * sizeof(float),
                          cudaMemcpyDeviceToHost), "download batch");
    check_cuda(cudaMemcpy(ref.data(), d_ref.ptr, ref.size() * sizeof(float),
                          cudaMemcpyDeviceToHost), "download reference");
    float max_abs = 0.0f;
    for (size_t i = 0; i < batch.size(); ++i) max_abs = std::max(max_abs, std::fabs(batch[i] - ref[i]));
    std::cout << "rows=" << rows << " max_count=" << max_count
              << " max_abs=" << max_abs << "\n";
    if (max_abs > 1.0e-7f) throw std::runtime_error("batched attention differs from row reference");

    std::vector<float> first = batch;
    for (int iter = 0; iter < 3; ++iter) {
        if (!pocket::indexed_cached_attention_rows_cuda(
                d_q.ptr, d_kv.ptr, d_starts.ptr, d_indices.ptr, d_sink.ptr,
                d_batch.ptr, rows, heads, head_dim, max_count, scale)) {
            throw std::runtime_error("repeated batched attention launch failed");
        }
        check_cuda(cudaDeviceSynchronize(), "sync repeated attention");
        check_cuda(cudaMemcpy(batch.data(), d_batch.ptr, batch.size() * sizeof(float),
                              cudaMemcpyDeviceToHost), "download repeated batch");
        if (batch != first) throw std::runtime_error("batched attention is not bitwise reproducible");
    }
}

void run_batch_kv_case() {
    constexpr int rows = 4;
    constexpr int heads = 4;
    constexpr int head_dim = 64;
    constexpr int cache_slots = 8;
    const std::vector<int32_t> starts = {0, 5, 11, 18, 26};
    // Non-negative entries address the pre-batch live ring. -2, -3, ...
    // address current continuation rows 0, 1, ... without overwriting it.
    const std::vector<int32_t> indices = {
        5, 6, 7, -2, -1,
        6, 7, 0, -2, -3, -1,
        7, 0, 1, -2, -3, -4, -1,
        0, 1, 2, 3, -2, -3, -4, -5};
    const int max_count = 8;

    std::vector<float> q(static_cast<size_t>(rows) * heads * head_dim);
    std::vector<float> live_kv(static_cast<size_t>(cache_slots) * head_dim);
    std::vector<float> batch_kv(static_cast<size_t>(rows) * head_dim);
    std::vector<float> sink(heads);
    for (size_t i = 0; i < q.size(); ++i) q[i] = std::sin(static_cast<float>(i) * 0.017f) * 0.35f;
    for (size_t i = 0; i < live_kv.size(); ++i) live_kv[i] = std::cos(static_cast<float>(i) * 0.011f) * 0.25f;
    for (size_t i = 0; i < batch_kv.size(); ++i) batch_kv[i] = std::sin(static_cast<float>(i) * 0.023f + 0.4f) * 0.3f;
    for (int h = 0; h < heads; ++h) sink[h] = -0.15f + 0.02f * h;

    std::vector<float> merged_kv = live_kv;
    merged_kv.insert(merged_kv.end(), batch_kv.begin(), batch_kv.end());
    std::vector<int32_t> merged_indices = indices;
    for (int32_t& idx : merged_indices) {
        if (idx <= -2) idx = cache_slots + (-idx - 2);
    }

    DeviceBuffer<float> d_q(q.size()), d_live(live_kv.size()), d_batch_kv(batch_kv.size());
    DeviceBuffer<float> d_merged(merged_kv.size()), d_sink(sink.size());
    DeviceBuffer<float> d_batch(q.size()), d_ref(q.size());
    DeviceBuffer<int32_t> d_starts(starts.size()), d_indices(indices.size());
    DeviceBuffer<int32_t> d_merged_indices(merged_indices.size());
    upload(d_q, q);
    upload(d_live, live_kv);
    upload(d_batch_kv, batch_kv);
    upload(d_merged, merged_kv);
    upload(d_sink, sink);
    upload(d_starts, starts);
    upload(d_indices, indices);
    upload(d_merged_indices, merged_indices);

    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    if (!pocket::indexed_cached_attention_rows_batch_kv_cuda(
            d_q.ptr, d_live.ptr, d_batch_kv.ptr, d_starts.ptr, d_indices.ptr,
            d_sink.ptr, d_batch.ptr, rows, heads, head_dim, max_count, scale)) {
        throw std::runtime_error("batch-local continuation attention launch failed");
    }
    if (!pocket::indexed_cached_attention_rows_cuda(
            d_q.ptr, d_merged.ptr, d_starts.ptr, d_merged_indices.ptr,
            d_sink.ptr, d_ref.ptr, rows, heads, head_dim, max_count, scale)) {
        throw std::runtime_error("merged-cache reference attention launch failed");
    }
    check_cuda(cudaDeviceSynchronize(), "sync batch-local attention");

    std::vector<float> batch(q.size()), ref(q.size());
    check_cuda(cudaMemcpy(batch.data(), d_batch.ptr, batch.size() * sizeof(float),
                          cudaMemcpyDeviceToHost), "download batch-local result");
    check_cuda(cudaMemcpy(ref.data(), d_ref.ptr, ref.size() * sizeof(float),
                          cudaMemcpyDeviceToHost), "download merged reference");
    float max_abs = 0.0f;
    for (size_t i = 0; i < batch.size(); ++i) max_abs = std::max(max_abs, std::fabs(batch[i] - ref[i]));
    std::cout << "batch_kv rows=" << rows << " max_abs=" << max_abs << "\n";
    if (max_abs > 1.0e-7f) throw std::runtime_error("batch-local KV attention differs from merged reference");

    const std::vector<float> first = batch;
    for (int iter = 0; iter < 3; ++iter) {
        if (!pocket::indexed_cached_attention_rows_batch_kv_cuda(
                d_q.ptr, d_live.ptr, d_batch_kv.ptr, d_starts.ptr, d_indices.ptr,
                d_sink.ptr, d_batch.ptr, rows, heads, head_dim, max_count, scale)) {
            throw std::runtime_error("repeated batch-local attention launch failed");
        }
        check_cuda(cudaDeviceSynchronize(), "sync repeated batch-local attention");
        check_cuda(cudaMemcpy(batch.data(), d_batch.ptr, batch.size() * sizeof(float),
                              cudaMemcpyDeviceToHost), "download repeated batch-local result");
        if (batch != first) throw std::runtime_error("batch-local attention is not bitwise reproducible");
    }
}

}  // namespace

int main() {
    try {
        if (!pocket::cuda_runtime_available()) {
            std::cout << "[SKIP] CUDA runtime is not available\n";
            return 0;
        }
        run_case(3, 4, 64, 24,
                 {0, 5, 12, 21},
                 {14, 15, 0, 1, 16,
                  15, 0, 1, 2, 17, 18, 20,
                  0, 1, 2, 3, 4, 16, 19, 21, 23});
        run_case(6, 8, 128, 128,
                 {0, 3, 7, 12, 18, 25, 33},
                 {62, 63, 64,
                  63, 0, 64, 80,
                  0, 1, 2, 65, 90,
                  1, 2, 3, 66, 91, 110,
                  2, 3, 4, 5, 67, 92, 111,
                  3, 4, 5, 6, 7, 68, 93, 127});
        run_batch_kv_case();
        std::cout << "[PASS] continuation indexed attention\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "[FAIL] " << ex.what() << "\n";
        return 1;
    }
}
