// Measures the per-row cost of the Qwen3.8 TP4 projection kernels at the shard
// shapes that a DFlash2 verification actually issues. Speculative decoding can
// only win when an 8-row verify costs far less than 8 sequential 1-row decodes;
// this benchmark reports that ratio per projection so the speculative ceiling is
// measured instead of assumed.

#include "qwen_ops.hpp"

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
#include <string>
#include <vector>

namespace {

using pocket::qwen_fp8_e4m3_fp16scale_matmul_rows_f16_cuda;
using pocket::qwen_fp8_e4m3_fp16scale_matvec_f16_cuda;

constexpr int kFp8Block = 128;

void check(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(status));
        std::exit(1);
    }
}

uint16_t to_half(float value) {
    const __half h = __float2half(value);
    uint16_t bits = 0;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
}

struct Shape {
    const char* name;
    int out_rows;
    int cols;
};

// Real TP4 shard shapes for Qwen3.8-27B: hidden 5120, linear key/value shards
// 512/1536, full attention 3072 fused q/gate with 256-wide kv, MLP shard 4352.
const Shape kShapes[] = {
    {"linear.qkv", 2560, 5120},
    {"linear.z", 1536, 5120},
    {"linear.out", 5120, 1536},
    {"full.q_gate", 3072, 5120},
    {"full.k", 256, 5120},
    {"full.v", 256, 5120},
    {"full.out", 5120, 1536},
    {"mlp.gate", 4352, 5120},
    {"mlp.up", 4352, 5120},
    {"mlp.down", 5120, 4352},
};

struct DeviceBuffers {
    uint16_t* x = nullptr;
    uint8_t* weight = nullptr;
    uint16_t* scale = nullptr;
    uint16_t* weight_f16 = nullptr;
    uint16_t* y = nullptr;
    ~DeviceBuffers() {
        cudaFree(x);
        cudaFree(weight);
        cudaFree(scale);
        cudaFree(weight_f16);
        cudaFree(y);
    }
};

template <typename Launch>
double time_launch(Launch launch, const char* what, int iters) {
    if (!launch()) {
        std::fprintf(stderr, "%s launch failed\n", what);
        std::exit(1);
    }
    check(cudaDeviceSynchronize(), "warmup sync");
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    check(cudaEventCreate(&start), "event create");
    check(cudaEventCreate(&stop), "event create");
    check(cudaEventRecord(start), "event record");
    for (int i = 0; i < iters; ++i) {
        if (!launch()) std::exit(1);
    }
    check(cudaEventRecord(stop), "event record");
    check(cudaEventSynchronize(stop), "event sync");
    float ms = 0.0f;
    check(cudaEventElapsedTime(&ms, start, stop), "event elapsed");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return static_cast<double>(ms) / iters;
}

double time_rows(const DeviceBuffers& buffers, int batch, int out_rows,
                 int cols, int scale_stride, int iters) {
    auto launch = [&]() {
        if (batch == 1) {
            return qwen_fp8_e4m3_fp16scale_matvec_f16_cuda(
                buffers.x, buffers.weight, buffers.scale, buffers.y, out_rows,
                cols, cols, scale_stride);
        }
        return qwen_fp8_e4m3_fp16scale_matmul_rows_f16_cuda(
            buffers.x, buffers.weight, buffers.scale, buffers.y, batch, out_rows,
            cols, cols, out_rows, cols, scale_stride);
    };
    return time_launch(launch, "online FP8 projection", iters);
}

double time_resident_rows(const DeviceBuffers& buffers, int batch, int out_rows,
                          int cols, int iters) {
    auto launch = [&]() {
        return pocket::qwen_fp16_matmul_rows_f16_cublas_cuda(
            buffers.x, buffers.weight_f16, buffers.y, batch, out_rows, cols,
            cols, out_rows, cols);
    };
    return time_launch(launch, "resident FP16 projection", iters);
}

// Full-attention K and V have the same TP4 shape and consume the same normalized
// activation.  This measures the verify-only candidate that replaces two
// [256, 5120] GEMMs with one [512, 5120] GEMM followed by a row-pair split.
// The caller uses rows 2..8, matching target speculative verification; prefill
// remains on the established separate-projection path.
void bench_full_kv_projection(int iters, std::mt19937& rng) {
    constexpr int kRows = 256;
    constexpr int kCols = 5120;
    const int row_counts[] = {2, 3, 4, 5, 8};
    const size_t weight_elements = static_cast<size_t>(kRows) * kCols;
    const size_t fused_weight_elements = 2 * weight_elements;
    uint16_t* x = nullptr;
    uint16_t* k_weight = nullptr;
    uint16_t* v_weight = nullptr;
    uint16_t* kv_weight = nullptr;
    uint16_t* k_separate = nullptr;
    uint16_t* v_separate = nullptr;
    uint16_t* k_fused = nullptr;
    uint16_t* v_fused = nullptr;
    uint16_t* kv_packed = nullptr;
    check(cudaMalloc(&x, static_cast<size_t>(8) * kCols * sizeof(uint16_t)),
          "malloc full K/V x");
    check(cudaMalloc(&k_weight, weight_elements * sizeof(uint16_t)),
          "malloc full K weight");
    check(cudaMalloc(&v_weight, weight_elements * sizeof(uint16_t)),
          "malloc full V weight");
    check(cudaMalloc(&kv_weight, fused_weight_elements * sizeof(uint16_t)),
          "malloc fused full K/V weight");
    check(cudaMalloc(&k_separate, static_cast<size_t>(8) * kRows * sizeof(uint16_t)),
          "malloc separate K output");
    check(cudaMalloc(&v_separate, static_cast<size_t>(8) * kRows * sizeof(uint16_t)),
          "malloc separate V output");
    check(cudaMalloc(&k_fused, static_cast<size_t>(8) * kRows * sizeof(uint16_t)),
          "malloc fused K output");
    check(cudaMalloc(&v_fused, static_cast<size_t>(8) * kRows * sizeof(uint16_t)),
          "malloc fused V output");
    check(cudaMalloc(&kv_packed, static_cast<size_t>(8) * 2 * kRows * sizeof(uint16_t)),
          "malloc fused K/V output");

    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<uint16_t> host_x(static_cast<size_t>(8) * kCols);
    std::vector<uint16_t> host_k(weight_elements);
    std::vector<uint16_t> host_v(weight_elements);
    for (uint16_t& value : host_x) value = to_half(dist(rng) * 0.1f);
    for (uint16_t& value : host_k) value = to_half(dist(rng) * 0.05f);
    for (uint16_t& value : host_v) value = to_half(dist(rng) * 0.05f);
    check(cudaMemcpy(x, host_x.data(), host_x.size() * sizeof(uint16_t),
                     cudaMemcpyHostToDevice), "copy full K/V x");
    check(cudaMemcpy(k_weight, host_k.data(), host_k.size() * sizeof(uint16_t),
                     cudaMemcpyHostToDevice), "copy full K weight");
    check(cudaMemcpy(v_weight, host_v.data(), host_v.size() * sizeof(uint16_t),
                     cudaMemcpyHostToDevice), "copy full V weight");
    std::vector<uint16_t> host_kv(fused_weight_elements);
    std::copy(host_k.begin(), host_k.end(), host_kv.begin());
    std::copy(host_v.begin(), host_v.end(), host_kv.begin() + weight_elements);
    check(cudaMemcpy(kv_weight, host_kv.data(), host_kv.size() * sizeof(uint16_t),
                     cudaMemcpyHostToDevice), "copy fused full K/V weight");

    std::printf("qwen_verify_full_kv rows=2,3,4,5,8 k_rows=%d cols=%d iters=%d\n",
                kRows, kCols, iters);
    for (int rows : row_counts) {
        auto separate = [&]() {
            return pocket::qwen_fp16_matmul_rows_f16_cublas_cuda(
                       x, k_weight, k_separate, rows, kRows, kCols, kCols,
                       kRows, kCols) &&
                   pocket::qwen_fp16_matmul_rows_f16_cublas_cuda(
                       x, v_weight, v_separate, rows, kRows, kCols, kCols,
                       kRows, kCols);
        };
        auto fused = [&]() {
            return pocket::qwen_fp16_matmul_rows_f16_cublas_cuda(
                       x, kv_weight, kv_packed, rows, 2 * kRows, kCols, kCols,
                       2 * kRows, kCols) &&
                   pocket::qwen_split_rows_pair_f16(
                       kv_packed, k_fused, v_fused, rows, kRows);
        };
        const double separate_ms = time_launch(
            separate, "separate full K/V projection", iters);
        const double fused_ms = time_launch(
            fused, "fused full K/V projection", iters);
        if (!separate() || !fused()) std::exit(1);
        check(cudaDeviceSynchronize(), "full K/V correctness sync");
        std::vector<uint16_t> host_k_separate(static_cast<size_t>(rows) * kRows);
        std::vector<uint16_t> host_v_separate(static_cast<size_t>(rows) * kRows);
        std::vector<uint16_t> host_k_fused(static_cast<size_t>(rows) * kRows);
        std::vector<uint16_t> host_v_fused(static_cast<size_t>(rows) * kRows);
        check(cudaMemcpy(host_k_separate.data(), k_separate,
                         host_k_separate.size() * sizeof(uint16_t),
                         cudaMemcpyDeviceToHost), "copy separate K result");
        check(cudaMemcpy(host_v_separate.data(), v_separate,
                         host_v_separate.size() * sizeof(uint16_t),
                         cudaMemcpyDeviceToHost), "copy separate V result");
        check(cudaMemcpy(host_k_fused.data(), k_fused,
                         host_k_fused.size() * sizeof(uint16_t),
                         cudaMemcpyDeviceToHost), "copy fused K result");
        check(cudaMemcpy(host_v_fused.data(), v_fused,
                         host_v_fused.size() * sizeof(uint16_t),
                         cudaMemcpyDeviceToHost), "copy fused V result");
        double max_abs = 0.0;
        for (size_t index = 0; index < host_k_separate.size(); ++index) {
            max_abs = std::max(max_abs, static_cast<double>(std::fabs(
                __half2float(__ushort_as_half(host_k_separate[index])) -
                __half2float(__ushort_as_half(host_k_fused[index])))));
            max_abs = std::max(max_abs, static_cast<double>(std::fabs(
                __half2float(__ushort_as_half(host_v_separate[index])) -
                __half2float(__ushort_as_half(host_v_fused[index])))));
        }
        std::printf("  rows=%d separate=%.4f ms fused_split=%.4f ms speedup=%.3f "
                    "max_abs=%.3e\n", rows, separate_ms, fused_ms,
                    fused_ms > 0.0 ? separate_ms / fused_ms : 0.0, max_abs);
    }
    cudaFree(x);
    cudaFree(k_weight);
    cudaFree(v_weight);
    cudaFree(kv_weight);
    cudaFree(k_separate);
    cudaFree(v_separate);
    cudaFree(k_fused);
    cudaFree(v_fused);
    cudaFree(kv_packed);
}

// Linear-attention a/b are BF16 checkpoint matrices converted to FP16 at load:
// two 12x5120 weights are row-concatenated into one 24x5120 projection. This
// exact target shape is intentionally much shorter than the resident FP8
// projections above, so benchmark the generic small-batch reduction and tensor-
// core GEMM separately instead of inferring their crossover from large matrices.
void bench_linear_ab(int batch, int iters, std::mt19937& rng) {
    constexpr int kRows = 24;
    constexpr int kCols = 5120;
    const size_t x_elements = static_cast<size_t>(batch) * kCols;
    const size_t weight_elements = static_cast<size_t>(kRows) * kCols;
    const size_t y_elements = static_cast<size_t>(batch) * kRows;
    uint16_t* x = nullptr;
    uint16_t* weight = nullptr;
    uint16_t* generic_y = nullptr;
    uint16_t* cublas_y = nullptr;
    check(cudaMalloc(&x, x_elements * sizeof(uint16_t)), "malloc linear.ab x");
    check(cudaMalloc(&weight, weight_elements * sizeof(uint16_t)),
          "malloc linear.ab weight");
    check(cudaMalloc(&generic_y, y_elements * sizeof(uint16_t)),
          "malloc linear.ab generic output");
    check(cudaMalloc(&cublas_y, y_elements * sizeof(uint16_t)),
          "malloc linear.ab cublas output");

    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<uint16_t> host_x(x_elements);
    std::vector<uint16_t> host_weight(weight_elements);
    for (uint16_t& value : host_x) value = to_half(dist(rng) * 0.1f);
    for (uint16_t& value : host_weight) value = to_half(dist(rng) * 0.05f);
    check(cudaMemcpy(x, host_x.data(), host_x.size() * sizeof(uint16_t),
                     cudaMemcpyHostToDevice), "copy linear.ab x");
    check(cudaMemcpy(weight, host_weight.data(),
                     host_weight.size() * sizeof(uint16_t),
                     cudaMemcpyHostToDevice), "copy linear.ab weight");

    auto generic = [&]() {
        return pocket::qwen_fp16_matmul_rows_f16(
            x, weight, generic_y, batch, kRows, kCols, kCols, kRows, kCols);
    };
    auto cublas = [&]() {
        return pocket::qwen_fp16_matmul_rows_f16_cublas_cuda(
            x, weight, cublas_y, batch, kRows, kCols, kCols, kRows, kCols);
    };
    const double generic_ms = time_launch(generic, "linear.ab generic", iters);
    const double cublas_ms = time_launch(cublas, "linear.ab cuBLAS", iters);

    if (!generic() || !cublas()) std::exit(1);
    check(cudaDeviceSynchronize(), "linear.ab correctness sync");
    std::vector<uint16_t> host_generic(y_elements);
    std::vector<uint16_t> host_cublas(y_elements);
    check(cudaMemcpy(host_generic.data(), generic_y,
                     host_generic.size() * sizeof(uint16_t),
                     cudaMemcpyDeviceToHost), "copy linear.ab generic result");
    check(cudaMemcpy(host_cublas.data(), cublas_y,
                     host_cublas.size() * sizeof(uint16_t),
                     cudaMemcpyDeviceToHost), "copy linear.ab cublas result");
    double max_abs = 0.0;
    for (size_t index = 0; index < y_elements; ++index) {
        const float generic_value = __half2float(
            __ushort_as_half(host_generic[index]));
        const float cublas_value = __half2float(
            __ushort_as_half(host_cublas[index]));
        max_abs = std::max(max_abs, static_cast<double>(
            std::fabs(generic_value - cublas_value)));
    }
    std::printf("qwen_verify_linear_ab rows=%d out=%d cols=%d iters=%d "
                "generic=%.4f ms cublas=%.4f ms speedup=%.3f max_abs=%.3e\n",
                batch, kRows, kCols, iters, generic_ms, cublas_ms,
                cublas_ms > 0.0 ? generic_ms / cublas_ms : 0.0, max_abs);

    cudaFree(x);
    cudaFree(weight);
    cudaFree(generic_y);
    cudaFree(cublas_y);
}

double time_resident_swiglu(const DeviceBuffers& gate, const DeviceBuffers& up,
                            int batch, int out_rows, int cols, int iters) {
    uint16_t* gate_output = nullptr;
    uint16_t* up_output = nullptr;
    const size_t elements = static_cast<size_t>(batch) * out_rows;
    check(cudaMalloc(&gate_output, elements * sizeof(uint16_t)),
          "malloc resident gate output");
    check(cudaMalloc(&up_output, elements * sizeof(uint16_t)),
          "malloc resident up output");
    auto launch = [&]() {
        return pocket::qwen_fp16_matmul_rows_f16_cublas_cuda(
                   gate.x, gate.weight_f16, gate_output, batch, out_rows, cols,
                   cols, out_rows, cols) &&
               pocket::qwen_fp16_matmul_rows_f16_cublas_cuda(
                   gate.x, up.weight_f16, up_output, batch, out_rows, cols,
                   cols, out_rows, cols) &&
               pocket::qwen_silu_mul_rows_f16(
                   gate_output, up_output, gate.y, batch, out_rows);
    };
    const double result = time_launch(launch, "resident FP16 SwiGLU", iters);
    cudaFree(gate_output);
    cudaFree(up_output);
    return result;
}

// The fused SwiGLU is the single largest verify phase after attention, so time it
// at the real TP4 MLP shard shape. QWEN_FP8_SWIGLU_TILE selects the K-tile; this
// harness is far less noisy than the end-to-end bench (~1% run to run versus
// +-4 ms/step), so it is the right place to pick that default.
double time_swiglu(const DeviceBuffers& gate, const DeviceBuffers& up, int batch,
                   int out_rows, int cols, int scale_stride, int iters) {
    auto launch = [&]() {
        return pocket::qwen_fp8_e4m3_fp16scale_swiglu_small_batch_f16_cuda(
            gate.x, gate.weight, gate.scale, up.weight, up.scale, gate.y, batch,
            out_rows, cols, cols, out_rows, cols, scale_stride, nullptr);
    };
    if (!launch()) {
        std::fprintf(stderr, "swiglu launch failed batch=%d rows=%d\n", batch,
                     out_rows);
        std::exit(1);
    }
    check(cudaDeviceSynchronize(), "swiglu warmup sync");
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    check(cudaEventCreate(&start), "event create");
    check(cudaEventCreate(&stop), "event create");
    check(cudaEventRecord(start), "event record");
    for (int i = 0; i < iters; ++i) (void)launch();
    check(cudaEventRecord(stop), "event record");
    check(cudaEventSynchronize(stop), "event sync");
    float ms = 0.0f;
    check(cudaEventElapsedTime(&ms, start, stop), "event elapsed");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return static_cast<double>(ms) / iters;
}

void bench_residual_rmsnorm(int rows, int iters, std::mt19937& rng) {
    constexpr int kCols = 5120;
    constexpr float kEps = 1.0e-6f;
    const size_t elements = static_cast<size_t>(rows) * kCols;
    uint16_t* hidden = nullptr;
    uint16_t* delta = nullptr;
    uint16_t* gamma = nullptr;
    uint16_t* residual = nullptr;
    uint16_t* normalized = nullptr;
    check(cudaMalloc(&hidden, elements * sizeof(uint16_t)), "malloc hidden");
    check(cudaMalloc(&delta, elements * sizeof(uint16_t)), "malloc delta");
    check(cudaMalloc(&gamma, kCols * sizeof(uint16_t)), "malloc gamma");
    check(cudaMalloc(&residual, elements * sizeof(uint16_t)), "malloc residual");
    check(cudaMalloc(&normalized, elements * sizeof(uint16_t)), "malloc norm");
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<uint16_t> host(elements);
    for (uint16_t& value : host) value = to_half(dist(rng));
    check(cudaMemcpy(hidden, host.data(), elements * sizeof(uint16_t),
                     cudaMemcpyHostToDevice), "copy hidden");
    for (uint16_t& value : host) value = to_half(dist(rng) * 0.2f);
    check(cudaMemcpy(delta, host.data(), elements * sizeof(uint16_t),
                     cudaMemcpyHostToDevice), "copy delta");
    host.resize(kCols);
    for (uint16_t& value : host) value = to_half(dist(rng) * 0.1f);
    check(cudaMemcpy(gamma, host.data(), kCols * sizeof(uint16_t),
                     cudaMemcpyHostToDevice), "copy gamma");

    auto reference = [&]() {
        if (cudaMemcpy(residual, hidden, elements * sizeof(uint16_t),
                       cudaMemcpyDeviceToDevice) != cudaSuccess) return false;
        return pocket::qwen_add_inplace_f16(
                   residual, delta, static_cast<int>(elements)) &&
               pocket::qwen_rmsnorm_fp16_gamma_rows_f16(
                   residual, gamma, normalized, rows, kCols, kEps);
    };
    auto fused = [&]() {
        return pocket::qwen_residual_add_rmsnorm_fp16_gamma_rows_f16(
            hidden, delta, gamma, residual, normalized, rows, kCols, kEps);
    };
    auto event_time = [&](auto launch) {
        if (!launch()) std::exit(1);
        check(cudaDeviceSynchronize(), "residual norm warmup sync");
        cudaEvent_t start = nullptr;
        cudaEvent_t stop = nullptr;
        check(cudaEventCreate(&start), "event create");
        check(cudaEventCreate(&stop), "event create");
        check(cudaEventRecord(start), "event record");
        for (int i = 0; i < iters; ++i) {
            if (!launch()) std::exit(1);
        }
        check(cudaEventRecord(stop), "event record");
        check(cudaEventSynchronize(stop), "event sync");
        float elapsed = 0.0f;
        check(cudaEventElapsedTime(&elapsed, start, stop), "event elapsed");
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        return static_cast<double>(elapsed) / iters;
    };
    auto host_time = [&](auto launch) {
        if (!launch()) std::exit(1);
        check(cudaDeviceSynchronize(), "residual norm host warmup sync");
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < iters; ++i) {
            if (!launch()) std::exit(1);
        }
        const auto submitted = std::chrono::steady_clock::now();
        check(cudaDeviceSynchronize(), "residual norm host final sync");
        const auto completed = std::chrono::steady_clock::now();
        const double submit_ms = std::chrono::duration<double, std::milli>(
            submitted - start).count() / iters;
        const double wall_ms = std::chrono::duration<double, std::milli>(
            completed - start).count() / iters;
        return std::pair<double, double>{submit_ms, wall_ms};
    };
    const double reference_event = event_time(reference);
    const double fused_event = event_time(fused);
    const auto reference_host = host_time(reference);
    const auto fused_host = host_time(fused);
    std::printf("qwen_residual_rmsnorm rows=%d cols=%d iters=%d "
                "reference_event=%.4f ms fused_event=%.4f ms speedup=%.3f "
                "reference_submit=%.4f ms fused_submit=%.4f ms "
                "reference_wall=%.4f ms fused_wall=%.4f ms\n",
                rows, kCols, iters, reference_event, fused_event,
                reference_event / fused_event, reference_host.first,
                fused_host.first, reference_host.second, fused_host.second);
    cudaFree(hidden);
    cudaFree(delta);
    cudaFree(gamma);
    cudaFree(residual);
    cudaFree(normalized);
}

// The DFlash2 drafter runs FP16 weights with FP32 output. Its LM head is a tall
// skinny GEMM (local vocab x hidden) at 7 rows, where cuBLAS wastes most of its
// tile. Compare it against the warp-per-output-channel kernel.
struct F16Shape {
    const char* name;
    int out_rows;
    int cols;
};

const F16Shape kF16Shapes[] = {
    {"dflash2.lm_head", 62080, 5120},
    {"dflash2.qkv", 4096, 5120},
    {"dflash2.gate", 4352, 5120},
    {"dflash2.down", 5120, 4352},
    {"dflash2.selector", 256, 5120},
};

struct F16Buffers {
    uint16_t* x = nullptr;
    uint16_t* weight = nullptr;
    float* y = nullptr;
    ~F16Buffers() {
        cudaFree(x);
        cudaFree(weight);
        cudaFree(y);
    }
};

double time_f16(const F16Buffers& buffers, int batch, int out_rows, int cols,
                bool cublas, int iters) {
    auto launch = [&]() {
        if (cublas) {
            return pocket::qwen_fp16_matmul_rows_f16_f32_cublas_cuda(
                buffers.x, buffers.weight, buffers.y, batch, out_rows, cols,
                cols, out_rows, cols);
        }
        // Cost probe for the warp-per-output-channel path. The FP16-output entry
        // point is the only one that currently allows the small-batch kernel; the
        // arithmetic and memory traffic match the FP32-output variant.
        return pocket::qwen_fp16_matmul_rows_f16(
            buffers.x, buffers.weight,
            reinterpret_cast<uint16_t*>(buffers.y), batch, out_rows, cols, cols,
            out_rows, cols);
    };
    if (!launch()) {
        std::fprintf(stderr, "f16 launch failed batch=%d rows=%d cols=%d cublas=%d\n",
                     batch, out_rows, cols, cublas ? 1 : 0);
        std::exit(1);
    }
    check(cudaDeviceSynchronize(), "f16 warmup sync");
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    check(cudaEventCreate(&start), "event create");
    check(cudaEventCreate(&stop), "event create");
    check(cudaEventRecord(start), "event record");
    for (int i = 0; i < iters; ++i) (void)launch();
    check(cudaEventRecord(stop), "event record");
    check(cudaEventSynchronize(stop), "event sync");
    float ms = 0.0f;
    check(cudaEventElapsedTime(&ms, start, stop), "event elapsed");
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return static_cast<double>(ms) / iters;
}

// The DFlash2 context projector is the widest reduction in the drafter:
// taps [rows, 5 * 5120] times fc.weight [5120, 25600]. Real TP4 profiles show it
// dominating prepare_context, so compare the two available GEMM entry points at
// the exact shape.
void bench_context_projection(int iters, std::mt19937& rng) {
    constexpr int kWidth = 25600;
    constexpr int kHidden = 5120;
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    const int row_counts[] = {1, 2, 3, 8};
    uint16_t* x = nullptr;
    uint16_t* weight = nullptr;
    uint16_t* y = nullptr;
    float* y32 = nullptr;
    const size_t weight_elements =
        static_cast<size_t>(kHidden) * kWidth;
    check(cudaMalloc(&x, static_cast<size_t>(8) * kWidth * sizeof(uint16_t)),
          "malloc taps");
    check(cudaMalloc(&weight, weight_elements * sizeof(uint16_t)),
          "malloc projector");
    check(cudaMalloc(&y, static_cast<size_t>(8) * kHidden * sizeof(uint16_t)),
          "malloc projected");
    check(cudaMalloc(&y32, static_cast<size_t>(8) * kHidden * sizeof(float)),
          "malloc projected f32");
    std::vector<uint16_t> host(static_cast<size_t>(8) * kWidth);
    for (uint16_t& value : host) value = to_half(dist(rng) * 0.1f);
    check(cudaMemcpy(x, host.data(), host.size() * sizeof(uint16_t),
                     cudaMemcpyHostToDevice), "copy taps");
    std::vector<uint16_t> host_weight(weight_elements);
    for (uint16_t& value : host_weight) value = to_half(dist(rng) * 0.05f);
    check(cudaMemcpy(weight, host_weight.data(),
                     host_weight.size() * sizeof(uint16_t),
                     cudaMemcpyHostToDevice), "copy projector");
    const double bytes =
        static_cast<double>(weight_elements) * sizeof(uint16_t);
    std::printf("dflash2_context_projection hidden=%d width=%d weight=%.1f MiB "
                "iters=%d\n", kHidden, kWidth, bytes / (1024.0 * 1024.0), iters);
    for (int rows : row_counts) {
        auto time_one = [&](int which) {
            auto launch = [&]() {
                if (which == 0) {
                    return pocket::qwen_dspark_fp16_gemm_rows_f16_cuda(
                        x, weight, y, rows, kHidden, kWidth);
                }
                return pocket::qwen_fp16_matmul_rows_f16_f32_cublas_cuda(
                    x, weight, y32, rows, kHidden, kWidth, kWidth, kHidden,
                    kWidth);
            };
            if (!launch()) {
                std::fprintf(stderr, "context projection launch failed rows=%d "
                             "which=%d\n", rows, which);
                std::exit(1);
            }
            check(cudaDeviceSynchronize(), "context warmup sync");
            cudaEvent_t start = nullptr;
            cudaEvent_t stop = nullptr;
            check(cudaEventCreate(&start), "event create");
            check(cudaEventCreate(&stop), "event create");
            check(cudaEventRecord(start), "event record");
            for (int i = 0; i < iters; ++i) (void)launch();
            check(cudaEventRecord(stop), "event record");
            check(cudaEventSynchronize(stop), "event sync");
            float ms = 0.0f;
            check(cudaEventElapsedTime(&ms, start, stop), "event elapsed");
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            return static_cast<double>(ms) / iters;
        };
        const double direct = time_one(0);
        const double staged = time_one(1);
        std::printf("  rows=%d direct_TN=%.4f ms (%.0f GB/s) "
                    "staged_f32=%.4f ms (%.0f GB/s) speedup=%.3f\n",
                    rows, direct, bytes / (direct * 1.0e6), staged,
                    bytes / (staged * 1.0e6),
                    staged > 0.0 ? direct / staged : 0.0);
    }
    cudaFree(x);
    cudaFree(weight);
    cudaFree(y);
    cudaFree(y32);
}

void bench_f16_draft(int draft_rows, int iters, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::printf("dflash2_draft_projections rows=%d iters=%d\n", draft_rows, iters);
    double cublas_total = 0.0;
    double warp_total = 0.0;
    for (const F16Shape& shape : kF16Shapes) {
        F16Buffers buffers;
        const size_t x_elements = static_cast<size_t>(draft_rows) * shape.cols;
        const size_t weight_elements =
            static_cast<size_t>(shape.out_rows) * shape.cols;
        const size_t y_elements =
            static_cast<size_t>(draft_rows) * shape.out_rows;
        check(cudaMalloc(&buffers.x, x_elements * sizeof(uint16_t)), "malloc x");
        check(cudaMalloc(&buffers.weight, weight_elements * sizeof(uint16_t)),
              "malloc weight");
        check(cudaMalloc(&buffers.y, y_elements * sizeof(float)), "malloc y");
        std::vector<uint16_t> host(x_elements);
        for (uint16_t& value : host) value = to_half(dist(rng) * 0.1f);
        check(cudaMemcpy(buffers.x, host.data(), host.size() * sizeof(uint16_t),
                         cudaMemcpyHostToDevice), "copy x");
        std::vector<uint16_t> host_weight(weight_elements);
        for (uint16_t& value : host_weight) value = to_half(dist(rng) * 0.05f);
        check(cudaMemcpy(buffers.weight, host_weight.data(),
                         host_weight.size() * sizeof(uint16_t),
                         cudaMemcpyHostToDevice), "copy weight");
        const double cublas_ms =
            time_f16(buffers, draft_rows, shape.out_rows, shape.cols, true, iters);
        const double warp_ms =
            time_f16(buffers, draft_rows, shape.out_rows, shape.cols, false, iters);
        cublas_total += cublas_ms;
        warp_total += warp_ms;
        const double bytes =
            static_cast<double>(weight_elements) * sizeof(uint16_t);
        std::printf(
            "  %-16s out=%6d cols=%5d cublas=%.4f ms (%.0f GB/s) "
            "warp=%.4f ms (%.0f GB/s) speedup=%.3f\n",
            shape.name, shape.out_rows, shape.cols, cublas_ms,
            bytes / (cublas_ms * 1.0e6), warp_ms, bytes / (warp_ms * 1.0e6),
            warp_ms > 0.0 ? cublas_ms / warp_ms : 0.0);
    }
    std::printf("  total cublas=%.4f ms warp=%.4f ms draft_speedup=%.3f\n",
                cublas_total, warp_total,
                warp_total > 0.0 ? cublas_total / warp_total : 0.0);
}

}  // namespace

int main(int argc, char** argv) {
    using namespace pocket;
    int iters = 50;
    int verify_rows = 8;
    bool mlp_only = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--iters" && i + 1 < argc) iters = std::atoi(argv[++i]);
        else if (arg == "--rows" && i + 1 < argc) verify_rows = std::atoi(argv[++i]);
        else if (arg == "--mlp-only") mlp_only = true;
    }
    if (iters <= 0 || verify_rows <= 1) {
        std::fprintf(stderr, "usage: bench_qwen_verify_batch [--iters N] [--rows R>=2]\n");
        return 1;
    }
    check(cudaSetDevice(0), "cudaSetDevice");

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    bench_residual_rmsnorm(verify_rows, std::max(iters, 1000), rng);
    bench_linear_ab(verify_rows, std::max(iters, 1000), rng);
    bench_full_kv_projection(std::max(iters, 1000), rng);

    double serial_total = 0.0;
    double verify_total = 0.0;
    double resident_total = 0.0;
    // Fills one FP8 weight/scale set plus activations at the requested shape.
    auto fill_buffers = [&](DeviceBuffers& buffers, int out_rows, int cols,
                            int scale_stride) {
        const size_t x_elements = static_cast<size_t>(verify_rows) * cols;
        const size_t weight_elements = static_cast<size_t>(out_rows) * cols;
        const size_t scale_elements =
            static_cast<size_t>(out_rows) * scale_stride;
        const size_t y_elements = static_cast<size_t>(verify_rows) * out_rows;
        check(cudaMalloc(&buffers.x, x_elements * sizeof(uint16_t)), "malloc x");
        check(cudaMalloc(&buffers.weight, weight_elements), "malloc weight");
        check(cudaMalloc(&buffers.scale, scale_elements * sizeof(uint16_t)),
              "malloc scale");
        check(cudaMalloc(&buffers.weight_f16,
                         weight_elements * sizeof(uint16_t)),
              "malloc resident weight");
        check(cudaMalloc(&buffers.y, y_elements * sizeof(uint16_t)), "malloc y");
        std::vector<uint16_t> host_x(x_elements);
        for (uint16_t& value : host_x) value = to_half(dist(rng) * 0.1f);
        std::vector<uint8_t> host_weight(weight_elements);
        for (uint8_t& code : host_weight) {
            code = static_cast<uint8_t>(32 + (rng() % 192));
        }
        std::vector<uint16_t> host_scale(scale_elements);
        for (uint16_t& value : host_scale) value = to_half(0.02f);
        check(cudaMemcpy(buffers.x, host_x.data(),
                         host_x.size() * sizeof(uint16_t),
                         cudaMemcpyHostToDevice), "copy x");
        check(cudaMemcpy(buffers.weight, host_weight.data(), host_weight.size(),
                         cudaMemcpyHostToDevice), "copy weight");
        check(cudaMemcpy(buffers.scale, host_scale.data(),
                         host_scale.size() * sizeof(uint16_t),
                         cudaMemcpyHostToDevice), "copy scale");
        if (!pocket::qwen_fp8_e4m3_fp16scale_dequantize_f16_cuda(
                buffers.weight, buffers.scale, buffers.weight_f16, out_rows,
                cols, cols, scale_stride)) {
            std::fprintf(stderr, "resident weight dequantization failed\n");
            std::exit(1);
        }
        check(cudaDeviceSynchronize(), "resident weight dequantization sync");
    };

    {
        // Real TP4 MLP shard: intermediate 17408 / 4 ranks = 4352 rows.
        constexpr int kSwigluRows = 4352;
        constexpr int kSwigluCols = 5120;
        const int scale_stride = (kSwigluCols + kFp8Block - 1) / kFp8Block;
        DeviceBuffers gate;
        DeviceBuffers up;
        fill_buffers(gate, kSwigluRows, kSwigluCols, scale_stride);
        fill_buffers(up, kSwigluRows, kSwigluCols, scale_stride);
        const char* tile = std::getenv("QWEN_FP8_SWIGLU_TILE");
        const char* warps = std::getenv("QWEN_FP8_SWIGLU_WARPS");
        const double fused = time_swiglu(gate, up, verify_rows, kSwigluRows,
                                         kSwigluCols, scale_stride, iters);
        // Two separate online matmuls are the non-fused raw-FP8 alternative.
        const double separate =
            time_rows(gate, verify_rows, kSwigluRows, kSwigluCols, scale_stride,
                      iters) +
            time_rows(up, verify_rows, kSwigluRows, kSwigluCols, scale_stride,
                      iters);
        const double resident = time_resident_swiglu(
            gate, up, verify_rows, kSwigluRows, kSwigluCols, iters);
        std::printf("qwen_verify_swiglu rows=%d out=%d cols=%d tile=%s warps=%s "
                    "fused=%.4f ms separate=%.4f ms resident=%.4f ms "
                    "resident_speedup=%.3f\n",
                    verify_rows, kSwigluRows, kSwigluCols,
                    tile == nullptr ? "default" : tile,
                    warps == nullptr ? "default" : warps, fused, separate,
                    resident, resident > 0.0 ? fused / resident : 0.0);
    }

    const char* projection_warps =
        std::getenv("QWEN_FP8_F16_SMALL_BATCH_WARPS");
    std::printf("qwen_verify_batch rows=%d iters=%d warps=%s\n", verify_rows,
                iters, projection_warps == nullptr ? "default" : projection_warps);
    for (const Shape& shape : kShapes) {
        if (mlp_only && std::strncmp(shape.name, "mlp.", 4) != 0) continue;
        const int scale_stride = (shape.cols + kFp8Block - 1) / kFp8Block;
        DeviceBuffers buffers;
        const size_t x_elements =
            static_cast<size_t>(verify_rows) * shape.cols;
        const size_t weight_elements =
            static_cast<size_t>(shape.out_rows) * shape.cols;
        const size_t scale_elements =
            static_cast<size_t>(shape.out_rows) * scale_stride;
        const size_t y_elements =
            static_cast<size_t>(verify_rows) * shape.out_rows;
        check(cudaMalloc(&buffers.x, x_elements * sizeof(uint16_t)), "malloc x");
        check(cudaMalloc(&buffers.weight, weight_elements), "malloc weight");
        check(cudaMalloc(&buffers.scale, scale_elements * sizeof(uint16_t)),
              "malloc scale");
        check(cudaMalloc(&buffers.weight_f16,
                         weight_elements * sizeof(uint16_t)),
              "malloc resident weight");
        check(cudaMalloc(&buffers.y, y_elements * sizeof(uint16_t)), "malloc y");

        std::vector<uint16_t> host_x(x_elements);
        for (uint16_t& value : host_x) value = to_half(dist(rng) * 0.1f);
        std::vector<uint8_t> host_weight(weight_elements);
        // 32..223 stays inside finite E4M3 codes and avoids NaN encodings.
        for (uint8_t& code : host_weight) {
            code = static_cast<uint8_t>(32 + (rng() % 192));
        }
        std::vector<uint16_t> host_scale(scale_elements);
        for (uint16_t& value : host_scale) value = to_half(0.02f);
        check(cudaMemcpy(buffers.x, host_x.data(),
                         host_x.size() * sizeof(uint16_t),
                         cudaMemcpyHostToDevice), "copy x");
        check(cudaMemcpy(buffers.weight, host_weight.data(), host_weight.size(),
                         cudaMemcpyHostToDevice), "copy weight");
        check(cudaMemcpy(buffers.scale, host_scale.data(),
                         host_scale.size() * sizeof(uint16_t),
                         cudaMemcpyHostToDevice), "copy scale");
        if (!pocket::qwen_fp8_e4m3_fp16scale_dequantize_f16_cuda(
                buffers.weight, buffers.scale, buffers.weight_f16,
                shape.out_rows, shape.cols, shape.cols, scale_stride)) {
            std::fprintf(stderr, "resident weight dequantization failed\n");
            std::exit(1);
        }
        check(cudaDeviceSynchronize(), "resident weight dequantization sync");

        const double one_row = time_rows(buffers, 1, shape.out_rows, shape.cols,
                                         scale_stride, iters);
        const double batched = time_rows(buffers, verify_rows, shape.out_rows,
                                         shape.cols, scale_stride, iters);
        const double resident = time_resident_rows(
            buffers, verify_rows, shape.out_rows, shape.cols, iters);
        const double serial = one_row * verify_rows;
        serial_total += serial;
        verify_total += batched;
        resident_total += resident;
        std::printf(
            "  %-12s out=%5d cols=%5d row1=%.4f ms serial%d=%.4f ms "
            "batch%d=%.4f ms resident=%.4f ms resident_speedup=%.3f\n",
            shape.name, shape.out_rows, shape.cols, one_row, verify_rows,
            serial, verify_rows, batched, resident,
            resident > 0.0 ? batched / resident : 0.0);
    }
    std::printf(
        "  total serial%d=%.4f ms batch%d=%.4f ms resident=%.4f ms "
        "projection_speedup=%.3f resident_speedup=%.3f\n",
        verify_rows, serial_total, verify_rows, verify_total, resident_total,
        verify_total > 0.0 ? serial_total / verify_total : 0.0,
        resident_total > 0.0 ? verify_total / resident_total : 0.0);
    if (!mlp_only) {
        // DFlash2 proposes verify_rows - 1 draft rows per block.
        bench_f16_draft(verify_rows - 1, iters, rng);
        bench_context_projection(iters, rng);
    }
    return 0;
}
