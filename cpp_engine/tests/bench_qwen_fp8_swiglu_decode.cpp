// Decode-shape A/B for the fused FP8 gate/up SwiGLU matvec: the scalar column
// loop against the uchar4 vectorized variant. Reports per-call time, the
// implied weight-stream bandwidth, and each variant's error against a
// double-precision reference so a reduction-order change can be judged on
// accuracy rather than only on speed.
#include <cuda_runtime.h>

#include <cmath>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "qwen_ops.hpp"

namespace {

constexpr int kBlock = 128;

float fp8_e4m3_to_float_host(uint8_t code) {
    const uint32_t sign = static_cast<uint32_t>(code >> 7) << 31;
    const int exponent = (code >> 3) & 0xf;
    const int mantissa = code & 0x7;
    if (exponent == 0) {
        if (mantissa == 0) {
            float zero;
            const uint32_t bits = sign;
            std::memcpy(&zero, &bits, sizeof(zero));
            return zero;
        }
        const float value = std::ldexp(static_cast<float>(mantissa) / 8.0f, -6);
        return (sign != 0u) ? -value : value;
    }
    if (exponent == 0xf && mantissa == 0x7) {
        return (sign != 0u) ? -std::numeric_limits<float>::quiet_NaN()
                            : std::numeric_limits<float>::quiet_NaN();
    }
    const float value = std::ldexp(1.0f + static_cast<float>(mantissa) / 8.0f,
                                   exponent - 7);
    return (sign != 0u) ? -value : value;
}

uint16_t float_to_half_host(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t mantissa = bits & 0x7fffffu;
    if (exponent <= 0) return static_cast<uint16_t>(sign);
    if (exponent >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) |
                                (mantissa >> 13));
}

float half_to_float_host(uint16_t value) {
    const uint32_t sign = static_cast<uint32_t>(value & 0x8000u) << 16;
    const uint32_t exponent = (value >> 10) & 0x1fu;
    const uint32_t mantissa = value & 0x3ffu;
    uint32_t bits;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            int shift = 0;
            uint32_t m = mantissa;
            while ((m & 0x400u) == 0u) {
                m <<= 1;
                ++shift;
            }
            m &= 0x3ffu;
            bits = sign | (static_cast<uint32_t>(127 - 15 - shift) << 23) | (m << 13);
        }
    } else if (exponent == 31) {
        bits = sign | 0x7f800000u | (mantissa << 13);
    } else {
        bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    }
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

double silu_double(double v) { return v / (1.0 + std::exp(-v)); }

struct Case {
    int rows;
    int cols;
    const char* label;
};

}  // namespace

int main() {
    // Qwen3.8-27B-FP8 TP4: hidden 5120, intermediate 17408 sharded to 4352.
    const std::vector<Case> cases = {
        {4352, 5120, "tp4 mlp gate/up"},
        {8704, 5120, "tp2 mlp gate/up"},
        {17408, 5120, "tp1 mlp gate/up"},
    };
    const char* iters_env = std::getenv("BENCH_ITERS");
    const int iters = iters_env != nullptr && std::atoi(iters_env) > 0
                          ? std::atoi(iters_env) : 200;

    std::mt19937 rng(1234);
    // 0x7f and 0xff are E4M3 NaN. A single NaN weight makes every comparison
    // below false, and std::max(0.0, NaN) returns 0.0, so the error column
    // would silently read zero.
    // Codes are drawn from the middle of the E4M3 exponent range, which avoids
    // both NaN (0x7f/0xff) and magnitudes that would overflow the FP16 output
    // once 5120 columns are accumulated.
    auto sample_code = [&](std::mt19937& gen) {
        std::uniform_int_distribution<int> sign(0, 1);
        std::uniform_int_distribution<int> exponent(1, 4);
        std::uniform_int_distribution<int> mantissa(0, 7);
        return static_cast<uint8_t>((sign(gen) << 7) | (exponent(gen) << 3) |
                                    mantissa(gen));
    };

    std::printf("%-18s %6s %6s %10s %10s %10s %10s %8s %8s %8s %8s %9s\n",
                "case", "rows", "cols", "scalar_ms", "vec4_ms", "vec8_ms",
                "vec16_ms", "sc_GB/s", "v4_GB/s", "v8_GB/s", "v16_GB/s",
                "worst_err");

    for (const Case& c : cases) {
        const int scale_rows = (c.rows + kBlock - 1) / kBlock;
        const int scale_cols = (c.cols + kBlock - 1) / kBlock;
        std::vector<uint16_t> x(c.cols);
        std::vector<uint8_t> gw(static_cast<size_t>(c.rows) * c.cols);
        std::vector<uint8_t> uw(gw.size());
        std::vector<uint16_t> gs(static_cast<size_t>(scale_rows) * scale_cols);
        std::vector<uint16_t> us(gs.size());
        for (uint16_t& v : x) v = float_to_half_host(
            std::uniform_real_distribution<float>(-1.0f, 1.0f)(rng));
        for (uint8_t& v : gw) v = sample_code(rng);
        for (uint8_t& v : uw) v = sample_code(rng);
        for (uint16_t& v : gs) v = float_to_half_host(
            std::uniform_real_distribution<float>(0.01f, 0.05f)(rng));
        for (uint16_t& v : us) v = float_to_half_host(
            std::uniform_real_distribution<float>(0.01f, 0.05f)(rng));

        uint16_t* d_x = nullptr;
        uint8_t* d_gw = nullptr;
        uint8_t* d_uw = nullptr;
        uint16_t* d_gs = nullptr;
        uint16_t* d_us = nullptr;
        uint16_t* d_y = nullptr;
        if (cudaMalloc(&d_x, x.size() * 2) != cudaSuccess ||
            cudaMalloc(&d_gw, gw.size()) != cudaSuccess ||
            cudaMalloc(&d_uw, uw.size()) != cudaSuccess ||
            cudaMalloc(&d_gs, gs.size() * 2) != cudaSuccess ||
            cudaMalloc(&d_us, us.size() * 2) != cudaSuccess ||
            cudaMalloc(&d_y, static_cast<size_t>(c.rows) * 2) != cudaSuccess) {
            std::printf("[FAIL] alloc\n");
            return 1;
        }
        cudaMemcpy(d_x, x.data(), x.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(d_gw, gw.data(), gw.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_uw, uw.data(), uw.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_gs, gs.data(), gs.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(d_us, us.data(), us.size() * 2, cudaMemcpyHostToDevice);

        // Double-precision reference in plain column order.
        std::vector<double> ref(c.rows);
        for (int row = 0; row < c.rows; ++row) {
            double g = 0.0;
            double u = 0.0;
            for (int col = 0; col < c.cols; ++col) {
                const double xv = half_to_float_host(x[col]);
                const size_t wi = static_cast<size_t>(row) * c.cols + col;
                const size_t si = static_cast<size_t>(row / kBlock) * scale_cols +
                                  col / kBlock;
                g += xv * fp8_e4m3_to_float_host(gw[wi]) * half_to_float_host(gs[si]);
                u += xv * fp8_e4m3_to_float_host(uw[wi]) * half_to_float_host(us[si]);
            }
            ref[row] = silu_double(g) * u;
        }

        auto run = [&](bool vectorized, int cols_per_lane, double* ms,
                       double* err) {
            setenv("QWEN_FP8_F16_SWIGLU_VECTORIZE", vectorized ? "1" : "0", 1);
            char width[8];
            std::snprintf(width, sizeof(width), "%d", cols_per_lane);
            setenv("QWEN_FP8_F16_SWIGLU_COLS_PER_LANE", width, 1);
            // The dispatch caches nothing, so the env var takes effect per call.
            auto launch = [&] {
                return pocket::qwen_fp8_e4m3_fp16scale_swiglu_matvec_f16_cuda(
                    d_x, d_gw, d_gs, d_uw, d_us, d_y, c.rows, c.cols, c.cols,
                    scale_cols, nullptr);
            };
            if (!launch() || cudaDeviceSynchronize() != cudaSuccess) {
                std::printf("[FAIL] launch vectorized=%d\n",
                            vectorized ? 1 : 0);
                std::exit(1);
            }
            std::vector<uint16_t> y(c.rows);
            cudaMemcpy(y.data(), d_y, y.size() * 2, cudaMemcpyDeviceToHost);
            double worst = 0.0;
            for (int row = 0; row < c.rows; ++row) {
                const double got = half_to_float_host(y[row]);
                const double denom = std::max(1e-3, std::fabs(ref[row]));
                worst = std::max(worst, std::fabs(got - ref[row]) / denom);
            }
            *err = worst;

            cudaEvent_t start;
            cudaEvent_t stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);
            for (int i = 0; i < iters; ++i) launch();
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float elapsed = 0.0f;
            cudaEventElapsedTime(&elapsed, start, stop);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            *ms = static_cast<double>(elapsed) / iters;
        };

        double scalar_ms = 0.0;
        double vec4_ms = 0.0;
        double vec8_ms = 0.0;
        double vec16_ms = 0.0;
        double scalar_err = 0.0;
        double vec4_err = 0.0;
        double vec8_err = 0.0;
        double vec16_err = 0.0;
        run(false, 4, &scalar_ms, &scalar_err);
        run(true, 4, &vec4_ms, &vec4_err);
        run(true, 8, &vec8_ms, &vec8_err);
        run(true, 16, &vec16_ms, &vec16_err);

        // Two FP8 weight planes dominate the stream.
        const double bytes = 2.0 * static_cast<double>(c.rows) * c.cols;
        std::printf("%-18s %6d %6d %10.4f %10.4f %10.4f %10.4f "
                    "%8.1f %8.1f %8.1f %8.1f %9.2e\n",
                    c.label, c.rows, c.cols, scalar_ms, vec4_ms, vec8_ms,
                    vec16_ms, bytes / (scalar_ms * 1e-3) / 1e9,
                    bytes / (vec4_ms * 1e-3) / 1e9,
                    bytes / (vec8_ms * 1e-3) / 1e9,
                    bytes / (vec16_ms * 1e-3) / 1e9,
                    std::max(std::max(scalar_err, vec4_err),
                             std::max(vec8_err, vec16_err)));

        cudaFree(d_x);
        cudaFree(d_gw);
        cudaFree(d_uw);
        cudaFree(d_gs);
        cudaFree(d_us);
        cudaFree(d_y);
    }

    // The plain matvec is the other large decode kernel: the down projection
    // (5120 x 4352 per rank) plus every attention projection.
    struct MatvecCase {
        int rows;
        int cols;
        const char* label;
    };
    const std::vector<MatvecCase> matvec_cases = {
        {5120, 4352, "tp4 mlp down"},
        {5120, 8704, "tp2 mlp down"},
        {1536, 5120, "tp4 full q"},
        {3072, 5120, "tp2 full q"},
        {256, 5120, "tp4 full kv"},
        {512, 5120, "tp2 full kv"},
        {2560, 5120, "tp4 linear qkv"},
        {5120, 5120, "tp2 linear qkv"},
        {5120, 2560, "tp4 linear out"},
        {5120, 5120, "tp2 linear out"},
    };
    std::printf("\n%-18s %6s %6s %10s %10s %10s %10s %8s %8s %8s %8s %9s\n",
                "matvec case", "rows", "cols", "vec4_ms", "vec8_ms",
                "rows2_ms", "rows4_ms", "v4_GB/s", "v8_GB/s", "r2_GB/s",
                "r4_GB/s", "worst_err");
    for (const MatvecCase& c : matvec_cases) {
        const int scale_rows = (c.rows + kBlock - 1) / kBlock;
        const int scale_cols = (c.cols + kBlock - 1) / kBlock;
        std::vector<uint16_t> x(c.cols);
        std::vector<uint8_t> w(static_cast<size_t>(c.rows) * c.cols);
        std::vector<uint16_t> s(static_cast<size_t>(scale_rows) * scale_cols);
        std::uniform_real_distribution<float> x_dist(-0.5f, 0.5f);
        std::uniform_real_distribution<float> s_dist(0.01f, 0.05f);
        for (uint16_t& value : x) value = float_to_half_host(x_dist(rng));
        for (uint8_t& value : w) value = sample_code(rng);
        for (uint16_t& value : s) value = float_to_half_host(s_dist(rng));

        uint16_t* d_x = nullptr;
        uint8_t* d_w = nullptr;
        uint16_t* d_s = nullptr;
        uint16_t* d_y = nullptr;
        cudaMalloc(&d_x, x.size() * 2);
        cudaMalloc(&d_w, w.size());
        cudaMalloc(&d_s, s.size() * 2);
        cudaMalloc(&d_y, static_cast<size_t>(c.rows) * 2);
        cudaMemcpy(d_x, x.data(), x.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(d_w, w.data(), w.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_s, s.data(), s.size() * 2, cudaMemcpyHostToDevice);

        std::vector<double> ref(c.rows);
        for (int row = 0; row < c.rows; ++row) {
            double sum = 0.0;
            for (int col = 0; col < c.cols; ++col) {
                const size_t wi = static_cast<size_t>(row) * c.cols + col;
                const size_t si =
                    static_cast<size_t>(row / kBlock) * scale_cols + col / kBlock;
                sum += half_to_float_host(x[col]) *
                       fp8_e4m3_to_float_host(w[wi]) * half_to_float_host(s[si]);
            }
            ref[row] = sum;
        }

        auto run = [&](int cols_per_lane, int rows_per_warp, double* ms,
                       double* err) {
            setenv("QWEN_FP8_F16_VECTORIZE", "1", 1);
            setenv("QWEN_FP8_F16_COLS_PER_LANE",
                   cols_per_lane == 4 ? "4" : "8", 1);
            if (rows_per_warp > 1) {
                setenv("QWEN_FP8_F16_MULTIROW", "1", 1);
                setenv("QWEN_FP8_F16_MULTIROW_ROWS",
                       rows_per_warp == 2 ? "2" : "4", 1);
            } else {
                unsetenv("QWEN_FP8_F16_MULTIROW");
                unsetenv("QWEN_FP8_F16_MULTIROW_ROWS");
            }
            auto launch = [&] {
                return pocket::qwen_fp8_e4m3_fp16scale_matvec_f16_cuda(
                    d_x, d_w, d_s, d_y, c.rows, c.cols, c.cols, scale_cols,
                    nullptr);
            };
            if (!launch() || cudaDeviceSynchronize() != cudaSuccess) {
                std::printf("[FAIL] matvec launch cols_per_lane=%d\n",
                            cols_per_lane);
                std::exit(1);
            }
            std::vector<uint16_t> y(c.rows);
            cudaMemcpy(y.data(), d_y, y.size() * 2, cudaMemcpyDeviceToHost);
            double worst = 0.0;
            for (int row = 0; row < c.rows; ++row) {
                const double denom = std::max(1e-3, std::fabs(ref[row]));
                worst = std::max(
                    worst, std::fabs(half_to_float_host(y[row]) - ref[row]) / denom);
            }
            *err = worst;

            cudaEvent_t start;
            cudaEvent_t stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);
            for (int i = 0; i < iters; ++i) launch();
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float elapsed = 0.0f;
            cudaEventElapsedTime(&elapsed, start, stop);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            *ms = static_cast<double>(elapsed) / iters;
        };

        double vec4_ms = 0.0;
        double vec8_ms = 0.0;
        double rows2_ms = 0.0;
        double rows4_ms = 0.0;
        double vec4_err = 0.0;
        double vec8_err = 0.0;
        double rows2_err = 0.0;
        double rows4_err = 0.0;
        run(4, 1, &vec4_ms, &vec4_err);
        run(8, 1, &vec8_ms, &vec8_err);
        run(8, 2, &rows2_ms, &rows2_err);
        run(8, 4, &rows4_ms, &rows4_err);
        unsetenv("QWEN_FP8_F16_MULTIROW");
        unsetenv("QWEN_FP8_F16_MULTIROW_ROWS");
        unsetenv("QWEN_FP8_F16_COLS_PER_LANE");
        unsetenv("QWEN_FP8_F16_VECTORIZE");

        const double bytes = static_cast<double>(c.rows) * c.cols;
        std::printf("%-18s %6d %6d %10.4f %10.4f %10.4f %10.4f "
                    "%8.1f %8.1f %8.1f %8.1f %9.2e\n", c.label, c.rows,
                    c.cols, vec4_ms, vec8_ms, rows2_ms, rows4_ms,
                    bytes / (vec4_ms * 1e-3) / 1e9,
                    bytes / (vec8_ms * 1e-3) / 1e9,
                    bytes / (rows2_ms * 1e-3) / 1e9,
                    bytes / (rows4_ms * 1e-3) / 1e9,
                    std::max(std::max(vec4_err, vec8_err),
                             std::max(rows2_err, rows4_err)));

        cudaFree(d_x);
        cudaFree(d_w);
        cudaFree(d_s);
        cudaFree(d_y);
    }
    // TP2 DeltaNet qkv+z: compare the established two launches with the
    // two-weight single-grid candidate. Both read the same 5120-element
    // activation and preserve each output row's wide-8 reduction order.
    {
        constexpr int cols = 5120;
        constexpr int qkv_rows = 5120;
        constexpr int z_rows = 3072;
        constexpr int scale_cols = cols / kBlock;
        const int qkv_scale_rows = (qkv_rows + kBlock - 1) / kBlock;
        const int z_scale_rows = (z_rows + kBlock - 1) / kBlock;
        std::vector<uint16_t> x(cols);
        std::vector<uint8_t> qkv_weight(
            static_cast<size_t>(qkv_rows) * cols);
        std::vector<uint8_t> z_weight(static_cast<size_t>(z_rows) * cols);
        std::vector<uint16_t> qkv_scale(
            static_cast<size_t>(qkv_scale_rows) * scale_cols);
        std::vector<uint16_t> z_scale(
            static_cast<size_t>(z_scale_rows) * scale_cols);
        std::uniform_real_distribution<float> x_dist(-0.5f, 0.5f);
        std::uniform_real_distribution<float> s_dist(0.01f, 0.05f);
        for (uint16_t& value : x) value = float_to_half_host(x_dist(rng));
        for (uint8_t& value : qkv_weight) value = sample_code(rng);
        for (uint8_t& value : z_weight) value = sample_code(rng);
        for (uint16_t& value : qkv_scale) value = float_to_half_host(s_dist(rng));
        for (uint16_t& value : z_scale) value = float_to_half_host(s_dist(rng));

        uint16_t* d_x = nullptr;
        uint8_t* d_qkv_weight = nullptr;
        uint8_t* d_z_weight = nullptr;
        uint16_t* d_qkv_scale = nullptr;
        uint16_t* d_z_scale = nullptr;
        uint16_t* d_qkv_reference = nullptr;
        uint16_t* d_z_reference = nullptr;
        uint16_t* d_qkv_dual = nullptr;
        uint16_t* d_z_dual = nullptr;
        if (cudaMalloc(&d_x, x.size() * 2) != cudaSuccess ||
            cudaMalloc(&d_qkv_weight, qkv_weight.size()) != cudaSuccess ||
            cudaMalloc(&d_z_weight, z_weight.size()) != cudaSuccess ||
            cudaMalloc(&d_qkv_scale, qkv_scale.size() * 2) != cudaSuccess ||
            cudaMalloc(&d_z_scale, z_scale.size() * 2) != cudaSuccess ||
            cudaMalloc(&d_qkv_reference,
                       static_cast<size_t>(qkv_rows) * 2) != cudaSuccess ||
            cudaMalloc(&d_z_reference,
                       static_cast<size_t>(z_rows) * 2) != cudaSuccess ||
            cudaMalloc(&d_qkv_dual,
                       static_cast<size_t>(qkv_rows) * 2) != cudaSuccess ||
            cudaMalloc(&d_z_dual,
                       static_cast<size_t>(z_rows) * 2) != cudaSuccess) {
            std::printf("[FAIL] dual projection alloc\n");
            return 1;
        }
        cudaMemcpy(d_x, x.data(), x.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(d_qkv_weight, qkv_weight.data(), qkv_weight.size(),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_z_weight, z_weight.data(), z_weight.size(),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_qkv_scale, qkv_scale.data(), qkv_scale.size() * 2,
                   cudaMemcpyHostToDevice);
        cudaMemcpy(d_z_scale, z_scale.data(), z_scale.size() * 2,
                   cudaMemcpyHostToDevice);
        setenv("QWEN_FP8_F16_VECTORIZE", "1", 1);
        setenv("QWEN_FP8_F16_COLS_PER_LANE", "8", 1);
        unsetenv("QWEN_FP8_F16_MULTIROW");
        auto separate = [&] {
            return pocket::qwen_fp8_e4m3_fp16scale_matvec_f16_cuda(
                       d_x, d_qkv_weight, d_qkv_scale, d_qkv_reference,
                       qkv_rows, cols, cols, scale_cols) &&
                   pocket::qwen_fp8_e4m3_fp16scale_matvec_f16_cuda(
                       d_x, d_z_weight, d_z_scale, d_z_reference, z_rows, cols,
                       cols, scale_cols);
        };
        auto dual = [&] {
            return pocket::qwen_fp8_e4m3_fp16scale_matvec_dual_f16_cuda(
                d_x, d_qkv_weight, d_qkv_scale, d_qkv_dual, qkv_rows, cols,
                scale_cols, d_z_weight, d_z_scale, d_z_dual, z_rows, cols,
                scale_cols, cols);
        };
        if (!separate() || !dual() || cudaDeviceSynchronize() != cudaSuccess) {
            std::printf("[FAIL] dual projection launch\n");
            return 1;
        }
        std::vector<uint16_t> qkv_reference(qkv_rows), qkv_dual(qkv_rows);
        std::vector<uint16_t> z_reference(z_rows), z_dual(z_rows);
        cudaMemcpy(qkv_reference.data(), d_qkv_reference,
                   qkv_reference.size() * 2, cudaMemcpyDeviceToHost);
        cudaMemcpy(qkv_dual.data(), d_qkv_dual, qkv_dual.size() * 2,
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(z_reference.data(), d_z_reference, z_reference.size() * 2,
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(z_dual.data(), d_z_dual, z_dual.size() * 2,
                   cudaMemcpyDeviceToHost);
        const bool exact = qkv_reference == qkv_dual && z_reference == z_dual;

        auto time = [&](auto&& launch) {
            for (int i = 0; i < 20; ++i) launch();
            cudaDeviceSynchronize();
            cudaEvent_t start;
            cudaEvent_t stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);
            for (int i = 0; i < iters; ++i) launch();
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float elapsed = 0.0f;
            cudaEventElapsedTime(&elapsed, start, stop);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            return static_cast<double>(elapsed) / iters;
        };
        const double separate_ms = time(separate);
        const double dual_ms = time(dual);
        const double bytes =
            static_cast<double>(qkv_rows + z_rows) * cols;
        std::printf("\n%-18s %10s %10s %9s %9s %10s\n",
                    "dual projection", "separate_ms", "dual_ms",
                    "sep_GB/s", "dual_GB/s", "bit_exact");
        std::printf("%-18s %10.4f %10.4f %9.1f %9.1f %10s\n",
                    "tp2 qkv+z", separate_ms, dual_ms,
                    bytes / (separate_ms * 1e-3) / 1e9,
                    bytes / (dual_ms * 1e-3) / 1e9,
                    exact ? "yes" : "no");
        unsetenv("QWEN_FP8_F16_COLS_PER_LANE");
        unsetenv("QWEN_FP8_F16_VECTORIZE");
        cudaFree(d_x);
        cudaFree(d_qkv_weight);
        cudaFree(d_z_weight);
        cudaFree(d_qkv_scale);
        cudaFree(d_z_scale);
        cudaFree(d_qkv_reference);
        cudaFree(d_z_reference);
        cudaFree(d_qkv_dual);
        cudaFree(d_z_dual);
        if (!exact) {
            std::printf("[FAIL] dual projection mismatch\n");
            return 1;
        }
    }

    // TP2 full attention projects Q/gate, K, and V from the same hidden row.
    // Grouping them removes the two underfilled 512-row grids.
    {
        constexpr int cols = 5120;
        constexpr int q_rows = 6144;
        constexpr int kv_rows = 512;
        constexpr int scale_cols = cols / 128;
        std::mt19937 rng(20260830);
        std::vector<uint16_t> x(cols);
        std::vector<uint8_t> q_weight(static_cast<size_t>(q_rows) * cols);
        std::vector<uint8_t> k_weight(static_cast<size_t>(kv_rows) * cols);
        std::vector<uint8_t> v_weight(static_cast<size_t>(kv_rows) * cols);
        std::vector<uint16_t> q_scale(
            static_cast<size_t>((q_rows + 127) / 128) * scale_cols);
        std::vector<uint16_t> k_scale(
            static_cast<size_t>((kv_rows + 127) / 128) * scale_cols);
        std::vector<uint16_t> v_scale(
            static_cast<size_t>((kv_rows + 127) / 128) * scale_cols);
        std::uniform_real_distribution<float> x_dist(-0.5f, 0.5f);
        std::uniform_real_distribution<float> s_dist(0.01f, 0.05f);
        for (uint16_t& value : x) value = float_to_half_host(x_dist(rng));
        for (uint8_t& value : q_weight) value = sample_code(rng);
        for (uint8_t& value : k_weight) value = sample_code(rng);
        for (uint8_t& value : v_weight) value = sample_code(rng);
        for (uint16_t& value : q_scale) value = float_to_half_host(s_dist(rng));
        for (uint16_t& value : k_scale) value = float_to_half_host(s_dist(rng));
        for (uint16_t& value : v_scale) value = float_to_half_host(s_dist(rng));

        uint16_t* d_x = nullptr;
        uint8_t *d_q_weight = nullptr, *d_k_weight = nullptr,
                *d_v_weight = nullptr;
        uint16_t *d_q_scale = nullptr, *d_k_scale = nullptr, *d_v_scale = nullptr;
        uint16_t *d_q_reference = nullptr, *d_k_reference = nullptr,
                 *d_v_reference = nullptr, *d_q_triple = nullptr,
                 *d_k_triple = nullptr, *d_v_triple = nullptr;
        if (cudaMalloc(&d_x, x.size() * 2) != cudaSuccess ||
            cudaMalloc(&d_q_weight, q_weight.size()) != cudaSuccess ||
            cudaMalloc(&d_k_weight, k_weight.size()) != cudaSuccess ||
            cudaMalloc(&d_v_weight, v_weight.size()) != cudaSuccess ||
            cudaMalloc(&d_q_scale, q_scale.size() * 2) != cudaSuccess ||
            cudaMalloc(&d_k_scale, k_scale.size() * 2) != cudaSuccess ||
            cudaMalloc(&d_v_scale, v_scale.size() * 2) != cudaSuccess ||
            cudaMalloc(&d_q_reference, static_cast<size_t>(q_rows) * 2) != cudaSuccess ||
            cudaMalloc(&d_k_reference, static_cast<size_t>(kv_rows) * 2) != cudaSuccess ||
            cudaMalloc(&d_v_reference, static_cast<size_t>(kv_rows) * 2) != cudaSuccess ||
            cudaMalloc(&d_q_triple, static_cast<size_t>(q_rows) * 2) != cudaSuccess ||
            cudaMalloc(&d_k_triple, static_cast<size_t>(kv_rows) * 2) != cudaSuccess ||
            cudaMalloc(&d_v_triple, static_cast<size_t>(kv_rows) * 2) != cudaSuccess) {
            std::printf("[FAIL] triple projection alloc\n");
            return 1;
        }
        cudaMemcpy(d_x, x.data(), x.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(d_q_weight, q_weight.data(), q_weight.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_k_weight, k_weight.data(), k_weight.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_v_weight, v_weight.data(), v_weight.size(), cudaMemcpyHostToDevice);
        cudaMemcpy(d_q_scale, q_scale.data(), q_scale.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(d_k_scale, k_scale.data(), k_scale.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(d_v_scale, v_scale.data(), v_scale.size() * 2, cudaMemcpyHostToDevice);
        setenv("QWEN_FP8_F16_VECTORIZE", "1", 1);
        setenv("QWEN_FP8_F16_COLS_PER_LANE", "8", 1);
        unsetenv("QWEN_FP8_F16_MULTIROW");
        auto separate = [&] {
            return pocket::qwen_fp8_e4m3_fp16scale_matvec_f16_cuda(
                       d_x, d_q_weight, d_q_scale, d_q_reference, q_rows, cols,
                       cols, scale_cols) &&
                   pocket::qwen_fp8_e4m3_fp16scale_matvec_f16_cuda(
                       d_x, d_k_weight, d_k_scale, d_k_reference, kv_rows, cols,
                       cols, scale_cols) &&
                   pocket::qwen_fp8_e4m3_fp16scale_matvec_f16_cuda(
                       d_x, d_v_weight, d_v_scale, d_v_reference, kv_rows, cols,
                       cols, scale_cols);
        };
        auto triple = [&] {
            return pocket::qwen_fp8_e4m3_fp16scale_matvec_triple_f16_cuda(
                d_x, d_q_weight, d_q_scale, d_q_triple, q_rows, cols,
                scale_cols, d_k_weight, d_k_scale, d_k_triple, kv_rows, cols,
                scale_cols, d_v_weight, d_v_scale, d_v_triple, kv_rows, cols,
                scale_cols, cols);
        };
        if (!separate() || !triple() || cudaDeviceSynchronize() != cudaSuccess) {
            std::printf("[FAIL] triple projection launch\n");
            return 1;
        }
        std::vector<uint16_t> q_reference(q_rows), q_triple(q_rows);
        std::vector<uint16_t> k_reference(kv_rows), k_triple(kv_rows);
        std::vector<uint16_t> v_reference(kv_rows), v_triple(kv_rows);
        cudaMemcpy(q_reference.data(), d_q_reference, q_reference.size() * 2,
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(q_triple.data(), d_q_triple, q_triple.size() * 2,
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(k_reference.data(), d_k_reference, k_reference.size() * 2,
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(k_triple.data(), d_k_triple, k_triple.size() * 2,
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(v_reference.data(), d_v_reference, v_reference.size() * 2,
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(v_triple.data(), d_v_triple, v_triple.size() * 2,
                   cudaMemcpyDeviceToHost);
        const bool exact = q_reference == q_triple && k_reference == k_triple &&
                           v_reference == v_triple;
        auto time = [&](auto&& launch) {
            for (int i = 0; i < 20; ++i) launch();
            cudaDeviceSynchronize();
            cudaEvent_t start;
            cudaEvent_t stop;
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
            cudaEventRecord(start);
            for (int i = 0; i < iters; ++i) launch();
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);
            float elapsed = 0.0f;
            cudaEventElapsedTime(&elapsed, start, stop);
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
            return static_cast<double>(elapsed) / iters;
        };
        const double separate_ms = time(separate);
        const double triple_ms = time(triple);
        const double bytes = static_cast<double>(q_rows + 2 * kv_rows) * cols;
        std::printf("\n%-18s %10s %10s %9s %9s %10s\n",
                    "triple projection", "separate_ms", "triple_ms",
                    "sep_GB/s", "triple_GB/s", "bit_exact");
        std::printf("%-18s %10.4f %10.4f %9.1f %9.1f %10s\n",
                    "tp2 full q/k/v", separate_ms, triple_ms,
                    bytes / (separate_ms * 1e-3) / 1e9,
                    bytes / (triple_ms * 1e-3) / 1e9,
                    exact ? "yes" : "NO");
        if (!exact) {
            std::printf("[FAIL] triple projection bit exactness\n");
            return 1;
        }
        unsetenv("QWEN_FP8_F16_COLS_PER_LANE");
        unsetenv("QWEN_FP8_F16_VECTORIZE");
        cudaFree(d_x);
        cudaFree(d_q_weight);
        cudaFree(d_k_weight);
        cudaFree(d_v_weight);
        cudaFree(d_q_scale);
        cudaFree(d_k_scale);
        cudaFree(d_v_scale);
        cudaFree(d_q_reference);
        cudaFree(d_k_reference);
        cudaFree(d_v_reference);
        cudaFree(d_q_triple);
        cudaFree(d_k_triple);
        cudaFree(d_v_triple);
    }

    std::printf("[PASS] bench_qwen_fp8_swiglu_decode\n");
    return 0;
}
