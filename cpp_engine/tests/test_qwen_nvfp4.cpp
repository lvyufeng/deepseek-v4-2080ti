#include "cuda_ops.hpp"
#include "qwen_ops.hpp"
#include "qwen_weights.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

uint16_t to_half(float value) {
    return __half_as_ushort(__float2half(value));
}

float from_half(uint16_t bits) {
    return __half2float(__ushort_as_half(bits));
}

float from_fp8_e4m3(uint8_t code) {
    const uint32_t sign = static_cast<uint32_t>(code & 0x80u) << 24;
    const uint32_t exponent = (code >> 3) & 0xfu;
    const uint32_t mantissa = code & 0x7u;
    uint32_t bits;
    if (exponent != 0) {
        bits = sign | ((exponent + 120u) << 23) | (mantissa << 20);
    } else if (mantissa >= 4) {
        bits = sign | (120u << 23) | ((mantissa - 4u) << 21);
    } else if (mantissa >= 2) {
        bits = sign | (119u << 23) | ((mantissa - 2u) << 22);
    } else {
        bits = sign | (mantissa == 0 ? 0u : (118u << 23));
    }
    float output;
    std::memcpy(&output, &bits, sizeof(output));
    return output;
}

float from_e2m1(uint8_t code) {
    static constexpr float values[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
       -0.0f,-0.5f,-1.0f,-1.5f,-2.0f,-3.0f,-4.0f,-6.0f};
    return values[code & 15u];
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

}  // namespace

namespace {

// The wide 128x64 tile masks partial tiles instead of shrinking the launch, so
// the tail behavior needs a shape that is not a multiple of either tile side.
// FP16 output cannot resolve the absolute reference at this magnitude, so the
// wide result is compared against the FP32 WMMA output of the same Q8 input.
bool wide_n64_tail_matches_wmma() {
    constexpr int batch = 129;
    constexpr int rows = 67;
    constexpr int cols = 320;
    constexpr int blocks_per_row = cols / 64;
    constexpr float global_factor = 0.125f;

    std::vector<pocket::QwenNvfp4Block64> blocks(rows * blocks_per_row);
    for (size_t index = 0; index < blocks.size(); ++index) {
        auto& record = blocks[index];
        static constexpr uint8_t scales[4] = {0x38, 0x34, 0x3c, 0x30};
        for (int group = 0; group < 4; ++group) {
            record.d[group] = scales[(index + group) % 4];
        }
        for (int byte = 0; byte < 32; ++byte) {
            const uint8_t low = static_cast<uint8_t>((index + byte) & 15);
            const uint8_t high = static_cast<uint8_t>((index + byte * 3 + 1) & 15);
            record.qs[byte] = static_cast<uint8_t>(low | (high << 4));
        }
    }
    std::vector<uint16_t> input(static_cast<size_t>(batch) * cols);
    for (int sample = 0; sample < batch; ++sample) {
        for (int col = 0; col < cols; ++col) {
            input[static_cast<size_t>(sample) * cols + col] = to_half(
                static_cast<float>(((sample * 5 + col) % 11) - 5) / 16.0f);
        }
    }

    DeviceBuffer dx, dw, dq8, dq8_scale, dy_wide, dy_wmma;
    uint16_t* d_x = dx.allocate<uint16_t>(input.size());
    uint8_t* d_blocks = dw.allocate<uint8_t>(
        blocks.size() * sizeof(pocket::QwenNvfp4Block64));
    int8_t* d_q8 = dq8.allocate<int8_t>(static_cast<size_t>(batch) * cols);
    float* d_q8_scale = dq8_scale.allocate<float>(
        static_cast<size_t>(batch) * cols / 32);
    uint16_t* d_wide = dy_wide.allocate<uint16_t>(
        static_cast<size_t>(batch) * rows);
    float* d_wmma = dy_wmma.allocate<float>(static_cast<size_t>(batch) * rows);
    if (!d_x || !d_blocks || !d_q8 || !d_q8_scale || !d_wide || !d_wmma ||
        cudaMemcpy(d_x, input.data(), input.size() * sizeof(uint16_t),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_blocks, blocks.data(),
                   blocks.size() * sizeof(pocket::QwenNvfp4Block64),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        !pocket::qwen_nvfp4_quantize_q8_group32_f16_cuda(
            d_x, d_q8, d_q8_scale, batch, cols, cols) ||
        !pocket::qwen_nvfp4_group16_matmul_q8_wmma_f32_cuda(
            d_q8, d_q8_scale, d_blocks, d_wmma, batch, rows, cols, cols,
            rows, blocks_per_row, global_factor) ||
        !pocket::qwen_nvfp4_group16_matmul_q8_wide_n64_f16_cuda(
            d_q8, d_q8_scale, d_blocks, d_wide, batch, rows, cols, cols,
            rows, blocks_per_row, global_factor) ||
        cudaDeviceSynchronize() != cudaSuccess) {
        std::printf("[FAIL] NVFP4 wide-N64 tail launch\n");
        return false;
    }
    std::vector<uint16_t> got_wide(static_cast<size_t>(batch) * rows);
    std::vector<float> got_wmma(static_cast<size_t>(batch) * rows);
    cudaMemcpy(got_wide.data(), d_wide, got_wide.size() * sizeof(uint16_t),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(got_wmma.data(), d_wmma, got_wmma.size() * sizeof(float),
               cudaMemcpyDeviceToHost);

    double worst_relative = 0.0;
    double magnitude = 0.0;
    for (size_t index = 0; index < got_wide.size(); ++index) {
        const double expected = got_wmma[index];
        const double actual = from_half(got_wide[index]);
        magnitude = std::max(magnitude, std::fabs(expected));
        worst_relative = std::max(worst_relative,
            std::fabs(actual - expected) /
                std::max(1.0, std::fabs(expected)));
    }
    // A zero-magnitude comparison would pass trivially, so require the fixture
    // to have actually produced output before trusting the agreement.
    if (magnitude < 1.0 || worst_relative > 1.0e-3) {
        std::printf(
            "[FAIL] NVFP4 wide-N64 tail batch=%d rows=%d cols=%d "
            "relative=%.3e magnitude=%.3e\n",
            batch, rows, cols, worst_relative, magnitude);
        return false;
    }
    std::printf(
        "wide_n64_tail batch=%d rows=%d cols=%d relative=%.3e magnitude=%.3e\n",
        batch, rows, cols, worst_relative, magnitude);
    return true;
}

}  // namespace

int main() {
    if (!pocket::cuda_runtime_available()) {
        std::printf("[SKIP] test_qwen_nvfp4 requires CUDA\n");
        return 0;
    }
    if (!wide_n64_tail_matches_wmma()) return 1;
    constexpr int batch = 3;
    constexpr int rows = 5;
    constexpr int cols = 128;
    constexpr int blocks_per_row = cols / 64;
    constexpr float global_factor = 0.25f;
    std::vector<pocket::QwenNvfp4Block64> blocks(rows * blocks_per_row);
    for (int row = 0; row < rows; ++row) {
        for (int block = 0; block < blocks_per_row; ++block) {
            auto& record = blocks[row * blocks_per_row + block];
            record.d[0] = 0x38;  // 1.0
            record.d[1] = 0x40;  // 2.0
            record.d[2] = 0x30;  // 0.5
            record.d[3] = 0x3c;  // 1.5
            for (int byte = 0; byte < 32; ++byte) {
                const uint8_t low = static_cast<uint8_t>((2 * byte + row) & 15);
                const uint8_t high = static_cast<uint8_t>((2 * byte + row + 1) & 15);
                record.qs[byte] = static_cast<uint8_t>(low | (high << 4));
            }
        }
    }
    std::vector<uint16_t> input(batch * cols);
    for (int sample = 0; sample < batch; ++sample) {
        for (int col = 0; col < cols; ++col) {
            input[sample * cols + col] =
                to_half(static_cast<float>((sample + 1) * ((col % 7) - 3)) / 8.0f);
        }
    }
    std::vector<float> expected(batch * rows);
    for (int sample = 0; sample < batch; ++sample) {
        for (int row = 0; row < rows; ++row) {
            float sum = 0.0f;
            for (int col = 0; col < cols; ++col) {
                const auto& record = blocks[
                    row * blocks_per_row + col / 64];
                const int local = col % 64;
                const uint8_t packed = record.qs[local / 2];
                const uint8_t code = local & 1 ? packed >> 4 : packed & 15u;
                sum += from_half(input[sample * cols + col]) *
                       from_e2m1(code) *
                       from_fp8_e4m3(record.d[local / 16]);
            }
            expected[sample * rows + row] = sum * global_factor;
        }
    }

    DeviceBuffer dx, dw, dw_up, dy_half, dy_float, dq8, dq8_scale,
                 dy_dp4a_half, dy_dp4a_float, dy_wmma_half, dy_wmma_float,
                 dy_wide_half, dy_swiglu;
    uint16_t* d_x = dx.allocate<uint16_t>(input.size());
    uint8_t* d_blocks = dw.allocate<uint8_t>(
        blocks.size() * sizeof(pocket::QwenNvfp4Block64));
    uint8_t* d_up_blocks = dw_up.allocate<uint8_t>(
        blocks.size() * sizeof(pocket::QwenNvfp4Block64));
    uint16_t* d_half = dy_half.allocate<uint16_t>(batch * rows);
    float* d_float = dy_float.allocate<float>(batch * rows);
    int8_t* d_q8 = dq8.allocate<int8_t>(batch * cols);
    float* d_q8_scale = dq8_scale.allocate<float>(batch * cols / 32);
    uint16_t* d_dp4a_half = dy_dp4a_half.allocate<uint16_t>(batch * rows);
    float* d_dp4a_float = dy_dp4a_float.allocate<float>(batch * rows);
    uint16_t* d_wmma_half = dy_wmma_half.allocate<uint16_t>(batch * rows);
    float* d_wmma_float = dy_wmma_float.allocate<float>(batch * rows);
    uint16_t* d_wide_half = dy_wide_half.allocate<uint16_t>(batch * rows);
    uint16_t* d_swiglu = dy_swiglu.allocate<uint16_t>(batch * rows);
    if (!d_x || !d_blocks || !d_up_blocks || !d_half || !d_float || !d_q8 ||
        !d_q8_scale || !d_dp4a_half || !d_dp4a_float || !d_wmma_half ||
        !d_wmma_float || !d_wide_half || !d_swiglu ||
        cudaMemcpy(d_x, input.data(), input.size() * sizeof(uint16_t),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_blocks, blocks.data(),
                   blocks.size() * sizeof(pocket::QwenNvfp4Block64),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_up_blocks, blocks.data(),
                   blocks.size() * sizeof(pocket::QwenNvfp4Block64),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        !pocket::qwen_nvfp4_group16_matmul_rows_f16_cuda(
            d_x, d_blocks, d_half, batch, rows, cols, cols, rows,
            blocks_per_row, global_factor) ||
        !pocket::qwen_nvfp4_group16_matmul_rows_f16_f32_cuda(
            d_x, d_blocks, d_float, batch, rows, cols, cols, rows,
            blocks_per_row, global_factor) ||
        !pocket::qwen_nvfp4_quantize_q8_group32_f16_cuda(
            d_x, d_q8, d_q8_scale, batch, cols, cols) ||
        !pocket::qwen_nvfp4_group16_matmul_q8_f16_cuda(
            d_q8, d_q8_scale, d_blocks, d_dp4a_half, batch, rows, cols,
            cols, rows, blocks_per_row, global_factor) ||
        !pocket::qwen_nvfp4_group16_matmul_q8_f32_cuda(
            d_q8, d_q8_scale, d_blocks, d_dp4a_float, batch, rows, cols,
            cols, rows, blocks_per_row, global_factor) ||
        !pocket::qwen_nvfp4_group16_matmul_q8_wmma_f16_cuda(
            d_q8, d_q8_scale, d_blocks, d_wmma_half, batch, rows, cols,
            cols, rows, blocks_per_row, global_factor) ||
        !pocket::qwen_nvfp4_group16_matmul_q8_wmma_f32_cuda(
            d_q8, d_q8_scale, d_blocks, d_wmma_float, batch, rows, cols,
            cols, rows, blocks_per_row, global_factor) ||
        !pocket::qwen_nvfp4_group16_matmul_q8_wide_n64_f16_cuda(
            d_q8, d_q8_scale, d_blocks, d_wide_half, batch, rows, cols,
            cols, rows, blocks_per_row, global_factor) ||
        !pocket::qwen_nvfp4_group16_swiglu_q8_wmma_f16_cuda(
            d_q8, d_q8_scale, d_blocks, global_factor,
            d_up_blocks, global_factor, d_swiglu, batch, rows, cols,
            cols, rows, blocks_per_row) ||
        cudaDeviceSynchronize() != cudaSuccess) {
        std::printf("[FAIL] NVFP4 launch\n");
        return 1;
    }
    std::vector<uint16_t> got_half(batch * rows);
    std::vector<float> got_float(batch * rows);
    std::vector<int8_t> got_q8(batch * cols);
    std::vector<float> got_q8_scale(batch * cols / 32);
    std::vector<uint16_t> got_dp4a_half(batch * rows);
    std::vector<float> got_dp4a_float(batch * rows);
    std::vector<uint16_t> got_wmma_half(batch * rows);
    std::vector<float> got_wmma_float(batch * rows);
    std::vector<uint16_t> got_wide_half(batch * rows);
    std::vector<uint16_t> got_swiglu(batch * rows);
    cudaMemcpy(got_half.data(), d_half, got_half.size() * sizeof(uint16_t),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(got_float.data(), d_float, got_float.size() * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(got_q8.data(), d_q8, got_q8.size() * sizeof(int8_t),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(got_q8_scale.data(), d_q8_scale,
               got_q8_scale.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(got_dp4a_half.data(), d_dp4a_half,
               got_dp4a_half.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(got_dp4a_float.data(), d_dp4a_float,
               got_dp4a_float.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(got_wmma_half.data(), d_wmma_half,
               got_wmma_half.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(got_wmma_float.data(), d_wmma_float,
               got_wmma_float.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(got_wide_half.data(), d_wide_half,
               got_wide_half.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(got_swiglu.data(), d_swiglu,
               got_swiglu.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    std::vector<int8_t> expected_q8(batch * cols);
    std::vector<float> expected_q8_scale(batch * cols / 32);
    for (int sample = 0; sample < batch; ++sample) {
        for (int group = 0; group < cols / 32; ++group) {
            float maximum = 0.0f;
            const int begin = group * 32;
            for (int local = 0; local < 32; ++local) {
                maximum = std::max(maximum, std::fabs(
                    from_half(input[sample * cols + begin + local])));
            }
            const float scale = std::max(maximum, 1.0e-8f) / 127.0f;
            expected_q8_scale[sample * (cols / 32) + group] = scale;
            for (int local = 0; local < 32; ++local) {
                const float value = from_half(
                    input[sample * cols + begin + local]) / scale;
                const int quantized = std::max(-127, std::min(
                    127, static_cast<int>(std::nearbyint(value))));
                expected_q8[sample * cols + begin + local] =
                    static_cast<int8_t>(quantized);
            }
        }
    }
    std::vector<float> expected_q8_output(batch * rows);
    for (int sample = 0; sample < batch; ++sample) {
        for (int row = 0; row < rows; ++row) {
            float sum = 0.0f;
            for (int col = 0; col < cols; ++col) {
                const auto& record = blocks[
                    row * blocks_per_row + col / 64];
                const int local = col % 64;
                const uint8_t packed = record.qs[local / 2];
                const uint8_t code = local & 1 ? packed >> 4 : packed & 15u;
                const float activation = static_cast<float>(
                    expected_q8[sample * cols + col]) *
                    expected_q8_scale[sample * (cols / 32) + col / 32];
                sum += activation * from_e2m1(code) *
                       from_fp8_e4m3(record.d[local / 16]);
            }
            expected_q8_output[sample * rows + row] = sum * global_factor;
        }
    }

    double q8_scale_error = 0.0;
    int q8_code_mismatches = 0;
    for (size_t index = 0; index < expected_q8.size(); ++index) {
        if (got_q8[index] != expected_q8[index]) ++q8_code_mismatches;
    }
    for (size_t index = 0; index < expected_q8_scale.size(); ++index) {
        q8_scale_error = std::max(q8_scale_error, std::fabs(
            static_cast<double>(got_q8_scale[index]) - expected_q8_scale[index]));
    }
    double half_error = 0.0;
    double float_error = 0.0;
    double q8_quantization_error = 0.0;
    double dp4a_half_kernel_error = 0.0;
    double dp4a_float_kernel_error = 0.0;
    double wmma_half_kernel_error = 0.0;
    double wmma_float_kernel_error = 0.0;
    double wide_half_kernel_error = 0.0;
    double swiglu_kernel_error = 0.0;
    for (size_t index = 0; index < expected.size(); ++index) {
        half_error = std::max(half_error, std::fabs(
            static_cast<double>(from_half(got_half[index])) - expected[index]));
        float_error = std::max(float_error, std::fabs(
            static_cast<double>(got_float[index]) - expected[index]));
        q8_quantization_error = std::max(q8_quantization_error, std::fabs(
            static_cast<double>(expected_q8_output[index]) - expected[index]));
        dp4a_half_kernel_error = std::max(dp4a_half_kernel_error, std::fabs(
            static_cast<double>(from_half(got_dp4a_half[index])) -
            expected_q8_output[index]));
        dp4a_float_kernel_error = std::max(dp4a_float_kernel_error, std::fabs(
            static_cast<double>(got_dp4a_float[index]) -
            expected_q8_output[index]));
        wmma_half_kernel_error = std::max(wmma_half_kernel_error, std::fabs(
            static_cast<double>(from_half(got_wmma_half[index])) -
            expected_q8_output[index]));
        wmma_float_kernel_error = std::max(wmma_float_kernel_error, std::fabs(
            static_cast<double>(got_wmma_float[index]) -
            expected_q8_output[index]));
        wide_half_kernel_error = std::max(wide_half_kernel_error, std::fabs(
            static_cast<double>(from_half(got_wide_half[index])) -
            expected_q8_output[index]));
        const float gate = expected_q8_output[index];
        const float expected_swiglu = gate / (1.0f + std::exp(-gate)) * gate;
        swiglu_kernel_error = std::max(swiglu_kernel_error, std::fabs(
            static_cast<double>(from_half(got_swiglu[index])) -
            expected_swiglu));
    }
    if (q8_code_mismatches != 0 || q8_scale_error > 1.0e-7 ||
        half_error > 2.0e-2 || float_error > 2.0e-4 ||
        dp4a_half_kernel_error > 2.0e-2 ||
        dp4a_float_kernel_error > 2.0e-4 ||
        wmma_half_kernel_error > 2.0e-2 ||
        wmma_float_kernel_error > 2.0e-4 ||
        wide_half_kernel_error > 2.0e-2 ||
        swiglu_kernel_error > 5.0e-2) {
        std::printf(
            "[FAIL] NVFP4 numerical half=%.3e float=%.3e "
            "q8_codes=%d q8_scale=%.3e q8_quant=%.3e "
            "dp4a_half_kernel=%.3e dp4a_float_kernel=%.3e "
            "wmma_half_kernel=%.3e wmma_float_kernel=%.3e "
            "wide_half_kernel=%.3e swiglu_kernel=%.3e\n",
            half_error, float_error, q8_code_mismatches, q8_scale_error,
            q8_quantization_error, dp4a_half_kernel_error,
            dp4a_float_kernel_error, wmma_half_kernel_error,
            wmma_float_kernel_error, wide_half_kernel_error,
            swiglu_kernel_error);
        return 1;
    }
    std::printf(
        "[PASS] test_qwen_nvfp4 half=%.3e float=%.3e "
        "q8_codes=%d q8_scale=%.3e q8_quant=%.3e "
        "dp4a_half_kernel=%.3e dp4a_float_kernel=%.3e "
        "wmma_half_kernel=%.3e wmma_float_kernel=%.3e "
        "wide_half_kernel=%.3e swiglu_kernel=%.3e\n",
        half_error, float_error, q8_code_mismatches, q8_scale_error,
        q8_quantization_error, dp4a_half_kernel_error,
        dp4a_float_kernel_error, wmma_half_kernel_error,
        wmma_float_kernel_error, wide_half_kernel_error,
        swiglu_kernel_error);
    return 0;
}
