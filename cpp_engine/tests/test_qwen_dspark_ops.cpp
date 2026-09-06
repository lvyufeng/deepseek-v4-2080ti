#include "qwen_ops.hpp"
#include "qwen_weights.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void check_cuda(cudaError_t status, const std::string& message) {
    if (status != cudaSuccess) {
        throw std::runtime_error(message + ": " + cudaGetErrorString(status));
    }
}

uint16_t half_bits(float value) {
    const uint32_t bits = *reinterpret_cast<const uint32_t*>(&value);
    const uint16_t bf16 = static_cast<uint16_t>(bits >> 16);
    return pocket::qwen_bf16_to_fp16_bits(bf16);
}

float half_value(uint16_t value) {
    uint16_t exponent = static_cast<uint16_t>((value >> 10) & 0x1f);
    uint16_t mantissa = static_cast<uint16_t>(value & 0x03ff);
    const float sign = (value & 0x8000) ? -1.0f : 1.0f;
    if (exponent == 0) {
        return sign * std::ldexp(static_cast<float>(mantissa), -24);
    }
    if (exponent == 31) return mantissa ? NAN : sign * INFINITY;
    return sign * std::ldexp(1.0f + static_cast<float>(mantissa) / 1024.0f,
                             static_cast<int>(exponent) - 15);
}

template <typename T>
T* upload(const std::vector<T>& values) {
    T* device = nullptr;
    check_cuda(cudaMalloc(&device, values.size() * sizeof(T)), "cudaMalloc");
    check_cuda(cudaMemcpy(device, values.data(), values.size() * sizeof(T),
                          cudaMemcpyHostToDevice), "cudaMemcpy H2D");
    return device;
}

template <typename T>
std::vector<T> download(const T* device, size_t count) {
    std::vector<T> output(count);
    check_cuda(cudaMemcpy(output.data(), device, count * sizeof(T),
                          cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
    return output;
}

void test_gemm() {
    const int batch = 3;
    const int rows = 5;
    const int cols = 7;
    std::vector<float> x_values(static_cast<size_t>(batch) * cols);
    std::vector<float> weight_values(static_cast<size_t>(rows) * cols);
    for (size_t index = 0; index < x_values.size(); ++index) {
        x_values[index] = static_cast<float>(static_cast<int>(index % 9) - 4) / 8.0f;
    }
    for (size_t index = 0; index < weight_values.size(); ++index) {
        weight_values[index] = static_cast<float>(static_cast<int>(index % 11) - 5) / 7.0f;
    }
    std::vector<uint16_t> x;
    std::vector<uint16_t> weight;
    for (float value : x_values) x.push_back(half_bits(value));
    for (float value : weight_values) weight.push_back(half_bits(value));
    uint16_t* d_x = upload(x);
    uint16_t* d_weight = upload(weight);
    uint16_t* d_output = nullptr;
    check_cuda(cudaMalloc(&d_output,
                          static_cast<size_t>(batch) * rows * sizeof(uint16_t)),
               "GEMM output");
    require(pocket::qwen_dspark_fp16_gemm_rows_f16_cuda(
                d_x, d_weight, d_output, batch, rows, cols),
            "DSpark GEMM launch failed");
    const auto output = download(d_output, static_cast<size_t>(batch) * rows);
    for (int sample = 0; sample < batch; ++sample) {
        for (int row = 0; row < rows; ++row) {
            float expected = 0.0f;
            for (int col = 0; col < cols; ++col) {
                expected += x_values[static_cast<size_t>(sample) * cols + col] *
                    weight_values[static_cast<size_t>(row) * cols + col];
            }
            const float actual = half_value(
                output[static_cast<size_t>(sample) * rows + row]);
            if (std::abs(actual - expected) >= 0.006f) {
                throw std::runtime_error(
                    "DSpark GEMM mismatch sample=" + std::to_string(sample) +
                    " row=" + std::to_string(row) +
                    " expected=" + std::to_string(expected) +
                    " actual=" + std::to_string(actual));
            }
        }
    }
    cudaFree(d_x);
    cudaFree(d_weight);
    cudaFree(d_output);
}

void test_rmsnorm() {
    const int rows = 2;
    const int cols = 4;
    const std::vector<float> x_values = {1, 2, 3, 4, -2, 1, 0.5f, -1};
    const std::vector<float> gamma_values = {0.5f, 1.0f, 1.5f, 2.0f};
    std::vector<uint16_t> x;
    std::vector<uint16_t> gamma;
    for (float value : x_values) x.push_back(half_bits(value));
    for (float value : gamma_values) gamma.push_back(half_bits(value));
    uint16_t* d_x = upload(x);
    uint16_t* d_gamma = upload(gamma);
    uint16_t* d_y = nullptr;
    check_cuda(cudaMalloc(&d_y, x.size() * sizeof(uint16_t)), "rmsnorm output");
    require(pocket::qwen_dspark_rmsnorm_f16_cuda(
                d_x, d_gamma, d_y, rows, cols, 1.0e-6f),
            "RMSNorm launch failed");
    const auto y = download(d_y, x.size());
    for (int row = 0; row < rows; ++row) {
        float mean_square = 0.0f;
        for (int col = 0; col < cols; ++col) {
            const float value = x_values[row * cols + col];
            mean_square += value * value;
        }
        const float inverse = 1.0f / std::sqrt(mean_square / cols + 1.0e-6f);
        for (int col = 0; col < cols; ++col) {
            const float expected = x_values[row * cols + col] * inverse *
                gamma_values[col];
            require(std::abs(half_value(y[row * cols + col]) - expected) < 0.004f,
                    "standard RMSNorm mismatch");
        }
    }
    cudaFree(d_x);
    cudaFree(d_gamma);
    cudaFree(d_y);
}

void test_rope() {
    const int rows = 2;
    const int heads = 1;
    const int dim = 4;
    std::vector<uint16_t> x = {
        half_bits(1), half_bits(2), half_bits(3), half_bits(4),
        half_bits(2), half_bits(-1), half_bits(0.5f), half_bits(3)};
    const std::vector<float> frequencies = {1.0f, 0.25f};
    uint16_t* d_x = upload(x);
    float* d_frequencies = upload(frequencies);
    require(pocket::qwen_dspark_yarn_rope_f16_cuda(
                d_x, d_frequencies, rows, heads, dim, 3, 1.25f),
            "YaRN RoPE launch failed");
    const auto output = download(d_x, x.size());
    for (int row = 0; row < rows; ++row) {
        for (int pair = 0; pair < dim / 2; ++pair) {
            const float angle = (3 + row) * frequencies[pair];
            const float cosine = std::cos(angle) * 1.25f;
            const float sine = std::sin(angle) * 1.25f;
            const float left = half_value(x[row * dim + pair]);
            const float right = half_value(x[row * dim + dim / 2 + pair]);
            const float expected_left = left * cosine - right * sine;
            const float expected_right = right * cosine + left * sine;
            require(std::abs(half_value(output[row * dim + pair]) -
                             expected_left) < 0.005f,
                    "YaRN left mismatch");
            require(std::abs(half_value(output[row * dim + dim / 2 + pair]) -
                             expected_right) < 0.005f,
                    "YaRN right mismatch");
        }
    }
    cudaFree(d_x);
    cudaFree(d_frequencies);
}

void test_attention() {
    const int block_rows = 2;
    const int q_heads = 2;
    const int kv_heads = 1;
    const int dim = 4;
    const int context = 2;
    const std::vector<float> q_values = {
        1, 0, 0, 0, 0, 1, 0, 0,
        0, 0, 1, 0, 0, 0, 0, 1};
    const std::vector<float> context_k_values = {1, 0, 0, 0, 0, 1, 0, 0};
    const std::vector<float> context_v_values = {1, 2, 3, 4, 2, 0, 1, -1};
    const std::vector<float> block_k_values = {0, 0, 1, 0, 0, 0, 0, 1};
    const std::vector<float> block_v_values = {-1, 1, 2, 0, 3, -2, 0, 1};
    auto convert = [](const std::vector<float>& values) {
        std::vector<uint16_t> output;
        for (float value : values) output.push_back(half_bits(value));
        return output;
    };
    const auto q = convert(q_values);
    const auto context_k = convert(context_k_values);
    const auto context_v = convert(context_v_values);
    const auto block_k = convert(block_k_values);
    const auto block_v = convert(block_v_values);
    uint16_t* d_q = upload(q);
    uint16_t* d_context_k = upload(context_k);
    uint16_t* d_context_v = upload(context_v);
    uint16_t* d_block_k = upload(block_k);
    uint16_t* d_block_v = upload(block_v);
    uint16_t* d_output = nullptr;
    check_cuda(cudaMalloc(&d_output, q.size() * sizeof(uint16_t)),
               "attention output");
    require(pocket::qwen_dspark_dual_source_gqa_f16_cuda(
                d_q, d_context_k, d_context_v, d_block_k, d_block_v,
                d_output, block_rows, q_heads, kv_heads, dim, context, 8),
            "dual-source attention launch failed");
    const auto output = download(d_output, q.size());
    const float scale = 1.0f / std::sqrt(static_cast<float>(dim));
    const std::vector<std::vector<float>> keys = {
        {1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}, {0, 0, 0, 1}};
    const std::vector<std::vector<float>> values = {
        {1, 2, 3, 4}, {2, 0, 1, -1}, {-1, 1, 2, 0}, {3, -2, 0, 1}};
    for (int row = 0; row < block_rows; ++row) {
        for (int head = 0; head < q_heads; ++head) {
            const int q_base = (row * q_heads + head) * dim;
            std::vector<float> scores(4);
            float maximum = -INFINITY;
            for (int key = 0; key < 4; ++key) {
                float dot = 0.0f;
                for (int col = 0; col < dim; ++col) {
                    dot += q_values[q_base + col] * keys[key][col];
                }
                scores[key] = dot * scale;
                maximum = std::max(maximum, scores[key]);
            }
            float denominator = 0.0f;
            for (float& score : scores) {
                score = std::exp(score - maximum);
                denominator += score;
            }
            for (int col = 0; col < dim; ++col) {
                float expected = 0.0f;
                for (int key = 0; key < 4; ++key) {
                    expected += scores[key] / denominator * values[key][col];
                }
                require(std::abs(half_value(output[q_base + col]) - expected) <
                                0.004f,
                        "dual-source attention mismatch");
            }
        }
    }
    cudaFree(d_q);
    cudaFree(d_context_k);
    cudaFree(d_context_v);
    cudaFree(d_block_k);
    cudaFree(d_block_v);
    cudaFree(d_output);
}

void test_heads() {
    const int local_vocab = 3;
    const int rank = 2;
    std::vector<float> logits = {0.25f, -1.0f, 2.0f};
    std::vector<uint16_t> embedding = {half_bits(2), half_bits(-1)};
    std::vector<uint16_t> weight = {
        half_bits(1), half_bits(0.5f),
        half_bits(-2), half_bits(1),
        half_bits(0.25f), half_bits(-0.75f)};
    float* d_logits = upload(logits);
    uint16_t* d_embedding = upload(embedding);
    uint16_t* d_weight = upload(weight);
    require(pocket::qwen_dspark_add_markov_bias_f32_cuda(
                d_logits, d_embedding, d_weight, local_vocab, rank),
            "Markov launch failed");
    const auto corrected = download(d_logits, logits.size());
    const std::vector<float> expected = {1.75f, -6.0f, 3.25f};
    for (size_t index = 0; index < expected.size(); ++index) {
        require(std::abs(corrected[index] - expected[index]) < 0.003f,
                "Markov bias mismatch");
    }

    const int rows = 2;
    const int hidden_size = 2;
    const std::vector<uint16_t> hidden = {
        half_bits(1), half_bits(2), half_bits(-1), half_bits(0.5f)};
    const std::vector<uint16_t> markov = {
        half_bits(0.25f), half_bits(-2), half_bits(1), half_bits(3)};
    const std::vector<uint16_t> confidence_weight = {
        half_bits(0.5f), half_bits(-1), half_bits(2), half_bits(0.25f)};
    const std::vector<uint16_t> bias = {half_bits(-0.5f)};
    uint16_t* d_hidden = upload(hidden);
    uint16_t* d_markov = upload(markov);
    uint16_t* d_confidence_weight = upload(confidence_weight);
    uint16_t* d_bias = upload(bias);
    float* d_confidence = nullptr;
    check_cuda(cudaMalloc(&d_confidence, rows * sizeof(float)),
               "confidence output");
    require(pocket::qwen_dspark_confidence_f16_cuda(
                d_hidden, d_markov, d_confidence_weight, d_bias,
                d_confidence, rows, hidden_size, rank),
            "confidence launch failed");
    const auto confidence = download(d_confidence, rows);
    const std::vector<float> raw = {-2.0f, 1.25f};
    for (int row = 0; row < rows; ++row) {
        const float expected_confidence = 1.0f / (1.0f + std::exp(-raw[row]));
        require(std::abs(confidence[row] - expected_confidence) < 0.003f,
                "confidence mismatch");
    }
    cudaFree(d_logits);
    cudaFree(d_embedding);
    cudaFree(d_weight);
    cudaFree(d_hidden);
    cudaFree(d_markov);
    cudaFree(d_confidence_weight);
    cudaFree(d_bias);
    cudaFree(d_confidence);
}

}  // namespace

int main() {
    try {
        test_gemm();
        test_rmsnorm();
        test_rope();
        test_attention();
        test_heads();
        check_cuda(cudaDeviceSynchronize(), "DSpark op synchronization");
        std::cout << "[PASS] qwen_dspark_ops\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "[FAIL] " << error.what() << "\n";
        return 1;
    }
}
