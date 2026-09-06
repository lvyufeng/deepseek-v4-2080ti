#include "cuda_ops.hpp"
#include "gguf_reader.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Args {
    std::string model;
    std::string tensor;
    int rows = 7;
};

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            args.model = argv[++i];
        } else if (arg == "--tensor" && i + 1 < argc) {
            args.tensor = argv[++i];
        } else if (arg == "--rows" && i + 1 < argc) {
            args.rows = std::stoi(argv[++i]);
        } else {
            throw std::runtime_error("unknown or incomplete argument: " + arg);
        }
    }
    return args;
}

uint16_t f32_to_f16_bits(float value) {
    uint32_t x;
    std::memcpy(&x, &value, sizeof(x));
    uint32_t sign = (x >> 16) & 0x8000;
    int exp = static_cast<int>((x >> 23) & 0xff) - 127 + 15;
    uint32_t mant = x & 0x7fffff;
    if (exp <= 0) return static_cast<uint16_t>(sign);
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7c00);
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (mant >> 13));
}

float f16_bits_to_f32(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x03ff;
    uint32_t out;
    if (exp == 0) {
        out = sign;
    } else if (exp == 31) {
        out = sign | 0x7f800000 | (mant << 13);
    } else {
        out = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float value;
    std::memcpy(&value, &out, sizeof(value));
    return value;
}

void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
    }
}

std::vector<float> make_input(int cols) {
    std::vector<float> x(cols);
    for (int i = 0; i < cols; ++i) {
        x[i] = static_cast<float>((i % 17) - 8) * 0.125f;
    }
    return x;
}

std::vector<float> q8_0_cpu_ref(const std::vector<float>& x, const uint8_t* w, int rows, int cols) {
    const int blocks_per_row = (cols + 31) / 32;
    std::vector<float> ref(rows, 0.0f);
    for (int r = 0; r < rows; ++r) {
        for (int b = 0; b < blocks_per_row; ++b) {
            const uint8_t* block = w + (static_cast<size_t>(r) * blocks_per_row + b) * 34;
            const uint16_t scale_h = static_cast<uint16_t>(block[0]) | (static_cast<uint16_t>(block[1]) << 8);
            const float scale = f16_bits_to_f32(scale_h);
            const auto* qs = reinterpret_cast<const int8_t*>(block + 2);
            for (int j = 0; j < 32; ++j) {
                const int col = b * 32 + j;
                if (col < cols) {
                    ref[r] += scale * static_cast<float>(qs[j]) * x[col];
                }
            }
        }
    }
    return ref;
}

std::vector<uint8_t> make_synthetic_q8(int rows, int cols) {
    const int blocks_per_row = (cols + 31) / 32;
    std::vector<uint8_t> w(static_cast<size_t>(rows) * blocks_per_row * 34);
    for (int r = 0; r < rows; ++r) {
        for (int b = 0; b < blocks_per_row; ++b) {
            uint8_t* block = w.data() + (static_cast<size_t>(r) * blocks_per_row + b) * 34;
            const float scale = 0.03125f * static_cast<float>(r + 1);
            const uint16_t scale_h = f32_to_f16_bits(scale);
            block[0] = static_cast<uint8_t>(scale_h & 0xff);
            block[1] = static_cast<uint8_t>(scale_h >> 8);
            auto* qs = reinterpret_cast<int8_t*>(block + 2);
            for (int j = 0; j < 32; ++j) {
                const int col = b * 32 + j;
                qs[j] = static_cast<int8_t>(((r * 13 + col * 7) % 31) - 15);
            }
        }
    }
    return w;
}

void run_case(const std::vector<float>& x, const uint8_t* w, size_t w_bytes, const std::vector<float>& ref, int rows, int cols, const std::string& label) {
    float* d_x = nullptr;
    uint8_t* d_w = nullptr;
    float* d_y = nullptr;
    check_cuda(cudaMalloc(&d_x, x.size() * sizeof(float)), "cudaMalloc x");
    check_cuda(cudaMalloc(&d_w, w_bytes), "cudaMalloc w");
    check_cuda(cudaMalloc(&d_y, ref.size() * sizeof(float)), "cudaMalloc y");
    check_cuda(cudaMemcpy(d_x, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice), "copy x");
    check_cuda(cudaMemcpy(d_w, w, w_bytes, cudaMemcpyHostToDevice), "copy w");
    if (!pocket::q8_0_matvec_cuda(d_x, d_w, d_y, rows, cols)) {
        throw std::runtime_error("q8_0_matvec_cuda launch failed");
    }
    check_cuda(cudaDeviceSynchronize(), "sync kernel");
    std::vector<float> got(rows);
    check_cuda(cudaMemcpy(got.data(), d_y, got.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy y");
    cudaFree(d_x);
    cudaFree(d_w);
    cudaFree(d_y);

    float max_abs = 0.0f;
    float mean_abs = 0.0f;
    for (int i = 0; i < rows; ++i) {
        const float diff = std::fabs(got[i] - ref[i]);
        max_abs = std::max(max_abs, diff);
        mean_abs += diff;
    }
    mean_abs /= std::max(rows, 1);
    if (max_abs > 1e-3f) {
        std::cerr << "[FAIL] " << label << " max_abs=" << max_abs << " mean_abs=" << mean_abs << "\n";
        std::exit(1);
    }
    std::cout << "[PASS] " << label << " max_abs=" << max_abs << " mean_abs=" << mean_abs << " rows=" << rows << " cols=" << cols << "\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (!pocket::cuda_runtime_available()) {
            std::cout << "[SKIP] CUDA runtime is not available\n";
            return 0;
        }
        Args args = parse_args(argc, argv);
        if (!args.model.empty()) {
            if (args.tensor.empty()) {
                throw std::runtime_error("--tensor is required with --model");
            }
            pocket::GGUFFile file(args.model);
            const auto* info = file.find_tensor(args.tensor);
            if (info == nullptr) {
                throw std::runtime_error("tensor not found: " + args.tensor);
            }
            if (info->dtype != pocket::DType::Q8_0 || info->shape.size() != 2) {
                throw std::runtime_error("expected a 2D q8_0 tensor");
            }
            const int cols = static_cast<int>(info->shape[0]);
            const int tensor_rows = static_cast<int>(info->shape[1]);
            const int rows = std::min(args.rows, tensor_rows);
            const int blocks_per_row = (cols + 31) / 32;
            const size_t bytes = static_cast<size_t>(rows) * blocks_per_row * 34;
            auto x = make_input(cols);
            auto view = file.tensor_view(*info);
            auto ref = q8_0_cpu_ref(x, view.data, rows, cols);
            run_case(x, view.data, bytes, ref, rows, cols, args.tensor);
            return 0;
        }

        constexpr int rows = 7;
        constexpr int cols = 96;
        auto x = make_input(cols);
        auto w = make_synthetic_q8(rows, cols);
        auto ref = q8_0_cpu_ref(x, w.data(), rows, cols);
        run_case(x, w.data(), w.size(), ref, rows, cols, "synthetic_q8_0");
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "[FAIL] " << ex.what() << "\n";
        return 1;
    }
}
