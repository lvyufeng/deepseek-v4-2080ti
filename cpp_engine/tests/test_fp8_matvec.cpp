#include "cuda_ops.hpp"
#include "safetensors_reader.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Args {
    std::string ckpt;
    std::string tensor;
    int rows = 128;
};

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--ckpt" && i + 1 < argc) {
            args.ckpt = argv[++i];
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

void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
}

std::vector<float> make_input(int cols) {
    std::vector<float> x(cols);
    for (int i = 0; i < cols; ++i) x[i] = static_cast<float>((i % 23) - 11) * 0.03125f;
    return x;
}

std::vector<float> make_input_rows(int batch, int cols) {
    std::vector<float> x(static_cast<size_t>(batch) * cols);
    for (int b = 0; b < batch; ++b) {
        for (int i = 0; i < cols; ++i) {
            x[static_cast<size_t>(b) * cols + i] = static_cast<float>(((b * 7 + i) % 29) - 14) * 0.015625f;
        }
    }
    return x;
}

float fp8_e4m3_value(uint8_t code) {
    const int sign = (code >> 7) & 1;
    const int exp = (code >> 3) & 0xf;
    const int mant = code & 0x7;
    float value;
    if (exp == 0) value = std::ldexp(static_cast<float>(mant) * 0.125f, -6);
    else value = std::ldexp(1.0f + static_cast<float>(mant) * 0.125f, exp - 7);
    return sign ? -value : value;
}

float e8m0_value(uint8_t code) {
    return std::exp2(static_cast<float>(static_cast<int>(code) - 127));
}

std::vector<float> fp8_cpu_ref(const std::vector<float>& x, const uint8_t* w, const uint8_t* scale, int rows, int cols) {
    constexpr int block = 128;
    const int scale_cols = cols / block;
    std::vector<float> ref(rows, 0.0f);
    for (int r = 0; r < rows; ++r) {
        const int rb = r / block;
        for (int c = 0; c < cols; ++c) {
            const int cb = c / block;
            ref[r] += fp8_e4m3_value(w[static_cast<size_t>(r) * cols + c]) * e8m0_value(scale[static_cast<size_t>(rb) * scale_cols + cb]) * x[c];
        }
    }
    return ref;
}

std::vector<float> fp8_cpu_ref_rows(const std::vector<float>& x, const uint8_t* w, const uint8_t* scale, int batch, int rows, int cols) {
    constexpr int block = 128;
    const int scale_cols = cols / block;
    std::vector<float> ref(static_cast<size_t>(batch) * rows, 0.0f);
    for (int b = 0; b < batch; ++b) {
        for (int r = 0; r < rows; ++r) {
            const int rb = r / block;
            float sum = 0.0f;
            for (int c = 0; c < cols; ++c) {
                const int cb = c / block;
                sum += fp8_e4m3_value(w[static_cast<size_t>(r) * cols + c]) * e8m0_value(scale[static_cast<size_t>(rb) * scale_cols + cb]) * x[static_cast<size_t>(b) * cols + c];
            }
            ref[static_cast<size_t>(b) * rows + r] = sum;
        }
    }
    return ref;
}

void run_matmul_case(const std::vector<float>& x, const uint8_t* w, const uint8_t* scale, int batch, int rows, int cols, const std::string& label) {
    const size_t weight_bytes = static_cast<size_t>(rows) * cols;
    const size_t scale_bytes = static_cast<size_t>((rows + 127) / 128) * (cols / 128);
    auto ref = fp8_cpu_ref_rows(x, w, scale, batch, rows, cols);

    float* d_x = nullptr;
    uint8_t* d_w = nullptr;
    uint8_t* d_scale = nullptr;
    float* d_y = nullptr;
    check_cuda(cudaMalloc(&d_x, x.size() * sizeof(float)), "cudaMalloc matmul x");
    check_cuda(cudaMalloc(&d_w, weight_bytes), "cudaMalloc matmul weight");
    check_cuda(cudaMalloc(&d_scale, scale_bytes), "cudaMalloc matmul scale");
    check_cuda(cudaMalloc(&d_y, ref.size() * sizeof(float)), "cudaMalloc matmul y");
    check_cuda(cudaMemcpy(d_x, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice), "copy matmul x");
    check_cuda(cudaMemcpy(d_w, w, weight_bytes, cudaMemcpyHostToDevice), "copy matmul weight");
    check_cuda(cudaMemcpy(d_scale, scale, scale_bytes, cudaMemcpyHostToDevice), "copy matmul scale");
    if (!pocket::fp8_e4m3_e8m0_matmul_cuda(d_x, d_w, d_scale, d_y, batch, rows, cols)) {
        throw std::runtime_error("fp8_e4m3_e8m0_matmul_cuda launch failed");
    }
    check_cuda(cudaDeviceSynchronize(), "sync matmul kernel");
    std::vector<float> got(ref.size());
    check_cuda(cudaMemcpy(got.data(), d_y, got.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy matmul y");
    cudaFree(d_x);
    cudaFree(d_w);
    cudaFree(d_scale);
    cudaFree(d_y);

    float max_abs = 0.0f;
    float mean_abs = 0.0f;
    for (size_t i = 0; i < ref.size(); ++i) {
        const float diff = std::fabs(got[i] - ref[i]);
        max_abs = std::max(max_abs, diff);
        mean_abs += diff;
    }
    mean_abs /= std::max<size_t>(ref.size(), 1);
    if (max_abs > 5e-2f) {
        std::cerr << "[FAIL] " << label << " matmul max_abs=" << max_abs << " mean_abs=" << mean_abs << "\n";
        std::exit(1);
    }
    std::cout << "[PASS] " << label << " matmul max_abs=" << max_abs << " mean_abs=" << mean_abs << " batch=" << batch << " rows=" << rows << " cols=" << cols << "\n";
}

void run_case(const std::vector<float>& x, const uint8_t* w, const uint8_t* scale, int rows, int cols, const std::string& label) {
    const size_t weight_bytes = static_cast<size_t>(rows) * cols;
    const size_t scale_bytes = static_cast<size_t>((rows + 127) / 128) * (cols / 128);
    auto ref = fp8_cpu_ref(x, w, scale, rows, cols);

    float* d_x = nullptr;
    uint8_t* d_w = nullptr;
    uint8_t* d_scale = nullptr;
    float* d_y = nullptr;
    check_cuda(cudaMalloc(&d_x, x.size() * sizeof(float)), "cudaMalloc x");
    check_cuda(cudaMalloc(&d_w, weight_bytes), "cudaMalloc weight");
    check_cuda(cudaMalloc(&d_scale, scale_bytes), "cudaMalloc scale");
    check_cuda(cudaMalloc(&d_y, ref.size() * sizeof(float)), "cudaMalloc y");
    check_cuda(cudaMemcpy(d_x, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice), "copy x");
    check_cuda(cudaMemcpy(d_w, w, weight_bytes, cudaMemcpyHostToDevice), "copy weight");
    check_cuda(cudaMemcpy(d_scale, scale, scale_bytes, cudaMemcpyHostToDevice), "copy scale");
    if (!pocket::fp8_e4m3_e8m0_matvec_cuda(d_x, d_w, d_scale, d_y, rows, cols)) {
        throw std::runtime_error("fp8_e4m3_e8m0_matvec_cuda launch failed");
    }
    check_cuda(cudaDeviceSynchronize(), "sync kernel");
    std::vector<float> got(rows);
    check_cuda(cudaMemcpy(got.data(), d_y, got.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy y");
    cudaFree(d_x);
    cudaFree(d_w);
    cudaFree(d_scale);
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

std::string scale_name_for(const std::string& tensor) {
    const std::string suffix = ".weight";
    if (tensor.size() >= suffix.size() && tensor.compare(tensor.size() - suffix.size(), suffix.size(), suffix) == 0) {
        return tensor.substr(0, tensor.size() - suffix.size()) + ".scale";
    }
    return tensor + ".scale";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (!pocket::cuda_runtime_available()) {
            std::cout << "[SKIP] CUDA runtime is not available\n";
            return 0;
        }
        Args args = parse_args(argc, argv);
        if (args.ckpt.empty() || args.tensor.empty()) throw std::runtime_error("--ckpt and --tensor are required");
        pocket::SafeTensorsIndex index(args.ckpt);
        const std::string* shard_name = index.shard_for_tensor(args.tensor);
        if (shard_name == nullptr) throw std::runtime_error("tensor not found: " + args.tensor);
        const std::string scale_name = scale_name_for(args.tensor);
        const std::string* scale_shard = index.shard_for_tensor(scale_name);
        if (scale_shard == nullptr) throw std::runtime_error("scale tensor not found: " + scale_name);
        if (*shard_name != *scale_shard) throw std::runtime_error("weight and scale are in different shards");
        pocket::SafeTensorsShard shard(index.shard_path(*shard_name));
        const auto* weight = shard.find_tensor(args.tensor);
        const auto* scale = shard.find_tensor(scale_name);
        if (weight == nullptr || scale == nullptr) throw std::runtime_error("tensor missing from shard header");
        if (weight->dtype != pocket::SafeDType::F8_E4M3 || weight->shape.size() != 2) throw std::runtime_error("expected 2D F8_E4M3 weight");
        if (scale->dtype != pocket::SafeDType::F8_E8M0 || scale->shape.size() != 2) throw std::runtime_error("expected 2D F8_E8M0 scale");
        const int rows = std::min(args.rows, static_cast<int>(weight->shape[0]));
        const int cols = static_cast<int>(weight->shape[1]);
        if (rows % 128 != 0 || cols % 128 != 0) throw std::runtime_error("rows and cols must be multiples of 128");
        if (scale->shape[0] < static_cast<uint64_t>(rows / 128) || scale->shape[1] != static_cast<uint64_t>(cols / 128)) {
            throw std::runtime_error("weight/scale shape mismatch");
        }
        auto x = make_input(cols);
        run_case(x, shard.tensor_data(*weight), shard.tensor_data(*scale), rows, cols, args.tensor);
        if (static_cast<int>(weight->shape[0]) >= 128) {
            auto x_rows = make_input_rows(16, cols);
            run_matmul_case(x_rows, shard.tensor_data(*weight), shard.tensor_data(*scale), 16, rows, cols, args.tensor);
        }
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "[FAIL] " << ex.what() << "\n";
        return 1;
    }
}
