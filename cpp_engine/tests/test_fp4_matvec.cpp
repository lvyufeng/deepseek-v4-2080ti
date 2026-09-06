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
    int rows = 7;
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
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
    }
}

std::vector<float> make_input(int cols) {
    std::vector<float> x(cols);
    for (int i = 0; i < cols; ++i) {
        x[i] = static_cast<float>((i % 19) - 9) * 0.0625f;
    }
    return x;
}

float fp4_value(uint8_t code) {
    static constexpr float table[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
    };
    return table[code & 0x0f];
}

float e8m0_value(uint8_t code) {
    return std::exp2(static_cast<float>(static_cast<int>(code) - 127));
}

std::vector<float> fp4_cpu_ref(const std::vector<float>& x, const uint8_t* w, const uint8_t* scale, int rows, int cols) {
    const int packed_cols = cols / 2;
    const int scale_cols = cols / 32;
    std::vector<float> ref(rows, 0.0f);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            const uint8_t packed = w[static_cast<size_t>(r) * packed_cols + c / 2];
            const uint8_t code = (c & 1) ? static_cast<uint8_t>(packed >> 4) : static_cast<uint8_t>(packed & 0x0f);
            const float s = e8m0_value(scale[static_cast<size_t>(r) * scale_cols + c / 32]);
            ref[r] += fp4_value(code) * s * x[c];
        }
    }
    return ref;
}

std::vector<uint8_t> make_synthetic_weight(int rows, int cols) {
    const int packed_cols = cols / 2;
    std::vector<uint8_t> weight(static_cast<size_t>(rows) * packed_cols);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; c += 2) {
            const uint8_t lo = static_cast<uint8_t>((r * 3 + c) & 0x0f);
            const uint8_t hi = static_cast<uint8_t>((r * 5 + c + 1) & 0x0f);
            weight[static_cast<size_t>(r) * packed_cols + c / 2] = static_cast<uint8_t>(lo | (hi << 4));
        }
    }
    return weight;
}

std::vector<uint8_t> make_synthetic_scale(int rows, int cols) {
    const int scale_cols = cols / 32;
    std::vector<uint8_t> scale(static_cast<size_t>(rows) * scale_cols);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < scale_cols; ++c) {
            scale[static_cast<size_t>(r) * scale_cols + c] = static_cast<uint8_t>(124 + ((r + c) % 7));
        }
    }
    return scale;
}

void run_case(const std::vector<float>& x, const uint8_t* w, const uint8_t* scale, int rows, int cols, const std::string& label) {
    const size_t weight_bytes = static_cast<size_t>(rows) * (cols / 2);
    const size_t scale_bytes = static_cast<size_t>(rows) * (cols / 32);
    auto ref = fp4_cpu_ref(x, w, scale, rows, cols);

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
    if (!pocket::fp4_e2m1_e8m0_matvec_cuda(d_x, d_w, d_scale, d_y, rows, cols)) {
        throw std::runtime_error("fp4_e2m1_e8m0_matvec_cuda launch failed");
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
        if (!args.ckpt.empty()) {
            if (args.tensor.empty()) {
                throw std::runtime_error("--tensor is required with --ckpt");
            }
            pocket::SafeTensorsIndex index(args.ckpt);
            const std::string* weight_shard_name = index.shard_for_tensor(args.tensor);
            if (weight_shard_name == nullptr) throw std::runtime_error("tensor not found: " + args.tensor);
            pocket::SafeTensorsShard shard(index.shard_path(*weight_shard_name));
            const pocket::SafeFp4TensorPair pair = pocket::resolve_fp4_tensor_pair(index, shard, args.tensor);
            const auto* weight_info = shard.find_tensor(pair.weight_name);
            const auto* scale_info = shard.find_tensor(pair.scale_name);
            const int cols = static_cast<int>(pair.cols);
            const int rows = std::min(args.rows, static_cast<int>(pair.rows));
            auto x = make_input(cols);
            run_case(x, shard.tensor_data(*weight_info), shard.tensor_data(*scale_info), rows, cols, pair.weight_name);
            return 0;
        }

        constexpr int rows = 7;
        constexpr int cols = 96;
        auto x = make_input(cols);
        auto w = make_synthetic_weight(rows, cols);
        auto scale = make_synthetic_scale(rows, cols);
        run_case(x, w.data(), scale.data(), rows, cols, "synthetic_fp4_e2m1_e8m0");
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "[FAIL] " << ex.what() << "\n";
        return 1;
    }
}
