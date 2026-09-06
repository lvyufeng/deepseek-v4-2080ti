#include "cuda_ops.hpp"
#include "safetensors_reader.hpp"

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
    std::string ckpt;
    std::string tensor = "layers.0.attn_norm.weight";
};

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--ckpt" && i + 1 < argc) args.ckpt = argv[++i];
        else if (arg == "--tensor" && i + 1 < argc) args.tensor = argv[++i];
        else throw std::runtime_error("unknown or incomplete argument: " + arg);
    }
    return args;
}

void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
}

float bf16_to_float(uint16_t bits) {
    uint32_t value = static_cast<uint32_t>(bits) << 16;
    float out;
    std::memcpy(&out, &value, sizeof(out));
    return out;
}

std::vector<float> make_input(int cols) {
    std::vector<float> x(cols);
    for (int i = 0; i < cols; ++i) x[i] = static_cast<float>((i % 29) - 14) * 0.015625f;
    return x;
}

std::vector<float> rmsnorm_cpu_ref(const std::vector<float>& x, const uint16_t* gamma, float eps) {
    double sum_sq = 0.0;
    for (float v : x) sum_sq += static_cast<double>(v) * v;
    const float inv = 1.0f / std::sqrt(static_cast<float>(sum_sq / x.size()) + eps);
    std::vector<float> ref(x.size());
    for (size_t i = 0; i < x.size(); ++i) ref[i] = x[i] * inv * bf16_to_float(gamma[i]);
    return ref;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (!pocket::cuda_runtime_available()) {
            std::cout << "[SKIP] CUDA runtime is not available\n";
            return 0;
        }
        Args args = parse_args(argc, argv);
        if (args.ckpt.empty()) throw std::runtime_error("--ckpt is required");
        pocket::SafeTensorsIndex index(args.ckpt);
        const std::string* shard_name = index.shard_for_tensor(args.tensor);
        if (shard_name == nullptr) throw std::runtime_error("tensor not found: " + args.tensor);
        pocket::SafeTensorsShard shard(index.shard_path(*shard_name));
        const auto* info = shard.find_tensor(args.tensor);
        if (info == nullptr) throw std::runtime_error("tensor missing from shard header");
        if (info->dtype != pocket::SafeDType::BF16 || info->shape.size() != 1) throw std::runtime_error("expected 1D BF16 gamma");
        const int cols = static_cast<int>(info->shape[0]);
        auto x = make_input(cols);
        auto* gamma = reinterpret_cast<const uint16_t*>(shard.tensor_data(*info));
        constexpr float eps = 1e-6f;
        auto ref = rmsnorm_cpu_ref(x, gamma, eps);

        float* d_x = nullptr;
        uint16_t* d_gamma = nullptr;
        float* d_y = nullptr;
        check_cuda(cudaMalloc(&d_x, x.size() * sizeof(float)), "cudaMalloc x");
        check_cuda(cudaMalloc(&d_gamma, info->nbytes), "cudaMalloc gamma");
        check_cuda(cudaMalloc(&d_y, ref.size() * sizeof(float)), "cudaMalloc y");
        check_cuda(cudaMemcpy(d_x, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice), "copy x");
        check_cuda(cudaMemcpy(d_gamma, gamma, info->nbytes, cudaMemcpyHostToDevice), "copy gamma");
        if (!pocket::rmsnorm_bf16_gamma_cuda(d_x, d_gamma, d_y, cols, eps)) throw std::runtime_error("rmsnorm launch failed");
        check_cuda(cudaDeviceSynchronize(), "sync kernel");
        std::vector<float> got(cols);
        check_cuda(cudaMemcpy(got.data(), d_y, got.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy y");
        cudaFree(d_x);
        cudaFree(d_gamma);
        cudaFree(d_y);

        float max_abs = 0.0f;
        float mean_abs = 0.0f;
        for (int i = 0; i < cols; ++i) {
            const float diff = std::fabs(got[i] - ref[i]);
            max_abs = std::max(max_abs, diff);
            mean_abs += diff;
        }
        mean_abs /= cols;
        if (max_abs > 1e-6f) {
            std::cerr << "[FAIL] " << args.tensor << " max_abs=" << max_abs << " mean_abs=" << mean_abs << "\n";
            return 1;
        }
        std::cout << "[PASS] " << args.tensor << " max_abs=" << max_abs << " mean_abs=" << mean_abs << " cols=" << cols << "\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "[FAIL] " << ex.what() << "\n";
        return 1;
    }
}
