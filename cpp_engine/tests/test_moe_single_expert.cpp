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

void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
}

float bf16_to_float(uint16_t bits) {
    uint32_t value = static_cast<uint32_t>(bits) << 16;
    float out;
    std::memcpy(&out, &value, sizeof(out));
    return out;
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

std::vector<float> make_input(int cols) {
    std::vector<float> x(cols);
    for (int i = 0; i < cols; ++i) x[i] = static_cast<float>((i % 37) - 18) * 0.01f;
    return x;
}

std::vector<float> rmsnorm_cpu(const std::vector<float>& x, const uint16_t* gamma, float eps) {
    double sum_sq = 0.0;
    for (float v : x) sum_sq += static_cast<double>(v) * v;
    const float inv = 1.0f / std::sqrt(static_cast<float>(sum_sq / x.size()) + eps);
    std::vector<float> y(x.size());
    for (size_t i = 0; i < x.size(); ++i) y[i] = x[i] * inv * bf16_to_float(gamma[i]);
    return y;
}

std::vector<float> fp4_matvec_cpu(const std::vector<float>& x, const uint8_t* w, const uint8_t* scale, int rows, int cols) {
    const int packed_cols = cols / 2;
    const int scale_cols = cols / 32;
    std::vector<float> y(rows, 0.0f);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            const uint8_t packed = w[static_cast<size_t>(r) * packed_cols + c / 2];
            const uint8_t code = (c & 1) ? static_cast<uint8_t>(packed >> 4) : static_cast<uint8_t>(packed & 0x0f);
            y[r] += fp4_value(code) * e8m0_value(scale[static_cast<size_t>(r) * scale_cols + c / 32]) * x[c];
        }
    }
    return y;
}

std::vector<float> silu_mul_cpu(const std::vector<float>& gate, const std::vector<float>& up) {
    std::vector<float> y(gate.size());
    for (size_t i = 0; i < y.size(); ++i) y[i] = gate[i] / (1.0f + std::exp(-gate[i])) * up[i];
    return y;
}

struct TensorPair {
    pocket::SafeTensorsShard shard;
    const pocket::SafeTensorInfo* weight;
    const pocket::SafeTensorInfo* scale;
    pocket::SafeFp4TensorPair pair;
};

TensorPair open_pair(const pocket::SafeTensorsIndex& index, const std::string& name) {
    const std::string* shard_name = index.shard_for_tensor(name);
    if (shard_name == nullptr) throw std::runtime_error("missing tensor: " + name);
    TensorPair out{pocket::SafeTensorsShard(index.shard_path(*shard_name)), nullptr, nullptr, {}};
    out.pair = pocket::resolve_fp4_tensor_pair(index, out.shard, name);
    out.weight = out.shard.find_tensor(out.pair.weight_name);
    out.scale = out.shard.find_tensor(out.pair.scale_name);
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            std::cerr << "usage: test_moe_single_expert <ckpt_dir>\n";
            return 2;
        }
        if (!pocket::cuda_runtime_available()) {
            std::cout << "[SKIP] CUDA runtime is not available\n";
            return 0;
        }
        const std::string ckpt = argv[1];
        pocket::SafeTensorsIndex index(ckpt);

        const std::string norm_name = "layers.0.ffn_norm.weight";
        const std::string* norm_shard_name = index.shard_for_tensor(norm_name);
        if (norm_shard_name == nullptr) throw std::runtime_error("missing norm tensor");
        pocket::SafeTensorsShard norm_shard(index.shard_path(*norm_shard_name));
        const auto* norm_info = norm_shard.find_tensor(norm_name);
        auto* gamma = reinterpret_cast<const uint16_t*>(norm_shard.tensor_data(*norm_info));

        TensorPair w1 = open_pair(index, "layers.0.ffn.experts.0.w1.weight");
        TensorPair w2 = open_pair(index, "layers.0.ffn.experts.0.w2.weight");
        TensorPair w3 = open_pair(index, "layers.0.ffn.experts.0.w3.weight");
        const int dim = static_cast<int>(w1.pair.cols);
        const int inter = static_cast<int>(w1.pair.rows);
        if (w3.pair.rows != static_cast<uint64_t>(inter) || w2.pair.cols != static_cast<uint64_t>(inter)) throw std::runtime_error("expert shape mismatch");

        auto x = make_input(dim);
        auto norm = rmsnorm_cpu(x, gamma, 1e-6f);
        auto gate = fp4_matvec_cpu(norm, w1.shard.tensor_data(*w1.weight), w1.shard.tensor_data(*w1.scale), inter, dim);
        auto up = fp4_matvec_cpu(norm, w3.shard.tensor_data(*w3.weight), w3.shard.tensor_data(*w3.scale), inter, dim);
        auto hidden = silu_mul_cpu(gate, up);
        auto ref = fp4_matvec_cpu(hidden, w2.shard.tensor_data(*w2.weight), w2.shard.tensor_data(*w2.scale), dim, inter);

        float* d_x = nullptr;
        uint16_t* d_gamma = nullptr;
        float* d_norm = nullptr;
        uint8_t* d_w1 = nullptr;
        uint8_t* d_s1 = nullptr;
        uint8_t* d_w2 = nullptr;
        uint8_t* d_s2 = nullptr;
        uint8_t* d_w3 = nullptr;
        uint8_t* d_s3 = nullptr;
        float* d_gate = nullptr;
        float* d_up = nullptr;
        float* d_hidden = nullptr;
        float* d_y = nullptr;
        check_cuda(cudaMalloc(&d_x, x.size() * sizeof(float)), "cudaMalloc x");
        check_cuda(cudaMalloc(&d_gamma, norm_info->nbytes), "cudaMalloc gamma");
        check_cuda(cudaMalloc(&d_norm, x.size() * sizeof(float)), "cudaMalloc norm");
        check_cuda(cudaMalloc(&d_gate, static_cast<size_t>(inter) * sizeof(float)), "cudaMalloc gate");
        check_cuda(cudaMalloc(&d_up, static_cast<size_t>(inter) * sizeof(float)), "cudaMalloc up");
        check_cuda(cudaMalloc(&d_hidden, static_cast<size_t>(inter) * sizeof(float)), "cudaMalloc hidden");
        check_cuda(cudaMalloc(&d_y, static_cast<size_t>(dim) * sizeof(float)), "cudaMalloc y");
        check_cuda(cudaMalloc(&d_w1, w1.weight->nbytes), "cudaMalloc w1");
        check_cuda(cudaMalloc(&d_s1, w1.scale->nbytes), "cudaMalloc s1");
        check_cuda(cudaMalloc(&d_w2, w2.weight->nbytes), "cudaMalloc w2");
        check_cuda(cudaMalloc(&d_s2, w2.scale->nbytes), "cudaMalloc s2");
        check_cuda(cudaMalloc(&d_w3, w3.weight->nbytes), "cudaMalloc w3");
        check_cuda(cudaMalloc(&d_s3, w3.scale->nbytes), "cudaMalloc s3");
        check_cuda(cudaMemcpy(d_x, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice), "copy x");
        check_cuda(cudaMemcpy(d_gamma, gamma, norm_info->nbytes, cudaMemcpyHostToDevice), "copy gamma");
        check_cuda(cudaMemcpy(d_w1, w1.shard.tensor_data(*w1.weight), w1.weight->nbytes, cudaMemcpyHostToDevice), "copy w1");
        check_cuda(cudaMemcpy(d_s1, w1.shard.tensor_data(*w1.scale), w1.scale->nbytes, cudaMemcpyHostToDevice), "copy s1");
        check_cuda(cudaMemcpy(d_w2, w2.shard.tensor_data(*w2.weight), w2.weight->nbytes, cudaMemcpyHostToDevice), "copy w2");
        check_cuda(cudaMemcpy(d_s2, w2.shard.tensor_data(*w2.scale), w2.scale->nbytes, cudaMemcpyHostToDevice), "copy s2");
        check_cuda(cudaMemcpy(d_w3, w3.shard.tensor_data(*w3.weight), w3.weight->nbytes, cudaMemcpyHostToDevice), "copy w3");
        check_cuda(cudaMemcpy(d_s3, w3.shard.tensor_data(*w3.scale), w3.scale->nbytes, cudaMemcpyHostToDevice), "copy s3");

        if (!pocket::rmsnorm_bf16_gamma_cuda(d_x, d_gamma, d_norm, dim, 1e-6f)) throw std::runtime_error("rmsnorm launch failed");
        if (!pocket::fp4_e2m1_e8m0_matvec_cuda(d_norm, d_w1, d_s1, d_gate, inter, dim)) throw std::runtime_error("w1 launch failed");
        if (!pocket::fp4_e2m1_e8m0_matvec_cuda(d_norm, d_w3, d_s3, d_up, inter, dim)) throw std::runtime_error("w3 launch failed");
        if (!pocket::silu_mul_cuda(d_gate, d_up, d_hidden, inter)) throw std::runtime_error("silu_mul launch failed");
        if (!pocket::fp4_e2m1_e8m0_matvec_cuda(d_hidden, d_w2, d_s2, d_y, dim, inter)) throw std::runtime_error("w2 launch failed");
        check_cuda(cudaDeviceSynchronize(), "sync kernels");

        std::vector<float> got(dim);
        check_cuda(cudaMemcpy(got.data(), d_y, got.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy y");
        cudaFree(d_x); cudaFree(d_gamma); cudaFree(d_norm); cudaFree(d_w1); cudaFree(d_s1); cudaFree(d_w2); cudaFree(d_s2); cudaFree(d_w3); cudaFree(d_s3); cudaFree(d_gate); cudaFree(d_up); cudaFree(d_hidden); cudaFree(d_y);

        float max_abs = 0.0f;
        float mean_abs = 0.0f;
        for (int i = 0; i < dim; ++i) {
            const float diff = std::fabs(got[i] - ref[i]);
            max_abs = std::max(max_abs, diff);
            mean_abs += diff;
        }
        mean_abs /= dim;
        if (max_abs > 1e-2f) {
            std::cerr << "[FAIL] moe_single_expert max_abs=" << max_abs << " mean_abs=" << mean_abs << "\n";
            return 1;
        }
        std::cout << "[PASS] moe_single_expert max_abs=" << max_abs << " mean_abs=" << mean_abs << " dim=" << dim << " inter=" << inter << "\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "[FAIL] " << ex.what() << "\n";
        return 1;
    }
}
