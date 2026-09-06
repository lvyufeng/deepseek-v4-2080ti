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

std::vector<float> row_ref(const uint16_t* matrix, int row, int cols) {
    std::vector<float> y(cols);
    const uint16_t* src = matrix + static_cast<size_t>(row) * cols;
    for (int i = 0; i < cols; ++i) y[i] = bf16_to_float(src[i]);
    return y;
}

std::vector<float> matvec_ref(const std::vector<float>& x, const uint16_t* w, int rows, int cols) {
    std::vector<float> y(rows, 0.0f);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) y[r] += bf16_to_float(w[static_cast<size_t>(r) * cols + c]) * x[c];
    }
    return y;
}

std::vector<float> matvec_ref_double(const std::vector<float>& x, const uint16_t* w, int rows, int cols) {
    std::vector<float> y(rows, 0.0f);
    for (int r = 0; r < rows; ++r) {
        double sum = 0.0;
        for (int c = 0; c < cols; ++c) sum += static_cast<double>(bf16_to_float(w[static_cast<size_t>(r) * cols + c])) * x[c];
        y[r] = static_cast<float>(sum);
    }
    return y;
}

const pocket::SafeTensorInfo* require_tensor(const pocket::SafeTensorsShard& shard, const std::string& name, pocket::SafeDType dtype) {
    const auto* info = shard.find_tensor(name);
    if (info == nullptr) throw std::runtime_error("missing tensor: " + name);
    if (info->dtype != dtype || info->shape.size() != 2) throw std::runtime_error("bad tensor: " + name);
    return info;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2) {
            std::cerr << "usage: test_embedding_head <ckpt_dir>\n";
            return 2;
        }
        if (!pocket::cuda_runtime_available()) {
            std::cout << "[SKIP] CUDA runtime is not available\n";
            return 0;
        }
        const std::string ckpt = argv[1];
        pocket::SafeTensorsIndex index(ckpt);
        const std::string* embed_shard_name = index.shard_for_tensor("embed.weight");
        const std::string* head_shard_name = index.shard_for_tensor("head.weight");
        if (embed_shard_name == nullptr || head_shard_name == nullptr) throw std::runtime_error("missing embed/head tensor");
        pocket::SafeTensorsShard embed_shard(index.shard_path(*embed_shard_name));
        pocket::SafeTensorsShard head_shard(index.shard_path(*head_shard_name));
        const auto* embed = require_tensor(embed_shard, "embed.weight", pocket::SafeDType::BF16);
        const auto* head = require_tensor(head_shard, "head.weight", pocket::SafeDType::BF16);
        const int token = 1234;
        const int dim = static_cast<int>(embed->shape[1]);
        const int vocab_rows = 128;
        if (head->shape[1] != static_cast<uint64_t>(dim)) throw std::runtime_error("embed/head dim mismatch");

        auto* embed_data = reinterpret_cast<const uint16_t*>(embed_shard.tensor_data(*embed));
        auto* head_data = reinterpret_cast<const uint16_t*>(head_shard.tensor_data(*head));
        auto x_ref = row_ref(embed_data, token, dim);
        auto logits_ref = matvec_ref(x_ref, head_data, vocab_rows, dim);
        auto logits_ref_double = matvec_ref_double(x_ref, head_data, vocab_rows, dim);

        uint16_t* d_embed = nullptr;
        uint16_t* d_head = nullptr;
        float* d_x = nullptr;
        float* d_logits = nullptr;
        float* d_logits_cpu_order = nullptr;
        const uint16_t* token_row = embed_data + static_cast<size_t>(token) * dim;
        check_cuda(cudaMalloc(&d_embed, static_cast<size_t>(dim) * sizeof(uint16_t)), "cudaMalloc embed");
        check_cuda(cudaMalloc(&d_head, static_cast<size_t>(vocab_rows) * dim * sizeof(uint16_t)), "cudaMalloc head");
        check_cuda(cudaMalloc(&d_x, static_cast<size_t>(dim) * sizeof(float)), "cudaMalloc x");
        check_cuda(cudaMalloc(&d_logits, static_cast<size_t>(vocab_rows) * sizeof(float)), "cudaMalloc logits");
        check_cuda(cudaMalloc(&d_logits_cpu_order, static_cast<size_t>(vocab_rows) * sizeof(float)), "cudaMalloc logits cpu order");
        check_cuda(cudaMemcpy(d_embed, token_row, static_cast<size_t>(dim) * sizeof(uint16_t), cudaMemcpyHostToDevice), "copy embed row");
        check_cuda(cudaMemcpy(d_head, head_data, static_cast<size_t>(vocab_rows) * dim * sizeof(uint16_t), cudaMemcpyHostToDevice), "copy head");
        if (!pocket::bf16_row_to_float_cuda(d_embed, d_x, 0, dim)) throw std::runtime_error("embed lookup launch failed");
        if (!pocket::bf16_matvec_cuda(d_x, d_head, d_logits, vocab_rows, dim)) throw std::runtime_error("head matvec launch failed");
        if (!pocket::bf16_matvec_cpu_order_cuda(d_x, d_head, d_logits_cpu_order, vocab_rows, dim)) throw std::runtime_error("head cpu-order matvec launch failed");
        check_cuda(cudaDeviceSynchronize(), "sync kernels");
        std::vector<float> x(dim);
        std::vector<float> logits(vocab_rows);
        std::vector<float> logits_cpu_order(vocab_rows);
        check_cuda(cudaMemcpy(x.data(), d_x, x.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy x");
        check_cuda(cudaMemcpy(logits.data(), d_logits, logits.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy logits");
        check_cuda(cudaMemcpy(logits_cpu_order.data(), d_logits_cpu_order, logits_cpu_order.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy logits cpu order");
        cudaFree(d_embed);
        cudaFree(d_head);
        cudaFree(d_x);
        cudaFree(d_logits);
        cudaFree(d_logits_cpu_order);

        float max_x = 0.0f;
        for (int i = 0; i < dim; ++i) max_x = std::max(max_x, std::fabs(x[i] - x_ref[i]));
        float max_logits = 0.0f;
        float max_logits_cpu_order = 0.0f;
        for (int i = 0; i < vocab_rows; ++i) {
            max_logits = std::max(max_logits, std::fabs(logits[i] - logits_ref[i]));
            max_logits_cpu_order = std::max(max_logits_cpu_order, std::fabs(logits_cpu_order[i] - logits_ref_double[i]));
        }
        if (max_x > 0.0f || max_logits > 1e-4f || max_logits_cpu_order > 0.0f) {
            std::cerr << "[FAIL] embedding_head max_x=" << max_x << " max_logits=" << max_logits << " max_logits_cpu_order=" << max_logits_cpu_order << "\n";
            return 1;
        }
        std::cout << "[PASS] embedding_head max_x=" << max_x << " max_logits=" << max_logits << " max_logits_cpu_order=" << max_logits_cpu_order << " token=" << token << " rows=" << vocab_rows << " dim=" << dim << "\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "[FAIL] " << ex.what() << "\n";
        return 1;
    }
}
