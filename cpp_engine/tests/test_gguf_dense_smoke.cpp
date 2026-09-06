// Minimum smoke test for the GGUF dense forward primitives.
//
// Validates that:
//   1. open_weight_source() returns a working GGUFWeightSource for a .gguf
//      checkpoint and that the mapping table resolves the dense (non-MoE)
//      tensors needed for forward (embed/head/norm/attn projections).
//   2. f16_row_to_float_cuda correctly dequantizes a single token's embedding
//      row from the F16 embed matrix (mirrors the FP4 path's bf16_row_to_float
//      step for embed lookup).
//
// This is the first wired-up brick of the GGUF dense forward path; the full
// embed→attn→head chain will be assembled in follow-up commits.

#include "cuda_ops.hpp"
#include "weight_source.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

#define CHECK_CUDA(expr) do { \
    cudaError_t _err = (expr); \
    if (_err != cudaSuccess) { \
        throw std::runtime_error(std::string("cuda: ") + cudaGetErrorString(_err)); \
    } \
} while (0)

const char* dt_name(pocket::DType dt) {
    using pocket::DType;
    switch (dt) {
        case DType::F16: return "f16";
        case DType::BF16: return "bf16";
        case DType::F32: return "f32";
        case DType::Q8_0: return "q8_0";
        case DType::Q2_K: return "q2_k";
        case DType::IQ2_XXS: return "iq2_xxs";
        default: return "unknown";
    }
}

void describe(const pocket::WeightSource& ws, const std::string& name) {
    if (!ws.has(name)) {
        std::printf("  %-44s MISSING\n", name.c_str());
        return;
    }
    auto view = ws.require(name);
    std::string shape = "[";
    for (size_t i = 0; i < view.shape.size(); ++i) {
        if (i) shape += ",";
        shape += std::to_string(view.shape[i]);
    }
    shape += "]";
    std::printf("  %-44s %-7s %-22s nbytes=%llu\n",
                name.c_str(), dt_name(view.dtype), shape.c_str(),
                static_cast<unsigned long long>(view.nbytes));
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: test_gguf_dense_smoke <model.gguf>\n";
        return 2;
    }
    try {
        auto ws = pocket::open_weight_source(argv[1]);
        if (ws->format() != pocket::WeightSource::Format::GGUF_Q2) {
            throw std::runtime_error("test requires a GGUF Q2 checkpoint");
        }

        std::printf("=== layer 0 dense tensors ===\n");
        describe(*ws, "embed.weight");
        describe(*ws, "head.weight");
        describe(*ws, "norm.weight");
        describe(*ws, "layers.0.attn_norm.weight");
        describe(*ws, "layers.0.ffn_norm.weight");
        describe(*ws, "layers.0.attn.wq_a.weight");
        describe(*ws, "layers.0.attn.q_norm.weight");
        describe(*ws, "layers.0.attn.wq_b.weight");
        describe(*ws, "layers.0.attn.wkv.weight");
        describe(*ws, "layers.0.attn.kv_norm.weight");
        describe(*ws, "layers.0.attn.wo_a.weight");
        describe(*ws, "layers.0.attn.wo_b.weight");
        describe(*ws, "layers.0.attn.attn_sink");
        describe(*ws, "layers.0.ffn.shared_experts.w1.weight");
        describe(*ws, "layers.0.ffn.shared_experts.w2.weight");
        describe(*ws, "layers.0.ffn.shared_experts.w3.weight");

        // --- Embed F16 lookup smoke test ---
        auto embed = ws->require("embed.weight");
        if (embed.dtype != pocket::DType::F16) {
            throw std::runtime_error(std::string("embed dtype expected f16, got ") +
                                     dt_name(embed.dtype));
        }
        if (embed.shape.size() != 2) {
            throw std::runtime_error("embed shape rank != 2");
        }
        // GGUF stores token_embd as [dim, vocab]; per-token row is contiguous
        // (dim elements per token), matching FP4's [vocab, dim] row-major.
        const int dim = static_cast<int>(embed.shape[0]);
        const int vocab = static_cast<int>(embed.shape[1]);
        const int token = 1234;
        if (token >= vocab) {
            throw std::runtime_error("token id out of vocab");
        }

        const uint16_t* host_row = reinterpret_cast<const uint16_t*>(embed.data) +
                                   static_cast<size_t>(token) * dim;

        uint16_t* d_row_f16 = nullptr;
        float* d_x_f32 = nullptr;
        CHECK_CUDA(cudaMalloc(&d_row_f16, dim * sizeof(uint16_t)));
        CHECK_CUDA(cudaMalloc(&d_x_f32, dim * sizeof(float)));
        CHECK_CUDA(cudaMemcpy(d_row_f16, host_row, dim * sizeof(uint16_t),
                              cudaMemcpyHostToDevice));

        if (!pocket::f16_row_to_float_cuda(d_row_f16, d_x_f32, /*row=*/0, dim)) {
            throw std::runtime_error("f16_row_to_float_cuda failed");
        }
        CHECK_CUDA(cudaDeviceSynchronize());

        std::vector<float> h_x(dim);
        CHECK_CUDA(cudaMemcpy(h_x.data(), d_x_f32, dim * sizeof(float),
                              cudaMemcpyDeviceToHost));

        double sumsq = 0.0;
        for (float v : h_x) sumsq += static_cast<double>(v) * static_cast<double>(v);
        const float rms = static_cast<float>(std::sqrt(sumsq / dim));

        std::printf("\n=== embed lookup ===\n");
        std::printf("token=%d dim=%d vocab=%d\n", token, dim, vocab);
        std::printf("first[0..7] = %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n",
                    h_x[0], h_x[1], h_x[2], h_x[3],
                    h_x[4], h_x[5], h_x[6], h_x[7]);
        std::printf("rms=%.4f\n", rms);

        cudaFree(d_row_f16);
        cudaFree(d_x_f32);

        if (!(rms > 1e-6f && rms < 100.0f)) {
            std::cerr << "[FAIL] rms outside plausible range\n";
            return 1;
        }
        std::cout << "[PASS] gguf dense smoke\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "[FAIL] " << ex.what() << "\n";
        return 1;
    }
}
