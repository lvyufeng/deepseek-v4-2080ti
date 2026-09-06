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
    std::string weight = "layers.0.attn.wkv.weight";
    int rows = 128;
};

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--ckpt" && i + 1 < argc) args.ckpt = argv[++i];
        else if (arg == "--weight" && i + 1 < argc) args.weight = argv[++i];
        else if (arg == "--rows" && i + 1 < argc) args.rows = std::stoi(argv[++i]);
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

float fp8_e4m3_value(uint8_t code) {
    const int sign = (code >> 7) & 1;
    const int exp = (code >> 3) & 0xf;
    const int mant = code & 0x7;
    const float value = exp == 0 ? std::ldexp(static_cast<float>(mant) * 0.125f, -6) : std::ldexp(1.0f + static_cast<float>(mant) * 0.125f, exp - 7);
    return sign ? -value : value;
}

float e8m0_value(uint8_t code) {
    return std::exp2(static_cast<float>(static_cast<int>(code) - 127));
}

std::vector<float> make_input(int cols) {
    std::vector<float> x(cols);
    for (int i = 0; i < cols; ++i) x[i] = static_cast<float>((i % 31) - 15) * 0.015625f;
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

std::vector<float> fp8_matvec_cpu(const std::vector<float>& x, const uint8_t* w, const uint8_t* scale, int rows, int cols) {
    constexpr int block = 128;
    const int scale_cols = cols / block;
    std::vector<float> y(rows, 0.0f);
    for (int r = 0; r < rows; ++r) {
        const int rb = r / block;
        for (int c = 0; c < cols; ++c) {
            y[r] += fp8_e4m3_value(w[static_cast<size_t>(r) * cols + c]) * e8m0_value(scale[static_cast<size_t>(rb) * scale_cols + c / block]) * x[c];
        }
    }
    return y;
}

std::string scale_name_for(const std::string& weight) {
    const std::string suffix = ".weight";
    if (weight.size() >= suffix.size() && weight.compare(weight.size() - suffix.size(), suffix.size(), suffix) == 0) {
        return weight.substr(0, weight.size() - suffix.size()) + ".scale";
    }
    return weight + ".scale";
}

void test_head_rmsnorm_rope_kernel() {
    constexpr int heads = 2;
    constexpr int head_dim = 6;
    constexpr int rope_dim = 4;
    constexpr int position = 3;
    constexpr float theta = 10000.0f;
    constexpr float eps = 1e-6f;
    std::vector<float> x = {
        0.25f, -0.5f, 0.75f, 1.0f, -1.25f, 0.5f,
        -0.125f, 0.375f, -0.625f, 0.875f, 1.125f, -1.5f,
    };
    std::vector<float> ref = x;
    for (int h = 0; h < heads; ++h) {
        float sum_sq = 0.0f;
        for (int i = 0; i < head_dim; ++i) sum_sq += ref[h * head_dim + i] * ref[h * head_dim + i];
        const float inv = 1.0f / std::sqrt(sum_sq / head_dim + eps);
        for (int i = 0; i < head_dim; ++i) ref[h * head_dim + i] *= inv;
        const int rope_start = head_dim - rope_dim;
        for (int pair = 0; pair < rope_dim; pair += 2) {
            const float angle = static_cast<float>(position) / std::pow(theta, static_cast<float>(pair) / rope_dim);
            const float c = std::cos(angle);
            const float s = std::sin(angle);
            const int base = h * head_dim + rope_start + pair;
            const float a = ref[base];
            const float b = ref[base + 1];
            ref[base] = a * c - b * s;
            ref[base + 1] = a * s + b * c;
        }
    }

    float* d_x = nullptr;
    check_cuda(cudaMalloc(&d_x, x.size() * sizeof(float)), "cudaMalloc rope x");
    check_cuda(cudaMemcpy(d_x, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice), "copy rope x");
    if (!pocket::head_rmsnorm_rope_cuda(d_x, heads, head_dim, rope_dim, position, theta, false, eps)) throw std::runtime_error("head rmsnorm rope launch failed");
    check_cuda(cudaDeviceSynchronize(), "sync head rmsnorm rope");
    std::vector<float> got(x.size());
    check_cuda(cudaMemcpy(got.data(), d_x, got.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy rope x");
    cudaFree(d_x);
    for (size_t i = 0; i < got.size(); ++i) {
        if (std::fabs(got[i] - ref[i]) > 1e-6f) throw std::runtime_error("head rmsnorm rope mismatch");
    }
}

void test_cached_attention_kernel() {
    constexpr int heads = 2;
    constexpr int head_dim = 3;
    constexpr int cache_len = 3;
    const float scale = 0.25f;
    std::vector<float> q = {0.25f, -0.5f, 0.75f, -1.0f, 0.5f, 0.125f};
    std::vector<float> kv = {
        0.5f, -0.25f, 0.75f,
        -0.125f, 0.625f, 1.0f,
        0.875f, -0.5f, 0.25f,
    };
    std::vector<float> sink = {0.0f, -0.25f};
    std::vector<float> ref(q.size());
    for (int h = 0; h < heads; ++h) {
        std::vector<float> logits(cache_len);
        float max_logit = sink[h];
        for (int t = 0; t < cache_len; ++t) {
            float dot = 0.0f;
            for (int i = 0; i < head_dim; ++i) dot += q[h * head_dim + i] * kv[t * head_dim + i];
            logits[t] = dot * scale;
            max_logit = std::max(max_logit, logits[t]);
        }
        float denom = std::exp(sink[h] - max_logit);
        for (float logit : logits) denom += std::exp(logit - max_logit);
        for (int i = 0; i < head_dim; ++i) {
            float out = 0.0f;
            for (int t = 0; t < cache_len; ++t) out += std::exp(logits[t] - max_logit) / denom * kv[t * head_dim + i];
            ref[h * head_dim + i] = out;
        }
    }

    float* d_q = nullptr;
    float* d_kv = nullptr;
    float* d_sink = nullptr;
    float* d_y = nullptr;
    check_cuda(cudaMalloc(&d_q, q.size() * sizeof(float)), "cudaMalloc cached q");
    check_cuda(cudaMalloc(&d_kv, kv.size() * sizeof(float)), "cudaMalloc cached kv");
    check_cuda(cudaMalloc(&d_sink, sink.size() * sizeof(float)), "cudaMalloc cached sink");
    check_cuda(cudaMalloc(&d_y, q.size() * sizeof(float)), "cudaMalloc cached y");
    check_cuda(cudaMemcpy(d_q, q.data(), q.size() * sizeof(float), cudaMemcpyHostToDevice), "copy cached q");
    check_cuda(cudaMemcpy(d_kv, kv.data(), kv.size() * sizeof(float), cudaMemcpyHostToDevice), "copy cached kv");
    check_cuda(cudaMemcpy(d_sink, sink.data(), sink.size() * sizeof(float), cudaMemcpyHostToDevice), "copy cached sink");
    if (!pocket::cached_single_token_attention_cuda(d_q, d_kv, d_sink, d_y, heads, head_dim, cache_len, scale)) throw std::runtime_error("cached attention launch failed");
    check_cuda(cudaDeviceSynchronize(), "sync cached attention");
    std::vector<float> got(q.size());
    check_cuda(cudaMemcpy(got.data(), d_y, got.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy cached y");
    cudaFree(d_q);
    cudaFree(d_kv);
    cudaFree(d_sink);
    cudaFree(d_y);
    for (size_t i = 0; i < got.size(); ++i) {
        if (std::fabs(got[i] - ref[i]) > 1e-6f) throw std::runtime_error("cached attention mismatch");
    }
}

void test_prefill_headpair_serial_matches_indexed() {
    constexpr int tokens = 4;
    constexpr int heads = 5;
    constexpr int head_dim = 64;
    constexpr int topk = 4;
    constexpr int kv_len = 4;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    std::vector<float> q(static_cast<size_t>(tokens) * heads * head_dim);
    std::vector<float> kv(static_cast<size_t>(kv_len) * head_dim);
    std::vector<float> sink(heads);
    std::vector<int32_t> indices(static_cast<size_t>(tokens) * topk, -1);
    for (size_t i = 0; i < q.size(); ++i) q[i] = std::sin(static_cast<float>(i) * 0.013f) * 0.25f;
    for (size_t i = 0; i < kv.size(); ++i) kv[i] = std::cos(static_cast<float>(i) * 0.017f) * 0.2f;
    for (int h = 0; h < heads; ++h) sink[h] = -0.15f + 0.07f * static_cast<float>(h);
    for (int t = 0; t < tokens; ++t) {
        for (int k = 0; k < topk; ++k) indices[static_cast<size_t>(t) * topk + k] = (k <= t) ? k : -1;
    }

    float* d_q = nullptr;
    float* d_kv = nullptr;
    float* d_sink = nullptr;
    float* d_indexed = nullptr;
    float* d_serial = nullptr;
    int32_t* d_indices = nullptr;
    const size_t out_elems = q.size();
    check_cuda(cudaMalloc(&d_q, q.size() * sizeof(float)), "cudaMalloc prefill q");
    check_cuda(cudaMalloc(&d_kv, kv.size() * sizeof(float)), "cudaMalloc prefill kv");
    check_cuda(cudaMalloc(&d_sink, sink.size() * sizeof(float)), "cudaMalloc prefill sink");
    check_cuda(cudaMalloc(&d_indices, indices.size() * sizeof(int32_t)), "cudaMalloc prefill indices");
    check_cuda(cudaMalloc(&d_indexed, out_elems * sizeof(float)), "cudaMalloc prefill indexed");
    check_cuda(cudaMalloc(&d_serial, out_elems * sizeof(float)), "cudaMalloc prefill serial");
    check_cuda(cudaMemcpy(d_q, q.data(), q.size() * sizeof(float), cudaMemcpyHostToDevice), "copy prefill q");
    check_cuda(cudaMemcpy(d_kv, kv.data(), kv.size() * sizeof(float), cudaMemcpyHostToDevice), "copy prefill kv");
    check_cuda(cudaMemcpy(d_sink, sink.data(), sink.size() * sizeof(float), cudaMemcpyHostToDevice), "copy prefill sink");
    check_cuda(cudaMemcpy(d_indices, indices.data(), indices.size() * sizeof(int32_t), cudaMemcpyHostToDevice), "copy prefill indices");
    if (!pocket::prefill_sparse_attention_indexed_cuda(d_q, d_kv, d_sink, d_indices, d_indexed, tokens, heads, kv_len, topk, head_dim, scale)) throw std::runtime_error("prefill indexed attention launch failed");
    if (!pocket::prefill_sparse_attention_headpair_serial_cuda(d_q, d_kv, d_sink, d_indices, d_serial, tokens, heads, kv_len, topk, head_dim, scale)) throw std::runtime_error("prefill headpair serial attention launch failed");
    check_cuda(cudaDeviceSynchronize(), "sync prefill attention compare");
    std::vector<float> indexed(out_elems);
    std::vector<float> serial(out_elems);
    check_cuda(cudaMemcpy(indexed.data(), d_indexed, out_elems * sizeof(float), cudaMemcpyDeviceToHost), "copy indexed attention");
    check_cuda(cudaMemcpy(serial.data(), d_serial, out_elems * sizeof(float), cudaMemcpyDeviceToHost), "copy serial attention");
    cudaFree(d_q);
    cudaFree(d_kv);
    cudaFree(d_sink);
    cudaFree(d_indices);
    cudaFree(d_indexed);
    cudaFree(d_serial);
    float max_abs = 0.0f;
    for (size_t i = 0; i < out_elems; ++i) max_abs = std::max(max_abs, std::fabs(indexed[i] - serial[i]));
    if (max_abs > 1e-7f) throw std::runtime_error("prefill headpair serial mismatch vs indexed");
}

void test_single_token_attention_kernel() {
    constexpr int heads = 3;
    constexpr int head_dim = 4;
    const float scale = 0.5f;
    std::vector<float> q = {
        0.25f, -0.5f, 0.75f, 1.0f,
        -1.0f, 0.5f, 0.25f, -0.75f,
        0.125f, 0.25f, -0.375f, 0.5f,
    };
    std::vector<float> kv = {0.5f, -0.25f, 0.75f, 1.25f};
    std::vector<float> sink = {0.0f, -0.5f, 0.25f};
    std::vector<float> ref(q.size());
    for (int h = 0; h < heads; ++h) {
        float dot = 0.0f;
        for (int i = 0; i < head_dim; ++i) dot += q[h * head_dim + i] * kv[i];
        const float token_logit = dot * scale;
        const float m = std::max(token_logit, sink[h]);
        const float token_weight = std::exp(token_logit - m) / (std::exp(token_logit - m) + std::exp(sink[h] - m));
        for (int i = 0; i < head_dim; ++i) ref[h * head_dim + i] = kv[i] * token_weight;
    }

    float* d_q = nullptr;
    float* d_kv = nullptr;
    float* d_sink = nullptr;
    float* d_y = nullptr;
    check_cuda(cudaMalloc(&d_q, q.size() * sizeof(float)), "cudaMalloc attn q");
    check_cuda(cudaMalloc(&d_kv, kv.size() * sizeof(float)), "cudaMalloc attn kv");
    check_cuda(cudaMalloc(&d_sink, sink.size() * sizeof(float)), "cudaMalloc attn sink");
    check_cuda(cudaMalloc(&d_y, q.size() * sizeof(float)), "cudaMalloc attn y");
    check_cuda(cudaMemcpy(d_q, q.data(), q.size() * sizeof(float), cudaMemcpyHostToDevice), "copy attn q");
    check_cuda(cudaMemcpy(d_kv, kv.data(), kv.size() * sizeof(float), cudaMemcpyHostToDevice), "copy attn kv");
    check_cuda(cudaMemcpy(d_sink, sink.data(), sink.size() * sizeof(float), cudaMemcpyHostToDevice), "copy attn sink");
    if (!pocket::single_token_sparse_attention_cuda(d_q, d_kv, d_sink, d_y, heads, head_dim, scale)) throw std::runtime_error("single token attention launch failed");
    check_cuda(cudaDeviceSynchronize(), "sync single token attention");
    std::vector<float> got(q.size());
    check_cuda(cudaMemcpy(got.data(), d_y, got.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy attn y");
    cudaFree(d_q);
    cudaFree(d_kv);
    cudaFree(d_sink);
    cudaFree(d_y);
    for (size_t i = 0; i < got.size(); ++i) {
        if (std::fabs(got[i] - ref[i]) > 1e-6f) throw std::runtime_error("single token attention mismatch");
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (!pocket::cuda_runtime_available()) {
            std::cout << "[SKIP] CUDA runtime is not available\n";
            return 0;
        }
        test_head_rmsnorm_rope_kernel();
        test_single_token_attention_kernel();
        test_cached_attention_kernel();
        test_prefill_headpair_serial_matches_indexed();
        Args args = parse_args(argc, argv);
        if (args.ckpt.empty()) throw std::runtime_error("--ckpt is required");
        pocket::SafeTensorsIndex index(args.ckpt);

        const std::string norm_name = "layers.0.attn_norm.weight";
        const std::string scale_name = scale_name_for(args.weight);
        const std::string* norm_shard_name = index.shard_for_tensor(norm_name);
        const std::string* weight_shard_name = index.shard_for_tensor(args.weight);
        if (norm_shard_name == nullptr || weight_shard_name == nullptr) throw std::runtime_error("missing tensor in index");
        pocket::SafeTensorsShard norm_shard(index.shard_path(*norm_shard_name));
        pocket::SafeTensorsShard weight_shard(index.shard_path(*weight_shard_name));
        const auto* norm_info = norm_shard.find_tensor(norm_name);
        const auto* weight_info = weight_shard.find_tensor(args.weight);
        const auto* scale_info = weight_shard.find_tensor(scale_name);
        if (norm_info == nullptr || weight_info == nullptr || scale_info == nullptr) throw std::runtime_error("missing tensor in shard");
        const int cols = static_cast<int>(weight_info->shape[1]);
        const int rows = std::min(args.rows, static_cast<int>(weight_info->shape[0]));
        if (norm_info->shape[0] != static_cast<uint64_t>(cols)) throw std::runtime_error("norm/weight shape mismatch");
        if (rows % 128 != 0 || cols % 128 != 0) throw std::runtime_error("rows/cols must be multiples of 128");

        auto x = make_input(cols);
        auto* gamma = reinterpret_cast<const uint16_t*>(norm_shard.tensor_data(*norm_info));
        auto norm_ref = rmsnorm_cpu(x, gamma, 1e-6f);
        auto ref = fp8_matvec_cpu(norm_ref, weight_shard.tensor_data(*weight_info), weight_shard.tensor_data(*scale_info), rows, cols);

        float* d_x = nullptr;
        uint16_t* d_gamma = nullptr;
        float* d_norm = nullptr;
        uint8_t* d_w = nullptr;
        uint8_t* d_scale = nullptr;
        float* d_y = nullptr;
        check_cuda(cudaMalloc(&d_x, x.size() * sizeof(float)), "cudaMalloc x");
        check_cuda(cudaMalloc(&d_gamma, norm_info->nbytes), "cudaMalloc gamma");
        check_cuda(cudaMalloc(&d_norm, x.size() * sizeof(float)), "cudaMalloc norm");
        check_cuda(cudaMalloc(&d_w, static_cast<size_t>(rows) * cols), "cudaMalloc weight");
        check_cuda(cudaMalloc(&d_scale, static_cast<size_t>(rows / 128) * (cols / 128)), "cudaMalloc scale");
        check_cuda(cudaMalloc(&d_y, ref.size() * sizeof(float)), "cudaMalloc y");
        check_cuda(cudaMemcpy(d_x, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice), "copy x");
        check_cuda(cudaMemcpy(d_gamma, gamma, norm_info->nbytes, cudaMemcpyHostToDevice), "copy gamma");
        check_cuda(cudaMemcpy(d_w, weight_shard.tensor_data(*weight_info), static_cast<size_t>(rows) * cols, cudaMemcpyHostToDevice), "copy weight");
        check_cuda(cudaMemcpy(d_scale, weight_shard.tensor_data(*scale_info), static_cast<size_t>(rows / 128) * (cols / 128), cudaMemcpyHostToDevice), "copy scale");
        if (!pocket::rmsnorm_bf16_gamma_cuda(d_x, d_gamma, d_norm, cols, 1e-6f)) throw std::runtime_error("rmsnorm launch failed");
        if (!pocket::fp8_e4m3_e8m0_matvec_cuda(d_norm, d_w, d_scale, d_y, rows, cols)) throw std::runtime_error("fp8 matvec launch failed");
        check_cuda(cudaDeviceSynchronize(), "sync kernels");
        std::vector<float> got(rows);
        check_cuda(cudaMemcpy(got.data(), d_y, got.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy y");
        cudaFree(d_x);
        cudaFree(d_gamma);
        cudaFree(d_norm);
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
        mean_abs /= rows;
        if (max_abs > 1e-3f) {
            std::cerr << "[FAIL] " << args.weight << " max_abs=" << max_abs << " mean_abs=" << mean_abs << "\n";
            return 1;
        }
        std::cout << "[PASS] " << args.weight << " max_abs=" << max_abs << " mean_abs=" << mean_abs << " rows=" << rows << " cols=" << cols << "\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "[FAIL] " << ex.what() << "\n";
        return 1;
    }
}
