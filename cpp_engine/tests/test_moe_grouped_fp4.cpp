#include "cuda_ops.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
    }
}

uint8_t random_fp4_byte(std::mt19937& rng) {
    std::uniform_int_distribution<int> code(0, 15);
    const uint8_t lo = static_cast<uint8_t>(code(rng));
    const uint8_t hi = static_cast<uint8_t>(code(rng));
    return static_cast<uint8_t>((hi << 4) | lo);
}

uint8_t random_e8m0_byte(std::mt19937& rng) {
    std::uniform_int_distribution<int> dist(120, 134);
    return static_cast<uint8_t>(dist(rng));
}

std::vector<uint8_t> random_fp4_weight(int rows, int cols, std::mt19937& rng) {
    std::vector<uint8_t> w(static_cast<size_t>(rows) * (cols / 2));
    for (auto& b : w) b = random_fp4_byte(rng);
    return w;
}

std::vector<uint8_t> random_fp4_scale(int rows, int cols, std::mt19937& rng) {
    std::vector<uint8_t> s(static_cast<size_t>(rows) * (cols / 32));
    for (auto& b : s) b = random_e8m0_byte(rng);
    return s;
}

void compare(const std::vector<float>& a, const std::vector<float>& b, float tol, const std::string& tag) {
    if (a.size() != b.size()) throw std::runtime_error(tag + ": size mismatch");
    float max_abs = 0.0f;
    float mean_abs = 0.0f;
    int dump = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const float diff = std::fabs(a[i] - b[i]);
        if (diff > max_abs) max_abs = diff;
        mean_abs += diff;
        if (diff > tol && dump < 8) {
            std::cerr << tag << " idx=" << i << " grouped=" << a[i] << " ref=" << b[i] << " diff=" << diff << "\n";
            ++dump;
        }
    }
    mean_abs /= static_cast<float>(a.size());
    std::cout << tag << " max_abs=" << max_abs << " mean_abs=" << mean_abs << " size=" << a.size() << "\n";
    if (max_abs > tol) throw std::runtime_error(tag + ": max_abs above tolerance");
}

}  // namespace

int main() {
    try {
        const int tokens = 5;
        const int n_local_experts = 4;
        const int topk = 2;
        const int dim = 64;
        const int inter_dim = 128;
        const float swiglu_limit = 7.0f;

        std::mt19937 rng(0xc0ffee);

        std::uniform_real_distribution<float> xdist(-0.5f, 0.5f);
        std::vector<float> x(static_cast<size_t>(tokens) * dim);
        for (auto& v : x) v = xdist(rng);

        std::vector<std::vector<uint8_t>> w1q(n_local_experts), w1s(n_local_experts);
        std::vector<std::vector<uint8_t>> w3q(n_local_experts), w3s(n_local_experts);
        std::vector<std::vector<uint8_t>> w2q(n_local_experts), w2s(n_local_experts);
        for (int e = 0; e < n_local_experts; ++e) {
            w1q[e] = random_fp4_weight(inter_dim, dim, rng);
            w1s[e] = random_fp4_scale(inter_dim, dim, rng);
            w3q[e] = random_fp4_weight(inter_dim, dim, rng);
            w3s[e] = random_fp4_scale(inter_dim, dim, rng);
            w2q[e] = random_fp4_weight(dim, inter_dim, rng);
            w2s[e] = random_fp4_scale(dim, inter_dim, rng);
        }

        const size_t w1_elem = static_cast<size_t>(inter_dim) * (dim / 2);
        const size_t w1s_elem = static_cast<size_t>(inter_dim) * (dim / 32);
        const size_t w2_elem = static_cast<size_t>(dim) * (inter_dim / 2);
        const size_t w2s_elem = static_cast<size_t>(dim) * (inter_dim / 32);

        std::vector<uint8_t> packed_w1q(static_cast<size_t>(n_local_experts) * w1_elem);
        std::vector<uint8_t> packed_w1s(static_cast<size_t>(n_local_experts) * w1s_elem);
        std::vector<uint8_t> packed_w3q(static_cast<size_t>(n_local_experts) * w1_elem);
        std::vector<uint8_t> packed_w3s(static_cast<size_t>(n_local_experts) * w1s_elem);
        std::vector<uint8_t> packed_w2q(static_cast<size_t>(n_local_experts) * w2_elem);
        std::vector<uint8_t> packed_w2s(static_cast<size_t>(n_local_experts) * w2s_elem);
        for (int e = 0; e < n_local_experts; ++e) {
            std::memcpy(packed_w1q.data() + static_cast<size_t>(e) * w1_elem, w1q[e].data(), w1_elem);
            std::memcpy(packed_w1s.data() + static_cast<size_t>(e) * w1s_elem, w1s[e].data(), w1s_elem);
            std::memcpy(packed_w3q.data() + static_cast<size_t>(e) * w1_elem, w3q[e].data(), w1_elem);
            std::memcpy(packed_w3s.data() + static_cast<size_t>(e) * w1s_elem, w3s[e].data(), w1s_elem);
            std::memcpy(packed_w2q.data() + static_cast<size_t>(e) * w2_elem, w2q[e].data(), w2_elem);
            std::memcpy(packed_w2s.data() + static_cast<size_t>(e) * w2s_elem, w2s[e].data(), w2s_elem);
        }

        std::uniform_real_distribution<float> wdist(0.05f, 0.45f);
        std::vector<std::vector<int>> token_routes(tokens);
        std::vector<std::vector<float>> token_weights(tokens);
        std::vector<std::vector<int>> expert_routes(n_local_experts);
        std::vector<std::vector<float>> expert_token_weights(n_local_experts);
        std::vector<std::vector<int64_t>> expert_ordinals(n_local_experts);
        for (int t = 0; t < tokens; ++t) {
            std::vector<int> available;
            for (int e = 0; e < n_local_experts; ++e) available.push_back(e);
            std::shuffle(available.begin(), available.end(), rng);
            for (int k = 0; k < topk; ++k) {
                const int expert = available[k];
                const float w = wdist(rng);
                token_routes[t].push_back(expert);
                token_weights[t].push_back(w);
                expert_routes[expert].push_back(t);
                expert_token_weights[expert].push_back(w);
                expert_ordinals[expert].push_back(static_cast<int64_t>(t));
            }
        }

        std::vector<int32_t> seg_starts(n_local_experts + 1, 0);
        for (int e = 0; e < n_local_experts; ++e) seg_starts[e + 1] = seg_starts[e] + static_cast<int32_t>(expert_routes[e].size());
        const int routes = seg_starts[n_local_experts];
        std::vector<int64_t> route_tokens(routes);
        std::vector<float> route_weights(routes);
        int max_count = 0;
        for (int e = 0; e < n_local_experts; ++e) {
            const int start = seg_starts[e];
            for (size_t i = 0; i < expert_routes[e].size(); ++i) {
                route_tokens[start + i] = expert_ordinals[e][i];
                route_weights[start + i] = expert_token_weights[e][i];
            }
            max_count = std::max(max_count, static_cast<int>(expert_routes[e].size()));
        }
        if (max_count == 0) throw std::runtime_error("max_count=0");

        float* d_x = nullptr;
        float* d_y = nullptr;
        int64_t* d_route_tokens = nullptr;
        float* d_route_weights = nullptr;
        int32_t* d_seg_starts = nullptr;
        uint8_t* d_w1q = nullptr;
        uint8_t* d_w1s = nullptr;
        uint8_t* d_w3q = nullptr;
        uint8_t* d_w3s = nullptr;
        uint8_t* d_w2q = nullptr;
        uint8_t* d_w2s = nullptr;
        check_cuda(cudaMalloc(&d_x, x.size() * sizeof(float)), "malloc x");
        check_cuda(cudaMalloc(&d_y, x.size() * sizeof(float)), "malloc y");
        check_cuda(cudaMalloc(&d_route_tokens, route_tokens.size() * sizeof(int64_t)), "malloc routes");
        check_cuda(cudaMalloc(&d_route_weights, route_weights.size() * sizeof(float)), "malloc weights");
        check_cuda(cudaMalloc(&d_seg_starts, seg_starts.size() * sizeof(int32_t)), "malloc seg");
        check_cuda(cudaMalloc(&d_w1q, packed_w1q.size()), "malloc w1q");
        check_cuda(cudaMalloc(&d_w1s, packed_w1s.size()), "malloc w1s");
        check_cuda(cudaMalloc(&d_w3q, packed_w3q.size()), "malloc w3q");
        check_cuda(cudaMalloc(&d_w3s, packed_w3s.size()), "malloc w3s");
        check_cuda(cudaMalloc(&d_w2q, packed_w2q.size()), "malloc w2q");
        check_cuda(cudaMalloc(&d_w2s, packed_w2s.size()), "malloc w2s");

        check_cuda(cudaMemcpy(d_x, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice), "copy x");
        check_cuda(cudaMemcpy(d_route_tokens, route_tokens.data(), route_tokens.size() * sizeof(int64_t), cudaMemcpyHostToDevice), "copy routes");
        check_cuda(cudaMemcpy(d_route_weights, route_weights.data(), route_weights.size() * sizeof(float), cudaMemcpyHostToDevice), "copy weights");
        check_cuda(cudaMemcpy(d_seg_starts, seg_starts.data(), seg_starts.size() * sizeof(int32_t), cudaMemcpyHostToDevice), "copy seg");
        check_cuda(cudaMemcpy(d_w1q, packed_w1q.data(), packed_w1q.size(), cudaMemcpyHostToDevice), "copy w1q");
        check_cuda(cudaMemcpy(d_w1s, packed_w1s.data(), packed_w1s.size(), cudaMemcpyHostToDevice), "copy w1s");
        check_cuda(cudaMemcpy(d_w3q, packed_w3q.data(), packed_w3q.size(), cudaMemcpyHostToDevice), "copy w3q");
        check_cuda(cudaMemcpy(d_w3s, packed_w3s.data(), packed_w3s.size(), cudaMemcpyHostToDevice), "copy w3s");
        check_cuda(cudaMemcpy(d_w2q, packed_w2q.data(), packed_w2q.size(), cudaMemcpyHostToDevice), "copy w2q");
        check_cuda(cudaMemcpy(d_w2s, packed_w2s.data(), packed_w2s.size(), cudaMemcpyHostToDevice), "copy w2s");

        if (!pocket::moe_prefill_fp4_grouped_cuda(
                d_x,
                d_route_tokens,
                d_route_weights,
                d_seg_starts,
                d_w1q,
                d_w1s,
                d_w2q,
                d_w2s,
                d_w3q,
                d_w3s,
                d_y,
                tokens,
                topk,
                routes,
                n_local_experts,
                max_count,
                dim,
                inter_dim,
                swiglu_limit)) {
            throw std::runtime_error("moe_prefill_fp4_grouped_cuda failed");
        }
        check_cuda(cudaDeviceSynchronize(), "sync grouped");
        std::vector<float> grouped(x.size());
        check_cuda(cudaMemcpy(grouped.data(), d_y, grouped.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy grouped");

        std::vector<float> per_token(x.size(), 0.0f);
        float* d_x_token = nullptr;
        float* d_y_token = nullptr;
        int64_t* d_indices = nullptr;
        float* d_weights = nullptr;
        check_cuda(cudaMalloc(&d_x_token, dim * sizeof(float)), "malloc x_token");
        check_cuda(cudaMalloc(&d_y_token, dim * sizeof(float)), "malloc y_token");
        check_cuda(cudaMalloc(&d_indices, topk * sizeof(int64_t)), "malloc indices");
        check_cuda(cudaMalloc(&d_weights, topk * sizeof(float)), "malloc weights");
        for (int t = 0; t < tokens; ++t) {
            std::vector<int64_t> idx(token_routes[t].begin(), token_routes[t].end());
            std::vector<float> wgt = token_weights[t];
            check_cuda(cudaMemcpy(d_x_token, x.data() + t * dim, dim * sizeof(float), cudaMemcpyHostToDevice), "copy x_token");
            check_cuda(cudaMemcpy(d_indices, idx.data(), idx.size() * sizeof(int64_t), cudaMemcpyHostToDevice), "copy idx");
            check_cuda(cudaMemcpy(d_weights, wgt.data(), wgt.size() * sizeof(float), cudaMemcpyHostToDevice), "copy wgt");
            if (!pocket::moe_single_token_fp4_cuda(
                    d_x_token,
                    d_indices,
                    d_weights,
                    d_w1q,
                    d_w1s,
                    d_w2q,
                    d_w2s,
                    d_w3q,
                    d_w3s,
                    d_y_token,
                    topk,
                    0,
                    n_local_experts,
                    dim,
                    inter_dim,
                    swiglu_limit)) {
                throw std::runtime_error("single token launch failed");
            }
            check_cuda(cudaDeviceSynchronize(), "sync single token");
            check_cuda(cudaMemcpy(per_token.data() + t * dim, d_y_token, dim * sizeof(float), cudaMemcpyDeviceToHost), "copy y_token");
        }

        compare(grouped, per_token, 5e-2f, "grouped_vs_per_token");

        cudaFree(d_x); cudaFree(d_y); cudaFree(d_route_tokens); cudaFree(d_route_weights); cudaFree(d_seg_starts);
        cudaFree(d_w1q); cudaFree(d_w1s); cudaFree(d_w3q); cudaFree(d_w3s); cudaFree(d_w2q); cudaFree(d_w2s);
        cudaFree(d_x_token); cudaFree(d_y_token); cudaFree(d_indices); cudaFree(d_weights);

        std::cout << "[PASS] moe_grouped_fp4 tokens=" << tokens
                  << " routes=" << routes
                  << " max_count=" << max_count
                  << " n_local_experts=" << n_local_experts << "\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "[FAIL] " << ex.what() << "\n";
        return 1;
    }
}
