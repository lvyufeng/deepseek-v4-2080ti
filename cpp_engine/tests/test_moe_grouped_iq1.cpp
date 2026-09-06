#include "cuda_ops.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
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

uint16_t float_to_half_bits(float x) {
    __half h = __float2half(x);
    return *reinterpret_cast<uint16_t*>(&h);
}

std::vector<uint8_t> random_iq1_weight(int rows, int cols, std::mt19937& rng) {
    const int blocks_per_row = (cols + 255) / 256;
    std::vector<uint8_t> w(static_cast<size_t>(rows) * blocks_per_row * 56);
    std::uniform_int_distribution<int> byte_dist(0, 255);
    std::uniform_int_distribution<int> qh_dist(0, 127);  // keep delta positive for stability
    std::uniform_int_distribution<int> sc_dist(0, 7);
    std::uniform_real_distribution<float> d_dist(0.01f, 0.04f);
    for (int r = 0; r < rows; ++r) {
        for (int b = 0; b < blocks_per_row; ++b) {
            uint8_t* block = w.data() + (static_cast<size_t>(r) * blocks_per_row + b) * 56;
            for (int i = 0; i < 32; ++i) block[i] = static_cast<uint8_t>(byte_dist(rng));
            for (int i = 0; i < 16; ++i) block[32 + i] = static_cast<uint8_t>(qh_dist(rng));
            uint16_t sc[4] = {0, 0, 0, 0};
            for (int s = 0; s < 8; ++s) {
                sc[s >> 2] |= static_cast<uint16_t>(sc_dist(rng) << ((s & 3) * 3));
            }
            const uint16_t d_bits = float_to_half_bits(d_dist(rng));
            sc[0] |= static_cast<uint16_t>((d_bits & 0x000fu) << 12);
            sc[1] |= static_cast<uint16_t>(d_bits & 0x00f0u) << 8;
            sc[2] |= static_cast<uint16_t>(d_bits & 0x0f00u) << 4;
            sc[3] |= static_cast<uint16_t>(d_bits & 0xf000u);
            std::memcpy(block + 48, sc, sizeof(sc));
        }
    }
    return w;
}

int env_int_or_default(const char* name, int fallback) {
    const char* v = std::getenv(name);
    if (v == nullptr || *v == '\0') return fallback;
    char* end = nullptr;
    const long parsed = std::strtol(v, &end, 10);
    return (end == v) ? fallback : static_cast<int>(parsed);
}

void compare(const std::vector<float>& a, const std::vector<float>& b, float tol, const std::string& tag) {
    if (a.size() != b.size()) throw std::runtime_error(tag + ": size mismatch");
    float max_abs = 0.0f;
    float mean_abs = 0.0f;
    int dump = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        const float diff = std::fabs(a[i] - b[i]);
        max_abs = std::max(max_abs, diff);
        mean_abs += diff;
        if (diff > tol && dump < 8) {
            std::cerr << tag << " idx=" << i << " grouped=" << a[i]
                      << " ref=" << b[i] << " diff=" << diff << "\n";
            ++dump;
        }
    }
    mean_abs /= static_cast<float>(a.size());
    std::cout << tag << " max_abs=" << max_abs << " mean_abs=" << mean_abs
              << " size=" << a.size() << "\n";
    if (max_abs > tol) throw std::runtime_error(tag + ": max_abs above tolerance");
}

}  // namespace

int main() {
    try {
        const int tokens = 16;
        const int n_local_experts = 4;
        const int topk = 2;
        const int dim = 256;
        const int inter_dim = 256;
        const float swiglu_limit = 7.0f;
        std::mt19937 rng(0x1f00d);

        std::uniform_real_distribution<float> xdist(-0.25f, 0.25f);
        std::vector<float> x(static_cast<size_t>(tokens) * dim);
        for (auto& v : x) v = xdist(rng);

        const size_t w13_bytes_per_expert = static_cast<size_t>(inter_dim) * ((dim + 255) / 256) * 56;
        const size_t w2_bytes_per_expert = static_cast<size_t>(dim) * ((inter_dim + 255) / 256) * 56;
        std::vector<uint8_t> w1(static_cast<size_t>(n_local_experts) * w13_bytes_per_expert);
        std::vector<uint8_t> w3(static_cast<size_t>(n_local_experts) * w13_bytes_per_expert);
        std::vector<uint8_t> w2(static_cast<size_t>(n_local_experts) * w2_bytes_per_expert);
        for (int e = 0; e < n_local_experts; ++e) {
            auto ew1 = random_iq1_weight(inter_dim, dim, rng);
            auto ew3 = random_iq1_weight(inter_dim, dim, rng);
            auto ew2 = random_iq1_weight(dim, inter_dim, rng);
            std::memcpy(w1.data() + static_cast<size_t>(e) * w13_bytes_per_expert, ew1.data(), w13_bytes_per_expert);
            std::memcpy(w3.data() + static_cast<size_t>(e) * w13_bytes_per_expert, ew3.data(), w13_bytes_per_expert);
            std::memcpy(w2.data() + static_cast<size_t>(e) * w2_bytes_per_expert, ew2.data(), w2_bytes_per_expert);
        }

        std::uniform_real_distribution<float> wdist(0.05f, 0.45f);
        std::vector<std::vector<int>> token_routes(tokens);
        std::vector<std::vector<float>> token_weights(tokens);
        std::vector<std::vector<int64_t>> expert_tokens(n_local_experts);
        std::vector<std::vector<float>> expert_weights(n_local_experts);
        for (int t = 0; t < tokens; ++t) {
            std::vector<int> experts;
            for (int e = 0; e < n_local_experts; ++e) experts.push_back(e);
            std::shuffle(experts.begin(), experts.end(), rng);
            for (int k = 0; k < topk; ++k) {
                const int e = experts[k];
                const float weight = wdist(rng);
                token_routes[t].push_back(e);
                token_weights[t].push_back(weight);
                expert_tokens[e].push_back(t);
                expert_weights[e].push_back(weight);
            }
        }

        std::vector<int32_t> seg_starts(n_local_experts + 1, 0);
        for (int e = 0; e < n_local_experts; ++e) {
            seg_starts[e + 1] = seg_starts[e] + static_cast<int32_t>(expert_tokens[e].size());
        }
        const int routes = seg_starts[n_local_experts];
        int max_count = 0;
        std::vector<int64_t> route_tokens(routes);
        std::vector<float> route_weights(routes);
        for (int e = 0; e < n_local_experts; ++e) {
            max_count = std::max(max_count, static_cast<int>(expert_tokens[e].size()));
            const int start = seg_starts[e];
            for (size_t i = 0; i < expert_tokens[e].size(); ++i) {
                route_tokens[start + static_cast<int>(i)] = expert_tokens[e][i];
                route_weights[start + static_cast<int>(i)] = expert_weights[e][i];
            }
        }

        std::vector<int32_t> tile_experts;
        std::vector<int32_t> tile_rows;
        for (int e = 0; e < n_local_experts; ++e) {
            const int count = static_cast<int>(expert_tokens[e].size());
            for (int row = 0; row < count; row += 16) {
                tile_experts.push_back(e);
                tile_rows.push_back(row);
            }
        }

        float *d_x = nullptr, *d_y = nullptr, *d_hidden = nullptr, *d_hidden_scale = nullptr, *d_x_scale = nullptr;
        int8_t *d_hidden_q = nullptr, *d_x_q = nullptr;
        int32_t *d_tile_experts = nullptr, *d_tile_rows = nullptr;
        int64_t* d_route_tokens = nullptr;
        float* d_route_weights = nullptr;
        int32_t* d_seg_starts = nullptr;
        uint8_t *d_w1 = nullptr, *d_w2 = nullptr, *d_w3 = nullptr;
        check_cuda(cudaMalloc(&d_x, x.size() * sizeof(float)), "malloc x");
        check_cuda(cudaMalloc(&d_y, x.size() * sizeof(float)), "malloc y");
        check_cuda(cudaMalloc(&d_hidden, static_cast<size_t>(routes) * inter_dim * sizeof(float)), "malloc hidden");
        if (env_int_or_default("POCKETLLM_IQ1_GROUPED_W2_Q8", 1) != 0) {
            check_cuda(cudaMalloc(&d_hidden_q, static_cast<size_t>(routes) * inter_dim), "malloc hidden q");
            check_cuda(cudaMalloc(&d_hidden_scale, static_cast<size_t>(routes) * ((inter_dim + 15) / 16) * sizeof(float)), "malloc hidden scale");
        }
        if (env_int_or_default("POCKETLLM_IQ1_GROUPED_GEMM", 1) != 0) {
            check_cuda(cudaMalloc(&d_x_q, static_cast<size_t>(routes) * dim), "malloc x q");
            check_cuda(cudaMalloc(&d_x_scale, static_cast<size_t>(routes) * ((dim + 15) / 16) * sizeof(float)), "malloc x scale");
            check_cuda(cudaMalloc(&d_tile_experts, static_cast<size_t>(tile_experts.size()) * sizeof(int32_t)), "malloc tile experts");
            check_cuda(cudaMalloc(&d_tile_rows, static_cast<size_t>(tile_rows.size()) * sizeof(int32_t)), "malloc tile rows");
        }
        check_cuda(cudaMalloc(&d_route_tokens, route_tokens.size() * sizeof(int64_t)), "malloc route tokens");
        check_cuda(cudaMalloc(&d_route_weights, route_weights.size() * sizeof(float)), "malloc route weights");
        check_cuda(cudaMalloc(&d_seg_starts, seg_starts.size() * sizeof(int32_t)), "malloc seg starts");
        check_cuda(cudaMalloc(&d_w1, w1.size()), "malloc w1");
        check_cuda(cudaMalloc(&d_w2, w2.size()), "malloc w2");
        check_cuda(cudaMalloc(&d_w3, w3.size()), "malloc w3");
        check_cuda(cudaMemcpy(d_x, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice), "copy x");
        check_cuda(cudaMemcpy(d_route_tokens, route_tokens.data(), route_tokens.size() * sizeof(int64_t), cudaMemcpyHostToDevice), "copy route tokens");
        check_cuda(cudaMemcpy(d_route_weights, route_weights.data(), route_weights.size() * sizeof(float), cudaMemcpyHostToDevice), "copy route weights");
        check_cuda(cudaMemcpy(d_seg_starts, seg_starts.data(), seg_starts.size() * sizeof(int32_t), cudaMemcpyHostToDevice), "copy seg starts");
        if (d_tile_experts != nullptr && !tile_experts.empty()) {
            check_cuda(cudaMemcpy(d_tile_experts, tile_experts.data(), tile_experts.size() * sizeof(int32_t), cudaMemcpyHostToDevice), "copy tile experts");
            check_cuda(cudaMemcpy(d_tile_rows, tile_rows.data(), tile_rows.size() * sizeof(int32_t), cudaMemcpyHostToDevice), "copy tile rows");
        }
        check_cuda(cudaMemcpy(d_w1, w1.data(), w1.size(), cudaMemcpyHostToDevice), "copy w1");
        check_cuda(cudaMemcpy(d_w2, w2.data(), w2.size(), cudaMemcpyHostToDevice), "copy w2");
        check_cuda(cudaMemcpy(d_w3, w3.data(), w3.size(), cudaMemcpyHostToDevice), "copy w3");

        pocket::MoePrefillIq1GroupedWorkspace ws;
        ws.d_hidden = d_hidden;
        ws.d_hidden_q = d_hidden_q;
        ws.d_hidden_scale = d_hidden_scale;
        ws.d_x_q = d_x_q;
        ws.d_x_scale = d_x_scale;
        ws.d_tile_experts = d_tile_experts;
        ws.d_tile_rows = d_tile_rows;
        ws.routes_cap = routes;
        ws.tile_cap = static_cast<int>(tile_experts.size());
        ws.tile_count = static_cast<int>(tile_experts.size());
        ws.dim = dim;
        ws.inter_dim = inter_dim;
        if (!pocket::moe_prefill_iq1_grouped_cuda_with_workspace(
                d_x, d_route_tokens, d_route_weights, d_seg_starts,
                d_w1, d_w2, d_w3, d_y,
                tokens, routes, n_local_experts, max_count,
                dim, inter_dim, swiglu_limit, ws)) {
            throw std::runtime_error("grouped iq1 launch failed");
        }
        check_cuda(cudaDeviceSynchronize(), "sync grouped");
        std::vector<float> grouped(x.size());
        check_cuda(cudaMemcpy(grouped.data(), d_y, grouped.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy grouped");

        std::vector<float> ref(x.size(), 0.0f);
        float *d_x_token = nullptr, *d_y_token = nullptr, *d_hidden_token = nullptr;
        int64_t* d_slots = nullptr;
        float* d_weights = nullptr;
        check_cuda(cudaMalloc(&d_x_token, dim * sizeof(float)), "malloc x token");
        check_cuda(cudaMalloc(&d_y_token, dim * sizeof(float)), "malloc y token");
        check_cuda(cudaMalloc(&d_hidden_token, topk * inter_dim * sizeof(float)), "malloc hidden token");
        check_cuda(cudaMalloc(&d_slots, topk * sizeof(int64_t)), "malloc slots");
        check_cuda(cudaMalloc(&d_weights, topk * sizeof(float)), "malloc weights");
        for (int t = 0; t < tokens; ++t) {
            std::vector<int64_t> slots(token_routes[t].begin(), token_routes[t].end());
            check_cuda(cudaMemcpy(d_x_token, x.data() + static_cast<size_t>(t) * dim, dim * sizeof(float), cudaMemcpyHostToDevice), "copy x token");
            check_cuda(cudaMemcpy(d_slots, slots.data(), slots.size() * sizeof(int64_t), cudaMemcpyHostToDevice), "copy slots");
            check_cuda(cudaMemcpy(d_weights, token_weights[t].data(), token_weights[t].size() * sizeof(float), cudaMemcpyHostToDevice), "copy weights");
            check_cuda(cudaMemset(d_y_token, 0, dim * sizeof(float)), "zero y token");
            if (!pocket::iq1_moe_single_w13_swiglu_cuda(
                    d_x_token, d_slots, d_weights, d_w1, d_w3, d_hidden_token,
                    topk, n_local_experts, dim, inter_dim, swiglu_limit)) {
                throw std::runtime_error("single w13 launch failed");
            }
            if (!pocket::iq1_moe_single_w2_cuda(
                    d_hidden_token, d_slots, d_w2, d_y_token,
                    topk, n_local_experts, dim, inter_dim)) {
                throw std::runtime_error("single w2 launch failed");
            }
            check_cuda(cudaDeviceSynchronize(), "sync single");
            check_cuda(cudaMemcpy(ref.data() + static_cast<size_t>(t) * dim, d_y_token, dim * sizeof(float), cudaMemcpyDeviceToHost), "copy y token");
        }

        const bool q8_w2 = env_int_or_default("POCKETLLM_IQ1_GROUPED_W2_Q8", 1) != 0;
        const bool gemm = env_int_or_default("POCKETLLM_IQ1_GROUPED_GEMM", 1) != 0;
        compare(grouped, ref, gemm ? 8e-2f : (q8_w2 ? 5e-3f : 2e-4f), "iq1_grouped_vs_single");

        cudaFree(d_x); cudaFree(d_y); cudaFree(d_hidden); cudaFree(d_hidden_q); cudaFree(d_hidden_scale); cudaFree(d_x_q); cudaFree(d_x_scale); cudaFree(d_tile_experts); cudaFree(d_tile_rows); cudaFree(d_route_tokens); cudaFree(d_route_weights); cudaFree(d_seg_starts);
        cudaFree(d_w1); cudaFree(d_w2); cudaFree(d_w3);
        cudaFree(d_x_token); cudaFree(d_y_token); cudaFree(d_hidden_token); cudaFree(d_slots); cudaFree(d_weights);

        std::cout << "[PASS] moe_grouped_iq1 tokens=" << tokens
                  << " routes=" << routes
                  << " max_count=" << max_count
                  << " n_local_experts=" << n_local_experts << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[FAIL] " << e.what() << "\n";
        return 1;
    }
}
