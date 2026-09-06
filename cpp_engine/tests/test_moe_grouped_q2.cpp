#include "cuda_ops.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

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

uint16_t float_to_half_bits(float x) {
    __half h = __float2half(x);
    return *reinterpret_cast<uint16_t*>(&h);
}

std::vector<uint8_t> random_iq2_xxs_weight(int rows, int cols, std::mt19937& rng) {
    const int blocks_per_row = (cols + 255) / 256;
    std::vector<uint8_t> w(static_cast<size_t>(rows) * blocks_per_row * 66);
    std::uniform_int_distribution<int> grid_dist(0, 255);
    std::uniform_int_distribution<int> sign_dist(0, 127);
    std::uniform_int_distribution<int> high_dist(0, 1);
    std::uniform_real_distribution<float> d_dist(0.005f, 0.025f);
    for (int r = 0; r < rows; ++r) {
        for (int b = 0; b < blocks_per_row; ++b) {
            uint8_t* block = w.data() + (static_cast<size_t>(r) * blocks_per_row + b) * 66;
            const uint16_t d_bits = float_to_half_bits(d_dist(rng));
            block[0] = static_cast<uint8_t>(d_bits & 0xffu);
            block[1] = static_cast<uint8_t>(d_bits >> 8);
            for (int sub = 0; sub < 8; ++sub) {
                uint8_t* chunk = block + 2 + sub * 8;
                for (int part = 0; part < 4; ++part) {
                    chunk[part] = static_cast<uint8_t>(grid_dist(rng));
                }
                uint32_t aux = static_cast<uint32_t>(high_dist(rng)) << 28;
                for (int part = 0; part < 4; ++part) {
                    aux |= static_cast<uint32_t>(sign_dist(rng)) << (7 * part);
                }
                chunk[4] = static_cast<uint8_t>(aux & 0xffu);
                chunk[5] = static_cast<uint8_t>((aux >> 8) & 0xffu);
                chunk[6] = static_cast<uint8_t>((aux >> 16) & 0xffu);
                chunk[7] = static_cast<uint8_t>((aux >> 24) & 0xffu);
            }
        }
    }
    return w;
}

std::vector<uint8_t> random_q2_k_weight(int rows, int cols, std::mt19937& rng) {
    const int blocks_per_row = (cols + 255) / 256;
    std::vector<uint8_t> w(static_cast<size_t>(rows) * blocks_per_row * 84);
    std::uniform_int_distribution<int> q_dist(0, 255);
    std::uniform_int_distribution<int> qscale_dist(1, 7);
    std::uniform_int_distribution<int> minscale_dist(0, 2);
    std::uniform_real_distribution<float> d_dist(0.001f, 0.01f);
    std::uniform_real_distribution<float> dmin_dist(0.0f, 0.002f);
    for (int r = 0; r < rows; ++r) {
        for (int b = 0; b < blocks_per_row; ++b) {
            uint8_t* block = w.data() + (static_cast<size_t>(r) * blocks_per_row + b) * 84;
            for (int i = 0; i < 16; ++i) {
                block[i] = static_cast<uint8_t>(qscale_dist(rng) | (minscale_dist(rng) << 4));
            }
            for (int i = 0; i < 64; ++i) block[16 + i] = static_cast<uint8_t>(q_dist(rng));
            const uint16_t d_bits = float_to_half_bits(d_dist(rng));
            const uint16_t dmin_bits = float_to_half_bits(dmin_dist(rng));
            block[80] = static_cast<uint8_t>(d_bits & 0xffu);
            block[81] = static_cast<uint8_t>(d_bits >> 8);
            block[82] = static_cast<uint8_t>(dmin_bits & 0xffu);
            block[83] = static_cast<uint8_t>(dmin_bits >> 8);
        }
    }
    return w;
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
        const int tokens = 5;
        const int n_local_experts = 4;
        const int topk = 2;
        const int dim = 256;
        const int inter_dim = 256;
        const float swiglu_limit = 7.0f;
        std::mt19937 rng(0x2202);

        std::uniform_real_distribution<float> xdist(-0.25f, 0.25f);
        std::vector<float> x(static_cast<size_t>(tokens) * dim);
        for (auto& v : x) v = xdist(rng);

        const size_t w13_bytes_per_expert = static_cast<size_t>(inter_dim) * ((dim + 255) / 256) * 66;
        const size_t w2_bytes_per_expert = static_cast<size_t>(dim) * ((inter_dim + 255) / 256) * 84;
        std::vector<uint8_t> w1(static_cast<size_t>(n_local_experts) * w13_bytes_per_expert);
        std::vector<uint8_t> w3(static_cast<size_t>(n_local_experts) * w13_bytes_per_expert);
        std::vector<uint8_t> w2(static_cast<size_t>(n_local_experts) * w2_bytes_per_expert);
        for (int e = 0; e < n_local_experts; ++e) {
            auto ew1 = random_iq2_xxs_weight(inter_dim, dim, rng);
            auto ew3 = random_iq2_xxs_weight(inter_dim, dim, rng);
            auto ew2 = random_q2_k_weight(dim, inter_dim, rng);
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

        float *d_x = nullptr, *d_y = nullptr, *d_x_route = nullptr, *d_x_scale = nullptr;
        float *d_gate = nullptr, *d_up = nullptr, *d_hidden_scale = nullptr;
        int8_t *d_x_q = nullptr, *d_hidden_q = nullptr;
        int64_t *d_route_tokens = nullptr, *d_route_slots = nullptr;
        float* d_route_weights = nullptr;
        int32_t* d_seg_starts = nullptr;
        uint8_t *d_w1 = nullptr, *d_w2 = nullptr, *d_w3 = nullptr;
        const int x_groups = (dim + 31) / 32;
        const int hidden_groups = (inter_dim + 15) / 16;

        check_cuda(cudaMalloc(&d_x, x.size() * sizeof(float)), "malloc x");
        check_cuda(cudaMalloc(&d_y, x.size() * sizeof(float)), "malloc y");
        check_cuda(cudaMalloc(&d_x_route, static_cast<size_t>(routes) * dim * sizeof(float)), "malloc x route");
        check_cuda(cudaMalloc(&d_x_q, static_cast<size_t>(routes) * dim * sizeof(int8_t)), "malloc x q");
        check_cuda(cudaMalloc(&d_x_scale, static_cast<size_t>(routes) * x_groups * sizeof(float)), "malloc x scale");
        check_cuda(cudaMalloc(&d_route_slots, static_cast<size_t>(routes) * sizeof(int64_t)), "malloc route slots");
        check_cuda(cudaMalloc(&d_gate, static_cast<size_t>(routes) * inter_dim * sizeof(float)), "malloc gate");
        check_cuda(cudaMalloc(&d_up, static_cast<size_t>(routes) * inter_dim * sizeof(float)), "malloc up");
        check_cuda(cudaMalloc(&d_hidden_q, static_cast<size_t>(routes) * inter_dim * sizeof(int8_t)), "malloc hidden q");
        check_cuda(cudaMalloc(&d_hidden_scale, static_cast<size_t>(routes) * hidden_groups * sizeof(float)), "malloc hidden scale");
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
        check_cuda(cudaMemcpy(d_w1, w1.data(), w1.size(), cudaMemcpyHostToDevice), "copy w1");
        check_cuda(cudaMemcpy(d_w2, w2.data(), w2.size(), cudaMemcpyHostToDevice), "copy w2");
        check_cuda(cudaMemcpy(d_w3, w3.data(), w3.size(), cudaMemcpyHostToDevice), "copy w3");

        pocket::MoePrefillQ2GroupedWorkspace ws;
        ws.d_x_route = d_x_route;
        ws.d_x_q = d_x_q;
        ws.d_x_scale = d_x_scale;
        ws.d_route_slots = d_route_slots;
        ws.d_gate = d_gate;
        ws.d_up = d_up;
        ws.d_hidden_q = d_hidden_q;
        ws.d_hidden_scale = d_hidden_scale;
        ws.routes_cap = routes;
        ws.dim = dim;
        ws.inter_dim = inter_dim;
        if (!pocket::moe_prefill_q2_grouped_cuda_with_workspace(
                d_x, d_route_tokens, d_route_weights, d_seg_starts,
                d_w1, d_w2, d_w3, d_y,
                tokens, routes, n_local_experts, max_count,
                dim, inter_dim, swiglu_limit, ws)) {
            throw std::runtime_error("grouped q2 launch failed");
        }
        check_cuda(cudaDeviceSynchronize(), "sync grouped");
        std::vector<float> grouped(x.size());
        check_cuda(cudaMemcpy(grouped.data(), d_y, grouped.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy grouped");

        std::vector<float> ref(x.size(), 0.0f);
        float *d_x_token = nullptr, *d_y_token = nullptr, *d_x_scale_token = nullptr;
        float *d_gate_token = nullptr, *d_up_token = nullptr, *d_hidden_scale_token = nullptr;
        int8_t *d_x_q_token = nullptr, *d_hidden_q_token = nullptr;
        int64_t* d_slots = nullptr;
        float* d_weights = nullptr;
        check_cuda(cudaMalloc(&d_x_token, dim * sizeof(float)), "malloc x token");
        check_cuda(cudaMalloc(&d_y_token, dim * sizeof(float)), "malloc y token");
        check_cuda(cudaMalloc(&d_x_q_token, dim * sizeof(int8_t)), "malloc x q token");
        check_cuda(cudaMalloc(&d_x_scale_token, x_groups * sizeof(float)), "malloc x scale token");
        check_cuda(cudaMalloc(&d_gate_token, topk * inter_dim * sizeof(float)), "malloc gate token");
        check_cuda(cudaMalloc(&d_up_token, topk * inter_dim * sizeof(float)), "malloc up token");
        check_cuda(cudaMalloc(&d_hidden_q_token, topk * inter_dim * sizeof(int8_t)), "malloc hidden q token");
        check_cuda(cudaMalloc(&d_hidden_scale_token, topk * hidden_groups * sizeof(float)), "malloc hidden scale token");
        check_cuda(cudaMalloc(&d_slots, topk * sizeof(int64_t)), "malloc slots");
        check_cuda(cudaMalloc(&d_weights, topk * sizeof(float)), "malloc weights");
        for (int t = 0; t < tokens; ++t) {
            std::vector<int64_t> slots(token_routes[t].begin(), token_routes[t].end());
            check_cuda(cudaMemcpy(d_x_token, x.data() + static_cast<size_t>(t) * dim, dim * sizeof(float), cudaMemcpyHostToDevice), "copy x token");
            check_cuda(cudaMemcpy(d_slots, slots.data(), slots.size() * sizeof(int64_t), cudaMemcpyHostToDevice), "copy slots");
            check_cuda(cudaMemcpy(d_weights, token_weights[t].data(), token_weights[t].size() * sizeof(float), cudaMemcpyHostToDevice), "copy weights");
            check_cuda(cudaMemset(d_y_token, 0, dim * sizeof(float)), "zero y token");
            if (!pocket::q2_quantize_x_q8_1_cuda(d_x_token, d_x_q_token, d_x_scale_token, 1, dim)) {
                throw std::runtime_error("single x quant launch failed");
            }
            if (!pocket::q2_moe_single_w13_iq2_xxs_cuda(
                    d_x_q_token, d_x_scale_token, d_slots, d_w1, d_w3,
                    d_gate_token, d_up_token,
                    topk, n_local_experts, dim, inter_dim)) {
                throw std::runtime_error("single w13 launch failed");
            }
            if (!pocket::q2_route_swiglu_quantize_hidden_q8_1_cuda(
                    d_gate_token, d_up_token, d_weights, d_hidden_q_token, d_hidden_scale_token,
                    topk, inter_dim, swiglu_limit)) {
                throw std::runtime_error("single swiglu quant launch failed");
            }
            if (!pocket::q2_moe_single_w2_q2k_cuda(
                    d_hidden_q_token, d_hidden_scale_token, d_slots, d_w2, d_y_token,
                    topk, n_local_experts, dim, inter_dim)) {
                throw std::runtime_error("single w2 launch failed");
            }
            check_cuda(cudaDeviceSynchronize(), "sync single");
            check_cuda(cudaMemcpy(ref.data() + static_cast<size_t>(t) * dim, d_y_token, dim * sizeof(float), cudaMemcpyDeviceToHost), "copy y token");
        }

        compare(grouped, ref, 2e-4f, "q2_grouped_vs_single");

        cudaFree(d_x); cudaFree(d_y); cudaFree(d_x_route); cudaFree(d_x_q); cudaFree(d_x_scale); cudaFree(d_route_slots);
        cudaFree(d_gate); cudaFree(d_up); cudaFree(d_hidden_q); cudaFree(d_hidden_scale);
        cudaFree(d_route_tokens); cudaFree(d_route_weights); cudaFree(d_seg_starts);
        cudaFree(d_w1); cudaFree(d_w2); cudaFree(d_w3);
        cudaFree(d_x_token); cudaFree(d_y_token); cudaFree(d_x_q_token); cudaFree(d_x_scale_token);
        cudaFree(d_gate_token); cudaFree(d_up_token); cudaFree(d_hidden_q_token); cudaFree(d_hidden_scale_token);
        cudaFree(d_slots); cudaFree(d_weights);

        std::cout << "[PASS] moe_grouped_q2 tokens=" << tokens
                  << " routes=" << routes
                  << " max_count=" << max_count
                  << " n_local_experts=" << n_local_experts << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "[FAIL] " << e.what() << "\n";
        return 1;
    }
}
