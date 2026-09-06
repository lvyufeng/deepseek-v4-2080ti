// Active-slot small-batch FP4 MoE agreement with repeated single-token calls.

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

template <typename T>
struct DeviceBuffer {
    T* ptr = nullptr;
    DeviceBuffer() = default;
    explicit DeviceBuffer(size_t count) {
        if (count != 0) check_cuda(cudaMalloc(&ptr, count * sizeof(T)), "cudaMalloc");
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer() { if (ptr != nullptr) cudaFree(ptr); }
};

template <typename T>
void copy_to_device(DeviceBuffer<T>& dst, const std::vector<T>& src) {
    if (!src.empty()) {
        check_cuda(cudaMemcpy(dst.ptr, src.data(), src.size() * sizeof(T),
                              cudaMemcpyHostToDevice), "copy to device");
    }
}

struct RouteLayout {
    std::vector<int32_t> experts;
    std::vector<int32_t> starts;
    std::vector<int32_t> tokens;
    std::vector<float> weights;
};

RouteLayout build_routes(const std::vector<int64_t>& indices,
                         const std::vector<float>& weights,
                         int tokens, int topk, int n_local) {
    RouteLayout out;
    out.starts.push_back(0);
    for (int expert = 0; expert < n_local; ++expert) {
        bool active = false;
        for (int token = 0; token < tokens; ++token) {
            for (int k = 0; k < topk; ++k) {
                const int idx = token * topk + k;
                if (indices[idx] != expert) continue;
                if (!active) {
                    out.experts.push_back(expert);
                    active = true;
                }
                out.tokens.push_back(token);
                out.weights.push_back(weights[idx]);
            }
        }
        if (active) out.starts.push_back(static_cast<int32_t>(out.tokens.size()));
    }
    return out;
}

struct CaseBuffers {
    DeviceBuffer<float> x, y, ref, weights, pair_weights;
    DeviceBuffer<int64_t> indices;
    DeviceBuffer<int32_t> slot_expert, slot_starts, slot_tokens;
    DeviceBuffer<uint8_t> w1q, w1s, w2q, w2s, w3q, w3s;
    DeviceBuffer<int8_t> x_q, hidden_q;
    DeviceBuffer<float> x_scale, gate, up, hidden_scale, partials;

    CaseBuffers(int tokens, int topk, int slots, int pairs, int n_local,
                int dim, int inter)
        : x(static_cast<size_t>(tokens) * dim),
          y(static_cast<size_t>(tokens) * dim),
          ref(static_cast<size_t>(tokens) * dim),
          weights(static_cast<size_t>(tokens) * topk),
          pair_weights(pairs),
          indices(static_cast<size_t>(tokens) * topk),
          slot_expert(slots), slot_starts(static_cast<size_t>(slots) + 1),
          slot_tokens(pairs),
          w1q(static_cast<size_t>(n_local) * inter * (dim / 2)),
          w1s(static_cast<size_t>(n_local) * inter * (dim / 32)),
          w2q(static_cast<size_t>(n_local) * dim * (inter / 2)),
          w2s(static_cast<size_t>(n_local) * dim * (inter / 32)),
          w3q(static_cast<size_t>(n_local) * inter * (dim / 2)),
          w3s(static_cast<size_t>(n_local) * inter * (dim / 32)),
          x_q(static_cast<size_t>(tokens) * dim), x_scale(tokens),
          gate(static_cast<size_t>(pairs) * inter),
          up(static_cast<size_t>(pairs) * inter),
          hidden_q(static_cast<size_t>(pairs) * inter), hidden_scale(pairs),
          partials(static_cast<size_t>(pairs) * dim) {}
};

void fill_bytes(uint8_t* dst, size_t count, std::mt19937& rng, bool scale) {
    std::vector<uint8_t> host(count);
    if (scale) {
        std::uniform_int_distribution<int> dist(120, 135);
        for (auto& v : host) v = static_cast<uint8_t>(dist(rng));
    } else {
        for (auto& v : host) v = static_cast<uint8_t>(rng() & 0xff);
    }
    check_cuda(cudaMemcpy(dst, host.data(), host.size(), cudaMemcpyHostToDevice),
               "copy random bytes");
}

void compare(const std::vector<float>& got, const std::vector<float>& want,
             float rel_limit, const std::string& label) {
    float max_abs = 0.0f;
    float max_rel = 0.0f;
    for (size_t i = 0; i < got.size(); ++i) {
        const float diff = std::fabs(got[i] - want[i]);
        max_abs = std::max(max_abs, diff);
        max_rel = std::max(max_rel, diff / std::max(std::fabs(want[i]), 1.0e-9f));
    }
    std::cout << label << " max_abs=" << max_abs << " max_rel=" << max_rel << "\n";
    if (max_rel > rel_limit) {
        throw std::runtime_error(label + ": relative error exceeds bound");
    }
}

void run_case(int tokens, int topk, int n_local, int dim, int inter,
              int seed, float rel_limit, bool all_experts) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> xdist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> wdist(0.1f, 1.1f);
    std::vector<float> x(static_cast<size_t>(tokens) * dim);
    for (auto& v : x) v = xdist(rng);
    std::vector<int64_t> indices(static_cast<size_t>(tokens) * topk);
    std::vector<float> weights(indices.size());
    for (int token = 0; token < tokens; ++token) {
        std::vector<int> experts(n_local);
        for (int e = 0; e < n_local; ++e) experts[e] = e;
        if (!all_experts) std::shuffle(experts.begin(), experts.end(), rng);
        for (int k = 0; k < topk; ++k) {
            indices[token * topk + k] = experts[k];
            weights[token * topk + k] = wdist(rng);
        }
    }
    const RouteLayout routes = build_routes(indices, weights, tokens, topk, n_local);
    const int slots = static_cast<int>(routes.experts.size());
    const int pairs = static_cast<int>(routes.tokens.size());
    CaseBuffers b(tokens, topk, slots, pairs, n_local, dim, inter);
    copy_to_device(b.x, x);
    copy_to_device(b.indices, indices);
    copy_to_device(b.weights, weights);
    copy_to_device(b.slot_expert, routes.experts);
    copy_to_device(b.slot_starts, routes.starts);
    copy_to_device(b.slot_tokens, routes.tokens);
    copy_to_device(b.pair_weights, routes.weights);
    fill_bytes(b.w1q.ptr, static_cast<size_t>(n_local) * inter * (dim / 2), rng, false);
    fill_bytes(b.w1s.ptr, static_cast<size_t>(n_local) * inter * (dim / 32), rng, true);
    fill_bytes(b.w2q.ptr, static_cast<size_t>(n_local) * dim * (inter / 2), rng, false);
    fill_bytes(b.w2s.ptr, static_cast<size_t>(n_local) * dim * (inter / 32), rng, true);
    fill_bytes(b.w3q.ptr, static_cast<size_t>(n_local) * inter * (dim / 2), rng, false);
    fill_bytes(b.w3s.ptr, static_cast<size_t>(n_local) * inter * (dim / 32), rng, true);

    pocket::MoeMultiTokenFp4Workspace ws;
    ws.d_x_q = b.x_q.ptr;
    ws.d_x_scale = b.x_scale.ptr;
    ws.d_gate = b.gate.ptr;
    ws.d_up = b.up.ptr;
    ws.d_hidden_q = b.hidden_q.ptr;
    ws.d_hidden_scale = b.hidden_scale.ptr;
    ws.d_partials = b.partials.ptr;
    ws.tokens_cap = tokens;
    ws.pairs_cap = pairs;
    ws.dim = dim;
    ws.inter_dim = inter;
    if (!pocket::moe_multi_token_fp4_cuda_with_workspace(
            b.x.ptr, b.slot_expert.ptr, b.slot_starts.ptr, b.slot_tokens.ptr,
            b.pair_weights.ptr, b.w1q.ptr, b.w1s.ptr, b.w2q.ptr, b.w2s.ptr,
            b.w3q.ptr, b.w3s.ptr, b.y.ptr, tokens, slots, pairs, n_local,
            dim, inter, 7.0f, ws)) {
        throw std::runtime_error("multi-token launch failed");
    }
    check_cuda(cudaDeviceSynchronize(), "sync multi-token");

    for (int token = 0; token < tokens; ++token) {
        if (!pocket::moe_single_token_fp4_cuda(
                b.x.ptr + static_cast<size_t>(token) * dim,
                b.indices.ptr + static_cast<size_t>(token) * topk,
                b.weights.ptr + static_cast<size_t>(token) * topk,
                b.w1q.ptr, b.w1s.ptr, b.w2q.ptr, b.w2s.ptr, b.w3q.ptr, b.w3s.ptr,
                b.ref.ptr + static_cast<size_t>(token) * dim, topk, 0, n_local,
                dim, inter, 7.0f)) {
            throw std::runtime_error("single-token launch failed");
        }
    }
    check_cuda(cudaDeviceSynchronize(), "sync reference");
    std::vector<float> got(static_cast<size_t>(tokens) * dim);
    std::vector<float> want(got.size());
    check_cuda(cudaMemcpy(got.data(), b.y.ptr, got.size() * sizeof(float),
                          cudaMemcpyDeviceToHost), "copy multi output");
    check_cuda(cudaMemcpy(want.data(), b.ref.ptr, want.size() * sizeof(float),
                          cudaMemcpyDeviceToHost), "copy reference output");
    compare(got, want, rel_limit,
            "tokens=" + std::to_string(tokens) + " topk=" + std::to_string(topk) +
            " dim=" + std::to_string(dim));

    for (int iter = 0; iter < 3; ++iter) {
        if (!pocket::moe_multi_token_fp4_cuda_with_workspace(
                b.x.ptr, b.slot_expert.ptr, b.slot_starts.ptr, b.slot_tokens.ptr,
                b.pair_weights.ptr, b.w1q.ptr, b.w1s.ptr, b.w2q.ptr, b.w2s.ptr,
                b.w3q.ptr, b.w3s.ptr, b.y.ptr, tokens, slots, pairs, n_local,
                dim, inter, 7.0f, ws)) {
            throw std::runtime_error("repeat multi-token launch failed");
        }
        check_cuda(cudaDeviceSynchronize(), "sync repeated multi-token");
        std::vector<float> repeated(got.size());
        check_cuda(cudaMemcpy(repeated.data(), b.y.ptr, repeated.size() * sizeof(float),
                              cudaMemcpyDeviceToHost), "copy repeated output");
        if (std::memcmp(got.data(), repeated.data(), got.size() * sizeof(float)) != 0) {
            throw std::runtime_error("multi-token result is not bitwise reproducible");
        }
    }
}

void test_empty_routes() {
    constexpr int tokens = 3;
    constexpr int dim = 256;
    DeviceBuffer<float> x(static_cast<size_t>(tokens) * dim);
    DeviceBuffer<float> y(static_cast<size_t>(tokens) * dim);
    DeviceBuffer<int32_t> starts(1);
    DeviceBuffer<uint8_t> dummy(1);
    DeviceBuffer<int8_t> x_q(static_cast<size_t>(tokens) * dim);
    DeviceBuffer<float> x_scale(tokens);
    check_cuda(cudaMemset(starts.ptr, 0, sizeof(int32_t)), "zero starts");
    pocket::MoeMultiTokenFp4Workspace ws;
    ws.d_x_q = x_q.ptr;
    ws.d_x_scale = x_scale.ptr;
    ws.tokens_cap = tokens;
    ws.pairs_cap = 0;
    ws.dim = dim;
    ws.inter_dim = 128;
    if (!pocket::moe_multi_token_fp4_cuda_with_workspace(
            x.ptr, nullptr, starts.ptr, nullptr, nullptr, dummy.ptr, dummy.ptr,
            dummy.ptr, dummy.ptr, dummy.ptr, dummy.ptr, y.ptr, tokens, 0, 0, 8,
            dim, 128, 7.0f, ws)) {
        throw std::runtime_error("empty-route launch failed");
    }
    std::vector<float> host(static_cast<size_t>(tokens) * dim, 1.0f);
    check_cuda(cudaMemcpy(host.data(), y.ptr, host.size() * sizeof(float),
                          cudaMemcpyDeviceToHost), "copy empty output");
    if (std::any_of(host.begin(), host.end(), [](float v) { return v != 0.0f; })) {
        throw std::runtime_error("empty routes did not produce zero rows");
    }
}

}  // namespace

int main() {
    try {
        for (int tokens : {1, 2, 3, 6}) {
            run_case(tokens, 3, 8, 256, 128, 1000 + tokens, 4.0e-4f, false);
        }
        run_case(10, 8, 8, 256, 128, 77, 4.0e-3f, true);
        test_empty_routes();
        run_case(2, 6, 64, 4096, 2048, 4002, 8.0e-3f, false);
        std::cout << "[PASS] small-batch FP4 MoE\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "[FAIL] " << ex.what() << "\n";
        return 1;
    }
}
