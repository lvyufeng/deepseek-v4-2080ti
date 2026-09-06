// Numerical check for the Gated DeltaNet recurrent step against a CPU
// reference, including multi-step state continuation.
//
// The kernel was rewritten for speed (cooperative q/k L2 norm in shared memory
// plus a single-pass state update). A zero-weight engine fixture cannot catch a
// regression here, so this uses random inputs and verifies the evolving state.

#include "cuda_ops.hpp"
#include "qwen_ops.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

namespace {

int failures = 0;

void fail(const std::string& what) {
    std::printf("[FAIL] %s\n", what.c_str());
    ++failures;
}

// One Gated DeltaNet step:
//   S <- S * exp(g); delta = (v - S^T k_hat) * beta; S += k_hat delta^T
//   out = (S^T q_hat) * q_scale
void delta_step_reference(std::vector<float>& state, const std::vector<float>& q,
                          const std::vector<float>& k, const std::vector<float>& v,
                          const std::vector<float>& g, const std::vector<float>& beta,
                          std::vector<float>& out, int heads, int key_heads,
                          int key_dim, int value_dim, float q_scale) {
    const int repeat = heads / key_heads;
    for (int head = 0; head < heads; ++head) {
        const int key_head = head / repeat;
        const float* q_head = q.data() + static_cast<size_t>(key_head) * key_dim;
        const float* k_head = k.data() + static_cast<size_t>(key_head) * key_dim;
        double q_sq = 0.0, k_sq = 0.0;
        for (int i = 0; i < key_dim; ++i) {
            q_sq += static_cast<double>(q_head[i]) * q_head[i];
            k_sq += static_cast<double>(k_head[i]) * k_head[i];
        }
        const float q_norm = static_cast<float>(1.0 / std::sqrt(q_sq + 1.0e-6));
        const float k_norm = static_cast<float>(1.0 / std::sqrt(k_sq + 1.0e-6));
        float* state_head = state.data() + static_cast<size_t>(head) * key_dim * value_dim;
        const float decay = std::exp(g[head]);
        for (int value = 0; value < value_dim; ++value) {
            float kv_mem = 0.0f;
            for (int i = 0; i < key_dim; ++i) {
                float& cell = state_head[static_cast<size_t>(i) * value_dim + value];
                cell *= decay;
                kv_mem += cell * k_head[i] * k_norm;
            }
            const float delta =
                (v[static_cast<size_t>(head) * value_dim + value] - kv_mem) * beta[head];
            float result = 0.0f;
            for (int i = 0; i < key_dim; ++i) {
                float& cell = state_head[static_cast<size_t>(i) * value_dim + value];
                cell += k_head[i] * k_norm * delta;
                result += cell * q_head[i] * q_norm;
            }
            out[static_cast<size_t>(head) * value_dim + value] = result * q_scale;
        }
    }
}

struct Device {
    float* state = nullptr;
    float* q = nullptr;
    float* k = nullptr;
    float* v = nullptr;
    float* g = nullptr;
    float* beta = nullptr;
    float* out = nullptr;
    ~Device() {
        cudaFree(state);
        cudaFree(q);
        cudaFree(k);
        cudaFree(v);
        cudaFree(g);
        cudaFree(beta);
        cudaFree(out);
    }
};

bool check_delta_rule() {
    // 48 value heads over 16 key heads matches the Qwen3.8 TP1 layout; the
    // 12-over-4 shape here keeps the same repeat factor of 3 at test size.
    const int heads = 12;
    const int key_heads = 4;
    const int key_dim = 128;
    const int value_dim = 128;
    const float q_scale = 0.08838834764f;  // 128^-0.5
    const int steps = 4;

    std::mt19937 rng(987654321);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> gate(-0.4f, -0.02f);
    std::uniform_real_distribution<float> beta_dist(0.05f, 0.95f);

    const size_t state_size = static_cast<size_t>(heads) * key_dim * value_dim;
    std::vector<float> ref_state(state_size);
    for (float& x : ref_state) x = dist(rng) * 0.05f;

    Device dev;
    const size_t qk_size = static_cast<size_t>(key_heads) * key_dim;
    const size_t out_size = static_cast<size_t>(heads) * value_dim;
    if (cudaMalloc(&dev.state, state_size * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dev.q, qk_size * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dev.k, qk_size * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dev.v, out_size * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dev.g, heads * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dev.beta, heads * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dev.out, out_size * sizeof(float)) != cudaSuccess) {
        return false;
    }
    cudaMemcpy(dev.state, ref_state.data(), state_size * sizeof(float), cudaMemcpyHostToDevice);

    double worst_out = 0.0;
    double worst_state = 0.0;
    for (int step = 0; step < steps; ++step) {
        std::vector<float> q(qk_size), k(qk_size), v(out_size);
        std::vector<float> g(heads), beta(heads);
        for (float& x : q) x = dist(rng);
        for (float& x : k) x = dist(rng);
        for (float& x : v) x = dist(rng);
        for (float& x : g) x = gate(rng);
        for (float& x : beta) x = beta_dist(rng);

        cudaMemcpy(dev.q, q.data(), q.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dev.k, k.data(), k.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dev.v, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dev.g, g.data(), g.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(dev.beta, beta.data(), beta.size() * sizeof(float), cudaMemcpyHostToDevice);

        if (!pocket::qwen_gated_delta_step_cuda(dev.state, dev.q, dev.k, dev.v, dev.g,
                                              dev.beta, dev.out, heads, key_heads,
                                              key_dim, value_dim, q_scale)) {
            fail("gated_delta_step launch failed");
            return true;
        }
        if (cudaDeviceSynchronize() != cudaSuccess) {
            fail("gated_delta_step sync failed");
            return true;
        }

        std::vector<float> got_out(out_size);
        std::vector<float> got_state(state_size);
        cudaMemcpy(got_out.data(), dev.out, got_out.size() * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(got_state.data(), dev.state, got_state.size() * sizeof(float), cudaMemcpyDeviceToHost);

        std::vector<float> want_out(out_size);
        delta_step_reference(ref_state, q, k, v, g, beta, want_out, heads, key_heads,
                             key_dim, value_dim, q_scale);

        for (size_t i = 0; i < out_size; ++i) {
            worst_out = std::max(worst_out, static_cast<double>(std::fabs(got_out[i] - want_out[i])));
        }
        for (size_t i = 0; i < state_size; ++i) {
            worst_state = std::max(worst_state,
                                   static_cast<double>(std::fabs(got_state[i] - ref_state[i])));
        }
    }

    // The kernel reorders the fp32 reductions relative to the scalar reference,
    // so exact equality is not expected; 2e-4 still catches a wrong norm, a
    // dropped decay, or a stale state cell.
    if (worst_out > 2.0e-4 || worst_state > 2.0e-4) {
        fail("delta_rule drift out=" + std::to_string(worst_out) +
             " state=" + std::to_string(worst_state));
    } else {
        std::printf("  delta_rule steps=%d heads=%d worst_out=%.3e worst_state=%.3e\n",
                    steps, heads, worst_out, worst_state);
    }
    return true;
}

// The prefill sequence kernel must reproduce the step kernel run token by
// token: same outputs, and the same final state so a following decode continues
// from an identical recurrence.
bool check_delta_sequence() {
    const int heads = 12;
    const int key_heads = 4;
    const int key_dim = 128;
    const int value_dim = 128;
    const float q_scale = 0.08838834764f;
    const int rows = 7;  // deliberately not a multiple of anything

    std::mt19937 rng(24681357);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> gate(-0.4f, -0.02f);
    std::uniform_real_distribution<float> beta_dist(0.05f, 0.95f);

    const size_t state_size = static_cast<size_t>(heads) * key_dim * value_dim;
    const size_t qk_row = static_cast<size_t>(key_heads) * key_dim;
    const size_t out_row = static_cast<size_t>(heads) * value_dim;
    std::vector<float> init_state(state_size);
    for (float& x : init_state) x = dist(rng) * 0.05f;
    std::vector<float> q(qk_row * rows), k(qk_row * rows), v(out_row * rows);
    std::vector<float> g(static_cast<size_t>(heads) * rows);
    std::vector<float> beta(g.size());
    for (float& x : q) x = dist(rng);
    for (float& x : k) x = dist(rng);
    for (float& x : v) x = dist(rng);
    for (float& x : g) x = gate(rng);
    for (float& x : beta) x = beta_dist(rng);

    float *d_state = nullptr, *d_q = nullptr, *d_k = nullptr, *d_v = nullptr;
    float *d_g = nullptr, *d_beta = nullptr, *d_out = nullptr;
    if (cudaMalloc(&d_state, state_size * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_q, q.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_k, k.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_v, v.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_g, g.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_beta, beta.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_out, out_row * rows * sizeof(float)) != cudaSuccess) {
        return false;
    }
    cudaMemcpy(d_q, q.data(), q.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, k.data(), k.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_g, g.data(), g.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, beta.data(), beta.size() * sizeof(float), cudaMemcpyHostToDevice);

    auto run_steps = [&](std::vector<float>& out, std::vector<float>& final_state) {
        cudaMemcpy(d_state, init_state.data(), state_size * sizeof(float), cudaMemcpyHostToDevice);
        for (int t = 0; t < rows; ++t) {
            if (!pocket::qwen_gated_delta_step_cuda(
                    d_state, d_q + static_cast<size_t>(t) * qk_row,
                    d_k + static_cast<size_t>(t) * qk_row,
                    d_v + static_cast<size_t>(t) * out_row,
                    d_g + static_cast<size_t>(t) * heads,
                    d_beta + static_cast<size_t>(t) * heads,
                    d_out + static_cast<size_t>(t) * out_row, heads, key_heads,
                    key_dim, value_dim, q_scale)) {
                return false;
            }
        }
        if (cudaDeviceSynchronize() != cudaSuccess) return false;
        cudaMemcpy(out.data(), d_out, out.size() * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(final_state.data(), d_state, final_state.size() * sizeof(float),
                   cudaMemcpyDeviceToHost);
        return true;
    };

    std::vector<float> step_out(out_row * rows), step_state(state_size);
    std::vector<float> seq_out(out_row * rows), seq_state(state_size);
    bool ok = run_steps(step_out, step_state);
    if (ok) {
        cudaMemcpy(d_state, init_state.data(), state_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemset(d_out, 0, seq_out.size() * sizeof(float));
        ok = pocket::qwen_gated_delta_sequence_cuda(d_state, d_q, d_k, d_v, d_g, d_beta, d_out,
                                                  rows, heads, key_heads, key_dim, value_dim,
                                                  q_scale);
        if (!ok) fail("gated_delta_sequence launch failed");
        if (ok && cudaDeviceSynchronize() != cudaSuccess) {
            fail("gated_delta_sequence sync failed");
            ok = false;
        }
        if (ok) {
            cudaMemcpy(seq_out.data(), d_out, seq_out.size() * sizeof(float), cudaMemcpyDeviceToHost);
            cudaMemcpy(seq_state.data(), d_state, seq_state.size() * sizeof(float),
                       cudaMemcpyDeviceToHost);
            double worst_out = 0.0;
            double worst_state = 0.0;
            for (size_t i = 0; i < seq_out.size(); ++i) {
                worst_out = std::max(worst_out,
                                     static_cast<double>(std::fabs(seq_out[i] - step_out[i])));
            }
            for (size_t i = 0; i < seq_state.size(); ++i) {
                worst_state = std::max(worst_state,
                                       static_cast<double>(std::fabs(seq_state[i] - step_state[i])));
            }
            // Both kernels perform the same operations in the same order, so the
            // only permitted difference is FMA contraction inside one token.
            if (worst_out > 1.0e-5 || worst_state > 1.0e-5) {
                fail("delta_sequence drift out=" + std::to_string(worst_out) +
                     " state=" + std::to_string(worst_state));
            } else {
                std::printf("  delta_sequence rows=%d heads=%d worst_out=%.3e worst_state=%.3e\n",
                            rows, heads, worst_out, worst_state);
            }
        }
    } else {
        fail("gated_delta_step reference loop failed");
    }

    cudaFree(d_state);
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_g);
    cudaFree(d_beta);
    cudaFree(d_out);
    return true;
}

}  // namespace

int main() {
    if (!pocket::cuda_runtime_available()) {
        std::printf("[SKIP] test_qwen_delta_rule requires a CUDA device\n");
        return 0;
    }
    if (!check_delta_rule()) {
        std::printf("[SKIP] device allocation failed\n");
        return 0;
    }
    if (!check_delta_sequence()) {
        std::printf("[SKIP] device allocation failed for sequence check\n");
        return 0;
    }
    if (failures != 0) {
        std::printf("test_qwen_delta_rule failures=%d\n", failures);
        return 1;
    }
    std::printf("[PASS] test_qwen_delta_rule\n");
    return 0;
}
