#include "cuda_ops.hpp"
#include "qwen_ops.hpp"

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace {

int failures = 0;

void fail(const char* message) {
    std::printf("[FAIL] %s\n", message);
    ++failures;
}

uint16_t to_half(float value) {
    return __half_as_ushort(__float2half(value));
}

float from_half(uint16_t bits) {
    return __half2float(__ushort_as_half(bits));
}

float from_fp8_e4m3(uint8_t code) {
    const uint32_t sign = static_cast<uint32_t>(code & 0x80u) << 24;
    const uint32_t exponent = (code >> 3) & 0xfu;
    const uint32_t mantissa = code & 0x7u;
    uint32_t bits;
    if (exponent != 0) {
        bits = sign | ((exponent + 120u) << 23) | (mantissa << 20);
    } else if (mantissa >= 4) {
        bits = sign | (120u << 23) | ((mantissa - 4u) << 21);
    } else if (mantissa >= 2) {
        bits = sign | (119u << 23) | ((mantissa - 2u) << 22);
    } else {
        bits = sign | (mantissa == 0 ? 0u : (118u << 23));
    }
    float output;
    std::memcpy(&output, &bits, sizeof(output));
    return output;
}

struct DeviceBuffer {
    void* data = nullptr;
    ~DeviceBuffer() { cudaFree(data); }
    template <typename T>
    T* allocate(size_t count) {
        if (cudaMalloc(&data, count * sizeof(T)) != cudaSuccess) return nullptr;
        return static_cast<T*>(data);
    }
};

bool check_fp16_cache(int context_len) {
    constexpr int q_heads = 6;
    constexpr int kv_heads = 1;
    constexpr int head_dim = 64;
    const int max_context = context_len + 3;
    std::mt19937 rng(1234 + context_len);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<uint16_t> q(q_heads * head_dim);
    std::vector<uint16_t> k(context_len * kv_heads * head_dim);
    std::vector<uint16_t> v(k.size());
    for (uint16_t& item : q) item = to_half(dist(rng));
    for (uint16_t& item : k) item = to_half(dist(rng));
    for (uint16_t& item : v) item = to_half(dist(rng));
    DeviceBuffer dq, dk_rows, dv_rows, dk_cache, dv_cache, dout, dscores;
    uint16_t* d_q = dq.allocate<uint16_t>(q.size());
    uint16_t* d_k_rows = dk_rows.allocate<uint16_t>(k.size());
    uint16_t* d_v_rows = dv_rows.allocate<uint16_t>(v.size());
    uint16_t* d_k_cache = dk_cache.allocate<uint16_t>(max_context * kv_heads * head_dim);
    uint16_t* d_v_cache = dv_cache.allocate<uint16_t>(max_context * kv_heads * head_dim);
    uint16_t* d_out = dout.allocate<uint16_t>(q.size());
    float* d_scores = dscores.allocate<float>(q_heads * context_len);
    if (!d_q || !d_k_rows || !d_v_rows || !d_k_cache || !d_v_cache || !d_out || !d_scores) return false;
    cudaMemcpy(d_q, q.data(), q.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k_rows, k.data(), k.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v_rows, v.data(), v.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
    if (!pocket::qwen_append_kv_cache_f16(d_k_rows, d_v_rows, d_k_cache, d_v_cache,
                                              context_len, kv_heads, head_dim, 0, max_context) ||
        !pocket::qwen_gqa_decode_attention_f16(d_q, d_k_cache, d_v_cache, d_out, d_scores,
                                                   q_heads, kv_heads, head_dim, context_len,
                                                   max_context) || cudaDeviceSynchronize() != cudaSuccess) {
        fail("FP16 cache launch");
        return true;
    }
    std::vector<uint16_t> got(q.size());
    cudaMemcpy(got.data(), d_out, got.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    double worst = 0.0;
    for (int head = 0; head < q_heads; ++head) {
        std::vector<double> scores(context_len);
        double maximum = -1.0e300;
        for (int pos = 0; pos < context_len; ++pos) {
            double dot = 0.0;
            for (int d = 0; d < head_dim; ++d) {
                dot += static_cast<double>(from_half(q[head * head_dim + d])) *
                       from_half(k[pos * head_dim + d]);
            }
            scores[pos] = dot * scale;
            maximum = std::max(maximum, scores[pos]);
        }
        double denominator = 0.0;
        for (double& score : scores) {
            score = std::exp(score - maximum);
            denominator += score;
        }
        for (int d = 0; d < head_dim; ++d) {
            double value = 0.0;
            for (int pos = 0; pos < context_len; ++pos) value += scores[pos] * from_half(v[pos * head_dim + d]);
            const float expected = static_cast<float>(value / denominator);
            worst = std::max(worst, std::fabs(static_cast<double>(from_half(got[head * head_dim + d])) - expected));
        }
    }
    if (worst > 3.0e-3) {
        fail("FP16 cache numerical check");
    } else {
        std::printf("  FP16 cache context=%d worst=%.3e\n", context_len, worst);
    }
    return true;
}

double max_half_difference(const std::vector<uint16_t>& lhs,
                          const std::vector<uint16_t>& rhs,
                          bool* finite) {
    if (lhs.size() != rhs.size()) {
        *finite = false;
        return INFINITY;
    }
    double worst = 0.0;
    *finite = true;
    for (size_t i = 0; i < lhs.size(); ++i) {
        const float left = from_half(lhs[i]);
        const float right = from_half(rhs[i]);
        if (!std::isfinite(left) || !std::isfinite(right)) *finite = false;
        worst = std::max(worst, std::fabs(static_cast<double>(left) - right));
    }
    return worst;
}

bool check_residual_add_rmsnorm_fused() {
    constexpr int cols = 5120;
    constexpr float eps = 1.0e-6f;
    const int row_cases[] = {2, 3, 5, 8};
    std::mt19937 rng(26391);
    std::uniform_real_distribution<float> hidden_dist(-2.0f, 2.0f);
    std::uniform_real_distribution<float> delta_dist(-0.5f, 0.5f);
    std::uniform_real_distribution<float> gamma_dist(-0.2f, 0.2f);
    for (int rows : row_cases) {
        const size_t elements = static_cast<size_t>(rows) * cols;
        std::vector<uint16_t> hidden(elements);
        std::vector<uint16_t> delta(elements);
        std::vector<uint16_t> gamma(cols);
        for (uint16_t& value : hidden) value = to_half(hidden_dist(rng));
        for (uint16_t& value : delta) value = to_half(delta_dist(rng));
        for (uint16_t& value : gamma) value = to_half(gamma_dist(rng));

        DeviceBuffer dhidden, ddelta, dgamma, dref_residual, dref_normalized,
            dfused_residual, dfused_normalized;
        uint16_t* d_hidden = dhidden.allocate<uint16_t>(elements);
        uint16_t* d_delta = ddelta.allocate<uint16_t>(elements);
        uint16_t* d_gamma = dgamma.allocate<uint16_t>(gamma.size());
        uint16_t* d_ref_residual = dref_residual.allocate<uint16_t>(elements);
        uint16_t* d_ref_normalized = dref_normalized.allocate<uint16_t>(elements);
        uint16_t* d_fused_residual = dfused_residual.allocate<uint16_t>(elements);
        uint16_t* d_fused_normalized = dfused_normalized.allocate<uint16_t>(elements);
        if (!d_hidden || !d_delta || !d_gamma || !d_ref_residual ||
            !d_ref_normalized || !d_fused_residual || !d_fused_normalized ||
            cudaMemcpy(d_hidden, hidden.data(), elements * sizeof(uint16_t),
                       cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(d_delta, delta.data(), elements * sizeof(uint16_t),
                       cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(d_gamma, gamma.data(), gamma.size() * sizeof(uint16_t),
                       cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(d_ref_residual, d_hidden, elements * sizeof(uint16_t),
                       cudaMemcpyDeviceToDevice) != cudaSuccess ||
            !pocket::qwen_add_inplace_f16(
                d_ref_residual, d_delta, static_cast<int>(elements)) ||
            !pocket::qwen_rmsnorm_fp16_gamma_rows_f16(
                d_ref_residual, d_gamma, d_ref_normalized, rows, cols, eps) ||
            !pocket::qwen_residual_add_rmsnorm_fp16_gamma_rows_f16(
                d_hidden, d_delta, d_gamma, d_fused_residual,
                d_fused_normalized, rows, cols, eps) ||
            cudaDeviceSynchronize() != cudaSuccess) {
            fail("FP16 fused residual RMSNorm launch");
            return true;
        }

        std::vector<uint16_t> ref_residual(elements);
        std::vector<uint16_t> ref_normalized(elements);
        std::vector<uint16_t> fused_residual(elements);
        std::vector<uint16_t> fused_normalized(elements);
        if (cudaMemcpy(ref_residual.data(), d_ref_residual,
                       elements * sizeof(uint16_t), cudaMemcpyDeviceToHost) !=
                cudaSuccess ||
            cudaMemcpy(ref_normalized.data(), d_ref_normalized,
                       elements * sizeof(uint16_t), cudaMemcpyDeviceToHost) !=
                cudaSuccess ||
            cudaMemcpy(fused_residual.data(), d_fused_residual,
                       elements * sizeof(uint16_t), cudaMemcpyDeviceToHost) !=
                cudaSuccess ||
            cudaMemcpy(fused_normalized.data(), d_fused_normalized,
                       elements * sizeof(uint16_t), cudaMemcpyDeviceToHost) !=
                cudaSuccess) {
            fail("FP16 fused residual RMSNorm copy");
            return true;
        }
        int residual_mismatches = 0;
        int normalized_mismatches = 0;
        for (size_t index = 0; index < elements; ++index) {
            residual_mismatches +=
                ref_residual[index] != fused_residual[index] ? 1 : 0;
            normalized_mismatches +=
                ref_normalized[index] != fused_normalized[index] ? 1 : 0;
        }
        if (residual_mismatches != 0 || normalized_mismatches != 0) {
            std::printf("  FP16 fused residual RMSNorm rows=%d cols=%d "
                        "residual_mismatch=%d normalized_mismatch=%d\n",
                        rows, cols, residual_mismatches,
                        normalized_mismatches);
            fail("FP16 fused residual RMSNorm bit exactness");
            return true;
        }
    }
    std::printf("  FP16 fused residual RMSNorm bit exact rows=2,3,5,8 "
                "cols=%d\n", cols);
    return true;
}

bool check_prefill_tiled(int seq_len, int head_dim, int position_offset,
                         int q_heads = 6, int kv_heads = 1) {

    const int max_context = position_offset + seq_len + 3;
    std::mt19937 rng(9000 + seq_len * 17 + head_dim + position_offset);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<uint16_t> q(static_cast<size_t>(seq_len) * q_heads * head_dim);
    std::vector<uint16_t> k(static_cast<size_t>(max_context) * kv_heads * head_dim);
    std::vector<uint16_t> v(k.size());
    for (uint16_t& item : q) item = to_half(dist(rng));
    for (uint16_t& item : k) item = to_half(dist(rng));
    for (uint16_t& item : v) item = to_half(dist(rng));

    DeviceBuffer dq, dk, dv, dold, dnew, dsparse;
    uint16_t* d_q = dq.allocate<uint16_t>(q.size());
    uint16_t* d_k = dk.allocate<uint16_t>(k.size());
    uint16_t* d_v = dv.allocate<uint16_t>(v.size());
    uint16_t* d_old = dold.allocate<uint16_t>(q.size());
    uint16_t* d_new = dnew.allocate<uint16_t>(q.size());
    uint16_t* d_sparse = dsparse.allocate<uint16_t>(q.size());
    if (!d_q || !d_k || !d_v || !d_old || !d_new || !d_sparse) return false;
    if (cudaMemcpy(d_q, q.data(), q.size() * sizeof(uint16_t), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_k, k.data(), k.size() * sizeof(uint16_t), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_v, v.data(), v.size() * sizeof(uint16_t), cudaMemcpyHostToDevice) != cudaSuccess ||
        !pocket::qwen_gqa_prefill_attention_f16(
            d_q, d_k, d_v, d_old, seq_len, q_heads, kv_heads, head_dim,
            position_offset, max_context) ||
        !pocket::qwen_gqa_prefill_attention_f16_tiled_cuda(
            d_q, d_k, d_v, d_new, seq_len, q_heads, kv_heads, head_dim,
            position_offset, max_context) ||
        !pocket::qwen_gqa_prefill_attention_f16_tiled_cuda(
            d_q, d_k, d_v, d_sparse, seq_len, q_heads, kv_heads, head_dim,
            position_offset, max_context, seq_len + position_offset + 8, 0) ||
        cudaDeviceSynchronize() != cudaSuccess) {
        fail("FP16 tiled prefill launch");
        return true;
    }
    std::vector<uint16_t> old_output(q.size());
    std::vector<uint16_t> new_output(q.size());
    std::vector<uint16_t> sparse_output(q.size());
    if (cudaMemcpy(old_output.data(), d_old, old_output.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(new_output.data(), d_new, new_output.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(sparse_output.data(), d_sparse, sparse_output.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost) != cudaSuccess) {
        fail("FP16 tiled prefill copy");
        return true;
    }
    bool finite = true;
    const double worst = max_half_difference(old_output, new_output, &finite);
    bool sparse_finite = true;
    const double sparse_worst = max_half_difference(old_output, sparse_output, &sparse_finite);
    if (!finite || worst > 4.0e-3 || !sparse_finite || sparse_worst > 4.0e-3) {
        fail("FP16 tiled prefill numerical check");
    } else {
        std::printf("  FP16 tiled prefill seq=%d dim=%d offset=%d worst=%.3e sparse_exact=%.3e\n",
                    seq_len, head_dim, position_offset, worst, sparse_worst);
    }
    return true;
}

bool check_verify_split(int seq_len, int head_dim, int position_offset) {
    constexpr int q_heads = 6;
    constexpr int kv_heads = 1;
    const int max_context = position_offset + seq_len + 3;
    const int splits = std::min(64, (position_offset + seq_len + 255) / 256);
    std::mt19937 rng(11000 + seq_len * 17 + head_dim + position_offset);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<uint16_t> q(static_cast<size_t>(seq_len) * q_heads * head_dim);
    std::vector<uint16_t> k(static_cast<size_t>(max_context) * kv_heads * head_dim);
    std::vector<uint16_t> v(k.size());
    for (uint16_t& item : q) item = to_half(dist(rng));
    for (uint16_t& item : k) item = to_half(dist(rng));
    for (uint16_t& item : v) item = to_half(dist(rng));

    DeviceBuffer dq, dk, dv, dref, dexact, dverify, dscores, dpartial;
    uint16_t* d_q = dq.allocate<uint16_t>(q.size());
    uint16_t* d_k = dk.allocate<uint16_t>(k.size());
    uint16_t* d_v = dv.allocate<uint16_t>(v.size());
    uint16_t* d_ref = dref.allocate<uint16_t>(q.size());
    uint16_t* d_exact = dexact.allocate<uint16_t>(q.size());
    uint16_t* d_verify = dverify.allocate<uint16_t>(q.size());
    float* d_scores = dscores.allocate<float>(
        static_cast<size_t>(seq_len) * q_heads * (position_offset + seq_len));
    float* d_partial = dpartial.allocate<float>(
        static_cast<size_t>(seq_len) * q_heads * splits * (head_dim + 2));
    if (!d_q || !d_k || !d_v || !d_ref || !d_exact || !d_verify ||
        !d_scores || !d_partial) return false;
    if (cudaMemcpy(d_q, q.data(), q.size() * sizeof(uint16_t), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_k, k.data(), k.size() * sizeof(uint16_t), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_v, v.data(), v.size() * sizeof(uint16_t), cudaMemcpyHostToDevice) != cudaSuccess ||
        !pocket::qwen_gqa_prefill_attention_f16(
            d_q, d_k, d_v, d_ref, seq_len, q_heads, kv_heads, head_dim,
            position_offset, max_context) ||
        !pocket::qwen_gqa_verify_attention_f16_exact_cuda(
            d_q, d_k, d_v, d_exact, d_scores, seq_len, q_heads, kv_heads,
            head_dim, position_offset, max_context) ||
        !pocket::qwen_gqa_verify_attention_f16(
            d_q, d_k, d_v, d_verify, d_partial, seq_len, q_heads, kv_heads,
            head_dim, position_offset, max_context, splits) ||
        cudaDeviceSynchronize() != cudaSuccess) {
        fail("FP16 split verify launch");
        return true;
    }
    std::vector<uint16_t> reference(q.size());
    std::vector<uint16_t> exact(q.size());
    std::vector<uint16_t> verify(q.size());
    if (cudaMemcpy(reference.data(), d_ref, reference.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(exact.data(), d_exact, exact.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(verify.data(), d_verify, verify.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost) != cudaSuccess) {
        fail("FP16 split verify copy");
        return true;
    }
    bool exact_finite = true;
    const double exact_worst = max_half_difference(reference, exact, &exact_finite);
    bool finite = true;
    const double worst = max_half_difference(reference, verify, &finite);
    if (!exact_finite || exact_worst > 4.0e-3) {
        fail("FP16 exact verify numerical check");
    } else if (!finite || worst > 4.0e-3) {
        fail("FP16 split verify numerical check");
    } else {
        std::printf("  FP16 verify seq=%d dim=%d offset=%d splits=%d exact=%.3e split=%.3e\n",
                    seq_len, head_dim, position_offset, splits, exact_worst, worst);
    }
    return true;
}

bool check_decode_fused(int context_len, int head_dim) {
    constexpr int q_heads = 6;
    constexpr int kv_heads = 1;
    std::mt19937 rng(12000 + context_len + head_dim);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<uint16_t> q(static_cast<size_t>(q_heads) * head_dim);
    std::vector<uint16_t> k(static_cast<size_t>(context_len) * kv_heads * head_dim);
    std::vector<uint16_t> v(k.size());
    for (uint16_t& item : q) item = to_half(dist(rng));
    for (uint16_t& item : k) item = to_half(dist(rng));
    for (uint16_t& item : v) item = to_half(dist(rng));

    DeviceBuffer dq, dk, dv, dold, dnew, dscores, dpartial, dsparse;
    uint16_t* d_q = dq.allocate<uint16_t>(q.size());
    uint16_t* d_k = dk.allocate<uint16_t>(k.size());
    uint16_t* d_v = dv.allocate<uint16_t>(k.size());
    uint16_t* d_old = dold.allocate<uint16_t>(q.size());
    uint16_t* d_new = dnew.allocate<uint16_t>(q.size());
    uint16_t* d_sparse = dsparse.allocate<uint16_t>(q.size());
    float* d_scores = dscores.allocate<float>(static_cast<size_t>(q_heads) * context_len);
    const int splits = std::min(64, (context_len + 2048 - 1) / 2048);
    float* d_partial = dpartial.allocate<float>(
        static_cast<size_t>(q_heads) * splits * (head_dim + 2));
    if (!d_q || !d_k || !d_v || !d_old || !d_new || !d_sparse ||
        !d_scores || !d_partial) return false;
    if (cudaMemcpy(d_q, q.data(), q.size() * sizeof(uint16_t), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_k, k.data(), k.size() * sizeof(uint16_t), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_v, v.data(), v.size() * sizeof(uint16_t), cudaMemcpyHostToDevice) != cudaSuccess ||
        !pocket::qwen_gqa_decode_attention_f16(
            d_q, d_k, d_v, d_old, d_scores, q_heads, kv_heads, head_dim,
            context_len, context_len) ||
        !pocket::qwen_gqa_decode_attention_f16_fused_cuda(
            d_q, d_k, d_v, d_new, d_partial, q_heads, kv_heads, head_dim,
            context_len, context_len) ||
        !pocket::qwen_gqa_decode_attention_f16_fused_cuda(
            d_q, d_k, d_v, d_sparse, d_partial, q_heads, kv_heads, head_dim,
            context_len, context_len, context_len + 8, 0) ||
        cudaDeviceSynchronize() != cudaSuccess) {
        fail("FP16 fused decode launch");
        return true;
    }
    std::vector<uint16_t> old_output(q.size());
    std::vector<uint16_t> new_output(q.size());
    std::vector<uint16_t> sparse_output(q.size());
    if (cudaMemcpy(old_output.data(), d_old, old_output.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(new_output.data(), d_new, new_output.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(sparse_output.data(), d_sparse, sparse_output.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost) != cudaSuccess) {
        fail("FP16 fused decode copy");
        return true;
    }
    bool finite = true;
    const double worst = max_half_difference(old_output, new_output, &finite);
    bool sparse_finite = true;
    const double sparse_worst = max_half_difference(old_output, sparse_output, &sparse_finite);
    if (!finite || worst > 4.0e-3 || !sparse_finite || sparse_worst > 4.0e-3) {
        std::printf("  FP16 fused decode context=%d dim=%d worst=%.3e finite=%d\n",
                    context_len, head_dim, worst, finite ? 1 : 0);
        fail("FP16 fused decode numerical check");
    } else {
        std::printf("  FP16 fused decode context=%d dim=%d worst=%.3e sparse_exact=%.3e\n",
                    context_len, head_dim, worst, sparse_worst);
    }
    return true;
}

bool check_decode_window_reference() {
    constexpr int q_heads = 6;
    constexpr int kv_heads = 1;
    constexpr int head_dim = 64;
    constexpr int context_len = 8192;
    constexpr int window = 257;
    constexpr int sink = 3;
    const int window_start = context_len - window;
    std::mt19937 rng(18000);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<uint16_t> q(static_cast<size_t>(q_heads) * head_dim);
    std::vector<uint16_t> k(static_cast<size_t>(context_len) * head_dim);
    std::vector<uint16_t> v(k.size());
    for (uint16_t& item : q) item = to_half(dist(rng));
    for (uint16_t& item : k) item = to_half(dist(rng));
    for (uint16_t& item : v) item = to_half(dist(rng));

    const int attended = sink + (context_len - window_start);
    DeviceBuffer dq, dk, dv, dout, dpartial;
    uint16_t* d_q = dq.allocate<uint16_t>(q.size());
    uint16_t* d_k = dk.allocate<uint16_t>(k.size());
    uint16_t* d_v = dv.allocate<uint16_t>(v.size());
    uint16_t* d_out = dout.allocate<uint16_t>(q.size());
    float* d_partial = dpartial.allocate<float>(
        static_cast<size_t>(q_heads) * (head_dim + 2));
    if (!d_q || !d_k || !d_v || !d_out || !d_partial) return false;
    if (cudaMemcpy(d_q, q.data(), q.size() * sizeof(uint16_t), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_k, k.data(), k.size() * sizeof(uint16_t), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_v, v.data(), v.size() * sizeof(uint16_t), cudaMemcpyHostToDevice) != cudaSuccess ||
        !pocket::qwen_gqa_decode_attention_f16_fused_cuda(
            d_q, d_k, d_v, d_out, d_partial, q_heads, kv_heads, head_dim,
            context_len, context_len, window, sink) ||
        cudaDeviceSynchronize() != cudaSuccess) {
        fail("FP16 sparse decode launch");
        return true;
    }
    std::vector<uint16_t> got(q.size());
    if (cudaMemcpy(got.data(), d_out, got.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost) != cudaSuccess) {
        fail("FP16 sparse decode copy");
        return true;
    }
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    double worst = 0.0;
    for (int head = 0; head < q_heads; ++head) {
        std::vector<double> scores;
        std::vector<int> positions;
        for (int pos = 0; pos < context_len; ++pos) {
            if (pos < sink || pos >= window_start) positions.push_back(pos);
        }
        scores.reserve(positions.size());
        double maximum = -1.0e300;
        for (int pos : positions) {
            double dot = 0.0;
            for (int d = 0; d < head_dim; ++d) {
                dot += static_cast<double>(from_half(q[head * head_dim + d])) *
                       from_half(k[pos * head_dim + d]);
            }
            scores.push_back(dot * scale);
            maximum = std::max(maximum, scores.back());
        }
        double denominator = 0.0;
        for (double& score : scores) {
            score = std::exp(score - maximum);
            denominator += score;
        }
        for (int d = 0; d < head_dim; ++d) {
            double value = 0.0;
            for (size_t i = 0; i < positions.size(); ++i) {
                value += scores[i] * from_half(v[positions[i] * head_dim + d]);
            }
            const double expected = value / denominator;
            worst = std::max(worst, std::fabs(
                static_cast<double>(from_half(got[head * head_dim + d])) - expected));
        }
    }
    if (worst > 4.0e-3) fail("FP16 sparse decode reference check");
    else std::printf("  FP16 sparse decode context=%d sink=%d window=%d attended=%d worst=%.3e\n",
                     context_len, sink, window, attended, worst);
    return true;
}

bool check_decode_grid_256k() {
    constexpr int q_heads = 6;
    constexpr int kv_heads = 1;
    constexpr int head_dim = 64;
    constexpr int context_len = 262144;
    DeviceBuffer dq, dk_cache, dv_cache, dout, dfused, dscores, dpartial;
    uint16_t* d_q = dq.allocate<uint16_t>(q_heads * head_dim);
    uint16_t* d_k_cache = dk_cache.allocate<uint16_t>(context_len * kv_heads * head_dim);
    uint16_t* d_v_cache = dv_cache.allocate<uint16_t>(context_len * kv_heads * head_dim);
    uint16_t* d_out = dout.allocate<uint16_t>(q_heads * head_dim);
    uint16_t* d_fused = dfused.allocate<uint16_t>(q_heads * head_dim);
    float* d_scores = dscores.allocate<float>(q_heads * context_len);
    const int splits = std::min(64, (context_len + 2048 - 1) / 2048);
    float* d_partial = dpartial.allocate<float>(
        static_cast<size_t>(q_heads) * splits * (head_dim + 2));
    if (!d_q || !d_k_cache || !d_v_cache || !d_out || !d_fused ||
        !d_scores || !d_partial) return false;
    if (cudaMemset(d_q, 0, q_heads * head_dim * sizeof(uint16_t)) != cudaSuccess ||
        cudaMemset(d_k_cache, 0, context_len * kv_heads * head_dim * sizeof(uint16_t)) != cudaSuccess ||
        cudaMemset(d_v_cache, 0, context_len * kv_heads * head_dim * sizeof(uint16_t)) != cudaSuccess ||
        !pocket::qwen_gqa_decode_attention_f16(
            d_q, d_k_cache, d_v_cache, d_out, d_scores, q_heads, kv_heads,
            head_dim, context_len, context_len) ||
        !pocket::qwen_gqa_decode_attention_f16_fused_cuda(
            d_q, d_k_cache, d_v_cache, d_fused, d_partial, q_heads, kv_heads,
            head_dim, context_len, context_len) ||
        cudaDeviceSynchronize() != cudaSuccess) {
        fail("FP16 256K decode grid launch");
        return true;
    }
    std::vector<uint16_t> got(q_heads * head_dim);
    std::vector<uint16_t> fused(q_heads * head_dim);
    cudaMemcpy(got.data(), d_out, got.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(fused.data(), d_fused, fused.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    for (uint16_t value : got) {
        if (value != 0) {
            fail("FP16 256K decode zero output");
            break;
        }
    }
    bool finite = true;
    const double worst = max_half_difference(got, fused, &finite);
    if (!finite || worst > 4.0e-3) fail("FP16 256K fused decode numerical check");
    std::printf("  FP16 decode context=%d grid=flattened fused worst=%.3e\n",
                context_len, worst);
    return true;
}

bool check_fp8_f16_projection() {
    std::mt19937 rng(24000);
    std::uniform_real_distribution<float> x_dist(-0.75f, 0.75f);
    std::uniform_real_distribution<float> scale_dist(0.005f, 0.02f);
    auto random_code = [&]() {
        // Keep the direct gate finite and away from the E4M3 NaN encodings.
        return static_cast<uint8_t>(32 + (rng() % 192));
    };
    auto max_error = [](const std::vector<uint16_t>& got,
                        const std::vector<float>& expected, int stride,
                        int rows, int batch) {
        double worst = 0.0;
        for (int b = 0; b < batch; ++b) {
            for (int r = 0; r < rows; ++r) {
                worst = std::max(worst, std::fabs(
                    static_cast<double>(from_half(got[static_cast<size_t>(b) * stride + r])) -
                    expected[static_cast<size_t>(b) * rows + r]));
            }
        }
        return worst;
    };

    // Decode shape crosses the multi-row occupancy threshold (4352 rows ->
    // 136 blocks at four rows per warp) and uses a non-power-of-two block count.
    constexpr int decode_rows = 4352;
    constexpr int decode_cols = 512;
    constexpr int decode_weight_stride = 512;
    constexpr int decode_scale_stride = 4;
    std::vector<uint16_t> decode_x(decode_cols);
    std::vector<uint8_t> decode_weight(static_cast<size_t>(decode_rows) * decode_weight_stride);
    std::vector<uint16_t> decode_scale(static_cast<size_t>(decode_rows / 128) * decode_scale_stride);
    for (uint16_t& value : decode_x) value = to_half(x_dist(rng));
    for (uint8_t& value : decode_weight) value = random_code();
    for (uint16_t& value : decode_scale) value = to_half(scale_dist(rng));
    std::vector<float> decode_expected(decode_rows);
    for (int r = 0; r < decode_rows; ++r) {
        float sum = 0.0f;
        for (int c = 0; c < decode_cols; ++c) {
            sum += from_half(decode_x[c]) * from_fp8_e4m3(decode_weight[
                static_cast<size_t>(r) * decode_weight_stride + c]) *
                   from_half(decode_scale[static_cast<size_t>(r / 128) * decode_scale_stride + c / 128]);
        }
        decode_expected[r] = sum;
    }
    DeviceBuffer d_decode_x, d_decode_weight, d_decode_scale, d_decode_multi,
        d_decode_vector, d_decode_fallback;
    uint16_t* dx = d_decode_x.allocate<uint16_t>(decode_x.size());
    uint8_t* dw = d_decode_weight.allocate<uint8_t>(decode_weight.size());
    uint16_t* ds = d_decode_scale.allocate<uint16_t>(decode_scale.size());
    uint16_t* dy_multi = d_decode_multi.allocate<uint16_t>(decode_rows);
    uint16_t* dy_vector = d_decode_vector.allocate<uint16_t>(decode_rows);
    uint16_t* dy_fallback = d_decode_fallback.allocate<uint16_t>(decode_rows);
    if (!dx || !dw || !ds || !dy_multi || !dy_vector || !dy_fallback ||
        cudaMemcpy(dx, decode_x.data(), decode_x.size() * sizeof(uint16_t), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(dw, decode_weight.data(), decode_weight.size() * sizeof(uint8_t), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(ds, decode_scale.data(), decode_scale.size() * sizeof(uint16_t), cudaMemcpyHostToDevice) != cudaSuccess) {
        fail("FP16 FP8 decode gate allocation/copy");
        return false;
    }
    setenv("QWEN_FP8_F16_MULTIROW", "1", 1);
    setenv("QWEN_FP8_F16_MULTIROW_ROWS", "4", 1);
    if (!pocket::qwen_fp8_e4m3_fp16scale_matvec_f16_cuda(
            dx, dw, ds, dy_multi, decode_rows, decode_cols,
            decode_weight_stride, decode_scale_stride) ||
        cudaDeviceSynchronize() != cudaSuccess) {
        fail("FP16 FP8 multi-row decode launch");
        return false;
    }
    setenv("QWEN_FP8_F16_MULTIROW", "0", 1);
    unsetenv("QWEN_FP8_F16_MULTIROW_ROWS");
    setenv("QWEN_FP8_F16_VECTORIZE", "1", 1);
    if (!pocket::qwen_fp8_e4m3_fp16scale_matvec_f16_cuda(
            dx, dw, ds, dy_vector, decode_rows, decode_cols,
            decode_weight_stride, decode_scale_stride) ||
        cudaDeviceSynchronize() != cudaSuccess) {
        fail("FP16 FP8 vectorized decode launch");
        return false;
    }
    setenv("QWEN_FP8_F16_MULTIROW", "0", 1);
    setenv("QWEN_FP8_F16_VECTORIZE", "0", 1);
    if (!pocket::qwen_fp8_e4m3_fp16scale_matvec_f16_cuda(
            dx, dw, ds, dy_fallback, decode_rows, decode_cols,
            decode_weight_stride, decode_scale_stride) ||
        cudaDeviceSynchronize() != cudaSuccess) {
        fail("FP16 FP8 scalar decode launch");
        return false;
    }
    std::vector<uint16_t> decode_multi(decode_rows), decode_vector(decode_rows),
        decode_fallback(decode_rows);
    cudaMemcpy(decode_multi.data(), dy_multi, decode_multi.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(decode_vector.data(), dy_vector, decode_vector.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(decode_fallback.data(), dy_fallback, decode_fallback.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    double decode_vector_dispatch_error = 0.0;
    double decode_multi_dispatch_error = 0.0;
    for (int r = 0; r < decode_rows; ++r) {
        decode_vector_dispatch_error = std::max(decode_vector_dispatch_error,
            std::fabs(static_cast<double>(from_half(decode_vector[r])) - from_half(decode_fallback[r])));
        decode_multi_dispatch_error = std::max(decode_multi_dispatch_error,
            std::fabs(static_cast<double>(from_half(decode_multi[r])) - from_half(decode_fallback[r])));
    }
    const double decode_error = max_error(decode_vector, decode_expected, 1, decode_rows, 1);
    const double decode_multi_error = max_error(decode_multi, decode_expected, 1, decode_rows, 1);
    if (decode_error > 2.0e-2 || decode_multi_error > 2.0e-2 ||
        decode_vector_dispatch_error > 2.0e-2 || decode_multi_dispatch_error > 2.0e-2) {
        fail("FP16 FP8 decode dispatch numerical check");
    } else {
        std::printf("  FP16 FP8 decode rows=%d cols=%d vector=%.3e multi=%.3e "
                    "vector_vs_scalar=%.3e multi_vs_scalar=%.3e\n",
                    decode_rows, decode_cols, decode_error, decode_multi_error,
                    decode_vector_dispatch_error, decode_multi_dispatch_error);
    }

    // The DeltaNet decode path can issue QKV and Z as one grid without copying
    // their raw FP8 tensors into permanent concatenated storage. The dual grid
    // must remain bit exact against the two established wide-8 launches.
    {
        constexpr int dual_cols = 512;
        constexpr int first_rows = 640;
        constexpr int second_rows = 384;
        constexpr int third_rows = 256;
        constexpr int first_scale_stride = dual_cols / 128;
        constexpr int second_scale_stride = dual_cols / 128;
        std::vector<uint16_t> dual_x(dual_cols);
        std::vector<uint8_t> first_weight(
            static_cast<size_t>(first_rows) * dual_cols);
        std::vector<uint8_t> second_weight(
            static_cast<size_t>(second_rows) * dual_cols);
        std::vector<uint16_t> first_scale(
            static_cast<size_t>((first_rows + 127) / 128) *
            first_scale_stride);
        std::vector<uint16_t> second_scale(
            static_cast<size_t>((second_rows + 127) / 128) *
            second_scale_stride);
        std::vector<uint8_t> third_weight(
            static_cast<size_t>(third_rows) * dual_cols);
        std::vector<uint16_t> third_scale(
            static_cast<size_t>((third_rows + 127) / 128) *
            second_scale_stride);
        for (uint16_t& value : dual_x) value = to_half(x_dist(rng));
        for (uint8_t& value : first_weight) value = random_code();
        for (uint8_t& value : second_weight) value = random_code();
        for (uint8_t& value : third_weight) value = random_code();
        for (uint16_t& value : first_scale) value = to_half(scale_dist(rng));
        for (uint16_t& value : second_scale) value = to_half(scale_dist(rng));
        for (uint16_t& value : third_scale) value = to_half(scale_dist(rng));

        DeviceBuffer d_dual_x, d_first_weight, d_second_weight, d_first_scale,
            d_second_scale, d_first_reference, d_second_reference,
            d_first_dual, d_second_dual, d_third_weight, d_third_scale,
            d_third_reference, d_third_grouped;
        uint16_t* dual_px = d_dual_x.allocate<uint16_t>(dual_x.size());
        uint8_t* first_pw = d_first_weight.allocate<uint8_t>(
            first_weight.size());
        uint8_t* second_pw = d_second_weight.allocate<uint8_t>(
            second_weight.size());
        uint16_t* first_ps = d_first_scale.allocate<uint16_t>(
            first_scale.size());
        uint16_t* second_ps = d_second_scale.allocate<uint16_t>(
            second_scale.size());
        uint16_t* first_reference =
            d_first_reference.allocate<uint16_t>(first_rows);
        uint16_t* second_reference =
            d_second_reference.allocate<uint16_t>(second_rows);
        uint16_t* first_dual = d_first_dual.allocate<uint16_t>(first_rows);
        uint16_t* second_dual = d_second_dual.allocate<uint16_t>(second_rows);
        uint8_t* third_pw = d_third_weight.allocate<uint8_t>(
            third_weight.size());
        uint16_t* third_ps = d_third_scale.allocate<uint16_t>(
            third_scale.size());
        uint16_t* third_reference =
            d_third_reference.allocate<uint16_t>(third_rows);
        uint16_t* third_grouped =
            d_third_grouped.allocate<uint16_t>(third_rows);
        if (!dual_px || !first_pw || !second_pw || !first_ps || !second_ps ||
            !first_reference || !second_reference || !first_dual ||
            !second_dual || !third_pw || !third_ps || !third_reference ||
            !third_grouped ||
            cudaMemcpy(dual_px, dual_x.data(), dual_x.size() * sizeof(uint16_t),
                       cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(first_pw, first_weight.data(), first_weight.size(),
                       cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(second_pw, second_weight.data(), second_weight.size(),
                       cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(first_ps, first_scale.data(),
                       first_scale.size() * sizeof(uint16_t),
                       cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(second_ps, second_scale.data(),
                       second_scale.size() * sizeof(uint16_t),
                       cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(third_pw, third_weight.data(), third_weight.size(),
                       cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(third_ps, third_scale.data(),
                       third_scale.size() * sizeof(uint16_t),
                       cudaMemcpyHostToDevice) != cudaSuccess) {
            fail("FP16 FP8 dual projection allocation/copy");
            return false;
        }
        setenv("QWEN_FP8_F16_VECTORIZE", "1", 1);
        setenv("QWEN_FP8_F16_COLS_PER_LANE", "8", 1);
        unsetenv("QWEN_FP8_F16_MULTIROW");
        bool dual_ok = true;
        for (int pass = 0; pass < 3 && dual_ok; ++pass) {
            dual_ok =
                pocket::qwen_fp8_e4m3_fp16scale_matvec_f16_cuda(
                    dual_px, first_pw, first_ps, first_reference, first_rows,
                    dual_cols, dual_cols, first_scale_stride) &&
                pocket::qwen_fp8_e4m3_fp16scale_matvec_f16_cuda(
                    dual_px, second_pw, second_ps, second_reference,
                    second_rows, dual_cols, dual_cols, second_scale_stride) &&
                pocket::qwen_fp8_e4m3_fp16scale_matvec_dual_f16_cuda(
                    dual_px, first_pw, first_ps, first_dual, first_rows,
                    dual_cols, first_scale_stride, second_pw, second_ps,
                    second_dual, second_rows, dual_cols, second_scale_stride,
                    dual_cols) &&
                cudaDeviceSynchronize() == cudaSuccess;
        }
        unsetenv("QWEN_FP8_F16_COLS_PER_LANE");
        unsetenv("QWEN_FP8_F16_VECTORIZE");
        if (!dual_ok) {
            fail("FP16 FP8 dual projection launch");
            return false;
        }
        std::vector<uint16_t> first_got(first_rows), first_want(first_rows);
        std::vector<uint16_t> second_got(second_rows), second_want(second_rows);
        cudaMemcpy(first_got.data(), first_dual,
                   first_got.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(first_want.data(), first_reference,
                   first_want.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(second_got.data(), second_dual,
                   second_got.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(second_want.data(), second_reference,
                   second_want.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
        if (first_got != first_want || second_got != second_want) {
            fail("FP16 FP8 dual projection bit exactness");
        } else {
            std::printf("  FP16 FP8 dual projection first=%d second=%d "
                        "cols=%d bit exact\n",
                        first_rows, second_rows, dual_cols);
        }

        bool triple_ok = true;
        for (int pass = 0; pass < 3 && triple_ok; ++pass) {
            triple_ok =
                pocket::qwen_fp8_e4m3_fp16scale_matvec_f16_cuda(
                    dual_px, third_pw, third_ps, third_reference, third_rows,
                    dual_cols, dual_cols, second_scale_stride) &&
                pocket::qwen_fp8_e4m3_fp16scale_matvec_triple_f16_cuda(
                    dual_px, first_pw, first_ps, first_dual, first_rows,
                    dual_cols, first_scale_stride, second_pw, second_ps,
                    second_dual, second_rows, dual_cols, second_scale_stride,
                    third_pw, third_ps, third_grouped, third_rows, dual_cols,
                    second_scale_stride, dual_cols) &&
                cudaDeviceSynchronize() == cudaSuccess;
        }
        if (!triple_ok) {
            fail("FP16 FP8 triple projection launch");
            return false;
        }
        std::vector<uint16_t> third_got(third_rows), third_want(third_rows);
        cudaMemcpy(first_got.data(), first_dual,
                   first_got.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(second_got.data(), second_dual,
                   second_got.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(third_got.data(), third_grouped,
                   third_got.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(third_want.data(), third_reference,
                   third_want.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
        if (first_got != first_want || second_got != second_want ||
            third_got != third_want) {
            fail("FP16 FP8 triple projection bit exactness");
        } else {
            std::printf("  FP16 FP8 triple projection rows=%d+%d+%d "
                        "cols=%d bit exact\n",
                        first_rows, second_rows, third_rows, dual_cols);
        }
    }

    // Small speculative batches reuse each FP8 weight across all input rows.
    constexpr int small_batch = 5;
    constexpr int small_rows = 130;
    constexpr int small_cols = 300;
    constexpr int small_x_stride = 304;
    constexpr int small_y_stride = 136;
    constexpr int small_weight_stride = 304;
    constexpr int small_scale_stride = 3;
    std::vector<uint16_t> small_x(
        static_cast<size_t>(small_batch) * small_x_stride);
    std::vector<uint8_t> small_weight(
        static_cast<size_t>(small_rows) * small_weight_stride);
    std::vector<uint16_t> small_scale(
        static_cast<size_t>((small_rows + 127) / 128) * small_scale_stride);
    for (uint16_t& value : small_x) value = to_half(x_dist(rng));
    for (uint8_t& value : small_weight) value = random_code();
    for (uint16_t& value : small_scale) value = to_half(scale_dist(rng));
    DeviceBuffer d_small_x, d_small_weight, d_small_scale, d_small_reuse,
        d_small_tiled;
    uint16_t* sx = d_small_x.allocate<uint16_t>(small_x.size());
    uint8_t* sw = d_small_weight.allocate<uint8_t>(small_weight.size());
    uint16_t* ss = d_small_scale.allocate<uint16_t>(small_scale.size());
    uint16_t* sy_reuse = d_small_reuse.allocate<uint16_t>(
        static_cast<size_t>(small_batch) * small_y_stride);
    uint16_t* sy_tiled = d_small_tiled.allocate<uint16_t>(
        static_cast<size_t>(small_batch) * small_y_stride);
    if (!sx || !sw || !ss || !sy_reuse || !sy_tiled ||
        cudaMemcpy(sx, small_x.data(), small_x.size() * sizeof(uint16_t),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(sw, small_weight.data(), small_weight.size(),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(ss, small_scale.data(), small_scale.size() * sizeof(uint16_t),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        fail("FP16 FP8 small-batch allocation/copy");
        return false;
    }
    setenv("QWEN_FP8_F16_SMALL_BATCH", "1", 1);
    if (!pocket::qwen_fp8_e4m3_fp16scale_matmul_rows_f16_cuda(
            sx, sw, ss, sy_reuse, small_batch, small_rows, small_cols,
            small_x_stride, small_y_stride, small_weight_stride,
            small_scale_stride) || cudaDeviceSynchronize() != cudaSuccess) {
        fail("FP16 FP8 small-batch reuse launch");
        return false;
    }
    setenv("QWEN_FP8_F16_SMALL_BATCH", "0", 1);
    if (!pocket::qwen_fp8_e4m3_fp16scale_matmul_rows_f16_cuda(
            sx, sw, ss, sy_tiled, small_batch, small_rows, small_cols,
            small_x_stride, small_y_stride, small_weight_stride,
            small_scale_stride) || cudaDeviceSynchronize() != cudaSuccess) {
        fail("FP16 FP8 small-batch tiled launch");
        return false;
    }
    std::vector<uint16_t> small_reuse(
        static_cast<size_t>(small_batch) * small_y_stride);
    std::vector<uint16_t> small_tiled(small_reuse.size());
    cudaMemcpy(small_reuse.data(), sy_reuse,
               small_reuse.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(small_tiled.data(), sy_tiled,
               small_tiled.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    double small_dispatch_error = 0.0;
    for (int sample = 0; sample < small_batch; ++sample) {
        for (int row = 0; row < small_rows; ++row) {
            const size_t at = static_cast<size_t>(sample) * small_y_stride + row;
            small_dispatch_error = std::max(
                small_dispatch_error,
                std::fabs(static_cast<double>(from_half(small_reuse[at])) -
                          from_half(small_tiled[at])));
        }
    }
    unsetenv("QWEN_FP8_F16_SMALL_BATCH");
    if (small_dispatch_error > 3.0e-2) {
        fail("FP16 FP8 small-batch numerical check");
    } else {
        std::printf("  FP16 FP8 small batch=%d rows=%d cols=%d "
                    "reuse_vs_tiled=%.3e\n",
                    small_batch, small_rows, small_cols,
                    small_dispatch_error);
    }

    // The shared-activation small-batch kernel keeps the per-lane column order of
    // the register-only kernel, so speculative verification must stay bit exact.
    // A near-tie greedy argmax would otherwise diverge from plain decode.
    {
        struct SharedCase {
            int batch;
            int rows;
            int cols;
        };
        // 5120/4352/1536 are the real Qwen3.8 TP4 shard widths; 300 and 2052
        // exercise a partial final tile and a non-128-multiple vector width.
        const SharedCase cases[] = {
            {2, 130, 5120}, {3, 96, 4352}, {4, 72, 1536},
            {5, 130, 300},  {8, 136, 2052}, {8, 40, 5120},
        };
        bool shared_ok = true;
        double worst = 0.0;
        // The paired shared kernel fuses two strided steps while keeping each
        // lane's columns and addition order, so it must match the register-only
        // path bit for bit.
        for (const SharedCase& item : cases) {
            const int x_stride = item.cols;
            const int y_stride = item.rows;
            const int weight_stride = item.cols;
            const int scale_stride = (item.cols + 127) / 128;
            std::vector<uint16_t> host_x(
                static_cast<size_t>(item.batch) * x_stride);
            std::vector<uint8_t> host_weight(
                static_cast<size_t>(item.rows) * weight_stride);
            std::vector<uint16_t> host_scale(
                static_cast<size_t>((item.rows + 127) / 128) * scale_stride);
            for (uint16_t& value : host_x) value = to_half(x_dist(rng));
            for (uint8_t& value : host_weight) value = random_code();
            for (uint16_t& value : host_scale) value = to_half(scale_dist(rng));
            DeviceBuffer dx, dw, ds, d_shared, d_plain;
            uint16_t* px = dx.allocate<uint16_t>(host_x.size());
            uint8_t* pw = dw.allocate<uint8_t>(host_weight.size());
            uint16_t* ps = ds.allocate<uint16_t>(host_scale.size());
            const size_t out_elements =
                static_cast<size_t>(item.batch) * y_stride;
            uint16_t* py_shared = d_shared.allocate<uint16_t>(out_elements);
            uint16_t* py_plain = d_plain.allocate<uint16_t>(out_elements);
            if (!px || !pw || !ps || !py_shared || !py_plain ||
                cudaMemcpy(px, host_x.data(),
                           host_x.size() * sizeof(uint16_t),
                           cudaMemcpyHostToDevice) != cudaSuccess ||
                cudaMemcpy(pw, host_weight.data(), host_weight.size(),
                           cudaMemcpyHostToDevice) != cudaSuccess ||
                cudaMemcpy(ps, host_scale.data(),
                           host_scale.size() * sizeof(uint16_t),
                           cudaMemcpyHostToDevice) != cudaSuccess) {
                fail("FP16 FP8 shared small-batch allocation/copy");
                shared_ok = false;
                break;
            }
            setenv("QWEN_FP8_F16_SMALL_BATCH", "1", 1);
            setenv("QWEN_FP8_F16_SMALL_BATCH_SHARED", "1", 1);
            const bool shared_launched =
                pocket::qwen_fp8_e4m3_fp16scale_matmul_rows_f16_cuda(
                    px, pw, ps, py_shared, item.batch, item.rows, item.cols,
                    x_stride, y_stride, weight_stride, scale_stride) &&
                cudaDeviceSynchronize() == cudaSuccess;
            setenv("QWEN_FP8_F16_SMALL_BATCH_SHARED", "0", 1);
            const bool plain_launched =
                pocket::qwen_fp8_e4m3_fp16scale_matmul_rows_f16_cuda(
                    px, pw, ps, py_plain, item.batch, item.rows, item.cols,
                    x_stride, y_stride, weight_stride, scale_stride) &&
                cudaDeviceSynchronize() == cudaSuccess;
            unsetenv("QWEN_FP8_F16_SMALL_BATCH_SHARED");
            unsetenv("QWEN_FP8_F16_SMALL_BATCH");
            if (!shared_launched || !plain_launched) {
                fail("FP16 FP8 shared small-batch launch");
                shared_ok = false;
                break;
            }
            std::vector<uint16_t> got(out_elements);
            std::vector<uint16_t> want(out_elements);
            cudaMemcpy(got.data(), py_shared, out_elements * sizeof(uint16_t),
                       cudaMemcpyDeviceToHost);
            cudaMemcpy(want.data(), py_plain, out_elements * sizeof(uint16_t),
                       cudaMemcpyDeviceToHost);
            int mismatches = 0;
            for (int sample = 0; sample < item.batch; ++sample) {
                for (int row = 0; row < item.rows; ++row) {
                    const size_t at =
                        static_cast<size_t>(sample) * y_stride + row;
                    if (got[at] != want[at]) {
                        ++mismatches;
                        worst = std::max(worst,
                            std::fabs(static_cast<double>(from_half(got[at])) -
                                      from_half(want[at])));
                    }
                }
            }
            if (mismatches != 0) {
                std::printf("  FP16 FP8 shared small-batch mismatch batch=%d "
                            "rows=%d cols=%d count=%d worst=%.3e\n",
                            item.batch, item.rows, item.cols, mismatches, worst);
                fail("FP16 FP8 shared small-batch bit exactness");
                shared_ok = false;
                break;
            }
        }
        if (shared_ok) {
            std::printf("  FP16 FP8 shared small-batch bit exact over %zu "
                        "shard shapes\n",
                        sizeof(cases) / sizeof(cases[0]));
        }
    }

    // Prefill shape triggers the 128-token x 64-row N64 tile and has padded
    // strides so both aligned vector loads and output indexing are exercised.
    constexpr int batch = 128;
    constexpr int rows = 130;
    constexpr int cols = 300;
    constexpr int x_stride = 304;
    constexpr int y_stride = 136;
    constexpr int weight_stride = 304;
    constexpr int scale_stride = 3;
    std::vector<uint16_t> prefill_x(static_cast<size_t>(batch) * x_stride);
    std::vector<uint8_t> prefill_weight(static_cast<size_t>(rows) * weight_stride);
    std::vector<uint16_t> prefill_scale(static_cast<size_t>((rows + 127) / 128) * scale_stride);
    for (uint16_t& value : prefill_x) value = to_half(x_dist(rng));
    for (uint8_t& value : prefill_weight) value = random_code();
    for (uint16_t& value : prefill_scale) value = to_half(scale_dist(rng));
    std::vector<float> prefill_expected(static_cast<size_t>(batch) * rows);
    for (int b = 0; b < batch; ++b) {
        for (int r = 0; r < rows; ++r) {
            float sum = 0.0f;
            for (int c = 0; c < cols; ++c) {
                sum += from_half(prefill_x[static_cast<size_t>(b) * x_stride + c]) *
                       from_fp8_e4m3(prefill_weight[static_cast<size_t>(r) * weight_stride + c]) *
                       from_half(prefill_scale[static_cast<size_t>(r / 128) * scale_stride + c / 128]);
            }
            prefill_expected[static_cast<size_t>(b) * rows + r] = sum;
        }
    }
    DeviceBuffer d_prefill_x, d_prefill_weight, d_prefill_scale, d_prefill_wide, d_prefill_fallback;
    uint16_t* px = d_prefill_x.allocate<uint16_t>(prefill_x.size());
    uint8_t* pw = d_prefill_weight.allocate<uint8_t>(prefill_weight.size());
    uint16_t* ps = d_prefill_scale.allocate<uint16_t>(prefill_scale.size());
    uint16_t* py_wide = d_prefill_wide.allocate<uint16_t>(static_cast<size_t>(batch) * y_stride);
    uint16_t* py_fallback = d_prefill_fallback.allocate<uint16_t>(static_cast<size_t>(batch) * y_stride);
    if (!px || !pw || !ps || !py_wide || !py_fallback ||
        cudaMemcpy(px, prefill_x.data(), prefill_x.size() * sizeof(uint16_t), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(pw, prefill_weight.data(), prefill_weight.size() * sizeof(uint8_t), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(ps, prefill_scale.data(), prefill_scale.size() * sizeof(uint16_t), cudaMemcpyHostToDevice) != cudaSuccess) {
        fail("FP16 FP8 wide prefill allocation/copy");
        return false;
    }
    setenv("QWEN_FP8_F16_PREFILL_WIDE_N64", "1", 1);
    if (!pocket::qwen_fp8_e4m3_fp16scale_matmul_rows_f16_cuda(
            px, pw, ps, py_wide, batch, rows, cols, x_stride, y_stride,
            weight_stride, scale_stride) || cudaDeviceSynchronize() != cudaSuccess) {
        fail("FP16 FP8 wide prefill launch");
        return false;
    }
    setenv("QWEN_FP8_F16_PREFILL_WIDE_N64", "0", 1);
    if (!pocket::qwen_fp8_e4m3_fp16scale_matmul_rows_f16_cuda(
            px, pw, ps, py_fallback, batch, rows, cols, x_stride, y_stride,
            weight_stride, scale_stride) || cudaDeviceSynchronize() != cudaSuccess) {
        fail("FP16 FP8 fallback prefill launch");
        return false;
    }
    std::vector<uint16_t> prefill_wide(static_cast<size_t>(batch) * y_stride);
    std::vector<uint16_t> prefill_fallback(prefill_wide.size());
    cudaMemcpy(prefill_wide.data(), py_wide, prefill_wide.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(prefill_fallback.data(), py_fallback, prefill_fallback.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    double prefill_dispatch_error = 0.0;
    for (int b = 0; b < batch; ++b) {
        for (int r = 0; r < rows; ++r) {
            prefill_dispatch_error = std::max(prefill_dispatch_error,
                std::fabs(static_cast<double>(from_half(prefill_wide[static_cast<size_t>(b) * y_stride + r])) -
                          from_half(prefill_fallback[static_cast<size_t>(b) * y_stride + r])));
        }
    }
    const double prefill_error = max_error(prefill_wide, prefill_expected, y_stride, rows, batch);
    // cuBLAS prefill path. It dequantises the weight block to FP16 scratch and
    // runs a tensor-core GEMM, so it accumulates in a different order than the
    // hand-written tiles. The check is against the same FP32 CPU reference the
    // other variants use, at the same tolerance, rather than bit equality.
    DeviceBuffer d_prefill_cublas;
    uint16_t* py_cublas =
        d_prefill_cublas.allocate<uint16_t>(static_cast<size_t>(batch) * y_stride);
    double prefill_cublas_error = 0.0;
    if (!py_cublas) {
        fail("FP16 FP8 cuBLAS prefill allocation");
        return false;
    }
    setenv("QWEN_FP8_F16_PREFILL_CUBLAS", "1", 1);
    if (!pocket::qwen_fp8_e4m3_fp16scale_matmul_rows_f16_cuda(
            px, pw, ps, py_cublas, batch, rows, cols, x_stride, y_stride,
            weight_stride, scale_stride) || cudaDeviceSynchronize() != cudaSuccess) {
        fail("FP16 FP8 cuBLAS prefill launch");
        return false;
    }
    unsetenv("QWEN_FP8_F16_PREFILL_CUBLAS");
    std::vector<uint16_t> prefill_cublas(prefill_wide.size());
    cudaMemcpy(prefill_cublas.data(), py_cublas,
               prefill_cublas.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    prefill_cublas_error =
        max_error(prefill_cublas, prefill_expected, y_stride, rows, batch);
    // Only the first `rows` entries of each `y_stride` row are written; the
    // stride padding stays untouched, so scanning it would read uninitialised
    // device memory rather than kernel output.
    for (int b = 0; b < batch; ++b) {
        for (int r = 0; r < rows; ++r) {
            const float value = from_half(
                prefill_cublas[static_cast<size_t>(b) * y_stride + r]);
            if (!std::isfinite(static_cast<double>(value))) {
                fail("FP16 FP8 cuBLAS prefill produced a non-finite value");
                return false;
            }
        }
    }
    unsetenv("QWEN_FP8_F16_MULTIROW");
    unsetenv("QWEN_FP8_F16_MULTIROW_ROWS");
    unsetenv("QWEN_FP8_F16_PREFILL_WIDE_N64");
    if (prefill_error > 3.0e-2 || prefill_dispatch_error > 3.0e-2 ||
        prefill_cublas_error > 3.0e-2) {
        fail("FP16 FP8 wide prefill numerical check");
    } else {
        std::printf("  FP16 FP8 prefill batch=%d rows=%d cols=%d error=%.3e "
                    "dispatch=%.3e cublas=%.3e\n",
                    batch, rows, cols, prefill_error, prefill_dispatch_error,
                    prefill_cublas_error);
    }
    return true;
}

bool check_resident_fp16_projection() {
    constexpr int batch = 8;
    constexpr int rows = 136;
    constexpr int cols = 512;
    constexpr int x_stride = 520;
    constexpr int y_stride = 144;
    constexpr int weight_stride = 520;
    constexpr int scale_stride = 5;
    const uint8_t finite_codes[] = {
        0x01, 0x03, 0x08, 0x18, 0x20, 0x28, 0x30, 0x38,
        0x40, 0x48, 0x81, 0x83, 0x88, 0x98, 0xa0, 0xa8,
        0xb0, 0xb8, 0xc0, 0xc8,
    };
    std::mt19937 rng(52081);
    std::uniform_real_distribution<float> x_dist(-0.5f, 0.5f);
    std::uniform_real_distribution<float> scale_dist(0.005f, 0.02f);
    std::vector<uint16_t> x(static_cast<size_t>(batch) * x_stride);
    std::vector<uint8_t> weight(static_cast<size_t>(rows) * weight_stride);
    std::vector<uint16_t> scale(
        static_cast<size_t>((rows + 127) / 128) * scale_stride);
    for (uint16_t& value : x) value = to_half(x_dist(rng));
    for (uint8_t& value : weight) {
        value = finite_codes[rng() % (sizeof(finite_codes) / sizeof(finite_codes[0]))];
    }
    for (uint16_t& value : scale) value = to_half(scale_dist(rng));

    std::vector<uint16_t> expected_weight(static_cast<size_t>(rows) * cols);
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            expected_weight[static_cast<size_t>(row) * cols + col] = to_half(
                from_fp8_e4m3(weight[static_cast<size_t>(row) * weight_stride + col]) *
                from_half(scale[static_cast<size_t>(row / 128) * scale_stride +
                                      col / 128]));
        }
    }

    DeviceBuffer dx, dw, ds, d_resident_weight, d_online, d_resident;
    uint16_t* d_x = dx.allocate<uint16_t>(x.size());
    uint8_t* d_weight = dw.allocate<uint8_t>(weight.size());
    uint16_t* d_scale = ds.allocate<uint16_t>(scale.size());
    uint16_t* d_dense = d_resident_weight.allocate<uint16_t>(
        expected_weight.size());
    uint16_t* d_online_y = d_online.allocate<uint16_t>(
        static_cast<size_t>(batch) * y_stride);
    uint16_t* d_resident_y = d_resident.allocate<uint16_t>(
        static_cast<size_t>(batch) * y_stride);
    if (!d_x || !d_weight || !d_scale || !d_dense || !d_online_y ||
        !d_resident_y ||
        cudaMemcpy(d_x, x.data(), x.size() * sizeof(uint16_t),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_weight, weight.data(), weight.size(),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_scale, scale.data(), scale.size() * sizeof(uint16_t),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        fail("resident FP16 projection allocation/copy");
        return false;
    }

    setenv("QWEN_FP8_F16_SMALL_BATCH", "1", 1);
    setenv("QWEN_FP8_F16_SMALL_BATCH_SHARED", "1", 1);
    const bool launched =
        pocket::qwen_fp8_e4m3_fp16scale_dequantize_f16_cuda(
            d_weight, d_scale, d_dense, rows, cols, weight_stride,
            scale_stride) &&
        pocket::qwen_fp8_e4m3_fp16scale_matmul_rows_f16_cuda(
            d_x, d_weight, d_scale, d_online_y, batch, rows, cols, x_stride,
            y_stride, weight_stride, scale_stride) &&
        pocket::qwen_fp16_matmul_rows_f16_cublas_cuda(
            d_x, d_dense, d_resident_y, batch, rows, cols, x_stride, y_stride,
            cols) &&
        cudaDeviceSynchronize() == cudaSuccess;
    unsetenv("QWEN_FP8_F16_SMALL_BATCH_SHARED");
    unsetenv("QWEN_FP8_F16_SMALL_BATCH");
    if (!launched) {
        fail("resident FP16 projection launch");
        return false;
    }

    std::vector<uint16_t> dense(expected_weight.size());
    std::vector<uint16_t> online(static_cast<size_t>(batch) * y_stride);
    std::vector<uint16_t> resident(online.size());
    if (cudaMemcpy(dense.data(), d_dense, dense.size() * sizeof(uint16_t),
                   cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(online.data(), d_online_y, online.size() * sizeof(uint16_t),
                   cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(resident.data(), d_resident_y,
                   resident.size() * sizeof(uint16_t),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        fail("resident FP16 projection copy");
        return false;
    }
    if (dense != expected_weight) {
        fail("resident FP16 weight expansion layout");
        return true;
    }

    double direct_abs = 0.0;
    double direct_rel = 0.0;
    double online_reference = 0.0;
    double resident_reference = 0.0;
    for (int sample = 0; sample < batch; ++sample) {
        for (int row = 0; row < rows; ++row) {
            float raw_sum = 0.0f;
            float rounded_sum = 0.0f;
            for (int col = 0; col < cols; ++col) {
                const float activation = from_half(
                    x[static_cast<size_t>(sample) * x_stride + col]);
                const float block_scale = from_half(
                    scale[static_cast<size_t>(row / 128) * scale_stride +
                          col / 128]);
                raw_sum += activation *
                    from_fp8_e4m3(weight[static_cast<size_t>(row) *
                                               weight_stride + col]) *
                    block_scale;
                rounded_sum += activation * from_half(
                    expected_weight[static_cast<size_t>(row) * cols + col]);
            }
            const size_t at = static_cast<size_t>(sample) * y_stride + row;
            const double online_value = from_half(online[at]);
            const double resident_value = from_half(resident[at]);
            const double difference = std::fabs(online_value - resident_value);
            direct_abs = std::max(direct_abs, difference);
            const double magnitude = std::max(
                std::fabs(online_value), std::fabs(resident_value));
            if (magnitude >= 0.25) {
                direct_rel = std::max(direct_rel, difference / magnitude);
            }
            online_reference = std::max(
                online_reference,
                std::fabs(online_value - static_cast<double>(raw_sum)));
            resident_reference = std::max(
                resident_reference,
                std::fabs(resident_value - static_cast<double>(rounded_sum)));
        }
    }
    if (direct_abs > 3.0e-2 || online_reference > 3.0e-2 ||
        resident_reference > 3.0e-2) {
        fail("resident FP16 projection numerical check");
    } else {
        std::printf("  resident FP16 projection batch=%d rows=%d cols=%d "
                    "online_vs_resident_abs=%.3e rel=%.3e "
                    "online_ref=%.3e resident_ref=%.3e\n",
                    batch, rows, cols, direct_abs, direct_rel,
                    online_reference, resident_reference);
    }
    return true;
}

bool check_fp8_prefill_tail(int cols) {
    constexpr int batch = 128;
    constexpr int rows = 128;
    constexpr int x_stride = 320;
    constexpr int y_stride = 136;
    constexpr int weight_stride = 320;
    const int scale_stride = (cols + 127) / 128;
    std::mt19937 rng(7000 + cols);
    std::uniform_real_distribution<float> x_dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> scale_dist(0.005f, 0.02f);
    auto random_code = [&]() {
        // Avoid the E4M3 NaN encodings so the scalar reference and device
        // conversion both stay finite.
        return static_cast<uint8_t>(32 + (rng() % 192));
    };
    std::vector<uint16_t> x(static_cast<size_t>(batch) * x_stride);
    std::vector<uint8_t> weight(static_cast<size_t>(rows) * weight_stride);
    std::vector<uint16_t> scale(static_cast<size_t>((rows + 127) / 128) * scale_stride);
    for (uint16_t& value : x) value = to_half(x_dist(rng));
    for (uint8_t& value : weight) value = random_code();
    for (uint16_t& value : scale) value = to_half(scale_dist(rng));
    std::vector<float> expected(static_cast<size_t>(batch) * rows);
    for (int b = 0; b < batch; ++b) {
        for (int r = 0; r < rows; ++r) {
            float sum = 0.0f;
            for (int c = 0; c < cols; ++c) {
                sum += from_half(x[static_cast<size_t>(b) * x_stride + c]) *
                    from_fp8_e4m3(weight[static_cast<size_t>(r) * weight_stride + c]) *
                    from_half(scale[static_cast<size_t>(r / 128) * scale_stride + c / 128]);
            }
            expected[static_cast<size_t>(b) * rows + r] = sum;
        }
    }
    DeviceBuffer dx, dw, ds, dy;
    uint16_t* px = dx.allocate<uint16_t>(x.size());
    uint8_t* pw = dw.allocate<uint8_t>(weight.size());
    uint16_t* ps = ds.allocate<uint16_t>(scale.size());
    uint16_t* py = dy.allocate<uint16_t>(static_cast<size_t>(batch) * y_stride);
    if (!px || !pw || !ps || !py ||
        cudaMemcpy(px, x.data(), x.size() * sizeof(uint16_t), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(pw, weight.data(), weight.size(), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(ps, scale.data(), scale.size() * sizeof(uint16_t), cudaMemcpyHostToDevice) != cudaSuccess) {
        fail("FP8 prefill tail allocation/copy");
        return false;
    }
    setenv("QWEN_FP8_F16_SMALL_BATCH", "0", 1);
    unsetenv("QWEN_FP8_F16_PREFILL_CUBLAS");
    setenv("QWEN_FP8_F16_PREFILL_WIDE_N64", "1", 1);
    const bool launched = pocket::qwen_fp8_e4m3_fp16scale_matmul_rows_f16_cuda(
        px, pw, ps, py, batch, rows, cols, x_stride, y_stride,
        weight_stride, scale_stride) && cudaDeviceSynchronize() == cudaSuccess;
    unsetenv("QWEN_FP8_F16_SMALL_BATCH");
    unsetenv("QWEN_FP8_F16_PREFILL_WIDE_N64");
    if (!launched) {
        fail("FP8 prefill tail launch");
        return false;
    }
    unsetenv("QWEN_FP8_F16_PREFILL_CUBLAS");
    std::vector<uint16_t> output(static_cast<size_t>(batch) * y_stride);
    if (cudaMemcpy(output.data(), py, output.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost) != cudaSuccess) {
        fail("FP8 prefill tail copy");
        return false;
    }
    double error = 0.0;
    for (int b = 0; b < batch; ++b) {
        for (int r = 0; r < rows; ++r) {
            error = std::max(error, std::fabs(
                static_cast<double>(from_half(output[static_cast<size_t>(b) * y_stride + r])) -
                expected[static_cast<size_t>(b) * rows + r]));
        }
    }
    if (error > 3.0e-2) {
        fail("FP8 prefill tail numerical check");
    } else {
        std::printf("  FP16 FP8 prefill tail cols=%d error=%.3e\n", cols, error);
    }
    return true;
}

bool check_fp8_channel_projection() {
    constexpr int batch = 37;
    constexpr int rows = 41;
    constexpr int cols = 67;
    constexpr int x_stride = 72;
    constexpr int y_stride = 48;
    constexpr int weight_stride = 72;
    std::mt19937 rng(34001);
    std::uniform_real_distribution<float> x_dist(-0.25f, 0.25f);
    std::uniform_real_distribution<float> scale_dist(0.002f, 0.01f);
    std::vector<uint16_t> x(static_cast<size_t>(batch) * x_stride);
    std::vector<uint8_t> weight(static_cast<size_t>(rows) * weight_stride);
    std::vector<uint16_t> scale(rows);
    for (uint16_t& value : x) value = to_half(x_dist(rng));
    for (uint8_t& value : weight) {
        value = static_cast<uint8_t>(32 + (rng() % 192));
    }
    for (uint16_t& value : scale) value = to_half(scale_dist(rng));
    std::vector<float> expected(static_cast<size_t>(batch) * rows);
    for (int sample = 0; sample < batch; ++sample) {
        for (int row = 0; row < rows; ++row) {
            float sum = 0.0f;
            for (int col = 0; col < cols; ++col) {
                sum += from_half(x[static_cast<size_t>(sample) * x_stride + col]) *
                    from_fp8_e4m3(weight[static_cast<size_t>(row) *
                                             weight_stride + col]);
            }
            expected[static_cast<size_t>(sample) * rows + row] =
                sum * from_half(scale[row]);
        }
    }
    DeviceBuffer dx, dw, ds, dy_half, dy_float;
    uint16_t* px = dx.allocate<uint16_t>(x.size());
    uint8_t* pw = dw.allocate<uint8_t>(weight.size());
    uint16_t* ps = ds.allocate<uint16_t>(scale.size());
    uint16_t* py_half = dy_half.allocate<uint16_t>(
        static_cast<size_t>(batch) * y_stride);
    float* py_float = dy_float.allocate<float>(
        static_cast<size_t>(batch) * y_stride);
    if (!px || !pw || !ps || !py_half || !py_float ||
        cudaMemcpy(px, x.data(), x.size() * sizeof(uint16_t),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(pw, weight.data(), weight.size(),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(ps, scale.data(), scale.size() * sizeof(uint16_t),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        !pocket::qwen_fp8_e4m3_channel_matmul_rows_f16_cuda(
            px, pw, ps, py_half, batch, rows, cols, x_stride, y_stride,
            weight_stride) ||
        !pocket::qwen_fp8_e4m3_channel_matmul_rows_f16_f32_cuda(
            px, pw, ps, py_float, batch, rows, cols, x_stride, y_stride,
            weight_stride) || cudaDeviceSynchronize() != cudaSuccess) {
        fail("FP8 channel projection launch");
        return true;
    }
    std::vector<uint16_t> got_half(static_cast<size_t>(batch) * y_stride);
    std::vector<float> got_float(static_cast<size_t>(batch) * y_stride);
    cudaMemcpy(got_half.data(), py_half,
               got_half.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(got_float.data(), py_float,
               got_float.size() * sizeof(float), cudaMemcpyDeviceToHost);
    double half_error = 0.0;
    double float_error = 0.0;
    int worst_half_sample = 0;
    int worst_half_row = 0;
    int worst_float_sample = 0;
    int worst_float_row = 0;
    for (int sample = 0; sample < batch; ++sample) {
        for (int row = 0; row < rows; ++row) {
            const size_t output_at = static_cast<size_t>(sample) * y_stride + row;
            const float reference =
                expected[static_cast<size_t>(sample) * rows + row];
            const double current_half = std::fabs(
                static_cast<double>(from_half(got_half[output_at])) - reference);
            const double current_float = std::fabs(
                static_cast<double>(got_float[output_at]) - reference);
            if (current_half > half_error) {
                half_error = current_half;
                worst_half_sample = sample;
                worst_half_row = row;
            }
            if (current_float > float_error) {
                float_error = current_float;
                worst_float_sample = sample;
                worst_float_row = row;
            }
        }
    }
    if (half_error > 3.0e-2 || float_error > 2.0e-4) {
        const size_t half_at = static_cast<size_t>(worst_half_sample) * y_stride +
            worst_half_row;
        const size_t float_at = static_cast<size_t>(worst_float_sample) * y_stride +
            worst_float_row;
        std::printf("    worst_half sample=%d row=%d got=%.6e expected=%.6e\n",
                    worst_half_sample, worst_half_row,
                    static_cast<double>(from_half(got_half[half_at])),
                    static_cast<double>(expected[
                        static_cast<size_t>(worst_half_sample) * rows +
                        worst_half_row]));
        std::printf("    worst_float sample=%d row=%d got=%.6e expected=%.6e\n",
                    worst_float_sample, worst_float_row,
                    static_cast<double>(got_float[float_at]),
                    static_cast<double>(expected[
                        static_cast<size_t>(worst_float_sample) * rows +
                        worst_float_row]));
        std::printf("  FP8 channel mismatch batch=%d rows=%d cols=%d half=%.3e float=%.3e\n",
                    batch, rows, cols, half_error, float_error);
        fail("FP8 channel projection numerical check");
    } else {
        std::printf("  FP8 channel batch=%d rows=%d cols=%d half=%.3e float=%.3e\n",
                    batch, rows, cols, half_error, float_error);
    }
    return true;
}

bool check_batched_argmax() {
    constexpr int rows = 5;
    constexpr int count = 263;
    constexpr int token_offset = 1000;
    std::vector<float> logits(static_cast<size_t>(rows) * count, -10.0f);
    logits[7] = 3.0f;
    logits[2] = 2.5f;
    logits[static_cast<size_t>(count) + 5] = 4.0f;
    logits[static_cast<size_t>(count) + 9] = 4.0f;
    logits[static_cast<size_t>(2) * count + 262] = 9.0f;
    logits[static_cast<size_t>(2) * count + 3] = 8.0f;
    logits[static_cast<size_t>(3) * count] = -1.0f;
    logits[static_cast<size_t>(3) * count + 11] = -2.0f;
    logits[static_cast<size_t>(4) * count + 129] = 0.5f;
    logits[static_cast<size_t>(4) * count + 130] = 0.25f;
    DeviceBuffer d_logits, d_tokens, d_values;
    float* device_logits = d_logits.allocate<float>(logits.size());
    int* device_tokens = d_tokens.allocate<int>(rows);
    float* device_values = d_values.allocate<float>(rows);
    if (!device_logits || !device_tokens || !device_values ||
        cudaMemcpy(device_logits, logits.data(), logits.size() * sizeof(float),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        !pocket::argmax_fp32_rows_cuda(device_logits, device_tokens, device_values,
                                     rows, count, token_offset) ||
        cudaDeviceSynchronize() != cudaSuccess) {
        fail("batched argmax launch");
        return false;
    }
    std::vector<int> tokens(rows);
    std::vector<float> values(rows);
    if (cudaMemcpy(tokens.data(), device_tokens, rows * sizeof(int),
                   cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(values.data(), device_values, rows * sizeof(float),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        fail("batched argmax copy");
        return false;
    }
    const std::vector<int> expected_tokens = {1007, 1005, 1262, 1000, 1129};
    const std::vector<float> expected_values = {3.0f, 4.0f, 9.0f, -1.0f, 0.5f};
    if (tokens != expected_tokens || values != expected_values) {
        fail("batched argmax numerical/tie check");
    } else {
        std::printf("  batched argmax rows=%d count=%d tie=lower-token\n",
                    rows, count);
    }
    return true;
}

bool check_region_copy() {
    constexpr uint8_t kGuard = 0xa7;
    std::vector<size_t> sizes;
    sizes.reserve(97);
    for (int index = 0; index < 48; ++index) {
        sizes.push_back(index == 17 ? 786432 + 13 : 786432);
        sizes.push_back(index == 9 ? 15360 + 7 : 15360);
    }
    sizes.push_back(10240 + 5);

    std::vector<DeviceBuffer> buffers(sizes.size());
    std::vector<uint8_t*> device_regions(sizes.size());
    std::vector<std::vector<uint8_t>> expected(sizes.size());
    std::vector<pocket::QwenCopyRegion> descriptors;
    descriptors.reserve(sizes.size());
    uint64_t packed_bytes = 0;
    uint64_t total_blocks = 0;
    for (size_t region = 0; region < sizes.size(); ++region) {
        const size_t bytes = sizes[region];
        device_regions[region] = buffers[region].allocate<uint8_t>(bytes + 32);
        if (device_regions[region] == nullptr) return false;
        expected[region].resize(bytes);
        for (size_t index = 0; index < bytes; ++index) {
            expected[region][index] = static_cast<uint8_t>(
                (region * 131 + index * 17 + (index >> 8)) & 0xffu);
        }
        std::vector<uint8_t> initial(bytes + 32, kGuard);
        std::copy(expected[region].begin(), expected[region].end(),
                  initial.begin() + 16);
        if (cudaMemcpy(device_regions[region], initial.data(), initial.size(),
                       cudaMemcpyHostToDevice) != cudaSuccess) {
            fail("region copy source upload");
            return true;
        }
        device_regions[region] += 16;
        pocket::QwenCopyRegion descriptor;
        descriptor.device_address = reinterpret_cast<uint64_t>(
            device_regions[region]);
        descriptor.packed_offset = packed_bytes;
        descriptor.bytes = bytes;
        descriptor.first_block = total_blocks;
        descriptors.push_back(descriptor);
        packed_bytes += bytes;
        total_blocks += pocket::qwen_copy_region_blocks(bytes);
    }

    DeviceBuffer ddescriptors, dpacked;
    auto* d_regions = ddescriptors.allocate<pocket::QwenCopyRegion>(
        descriptors.size());
    uint8_t* d_packed = dpacked.allocate<uint8_t>(packed_bytes + 32);
    if (!d_regions || !d_packed ||
        cudaMemcpy(d_regions, descriptors.data(),
                   descriptors.size() * sizeof(descriptors[0]),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemset(d_packed, kGuard, packed_bytes + 32) != cudaSuccess ||
        !pocket::qwen_gather_copy_regions(
            d_regions, static_cast<int>(descriptors.size()), d_packed + 16,
            total_blocks) ||
        cudaDeviceSynchronize() != cudaSuccess) {
        fail("region gather launch");
        return true;
    }

    std::vector<uint8_t> packed(packed_bytes + 32);
    if (cudaMemcpy(packed.data(), d_packed, packed.size(),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        fail("region gather copy");
        return true;
    }
    for (size_t index = 0; index < 16; ++index) {
        if (packed[index] != kGuard || packed[16 + packed_bytes + index] != kGuard) {
            fail("region gather guard check");
            return true;
        }
    }
    for (size_t region = 0; region < descriptors.size(); ++region) {
        const uint8_t* got = packed.data() + 16 +
            descriptors[region].packed_offset;
        if (!std::equal(expected[region].begin(), expected[region].end(), got)) {
            fail("region gather numerical check");
            return true;
        }
    }

    for (size_t region = 0; region < descriptors.size(); ++region) {
        if (cudaMemset(device_regions[region], 0, sizes[region]) != cudaSuccess) {
            fail("region scatter reset");
            return true;
        }
    }
    if (!pocket::qwen_scatter_copy_regions(
            d_regions, static_cast<int>(descriptors.size()), d_packed + 16,
            total_blocks) ||
        cudaDeviceSynchronize() != cudaSuccess) {
        fail("region scatter launch");
        return true;
    }
    for (size_t region = 0; region < descriptors.size(); ++region) {
        std::vector<uint8_t> got(sizes[region] + 32);
        if (cudaMemcpy(got.data(), device_regions[region] - 16, got.size(),
                       cudaMemcpyDeviceToHost) != cudaSuccess) {
            fail("region scatter copy");
            return true;
        }
        if (!std::all_of(got.begin(), got.begin() + 16,
                         [](uint8_t value) { return value == kGuard; }) ||
            !std::all_of(got.end() - 16, got.end(),
                         [](uint8_t value) { return value == kGuard; }) ||
            !std::equal(expected[region].begin(), expected[region].end(),
                        got.begin() + 16)) {
            fail("region scatter numerical check");
            return true;
        }
    }
    std::printf("  region gather/scatter regions=%zu bytes=%llu bit exact\n",
                descriptors.size(),
                static_cast<unsigned long long>(packed_bytes));
    return true;
}

bool check_strided_row_copy(int rows) {
    constexpr int source_stride = 7;
    constexpr int destination_stride = 19;
    constexpr int columns = 5;
    constexpr int destination_offset = 3;
    std::vector<uint16_t> source(static_cast<size_t>(rows) * source_stride);
    std::vector<uint16_t> destination(static_cast<size_t>(rows) * destination_stride,
                                      to_half(-7.0f));
    for (int row = 0; row < rows; ++row) {
        for (int column = 0; column < source_stride; ++column) {
            source[static_cast<size_t>(row) * source_stride + column] =
                to_half(static_cast<float>(row * 100 + column));
        }
    }
    DeviceBuffer dsource, ddestination;
    uint16_t* d_source = dsource.allocate<uint16_t>(source.size());
    uint16_t* d_destination = ddestination.allocate<uint16_t>(destination.size());
    if (!d_source || !d_destination) return false;
    if (cudaMemcpy(d_source, source.data(), source.size() * sizeof(uint16_t),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_destination, destination.data(),
                   destination.size() * sizeof(uint16_t),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        !pocket::qwen_copy_rows_strided_f16(
            d_source, source_stride, d_destination + destination_offset,
            destination_stride, rows, columns) ||
        cudaDeviceSynchronize() != cudaSuccess ||
        cudaMemcpy(destination.data(), d_destination,
                   destination.size() * sizeof(uint16_t),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        fail("FP16 strided row copy launch");
        return true;
    }
    for (int row = 0; row < rows; ++row) {
        for (int column = 0; column < destination_stride; ++column) {
            const bool copied = column >= destination_offset &&
                column < destination_offset + columns;
            const uint16_t expected = copied
                ? source[static_cast<size_t>(row) * source_stride +
                         column - destination_offset]
                : to_half(-7.0f);
            if (destination[static_cast<size_t>(row) * destination_stride + column] !=
                expected) {
                fail("FP16 strided row copy numerical check");
                return true;
            }
        }
    }
    std::printf("  FP16 strided row copy rows=%d\n", rows);
    return true;
}

// The decode fused gate/up SwiGLU matvec is the largest kernel in a decode
// step, so it has a vectorized variant that widens FP8 codes with bit
// arithmetic rather than a table. That shortcut only holds for E4M3's normal
// codes, so cover every code including the exponent-zero subnormals and both
// signs, and check the vectorized result against the scalar kernel it replaces.
bool check_fp8_swiglu_decode_vectorized() {
    // 5120 is the real Qwen3.8 MLP input width. 2052 is not a multiple of 128,
    // so it exercises a partial scale block, and 300 leaves a ragged tail that
    // the vectorized loop hands to the scalar path.
    struct Case {
        int rows;
        int cols;
    };
    const Case cases[] = {{4352, 5120}, {136, 2052}, {72, 300}};
    std::mt19937 rng(5150);
    std::uniform_real_distribution<float> x_dist(-0.5f, 0.5f);
    std::uniform_real_distribution<float> scale_dist(0.01f, 0.05f);

    for (const Case& item : cases) {
        const int weight_stride = item.cols;
        const int scale_stride = (item.cols + 127) / 128;
        const size_t weight_elements =
            static_cast<size_t>(item.rows) * weight_stride;
        std::vector<uint16_t> host_x(item.cols);
        std::vector<uint8_t> host_gate(weight_elements);
        std::vector<uint8_t> host_up(weight_elements);
        std::vector<uint16_t> host_gate_scale(
            static_cast<size_t>((item.rows + 127) / 128) * scale_stride);
        std::vector<uint16_t> host_up_scale(host_gate_scale.size());
        for (uint16_t& value : host_x) value = to_half(x_dist(rng));
        // Walk all 256 codes so the subnormal and NaN encodings are covered at
        // every lane position rather than sampled.
        for (size_t i = 0; i < weight_elements; ++i) {
            host_gate[i] = static_cast<uint8_t>(i & 0xffu);
            host_up[i] = static_cast<uint8_t>((i * 7u + 3u) & 0xffu);
        }
        for (uint16_t& value : host_gate_scale) value = to_half(scale_dist(rng));
        for (uint16_t& value : host_up_scale) value = to_half(scale_dist(rng));

        DeviceBuffer dx, dg, du, dgs, dus, d_vec, d_scalar;
        uint16_t* px = dx.allocate<uint16_t>(host_x.size());
        uint8_t* pg = dg.allocate<uint8_t>(host_gate.size());
        uint8_t* pu = du.allocate<uint8_t>(host_up.size());
        uint16_t* pgs = dgs.allocate<uint16_t>(host_gate_scale.size());
        uint16_t* pus = dus.allocate<uint16_t>(host_up_scale.size());
        uint16_t* py_vec = d_vec.allocate<uint16_t>(item.rows);
        uint16_t* py_scalar = d_scalar.allocate<uint16_t>(item.rows);
        if (!px || !pg || !pu || !pgs || !pus || !py_vec || !py_scalar ||
            cudaMemcpy(px, host_x.data(), host_x.size() * sizeof(uint16_t),
                       cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(pg, host_gate.data(), host_gate.size(),
                       cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(pu, host_up.data(), host_up.size(),
                       cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(pgs, host_gate_scale.data(),
                       host_gate_scale.size() * sizeof(uint16_t),
                       cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(pus, host_up_scale.data(),
                       host_up_scale.size() * sizeof(uint16_t),
                       cudaMemcpyHostToDevice) != cudaSuccess) {
            fail("FP8 SwiGLU decode vectorized allocation/copy");
            return true;
        }

        // Multi-row dispatch has its own kernel; pin the single-row path so the
        // comparison is vectorized against scalar and nothing else.
        setenv("QWEN_FP8_F16_MULTIROW", "0", 1);
        setenv("QWEN_FP8_F16_SWIGLU_VECTORIZE", "1", 1);
        const bool vec_ok = pocket::qwen_fp8_e4m3_fp16scale_swiglu_matvec_f16_cuda(
                                px, pg, pgs, pu, pus, py_vec, item.rows,
                                item.cols, weight_stride, scale_stride) &&
                            cudaDeviceSynchronize() == cudaSuccess;
        setenv("QWEN_FP8_F16_SWIGLU_VECTORIZE", "0", 1);
        const bool scalar_ok = pocket::qwen_fp8_e4m3_fp16scale_swiglu_matvec_f16_cuda(
                                   px, pg, pgs, pu, pus, py_scalar, item.rows,
                                   item.cols, weight_stride, scale_stride) &&
                               cudaDeviceSynchronize() == cudaSuccess;
        unsetenv("QWEN_FP8_F16_SWIGLU_VECTORIZE");
        unsetenv("QWEN_FP8_F16_MULTIROW");
        if (!vec_ok || !scalar_ok) {
            fail("FP8 SwiGLU decode vectorized launch");
            return true;
        }

        std::vector<uint16_t> got(item.rows), want(item.rows);
        cudaMemcpy(got.data(), py_vec, got.size() * sizeof(uint16_t),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(want.data(), py_scalar, want.size() * sizeof(uint16_t),
                   cudaMemcpyDeviceToHost);

        double worst_vs_scalar = 0.0;
        double worst_vs_host = 0.0;
        double worst_relative = 0.0;
        for (int row = 0; row < item.rows; ++row) {
            double gate = 0.0;
            double up = 0.0;
            const size_t base = static_cast<size_t>(row) * weight_stride;
            const size_t scale_base =
                static_cast<size_t>(row / 128) * scale_stride;
            for (int col = 0; col < item.cols; ++col) {
                const double xv = from_half(host_x[col]);
                gate += xv * from_fp8_e4m3(host_gate[base + col]) *
                        from_half(host_gate_scale[scale_base + col / 128]);
                up += xv * from_fp8_e4m3(host_up[base + col]) *
                      from_half(host_up_scale[scale_base + col / 128]);
            }
            const double reference =
                (gate / (1.0 + std::exp(-gate))) * up;
            const double vectorized = from_half(got[row]);
            const double scalar = from_half(want[row]);
            if (!std::isfinite(vectorized) || !std::isfinite(scalar)) {
                fail("FP8 SwiGLU decode vectorized produced a non-finite row");
                return true;
            }
            worst_vs_scalar = std::max(worst_vs_scalar,
                                       std::fabs(vectorized - scalar));
            worst_vs_host = std::max(worst_vs_host,
                                     std::fabs(vectorized - reference));
            const double magnitude = std::max(std::fabs(reference), 1.0e-3);
            worst_relative = std::max(worst_relative,
                                      std::fabs(vectorized - reference) / magnitude);
        }
        // The vectorized kernel sums four products before applying the block
        // scale, so it is not bit exact against the scalar kernel; hold it to
        // the same relative accuracy instead.
        if (worst_relative > 5.0e-3) {
            fail("FP8 SwiGLU decode vectorized numerical check");
            return true;
        }
        std::printf("  FP8 SwiGLU decode vectorized rows=%d cols=%d "
                    "vs_scalar=%.3e vs_host=%.3e rel=%.3e\n",
                    item.rows, item.cols, worst_vs_scalar, worst_vs_host,
                    worst_relative);
    }
    return true;
}

// The FP8 fused gate/up SwiGLU kernel runs on every speculative verify batch.
// Its shared-activation variant must stay bit exact against the register-only
// kernel or greedy near ties would diverge from plain decode.
bool check_fp8_fused_swiglu_shared() {
    struct Case {
        int batch;
        int rows;
        int cols;
    };
    // 5120 is the real Qwen3.8 MLP input width; 300 and 2052 cover a partial
    // final tile and a non-128-multiple vectorized width.
    const Case cases[] = {
        {2, 130, 5120}, {3, 96, 5120}, {4, 72, 2052},
        {5, 130, 300},  {8, 136, 5120}, {8, 40, 2052},
    };
    std::mt19937 rng(9182);
    std::uniform_real_distribution<float> x_dist(-0.5f, 0.5f);
    std::uniform_real_distribution<float> scale_dist(0.01f, 0.05f);
    auto random_code = [&rng]() {
        return static_cast<uint8_t>(32 + (rng() % 192));
    };
    for (const Case& item : cases) {
        const int x_stride = item.cols;
        const int y_stride = item.rows;
        const int weight_stride = item.cols;
        const int scale_stride = (item.cols + 127) / 128;
        std::vector<uint16_t> host_x(static_cast<size_t>(item.batch) * x_stride);
        std::vector<uint8_t> host_gate(
            static_cast<size_t>(item.rows) * weight_stride);
        std::vector<uint8_t> host_up(host_gate.size());
        std::vector<uint16_t> host_gate_scale(
            static_cast<size_t>((item.rows + 127) / 128) * scale_stride);
        std::vector<uint16_t> host_up_scale(host_gate_scale.size());
        for (uint16_t& value : host_x) value = to_half(x_dist(rng));
        for (uint8_t& value : host_gate) value = random_code();
        for (uint8_t& value : host_up) value = random_code();
        for (uint16_t& value : host_gate_scale) value = to_half(scale_dist(rng));
        for (uint16_t& value : host_up_scale) value = to_half(scale_dist(rng));
        DeviceBuffer dx, dg, du, dgs, dus, d_shared, d_plain;
        uint16_t* px = dx.allocate<uint16_t>(host_x.size());
        uint8_t* pg = dg.allocate<uint8_t>(host_gate.size());
        uint8_t* pu = du.allocate<uint8_t>(host_up.size());
        uint16_t* pgs = dgs.allocate<uint16_t>(host_gate_scale.size());
        uint16_t* pus = dus.allocate<uint16_t>(host_up_scale.size());
        const size_t out_elements = static_cast<size_t>(item.batch) * y_stride;
        uint16_t* py_shared = d_shared.allocate<uint16_t>(out_elements);
        uint16_t* py_plain = d_plain.allocate<uint16_t>(out_elements);
        if (!px || !pg || !pu || !pgs || !pus || !py_shared || !py_plain ||
            cudaMemcpy(px, host_x.data(), host_x.size() * sizeof(uint16_t),
                       cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(pg, host_gate.data(), host_gate.size(),
                       cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(pu, host_up.data(), host_up.size(),
                       cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(pgs, host_gate_scale.data(),
                       host_gate_scale.size() * sizeof(uint16_t),
                       cudaMemcpyHostToDevice) != cudaSuccess ||
            cudaMemcpy(pus, host_up_scale.data(),
                       host_up_scale.size() * sizeof(uint16_t),
                       cudaMemcpyHostToDevice) != cudaSuccess) {
            fail("FP8 fused SwiGLU shared allocation/copy");
            return true;
        }
        setenv("QWEN_FP8_F16_SMALL_BATCH_SHARED", "1", 1);
        const bool shared_ok =
            pocket::qwen_fp8_e4m3_fp16scale_swiglu_small_batch_f16_cuda(
                px, pg, pgs, pu, pus, py_shared, item.batch, item.rows,
                item.cols, x_stride, y_stride, weight_stride, scale_stride) &&
            cudaDeviceSynchronize() == cudaSuccess;
        setenv("QWEN_FP8_F16_SMALL_BATCH_SHARED", "0", 1);
        const bool plain_ok =
            pocket::qwen_fp8_e4m3_fp16scale_swiglu_small_batch_f16_cuda(
                px, pg, pgs, pu, pus, py_plain, item.batch, item.rows,
                item.cols, x_stride, y_stride, weight_stride, scale_stride) &&
            cudaDeviceSynchronize() == cudaSuccess;
        unsetenv("QWEN_FP8_F16_SMALL_BATCH_SHARED");
        if (!shared_ok || !plain_ok) {
            fail("FP8 fused SwiGLU shared launch");
            return true;
        }
        std::vector<uint16_t> got(out_elements);
        std::vector<uint16_t> want(out_elements);
        cudaMemcpy(got.data(), py_shared, out_elements * sizeof(uint16_t),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(want.data(), py_plain, out_elements * sizeof(uint16_t),
                   cudaMemcpyDeviceToHost);
        for (int sample = 0; sample < item.batch; ++sample) {
            for (int row = 0; row < item.rows; ++row) {
                const size_t at = static_cast<size_t>(sample) * y_stride + row;
                if (got[at] != want[at]) {
                    std::printf("  FP8 fused SwiGLU shared mismatch batch=%d "
                                "rows=%d cols=%d at sample=%d row=%d "
                                "%.6e vs %.6e\n",
                                item.batch, item.rows, item.cols, sample, row,
                                static_cast<double>(from_half(got[at])),
                                static_cast<double>(from_half(want[at])));
                    fail("FP8 fused SwiGLU shared bit exactness");
                    return true;
                }
            }
        }
    }
    std::printf("  FP8 fused SwiGLU shared bit exact over %zu shard shapes\n",
                sizeof(cases) / sizeof(cases[0]));
    return true;
}

bool check_fused_swiglu() {
    constexpr int batch = 8;
    constexpr int rows = 17;
    constexpr int cols = 12;
    std::mt19937 rng(731);
    std::uniform_real_distribution<float> dist(-0.25f, 0.25f);
    std::vector<uint16_t> input(static_cast<size_t>(batch) * cols);
    std::vector<uint16_t> gate(static_cast<size_t>(rows) * cols);
    std::vector<uint16_t> up(static_cast<size_t>(rows) * cols);
    for (uint16_t& value : input) value = to_half(dist(rng));
    for (uint16_t& value : gate) value = to_half(dist(rng));
    for (uint16_t& value : up) value = to_half(dist(rng));
    DeviceBuffer dinput, dgate, dup, doutput;
    uint16_t* d_input = dinput.allocate<uint16_t>(input.size());
    uint16_t* d_gate = dgate.allocate<uint16_t>(gate.size());
    uint16_t* d_up = dup.allocate<uint16_t>(up.size());
    uint16_t* d_output = doutput.allocate<uint16_t>(static_cast<size_t>(batch) * rows);
    if (!d_input || !d_gate || !d_up || !d_output) return false;
    if (cudaMemcpy(d_input, input.data(), input.size() * sizeof(uint16_t),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_gate, gate.data(), gate.size() * sizeof(uint16_t),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_up, up.data(), up.size() * sizeof(uint16_t),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        !pocket::qwen_fp16_swiglu_matmul_rows_f16(
            d_input, d_gate, d_up, d_output, batch, rows, cols, cols, rows,
            cols) || cudaDeviceSynchronize() != cudaSuccess) {
        fail("FP16 fused SwiGLU launch");
        return true;
    }
    std::vector<uint16_t> output(static_cast<size_t>(batch) * rows);
    if (cudaMemcpy(output.data(), d_output, output.size() * sizeof(uint16_t),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        fail("FP16 fused SwiGLU copy");
        return true;
    }
    double worst = 0.0;
    for (int sample = 0; sample < batch; ++sample) {
        for (int row = 0; row < rows; ++row) {
            float gate_sum = 0.0f;
            float up_sum = 0.0f;
            for (int column = 0; column < cols; ++column) {
                const float x = from_half(input[static_cast<size_t>(sample) * cols + column]);
                gate_sum += x * from_half(gate[static_cast<size_t>(row) * cols + column]);
                up_sum += x * from_half(up[static_cast<size_t>(row) * cols + column]);
            }
            const float expected = gate_sum / (1.0f + std::exp(-gate_sum)) * up_sum;
            worst = std::max(worst, std::fabs(static_cast<double>(
                from_half(output[static_cast<size_t>(sample) * rows + row])) - expected));
        }
    }
    if (worst > 2.0e-3) fail("FP16 fused SwiGLU numerical check");
    else std::printf("  FP16 fused SwiGLU batch=%d rows=%d worst=%.3e\n",
                     batch, rows, worst);
    return true;
}

bool check_gated_delta_prenormalized(int rows = 8, int heads = 12,
                                     int key_heads = 4, int passes = 1,
                                     float scale = 0.2f) {
    constexpr int dim = 128;
    const size_t state_elements = static_cast<size_t>(heads) * dim * dim;
    const size_t key_elements = static_cast<size_t>(rows) * key_heads * dim;
    const size_t value_elements = static_cast<size_t>(rows) * heads * dim;
    const size_t gate_elements = static_cast<size_t>(rows) * heads;
    std::mt19937 rng(17001 + rows * 31 + heads * 7 + key_heads);
    std::uniform_real_distribution<float> dist(-scale, scale);
    std::vector<float> state(state_elements);
    for (float& value : state) value = dist(rng);
    std::vector<uint16_t> q(key_elements), k(key_elements), v(value_elements);
    std::vector<uint16_t> g(gate_elements), beta(gate_elements);
    for (uint16_t& value : q) value = to_half(dist(rng));
    for (uint16_t& value : k) value = to_half(dist(rng));
    for (uint16_t& value : v) value = to_half(dist(rng));
    for (uint16_t& value : g) value = to_half(dist(rng) - 0.3f);
    for (uint16_t& value : beta) value = to_half(dist(rng) + 0.5f);

    float* d_state_reference = nullptr;
    float* d_state_normalized = nullptr;
    float* d_state_shared = nullptr;
    uint16_t* d_shared = nullptr;
    uint16_t* d_q = nullptr;
    uint16_t* d_k = nullptr;
    uint16_t* d_v = nullptr;
    uint16_t* d_g = nullptr;
    uint16_t* d_beta = nullptr;
    uint16_t* d_reference = nullptr;
    uint16_t* d_normalized = nullptr;
    float* d_q_normalized = nullptr;
    float* d_k_normalized = nullptr;
    auto malloc_copy = [](void** device, const void* host, size_t bytes) {
        return cudaMalloc(device, bytes) == cudaSuccess &&
               cudaMemcpy(*device, host, bytes, cudaMemcpyHostToDevice) == cudaSuccess;
    };
    if (!malloc_copy(reinterpret_cast<void**>(&d_state_reference), state.data(),
                     state.size() * sizeof(float)) ||
        !malloc_copy(reinterpret_cast<void**>(&d_state_normalized), state.data(),
                     state.size() * sizeof(float)) ||
        !malloc_copy(reinterpret_cast<void**>(&d_state_shared), state.data(),
                     state.size() * sizeof(float)) ||
        !malloc_copy(reinterpret_cast<void**>(&d_q), q.data(),
                     q.size() * sizeof(uint16_t)) ||
        !malloc_copy(reinterpret_cast<void**>(&d_k), k.data(),
                     k.size() * sizeof(uint16_t)) ||
        !malloc_copy(reinterpret_cast<void**>(&d_v), v.data(),
                     v.size() * sizeof(uint16_t)) ||
        !malloc_copy(reinterpret_cast<void**>(&d_g), g.data(),
                     g.size() * sizeof(uint16_t)) ||
        !malloc_copy(reinterpret_cast<void**>(&d_beta), beta.data(),
                     beta.size() * sizeof(uint16_t)) ||
        cudaMalloc(&d_reference, value_elements * sizeof(uint16_t)) != cudaSuccess ||
        cudaMalloc(&d_normalized, value_elements * sizeof(uint16_t)) != cudaSuccess ||
        cudaMalloc(&d_shared, value_elements * sizeof(uint16_t)) != cudaSuccess ||
        cudaMalloc(&d_q_normalized, key_elements * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_k_normalized, key_elements * sizeof(float)) != cudaSuccess) {
        fail("gated delta prenormalized allocation");
        return false;
    }
    const float q_scale = 1.0f / std::sqrt(static_cast<float>(dim));
    // Repeated passes reuse the evolving state, which is what a multi-round
    // speculative verify does. A kernel that mishandles the state load/store
    // shows up on the second pass even if the first one looks right.
    for (int pass = 0; pass < passes; ++pass) {
        if (!pocket::qwen_gated_delta_sequence_f16(
                d_state_reference, d_q, d_k, d_v, d_g, d_beta, d_reference,
                rows, heads, key_heads, dim, dim, q_scale) ||
            !pocket::qwen_normalize_gated_delta_qk_f16(
                d_q, d_k, d_q_normalized, d_k_normalized, rows, key_heads, dim) ||
            !pocket::qwen_gated_delta_sequence_normalized_f16(
                d_state_normalized, d_q_normalized, d_k_normalized, d_v, d_g,
                d_beta, d_normalized, rows, heads, key_heads, dim, dim, q_scale) ||
            !pocket::qwen_gated_delta_sequence_normalized_shared_f16(
                d_state_shared, d_q_normalized, d_k_normalized, d_v, d_g,
                d_beta, d_shared, rows, heads, key_heads, dim, dim, q_scale) ||
            cudaDeviceSynchronize() != cudaSuccess) {
            fail("gated delta prenormalized launch");
            return false;
        }
    }
    std::vector<uint16_t> reference(value_elements), normalized(value_elements);
    std::vector<uint16_t> shared(value_elements);
    std::vector<float> reference_state(state_elements), normalized_state(state_elements);
    std::vector<float> shared_state(state_elements);
    cudaMemcpy(reference.data(), d_reference,
               reference.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(normalized.data(), d_normalized,
               normalized.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(shared.data(), d_shared,
               shared.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(reference_state.data(), d_state_reference,
               reference_state.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(normalized_state.data(), d_state_normalized,
               normalized_state.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(shared_state.data(), d_state_shared,
               shared_state.size() * sizeof(float), cudaMemcpyDeviceToHost);
    double output_worst = 0.0;
    double shared_output_worst = 0.0;
    bool finite = true;
    for (size_t i = 0; i < reference.size(); ++i) {
        const double base = from_half(reference[i]);
        const double candidate = from_half(shared[i]);
        if (!std::isfinite(candidate)) finite = false;
        output_worst = std::max(
            output_worst, std::fabs(base - from_half(normalized[i])));
        shared_output_worst = std::max(shared_output_worst,
                                       std::fabs(base - candidate));
    }
    double state_worst = 0.0;
    double shared_state_worst = 0.0;
    for (size_t i = 0; i < reference_state.size(); ++i) {
        if (!std::isfinite(shared_state[i])) finite = false;
        state_worst = std::max(
            state_worst,
            std::fabs(static_cast<double>(reference_state[i]) -
                      normalized_state[i]));
        shared_state_worst = std::max(
            shared_state_worst,
            std::fabs(static_cast<double>(reference_state[i]) -
                      shared_state[i]));
    }
    if (output_worst != 0.0 || state_worst != 0.0) {
        fail("gated delta prenormalized bit exact");
    } else if (!finite) {
        fail("gated delta shared-state finite");
    } else if (shared_output_worst != 0.0 || shared_state_worst != 0.0) {
        fail("gated delta shared-state bit exact");
    } else {
        std::printf("  gated delta prenormalized rows=%d heads=%d key_heads=%d "
                    "passes=%d output=%.3e state=%.3e shared_output=%.3e "
                    "shared_state=%.3e\n",
                    rows, heads, key_heads, passes, output_worst, state_worst,
                    shared_output_worst, shared_state_worst);
    }
    cudaFree(d_state_reference);
    cudaFree(d_state_normalized);
    cudaFree(d_state_shared);
    cudaFree(d_shared);
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_g);
    cudaFree(d_beta);
    cudaFree(d_reference);
    cudaFree(d_normalized);
    cudaFree(d_q_normalized);
    cudaFree(d_k_normalized);
    return true;
}

// The per-token step kernel and the sequence kernel must agree exactly. Decode
// runs one token at a time, so whichever one the engine picks at rows == 1 has
// to produce the same state and output as the other, otherwise switching the
// gate would silently change generated tokens. The two differ only in where the
// recurrent state lives: the step kernel round-trips it through global memory
// while the sequence kernel holds it in registers.
bool check_gated_delta_step_matches_sequence(int rows = 1, int heads = 12,
                                            int key_heads = 4, int passes = 3,
                                            float scale = 0.2f) {
    constexpr int dim = 128;
    const size_t state_elements = static_cast<size_t>(heads) * dim * dim;
    const size_t key_elements = static_cast<size_t>(rows) * key_heads * dim;
    const size_t value_elements = static_cast<size_t>(rows) * heads * dim;
    const size_t gate_elements = static_cast<size_t>(rows) * heads;
    std::mt19937 rng(24101 + rows * 13 + heads * 5 + key_heads);
    std::uniform_real_distribution<float> dist(-scale, scale);
    std::vector<float> state(state_elements);
    for (float& value : state) value = dist(rng);
    std::vector<uint16_t> q(key_elements), k(key_elements), v(value_elements);
    std::vector<uint16_t> g(gate_elements), beta(gate_elements);
    for (uint16_t& value : q) value = to_half(dist(rng));
    for (uint16_t& value : k) value = to_half(dist(rng));
    for (uint16_t& value : v) value = to_half(dist(rng));
    for (uint16_t& value : g) value = to_half(dist(rng) - 0.3f);
    for (uint16_t& value : beta) value = to_half(dist(rng) + 0.5f);

    float* d_state_step = nullptr;
    float* d_state_sequence = nullptr;
    uint16_t* d_q = nullptr;
    uint16_t* d_k = nullptr;
    uint16_t* d_v = nullptr;
    uint16_t* d_g = nullptr;
    uint16_t* d_beta = nullptr;
    uint16_t* d_step = nullptr;
    uint16_t* d_sequence = nullptr;
    auto malloc_copy = [](void** device, const void* host, size_t bytes) {
        return cudaMalloc(device, bytes) == cudaSuccess &&
               cudaMemcpy(*device, host, bytes, cudaMemcpyHostToDevice) == cudaSuccess;
    };
    if (!malloc_copy(reinterpret_cast<void**>(&d_state_step), state.data(),
                     state.size() * sizeof(float)) ||
        !malloc_copy(reinterpret_cast<void**>(&d_state_sequence), state.data(),
                     state.size() * sizeof(float)) ||
        !malloc_copy(reinterpret_cast<void**>(&d_q), q.data(),
                     q.size() * sizeof(uint16_t)) ||
        !malloc_copy(reinterpret_cast<void**>(&d_k), k.data(),
                     k.size() * sizeof(uint16_t)) ||
        !malloc_copy(reinterpret_cast<void**>(&d_v), v.data(),
                     v.size() * sizeof(uint16_t)) ||
        !malloc_copy(reinterpret_cast<void**>(&d_g), g.data(),
                     g.size() * sizeof(uint16_t)) ||
        !malloc_copy(reinterpret_cast<void**>(&d_beta), beta.data(),
                     beta.size() * sizeof(uint16_t)) ||
        cudaMalloc(&d_step, value_elements * sizeof(uint16_t)) != cudaSuccess ||
        cudaMalloc(&d_sequence, value_elements * sizeof(uint16_t)) != cudaSuccess) {
        fail("gated delta step-vs-sequence allocation");
        return false;
    }
    const float q_scale = 1.0f / std::sqrt(static_cast<float>(dim));
    for (int pass = 0; pass < passes; ++pass) {
        bool launched = true;
        for (int row = 0; row < rows && launched; ++row) {
            launched = pocket::qwen_gated_delta_step_f16_cuda(
                d_state_step,
                d_q + static_cast<size_t>(row) * key_heads * dim,
                d_k + static_cast<size_t>(row) * key_heads * dim,
                d_v + static_cast<size_t>(row) * heads * dim,
                d_g + static_cast<size_t>(row) * heads,
                d_beta + static_cast<size_t>(row) * heads,
                d_step + static_cast<size_t>(row) * heads * dim,
                heads, key_heads, dim, dim, q_scale);
        }
        if (!launched ||
            !pocket::qwen_gated_delta_sequence_f16_cuda(
                d_state_sequence, d_q, d_k, d_v, d_g, d_beta, d_sequence,
                rows, heads, key_heads, dim, dim, q_scale) ||
            cudaDeviceSynchronize() != cudaSuccess) {
            fail("gated delta step-vs-sequence launch");
            return false;
        }
    }
    std::vector<uint16_t> step(value_elements), sequence(value_elements);
    std::vector<float> step_state(state_elements), sequence_state(state_elements);
    cudaMemcpy(step.data(), d_step, step.size() * sizeof(uint16_t),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(sequence.data(), d_sequence, sequence.size() * sizeof(uint16_t),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(step_state.data(), d_state_step,
               step_state.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(sequence_state.data(), d_state_sequence,
               sequence_state.size() * sizeof(float), cudaMemcpyDeviceToHost);
    double output_worst = 0.0;
    double state_worst = 0.0;
    bool finite = true;
    for (size_t i = 0; i < step.size(); ++i) {
        const double candidate = from_half(sequence[i]);
        if (!std::isfinite(candidate)) finite = false;
        output_worst = std::max(output_worst,
                                std::fabs(from_half(step[i]) - candidate));
    }
    for (size_t i = 0; i < step_state.size(); ++i) {
        if (!std::isfinite(sequence_state[i])) finite = false;
        state_worst = std::max(
            state_worst, std::fabs(static_cast<double>(step_state[i]) -
                                   sequence_state[i]));
    }
    if (!finite) {
        fail("gated delta step-vs-sequence finite");
    } else if (output_worst != 0.0 || state_worst != 0.0) {
        fail("gated delta step-vs-sequence bit exact");
    } else {
        std::printf("  gated delta step-vs-sequence rows=%d heads=%d "
                    "key_heads=%d passes=%d output=%.3e state=%.3e\n",
                    rows, heads, key_heads, passes, output_worst, state_worst);
    }
    cudaFree(d_state_step);
    cudaFree(d_state_sequence);
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_g);
    cudaFree(d_beta);
    cudaFree(d_step);
    cudaFree(d_sequence);
    return true;
}

bool check_fp8_cache() {
    constexpr int q_heads = 6;
    constexpr int kv_heads = 1;
    constexpr int head_dim = 64;
    constexpr int context_len = 17;
    constexpr int max_context = 32;
    constexpr int scale_block = 64;
    std::mt19937 rng(5678);
    std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
    std::vector<uint16_t> q(q_heads * head_dim);
    std::vector<uint16_t> k(context_len * head_dim);
    std::vector<uint16_t> v(k.size());
    for (uint16_t& item : q) item = to_half(dist(rng));
    for (uint16_t& item : k) item = to_half(dist(rng));
    for (uint16_t& item : v) item = to_half(dist(rng));
    DeviceBuffer dq, dk_rows, dv_rows, dk_cache, dv_cache, dks, dvs, dout, dscores;
    uint16_t* d_q = dq.allocate<uint16_t>(q.size());
    uint16_t* d_k_rows = dk_rows.allocate<uint16_t>(k.size());
    uint16_t* d_v_rows = dv_rows.allocate<uint16_t>(v.size());
    uint8_t* d_k_cache = dk_cache.allocate<uint8_t>(max_context * head_dim);
    uint8_t* d_v_cache = dv_cache.allocate<uint8_t>(max_context * head_dim);
    uint16_t* d_k_scale = dks.allocate<uint16_t>(max_context);
    uint16_t* d_v_scale = dvs.allocate<uint16_t>(max_context);
    uint16_t* d_out = dout.allocate<uint16_t>(q.size());
    float* d_scores = dscores.allocate<float>(q_heads * context_len);
    if (!d_q || !d_k_rows || !d_v_rows || !d_k_cache || !d_v_cache || !d_k_scale || !d_v_scale || !d_out || !d_scores) return false;
    cudaMemcpy(d_q, q.data(), q.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k_rows, k.data(), k.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v_rows, v.data(), v.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
    if (!pocket::qwen_append_kv_cache_fp8_cuda(d_k_rows, d_v_rows, d_k_cache, d_v_cache,
                                               d_k_scale, d_v_scale, context_len,
                                               kv_heads, head_dim, scale_block, 0,
                                               max_context) ||
        !pocket::qwen_gqa_decode_attention_fp8_cuda(d_q, d_k_cache, d_v_cache,
                                                   d_k_scale, d_v_scale, d_out,
                                                   d_scores, q_heads, kv_heads,
                                                   head_dim, scale_block,
                                                   context_len, max_context) ||
        cudaDeviceSynchronize() != cudaSuccess) {
        fail("FP8 cache launch");
        return true;
    }
    std::vector<uint16_t> got(q.size());
    cudaMemcpy(got.data(), d_out, got.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    std::vector<uint8_t> kq(k.size()), vq(v.size());
    std::vector<uint16_t> ks(context_len), vs(context_len);
    cudaMemcpy(ks.data(), d_k_scale, ks.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(vs.data(), d_v_scale, vs.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(kq.data(), d_k_cache, kq.size() * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(vq.data(), d_v_cache, vq.size() * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    double worst = 0.0;
    const float query_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    for (int head = 0; head < q_heads; ++head) {
        std::vector<double> scores(context_len);
        double maximum = -1.0e300;
        for (int pos = 0; pos < context_len; ++pos) {
            double dot = 0.0;
            const float key_scale = from_half(ks[pos]);
            for (int d = 0; d < head_dim; ++d) {
                dot += static_cast<double>(from_half(q[head * head_dim + d])) *
                       (from_fp8_e4m3(kq[pos * head_dim + d]) * key_scale);
            }
            scores[pos] = dot * query_scale;
            maximum = std::max(maximum, scores[pos]);
        }
        double denominator = 0.0;
        for (double& score : scores) {
            score = std::exp(score - maximum);
            denominator += score;
        }
        for (int d = 0; d < head_dim; ++d) {
            double value = 0.0;
            for (int pos = 0; pos < context_len; ++pos) {
                value += scores[pos] * (from_fp8_e4m3(vq[pos * head_dim + d]) * from_half(vs[pos]));
            }
            const float expected = static_cast<float>(value / denominator);
            worst = std::max(worst, std::fabs(static_cast<double>(from_half(got[head * head_dim + d])) - expected));
        }
    }
    // The quantization itself is checked against the exact max-abs scale.
    for (int pos = 0; pos < context_len; ++pos) {
        float max_k = 0.0f;
        float max_v = 0.0f;
        for (int d = 0; d < head_dim; ++d) {
            max_k = std::max(max_k, std::fabs(from_half(k[pos * head_dim + d])));
            max_v = std::max(max_v, std::fabs(from_half(v[pos * head_dim + d])));
        }
        if (std::fabs(from_half(ks[pos]) - (max_k > 0.0f ? max_k / 448.0f : 1.0f)) > 2.0e-3f ||
            std::fabs(from_half(vs[pos]) - (max_v > 0.0f ? max_v / 448.0f : 1.0f)) > 2.0e-3f) {
            fail("FP8 cache scale check");
            break;
        }
    }
    if (worst > 0.35) fail("FP8 cache numerical check");
    else std::printf("  FP8 cache context=%d worst=%.3e\n", context_len, worst);
    return true;
}

// Decode LM head. The matvec kernel splits each row across a warp instead of a
// block, so its summation order differs from the reference path and the two are
// compared on tolerance against a CPU double reference rather than for equality.
// The greedy pick is compared as well, since that is what actually reaches the
// caller: a logit spread that does not move the argmax cannot change a token.
bool check_fp16_matvec_logits(int rows, int cols) {
    std::mt19937 rng(0x5eed1234u ^ (static_cast<unsigned>(rows) << 8) ^
                     static_cast<unsigned>(cols));
    std::uniform_real_distribution<float> dist(-0.08f, 0.08f);
    std::vector<uint16_t> x(static_cast<size_t>(cols));
    std::vector<uint16_t> weight(static_cast<size_t>(rows) * cols);
    for (uint16_t& v : x) v = to_half(dist(rng));
    for (uint16_t& v : weight) v = to_half(dist(rng));

    DeviceBuffer dx, dw, d_reference, d_matvec;
    uint16_t* px = dx.allocate<uint16_t>(x.size());
    uint16_t* pw = dw.allocate<uint16_t>(weight.size());
    float* p_reference = d_reference.allocate<float>(static_cast<size_t>(rows));
    float* p_matvec = d_matvec.allocate<float>(static_cast<size_t>(rows));
    if (!px || !pw || !p_reference || !p_matvec ||
        cudaMemcpy(px, x.data(), x.size() * sizeof(uint16_t),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(pw, weight.data(), weight.size() * sizeof(uint16_t),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        fail("FP16 matvec logits allocation/copy");
        return true;
    }
    if (!pocket::qwen_fp16_matmul_rows_f16_f32_cuda(
            px, pw, p_reference, 1, rows, cols, cols, rows, cols) ||
        !pocket::qwen_fp16_matvec_rows_f16_f32_cuda(
            px, pw, p_matvec, 1, rows, cols, cols, rows, cols) ||
        cudaDeviceSynchronize() != cudaSuccess) {
        fail("FP16 matvec logits launch");
        return true;
    }
    std::vector<float> reference(static_cast<size_t>(rows));
    std::vector<float> candidate(reference.size());
    cudaMemcpy(reference.data(), p_reference, reference.size() * sizeof(float),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(candidate.data(), p_matvec, candidate.size() * sizeof(float),
               cudaMemcpyDeviceToHost);

    double worst_reference = 0.0;
    double worst_candidate = 0.0;
    bool finite = true;
    int best_reference = 0;
    int best_candidate = 0;
    for (int row = 0; row < rows; ++row) {
        double expected = 0.0;
        for (int col = 0; col < cols; ++col) {
            expected += static_cast<double>(from_half(x[col])) *
                        from_half(weight[static_cast<size_t>(row) * cols + col]);
        }
        if (!std::isfinite(candidate[row]) || !std::isfinite(reference[row])) {
            finite = false;
        }
        worst_reference = std::max(worst_reference,
                                   std::fabs(reference[row] - expected));
        worst_candidate = std::max(worst_candidate,
                                   std::fabs(candidate[row] - expected));
        if (reference[row] > reference[best_reference]) best_reference = row;
        if (candidate[row] > candidate[best_candidate]) best_candidate = row;
    }
    // Batched shapes must be refused, not silently run on the reference path.
    const bool rejects_batch = !pocket::qwen_fp16_matvec_rows_f16_f32_cuda(
        px, pw, p_matvec, 2, rows, cols, cols, rows, cols);
    if (!finite || worst_reference > 2.0e-3 || worst_candidate > 2.0e-3 ||
        best_reference != best_candidate || !rejects_batch) {
        std::printf("  rows=%d cols=%d ref=%.3e cand=%.3e argmax %d/%d batch_refused=%d\n",
                    rows, cols, worst_reference, worst_candidate,
                    best_reference, best_candidate,
                    static_cast<int>(rejects_batch));
        fail("FP16 matvec logits numerical check");
    } else {
        std::printf("  fp16 matvec logits rows=%6d cols=%5d ref=%.3e cand=%.3e "
                    "argmax=%d\n", rows, cols, worst_reference, worst_candidate,
                    best_candidate);
    }
    return true;
}

}  // namespace

int main() {
    if (!pocket::cuda_runtime_available()) {
        std::printf("[SKIP] test_qwen_half_ops requires a CUDA device\n");
        return 0;
    }
    check_fp16_cache(1);
    check_fp16_cache(17);
    check_fp16_cache(333);
    check_residual_add_rmsnorm_fused();
    check_prefill_tiled(1, 64, 0);
    check_prefill_tiled(2, 256, 0);
    check_prefill_tiled(7, 64, 3);
    check_prefill_tiled(17, 256, 5);
    check_prefill_tiled(128, 64, 7);
    check_prefill_tiled(333, 256, 11);
    check_prefill_tiled(17, 128, 5, 8, 2);
    check_prefill_tiled(33, 256, 7, 32, 8);
    check_fp8_prefill_tail(301);
    check_fp8_prefill_tail(302);
    check_fp8_prefill_tail(303);
    check_verify_split(2, 64, 0);
    check_verify_split(3, 256, 37);
    check_verify_split(5, 64, 4093);
    check_verify_split(8, 256, 8191);
    check_decode_fused(4096, 64);
    check_decode_fused(8192, 256);
    check_decode_fused(32768, 64);
    check_decode_window_reference();
    check_decode_grid_256k();
    check_fp8_f16_projection();
    check_fp8_swiglu_decode_vectorized();
    check_resident_fp16_projection();
    check_fp8_channel_projection();
    check_batched_argmax();
    check_region_copy();
    check_strided_row_copy(1);
    check_strided_row_copy(8);
    check_fused_swiglu();
    check_fp8_fused_swiglu_shared();
    check_gated_delta_prenormalized();
    check_gated_delta_prenormalized(2);
    check_gated_delta_prenormalized(3);
    check_gated_delta_prenormalized(5);
    check_gated_delta_prenormalized(7);
    check_gated_delta_prenormalized(8, 12, 2);
    check_gated_delta_prenormalized(8, 12, 12);
    check_gated_delta_prenormalized(8, 4, 1);
    check_gated_delta_prenormalized(8, 12, 4, 3);
    // rows == 1 is the decode shape the engine gate now routes to the sequence
    // kernel; the wider rows guard the verify shapes that still use the step loop.
    check_gated_delta_step_matches_sequence(1);
    check_gated_delta_step_matches_sequence(2);
    check_gated_delta_step_matches_sequence(4);
    check_gated_delta_step_matches_sequence(1, 12, 12);
    check_gated_delta_step_matches_sequence(1, 4, 1);
    check_gated_delta_prenormalized(8, 12, 4, 3, 2.0f);
    // Decode LM head. 62080x5120 is the real TP4 shard; the odd shapes cover a
    // row count that does not fill the last warp pair and a column count that
    // leaves a scalar tail past the 8-wide vector body.
    check_fp16_matvec_logits(62080, 5120);
    check_fp16_matvec_logits(1021, 5120);
    check_fp16_matvec_logits(255, 516);
    check_fp8_cache();
    if (failures != 0) {
        std::printf("test_qwen_half_ops failures=%d\n", failures);
        return 1;
    }
    std::printf("[PASS] test_qwen_half_ops\n");
    return 0;
}
