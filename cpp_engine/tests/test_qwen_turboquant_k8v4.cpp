// Numerical test for TurboQuant K8V4 KV-cache against reference.
#include "qwen_ops.hpp"
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

void check_cuda(cudaError_t status, const char* context) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", context, cudaGetErrorString(status));
        std::exit(1);
    }
}

uint16_t f16(float v) {
    __half x = __float2half(v);
    uint16_t b = 0;
    std::memcpy(&b, &x, 2);
    return b;
}

float f16_to_f32(uint16_t b) {
    __half x;
    std::memcpy(&x, &b, 2);
    return __half2float(x);
}

float dequantize_value_nibble(uint8_t nibble, float scale, float minimum) {
    return static_cast<float>(nibble) * scale + minimum;
}

// Dequantize one slot into FP32 K/V so the reference attention consumes exactly
// the bytes the kernel reads. Any mismatch is then kernel math, not quantization.
void dequantize_slot(const uint8_t* slot, int head_dim, std::vector<float>& k_out,
                     std::vector<float>& v_out) {
    k_out.resize(head_dim);
    v_out.resize(head_dim);

    uint16_t scale_bits, min_bits;
    const int value_bytes = head_dim / 2;
    std::memcpy(&scale_bits, slot + head_dim + value_bytes, 2);
    std::memcpy(&min_bits, slot + head_dim + value_bytes + 2, 2);
    const float scale = f16_to_f32(scale_bits);
    const float minimum = f16_to_f32(min_bits);

    for (int d = 0; d < head_dim; ++d) {
        __half k_h = __nv_cvt_fp8_to_halfraw(slot[d], __NV_E5M2);
        k_out[d] = __half2float(k_h);

        const uint8_t packed = slot[head_dim + (d >> 1)];
        const uint8_t nibble = (d & 1) ? (packed >> 4) : (packed & 0x0F);
        v_out[d] = dequantize_value_nibble(nibble, scale, minimum);
    }
}

// Reference attention over the dequantized cache, in FP32 with a plain
// max-subtracted softmax. `visible` selects the attended positions.
template <typename VisibleFn>
void reference_attention(const std::vector<uint8_t>& cache, const uint16_t* q_row,
                         int head, int kv_head, int kv_heads, int head_dim,
                         int slot_bytes, int context_len, VisibleFn visible,
                         std::vector<float>& out) {
    std::vector<float> scores(context_len, -INFINITY);
    std::vector<float> k_buf, v_buf;

    const float q_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    float max_score = -INFINITY;

    for (int pos = 0; pos < context_len; ++pos) {
        if (!visible(pos)) continue;
        const size_t slot_offset =
            (static_cast<size_t>(pos) * kv_heads + kv_head) * slot_bytes;
        dequantize_slot(cache.data() + slot_offset, head_dim, k_buf, v_buf);

        double dot = 0.0;
        for (int d = 0; d < head_dim; ++d) {
            dot += static_cast<double>(f16_to_f32(q_row[d])) * k_buf[d];
        }
        scores[pos] = static_cast<float>(dot) * q_scale;
        max_score = std::max(max_score, scores[pos]);
    }

    out.assign(head_dim, 0.0f);
    double sum = 0.0;
    for (int pos = 0; pos < context_len; ++pos) {
        if (!std::isfinite(scores[pos])) continue;
        const double weight = std::exp(static_cast<double>(scores[pos] - max_score));
        sum += weight;

        const size_t slot_offset =
            (static_cast<size_t>(pos) * kv_heads + kv_head) * slot_bytes;
        dequantize_slot(cache.data() + slot_offset, head_dim, k_buf, v_buf);
        for (int d = 0; d < head_dim; ++d) {
            out[d] += static_cast<float>(weight * v_buf[d]);
        }
    }
    if (sum > 0.0) {
        for (int d = 0; d < head_dim; ++d) out[d] /= static_cast<float>(sum);
    }
}

void test_store_and_decode(int seq_len, int kv_heads, int q_heads, int start_pos,
                          int head_dim) {
    std::printf("Testing store+decode: seq_len=%d kv_heads=%d q_heads=%d start_pos=%d head_dim=%d\n",
                seq_len, kv_heads, q_heads, start_pos, head_dim);

    const int kHeadDim = head_dim;
    const int kSlotBytes = head_dim + head_dim / 2 + 4;
    const int max_context = start_pos + seq_len + 16;

    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    const size_t kv_size = static_cast<size_t>(seq_len) * kv_heads * kHeadDim;
    const size_t q_size = static_cast<size_t>(q_heads) * kHeadDim;
    const size_t cache_size = static_cast<size_t>(max_context) * kv_heads * kSlotBytes;

    std::vector<uint16_t> h_k(kv_size);
    std::vector<uint16_t> h_v(kv_size);
    std::vector<uint16_t> h_q(q_size);
    std::vector<float> h_k_f32(kv_size);
    std::vector<float> h_v_f32(kv_size);

    for (size_t i = 0; i < kv_size; ++i) {
        h_k[i] = f16(dist(rng) * 0.1f);
        h_v[i] = f16(dist(rng) * 0.1f);
        // The kernel only ever sees the FP16 inputs, so the reference for
        // quantization error is the FP16 value, not the pre-rounding float.
        h_k_f32[i] = f16_to_f32(h_k[i]);
        h_v_f32[i] = f16_to_f32(h_v[i]);
    }
    for (size_t i = 0; i < q_size; ++i) {
        h_q[i] = f16(dist(rng) * 0.1f);
    }

    uint16_t *d_k = nullptr, *d_v = nullptr, *d_q = nullptr;
    uint8_t *d_cache = nullptr;
    uint16_t *d_out = nullptr;
    float *d_score_scratch = nullptr;

    check_cuda(cudaMalloc(&d_k, kv_size * sizeof(uint16_t)), "k");
    check_cuda(cudaMalloc(&d_v, kv_size * sizeof(uint16_t)), "v");
    check_cuda(cudaMalloc(&d_q, q_size * sizeof(uint16_t)), "q");
    check_cuda(cudaMalloc(&d_cache, cache_size), "cache");
    check_cuda(cudaMalloc(&d_out, q_size * sizeof(uint16_t)), "out");
    check_cuda(cudaMalloc(&d_score_scratch, q_heads * max_context * sizeof(float)), "scratch");

    check_cuda(cudaMemcpy(d_k, h_k.data(), kv_size * sizeof(uint16_t),
                          cudaMemcpyHostToDevice), "copy k");
    check_cuda(cudaMemcpy(d_v, h_v.data(), kv_size * sizeof(uint16_t),
                          cudaMemcpyHostToDevice), "copy v");
    check_cuda(cudaMemcpy(d_q, h_q.data(), q_size * sizeof(uint16_t),
                          cudaMemcpyHostToDevice), "copy q");
    check_cuda(cudaMemset(d_cache, 0, cache_size), "zero cache");

    // Store K/V into TurboQuant cache.
    bool ok_store = pocket::qwen_append_kv_cache_turboquant_k8v4_cuda(
        d_k, d_v, d_cache, seq_len, kv_heads, kHeadDim, start_pos, max_context);
    if (!ok_store) {
        std::fprintf(stderr, "Store kernel launch failed\n");
        std::exit(1);
    }
    check_cuda(cudaDeviceSynchronize(), "sync store");

    // Read back cache and verify slot layout.
    std::vector<uint8_t> h_cache(cache_size);
    check_cuda(cudaMemcpy(h_cache.data(), d_cache, cache_size,
                          cudaMemcpyDeviceToHost), "read cache");

    for (int token = 0; token < seq_len; ++token) {
        for (int head = 0; head < kv_heads; ++head) {
            const int cache_pos = start_pos + token;
            const size_t slot_offset = (static_cast<size_t>(cache_pos) * kv_heads + head) * kSlotBytes;
            const uint8_t* slot = h_cache.data() + slot_offset;

            // Read FP16 scale/min metadata.
            uint16_t scale_bits, min_bits;
            std::memcpy(&scale_bits, slot + kHeadDim + kHeadDim / 2, 2);
            std::memcpy(&min_bits, slot + kHeadDim + kHeadDim / 2 + 2, 2);
            float scale = f16_to_f32(scale_bits);
            float minimum = f16_to_f32(min_bits);

            if (!std::isfinite(scale) || !std::isfinite(minimum)) {
                std::fprintf(stderr, "FAIL: non-finite metadata at token=%d head=%d\n", token, head);
                std::exit(1);
            }

            // Verify key E5M2 round-trip.
            const size_t kv_offset = (static_cast<size_t>(token) * kv_heads + head) * kHeadDim;
            float max_key_error = 0.0f;
            for (int d = 0; d < kHeadDim; ++d) {
                uint8_t k_fp8 = slot[d];
                __half k_h = __nv_cvt_fp8_to_halfraw(k_fp8, __NV_E5M2);
                float k_decoded = __half2float(k_h);
                float k_orig = h_k_f32[kv_offset + d];
                float error = std::fabs(k_decoded - k_orig);
                if (error > max_key_error) max_key_error = error;
            }

            // Verify value 4-bit round-trip within quantization error.
            float max_value_error = 0.0f;
            for (int d = 0; d < kHeadDim; ++d) {
                int byte_idx = kHeadDim + (d >> 1);
                uint8_t packed = slot[byte_idx];
                uint8_t nibble = (d & 1) ? (packed >> 4) : (packed & 0x0F);
                float v_decoded = dequantize_value_nibble(nibble, scale, minimum);
                float v_orig = h_v_f32[kv_offset + d];
                float error = std::fabs(v_decoded - v_orig);
                if (error > max_value_error) max_value_error = error;
            }

            // E5M2 has 2-bit mantissa, expect ~1e-2 key error.
            // 4-bit uniform quantization: max error = scale/2.
            const float key_tol = 2e-2f;
            // Uniform 4-bit quantization caps the error at half a step; allow a
            // small slack for the FP16 rounding of the reconstruction itself.
            const float value_tol = scale * 0.5f + scale * 1e-2f + 1e-6f;
            if (max_key_error > key_tol) {
                std::fprintf(stderr, "FAIL: key error %.6e exceeds tolerance %.6e\n",
                             max_key_error, key_tol);
                std::exit(1);
            }
            if (max_value_error > value_tol) {
                std::fprintf(stderr, "FAIL: value error %.6e exceeds tolerance %.6e\n",
                             max_value_error, value_tol);
                std::exit(1);
            }
        }
    }

    // Decode attention with TurboQuant cache.
    const int context_len = start_pos + seq_len;
    bool ok_decode = pocket::qwen_gqa_decode_attention_turboquant_k8v4_cuda(
        d_q, d_cache, d_out, d_score_scratch, q_heads, kv_heads, kHeadDim,
        context_len, max_context, 0, 0);
    if (!ok_decode) {
        std::fprintf(stderr, "Decode kernel launch failed\n");
        std::exit(1);
    }
    check_cuda(cudaDeviceSynchronize(), "sync decode");

    std::vector<uint16_t> h_out(q_size);
    check_cuda(cudaMemcpy(h_out.data(), d_out, q_size * sizeof(uint16_t),
                          cudaMemcpyDeviceToHost), "read out");

    // Compare against the reference over the same quantized bytes.
    const int group = q_heads / kv_heads;
    std::vector<float> expected;
    float max_decode_error = 0.0f;
    for (int head = 0; head < q_heads; ++head) {
        reference_attention(h_cache, h_q.data() + static_cast<size_t>(head) * kHeadDim,
                            head, head / group, kv_heads, kHeadDim, kSlotBytes,
                            context_len, [](int) { return true; }, expected);
        for (int d = 0; d < kHeadDim; ++d) {
            const float got = f16_to_f32(h_out[static_cast<size_t>(head) * kHeadDim + d]);
            if (!std::isfinite(got)) {
                std::fprintf(stderr, "FAIL: non-finite decode output head=%d dim=%d\n",
                             head, d);
                std::exit(1);
            }
            max_decode_error = std::max(max_decode_error, std::fabs(got - expected[d]));
        }
    }
    // FP16 output storage plus FP32 accumulation ordering.
    const float decode_tol = 2e-3f;
    if (max_decode_error > decode_tol) {
        std::fprintf(stderr, "FAIL: decode error %.6e exceeds tolerance %.6e\n",
                     max_decode_error, decode_tol);
        std::exit(1);
    }

    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_q);
    cudaFree(d_cache);
    cudaFree(d_out);
    cudaFree(d_score_scratch);

    std::printf("  PASS (round-trip within quantization tol, decode max_error=%.3e)\n",
                max_decode_error);
}

void test_prefill_causal(int seq_len, int kv_heads, int q_heads, int position_offset,
                         int head_dim) {
    std::printf("Testing prefill causal: seq_len=%d kv_heads=%d q_heads=%d position_offset=%d head_dim=%d\n",
                seq_len, kv_heads, q_heads, position_offset, head_dim);

    const int kHeadDim = head_dim;
    const int kSlotBytes = head_dim + head_dim / 2 + 4;
    const int max_context = position_offset + seq_len + 16;

    std::mt19937 rng(54321);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    const size_t kv_size = static_cast<size_t>(seq_len) * kv_heads * kHeadDim;
    const size_t q_size = static_cast<size_t>(seq_len) * q_heads * kHeadDim;
    const size_t cache_size = static_cast<size_t>(max_context) * kv_heads * kSlotBytes;

    std::vector<uint16_t> h_k(kv_size);
    std::vector<uint16_t> h_v(kv_size);
    std::vector<uint16_t> h_q(q_size);

    for (size_t i = 0; i < kv_size; ++i) {
        h_k[i] = f16(dist(rng) * 0.1f);
        h_v[i] = f16(dist(rng) * 0.1f);
    }
    for (size_t i = 0; i < q_size; ++i) {
        h_q[i] = f16(dist(rng) * 0.1f);
    }

    uint16_t *d_k = nullptr, *d_v = nullptr, *d_q = nullptr;
    uint8_t *d_cache = nullptr;
    uint16_t *d_out = nullptr;

    check_cuda(cudaMalloc(&d_k, kv_size * sizeof(uint16_t)), "k");
    check_cuda(cudaMalloc(&d_v, kv_size * sizeof(uint16_t)), "v");
    check_cuda(cudaMalloc(&d_q, q_size * sizeof(uint16_t)), "q");
    check_cuda(cudaMalloc(&d_cache, cache_size), "cache");
    check_cuda(cudaMalloc(&d_out, q_size * sizeof(uint16_t)), "out");

    check_cuda(cudaMemcpy(d_k, h_k.data(), kv_size * sizeof(uint16_t),
                          cudaMemcpyHostToDevice), "copy k");
    check_cuda(cudaMemcpy(d_v, h_v.data(), kv_size * sizeof(uint16_t),
                          cudaMemcpyHostToDevice), "copy v");
    check_cuda(cudaMemcpy(d_q, h_q.data(), q_size * sizeof(uint16_t),
                          cudaMemcpyHostToDevice), "copy q");
    check_cuda(cudaMemset(d_cache, 0, cache_size), "zero cache");

    // Store K/V.
    bool ok_store = pocket::qwen_append_kv_cache_turboquant_k8v4_cuda(
        d_k, d_v, d_cache, seq_len, kv_heads, kHeadDim, position_offset, max_context);
    if (!ok_store) {
        std::fprintf(stderr, "Store kernel launch failed\n");
        std::exit(1);
    }
    check_cuda(cudaDeviceSynchronize(), "sync store");

    // Prefill attention.
    bool ok_prefill = pocket::qwen_gqa_prefill_attention_turboquant_k8v4_cuda(
        d_q, d_cache, d_out, seq_len, q_heads, kv_heads, kHeadDim,
        position_offset, max_context, 0, 0);
    if (!ok_prefill) {
        std::fprintf(stderr, "Prefill kernel launch failed\n");
        std::exit(1);
    }
    check_cuda(cudaDeviceSynchronize(), "sync prefill");

    std::vector<uint16_t> h_out(q_size);
    check_cuda(cudaMemcpy(h_out.data(), d_out, q_size * sizeof(uint16_t),
                          cudaMemcpyDeviceToHost), "read out");

    std::vector<uint8_t> h_cache(cache_size);
    check_cuda(cudaMemcpy(h_cache.data(), d_cache, cache_size,
                          cudaMemcpyDeviceToHost), "read cache");

    const int group = q_heads / kv_heads;
    std::vector<float> expected;
    float max_prefill_error = 0.0f;
    for (int token = 0; token < seq_len; ++token) {
        const int q_pos = position_offset + token;
        for (int head = 0; head < q_heads; ++head) {
            const size_t q_offset =
                (static_cast<size_t>(token) * q_heads + head) * kHeadDim;
            reference_attention(h_cache, h_q.data() + q_offset, head, head / group,
                                kv_heads, kHeadDim, kSlotBytes, q_pos + 1,
                                [](int) { return true; }, expected);
            for (int d = 0; d < kHeadDim; ++d) {
                const float got = f16_to_f32(h_out[q_offset + d]);
                if (!std::isfinite(got)) {
                    std::fprintf(stderr,
                                 "FAIL: non-finite prefill output token=%d head=%d dim=%d\n",
                                 token, head, d);
                    std::exit(1);
                }
                max_prefill_error =
                    std::max(max_prefill_error, std::fabs(got - expected[d]));
            }
        }
    }
    const float prefill_tol = 2e-3f;
    if (max_prefill_error > prefill_tol) {
        std::fprintf(stderr, "FAIL: prefill error %.6e exceeds tolerance %.6e\n",
                     max_prefill_error, prefill_tol);
        std::exit(1);
    }

    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_q);
    cudaFree(d_cache);
    cudaFree(d_out);

    std::printf("  PASS (causal mask vs reference, max_error=%.3e)\n", max_prefill_error);
}

void test_window_and_sink(int context_len, int q_heads, int kv_heads,
                          int attention_window, int attention_sink_tokens,
                          int head_dim) {
    std::printf("Testing window=%d sink=%d context=%d head_dim=%d\n",
                attention_window, attention_sink_tokens, context_len, head_dim);

    const int kHeadDim = head_dim;
    const int kSlotBytes = head_dim + head_dim / 2 + 4;
    const int max_context = context_len + 16;

    std::mt19937 rng(98765);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

    const size_t kv_size = static_cast<size_t>(context_len) * kv_heads * kHeadDim;
    const size_t q_size = static_cast<size_t>(q_heads) * kHeadDim;
    const size_t cache_size = static_cast<size_t>(max_context) * kv_heads * kSlotBytes;

    std::vector<uint16_t> h_k(kv_size);
    std::vector<uint16_t> h_v(kv_size);
    std::vector<uint16_t> h_q(q_size);

    for (size_t i = 0; i < kv_size; ++i) {
        h_k[i] = f16(dist(rng) * 0.1f);
        h_v[i] = f16(dist(rng) * 0.1f);
    }
    for (size_t i = 0; i < q_size; ++i) {
        h_q[i] = f16(dist(rng) * 0.1f);
    }

    uint16_t *d_k = nullptr, *d_v = nullptr, *d_q = nullptr;
    uint8_t *d_cache = nullptr;
    uint16_t *d_out = nullptr;
    float *d_score_scratch = nullptr;

    check_cuda(cudaMalloc(&d_k, kv_size * sizeof(uint16_t)), "k");
    check_cuda(cudaMalloc(&d_v, kv_size * sizeof(uint16_t)), "v");
    check_cuda(cudaMalloc(&d_q, q_size * sizeof(uint16_t)), "q");
    check_cuda(cudaMalloc(&d_cache, cache_size), "cache");
    check_cuda(cudaMalloc(&d_out, q_size * sizeof(uint16_t)), "out");
    check_cuda(cudaMalloc(&d_score_scratch, q_heads * max_context * sizeof(float)), "scratch");

    check_cuda(cudaMemcpy(d_k, h_k.data(), kv_size * sizeof(uint16_t),
                          cudaMemcpyHostToDevice), "copy k");
    check_cuda(cudaMemcpy(d_v, h_v.data(), kv_size * sizeof(uint16_t),
                          cudaMemcpyHostToDevice), "copy v");
    check_cuda(cudaMemcpy(d_q, h_q.data(), q_size * sizeof(uint16_t),
                          cudaMemcpyHostToDevice), "copy q");
    check_cuda(cudaMemset(d_cache, 0, cache_size), "zero cache");

    bool ok_store = pocket::qwen_append_kv_cache_turboquant_k8v4_cuda(
        d_k, d_v, d_cache, context_len, kv_heads, kHeadDim, 0, max_context);
    if (!ok_store) {
        std::fprintf(stderr, "Store kernel launch failed\n");
        std::exit(1);
    }
    check_cuda(cudaDeviceSynchronize(), "sync store");

    bool ok_decode = pocket::qwen_gqa_decode_attention_turboquant_k8v4_cuda(
        d_q, d_cache, d_out, d_score_scratch, q_heads, kv_heads, kHeadDim,
        context_len, max_context, attention_window, attention_sink_tokens);
    if (!ok_decode) {
        std::fprintf(stderr, "Decode kernel launch failed\n");
        std::exit(1);
    }
    check_cuda(cudaDeviceSynchronize(), "sync decode");

    std::vector<uint16_t> h_out(q_size);
    check_cuda(cudaMemcpy(h_out.data(), d_out, q_size * sizeof(uint16_t),
                          cudaMemcpyDeviceToHost), "read out");

    std::vector<uint8_t> h_cache(cache_size);
    check_cuda(cudaMemcpy(h_cache.data(), d_cache, cache_size,
                          cudaMemcpyDeviceToHost), "read cache");

    // Same two-range visibility the kernel applies: sink prefix plus the
    // trailing window.
    const int sink_end = attention_sink_tokens;
    const int window_start = std::max(sink_end, context_len - attention_window);
    auto visible = [sink_end, window_start](int pos) {
        return pos < sink_end || pos >= window_start;
    };

    const int group = q_heads / kv_heads;
    std::vector<float> expected;
    float max_window_error = 0.0f;
    int attended = 0;
    for (int pos = 0; pos < context_len; ++pos) {
        if (visible(pos)) ++attended;
    }
    if (attended >= context_len) {
        std::fprintf(stderr, "FAIL: window test does not actually mask anything\n");
        std::exit(1);
    }

    for (int head = 0; head < q_heads; ++head) {
        reference_attention(h_cache, h_q.data() + static_cast<size_t>(head) * kHeadDim,
                            head, head / group, kv_heads, kHeadDim, kSlotBytes,
                            context_len, visible, expected);
        for (int d = 0; d < kHeadDim; ++d) {
            const float got = f16_to_f32(h_out[static_cast<size_t>(head) * kHeadDim + d]);
            if (!std::isfinite(got)) {
                std::fprintf(stderr, "FAIL: non-finite windowed output head=%d dim=%d\n",
                             head, d);
                std::exit(1);
            }
            max_window_error = std::max(max_window_error, std::fabs(got - expected[d]));
        }
    }
    const float window_tol = 2e-3f;
    if (max_window_error > window_tol) {
        std::fprintf(stderr, "FAIL: window error %.6e exceeds tolerance %.6e\n",
                     max_window_error, window_tol);
        std::exit(1);
    }

    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_q);
    cudaFree(d_cache);
    cudaFree(d_out);
    cudaFree(d_score_scratch);

    std::printf("  PASS (window+sink vs reference, attended=%d/%d, max_error=%.3e)\n",
                attended, context_len, max_window_error);
}

}  // namespace

int main() {
    check_cuda(cudaSetDevice(0), "set device");

    // head_dim 128 and the real Qwen3.5 full-attention head_dim 256.
    for (int head_dim : {128, 256}) {
        test_store_and_decode(8, 4, 12, 0, head_dim);
        test_store_and_decode(32, 4, 12, 0, head_dim);
        test_store_and_decode(16, 4, 12, 64, head_dim);

        test_prefill_causal(8, 4, 12, 0, head_dim);
        test_prefill_causal(32, 4, 12, 0, head_dim);
        test_prefill_causal(16, 4, 12, 48, head_dim);

        test_window_and_sink(512, 12, 4, 128, 16, head_dim);
        test_window_and_sink(1024, 12, 4, 256, 32, head_dim);
    }

    // TP4 rank shape: one KV head per rank, six query heads.
    test_store_and_decode(32, 1, 6, 0, 256);
    test_prefill_causal(32, 1, 6, 0, 256);

    std::printf("All TurboQuant K8V4 tests PASS\n");
    return 0;
}
