// Numerical check for the online-softmax GQA kernels (decode + prefill) against
// a two-pass CPU softmax reference.
//
// The kernels stream the KV cache once and rescale a running max/denominator.
// head_dim=256 and 24-over-4 heads match the Qwen3.8 TP1 full-attention layout.

#include "cuda_ops.hpp"
#include "qwen_ops.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

int failures = 0;

void fail(const std::string& what) {
    std::printf("[FAIL] %s\n", what.c_str());
    ++failures;
}

uint16_t to_half(float value) {
    return __half_as_ushort(__float2half(value));
}

float from_half(uint16_t bits) {
    return __half2float(__ushort_as_half(bits));
}

// Reference: explicit max, exp, normalize over the causal window.
void attention_reference(const std::vector<float>& q, const std::vector<float>& k_cache,
                         const std::vector<float>& v_cache, std::vector<float>& out,
                         int q_heads, int kv_heads, int head_dim, int context_len) {
    const int repeat = q_heads / kv_heads;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    for (int head = 0; head < q_heads; ++head) {
        const int kv_head = head / repeat;
        const float* q_row = q.data() + static_cast<size_t>(head) * head_dim;
        std::vector<double> scores(context_len);
        double max_score = -1.0e300;
        for (int pos = 0; pos < context_len; ++pos) {
            const float* key = k_cache.data() +
                               static_cast<size_t>(pos) * kv_heads * head_dim +
                               static_cast<size_t>(kv_head) * head_dim;
            double dot = 0.0;
            for (int d = 0; d < head_dim; ++d) dot += static_cast<double>(q_row[d]) * key[d];
            scores[pos] = dot * scale;
            max_score = std::max(max_score, scores[pos]);
        }
        double denom = 0.0;
        for (int pos = 0; pos < context_len; ++pos) {
            scores[pos] = std::exp(scores[pos] - max_score);
            denom += scores[pos];
        }
        for (int d = 0; d < head_dim; ++d) {
            double acc = 0.0;
            for (int pos = 0; pos < context_len; ++pos) {
                const float* value = v_cache.data() +
                                     static_cast<size_t>(pos) * kv_heads * head_dim +
                                     static_cast<size_t>(kv_head) * head_dim;
                acc += scores[pos] * value[d];
            }
            out[static_cast<size_t>(head) * head_dim + d] =
                static_cast<float>(acc / (denom > 0.0 ? denom : 1.0));
        }
    }
}

struct Device {
    float* q = nullptr;
    float* k = nullptr;
    float* v = nullptr;
    float* out = nullptr;
    float* scores = nullptr;
    ~Device() {
        cudaFree(q);
        cudaFree(k);
        cudaFree(v);
        cudaFree(out);
        cudaFree(scores);
    }
};

bool run_decode(int q_heads, int kv_heads, int head_dim, int context_len, int max_context) {
    std::mt19937 rng(24680);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> q(static_cast<size_t>(q_heads) * head_dim);
    std::vector<float> k(static_cast<size_t>(max_context) * kv_heads * head_dim, 0.0f);
    std::vector<float> v(k.size(), 0.0f);
    for (float& x : q) x = dist(rng);
    for (size_t i = 0; i < static_cast<size_t>(context_len) * kv_heads * head_dim; ++i) {
        k[i] = dist(rng);
        v[i] = dist(rng);
    }

    Device dev;
    if (cudaMalloc(&dev.q, q.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dev.k, k.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dev.v, v.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dev.out, q.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&dev.scores, static_cast<size_t>(q_heads) * context_len * sizeof(float)) != cudaSuccess) {
        return false;
    }
    cudaMemcpy(dev.q, q.data(), q.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev.k, k.data(), k.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev.v, v.data(), v.size() * sizeof(float), cudaMemcpyHostToDevice);

    if (!pocket::qwen_gqa_decode_attention_cuda(dev.q, dev.k, dev.v, dev.out, dev.scores,
                                              q_heads, kv_heads, head_dim, context_len,
                                              max_context)) {
        fail("gqa decode launch failed");
        return true;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        fail("gqa decode sync failed");
        return true;
    }
    std::vector<float> got(q.size());
    cudaMemcpy(got.data(), dev.out, got.size() * sizeof(float), cudaMemcpyDeviceToHost);

    std::vector<float> want(q.size());
    attention_reference(q, k, v, want, q_heads, kv_heads, head_dim, context_len);
    double worst = 0.0;
    for (size_t i = 0; i < got.size(); ++i) {
        worst = std::max(worst, static_cast<double>(std::fabs(got[i] - want[i])));
    }
    if (worst > 1.0e-4) {
        fail("gqa decode ctx=" + std::to_string(context_len) + " worst=" + std::to_string(worst));
    } else {
        std::printf("  gqa decode heads=%d/%d head_dim=%d ctx=%4d worst=%.3e\n",
                    q_heads, kv_heads, head_dim, context_len, worst);
    }
    return true;
}

// Prefill: compare the tiled kernel (K/V loads shared across GQA heads and query
// rows) against both the reference streaming kernel and the CPU softmax. The
// tiled path is what long-context prefill dispatches to, so it needs its own
// numerical gate rather than inheriting the decode kernel's.
bool run_prefill_tiled(int rows, int q_heads, int kv_heads, int head_dim,
                       int position_offset, int max_context,
                       bool exercise_long_tiles = false) {
    std::mt19937 rng(13579);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    const int context_len = position_offset + rows;
    const size_t q_elements = static_cast<size_t>(rows) * q_heads * head_dim;
    const size_t kv_elements =
        static_cast<size_t>(max_context) * kv_heads * head_dim;
    std::vector<uint16_t> q(q_elements);
    std::vector<uint16_t> k(kv_elements, 0);
    std::vector<uint16_t> v(kv_elements, 0);
    std::vector<float> q_ref(q_elements);
    std::vector<float> k_ref(kv_elements, 0.0f);
    std::vector<float> v_ref(kv_elements, 0.0f);
    for (size_t i = 0; i < q_elements; ++i) {
        q_ref[i] = dist(rng);
        q[i] = to_half(q_ref[i]);
        // Round-trip so the reference sees exactly the FP16 values the kernel does.
        q_ref[i] = from_half(q[i]);
    }
    for (size_t i = 0;
         i < static_cast<size_t>(context_len) * kv_heads * head_dim; ++i) {
        k[i] = to_half(dist(rng));
        v[i] = to_half(dist(rng));
        k_ref[i] = from_half(k[i]);
        v_ref[i] = from_half(v[i]);
    }

    uint16_t* d_q = nullptr;
    uint16_t* d_k = nullptr;
    uint16_t* d_v = nullptr;
    uint16_t* d_tiled = nullptr;
    uint16_t* d_plain = nullptr;
    uint16_t* d_long_qr2 = nullptr;
    uint16_t* d_long = nullptr;
    uint16_t* d_flash = nullptr;
    uint16_t* d_mma = nullptr;
    const bool long_shape = exercise_long_tiles && rows >= 128 &&
        q_heads == 6 && kv_heads == 1 && head_dim == 256 &&
        context_len >= 2048;
    auto release = [&]() {
        cudaFree(d_q); cudaFree(d_k); cudaFree(d_v);
        cudaFree(d_tiled); cudaFree(d_plain); cudaFree(d_long_qr2);
        cudaFree(d_long); cudaFree(d_flash); cudaFree(d_mma);
    };
    if (cudaMalloc(&d_q, q.size() * sizeof(uint16_t)) != cudaSuccess ||
        cudaMalloc(&d_k, k.size() * sizeof(uint16_t)) != cudaSuccess ||
        cudaMalloc(&d_v, v.size() * sizeof(uint16_t)) != cudaSuccess ||
        cudaMalloc(&d_tiled, q.size() * sizeof(uint16_t)) != cudaSuccess ||
        cudaMalloc(&d_plain, q.size() * sizeof(uint16_t)) != cudaSuccess ||
        (long_shape && cudaMalloc(&d_long_qr2, q.size() * sizeof(uint16_t)) !=
            cudaSuccess) ||
        (long_shape && cudaMalloc(&d_long, q.size() * sizeof(uint16_t)) !=
            cudaSuccess) ||
        (long_shape && cudaMalloc(&d_flash, q.size() * sizeof(uint16_t)) !=
            cudaSuccess) ||
        (long_shape && cudaMalloc(&d_mma, q.size() * sizeof(uint16_t)) !=
            cudaSuccess)) {
        release();
        return false;
    }
    cudaMemcpy(d_q, q.data(), q.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, k.data(), k.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v.data(), v.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);

    setenv("POCKETLLM_QWEN_GQA_POS_TILE", "1", 1);
    // An unset LONG_TILE selects hpg6 and an unset MMA_TILE selects the
    // tensor-core kernel in production. Make this reference launch explicitly
    // generic on both so the candidate checks below cannot compare a kernel with
    // itself through inherited or default environment state.
    setenv("POCKETLLM_QWEN_GQA_LONG_TILE", "0", 1);
    setenv("POCKETLLM_QWEN_GQA_MMA_TILE", "0", 1);
    unsetenv("POCKETLLM_QWEN_GQA_FLASH_TILE");
    unsetenv("POCKETLLM_QWEN_GQA_QUERY_ROWS");
    const bool tiled_ok = pocket::qwen_gqa_prefill_attention_f16_tiled_cuda(
        d_q, d_k, d_v, d_tiled, rows, q_heads, kv_heads, head_dim,
        position_offset, max_context, 0, 0);
    bool long_qr2_ok = true;
    bool long_ok = true;
    bool flash_ok = true;
    bool mma_ok = true;
    if (long_shape) {
        setenv("POCKETLLM_QWEN_GQA_LONG_TILE", "1", 1);
        setenv("POCKETLLM_QWEN_GQA_QUERY_ROWS", "2", 1);
        long_qr2_ok = pocket::qwen_gqa_prefill_attention_f16_tiled_cuda(
            d_q, d_k, d_v, d_long_qr2, rows, q_heads, kv_heads, head_dim,
            position_offset, max_context, 0, 0);
        setenv("POCKETLLM_QWEN_GQA_QUERY_ROWS", "4", 1);
        long_ok = pocket::qwen_gqa_prefill_attention_f16_tiled_cuda(
            d_q, d_k, d_v, d_long, rows, q_heads, kv_heads, head_dim,
            position_offset, max_context, 0, 0);
        setenv("POCKETLLM_QWEN_GQA_FLASH_TILE", "1", 1);
        flash_ok = pocket::qwen_gqa_prefill_attention_f16_tiled_cuda(
            d_q, d_k, d_v, d_flash, rows, q_heads, kv_heads, head_dim,
            position_offset, max_context, 0, 0);
        unsetenv("POCKETLLM_QWEN_GQA_FLASH_TILE");
        unsetenv("POCKETLLM_QWEN_GQA_QUERY_ROWS");
        // Tensor-core path. It reassociates both dot products and rounds the
        // softmax numerator to half for the MMA operand, so it is checked against
        // the CPU reference on its own tolerance, never for equality with hpg6.
        setenv("POCKETLLM_QWEN_GQA_MMA_TILE", "1", 1);
        mma_ok = pocket::qwen_gqa_prefill_attention_f16_tiled_cuda(
            d_q, d_k, d_v, d_mma, rows, q_heads, kv_heads, head_dim,
            position_offset, max_context, 0, 0);
        setenv("POCKETLLM_QWEN_GQA_MMA_TILE", "0", 1);
        setenv("POCKETLLM_QWEN_GQA_LONG_TILE", "0", 1);
    }
    // The position-batched kernel only amortises barriers; it must reproduce the
    // per-position kernel bit for bit, so this is an equality check rather than a
    // tolerance check.
    uint16_t* d_seq = nullptr;
    bool seq_ok = cudaMalloc(&d_seq, q.size() * sizeof(uint16_t)) == cudaSuccess;
    setenv("POCKETLLM_QWEN_GQA_POS_TILE", "0", 1);
    seq_ok = seq_ok && pocket::qwen_gqa_prefill_attention_f16_tiled_cuda(
        d_q, d_k, d_v, d_seq, rows, q_heads, kv_heads, head_dim,
        position_offset, max_context, 0, 0);
    // The warp-per-combo kernel reassociates the dot-product reduction, so it is
    // checked against the CPU softmax reference on tolerance rather than for bit
    // equality with the per-position kernel.
    uint16_t* d_warp = nullptr;
    bool warp_ok = cudaMalloc(&d_warp, q.size() * sizeof(uint16_t)) == cudaSuccess;
    setenv("POCKETLLM_QWEN_GQA_POS_TILE", "1", 1);
    setenv("POCKETLLM_QWEN_GQA_WARP_COMBO", "1", 1);
    warp_ok = warp_ok && pocket::qwen_gqa_prefill_attention_f16_tiled_cuda(
        d_q, d_k, d_v, d_warp, rows, q_heads, kv_heads, head_dim,
        position_offset, max_context, 0, 0);
    unsetenv("POCKETLLM_QWEN_GQA_WARP_COMBO");
    unsetenv("POCKETLLM_QWEN_GQA_POS_TILE");
    const bool plain_ok = pocket::qwen_gqa_prefill_attention_f16(
        d_q, d_k, d_v, d_plain, rows, q_heads, kv_heads, head_dim,
        position_offset, max_context);
    if (!tiled_ok || !plain_ok || !long_qr2_ok || !long_ok || !flash_ok ||
        !mma_ok || cudaDeviceSynchronize() != cudaSuccess) {
        fail("gqa prefill tiled launch failed");
        release();
        return true;
    }
    std::vector<uint16_t> got_warp(q_elements);
    if (!warp_ok || cudaDeviceSynchronize() != cudaSuccess) {
        fail("gqa prefill warp-combo launch failed");
        cudaFree(d_warp);
        cudaFree(d_seq);
        release();
        return true;
    }
    cudaMemcpy(got_warp.data(), d_warp, got_warp.size() * sizeof(uint16_t),
               cudaMemcpyDeviceToHost);
    cudaFree(d_warp);
    std::vector<uint16_t> got_tiled(q_elements);
    std::vector<uint16_t> got_plain(q_elements);
    std::vector<uint16_t> got_seq(q_elements);
    std::vector<uint16_t> got_long_qr2;
    std::vector<uint16_t> got_long;
    std::vector<uint16_t> got_flash;
    std::vector<uint16_t> got_mma;
    if (long_shape) {
        got_long_qr2.resize(q_elements);
        got_long.resize(q_elements);
        got_flash.resize(q_elements);
        got_mma.resize(q_elements);
    }
    if (!seq_ok || cudaDeviceSynchronize() != cudaSuccess) {
        fail("gqa prefill position-batched fallback launch failed");
        cudaFree(d_seq);
        release();
        return true;
    }
    cudaMemcpy(got_seq.data(), d_seq, got_seq.size() * sizeof(uint16_t),
               cudaMemcpyDeviceToHost);
    cudaFree(d_seq);
    cudaMemcpy(got_tiled.data(), d_tiled, got_tiled.size() * sizeof(uint16_t),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(got_plain.data(), d_plain, got_plain.size() * sizeof(uint16_t),
               cudaMemcpyDeviceToHost);
    if (long_shape) {
        cudaMemcpy(got_long_qr2.data(), d_long_qr2,
                   got_long_qr2.size() * sizeof(uint16_t),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(got_long.data(), d_long, got_long.size() * sizeof(uint16_t),
                   cudaMemcpyDeviceToHost);
        cudaMemcpy(got_flash.data(), d_flash,
                   got_flash.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(got_mma.data(), d_mma,
                   got_mma.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost);
    }
    release();

    size_t pos_tile_mismatches = 0;
    for (size_t i = 0; i < got_tiled.size(); ++i) {
        if (got_tiled[i] != got_seq[i]) ++pos_tile_mismatches;
    }
    if (pos_tile_mismatches != 0) {
        fail("gqa prefill position-batched kernel differs from per-position "
             "kernel in " + std::to_string(pos_tile_mismatches) + " elements");
    }
    double worst_ref = 0.0;
    double worst_cross = 0.0;
    double worst_warp = 0.0;
    double worst_long_qr2 = 0.0;
    double worst_long = 0.0;
    double worst_flash = 0.0;
    double worst_mma = 0.0;
    bool long_qr2_finite = true;
    bool long_finite = true;
    bool flash_finite = true;
    bool mma_finite = true;
    size_t long_qr2_mismatches = 0;
    size_t long_mismatches = 0;
    size_t flash_mismatches = 0;
    size_t mma_mismatches = 0;
    size_t long_qr2_vs_qr4_mismatches = 0;
    double worst_long_qr2_vs_qr4 = 0.0;
    for (int row = 0; row < rows; ++row) {
        std::vector<float> row_q(static_cast<size_t>(q_heads) * head_dim);
        for (size_t i = 0; i < row_q.size(); ++i) {
            row_q[i] = q_ref[static_cast<size_t>(row) * q_heads * head_dim + i];
        }
        std::vector<float> want(row_q.size());
        attention_reference(row_q, k_ref, v_ref, want, q_heads, kv_heads,
                            head_dim, position_offset + row + 1);
        for (size_t i = 0; i < want.size(); ++i) {
            const size_t flat = static_cast<size_t>(row) * q_heads * head_dim + i;
            const double tiled = from_half(got_tiled[flat]);
            const double plain = from_half(got_plain[flat]);
            worst_ref = std::max(worst_ref, std::fabs(tiled - want[i]));
            worst_cross = std::max(worst_cross, std::fabs(tiled - plain));
            worst_warp = std::max(worst_warp,
                std::fabs(static_cast<double>(from_half(got_warp[flat])) -
                          want[i]));
            if (long_shape) {
                const double long_qr2_value = from_half(got_long_qr2[flat]);
                long_qr2_finite = long_qr2_finite && std::isfinite(long_qr2_value);
                worst_long_qr2 = std::max(worst_long_qr2,
                    std::fabs(long_qr2_value - want[i]));
                if (got_long_qr2[flat] != got_plain[flat]) ++long_qr2_mismatches;
                const double long_value = from_half(got_long[flat]);
                long_finite = long_finite && std::isfinite(long_value);
                worst_long = std::max(worst_long,
                    std::fabs(long_value - want[i]));
                if (got_long[flat] != got_plain[flat]) ++long_mismatches;
                if (got_long_qr2[flat] != got_long[flat]) {
                    ++long_qr2_vs_qr4_mismatches;
                }
                worst_long_qr2_vs_qr4 = std::max(worst_long_qr2_vs_qr4,
                    std::fabs(long_qr2_value - long_value));
                const double flash_value = from_half(got_flash[flat]);
                flash_finite = flash_finite && std::isfinite(flash_value);
                worst_flash = std::max(worst_flash,
                    std::fabs(flash_value - want[i]));
                if (got_flash[flat] != got_plain[flat]) ++flash_mismatches;
                const double mma_value = from_half(got_mma[flat]);
                mma_finite = mma_finite && std::isfinite(mma_value);
                worst_mma = std::max(worst_mma, std::fabs(mma_value - want[i]));
                if (got_mma[flat] != got_plain[flat]) ++mma_mismatches;
            }
        }
    }
    // The long-tile and flash-tile candidates reassociate the dot product and the
    // softmax denominator differently from the per-position kernel, so they are
    // gated on distance from the CPU reference. The FP16 bit mismatch count
    // against the per-position kernel is reported, not asserted.
    // The tensor-core path rounds the softmax numerator to half for the MMA
    // operand, one extra rounding step the scalar paths do not have, so it gets a
    // slightly wider band. Anything past 8e-3 means a real indexing or online
    // rescale bug, not accumulated FP16 noise.
    if (long_shape && (!long_qr2_finite || !long_finite || !flash_finite ||
                       !mma_finite || worst_long_qr2 > 2.0e-3 ||
                       worst_long > 2.0e-3 || worst_flash > 2.0e-3 ||
                       worst_mma > 8.0e-3)) {
        fail("gqa prefill long-tile rows=" + std::to_string(rows) +
             " ctx=" + std::to_string(context_len) + " qr2_ref=" +
             std::to_string(worst_long_qr2) + " qr4_ref=" +
             std::to_string(worst_long) + " flash_ref=" +
             std::to_string(worst_flash) + " mma_ref=" +
             std::to_string(worst_mma));
    } else if (long_shape) {
        std::printf("  gqa prefill long-tile rows=%3d ctx=%5d qr2_ref=%.3e "
                    "qr2_mismatches=%zu qr4_ref=%.3e qr4_mismatches=%zu "
                    "qr2_vs_qr4=%.3e/%zu flash_ref=%.3e "
                    "flash_mismatches=%zu mma_ref=%.3e mma_mismatches=%zu\n",
                    rows, context_len,
                    worst_long_qr2, long_qr2_mismatches, worst_long,
                    long_mismatches, worst_long_qr2_vs_qr4,
                    long_qr2_vs_qr4_mismatches, worst_flash, flash_mismatches,
                    worst_mma, mma_mismatches);
    }
    unsetenv("POCKETLLM_QWEN_GQA_LONG_TILE");
    unsetenv("POCKETLLM_QWEN_GQA_QUERY_ROWS");
    unsetenv("POCKETLLM_QWEN_GQA_FLASH_TILE");
    unsetenv("POCKETLLM_QWEN_GQA_MMA_TILE");
    unsetenv("POCKETLLM_QWEN_GQA_POS_TILE");
    (void)long_qr2_mismatches;
    (void)long_mismatches;
    (void)flash_mismatches;
    (void)mma_mismatches;
    (void)long_qr2_vs_qr4_mismatches;
    (void)worst_long_qr2_vs_qr4;
    // FP16 accumulation over thousands of keys; 2e-3 matches the engine's other
    // FP16 attention gates.
    if (worst_ref > 2.0e-3 || worst_cross > 2.0e-3 || worst_warp > 2.0e-3) {
        fail("gqa prefill tiled rows=" + std::to_string(rows) + " ctx=" +
             std::to_string(context_len) + " ref=" + std::to_string(worst_ref) +
             " cross=" + std::to_string(worst_cross) +
             " warp=" + std::to_string(worst_warp));
    } else {
        std::printf("  gqa prefill tiled rows=%3d ctx=%5d ref=%.3e cross=%.3e "
                    "warp=%.3e\n",
                    rows, context_len, worst_ref, worst_cross, worst_warp);
    }
    return true;
}

// Decode: the tensor-core split kernel and the scalar fused path at the real TP4
// decode shape, both launched from this process against the same inputs and both
// compared to the CPU softmax. The candidate reassociates both dot products and
// rounds the softmax numerator to half for the MMA operand, so it gets its own
// tolerance rather than inheriting the scalar kernel's bit-identical guarantee;
// the two are compared to each other on tolerance, never for equality.
//
// The explicit variant entry point exists because the production selector caches
// its environment on first use, so a single process cannot otherwise reach both
// kernels. It refuses an unsupported shape rather than falling back, so a
// candidate result here cannot secretly be the scalar kernel.
//
// `target_splits` is the expected candidate geometry for this context on the
// 68-SM SM75 test host. Contexts that do not divide it leave the last KV tile
// partial, and `expect_empty_splits` covers the separate case where trailing
// splits scan nothing at all: an early return there left scratch uninitialized
// and produced -inf logits from the second decode step onward.
bool run_decode_mma(int context_len, int target_splits,
                    int split_override = 0, bool expect_empty_splits = false,
                    int kv_heads = 1, bool check_variant_geometry = true,
                    float minimum_value = -0.1f,
                    float maximum_value = 0.1f) {
    const int q_heads = kv_heads * 6;
    constexpr int head_dim = 256;
    const int max_context = context_len + 16;
    const size_t q_elements = static_cast<size_t>(q_heads) * head_dim;
    const size_t kv_elements =
        static_cast<size_t>(max_context) * kv_heads * head_dim;
    std::mt19937 rng(24681 + context_len * 31 + target_splits);
    std::uniform_real_distribution<float> dist(minimum_value, maximum_value);
    std::vector<uint16_t> q(q_elements), k(kv_elements, 0), v(kv_elements, 0);
    for (uint16_t& value : q) value = to_half(dist(rng));
    for (size_t i = 0;
         i < static_cast<size_t>(context_len) * kv_heads * head_dim; ++i) {
        k[i] = to_half(dist(rng));
        v[i] = to_half(dist(rng));
    }

    const int mma_splits = split_override > 0
        ? pocket::qwen_gqa_decode_split_count_variant(
              context_len, pocket::QwenGqaDecodeVariant::TensorCore,
              split_override)
        : pocket::qwen_gqa_decode_split_count_variant(
              context_len, pocket::QwenGqaDecodeVariant::TensorCore);
    const int scalar_splits = pocket::qwen_gqa_decode_split_count_variant(
        context_len, pocket::QwenGqaDecodeVariant::Scalar, split_override);
    if (check_variant_geometry && split_override == 0 &&
        std::getenv("POCKETLLM_QWEN_DECODE_MMA_TARGET_SPLITS") == nullptr &&
        mma_splits != target_splits) {
        fail("gqa decode candidate split count ctx=" +
             std::to_string(context_len) + " got=" +
             std::to_string(mma_splits) + " want=" +
             std::to_string(target_splits));
        return true;
    }
    // Trailing splits scan nothing exactly when the count exceeds what the slice
    // length can fill. Production geometry re-derives the count to drop those, so
    // only the preserved override can reach the case.
    const int positions_per_split =
        (context_len + mma_splits - 1) / mma_splits;
    const int filled_splits =
        (context_len + positions_per_split - 1) / positions_per_split;
    const bool has_empty_splits = filled_splits < mma_splits;
    if (expect_empty_splits && !has_empty_splits) {
        fail("gqa decode empty-split case ctx=" + std::to_string(context_len) +
             " splits=" + std::to_string(mma_splits) + " filled=" +
             std::to_string(filled_splits) + " has no dry split");
        return true;
    }
    if (expect_empty_splits && scalar_splits != mma_splits) {
        fail("gqa decode empty-split case ctx=" + std::to_string(context_len) +
             " needs both variants on the same geometry, got scalar=" +
             std::to_string(scalar_splits) + " mma=" +
             std::to_string(mma_splits));
        return true;
    }

    uint16_t* d_q = nullptr;
    uint16_t* d_k = nullptr;
    uint16_t* d_v = nullptr;
    uint16_t* d_scalar = nullptr;
    uint16_t* d_mma = nullptr;
    float* d_partials = nullptr;
    auto release = [&]() {
        cudaFree(d_q); cudaFree(d_k); cudaFree(d_v);
        cudaFree(d_scalar); cudaFree(d_mma); cudaFree(d_partials);
    };
    // One scratch buffer serves both launches, so it is sized for whichever
    // variant asks for more splits. Reusing it also means a kernel that fails to
    // publish a partial reads the other kernel's leftovers, which is exactly the
    // failure the empty-split case is looking for.
    const size_t partial_elements = static_cast<size_t>(q_heads) *
        std::max(mma_splits, scalar_splits) *
        static_cast<size_t>(head_dim + 2);
    if (cudaMalloc(&d_q, q.size() * sizeof(uint16_t)) != cudaSuccess ||
        cudaMalloc(&d_k, k.size() * sizeof(uint16_t)) != cudaSuccess ||
        cudaMalloc(&d_v, v.size() * sizeof(uint16_t)) != cudaSuccess ||
        cudaMalloc(&d_scalar, q.size() * sizeof(uint16_t)) != cudaSuccess ||
        cudaMalloc(&d_mma, q.size() * sizeof(uint16_t)) != cudaSuccess ||
        cudaMalloc(&d_partials, partial_elements * sizeof(float)) != cudaSuccess) {
        release();
        return false;
    }
    if (cudaMemcpy(d_q, q.data(), q.size() * sizeof(uint16_t),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_k, k.data(), k.size() * sizeof(uint16_t),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_v, v.data(), v.size() * sizeof(uint16_t),
                   cudaMemcpyHostToDevice) != cudaSuccess) {
        release();
        return false;
    }
    if (!pocket::qwen_gqa_decode_attention_f16_fused_variant_cuda(
            d_q, d_k, d_v, d_scalar, d_partials, q_heads, kv_heads, head_dim,
            context_len, max_context, 0, 0,
            pocket::QwenGqaDecodeVariant::Scalar, split_override) ||
        cudaDeviceSynchronize() != cudaSuccess) {
        fail("gqa decode scalar launch ctx=" + std::to_string(context_len));
        release();
        return true;
    }
    if (!pocket::qwen_gqa_decode_attention_f16_fused_variant_cuda(
            d_q, d_k, d_v, d_mma, d_partials, q_heads, kv_heads, head_dim,
            context_len, max_context, 0, 0,
            pocket::QwenGqaDecodeVariant::TensorCore, split_override) ||
        cudaDeviceSynchronize() != cudaSuccess) {
        fail("gqa decode candidate launch ctx=" + std::to_string(context_len));
        release();
        return true;
    }
    std::vector<uint16_t> scalar(q_elements), candidate(q_elements);
    if (cudaMemcpy(scalar.data(), d_scalar, scalar.size() * sizeof(uint16_t),
                   cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(candidate.data(), d_mma, candidate.size() * sizeof(uint16_t),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        release();
        return false;
    }
    // Repeat the candidate on the same inputs. Its accumulator lives in shared
    // memory with one owning lane per element, so a repeat mismatch would mean a
    // race rather than a reassociation difference.
    std::vector<uint16_t> repeat(q_elements);
    if (!pocket::qwen_gqa_decode_attention_f16_fused_variant_cuda(
            d_q, d_k, d_v, d_mma, d_partials, q_heads, kv_heads, head_dim,
            context_len, max_context, 0, 0,
            pocket::QwenGqaDecodeVariant::TensorCore, split_override) ||
        cudaDeviceSynchronize() != cudaSuccess ||
        cudaMemcpy(repeat.data(), d_mma, repeat.size() * sizeof(uint16_t),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        fail("gqa decode candidate repeat ctx=" + std::to_string(context_len));
        release();
        return true;
    }
    release();
    size_t repeat_mismatches = 0;
    for (size_t i = 0; i < repeat.size(); ++i) {
        if (repeat[i] != candidate[i]) ++repeat_mismatches;
    }
    if (repeat_mismatches != 0) {
        fail("gqa decode candidate is not repeatable ctx=" +
             std::to_string(context_len) + " mismatches=" +
             std::to_string(repeat_mismatches));
    }

    // CPU reference over the exact FP16 values the kernels consumed.
    std::vector<float> q_ref(q_elements), k_ref(kv_elements), v_ref(kv_elements);
    for (size_t i = 0; i < q_elements; ++i) q_ref[i] = from_half(q[i]);
    for (size_t i = 0; i < kv_elements; ++i) {
        k_ref[i] = from_half(k[i]);
        v_ref[i] = from_half(v[i]);
    }
    std::vector<float> want(q_elements);
    attention_reference(q_ref, k_ref, v_ref, want, q_heads, kv_heads, head_dim,
                        context_len);
    double worst_scalar = 0.0;
    double worst_candidate = 0.0;
    double worst_cross = 0.0;
    bool finite = true;
    size_t cross_mismatches = 0;
    for (size_t i = 0; i < candidate.size(); ++i) {
        const double scalar_value = from_half(scalar[i]);
        const double candidate_value = from_half(candidate[i]);
        if (!std::isfinite(scalar_value) || !std::isfinite(candidate_value)) {
            finite = false;
        }
        worst_scalar = std::max(worst_scalar, std::fabs(scalar_value - want[i]));
        worst_candidate =
            std::max(worst_candidate, std::fabs(candidate_value - want[i]));
        worst_cross =
            std::max(worst_cross, std::fabs(candidate_value - scalar_value));
        if (scalar[i] != candidate[i]) ++cross_mismatches;
    }
    // The scalar path is the exact reference implementation, so it keeps the
    // tighter band; the candidate's extra half rounding of the softmax numerator
    // gets 4e-3, matching the prefill MMA gate.
    if (!finite || worst_scalar > 2.0e-3 || worst_candidate > 4.0e-3 ||
        worst_cross > 4.0e-3) {
        fail("gqa decode ctx=" + std::to_string(context_len) +
             " splits=" + std::to_string(mma_splits) +
             " finite=" + std::to_string(static_cast<int>(finite)) +
             " scalar=" + std::to_string(worst_scalar) +
             " mma=" + std::to_string(worst_candidate) +
             " cross=" + std::to_string(worst_cross));
    } else {
        std::printf("  gqa decode ctx=%6d mma_splits=%4d scalar_splits=%4d "
                    "empty=%d scalar_ref=%.3e mma_ref=%.3e cross=%.3e "
                    "bitdiff=%zu\n",
                    context_len, mma_splits, scalar_splits,
                    static_cast<int>(has_empty_splits), worst_scalar,
                    worst_candidate, worst_cross, cross_mismatches);
    }
    return true;
}

bool run_verify_crossover(int context_len) {
    constexpr int rows = 8;
    constexpr int q_heads = 6;
    constexpr int kv_heads = 1;
    constexpr int head_dim = 256;
    constexpr int splits = 64;
    const int position_offset = context_len - rows;
    const size_t q_elements = static_cast<size_t>(rows) * q_heads * head_dim;
    const size_t kv_elements = static_cast<size_t>(context_len) * kv_heads * head_dim;
    const size_t score_elements = static_cast<size_t>(rows) * q_heads * context_len;
    const size_t partial_elements = static_cast<size_t>(rows) * q_heads * splits *
        static_cast<size_t>(head_dim + 2);
    std::mt19937 rng(97531 + context_len);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    std::vector<uint16_t> q(q_elements), k(kv_elements), v(kv_elements);
    for (uint16_t& value : q) value = to_half(dist(rng));
    for (uint16_t& value : k) value = to_half(dist(rng));
    for (uint16_t& value : v) value = to_half(dist(rng));

    uint16_t* d_q = nullptr;
    uint16_t* d_k = nullptr;
    uint16_t* d_v = nullptr;
    uint16_t* d_exact = nullptr;
    uint16_t* d_split = nullptr;
    uint16_t* d_cublas = nullptr;
    float* d_scores = nullptr;
    float* d_partials = nullptr;
    auto release = [&]() {
        cudaFree(d_q); cudaFree(d_k); cudaFree(d_v);
        cudaFree(d_exact); cudaFree(d_split); cudaFree(d_cublas);
        cudaFree(d_scores); cudaFree(d_partials);
    };
    if (cudaMalloc(&d_q, q.size() * sizeof(uint16_t)) != cudaSuccess ||
        cudaMalloc(&d_k, k.size() * sizeof(uint16_t)) != cudaSuccess ||
        cudaMalloc(&d_v, v.size() * sizeof(uint16_t)) != cudaSuccess ||
        cudaMalloc(&d_exact, q.size() * sizeof(uint16_t)) != cudaSuccess ||
        cudaMalloc(&d_split, q.size() * sizeof(uint16_t)) != cudaSuccess ||
        cudaMalloc(&d_cublas, q.size() * sizeof(uint16_t)) != cudaSuccess ||
        cudaMalloc(&d_scores, score_elements * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&d_partials, partial_elements * sizeof(float)) != cudaSuccess) {
        release();
        return false;
    }
    if (cudaMemcpy(d_q, q.data(), q.size() * sizeof(uint16_t),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_k, k.data(), k.size() * sizeof(uint16_t),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(d_v, v.data(), v.size() * sizeof(uint16_t),
                   cudaMemcpyHostToDevice) != cudaSuccess ||
        !pocket::qwen_gqa_verify_attention_f16_exact_cuda(
            d_q, d_k, d_v, d_exact, d_scores, rows, q_heads, kv_heads,
            head_dim, position_offset, context_len) ||
        !pocket::qwen_gqa_verify_attention_f16(
            d_q, d_k, d_v, d_split, d_partials, rows, q_heads, kv_heads,
            head_dim, position_offset, context_len, splits) ||
        !pocket::qwen_gqa_verify_attention_f16_cublas_qk_cuda(
            d_q, d_k, d_v, d_cublas, d_scores, rows, q_heads, kv_heads,
            head_dim, position_offset, context_len) ||
        cudaDeviceSynchronize() != cudaSuccess) {
        fail("gqa verify crossover launch ctx=" + std::to_string(context_len));
        release();
        return true;
    }
    std::vector<uint16_t> exact(q_elements), split(q_elements), cublas(q_elements);
    if (cudaMemcpy(exact.data(), d_exact, exact.size() * sizeof(uint16_t),
                   cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(split.data(), d_split, split.size() * sizeof(uint16_t),
                   cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(cublas.data(), d_cublas, cublas.size() * sizeof(uint16_t),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
        fail("gqa verify crossover copy ctx=" + std::to_string(context_len));
        release();
        return true;
    }
    release();
    double split_worst = 0.0;
    double cublas_worst = 0.0;
    for (size_t index = 0; index < exact.size(); ++index) {
        const double reference = from_half(exact[index]);
        split_worst = std::max(split_worst, std::fabs(
            reference - from_half(split[index])));
        cublas_worst = std::max(cublas_worst, std::fabs(
            reference - from_half(cublas[index])));
    }
    if (split_worst > 2.0e-3 || cublas_worst > 2.0e-3) {
        fail("gqa verify crossover ctx=" + std::to_string(context_len) +
             " split=" + std::to_string(split_worst) +
             " cublas=" + std::to_string(cublas_worst));
    } else {
        const char* production_default = context_len <= 1024 ? "exact" : "split";
        std::printf("  gqa verify crossover ctx=%5d default=%s "
                    "exact_vs_split=%.3e exact_vs_cublas=%.3e\n",
                    context_len, production_default, split_worst, cublas_worst);
    }
    return true;
}

}  // namespace

int main() {
    if (!pocket::cuda_runtime_available()) {
        std::printf("[SKIP] test_qwen_gqa_attention requires a CUDA device\n");
        return 0;
    }
    // Qwen3.8 TP1 full attention: 24 q heads over 4 kv heads, head_dim 256.
    const int contexts[] = {1, 7, 128, 333};
    for (int ctx : contexts) {
        if (!run_decode(24, 4, 256, ctx, 512)) {
            std::printf("[SKIP] device allocation failed\n");
            return 0;
        }
    }
    // Long-context prefill shape: 16 q over 4 kv heads at head_dim 128 is the
    // TP4 shard of Qwen3.8 full attention. Includes a chunk landing deep into a
    // populated cache, which is where the tiled kernel is actually dispatched.
    const int prefill_offsets[] = {0, 512, 4096};
    for (int offset : prefill_offsets) {
        if (!run_prefill_tiled(64, 16, 4, 128, offset, 8192)) {
            std::printf("[SKIP] device allocation failed\n");
            return 0;
        }
    }
    // The position-batched kernel picks its head grouping from q_heads/kv_heads,
    // so exercise each branch: 4 (the TP4 shard above), 2, 1, and an odd ratio
    // that falls back to the original group of three.
    const int grouping_shapes[][2] = {{12, 6}, {8, 8}, {24, 8}};
    for (const auto& shape : grouping_shapes) {
        if (!run_prefill_tiled(64, shape[0], shape[1], 128, 512, 8192)) {
            std::printf("[SKIP] device allocation failed\n");
            return 0;
        }
    }
    // The production TP4 Qwen shard has six Q heads over one KV head. Use a
    // nonzero offset that makes total_context >= 2048 so this actually crosses
    // the production hpg6/flash dispatch gate, then compare every output element
    // against the exact generic and CPU reference paths.
    if (!run_prefill_tiled(128, 6, 1, 256, 1920, 2048, true)) {
        std::printf("[SKIP] device allocation failed\n");
        return 0;
    }
    // The tensor-core path owns 32 query rows and a 32-position KV tile per CTA.
    // Row 150 leaves a 22-row final block and context 2198 leaves a 22-position
    // final KV tile, so this covers both partial-tile masks at once. A kernel that
    // mishandles either would still pass the aligned case above.
    if (!run_prefill_tiled(150, 6, 1, 256, 2048, 2198, true)) {
        std::printf("[SKIP] device allocation failed\n");
        return 0;
    }
    // Decode at the real TP4 shape. 4096 is the minimum context the fused path
    // accepts; 8199 and 65537 leave the last KV tile partial. Both kernels run in
    // this process against the same inputs, so every case checks scalar against
    // the CPU reference, the candidate against the CPU reference, and the two
    // against each other, regardless of what POCKETLLM_QWEN_DECODE_MMA selects.
    // The candidate targets two CTA waves (136 on this host) at every context.
    // 4096 and 8199 land below that because the count is re-derived from the
    // rounded slice length: 31 positions cover 4096 in 133 splits and 61 cover
    // 8199 in 135, so the dead trailing CTAs are dropped.
    const int decode_cases[][2] = {
        {4096, 133}, {8199, 135}, {16384, 136}, {65537, 136},
    };
    for (const auto& decode_case : decode_cases) {
        if (!run_decode_mma(decode_case[0], decode_case[1])) {
            std::printf("[SKIP] device allocation failed\n");
            return 0;
        }
    }
    // TP2 has two independent 6Q:1KV groups in each rank. The same kernel grid
    // must address the interleaved [position, kv_head, channel] cache correctly
    // and publish both groups into the q-head-major partial layout.
    if (!run_decode_mma(8199, 135, 0, false, 2)) {
        std::printf("[SKIP] device allocation failed\n");
        return 0;
    }
    // Positive-only inputs expose stale scores from the ten padded MMA rows. Their
    // shared score slots are never written, so they must be masked before reduction;
    // otherwise a previous valid row can survive and make a full-mask shuffle group
    // use an arbitrary maximum. This was invisible to symmetric random fixtures.
    if (!run_decode_mma(4096, 133, 0, false, 2, true, 0.01f, 0.1f)) {
        std::printf("[SKIP] device allocation failed\n");
        return 0;
    }
    const int tp2_long_splits = pocket::qwen_gqa_decode_split_count(65537, 2, true);
    const char* mma_split_env =
        std::getenv("POCKETLLM_QWEN_DECODE_MMA_TARGET_SPLITS");
    const int mma_split_override =
        mma_split_env != nullptr ? std::atoi(mma_split_env) : 0;
    const int expected_tp2_long_splits =
        mma_split_override > 0 ? mma_split_override : 68;
    if (tp2_long_splits != expected_tp2_long_splits) {
        fail("gqa decode TP2 shape-aware split count got=" +
             std::to_string(tp2_long_splits) + " want=" +
             std::to_string(expected_tp2_long_splits));
    }
    // Empty trailing splits. 4100 positions over 200 splits is 21 each, which 196
    // splits already cover, so the last four CTAs scan nothing and must still
    // publish the neutral partial the merge expects. Production re-derives the
    // count to 196, so this needs the preserved-count test override. Silent
    // -inf logits from step two onward were the original failure here.
    if (!run_decode_mma(4100, 0, 200, true)) {
        std::printf("[SKIP] device allocation failed\n");
        return 0;
    }
    // Production currently uses exact through 1K context and split-K above it;
    // cuBLAS-QK is opt-in while its real-model crossover is validated. Cover
    // both sides of the current boundary and representative mid/long contexts.
    const int verify_contexts[] = {1024, 1025, 4096, 8192};
    for (int context : verify_contexts) {
        if (!run_verify_crossover(context)) {
            std::printf("[SKIP] device allocation failed\n");
            return 0;
        }
    }
    if (failures != 0) {
        std::printf("test_qwen_gqa_attention failures=%d\n", failures);
        return 1;
    }
    std::printf("[PASS] test_qwen_gqa_attention\n");
    return 0;
}
