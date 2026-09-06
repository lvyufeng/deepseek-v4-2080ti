// Numerical validation for the batched decode kernels: partial RoPE, KV append,
// GQA decode attention, the gated-delta step, and the causal conv.
//
// Each batched kernel is checked against N runs of the single-row kernel it
// replaces, driving every row with a distinct position/context length and its
// own state slot. A batching bug -- a missing slot stride, a shared tail, a
// position read from the wrong row -- would leave throughput untouched and
// silently corrupt one sequence with another's state, so the batched path needs
// its own numerical gate rather than inheriting the single-row one.

#include "qwen_cuda_ops.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

namespace {

using namespace pocket;

int failures = 0;
bool device_ready = true;

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

// Owns one device allocation so a failed test cannot leak into the next one.
struct Buffer {
    void* data = nullptr;
    size_t bytes = 0;

    ~Buffer() {
        if (data != nullptr) cudaFree(data);
    }

    bool allocate(size_t count) {
        bytes = count;
        return cudaMalloc(&data, count) == cudaSuccess;
    }

    template <typename T>
    T* as() {
        return static_cast<T*>(data);
    }

    bool upload(const void* host) {
        return cudaMemcpy(data, host, bytes, cudaMemcpyHostToDevice) ==
            cudaSuccess;
    }

    bool download(void* host) const {
        return cudaMemcpy(host, data, bytes, cudaMemcpyDeviceToHost) ==
            cudaSuccess;
    }

    bool clear() { return cudaMemset(data, 0, bytes) == cudaSuccess; }
};

double worst_difference(const std::vector<uint16_t>& left,
                        const std::vector<uint16_t>& right) {
    double worst = 0.0;
    for (size_t i = 0; i < left.size(); ++i) {
        worst = std::max(worst, std::fabs(
            static_cast<double>(from_half(left[i])) -
            static_cast<double>(from_half(right[i]))));
    }
    return worst;
}

// Bit-identical is the bar wherever the batched kernel only re-bases pointers.
size_t bit_mismatches(const std::vector<uint16_t>& left,
                      const std::vector<uint16_t>& right) {
    size_t count = 0;
    for (size_t i = 0; i < left.size(); ++i) {
        if (left[i] != right[i]) ++count;
    }
    return count;
}

// Partial RoPE. Each row rotates by its own absolute position, which is what
// distinguishes a decode batch from a prefill chunk: the rows are not
// consecutive, so `start_position + row` is wrong for every row but the first.
bool test_rope_batched() {
    constexpr int kRows = 5;
    constexpr int kQHeads = 6;
    constexpr int kKvHeads = 1;
    constexpr int kHeadDim = 256;
    constexpr int kRotaryDim = 64;
    const int positions[kRows] = {0, 1, 17, 1024, 4095};

    const size_t q_elements =
        static_cast<size_t>(kRows) * kQHeads * kHeadDim;
    const size_t k_elements =
        static_cast<size_t>(kRows) * kKvHeads * kHeadDim;
    std::mt19937 rng(24680);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<uint16_t> q_host(q_elements);
    std::vector<uint16_t> k_host(k_elements);
    for (uint16_t& value : q_host) value = to_half(dist(rng));
    for (uint16_t& value : k_host) value = to_half(dist(rng));

    Buffer q_sequential, k_sequential, q_batched, k_batched, device_positions;
    if (!q_sequential.allocate(q_elements * sizeof(uint16_t)) ||
        !k_sequential.allocate(k_elements * sizeof(uint16_t)) ||
        !q_batched.allocate(q_elements * sizeof(uint16_t)) ||
        !k_batched.allocate(k_elements * sizeof(uint16_t)) ||
        !device_positions.allocate(kRows * sizeof(int))) {
        device_ready = false;
        return false;
    }
    if (!q_sequential.upload(q_host.data()) ||
        !k_sequential.upload(k_host.data()) ||
        !q_batched.upload(q_host.data()) ||
        !k_batched.upload(k_host.data()) ||
        !device_positions.upload(positions)) {
        device_ready = false;
        return false;
    }

    // RoPE rotates in place, so each reference call takes the row's own slice.
    for (int row = 0; row < kRows; ++row) {
        const bool launched = qwen_partial_rope_rows_f16_cuda(
            q_sequential.as<uint16_t>() +
                static_cast<size_t>(row) * kQHeads * kHeadDim,
            k_sequential.as<uint16_t>() +
                static_cast<size_t>(row) * kKvHeads * kHeadDim,
            positions[row], 1, kRotaryDim, 10000.0f, kQHeads, kKvHeads,
            kHeadDim);
        if (!launched) {
            fail("batched rope reference launch failed");
            return true;
        }
    }
    if (!qwen_partial_rope_rows_f16_batched_cuda(
            q_batched.as<uint16_t>(), k_batched.as<uint16_t>(),
            device_positions.as<int>(), kRows, kRotaryDim, 10000.0f, kQHeads,
            kKvHeads, kHeadDim)) {
        fail("batched rope launch failed");
        return true;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        fail("batched rope sync failed");
        return true;
    }

    std::vector<uint16_t> q_want(q_elements), q_got(q_elements);
    std::vector<uint16_t> k_want(k_elements), k_got(k_elements);
    if (!q_sequential.download(q_want.data()) ||
        !q_batched.download(q_got.data()) ||
        !k_sequential.download(k_want.data()) ||
        !k_batched.download(k_got.data())) {
        device_ready = false;
        return false;
    }
    const size_t q_bad = bit_mismatches(q_want, q_got);
    const size_t k_bad = bit_mismatches(k_want, k_got);
    if (q_bad != 0 || k_bad != 0) {
        fail("batched rope q_mismatch=" + std::to_string(q_bad) +
             " k_mismatch=" + std::to_string(k_bad));
    } else {
        std::printf("  rope rows=%d positions=%d..%d bit-identical\n", kRows,
                    positions[0], positions[kRows - 1]);
    }
    return true;
}

// KV append. Every row writes one token into its own slot at its own position,
// so a missing slot stride would have the rows overwrite each other.
bool test_append_kv_batched() {
    constexpr int kRows = 4;
    constexpr int kKvHeads = 1;
    constexpr int kHeadDim = 256;
    constexpr int kMaxContext = 512;
    constexpr int kSlots = 6;
    const int positions[kRows] = {0, 7, 128, 511};
    const int slots[kRows] = {2, 5, 0, 3};

    const size_t row_elements = static_cast<size_t>(kKvHeads) * kHeadDim;
    const size_t slot_stride =
        static_cast<size_t>(kMaxContext) * kKvHeads * kHeadDim;
    const size_t cache_elements = static_cast<size_t>(kSlots) * slot_stride;
    std::mt19937 rng(1357);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<uint16_t> k_rows(static_cast<size_t>(kRows) * row_elements);
    std::vector<uint16_t> v_rows(k_rows.size());
    for (uint16_t& value : k_rows) value = to_half(dist(rng));
    for (uint16_t& value : v_rows) value = to_half(dist(rng));

    Buffer device_k_rows, device_v_rows, device_positions, device_slots;
    Buffer k_sequential, v_sequential, k_batched, v_batched;
    if (!device_k_rows.allocate(k_rows.size() * sizeof(uint16_t)) ||
        !device_v_rows.allocate(v_rows.size() * sizeof(uint16_t)) ||
        !device_positions.allocate(kRows * sizeof(int)) ||
        !device_slots.allocate(kRows * sizeof(int)) ||
        !k_sequential.allocate(cache_elements * sizeof(uint16_t)) ||
        !v_sequential.allocate(cache_elements * sizeof(uint16_t)) ||
        !k_batched.allocate(cache_elements * sizeof(uint16_t)) ||
        !v_batched.allocate(cache_elements * sizeof(uint16_t))) {
        device_ready = false;
        return false;
    }
    if (!device_k_rows.upload(k_rows.data()) ||
        !device_v_rows.upload(v_rows.data()) ||
        !device_positions.upload(positions) || !device_slots.upload(slots) ||
        !k_sequential.clear() || !v_sequential.clear() ||
        !k_batched.clear() || !v_batched.clear()) {
        device_ready = false;
        return false;
    }

    for (int row = 0; row < kRows; ++row) {
        const bool launched = qwen_append_kv_cache_f16_cuda(
            device_k_rows.as<uint16_t>() + static_cast<size_t>(row) * row_elements,
            device_v_rows.as<uint16_t>() + static_cast<size_t>(row) * row_elements,
            k_sequential.as<uint16_t>() +
                static_cast<size_t>(slots[row]) * slot_stride,
            v_sequential.as<uint16_t>() +
                static_cast<size_t>(slots[row]) * slot_stride,
            1, kKvHeads, kHeadDim, positions[row], kMaxContext);
        if (!launched) {
            fail("batched kv append reference launch failed");
            return true;
        }
    }
    if (!qwen_append_kv_cache_f16_batched_cuda(
            device_k_rows.as<uint16_t>(), device_v_rows.as<uint16_t>(),
            k_batched.as<uint16_t>(), v_batched.as<uint16_t>(), kRows,
            kKvHeads, kHeadDim, device_positions.as<int>(),
            device_slots.as<int>(), kMaxContext, slot_stride)) {
        fail("batched kv append launch failed");
        return true;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        fail("batched kv append sync failed");
        return true;
    }

    std::vector<uint16_t> k_want(cache_elements), k_got(cache_elements);
    std::vector<uint16_t> v_want(cache_elements), v_got(cache_elements);
    if (!k_sequential.download(k_want.data()) ||
        !k_batched.download(k_got.data()) ||
        !v_sequential.download(v_want.data()) ||
        !v_batched.download(v_got.data())) {
        device_ready = false;
        return false;
    }
    const size_t k_bad = bit_mismatches(k_want, k_got);
    const size_t v_bad = bit_mismatches(v_want, v_got);
    if (k_bad != 0 || v_bad != 0) {
        fail("batched kv append k_mismatch=" + std::to_string(k_bad) +
             " v_mismatch=" + std::to_string(v_bad));
    } else {
        std::printf("  kv append rows=%d slots=%d bit-identical\n", kRows,
                    kSlots);
    }
    return true;
}

// GQA decode attention. The split geometry comes from the longest row, so the
// shorter rows run some of their splits dry and must still publish the neutral
// partial the merge expects -- the same failure mode that produced silent -inf
// logits in the single-row path. Contexts are chosen so the batch spans both
// sides of that boundary.
bool test_gqa_decode_batched() {
    constexpr int kRows = 4;
    constexpr int kQHeads = 6;
    constexpr int kKvHeads = 1;
    constexpr int kHeadDim = 256;
    constexpr int kMaxContext = 8192;
    // The fused decode kernel declines contexts under 4096, so every row has to
    // clear that floor for the reference launches to be the same kernel.
    const int contexts[kRows] = {4096, 5000, 6144, 8192};
    const int longest = contexts[kRows - 1];
    // A non-identity mapping over more slots than rows, which is what the
    // scheduler actually produces once requests come and go.
    constexpr int kSlots = 7;
    const int slots[kRows] = {1, 3, 5, 6};

    const size_t slot_stride =
        static_cast<size_t>(kMaxContext) * kKvHeads * kHeadDim;
    const size_t cache_elements = static_cast<size_t>(kSlots) * slot_stride;
    const size_t q_row_elements = static_cast<size_t>(kQHeads) * kHeadDim;
    const size_t q_elements = static_cast<size_t>(kRows) * q_row_elements;

    std::mt19937 rng(97531);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<uint16_t> q_host(q_elements);
    std::vector<uint16_t> k_host(cache_elements, to_half(0.0f));
    std::vector<uint16_t> v_host(cache_elements, to_half(0.0f));
    for (uint16_t& value : q_host) value = to_half(dist(rng));
    for (int row = 0; row < kRows; ++row) {
        const size_t base = static_cast<size_t>(slots[row]) * slot_stride;
        const size_t live =
            static_cast<size_t>(contexts[row]) * kKvHeads * kHeadDim;
        for (size_t i = 0; i < live; ++i) {
            k_host[base + i] = to_half(dist(rng));
            v_host[base + i] = to_half(dist(rng));
        }
    }

    // Both paths need scratch for the widest geometry they will launch. The
    // batched split count is a public query for exactly this reason.
    const int batched_splits =
        qwen_gqa_decode_batched_split_count(longest, kKvHeads);
    if (batched_splits <= 0) {
        fail("batched decode split count is not positive");
        return true;
    }
    const size_t partial_row = static_cast<size_t>(kQHeads) * batched_splits *
        static_cast<size_t>(kHeadDim + 2);

    Buffer device_q, device_k, device_v, device_contexts, device_slots;
    Buffer out_sequential, out_batched, scratch_sequential, scratch_batched;
    if (!device_q.allocate(q_elements * sizeof(uint16_t)) ||
        !device_k.allocate(cache_elements * sizeof(uint16_t)) ||
        !device_v.allocate(cache_elements * sizeof(uint16_t)) ||
        !device_contexts.allocate(kRows * sizeof(int)) ||
        !device_slots.allocate(kRows * sizeof(int)) ||
        !out_sequential.allocate(q_elements * sizeof(uint16_t)) ||
        !out_batched.allocate(q_elements * sizeof(uint16_t)) ||
        !scratch_sequential.allocate(partial_row * sizeof(float)) ||
        !scratch_batched.allocate(
            static_cast<size_t>(kRows) * partial_row * sizeof(float))) {
        device_ready = false;
        return false;
    }
    if (!device_q.upload(q_host.data()) || !device_k.upload(k_host.data()) ||
        !device_v.upload(v_host.data()) ||
        !device_contexts.upload(contexts) || !device_slots.upload(slots) ||
        !out_sequential.clear() || !out_batched.clear()) {
        device_ready = false;
        return false;
    }

    for (int row = 0; row < kRows; ++row) {
        const size_t slot = static_cast<size_t>(slots[row]) * slot_stride;
        const bool launched = qwen_gqa_decode_attention_f16_fused_cuda(
            device_q.as<uint16_t>() + static_cast<size_t>(row) * q_row_elements,
            device_k.as<uint16_t>() + slot, device_v.as<uint16_t>() + slot,
            out_sequential.as<uint16_t>() +
                static_cast<size_t>(row) * q_row_elements,
            scratch_sequential.as<float>(), kQHeads, kKvHeads, kHeadDim,
            contexts[row], kMaxContext, 0, 0);
        if (!launched) {
            fail("batched decode reference launch failed at row " +
                 std::to_string(row));
            return true;
        }
        if (cudaDeviceSynchronize() != cudaSuccess) {
            fail("batched decode reference sync failed");
            return true;
        }
    }
    if (!qwen_gqa_decode_attention_f16_batched_cuda(
            device_q.as<uint16_t>(), device_k.as<uint16_t>(),
            device_v.as<uint16_t>(), out_batched.as<uint16_t>(),
            scratch_batched.as<float>(), device_contexts.as<int>(),
            device_slots.as<int>(), kRows, longest, slot_stride, kQHeads,
            kKvHeads, kHeadDim, kMaxContext, 0, 0)) {
        fail("batched decode launch failed");
        return true;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        fail("batched decode sync failed");
        return true;
    }

    std::vector<uint16_t> want(q_elements), got(q_elements);
    if (!out_sequential.download(want.data()) ||
        !out_batched.download(got.data())) {
        device_ready = false;
        return false;
    }
    // The reference launches pick their own split count per row, so a row whose
    // count differs from the batch's reduces its partials in a different order.
    // That is a float-addition reassociation, not a different computation, so
    // the bar is a tolerance rather than bit equality.
    const double worst = worst_difference(want, got);
    if (worst > 1.0e-3) {
        fail("batched decode worst=" + std::to_string(worst));
    } else {
        std::printf("  gqa decode rows=%d contexts=%d..%d splits=%d worst=%.3e\n",
                    kRows, contexts[0], longest, batched_splits, worst);
    }
    return true;
}

// Gated-delta step. Every row advances its own [key_dim, value_dim] state, so a
// shared state pointer would have the rows accumulate into each other. The
// states are seeded differently per row to make that visible.
bool test_gated_delta_step_batched() {
    constexpr int kRows = 3;
    constexpr int kHeads = 4;
    constexpr int kKeyHeads = 2;
    constexpr int kKeyDim = 128;
    constexpr int kValueDim = 128;

    constexpr int kSlots = 5;
    const int slots[kRows] = {4, 0, 2};

    const size_t state_stride =
        static_cast<size_t>(kHeads) * kKeyDim * kValueDim;
    const size_t state_elements = static_cast<size_t>(kSlots) * state_stride;
    const size_t key_row = static_cast<size_t>(kKeyHeads) * kKeyDim;
    const size_t value_row = static_cast<size_t>(kHeads) * kValueDim;

    std::mt19937 rng(86420);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> state_host(state_elements);
    for (float& value : state_host) value = dist(rng);
    std::vector<uint16_t> q_host(static_cast<size_t>(kRows) * key_row);
    std::vector<uint16_t> k_host(q_host.size());
    std::vector<uint16_t> v_host(static_cast<size_t>(kRows) * value_row);
    std::vector<uint16_t> g_host(static_cast<size_t>(kRows) * kHeads);
    std::vector<uint16_t> beta_host(g_host.size());
    for (uint16_t& value : q_host) value = to_half(dist(rng));
    for (uint16_t& value : k_host) value = to_half(dist(rng));
    for (uint16_t& value : v_host) value = to_half(dist(rng));
    // The gate is consumed as exp(g), so keep it negative as the real gate
    // kernel produces, and beta in (0, 1) as its sigmoid does.
    for (uint16_t& value : g_host) value = to_half(-0.5f - dist(rng));
    for (uint16_t& value : beta_host) value = to_half(0.5f + 0.5f * dist(rng));

    Buffer state_sequential, state_batched, device_q, device_k, device_v;
    Buffer device_g, device_beta, out_sequential, out_batched, device_slots;
    if (!state_sequential.allocate(state_elements * sizeof(float)) ||
        !state_batched.allocate(state_elements * sizeof(float)) ||
        !device_slots.allocate(kRows * sizeof(int)) ||
        !device_q.allocate(q_host.size() * sizeof(uint16_t)) ||
        !device_k.allocate(k_host.size() * sizeof(uint16_t)) ||
        !device_v.allocate(v_host.size() * sizeof(uint16_t)) ||
        !device_g.allocate(g_host.size() * sizeof(uint16_t)) ||
        !device_beta.allocate(beta_host.size() * sizeof(uint16_t)) ||
        !out_sequential.allocate(v_host.size() * sizeof(uint16_t)) ||
        !out_batched.allocate(v_host.size() * sizeof(uint16_t))) {
        device_ready = false;
        return false;
    }
    if (!state_sequential.upload(state_host.data()) ||
        !state_batched.upload(state_host.data()) ||
        !device_q.upload(q_host.data()) || !device_k.upload(k_host.data()) ||
        !device_v.upload(v_host.data()) || !device_g.upload(g_host.data()) ||
        !device_beta.upload(beta_host.data()) ||
        !device_slots.upload(slots) ||
        !out_sequential.clear() || !out_batched.clear()) {
        device_ready = false;
        return false;
    }

    const float q_scale = 1.0f / std::sqrt(static_cast<float>(kKeyDim));
    for (int row = 0; row < kRows; ++row) {
        const bool launched = qwen_gated_delta_step_f16_cuda(
            state_sequential.as<float>() +
                static_cast<size_t>(slots[row]) * state_stride,
            device_q.as<uint16_t>() + static_cast<size_t>(row) * key_row,
            device_k.as<uint16_t>() + static_cast<size_t>(row) * key_row,
            device_v.as<uint16_t>() + static_cast<size_t>(row) * value_row,
            device_g.as<uint16_t>() + static_cast<size_t>(row) * kHeads,
            device_beta.as<uint16_t>() + static_cast<size_t>(row) * kHeads,
            out_sequential.as<uint16_t>() + static_cast<size_t>(row) * value_row,
            kHeads, kKeyHeads, kKeyDim, kValueDim, q_scale);
        if (!launched) {
            fail("batched gated delta reference launch failed");
            return true;
        }
    }
    if (!qwen_gated_delta_step_batched_f16_cuda(
            state_batched.as<float>(), device_q.as<uint16_t>(),
            device_k.as<uint16_t>(), device_v.as<uint16_t>(),
            device_g.as<uint16_t>(), device_beta.as<uint16_t>(),
            out_batched.as<uint16_t>(), device_slots.as<int>(), kRows, kHeads,
            kKeyHeads, kKeyDim, kValueDim, q_scale, state_stride)) {
        fail("batched gated delta launch failed");
        return true;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        fail("batched gated delta sync failed");
        return true;
    }

    std::vector<uint16_t> want(v_host.size()), got(v_host.size());
    std::vector<float> state_want(state_elements), state_got(state_elements);
    if (!out_sequential.download(want.data()) ||
        !out_batched.download(got.data()) ||
        !state_sequential.download(state_want.data()) ||
        !state_batched.download(state_got.data())) {
        device_ready = false;
        return false;
    }
    const size_t output_bad = bit_mismatches(want, got);
    double state_worst = 0.0;
    for (size_t i = 0; i < state_elements; ++i) {
        state_worst = std::max(state_worst, std::fabs(
            static_cast<double>(state_want[i]) -
            static_cast<double>(state_got[i])));
    }
    // The advanced state matters as much as the returned row: a decode step that
    // writes the right output but the wrong state corrupts every later token.
    if (output_bad != 0 || state_worst != 0.0) {
        fail("batched gated delta output_mismatch=" +
             std::to_string(output_bad) + " state_worst=" +
             std::to_string(state_worst));
    } else {
        std::printf("  gated delta rows=%d heads=%d/%d state bit-identical\n",
                    kRows, kHeads, kKeyHeads);
    }
    return true;
}

// Causal conv. Each row convolves one new token against its own tail and then
// shifts that tail, so both the output and the updated tail are checked.
bool test_conv_silu_batched() {
    constexpr int kRows = 3;
    constexpr int kChannels = 512;
    constexpr int kKernel = 4;
    constexpr int kTailLength = kKernel - 1;

    constexpr int kSlots = 5;
    const int slots[kRows] = {3, 1, 4};

    const size_t tail_stride = static_cast<size_t>(kTailLength) * kChannels;
    const size_t tail_elements = static_cast<size_t>(kSlots) * tail_stride;
    const size_t row_elements = static_cast<size_t>(kChannels);

    std::mt19937 rng(13579);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<uint16_t> x_host(static_cast<size_t>(kRows) * row_elements);
    std::vector<uint16_t> weight_host(
        static_cast<size_t>(kChannels) * kKernel);
    std::vector<uint16_t> tail_host(tail_elements);
    for (uint16_t& value : x_host) value = to_half(dist(rng));
    for (uint16_t& value : weight_host) value = to_half(dist(rng));
    for (uint16_t& value : tail_host) value = to_half(dist(rng));

    Buffer device_x, device_weight, tail_sequential, tail_batched;
    Buffer out_sequential, out_batched, device_slots;
    if (!device_slots.allocate(kRows * sizeof(int)) ||
        !device_x.allocate(x_host.size() * sizeof(uint16_t)) ||
        !device_weight.allocate(weight_host.size() * sizeof(uint16_t)) ||
        !tail_sequential.allocate(tail_elements * sizeof(uint16_t)) ||
        !tail_batched.allocate(tail_elements * sizeof(uint16_t)) ||
        !out_sequential.allocate(x_host.size() * sizeof(uint16_t)) ||
        !out_batched.allocate(x_host.size() * sizeof(uint16_t))) {
        device_ready = false;
        return false;
    }
    if (!device_x.upload(x_host.data()) ||
        !device_weight.upload(weight_host.data()) ||
        !tail_sequential.upload(tail_host.data()) ||
        !tail_batched.upload(tail_host.data()) ||
        !device_slots.upload(slots) ||
        !out_sequential.clear() || !out_batched.clear()) {
        device_ready = false;
        return false;
    }

    for (int row = 0; row < kRows; ++row) {
        const bool launched = qwen_causal_depthwise_conv_silu_f16_cuda(
            device_x.as<uint16_t>() + static_cast<size_t>(row) * row_elements,
            device_weight.as<uint16_t>(),
            tail_sequential.as<uint16_t>() +
                static_cast<size_t>(slots[row]) * tail_stride,
            out_sequential.as<uint16_t>() +
                static_cast<size_t>(row) * row_elements,
            1, kChannels, kKernel, true);
        if (!launched) {
            fail("batched conv reference launch failed");
            return true;
        }
    }
    if (!qwen_causal_depthwise_conv_silu_f16_batched_cuda(
            device_x.as<uint16_t>(), device_weight.as<uint16_t>(),
            tail_batched.as<uint16_t>(), out_batched.as<uint16_t>(),
            device_slots.as<int>(), kRows, kChannels, kKernel, tail_stride)) {
        fail("batched conv launch failed");
        return true;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        fail("batched conv sync failed");
        return true;
    }

    std::vector<uint16_t> want(x_host.size()), got(x_host.size());
    std::vector<uint16_t> tail_want(tail_elements), tail_got(tail_elements);
    if (!out_sequential.download(want.data()) ||
        !out_batched.download(got.data()) ||
        !tail_sequential.download(tail_want.data()) ||
        !tail_batched.download(tail_got.data())) {
        device_ready = false;
        return false;
    }
    const size_t output_bad = bit_mismatches(want, got);
    const size_t tail_bad = bit_mismatches(tail_want, tail_got);
    if (output_bad != 0 || tail_bad != 0) {
        fail("batched conv output_mismatch=" + std::to_string(output_bad) +
             " tail_mismatch=" + std::to_string(tail_bad));
    } else {
        std::printf("  causal conv rows=%d channels=%d tail bit-identical\n",
                    kRows, kChannels);
    }
    return true;
}

}  // namespace

int main() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::printf("[SKIP] no CUDA device\n");
        return 0;
    }
    const bool completed = test_rope_batched() && test_append_kv_batched() &&
        test_gqa_decode_batched() && test_gated_delta_step_batched() &&
        test_conv_silu_batched();
    if (!completed && !device_ready) {
        std::printf("[SKIP] device allocation failed\n");
        return 0;
    }
    if (failures != 0) {
        std::printf("test_qwen_batched_decode failures=%d\n", failures);
        return 1;
    }
    std::printf("[PASS] test_qwen_batched_decode\n");
    return 0;
}
